#!/usr/bin/env python3
"""GRPO on the bank-read policy: on-policy rollouts under independent encoding, reward = SubEM.

Why RL here: distillation imitates the concat teacher and is capped by it; GRPO explores
trajectories under the ACTUAL bank-read decode and directly maximizes in-domain answer
correctness — the "beat the standard model on the training datasets' dev" goal.

Policy init: --warm-start distill LoRA is MERGED into the base, so the model with the fresh
trainable LoRA DISABLED is exactly the distill policy = the KL reference (no second model in
memory). Rollouts: per question, capture a gold+distractor bank once (no grad), sample G
completions in one batch (temperature), reward = SubEM, group-normalized advantage,
token-level policy gradient with k3 KL penalty to the reference.

Checkpoints saved every --save-every steps (overwrite) + final.
"""
import argparse, os, sys, json, random, logging
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.cross_batch.batch_crosscache import BatchCrossCache
from scripts.bench_distract import _load_arrow, _para_text
from scripts.bench_standard import bp, subem, extract_answer
from scripts.bench_selective import capture
from scripts.eval_multiquery import decode_texts

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--warm-start", required=True, help="distill LoRA to merge as policy init + KL ref")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--dataset", default="2wikimultihopqa",
                   help="flashrag name, or 'hotpotqa_hf' for the HF hotpot_qa distractor train split")
    p.add_argument("--split", default="train")
    p.add_argument("--think", action="store_true",
                   help="think-mode rollouts (enable_thinking prompts; raise --max-new to 192+)")
    p.add_argument("--flashrag-root",
                   default="/mnt/data/zichuanfu/.cache/huggingface/datasets/RUC-NLPIR___flash_rag_datasets")
    p.add_argument("--num-q", type=int, default=3000)
    p.add_argument("--n-paras-min", type=int, default=4)
    p.add_argument("--n-paras-max", type=int, default=8)
    p.add_argument("--group", type=int, default=8, help="G rollouts per question")
    p.add_argument("--max-new", type=int, default=32, help="nothink rollouts; raise for think")
    p.add_argument("--temp", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--beta", type=float, default=0.04, help="KL penalty to the distill reference")
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--seg-cap", type=int, default=768)
    p.add_argument("--max-plen", type=int, default=1600)
    p.add_argument("--save-every", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_items(root, name, split, num_q, rng):
    """gold paragraphs + global distractor pool, same parse as bench_distract.build_examples
    but with a split argument (train for GRPO, dev stays untouched for eval)."""
    rows = _load_arrow(root, name, split)
    rng.shuffle(rows)
    items, pool = [], []
    for r in rows:
        md = r["metadata"]
        if isinstance(md, str):
            md = json.loads(md)
        ctx = md.get("context"); sf = md.get("supporting_facts")
        if not isinstance(ctx, dict) or "title" not in ctx or not sf:
            continue
        titles = ctx["title"]
        texts = ctx.get("content") or ctx.get("text") or ctx.get("sentences")
        if texts is None or len(texts) != len(titles):
            continue
        gold_titles = set(sf.get("title", []))
        gold, others = [], []
        for t, tx in zip(titles, texts):
            s = _para_text(tx).strip()
            if not s:
                continue
            (gold if t in gold_titles else others).append(s)
        pool.extend(others)
        if gold:
            ans = r["golden_answers"]
            ans = ans if isinstance(ans, list) else [ans]
            items.append({"gold": gold, "question": r["question"], "answers": ans})
        if len(items) >= num_q and len(pool) > 5000:
            break
    return items[:num_q], pool


def load_hotpot_train(num_q, rng):
    """HF hotpot_qa distractor train: native gold + in-context distractor paragraphs."""
    from datasets import load_dataset
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="train")
    idx = list(range(len(ds))); rng.shuffle(idx)
    items, pool = [], []
    for i in idx:
        ex = ds[i]
        titles = ex["context"]["title"]; sents = ex["context"]["sentences"]
        gold_titles = set(ex["supporting_facts"]["title"])
        gold, others = [], []
        for t, sx in zip(titles, sents):
            s = " ".join(sx).strip()
            if not s:
                continue
            (gold if t in gold_titles else others).append(s)
        pool.extend(others)
        if gold:
            items.append({"gold": gold, "question": ex["question"], "answers": [ex["answer"]]})
        if len(items) >= num_q and len(pool) > 5000:
            break
    return items[:num_q], pool


@torch.no_grad()
def sample_group(model, tok, mgr, question, device, off, G, max_new, temp, think=False):
    """sample G completions in one batch reading the shared bank."""
    qp = bp(tok, "", question, think)
    enc = tok([qp], return_tensors="pt", add_special_tokens=False)
    qids = enc["input_ids"].to(device).repeat(G, 1)
    qattn = enc["attention_mask"].to(device).repeat(G, 1)
    eos = tok.eos_token_id; pad = tok.pad_token_id or eos
    mgr.start_use()
    pos = (qattn.long().cumsum(1) - 1).clamp(min=0) + off
    nxt_pos = off + qattn.sum(1)
    cur = qattn.clone(); nxt = qids.clone()
    fin = torch.zeros(G, dtype=torch.bool, device=device); pkv = None
    P = qids.shape[1]
    gen = qids.clone()
    for step in range(max_new):
        mgr.set_valid(cur)
        mgr.set_query_rows(cur.bool() if step == 0 else torch.ones(G, 1, dtype=torch.bool, device=device))
        pid = pos if step == 0 else nxt_pos.view(G, 1)
        out = model(input_ids=nxt, attention_mask=cur, position_ids=pid,
                    past_key_values=pkv, use_cache=True)
        pkv = out.past_key_values
        if step > 0:
            nxt_pos = nxt_pos + 1
        logits = out.logits[:, -1] / max(temp, 1e-4)
        t = torch.multinomial(F.softmax(logits, dim=-1), 1).squeeze(1)
        t = torch.where(fin, torch.full_like(t, pad), t)
        fin = fin | (t == eos)
        gen = torch.cat([gen, t.unsqueeze(1)], 1)
        cur = torch.cat([cur, (~fin).long().unsqueeze(1)], 1); nxt = t.unsqueeze(1)
        if fin.all():
            break
    texts = decode_texts(tok, gen, P, eos, pad)
    comps = []
    for g in range(G):
        ids = gen[g, P:].tolist()
        if eos in ids:
            ids = ids[: ids.index(eos) + 1]
        comps.append(ids)
    return texts, comps, qids[0].tolist()


def left_pad(seqs, pad_id, device):
    P = max(len(s) for s in seqs)
    out = torch.full((len(seqs), P), pad_id, dtype=torch.long, device=device)
    msk = torch.zeros((len(seqs), P), dtype=torch.long, device=device)
    for i, s in enumerate(seqs):
        out[i, P - len(s):] = torch.tensor(s, dtype=torch.long, device=device)
        msk[i, P - len(s):] = 1
    return out, msk


def token_logps(model, mgr, ids, attn, labels, off, device):
    """per-token logprob of labels (-100 masked) under the bank-read forward."""
    pos = (attn.long().cumsum(1) - 1).clamp(min=0) + off
    mgr.set_valid(attn); mgr.set_query_rows(attn.bool()); mgr.start_use()
    out = model(input_ids=ids, attention_mask=attn, position_ids=pos, use_cache=False)
    logp = F.log_softmax(out.logits[:, :-1].float(), dim=-1)
    lab = labels[:, 1:]
    mask = lab != -100
    gather = torch.gather(logp, 2, lab.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    return gather, mask


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda"
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token; tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "left"; tok.truncation_side = "left"

    base = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=torch.bfloat16, trust_remote_code=True,
        attn_implementation="sdpa").to(device)
    base = PeftModel.from_pretrained(base, args.warm_start).merge_and_unload()
    log.info("merged warm-start %s into base (= KL reference)", args.warm_start)
    lora = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.0,
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM")
    model = get_peft_model(base, lora)
    model.train(); model.print_trainable_parameters()
    mgr = BatchCrossCache(list(range(model.config.num_hidden_layers))); mgr.register(model)

    if args.dataset == "hotpotqa_hf":
        items, pool = load_hotpot_train(args.num_q, rng)
    else:
        items, pool = load_items(args.flashrag_root, args.dataset, args.split, args.num_q, rng)
    log.info("GRPO items=%d pool=%d", len(items), len(pool))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    step = 0; run_r = 0.0; run_loss = 0.0; skipped = 0
    for it in items:
        n_paras = rng.randint(args.n_paras_min, args.n_paras_max)
        need = max(0, n_paras - len(it["gold"]))
        paras = it["gold"] + rng.sample(pool, min(need, len(pool)))
        rng.shuffle(paras)
        model.eval()
        off = capture(model, tok, mgr, paras, it["question"], device, args.seg_cap, args.max_plen)
        texts, comps, qids = sample_group(model, tok, mgr, it["question"], device, off,
                                          args.group, args.max_new, args.temp, args.think)
        rewards = torch.tensor([subem(extract_answer(t), it["answers"]) for t in texts],
                               device=device)
        run_r += rewards.mean().item()
        if rewards.std() < 1e-6:
            skipped += 1; mgr.bank = {}; step += 1
            if step % 25 == 0:
                log.info("step%d avg_reward=%.3f loss=%.4f skipped=%d",
                         step, run_r / 25, run_loss / max(1, 25 - skipped), skipped)
                run_r = 0.0; run_loss = 0.0; skipped = 0
            if args.save_every and step % args.save_every == 0:
                model.save_pretrained(args.output_dir)
            continue
        adv = (rewards - rewards.mean()) / (rewards.std() + 1e-4)

        ids_l = [qids + c for c in comps]
        lab_l = [[-100] * len(qids) + c for c in comps]
        ids, attn = left_pad(ids_l, tok.pad_token_id, device)
        labels, _ = left_pad(lab_l, -100, device)
        labels = labels.masked_fill(attn == 0, -100)

        model.train()
        lp_pol, mask = token_logps(model, mgr, ids, attn, labels, off, device)
        with torch.no_grad(), model.disable_adapter():
            lp_ref, _ = token_logps(model, mgr, ids, attn, labels, off, device)
        # k3 KL estimator per token
        kl = torch.exp(lp_ref - lp_pol) - (lp_ref - lp_pol) - 1.0
        pg = -(adv.unsqueeze(1) * lp_pol)
        loss = ((pg + args.beta * kl) * mask).sum() / mask.sum().clamp(min=1)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step(); opt.zero_grad()
        run_loss += loss.item()
        mgr.bank = {}; step += 1
        if step % 8 == 0:
            torch.cuda.empty_cache()
        if step % 25 == 0:
            log.info("step%d avg_reward=%.3f loss=%.4f skipped=%d",
                     step, run_r / 25, run_loss / max(1, 25 - skipped), skipped)
            run_r = 0.0; run_loss = 0.0; skipped = 0
        if step % args.save_every == 0:
            model.save_pretrained(args.output_dir)
            log.info("checkpoint saved at step%d -> %s", step, args.output_dir)
    model.save_pretrained(args.output_dir)
    tok.save_pretrained(args.output_dir)
    log.info("GRPO done: %d steps -> %s", step, args.output_dir)


if __name__ == "__main__":
    main()
