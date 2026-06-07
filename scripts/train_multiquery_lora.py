#!/usr/bin/env python3
"""LoRA training for the symmetric multi-query cross-cache (two-phase).

Reuses eval_multiquery's two-phase forward. Phase-1 captures each context's KV
into a shared bank; phase-2 runs each query + its gold answer through the
bank-reading attention and trains a LoRA on (q,k,v,o) with teacher-forced CE on
the answer tokens only. Inference flow is unchanged; only LoRA weights are added.

Cheap pilot: capture runs under no_grad (--bank-grad off), so gradient flows only
through the query path -- the bank is treated as a fixed evidence store. This is
far cheaper (no G context forwards retained) and tests whether the query side can
learn to read the bank better. Use --bank-grad to also train the capture path.
"""
import argparse, os, sys, logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.cross_batch.batch_crosscache import BatchCrossCache
from src.datasets import load_multiquery_hotpot_groups
from src.models import Question
from src.prompts import build_single_prompt
from src.templates import build_chat_prompt

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--n-agents", type=int, default=4)
    p.add_argument("--paragraphs-per-agent", type=int, default=6)
    p.add_argument("--num-groups", type=int, default=2000)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--max-prompt-length", type=int, default=1600)
    p.add_argument("--mix-agents", default=None,
                   help="comma list of pool sizes G to MIX during training (e.g. 2,4,8) -> one general "
                        "LoRA that reads the bank across pool sizes, not tied to a single G")
    p.add_argument("--topk", type=int, default=0,
                   help=">0: train with selective reading (each query reads only its top-k contexts), "
                        "matching selective inference. 0 = read all (default)")
    p.add_argument("--distill-alpha", type=float, default=0.0,
                   help=">0: add KL(student||teacher) where teacher=Concat (query sees full union in-context). "
                        "Distills the oracle's cross-context reasoning into the cross-cache student.")
    p.add_argument("--distill-temp", type=float, default=2.0, help="KL softmax temperature")
    p.add_argument("--teacher-max-len", type=int, default=4096, help="truncation for the teacher's union prompt")
    p.add_argument("--bank-grad", action="store_true",
                   help="also backprop through the capture path (expensive); default detaches the bank")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def build_prompt(tok, context, question):
    q = Question(qid="q", text=question, priority=1.0, answer_tokens=12,
                 type_hint=None, references=[])
    sp, up = build_single_prompt(context, q, dataset="hotpot")
    return build_chat_prompt(tok, up, system_prompt=sp)


def context_mask_for(tok, prompts, ctxs, P, attn, max_plen):
    cm = torch.zeros(len(prompts), P, dtype=torch.bool)
    for i, (s, ctx) in enumerate(zip(prompts, ctxs)):
        ctx = ctx.strip(); Li = int(attn[i].sum()); pad = P - Li
        c0 = s.find(ctx)
        if c0 < 0:
            continue
        c1 = c0 + len(ctx)
        offs = tok(s, return_offsets_mapping=True, truncation=True, max_length=max_plen,
                   add_special_tokens=True)["offset_mapping"]
        for j, (a, b) in enumerate(offs):
            if b > a and a >= c0 and b <= c1 and pad + j < P:
                cm[i, pad + j] = True
    return cm


def left_pad(seqs, pad_id, device):
    P = max(len(s) for s in seqs)
    out = torch.full((len(seqs), P), pad_id, dtype=torch.long, device=device)
    msk = torch.zeros((len(seqs), P), dtype=torch.long, device=device)
    for i, s in enumerate(seqs):
        out[i, P - len(s):] = torch.tensor(s, dtype=torch.long, device=device)
        msk[i, P - len(s):] = 1
    return out, msk


def _maxsim_scores(qhs, qattn, chs, cm):
    qn = torch.nn.functional.normalize(qhs.float(), dim=-1)
    cn = torch.nn.functional.normalize(chs.float(), dim=-1)
    sims = torch.einsum("itd,jsd->ijts", qn, cn)
    cmask = cm.view(1, cm.shape[0], 1, cm.shape[1])
    sims = sims.masked_fill(~cmask, float("-inf"))
    mx = sims.max(dim=3).values
    qmask = qattn.bool().view(qattn.shape[0], 1, qattn.shape[1])
    mx = mx.masked_fill(~qmask, 0.0)
    return mx.sum(dim=2) / qmask.float().sum(dim=2).clamp(min=1)


def capture(model, mgr, tok, items, device, max_plen, want_hs=False):
    cp = [build_prompt(tok, it["context"], it["question"]) for it in items]
    enc = tok(cp, return_tensors="pt", padding=True, truncation=True, max_length=max_plen)
    cids = enc["input_ids"].to(device); cattn = enc["attention_mask"].to(device)
    cm = context_mask_for(tok, cp, [it["context"] for it in items], cids.shape[1], cattn, max_plen).to(device)
    mgr.set_enabled(True); mgr.exclude_self = False; mgr.set_allowed(None); mgr.set_realign(1.0, 1.0)
    mgr.start_capture(cm); mgr.set_valid(cattn)
    out = model(input_ids=cids, attention_mask=cattn, use_cache=False, output_hidden_states=want_hs)
    off = int(cattn.sum(1).max().item())
    return off, (out.hidden_states[-1] if want_hs else None), cm


@torch.no_grad()
def select_allowed(model, mgr, tok, items, device, chs, cm, topk, max_plen):
    qp = [build_prompt(tok, "", it["question"]) for it in items]
    enc = tok(qp, return_tensors="pt", padding=True, truncation=True, max_length=max_plen)
    qids = enc["input_ids"].to(device); qattn = enc["attention_mask"].to(device)
    mgr.set_enabled(False)
    qhs = model(input_ids=qids, attention_mask=qattn, output_hidden_states=True).hidden_states[-1]
    mgr.set_enabled(True)
    R = _maxsim_scores(qhs, qattn, chs, cm)
    G = qids.shape[0]
    topi = R.topk(min(topk, G), dim=1).indices
    allowed = torch.zeros(G, G, dtype=torch.bool, device=device)
    allowed.scatter_(1, topi, True)
    return allowed


def _union(items):
    seen = set(); paras = []
    for it in items:
        for p in it["context"].split("\n\n"):
            p = p.strip()
            if p and p not in seen:
                seen.add(p); paras.append(p)
    return "\n\n".join(paras)


def _ans_logits(logits, labels):
    """Answer-position logits (where the shifted label != -100), flattened [N, vocab]."""
    sl = logits[:, :-1, :]; lab = labels[:, 1:]
    return sl[lab != -100]


def use_loss(model, mgr, tok, items, device, off, max_plen, return_logits=False):
    eos = tok.eos_token_id; pad = tok.pad_token_id or eos
    ids_l, lab_l = [], []
    for it in items:
        qp = build_prompt(tok, "", it["question"])
        qids = tok(qp, add_special_tokens=True, truncation=True, max_length=max_plen)["input_ids"]
        aids = tok(" " + it["references"][0], add_special_tokens=False)["input_ids"] + [eos]
        ids_l.append(qids + aids)
        lab_l.append([-100] * len(qids) + aids)
    ids, attn = left_pad(ids_l, pad, device)
    labels, _ = left_pad(lab_l, -100, device)
    labels = labels.masked_fill(attn == 0, -100)
    base_pos = (attn.long().cumsum(1) - 1).clamp(min=0)
    pos = base_pos + off
    mgr.set_valid(attn); mgr.set_query_rows(attn.bool()); mgr.start_use()
    out = model(input_ids=ids, attention_mask=attn, position_ids=pos, labels=labels, use_cache=False)
    if return_logits:
        return out.loss, _ans_logits(out.logits, labels)
    return out.loss


@torch.no_grad()
def teacher_ans_logits(model, mgr, tok, items, device, max_plen):
    """Concat teacher: each query sees the FULL union in-context (standard attention)."""
    eos = tok.eos_token_id; pad = tok.pad_token_id or eos
    uni = _union(items)
    ids_l, lab_l = [], []
    for it in items:
        tp = build_prompt(tok, uni, it["question"])
        qids = tok(tp, add_special_tokens=True, truncation=True, max_length=max_plen)["input_ids"]
        aids = tok(" " + it["references"][0], add_special_tokens=False)["input_ids"] + [eos]
        ids_l.append(qids + aids)
        lab_l.append([-100] * len(qids) + aids)
    ids, attn = left_pad(ids_l, pad, device)
    labels, _ = left_pad(lab_l, -100, device)
    labels = labels.masked_fill(attn == 0, -100)
    mgr.set_enabled(False)
    out = model(input_ids=ids, attention_mask=attn, use_cache=False)
    mgr.set_enabled(True)
    return _ans_logits(out.logits, labels)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token; tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "left"; tok.truncation_side = "left"
    device = "cuda"
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=torch.bfloat16, trust_remote_code=True,
        attn_implementation="sdpa").to(device)
    lora = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)
    model.train()
    model.print_trainable_parameters()
    nl = model.config.num_hidden_layers
    mgr = BatchCrossCache(list(range(nl))); mgr.register(model)

    if args.mix_agents:
        # GENERAL recipe: mix several pool sizes G so one LoRA learns to read the shared bank
        # regardless of how many contexts are present (not tied to a single G).
        Gs = [int(x) for x in args.mix_agents.split(",")]
        per = max(1, args.num_groups // len(Gs))
        groups = []
        for gi, G in enumerate(Gs):
            groups += load_multiquery_hotpot_groups(
                split="train", n_agents=G, paragraphs_per_agent=args.paragraphs_per_agent,
                max_groups=per, seed=args.seed + gi, only_bridge=True,
                require_min_supporting=2, cross_question_distractor_pool=True)
        import random as _r; _r.Random(args.seed).shuffle(groups)
        log.info("MIXED-G train groups=%d over G=%s", len(groups), Gs)
    else:
        groups = load_multiquery_hotpot_groups(
            split="train", n_agents=args.n_agents, paragraphs_per_agent=args.paragraphs_per_agent,
            max_groups=args.num_groups, seed=args.seed, only_bridge=True,
            require_min_supporting=2, cross_question_distractor_pool=True)
    log.info("train groups=%d  bank_grad=%s", len(groups), args.bank_grad)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    step = 0; run = 0.0
    for ep in range(args.epochs):
        for g in groups:
            items = g["items"]
            want_hs = args.topk > 0
            if args.bank_grad:
                off, chs, cm = capture(model, mgr, tok, items, device, args.max_prompt_length, want_hs)
            else:
                with torch.no_grad():
                    off, chs, cm = capture(model, mgr, tok, items, device, args.max_prompt_length, want_hs)
            if args.topk > 0:
                mgr.set_allowed(select_allowed(model, mgr, tok, items, device, chs, cm,
                                               args.topk, args.max_prompt_length))
            else:
                mgr.set_allowed(None)
            if args.distill_alpha > 0:
                loss, s_log = use_loss(model, mgr, tok, items, device, off, args.max_prompt_length,
                                       return_logits=True)
                t_log = teacher_ans_logits(model, mgr, tok, items, device, args.teacher_max_len)
                if s_log.shape == t_log.shape:                    # same answer-token count -> KL aligned
                    T = args.distill_temp
                    kl = torch.nn.functional.kl_div(
                        torch.log_softmax(s_log / T, dim=-1),
                        torch.softmax(t_log.float() / T, dim=-1),
                        reduction="batchmean") * (T * T)
                    loss = loss + args.distill_alpha * kl
            else:
                loss = use_loss(model, mgr, tok, items, device, off, args.max_prompt_length)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step(); opt.zero_grad()
            run += loss.item(); step += 1
            if step % 50 == 0:
                log.info("ep%d step%d loss=%.4f", ep, step, run / 50); run = 0.0
    model.save_pretrained(args.output_dir)
    tok.save_pretrained(args.output_dir)
    log.info("saved LoRA adapter to %s", args.output_dir)


if __name__ == "__main__":
    main()
