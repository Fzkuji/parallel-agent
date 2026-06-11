#!/usr/bin/env python3
"""Multi-turn parallel encoding (see docs/multiturn-design.md).

Session-level accumulating bank: each turn's passages are encoded INDEPENDENTLY (蒙眼) and
APPENDED to a shared bank with a per-turn position offset (轮间错开); each turn's query reads
the WHOLE accumulated bank (睁眼 = all passages from all turns so far). This is what APE cannot
do — APE is single-turn, its query never sees a previous turn's passages.

Validation protocol (does turn-2 actually use turn-1's passages?):
  We build a 2-turn episode where turn-2's question is answerable ONLY from a passage that was
  introduced in turn-1. Two arms:
    multiturn : accumulating bank (turn-2 reads turn-1 + turn-2 passages)   -> should answer
    isolated  : fresh bank each turn (turn-2 reads ONLY turn-2 passages)    -> should FAIL
  If multiturn >> isolated on turn-2, the cross-turn bank works as designed.

Also reports a speed comparison vs single-turn APE-style read on the same passage set.
"""
import argparse, glob, json, os, random, sys, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.cross_batch.batch_crosscache import BatchCrossCache
from scripts.eval_multiquery import build_prompt as _bp_think, context_mask_for, decode_texts
from scripts.bench_distract import build_examples
from src.evaluation.basic import normalize_answer
from src.templates import build_chat_prompt
from src.prompts import build_single_prompt
from src.models import Question


def build_prompt(tok, context, question):
    """no-think build_prompt: Qwen3 outputs the answer directly (no <think>...). The thinking
    model otherwise spends all decode tokens reasoning and never reaches the answer -> SubEM floor."""
    q = Question(qid="q", text=question, priority=1.0, answer_tokens=12, type_hint=None, references=[])
    sp, up = build_single_prompt(context, q, dataset="hotpot")
    return build_chat_prompt(tok, up, system_prompt=sp, enable_thinking=False)

FLASHRAG = "/mnt/data/zichuanfu/.cache/huggingface/datasets/RUC-NLPIR___flash_rag_datasets"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--lora-path", default=None)
    p.add_argument("--dataset", default="2wikimultihopqa", choices=["2wikimultihopqa", "hotpotqa"])
    p.add_argument("--flashrag-root", default=FLASHRAG)
    p.add_argument("--num-episodes", type=int, default=50)
    p.add_argument("--paras-per-turn", type=int, default=4)
    p.add_argument("--max-new", type=int, default=32)
    p.add_argument("--max-plen", type=int, default=1600)
    p.add_argument("--seg-cap", type=int, default=768)
    p.add_argument("--arms", default="multiturn,ape_realign,isolated,nooffset",
                   help="subset of {multiturn, ape_realign, isolated, nooffset}")
    p.add_argument("--ape-temp", type=float, default=0.9, help="ape_realign bank temperature")
    p.add_argument("--ape-scale", type=float, default=0.9, help="ape_realign bank LSE scale")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gpu", type=int, default=5)
    return p.parse_args()


def subem(pred, answers):
    p = normalize_answer(pred)
    return 1.0 if any(normalize_answer(a) and normalize_answer(a) in p for a in answers) else 0.0


# ---------------------------------------------------------------- core: accumulating bank read
@torch.no_grad()
def capture_turn(model, tok, mgr, chunks, question, device, off, max_plen, seg_cap, fresh):
    """Encode this turn's passages INDEPENDENTLY into the bank at position offset `off`.

    fresh=True  -> reset bank first (isolated turn / first turn).
    fresh=False -> APPEND to the existing bank (cross-turn accumulation).
    Returns the position length consumed by this turn's passages (max segment length)."""
    cp = [build_prompt(tok, c, question) for c in chunks]
    enc = tok(cp, return_tensors="pt", padding=True, truncation=True,
              max_length=min(max_plen, seg_cap))
    cids = enc["input_ids"].to(device); cattn = enc["attention_mask"].to(device)
    cm = context_mask_for(tok, cp, chunks, cids.shape[1], cattn, max_plen).to(device)
    seg = cids.shape[1]
    # per-turn position offset: every segment is encoded from `off` (轮内重叠, 轮间错开)
    pid = (torch.arange(seg, device=device).unsqueeze(0).expand(cids.shape[0], seg) + off)
    mgr.set_enabled(True); mgr.exclude_self = False; mgr.set_allowed(None); mgr.set_realign(1.0, 1.0)
    if fresh:
        mgr.start_capture(cm)            # mode=capture + clears bank
    else:
        mgr.mode = "capture"             # APPEND: keep existing bank, capture fwd cats into it
    mgr.context_mask = cm.bool(); mgr.set_valid(cattn)
    model(input_ids=cids, attention_mask=cattn, position_ids=pid, use_cache=False)
    return int(cattn.sum(1).max().item())


@torch.no_grad()
def answer_turn(model, tok, mgr, question, device, q_off, max_new, max_plen, temp=1.0, scale=1.0):
    """Query reads the WHOLE accumulated bank. temp=scale=1 -> ours (flash fast path);
    temp=scale<1 -> APE realignment (sharpen bank + LSE down-weight, the APE-style read)."""
    qp = build_prompt(tok, "", question)
    enc = tok([qp], return_tensors="pt", truncation=True, max_length=max_plen)
    qids = enc["input_ids"].to(device); qattn = enc["attention_mask"].to(device)
    eos = tok.eos_token_id; pad = tok.pad_token_id or eos
    mgr.set_realign(temp, scale)
    mgr.start_use()
    pos = (qattn.long().cumsum(1) - 1).clamp(min=0) + q_off
    nxt_pos = q_off + qattn.sum(1)
    gen = qids.clone(); cur = qattn.clone()
    fin = torch.zeros(1, dtype=torch.bool, device=device); pkv = None; nxt = gen
    P = qids.shape[1]
    for step in range(max_new):
        mgr.set_valid(cur)
        mgr.set_query_rows(cur.bool() if step == 0 else torch.ones(1, 1, dtype=torch.bool, device=device))
        pid = pos if step == 0 else nxt_pos.view(1, 1)
        out = model(input_ids=nxt, attention_mask=cur, position_ids=pid,
                    past_key_values=pkv, use_cache=True)
        pkv = out.past_key_values
        if step > 0:
            nxt_pos = nxt_pos + 1
        t = out.logits[:, -1].argmax(-1); t = torch.where(fin, torch.full_like(t, pad), t)
        fin = fin | (t == eos)
        gen = torch.cat([gen, t.unsqueeze(1)], 1)
        cur = torch.cat([cur, (~fin).long().unsqueeze(1)], 1); nxt = t.unsqueeze(1)
        if fin.all():
            break
    return decode_texts(tok, gen, P, eos, pad)[0]


def run_episode(model, tok, mgr, ep, device, args, cfg):
    """cfg = {accumulate, no_offset, temp, scale}. Returns per-turn SubEM list."""
    scores = []
    off = 0
    for ti, turn in enumerate(ep["turns"]):
        fresh = (ti == 0) or (not cfg["accumulate"])
        # `off` = per-turn position offset (轮间错开). no_offset -> every turn from pos 0.
        turn_off = 0 if (cfg["no_offset"] or fresh) else off
        seg_len = capture_turn(model, tok, mgr, turn["passages"], turn["question"],
                               device, turn_off, args.max_plen, args.seg_cap, fresh)
        q_off = turn_off + seg_len
        pred = answer_turn(model, tok, mgr, turn["question"], device, q_off, args.max_new,
                           args.max_plen, cfg["temp"], cfg["scale"])
        scores.append(subem(pred, turn["answers"]))
        off = q_off + 48
    return scores


# ---------------------------------------------------------------- data: 2-turn cross-turn episodes
def build_episodes(parsed, pool, n_eps, paras_per_turn, rng):
    """Each episode = 2 turns, designed so turn-2 is answerable ONLY via the cross-turn bank.

      turn-1: question=qB, passages = gold_A + gold_B + distractors   (INTRODUCES gold_A; asks qB)
      turn-2: question=qA, passages = distractors only (NO gold_A)     (gold_A only in turn-1's bank)

    So gold_A enters the bank during turn-1 (while the model is answering an UNRELATED qB), and
    turn-2 asks qA whose evidence is gold_A — present only if the bank accumulated across turns.
      multiturn (accumulating bank) -> turn-2 CAN answer qA (reads turn-1's gold_A)
      isolated  (fresh bank)        -> turn-2 has only distractors -> should FAIL (≈0)
    Cross-turn gain = multiturn.turn2 − isolated.turn2. Turn-1 is just the carrier of gold_A; it is
    the same in both arms, so it does not enter the comparison.
    """
    def pad(paras, n):
        need = max(0, n - len(paras))
        out = paras + rng.sample(pool, min(need, len(pool)))
        rng.shuffle(out)
        return out

    eps = []
    for i in range(0, len(parsed) - 1, 2):
        A, B = parsed[i], parsed[i + 1]
        t1 = pad(A["gold"] + B["gold"], paras_per_turn)   # turn-1: gold_A + gold_B + distractors
        t2 = pad([], paras_per_turn)                      # turn-2: distractors only (no gold_A)
        eps.append({"turns": [
            {"passages": t1, "question": B["question"], "answers": B["answer"]},   # ask qB
            {"passages": t2, "question": A["question"], "answers": A["answer"]},   # ask qA (gold_A in bank only)
        ]})
        if len(eps) >= n_eps:
            break
    return eps


def main():
    args = parse_args()
    torch.cuda.set_device(args.gpu)
    device = f"cuda:{args.gpu}"
    rng = random.Random(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token; tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=torch.bfloat16, attn_implementation="sdpa").to(device).eval()
    if args.lora_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.lora_path).merge_and_unload().eval()
        print("loaded LoRA:", args.lora_path, flush=True)
    mgr = BatchCrossCache(list(range(len(model.model.layers))))
    mgr.register(model)

    parsed, pool = build_examples(args.flashrag_root, args.dataset, args.num_episodes, args.seed)
    eps = build_episodes(parsed, pool, args.num_episodes, args.paras_per_turn, rng)
    print(f"built {len(eps)} 2-turn episodes ({args.dataset}, {args.paras_per_turn} paras/turn)", flush=True)

    # arm -> setting. isolated == APE-single-turn-equivalent (fresh bank, no history).
    # ape_realign == accumulate bank but read with APE temperature/scale (the APE-style read).
    ARM_CFG = {
        "multiturn":   dict(accumulate=True,  no_offset=False, temp=1.0, scale=1.0),  # ours full
        "nooffset":    dict(accumulate=True,  no_offset=True,  temp=1.0, scale=1.0),  # ablate 错开
        "ape_realign": dict(accumulate=True,  no_offset=False, temp=args.ape_temp, scale=args.ape_scale),
        "isolated":    dict(accumulate=False, no_offset=False, temp=1.0, scale=1.0),  # = APE single-turn
    }
    if not eps:
        print(f"[ERROR] 0 episodes for {args.dataset} -- check build_examples field parsing", flush=True)
        return
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    for arm in arms:
        cfg = ARM_CFG[arm]
        t1s, t2s = [], []
        t0 = time.time()
        for ep in eps:
            s = run_episode(model, tok, mgr, ep, device, args, cfg)
            t1s.append(s[0]); t2s.append(s[1])
            torch.cuda.empty_cache()
        dt = time.time() - t0
        n = len(eps)
        print(f"[{arm:11s}] turn1={100*sum(t1s)/n:.1f}  turn2={100*sum(t2s)/n:.1f}  "
              f"({n} eps, {dt/n*1000:.0f}ms/ep)", flush=True)


if __name__ == "__main__":
    main()
