#!/usr/bin/env python3
"""Symmetric multi-query cross-cache (two phase).

Phase 1 (capture): each context is prefilled independently; its KV goes into a
SHARED bank. Phase 2 (use): each query sequence carries NO context of its own
(empty passage) -- every query attends the WHOLE context bank equally. This kills
the own-context anchor that made the asymmetric batched version fail: here all
contexts are equal siblings to every query, exactly like the single-sequence form
that worked, but now batched, supporting different questions, contexts computed
once each.

Conditions:
  Independent : each query sees only its own context (baseline)
  MultiQuery  : each query sees the shared context bank (this method)
"""
import argparse, json, logging, os, sys, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.cross_batch.batch_crosscache import BatchCrossCache
from src.datasets import load_distributed_hotpot_groups, load_multiquery_hotpot_groups
from src.evaluation.basic import compute_em
from src.inference import extract_answer
from src.models import Question
from src.prompts import build_single_prompt
from src.templates import build_chat_prompt

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--num-eval-groups", type=int, default=60)
    p.add_argument("--n-agents", type=int, default=4)
    p.add_argument("--paragraphs-per-agent", type=int, default=6)
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--max-prompt-length", type=int, default=1600)
    p.add_argument("--oracle-select", action="store_true",
                   help="bank only holds contexts that actually carry a supporting paragraph "
                        "(oracle selectivity; tests whether selective reading fixes has_supp dilution)")
    p.add_argument("--diff-questions", action="store_true",
                   help="G DIFFERENT questions share one pool (the genuine multi-query setting), "
                        "instead of dHotpot's same-question groups")
    p.add_argument("--topk", type=int, default=0,
                   help=">0: each query reads only its top-k most relevant contexts (selective). 0 = read all")
    p.add_argument("--selector", default="maxsim", choices=["mean", "maxsim"],
                   help="relevance scorer for top-k: mean=question/context mean-hidden cosine; "
                        "maxsim=late-interaction max token-level cosine (usually much better)")
    p.add_argument("--ape-temp", type=float, default=1.0,
                   help="APE attention temperature on the bank segment (T<1 sharpens; 1.0=off)")
    p.add_argument("--ape-scale", type=float, default=1.0,
                   help="APE LSE scaling S on the bank segment (S<1 down-weights; 1.0=off)")
    p.add_argument("--with-oracle", action="store_true",
                   help="also run Oracle: each query prefills the FULL pool in-context (upper bound)")
    p.add_argument("--no-offset", action="store_true",
                   help="ablation: drop the position offset (queries overlap the bank in position)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lora-path", default=None,
                   help="optional: load + merge a trained LoRA adapter before eval")
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


def build_prompt(tok, context, question):
    q = Question(qid="q", text=question, priority=1.0, answer_tokens=12,
                 type_hint=None, references=[])
    sp, up = build_single_prompt(context, q, dataset="hotpot")
    return build_chat_prompt(tok, up, system_prompt=sp)


def union_context(items):
    seen = set(); paras = []
    for it in items:
        for para in it["context"].split("\n\n"):
            para = para.strip()
            if para and para not in seen:
                seen.add(para); paras.append(para)
    return "\n\n".join(paras)


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


def decode_texts(tok, gen, P, eos, pad):
    out = []
    for i in range(gen.shape[0]):
        toks = [x for x in gen[i][P:].tolist() if x not in (eos, pad)]
        out.append(tok.decode(toks, skip_special_tokens=True).strip())
    return out


@torch.no_grad()
def independent(model, tok, mgr, items, device, max_new, max_plen):
    mgr.set_enabled(False)
    prompts = [build_prompt(tok, it["context"], it["question"]) for it in items]
    enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_plen)
    ids = enc["input_ids"].to(device); attn = enc["attention_mask"].to(device)
    G, P = ids.shape; eos = tok.eos_token_id; pad = tok.pad_token_id or eos
    gen = ids.clone(); cur = attn.clone(); fin = torch.zeros(G, dtype=torch.bool, device=device); pkv = None; nxt = gen
    for _ in range(max_new):
        out = model(input_ids=nxt, attention_mask=cur, past_key_values=pkv, use_cache=True); pkv = out.past_key_values
        t = out.logits[:, -1].argmax(-1); t = torch.where(fin, torch.full_like(t, pad), t); fin = fin | (t == eos)
        gen = torch.cat([gen, t.unsqueeze(1)], 1); cur = torch.cat([cur, (~fin).long().unsqueeze(1)], 1); nxt = t.unsqueeze(1)
        if fin.all():
            break
    mgr.set_enabled(True)
    return decode_texts(tok, gen, P, eos, pad)


def _masked_mean(hs, mask):
    m = mask.unsqueeze(-1).to(hs.dtype)
    return (hs * m).sum(1) / m.sum(1).clamp(min=1)


def _maxsim_scores(qhs, qattn, chs, cm):
    """Late-interaction (ColBERT-style) relevance R[i,j] of query i to context j.

    R[i,j] = mean over query-i tokens of ( max over context-j tokens of cosine ).
    qhs [G,Tq,d], qattn [G,Tq]; chs [G,Tc,d], cm [G,Tc]. Returns R [G,G] (float).
    """
    qn = torch.nn.functional.normalize(qhs.float(), dim=-1)          # [G,Tq,d]
    cn = torch.nn.functional.normalize(chs.float(), dim=-1)          # [G,Tc,d]
    sims = torch.einsum("itd,jsd->ijts", qn, cn)                     # [Gq,Gc,Tq,Tc]
    cmask = cm.view(1, cm.shape[0], 1, cm.shape[1])                  # [1,Gc,1,Tc]
    sims = sims.masked_fill(~cmask, float("-inf"))
    mx = sims.max(dim=3).values                                     # [Gq,Gc,Tq] max over ctx tokens
    qmask = qattn.bool().view(qattn.shape[0], 1, qattn.shape[1])    # [Gq,1,Tq]
    mx = mx.masked_fill(~qmask, 0.0)
    R = mx.sum(dim=2) / qmask.float().sum(dim=2).clamp(min=1)       # [Gq,Gc]
    return R


@torch.no_grad()
def multiquery(model, tok, mgr, items, device, max_new, max_plen, oracle_select=False,
               topk=0, no_offset=False, selector="maxsim", ape_temp=1.0, ape_scale=1.0):
    eos = tok.eos_token_id; pad = tok.pad_token_id or eos
    # ---- phase 1: capture each context's KV into the bank ----
    cp = [build_prompt(tok, it["context"], it["question"]) for it in items]
    enc = tok(cp, return_tensors="pt", padding=True, truncation=True, max_length=max_plen)
    cids = enc["input_ids"].to(device); cattn = enc["attention_mask"].to(device)
    cm = context_mask_for(tok, cp, [it["context"] for it in items], cids.shape[1], cattn, max_plen).to(device)
    if oracle_select:                              # only supporting contexts go into the bank
        for i, it in enumerate(items):
            if not it.get("has_supporting"):
                cm[i] = False
    mgr.set_enabled(True); mgr.exclude_self = False; mgr.set_allowed(None)
    mgr.start_capture(cm); mgr.set_valid(cattn)
    out1 = model(input_ids=cids, attention_mask=cattn, use_cache=False,
                 output_hidden_states=(topk > 0))
    off = 0 if no_offset else int(cattn.sum(1).max().item())   # put queries AFTER the bank
    # ---- phase 2: query sequences (empty passage) read the whole bank ----
    qp = [build_prompt(tok, "", it["question"]) for it in items]
    enc = tok(qp, return_tensors="pt", padding=True, truncation=True, max_length=max_plen)
    qids = enc["input_ids"].to(device); qattn = enc["attention_mask"].to(device)
    G, P = qids.shape
    if topk > 0 and topk < G:                      # selective: each query reads only its top-k contexts
        chs = out1.hidden_states[-1]                                  # [G, Tc, d]
        mgr.set_enabled(False)
        qhs = model(input_ids=qids, attention_mask=qattn, output_hidden_states=True).hidden_states[-1]
        mgr.set_enabled(True)
        if selector == "maxsim":                                     # late-interaction token-level
            R = _maxsim_scores(qhs, qattn, chs, cm)                  # [Gquery, Gctx]
        else:                                                        # mean-hidden cosine (cheap, weak)
            cn = torch.nn.functional.normalize(_masked_mean(chs, cm).float(), dim=-1)
            qn = torch.nn.functional.normalize(_masked_mean(qhs, qattn.bool()).float(), dim=-1)
            R = qn @ cn.t()
        topi = R.topk(topk, dim=1).indices
        allowed = torch.zeros(G, G, dtype=torch.bool, device=device)
        allowed.scatter_(1, topi, True)
        mgr.set_allowed(allowed)
    mgr.set_realign(ape_temp, ape_scale)
    mgr.start_use()
    base_pos = (qattn.long().cumsum(1) - 1).clamp(min=0)   # left-pad aware 0..
    pos = base_pos + off
    nxt_pos = off + qattn.sum(1)                            # [G] next position per sequence
    gen = qids.clone(); cur = qattn.clone(); fin = torch.zeros(G, dtype=torch.bool, device=device); pkv = None; nxt = gen
    for step in range(max_new):
        mgr.set_valid(cur)
        mgr.set_query_rows(cur.bool() if step == 0 else torch.ones(G, 1, dtype=torch.bool, device=device))
        pid = pos if step == 0 else nxt_pos.view(G, 1)
        out = model(input_ids=nxt, attention_mask=cur, position_ids=pid,
                    past_key_values=pkv, use_cache=True); pkv = out.past_key_values
        if step > 0:
            nxt_pos = nxt_pos + 1
        t = out.logits[:, -1].argmax(-1); t = torch.where(fin, torch.full_like(t, pad), t); fin = fin | (t == eos)
        gen = torch.cat([gen, t.unsqueeze(1)], 1); cur = torch.cat([cur, (~fin).long().unsqueeze(1)], 1); nxt = t.unsqueeze(1)
        if fin.all():
            break
    return decode_texts(tok, gen, P, eos, pad)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token; tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "left"
    tok.truncation_side = "left"     # keep the question/answer-cue at the end (Oracle union is long)
    device = "cuda"
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=torch.bfloat16, trust_remote_code=True,
        attn_implementation="sdpa").to(device).eval()
    if args.lora_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.lora_path).merge_and_unload().eval()
        log.info("merged LoRA adapter from %s", args.lora_path)
    nl = model.config.num_hidden_layers
    mgr = BatchCrossCache(list(range(nl))); mgr.register(model)
    log.info("MultiQuery cross-cache on all %d layers", nl)

    loader = load_multiquery_hotpot_groups if args.diff_questions else load_distributed_hotpot_groups
    groups = loader(
        split="validation", n_agents=args.n_agents, paragraphs_per_agent=args.paragraphs_per_agent,
        max_groups=args.num_eval_groups, seed=args.seed, only_bridge=True,
        require_min_supporting=2, cross_question_distractor_pool=True)
    log.info("groups=%d diff_questions=%s", len(groups), args.diff_questions)

    def newstat():
        return {"e": 0.0, "n": 0, "se": 0.0, "sn": 0, "ne": 0.0, "nn": 0}

    def accum(s, items, txts):
        for item, txt in zip(items, txts):
            a, _ = extract_answer(txt, "hotpot"); em = compute_em(a, item["references"])
            s["e"] += em; s["n"] += 1
            k = "s" if item.get("has_supporting") else "n"
            s[k + "e"] += em; s[k + "n"] += 1

    def plen(prompts):
        return sum(len(x) for x in tok(prompts, add_special_tokens=True)["input_ids"])

    sI, sM, sO = newstat(), newstat(), newstat()
    pf = {"indep": 0, "mq": 0, "oracle": 0}     # prefill tokens (context cost)
    tm = {"indep": 0.0, "mq": 0.0, "oracle": 0.0}   # wall-clock seconds (gi==0 skipped as warmup)
    n_timed = 0                                      # questions counted toward timing
    for gi, g in enumerate(groups):
        items = g["items"]; uni = union_context(items)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        it = independent(model, tok, mgr, items, device, args.max_new_tokens, args.max_prompt_length)
        torch.cuda.synchronize(); t1 = time.perf_counter()
        mq = multiquery(model, tok, mgr, items, device, args.max_new_tokens, args.max_prompt_length,
                        oracle_select=args.oracle_select, topk=args.topk, no_offset=args.no_offset,
                        selector=args.selector, ape_temp=args.ape_temp, ape_scale=args.ape_scale)
        torch.cuda.synchronize(); t2 = time.perf_counter()
        if gi > 0:
            tm["indep"] += t1 - t0; tm["mq"] += t2 - t1; n_timed += len(items)
        accum(sI, items, it); accum(sM, items, mq)
        # prefill-token accounting (the efficiency story: MQ == Oracle quality, G x cheaper)
        pf["indep"] += plen([build_prompt(tok, t["context"], t["question"]) for t in items])
        pf["mq"] += plen([build_prompt(tok, t["context"], t["question"]) for t in items]) \
                    + plen([build_prompt(tok, "", t["question"]) for t in items])
        pf["oracle"] += plen([build_prompt(tok, uni, t["question"]) for t in items])
        if args.with_oracle:
            oitems = [{**t, "context": uni} for t in items]
            torch.cuda.synchronize(); t3 = time.perf_counter()
            orc = independent(model, tok, mgr, oitems, device, args.max_new_tokens, args.max_prompt_length)
            torch.cuda.synchronize()
            if gi > 0:
                tm["oracle"] += time.perf_counter() - t3
            accum(sO, items, orc)
            if gi == 0:
                jj = min(2, len(items) - 1)
                log.info("ORACLE sample gold=%s oracle=%s | union_paras=%d",
                         items[jj]["references"][0], orc[jj][:40], len(uni.split("\n\n")))
        if gi == 0:
            j = min(2, len(items) - 1)
            log.info("sample gold=%s indep=%s multi=%s", items[j]["references"][0], it[j][:25], mq[j][:25])
    pc = lambda a, b: 100 * a / b if b else 0.0
    rows = [("Independent", sI), ("MultiQuery", sM)] + ([("Oracle", sO)] if args.with_oracle else [])
    for name, s in rows:
        log.info("%-12s overall=%.2f | has_supp=%.2f (n=%d) | no_supp=%.2f (n=%d)", name,
                 pc(s["e"], s["n"]), pc(s["se"], s["sn"]), s["sn"], pc(s["ne"], s["nn"]), s["nn"])
    log.info("PREFILL tokens: Independent=%d  MultiQuery=%d  Oracle=%d  | MQ/Oracle=%.3f",
             pf["indep"], pf["mq"], pf["oracle"], pf["mq"] / max(1, pf["oracle"]))
    nt = max(1, n_timed)
    log.info("TIME/question (s): Independent=%.4f  MultiQuery=%.4f%s  (n_timed=%d, warmup group skipped)",
             tm["indep"] / nt, tm["mq"] / nt,
             ("  Oracle=%.4f" % (tm["oracle"] / nt)) if args.with_oracle else "", n_timed)
    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump({"Independent": sI, "MultiQuery": sM, "Oracle": sO, "prefill": pf,
                   "time": tm, "n_timed": n_timed, "args": vars(args)}, f, indent=2)


if __name__ == "__main__":
    main()
