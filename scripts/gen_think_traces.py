"""Generate self-distillation thinking trajectories with vLLM (fast).

Teacher = Qwen3-8B full-attention over GOLD-ONLY passages, thinking enabled. For each
training question we generate a <think>...</think><answer>X</answer> trajectory and KEEP it
only if the extracted answer hits gold (the teacher solved the multi-hop in-context). The
bank-read student is later distilled to reproduce this trajectory while reading the
INDEPENDENT bank — testing whether a TRAINED reader can use generation-time thinking to
re-derive the cross-passage bridge that independent encoding drops.

Uses the SAME student bank-read system prompt (build_single_prompt, answer-tag) so the
distilled <answer> contract matches the extractor — NOT LB_PROMPT.
"""
import argparse, os, json, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.datasets import load_multiquery_hotpot_groups
from scripts.bench_longbench import best_f1
from src.inference import extract_box_answer
from scripts.train_multiquery_lora import build_prompt, _union


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher-path", default="/mnt/data/zichuanfu/models/Qwen3-8B")
    p.add_argument("--num-groups", type=int, default=1500)
    p.add_argument("--mix-agents", default="2,4")
    p.add_argument("--paragraphs-per-agent", type=int, default=6)
    p.add_argument("--max-new", type=int, default=1024)
    p.add_argument("--gpu-mem", type=float, default=0.85)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cot", action="store_true",
                   help="CoT teacher for non-thinking models (Qwen2.5): append a step-by-step "
                        "instruction to the teacher prompt, keep on answer-correct only (no <think> "
                        "tag required). Stored question stays PLAIN so the student matches inference.")
    args = p.parse_args()

    Gs = [int(x) for x in args.mix_agents.split(",")]
    per = max(1, args.num_groups // len(Gs))
    groups = []
    for gi, G in enumerate(Gs):
        groups += load_multiquery_hotpot_groups(
            split="train", n_agents=G, paragraphs_per_agent=args.paragraphs_per_agent,
            max_groups=per, seed=args.seed + gi, only_bridge=True,
            require_min_supporting=2, cross_question_distractor_pool=True)
    # flatten to (question, gold-only union, answer, items) per query
    items = []
    for g in groups:
        gi = [it for it in g["items"] if it.get("has_supporting", True)]
        uni = _union(gi if gi else g["items"])
        for it in g["items"]:
            items.append({"question": it["question"], "gold_union": uni,
                          "references": it["references"], "group_items": g["items"]})
    print(f"{len(items)} queries from {len(groups)} groups", flush=True)

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.teacher_path)
    llm = LLM(model=args.teacher_path, dtype="bfloat16", gpu_memory_utilization=args.gpu_mem,
              max_model_len=8192, enforce_eager=True)
    sp = SamplingParams(temperature=0.6, top_p=0.95, max_tokens=args.max_new)

    # build teacher prompts = student bank-read prompt but over the gold union (full attention)
    COT = ("\nThink step by step about how the passages connect before giving your final answer "
           "in <answer></answer> tags.")
    prompts = [build_prompt(tok, it["gold_union"], it["question"] + (COT if args.cot else ""))
               for it in items]
    outs = llm.generate(prompts, sp)

    kept = 0
    df = open(args.out, "w")
    for it, o in zip(items, outs):
        traj = o.outputs[0].text
        ans, _ = extract_box_answer(traj)
        if args.cot:
            ok = best_f1(ans, it["references"]) >= 0.5 and "<answer>" in traj
        else:
            ok = best_f1(ans, it["references"]) >= 0.5 and "</think>" in traj and "<answer>" in traj
        if ok:
            # store the full think+answer (strip leading whitespace; keep <think>..</answer>)
            ta = traj[traj.find("<think>"):] if "<think>" in traj else traj
            df.write(json.dumps({"question": it["question"], "references": it["references"],
                                 "group_items": it["group_items"], "think_answer": ta.strip()}) + "\n")
            kept += 1
    df.close()
    print(f"KEPT {kept}/{len(items)} teacher-correct trajectories -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
