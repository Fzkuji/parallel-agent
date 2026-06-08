"""Fast vLLM generation for the full-attention arms (concat/oracle) on Qwen3 with thinking.
Batched -> 10-50x faster than HF generate, so thinking-mode eval is actually runnable.
Dumps (question, gold, pred) JSONL for the offline 32B LLM-judge. Bank-read (ape/ours) arms
use the custom hook and can't go through vLLM — this covers only the full-attention arms.
"""
import argparse, os, json, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scripts.bench_longbench import split_passages, oracle_passages, LB_PROMPT, best_f1
from src.inference import extract_box_answer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--tasks", default="2wikimqa,hotpotqa,musique")
    p.add_argument("--data-dir", default="/mnt/data/zichuanfu/longbench_export")
    p.add_argument("--num-q", type=int, default=200)
    p.add_argument("--max-new", type=int, default=1024)
    p.add_argument("--gold-only", action="store_true")
    p.add_argument("--dump", required=True)
    p.add_argument("--gpu-mem", type=float, default=0.85)
    p.add_argument("--max-model-len", type=int, default=32768)
    args = p.parse_args()

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model_path)
    llm = LLM(model=args.model_path, dtype="bfloat16", gpu_memory_utilization=args.gpu_mem,
              max_model_len=args.max_model_len, enforce_eager=True)  # eager: avoids CUDA-graph hang
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_new)

    df = open(args.dump, "w")
    arm = "oracle" if args.gold_only else "concat"
    for task in [t.strip() for t in args.tasks.split(",") if t.strip()]:
        data = [json.loads(l) for l in open(os.path.join(args.data_dir, f"{task}.jsonl")) if l.strip()][: args.num_q]
        prompts, metas = [], []
        for ex in data:
            ps = split_passages(ex["context"])
            ctx = "\n\n".join(oracle_passages(ps, ex["answers"]) if args.gold_only else ps)
            msgs = [{"role": "user", "content": LB_PROMPT.format(context=ctx, input=ex["input"])}]
            prompts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
            metas.append((ex["input"], ex["answers"]))
        outs = llm.generate(prompts, sp)
        sc = 0.0
        for (q, gold), o in zip(metas, outs):
            pred = o.outputs[0].text
            ans, _ = extract_box_answer(pred)
            sc += best_f1(ans, gold)
            df.write(json.dumps({"task": task, "arm": arm, "question": q, "gold": gold, "pred": pred}) + "\n")
        print(f"{task}: qa_f1={100*sc/len(metas):.2f}  (n={len(metas)}, arm={arm})", flush=True)
    df.close()
    print("WROTE", args.dump)


if __name__ == "__main__":
    main()
