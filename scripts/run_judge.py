"""Step 2 of judge-based eval: load the 32B judge, score the prediction dumps from
bench_longbench (preds_<task>.jsonl), report judge-accuracy per arm ALONGSIDE qa_f1.
Run AFTER generation (separate process so the 8B gen model and 32B judge don't co-reside).
"""
import argparse, os, json, glob
from collections import defaultdict
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scripts.llm_judge import Judge
from scripts.bench_longbench import best_f1
from src.inference import extract_box_answer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--preds-dir", required=True, help="dir with preds_<task>.jsonl from bench_longbench")
    p.add_argument("--judge-path", default="/mnt/data/zichuanfu/models/Qwen2.5-32B-Instruct")
    p.add_argument("--bs", type=int, default=16)
    args = p.parse_args()

    rows = []
    for fp in sorted(glob.glob(os.path.join(args.preds_dir, "preds_*.jsonl"))):
        for line in open(fp):
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        print("no preds found in", args.preds_dir); return
    print(f"loaded {len(rows)} predictions; loading judge {args.judge_path} ...", flush=True)

    judge = Judge(args.judge_path)
    triples = [(r["question"], r["gold"], r["pred"]) for r in rows]
    verdicts = judge.judge_batch(triples, bs=args.bs)

    # aggregate by (task, arm): judge-acc and qa_f1 side by side
    agg = defaultdict(lambda: {"j": 0.0, "jn": 0, "f1": 0.0, "n": 0})
    for r, v in zip(rows, verdicts):
        key = (r["task"], r["arm"])
        ans, _ = extract_box_answer(r["pred"])
        agg[key]["f1"] += best_f1(ans, r["gold"]); agg[key]["n"] += 1
        if v is not None:
            agg[key]["j"] += v; agg[key]["jn"] += 1

    print(f"\n{'task':>10} {'arm':>14} {'qa_f1':>8} {'judge_acc':>10} {'n':>5}")
    for (task, arm), d in sorted(agg.items()):
        f1 = 100 * d["f1"] / max(1, d["n"])
        ja = 100 * d["j"] / max(1, d["jn"])
        print(f"{task:>10} {arm:>14} {f1:>8.2f} {ja:>10.2f} {d['n']:>5}")
    out = os.path.join(args.preds_dir, "judge_results.json")
    json.dump({f"{t}/{a}": {"qa_f1": round(100 * d["f1"] / max(1, d["n"]), 2),
                            "judge_acc": round(100 * d["j"] / max(1, d["jn"]), 2), "n": d["n"]}
               for (t, a), d in agg.items()}, open(out, "w"), indent=2)
    print("WROTE", out)


if __name__ == "__main__":
    main()
