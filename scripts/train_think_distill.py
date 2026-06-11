"""Self-distill thinking into the bank-read reader (stage 1 of the Qwen3 thinking plan).

Reads teacher trajectories from gen_think_traces.py (each = a question + the group's items +
the teacher's full <think>...</think><answer>X</answer> over the GOLD union). The student
captures the gold passages INDEPENDENTLY into the bank, then is teacher-forced to reproduce
the think+answer while reading only that independent bank. Tests whether a TRAINED reader can
use generation-time thinking to re-derive the cross-passage bridge independent encoding drops.

Decode-free (one teacher-forced forward per example) so no slow autoregressive rollout.
"""
import argparse, os, sys, json, logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.cross_batch.batch_crosscache import BatchCrossCache
from scripts.train_multiquery_lora import build_prompt, capture, use_loss

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--traj", required=True, help="JSONL from gen_think_traces.py")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--warm-start", default=None)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--max-prompt-length", type=int, default=2048)
    p.add_argument("--bank-grad", action="store_true", default=True)
    p.add_argument("--capture-all", action="store_true",
                   help="capture ALL group items (incl. non-supporting) like the answer-only arm, "
                        "for a fair thinking-vs-answer-target A/B")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def gold_items(group_items):
    """Capture only the supporting (gold) passages, independently, into the bank."""
    return [it for it in group_items if it.get("has_supporting", True)] or group_items


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token; tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "left"; tok.truncation_side = "left"
    device = "cuda"
    base = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=torch.bfloat16, trust_remote_code=True,
        attn_implementation="sdpa").to(device)
    if args.warm_start:
        model = PeftModel.from_pretrained(base, args.warm_start, is_trainable=True)
        log.info("warm-start from %s", args.warm_start)
    else:
        lora = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
                          target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM")
        model = get_peft_model(base, lora)
    model.train(); model.print_trainable_parameters()
    nl = model.config.num_hidden_layers
    mgr = BatchCrossCache(list(range(nl))); mgr.register(model)

    traj = [json.loads(l) for l in open(args.traj) if l.strip()]
    log.info("loaded %d think trajectories", len(traj))

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    step = 0; run = 0.0
    for ep in range(args.epochs):
        for t in traj:
            gi = t["group_items"] if args.capture_all else gold_items(t["group_items"])
            # capture gold passages independently (with grad on the capture path)
            if args.bank_grad:
                off, _, _ = capture(model, mgr, tok, gi, device, args.max_prompt_length, False)
            else:
                with torch.no_grad():
                    off, _, _ = capture(model, mgr, tok, gi, device, args.max_prompt_length, False)
            mgr.set_allowed(None)
            # one query, target = the teacher's full think+answer trajectory
            item = {"question": t["question"], "references": t["references"],
                    "think_answer": t.get("think_answer")}
            loss = use_loss(model, mgr, tok, [item], device, off, args.max_prompt_length)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step(); opt.zero_grad()
            run += loss.item(); step += 1
            mgr.bank = {}
            if step % 8 == 0:
                torch.cuda.empty_cache()
            if step % 25 == 0:
                log.info("ep%d step%d loss=%.4f", ep, step, run / 25); run = 0.0
    model.save_pretrained(args.output_dir)
    tok.save_pretrained(args.output_dir)
    log.info("saved think-distill LoRA to %s", args.output_dir)


if __name__ == "__main__":
    main()
