#!/usr/bin/env python3
"""Step 3: Eval for full-FT + multi-layer CSA hook architecture.

Loads a full-FT checkpoint (base model + CSA), registers hooks, then
runs greedy decoding twice per group:
  - Independent: csa_module.set_enabled(False) -> hooks bypass injection
  - CSA-v2:      csa_module.set_enabled(True)  -> hooks inject cross-batch info

DDP usage:
    torchrun --standalone --nproc_per_node=8 scripts/eval_csa_hook.py \\
        --dataset dhotpot --model-path /YOUR_PATH/Qwen2.5-7B-Instruct \\
        --checkpoint ./out/dhotpot_csa_pretrained/best_model.pt \\
        --num-eval-groups 200 --n-agents 4 --paragraphs-per-agent 9 \\
        --output-dir ./out/eval_dhotpot_hook
"""

import argparse
import json
import logging
import os
import sys

import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.cross_batch.attention import CrossSequenceAttentionV2
from src.cross_batch.multi_layer_hook import MultiLayerCSAModule, default_layer_indices
from src.datasets import load_distributed_hotpot_groups, load_squad_groups
from src.evaluation.basic import compute_em, compute_f1
from src.inference import extract_answer
from src.models import Question
from src.prompts import build_single_prompt
from src.templates import build_chat_prompt

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--checkpoint", required=True,
                   help="Full-FT checkpoint (model + CSA state)")
    p.add_argument("--dataset", choices=["squad", "dhotpot"], required=True)
    p.add_argument("--num-eval-groups", type=int, default=200)
    p.add_argument("--group-sizes", type=str, default="1,2,3,4,5",
                   help="for squad only: comma-separated G values")
    p.add_argument("--n-agents", type=int, default=4)
    p.add_argument("--paragraphs-per-agent", type=int, default=9)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--max-prompt-length", type=int, default=1536)
    p.add_argument("--csa-every", type=int, default=4)
    p.add_argument("--num-heads", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


def setup_distributed():
    if "LOCAL_RANK" not in os.environ:
        return 0, 1, -1
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ.get("RANK", local_rank))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return rank, world_size, local_rank


def make_questions(items_or_questions, dataset):
    if dataset == "squad":
        return [
            Question(qid=f"Q{i}", text=q["text"], priority=1.0,
                     answer_tokens=q.get("answer_tokens", 12), type_hint=None,
                     references=q.get("references", []))
            for i, q in enumerate(items_or_questions)
        ]
    return [
        Question(qid=f"A{i}", text=it["question"], priority=1.0,
                 answer_tokens=it.get("answer_tokens", 12), type_hint=None,
                 references=it.get("references", []))
        for i, it in enumerate(items_or_questions)
    ]


def build_prompts(tokenizer, items, dataset, shared_context=None):
    prompts = []
    for it, q in zip(items, make_questions(items, dataset)):
        ctx = shared_context if dataset == "squad" else it["context"]
        ds = "hotpot" if dataset == "dhotpot" else dataset
        sp, up = build_single_prompt(ctx, q, dataset=ds)
        prompts.append(build_chat_prompt(tokenizer, up, system_prompt=sp))
    return prompts


@torch.no_grad()
def greedy_generate(
    model, tokenizer, csa_module, prompts, max_new_tokens, max_prompt_length, device,
    enable_csa: bool,
):
    """Greedy decode with hook-based CSA optionally enabled.

    Forward the entire batch (size G) through the model — the hooks fire at
    every selected layer, with cross-batch attention if enabled.
    """
    enc = tokenizer(prompts, return_tensors="pt", padding=True,
                    truncation=True, max_length=max_prompt_length)
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    G, prompt_len = input_ids.shape

    csa_module.set_enabled(enable_csa)
    csa_module.set_context(question_emb=None, attention_mask=attention_mask)

    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id or eos_id

    generated = input_ids.clone()
    cur_mask = attention_mask.clone()
    finished = torch.zeros(G, dtype=torch.bool, device=device)

    past_key_values = None
    next_input = generated

    for step in range(max_new_tokens):
        # Re-set context attention mask each step so hook mean-pool sees the
        # current padded sequence correctly.
        csa_module.set_context(question_emb=None, attention_mask=cur_mask)
        outputs = model(
            input_ids=next_input,
            attention_mask=cur_mask,
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values
        logits = outputs.logits[:, -1, :]  # [G, V]
        next_tokens = logits.argmax(dim=-1)
        # Mask tokens for finished sequences
        next_tokens = torch.where(finished, torch.full_like(next_tokens, pad_id), next_tokens)
        finished = finished | (next_tokens == eos_id)

        generated = torch.cat([generated, next_tokens.unsqueeze(1)], dim=1)
        cur_mask = torch.cat([cur_mask, (~finished).long().unsqueeze(1)], dim=1)
        next_input = next_tokens.unsqueeze(1)

        if finished.all():
            break

    csa_module.clear_context()
    csa_module.set_enabled(True)
    return generated, prompt_len


def _all_reduce_dict(d, device):
    if not dist.is_initialized():
        return d
    keys = sorted(d.keys())
    tensor = torch.tensor([float(d[k]) for k in keys], device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    for i, k in enumerate(keys):
        d[k] = tensor[i].item()
    return d


@torch.no_grad()
def run_squad_condition(model, tokenizer, csa_module, eval_groups, group_size,
                        enable_csa, max_new_tokens, max_prompt_len, device,
                        rank=0, world_size=1):
    sums = {"em_sum": 0.0, "f1_sum": 0.0, "n": 0.0}
    for gi, group in enumerate(eval_groups):
        if gi % world_size != rank:
            continue
        if len(group["questions"]) < group_size:
            continue
        items = group["questions"][:group_size]
        prompts = build_prompts(tokenizer, items, dataset="squad",
                                shared_context=group["context"])
        seqs, plen = greedy_generate(
            model, tokenizer, csa_module, prompts,
            max_new_tokens, max_prompt_len, device, enable_csa,
        )
        for i, q in enumerate(make_questions(items, "squad")):
            tokens = []
            for t in seqs[i][plen:].tolist():
                if t in (tokenizer.eos_token_id, tokenizer.pad_token_id):
                    break
                tokens.append(t)
            text = tokenizer.decode(tokens, skip_special_tokens=True).strip()
            ans, _ = extract_answer(text, "squad")
            sums["em_sum"] += compute_em(ans, q.references)
            sums["f1_sum"] += compute_f1(ans, q.references)
            sums["n"] += 1.0
    sums = _all_reduce_dict(sums, device)
    n = max(sums["n"], 1.0)
    return {"em": 100 * sums["em_sum"] / n,
            "f1": 100 * sums["f1_sum"] / n,
            "n": int(sums["n"])}


@torch.no_grad()
def run_dhotpot_condition(model, tokenizer, csa_module, eval_groups, enable_csa,
                          max_new_tokens, max_prompt_len, device,
                          rank=0, world_size=1):
    sums = {k: 0.0 for k in [
        "all_em", "all_f1", "supp_em", "supp_f1", "nosupp_em", "nosupp_f1",
        "n", "supp_n", "nosupp_n",
    ]}
    for gi, group in enumerate(eval_groups):
        if gi % world_size != rank:
            continue
        items = group["items"]
        prompts = build_prompts(tokenizer, items, dataset="dhotpot")
        seqs, plen = greedy_generate(
            model, tokenizer, csa_module, prompts,
            max_new_tokens, max_prompt_len, device, enable_csa,
        )
        for i, it in enumerate(items):
            tokens = []
            for t in seqs[i][plen:].tolist():
                if t in (tokenizer.eos_token_id, tokenizer.pad_token_id):
                    break
                tokens.append(t)
            text = tokenizer.decode(tokens, skip_special_tokens=True).strip()
            ans, _ = extract_answer(text, "hotpot")
            em = compute_em(ans, it["references"])
            f1 = compute_f1(ans, it["references"])
            sums["all_em"] += em; sums["all_f1"] += f1; sums["n"] += 1.0
            if it.get("has_supporting"):
                sums["supp_em"] += em; sums["supp_f1"] += f1; sums["supp_n"] += 1.0
            else:
                sums["nosupp_em"] += em; sums["nosupp_f1"] += f1; sums["nosupp_n"] += 1.0
    sums = _all_reduce_dict(sums, device)
    pct = lambda a, b: 100 * a / b if b else 0.0
    return {
        "em": pct(sums["all_em"], sums["n"]),
        "f1": pct(sums["all_f1"], sums["n"]),
        "n": int(sums["n"]),
        "supp_em": pct(sums["supp_em"], sums["supp_n"]),
        "supp_f1": pct(sums["supp_f1"], sums["supp_n"]),
        "supp_n": int(sums["supp_n"]),
        "nosupp_em": pct(sums["nosupp_em"], sums["nosupp_n"]),
        "nosupp_f1": pct(sums["nosupp_f1"], sums["nosupp_n"]),
        "nosupp_n": int(sums["nosupp_n"]),
    }


def main():
    args = parse_args()
    rank, world_size, local_rank = setup_distributed()
    is_main = rank == 0

    if is_main:
        os.makedirs(args.output_dir, exist_ok=True)
        log.info("DDP eval: rank=%d world_size=%d local_rank=%d",
                 rank, world_size, local_rank)
        log.info("Loading model %s", args.model_path)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    if local_rank >= 0:
        device = f"cuda:{local_rank}"
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
        ).to(device)
    else:
        device = "cuda"
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, torch_dtype=torch.bfloat16,
            device_map="auto", trust_remote_code=True,
        )

    hidden_size = model.config.hidden_size
    num_layers = model.config.num_hidden_layers
    layer_indices = default_layer_indices(num_layers, every=args.csa_every)
    if is_main:
        log.info("CSA layer indices: %s", layer_indices)

    csa = CrossSequenceAttentionV2(
        hidden_size=hidden_size, num_heads=args.num_heads,
        use_gate=True, adaptive_top_k=True,
    )
    csa_module = MultiLayerCSAModule(csa=csa, layer_indices=layer_indices)
    csa_module.register(model)

    if is_main:
        log.info("Loading checkpoint %s", args.checkpoint)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    if "model" in ckpt:
        missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
        if is_main:
            if missing:
                log.warning("Model missing keys: %d", len(missing))
            if unexpected:
                log.warning("Model unexpected keys: %d", len(unexpected))
    if "csa_module" in ckpt:
        csa_module.load_state_dict(ckpt["csa_module"], strict=False)

    model.to(device).to(torch.bfloat16); model.eval()
    csa_module.to(device).to(torch.bfloat16); csa_module.eval()

    results = {}
    if args.dataset == "squad":
        group_sizes = [int(x) for x in args.group_sizes.split(",")]
        max_g = max(group_sizes)
        if is_main:
            log.info("Loading SQuAD val groups (need >=%d Qs/ctx)", max_g)
        eval_groups = load_squad_groups(
            split="validation",
            min_questions=max_g, max_questions=max_g,
            max_contexts=args.num_eval_groups,
            fixed_question_count=max_g, seed=args.seed,
        )
        if is_main:
            log.info("Got %d eval groups, sharded across %d ranks",
                     len(eval_groups), world_size)
        for G in group_sizes:
            for cond, enable in [("Independent", False), ("CSA-v2", True)]:
                r = run_squad_condition(model, tokenizer, csa_module, eval_groups, G,
                                        enable, args.max_new_tokens,
                                        args.max_prompt_length, device,
                                        rank=rank, world_size=world_size)
                results[f"G{G}_{cond}"] = r
                if is_main:
                    log.info("G=%d  %-12s  EM=%.2f  F1=%.2f  n=%d",
                             G, cond, r["em"], r["f1"], r["n"])
        if is_main:
            log.info("\n=== SUMMARY EM (%%) ===")
            log.info("%-13s | %s", "Strategy", "  ".join(f"G={g}" for g in group_sizes))
            for cond in ["Independent", "CSA-v2"]:
                row = [f"{results[f'G{g}_{cond}']['em']:5.2f}" for g in group_sizes]
                log.info("%-13s | %s", cond, "  ".join(row))
    else:
        if is_main:
            log.info("Loading dhotpot val groups (n_agents=%d)", args.n_agents)
        eval_groups = load_distributed_hotpot_groups(
            split="validation", n_agents=args.n_agents,
            paragraphs_per_agent=args.paragraphs_per_agent,
            max_groups=args.num_eval_groups, seed=args.seed,
            only_bridge=True, require_min_supporting=2,
            cross_question_distractor_pool=True,
        )
        if is_main:
            log.info("Got %d eval groups (%d total queries), sharded across %d ranks",
                     len(eval_groups), len(eval_groups) * args.n_agents, world_size)
        for cond, enable in [("Independent", False), ("CSA-v2", True)]:
            r = run_dhotpot_condition(model, tokenizer, csa_module, eval_groups,
                                      enable, args.max_new_tokens,
                                      args.max_prompt_length, device,
                                      rank=rank, world_size=world_size)
            results[cond] = r
            if is_main:
                log.info("%-12s overall EM=%.2f F1=%.2f  n=%d",
                         cond, r["em"], r["f1"], r["n"])
                log.info("              has_supp EM=%.2f F1=%.2f  n=%d",
                         r["supp_em"], r["supp_f1"], r["supp_n"])
                log.info("              no_supp  EM=%.2f F1=%.2f  n=%d",
                         r["nosupp_em"], r["nosupp_f1"], r["nosupp_n"])

        if is_main:
            log.info("\n=== SUMMARY ===")
            log.info("%-12s | %-12s | %-12s | %-12s",
                     "Strategy", "overall EM", "has_supp EM", "no_supp EM")
            for cond in ["Independent", "CSA-v2"]:
                r = results[cond]
                log.info("%-12s | %-12.2f | %-12.2f | %-12.2f",
                         cond, r["em"], r["supp_em"], r["nosupp_em"])
            i, c = results["Independent"], results["CSA-v2"]
            log.info("Δ (CSA-Independent): overall=%+.2f  has_supp=%+.2f  no_supp=%+.2f",
                     c["em"] - i["em"], c["supp_em"] - i["supp_em"],
                     c["nosupp_em"] - i["nosupp_em"])

    if is_main:
        out = os.path.join(args.output_dir, "results.json")
        with open(out, "w") as f:
            json.dump({"args": vars(args), "results": results}, f, indent=2)
        log.info("Saved %s", out)

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
