"""Offline LLM-as-a-judge for QA correctness (2026 standard), replacing/complementing token-F1.

Token-F1 systematically under-scores semantically-correct-but-verbose answers (e.g. Qwen3
thinking: "Ozalj, present day Croatia" vs gold "Ozalj"); LLM-judge correlates ~0.85 with
humans vs 0.40 for F1 (arXiv 2504.11972, ACL GEM 2026). We grade with a SimpleQA-style
question+gold+candidate -> CORRECT/INCORRECT prompt.

Design choices forced by the 2026 findings:
  - Judge = Qwen2.5-32B (NOT Qwen3): never judge a thinking model with the same family —
    Qwen3-Thinking self-preference bias is up to β=0.54.
  - STRIP <think> from the candidate before judging: avoids the "superficial reflection bias"
    where judges over-credit text containing "wait, let me think...".
  - The judge is PLAUSIBLE-BUT-UNPROVEN at 32B (2026 refuted a hard 30B floor); validate on a
    small human/EM-agreement slice and report judge-accuracy ALONGSIDE qa_f1, never instead.
"""
import re

# OpenAI SimpleQA grader, trimmed to a binary correct/incorrect for short-answer multi-hop QA.
JUDGE_PROMPT = """You are grading whether a predicted answer to a question is correct, given the gold (correct) answer(s).

Question: {question}
Gold answer(s): {gold}
Predicted answer: {pred}

Grade the predicted answer CORRECT if it contains the gold answer's key information and does not contradict it. A correct answer may be phrased differently, include extra correct detail, or be more verbose than the gold — only the semantic meaning matters; capitalization, punctuation, word order, and added correct context do not. Grade it INCORRECT if it misses the gold answer, states something that contradicts the gold, or does not actually answer the question.

Respond with exactly one word: CORRECT or INCORRECT."""


def strip_think(text):
    if "</think>" in text:
        text = text.split("</think>")[-1]
    # also strip an <answer> wrapper if present (judge sees the bare answer)
    m = re.search(r"<answer>(.*?)</answer>", text, re.IGNORECASE | re.DOTALL)
    if m:
        text = m.group(1)
    return text.strip()


def build_judge_messages(question, gold_list, pred):
    gold = " OR ".join(g for g in gold_list if g.strip())
    return [{"role": "user",
             "content": JUDGE_PROMPT.format(question=question, gold=gold, pred=strip_think(pred))}]


def parse_verdict(judge_out):
    """Return 1.0 for CORRECT, 0.0 for INCORRECT, None if unparseable."""
    t = judge_out.strip().upper()
    # take the LAST mention (a thinking-disabled 32B should answer directly, but be robust)
    has_c = "CORRECT" in t and "INCORRECT" not in t.split("CORRECT")[-1][:12]
    if t.startswith("CORRECT") or (re.search(r"\bCORRECT\b", t) and not re.search(r"\bINCORRECT\b", t)):
        return 1.0
    if re.search(r"\bINCORRECT\b", t):
        return 0.0
    if "CORRECT" in t:
        return 1.0
    return None


class Judge:
    """Batched Qwen2.5-32B judge. Loads once, judges (question, gold, pred) triples."""

    def __init__(self, model_path="/mnt/data/zichuanfu/models/Qwen2.5-32B-Instruct", device="cuda"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(model_path)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=torch.bfloat16, attn_implementation="sdpa", device_map="auto").eval()
        self.torch = torch

    def judge_batch(self, triples, bs=16):
        """triples = [(question, gold_list, pred), ...] -> [1.0/0.0/None]."""
        out = []
        for i in range(0, len(triples), bs):
            chunk = triples[i:i + bs]
            texts = [self.tok.apply_chat_template(build_judge_messages(q, g, p),
                                                  tokenize=False, add_generation_prompt=True)
                     for q, g, p in chunk]
            enc = self.tok(texts, return_tensors="pt", padding=True, truncation=True,
                           max_length=4096).to(self.model.device)
            with self.torch.no_grad():
                gen = self.model.generate(**enc, max_new_tokens=8, do_sample=False,
                                          pad_token_id=self.tok.pad_token_id)
            for j, row in enumerate(gen):
                dec = self.tok.decode(row[enc["input_ids"].shape[1]:], skip_special_tokens=True)
                out.append(parse_verdict(dec))
        return out
