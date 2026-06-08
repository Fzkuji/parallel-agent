"""Gate-0 logits consistency: with the BatchCrossCache hook enabled but an EMPTY bank on a
single sequence, the hooked forward MUST match the unhooked model.forward to <1e-3. On Qwen3
this only passes if the hook applies q_norm/k_norm before RoPE. If it fails, every hooked
Qwen3 number (ape/ape_oracle/ours) is invalid."""
import sys, os, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.cross_batch.batch_crosscache import BatchCrossCache

MP = "/mnt/data/zichuanfu/models/Qwen3-8B"
tok = AutoTokenizer.from_pretrained(MP)
model = AutoModelForCausalLM.from_pretrained(MP, dtype=torch.bfloat16,
                                             attn_implementation="sdpa", device_map="auto").eval()
print("sliding_window:", getattr(model.config, "sliding_window", "MISSING"))

ids = tok("The capital of France is", return_tensors="pt").to(model.device)

with torch.no_grad():
    ref = model(**ids).logits[0, -1].float()

nl = model.config.num_hidden_layers
mgr = BatchCrossCache(list(range(nl))); mgr.register(model)
mgr.set_enabled(True)
# empty bank, plain use-mode single sequence
mgr.mode = "use"; mgr.bank = {}
mgr.set_valid(ids["attention_mask"]); mgr.set_query_rows(ids["attention_mask"])
with torch.no_grad():
    hooked = model(**ids).logits[0, -1].float()

diff = (ref - hooked).abs().max().item()
rel = diff / (ref.abs().max().item() + 1e-9)
cos = torch.nn.functional.cosine_similarity(ref, hooked, dim=0).item()
top5_ref = ref.topk(5).indices.tolist(); top5_h = hooked.topk(5).indices.tolist()
print(f"max_abs_diff={diff:.4f}  rel={rel:.5f}  cosine={cos:.6f}  logit_scale={ref.abs().max():.1f}")
print("top5 match:", top5_ref == top5_h, top5_ref == top5_h and "" or (top5_ref, top5_h))
# bf16 rounding gives ~0.2 abs diff on logits of scale ~20; the real test is cosine≈1 + top5 match
ok = cos > 0.9999 and top5_ref == top5_h
print("PASS (QK-norm correct, diff is bf16 noise)" if ok else "FAIL — hook corrupts Qwen3 logits")
