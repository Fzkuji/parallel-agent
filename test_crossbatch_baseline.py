"""Test if CrossBatchGenerator with enable_cross_batch=False matches baseline."""
import sys
sys.path.insert(0, '.')
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.cross_batch import CrossBatchGenerator, CrossBatchAttention

model_name = "Qwen/Qwen2.5-7B-Instruct"
device = "cuda:0"

# Load model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map=device)

# Create CSA module (V=0, out_proj=0, gate=-100)
cross_batch_module = CrossBatchAttention(hidden_size=model.config.hidden_size)
cross_batch_module.to(device)

generator = CrossBatchGenerator(
    model=model,
    tokenizer=tokenizer,
    cross_batch_module=cross_batch_module,
    mix_method="attention",
    mix_layer=-1,
    device=device,
)

# Test prompt
prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n"
inputs = tokenizer([prompt], return_tensors="pt").to(device)

# Method 1: Standard generate
with torch.no_grad():
    output1 = model.generate(**inputs, max_new_tokens=20, do_sample=False)
text1 = tokenizer.decode(output1[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

# Method 2: CrossBatchGenerator with enable_cross_batch=False
output2 = generator.generate(
    input_ids=inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    max_new_tokens=20,
    do_sample=False,
    enable_cross_batch=False,
)
text2 = tokenizer.decode(output2["sequences"][0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

print("=== Standard generate ===")
print(text1)
print("\n=== CrossBatchGenerator (disabled) ===")
print(text2)
print("\n=== Are they equal? ===")
print(text1 == text2)
