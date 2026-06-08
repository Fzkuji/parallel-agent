import json, torch, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from transformers import AutoModelForCausalLM, AutoTokenizer
from scripts.bench_longbench import split_passages, oracle_passages, LB_PROMPT, best_f1

tok = AutoTokenizer.from_pretrained("/mnt/data/zichuanfu/models/Qwen3-8B")
m = AutoModelForCausalLM.from_pretrained("/mnt/data/zichuanfu/models/Qwen3-8B",
                                         dtype=torch.bfloat16, device_map="auto").eval()
data = [json.loads(l) for l in open("/mnt/data/zichuanfu/longbench_export/2wikimqa.jsonl")][:3]
for ex in data:
    ps = split_passages(ex["context"]); ctx = "\n\n".join(oracle_passages(ps, ex["answers"]))
    msgs = [{"role": "user", "content": LB_PROMPT.format(context=ctx, input=ex["input"])}]
    t = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(t, return_tensors="pt", truncation=True, max_length=16000).to(m.device)
    o = m.generate(**ids, max_new_tokens=1024, do_sample=False, pad_token_id=tok.pad_token_id)
    g = tok.decode(o[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
    has_close = "</think>" in g
    after = g.split("</think>")[-1].strip() if has_close else "(no </think>)"
    print("GOLD:", ex["answers"])
    print("GEN_TOKENS:", o.shape[1] - ids["input_ids"].shape[1], "CHARS:", len(g), "HAS_</think>:", has_close)
    print("ANSWER_AFTER_THINK:", repr(after[:200]))
    print("HEAD:", repr(g[:150]))
    print("---")
