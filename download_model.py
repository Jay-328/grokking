from transformers import GPTNeoXForCausalLM, AutoTokenizer

model_id = "EleutherAI/pythia-70m"

print("正在下载并加载模型...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = GPTNeoXForCausalLM.from_pretrained(model_id)

print("模型下载完成！")
# 测试生成
inputs = tokenizer("Hello, I am a language model,", return_tensors="pt")
tokens = model.generate(**inputs, max_new_tokens=10)
print("测试生成结果:", tokenizer.decode(tokens[0]))