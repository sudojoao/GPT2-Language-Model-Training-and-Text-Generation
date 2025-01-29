from transformers import AutoModelForCausalLM, AutoTokenizer
import os

model_dir = "./results/latest_checkpoint"

if not os.path.isdir(model_dir):
    raise ValueError(f"Model directory '{model_dir}' does not exist. Please check the path.")

model = AutoModelForCausalLM.from_pretrained(model_dir)
tokenizer = AutoTokenizer.from_pretrained(model_dir)

model.config.pad_token_id = model.config.eos_token_id

input_text = "Once upon a time"
input_ids = tokenizer.encode(input_text, return_tensors="pt")
attention_mask = input_ids.ne(model.config.pad_token_id).long()
output = model.generate(input_ids, attention_mask=attention_mask, max_length=100, num_return_sequences=1, do_sample=True, top_k=50)

generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)