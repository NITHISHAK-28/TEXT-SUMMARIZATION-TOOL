model_name = 'gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
model.eval()

def generate_text(prompt, max_length=100):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(inputs['input_ids'], max_length=max_length, do_sample=True, temperature=0.7)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Example use
prompt = "The impact of climate change on global agriculture"
print(generate_text(prompt))
