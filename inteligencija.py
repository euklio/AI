from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "gpt2"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def chat():
    print("Chatbot ready! Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        inputs = tokenizer(user_input, return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=100, pad_token_id=tokenizer.eos_token_id)

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Bot: {response}")

output = model.generate(
    max_length=100,
    repetition_penalty=8.2,
    temperature=0.8,
    top_k=50,
    pad_token_id=tokenizer.eos_token_id
)

response = tokenizer.decode(output[0], skip_special_tokens=True)
tokens = response.split()
unique_tokens = list(dict.fromkeys(tokens))
response = " ".join(unique_tokens)

chat()