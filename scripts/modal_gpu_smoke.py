import modal

app = modal.App("gpu-smoke-test")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers",
        "accelerate",
    )
)

@app.cls(gpu="L4", image=image, timeout=300)
class ModelRunner:
    @modal.enter()
    def load_model(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        model_id = "Qwen/Qwen2.5-3B-Instruct"
        print(f"Loading {model_id} on GPU...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
        )
        print("Model loaded successfully!")

    @modal.method()
    def generate_action(self, prompt: str) -> str:
        messages = [
            {"role": "system", "content": "You are playing Rock-Paper-Scissors. Respond in JSON: {\"action\": \"ROCK\" | \"PAPER\" | \"SCISSORS\"}."},
            {"role": "user", "content": prompt}
        ]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt").to("cuda")
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=25,
            temperature=0.7,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

@app.local_entrypoint()
def main():
    runner = ModelRunner()
    result = runner.generate_action.remote("The previous round was ROCK. What is your move?")
    print("Received generation:", result)
