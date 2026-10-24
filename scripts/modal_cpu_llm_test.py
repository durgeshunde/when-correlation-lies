import modal
import time

app = modal.App("cpu-llm-speed-test")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers",
        "accelerate",
    )
)

@app.cls(cpu=4, memory=8192, image=image, timeout=300)
class CPULLM:
    @modal.enter()
    def load(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        self.model_id = "Qwen/Qwen2.5-1.5B-Instruct"
        print(f"Loading {self.model_id} on CPU...")
        t0 = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        )
        print(f"Model loaded in {time.time() - t0:.2f}s")

    @modal.method()
    def generate(self, prompt: str) -> dict:
        t0 = time.time()
        messages = [
            {"role": "system", "content": "You are playing Rock-Paper-Scissors. Choose one: ROCK, PAPER, or SCISSORS. Respond in JSON: {\"action\": \"ROCK\" | \"PAPER\" | \"SCISSORS\", \"rationale\": \"short phrase\"}."},
            {"role": "user", "content": prompt}
        ]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt")
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=25,
            temperature=0.7,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        gen_time = time.time() - t0
        generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        out_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return {"text": out_text, "tokens": len(generated_tokens), "time_sec": gen_time}

@app.local_entrypoint()
def main():
    runner = CPULLM()
    res = runner.generate.remote("The previous round was ROCK. What is your move?")
    print("Result:", res)
