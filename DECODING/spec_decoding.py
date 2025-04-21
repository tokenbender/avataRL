import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MAIN_MODEL = "unsloth/Llama-3.2-3B-Instruct"
ASSIST_MODEL = "openai-community/gpt2-large"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

quant_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
)

tokenizer = AutoTokenizer.from_pretrained(MAIN_MODEL)
assistant_tokenizer = AutoTokenizer.from_pretrained(ASSIST_MODEL)

main_model = AutoModelForCausalLM.from_pretrained(
    MAIN_MODEL,
    quantization_config=quant_config,
    device_map="auto"
)

assist_model = AutoModelForCausalLM.from_pretrained(
    ASSIST_MODEL,
    quantization_config=quant_config,
    device_map="auto"
)

prompts = [
    "Explain the significance of the Higgs boson in simple terms.",
    "Write a short recipe for a vegan chocolate cake.",
    "Summarize the plot of Romeo and Juliet in one paragraph.",
    "Draft a polite follow-up email for a job application.",
    "Generate a motivational quote about perseverance."
]

gen_kwargs = {
    "max_length": 60,
    "do_sample": True,
    "top_k": 50,
    "top_p": 0.95,
}

def to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items()}

def benchmark(prompt: str, speculative: bool = False):
    batch = tokenizer(prompt, return_tensors="pt")
    batch = to_device(batch, device)
    extras = {}
    if speculative:
        extras.update({
            "assistant_model": assist_model,
            "tokenizer": tokenizer,
            "assistant_tokenizer": assistant_tokenizer,
            "num_assistant_tokens": 20,
        })
    start = time.time()
    output_ids = main_model.generate(**batch, **gen_kwargs, **extras)
    elapsed = time.time() - start
    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return text, elapsed

for idx, prompt in enumerate(prompts, 1):
    std_text, std_time = benchmark(prompt, speculative=False)
    spec_text, spec_time = benchmark(prompt, speculative=True)
    speedup = std_time / spec_time if spec_time > 0 else float("inf")
    print(f"\n=== Prompt {idx} ===")
    print(prompt)
    print(f"\n• Standard time: {std_time:.2f}s")
    print(f"• Speculative time: {spec_time:.2f}s")
    print(f"• Speed‑up: {speedup:.2f}×")
