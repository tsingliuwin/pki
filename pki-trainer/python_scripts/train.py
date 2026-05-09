import os
import argparse
import torch

# Unsloth MUST be imported first for optimizations, but only if we have a GPU
HAS_CUDA = torch.cuda.is_available()

if HAS_CUDA:
    from unsloth import FastLanguageModel
    print("[Python] GPU detected. Using Unsloth for accelerated training.")
else:
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
    from peft import LoraConfig, get_peft_model
    print("[Python] No GPU detected. Falling back to standard Transformers CPU training.")

from datasets import load_dataset
from trl import SFTTrainer

def train(dataset_path, output_dir):
    model_name = "Qwen/Qwen3-0.6B-Instruct"
    max_seq_length = 2048

    if HAS_CUDA:
        # GPU Path: Fast Unsloth 4-bit
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = model_name,
            max_seq_length = max_seq_length,
            load_in_4bit = True,
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r = 16,
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_alpha = 16,
            lora_dropout = 0,
            bias = "none",
            use_gradient_checkpointing = "unsloth",
            random_state = 3407,
        )
    else:
        # CPU Path: Standard Transformers FP32
        print("[Python] Loading model in FP32 for CPU training...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map="cpu"
        )
        
        # Manually apply LoRA
        config = LoraConfig(
            r=16,
            lora_alpha=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, config)
        model.print_trainable_parameters()

    # 3. Load Dataset
    def format_prompts(examples):
        instructions = examples["instruction"]
        inputs       = examples["input"]
        outputs      = examples["output"]
        texts = []
        for instruction, input, output in zip(instructions, inputs, outputs):
            text = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{instruction}\nContext: {input}<|im_end|>\n<|im_start|>assistant\n{output}<|im_end|>"
            texts.append(text)
        return { "text" : texts, }

    dataset = load_dataset("json", data_files=dataset_path, split="train")
    dataset = dataset.map(format_prompts, batched=True)

    # 4. Training
    trainer = SFTTrainer(
        model = model,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        args = TrainingArguments(
            per_device_train_batch_size = 1, # Small batch for CPU
            gradient_accumulation_steps = 4,
            warmup_steps = 2,
            max_steps = 10, # Very few steps for CPU-POC
            learning_rate = 2e-4,
            fp16 = False, # CPU doesn't support FP16 well
            bf16 = False,
            logging_steps = 1,
            output_dir = "outputs",
            report_to = "none",
            use_cpu = not HAS_CUDA,
        ),
    )

    trainer.train()

    # 5. Export
    print(f"[Python] Training complete. Exporting...")
    
    if HAS_CUDA:
        # Unsloth has a built-in GGUF exporter
        model.save_pretrained_gguf(output_dir, tokenizer, quantization_method = "q8_0")
    else:
        # On CPU, we merge manually. 
        print("[Python] CPU Merge: Merging LoRA into base model (FP32)...")
        merged_model = model.merge_and_unload()
        
        # Save as standard Safetensors
        merged_model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        print(f"[Python] Internalized FP32 model saved to {output_dir}.")
        print("[Python] Note: To use this in the Rust engine, you must manually quantize it to GGUF using llama.cpp.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()
    
    try:
        train(args.dataset, args.output)
    except Exception as e:
        print(f"[Python Error] {e}")
        import traceback
        traceback.print_exc()
        exit(1)
