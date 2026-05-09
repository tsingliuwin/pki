use anyhow::{Context, Result};
use candle_core::quantized::gguf_file;
use candle_core::{Device, Tensor};
use candle_transformers::models::quantized_qwen3::ModelWeights;
use std::io::Write;
use tokenizers::Tokenizer;

pub fn chat(model_path: &str, tokenizer_path: &str, adapter_path: Option<&str>, query: &str) -> Result<()> {
    println!("[pki-engine] Loading Tokenizer from {}...", tokenizer_path);
    let tokenizer = Tokenizer::from_file(tokenizer_path)
        .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

    println!("[pki-engine] Loading GGUF Model from {}...", model_path);
    let mut file = std::fs::File::open(model_path)?;
    let mut ct = gguf_file::Content::read(&mut file)
        .context("Failed to read GGUF file")?;

    // Older llama.cpp quantizes Qwen3 models under the "qwen2" arch label.
    // Remap qwen2.* metadata keys to qwen3.* so Candle's quantized_qwen3 can find them.
    let old_keys: Vec<String> = ct.metadata.keys().filter(|k| k.starts_with("qwen2.")).cloned().collect();
    for key in old_keys {
        if let Some(value) = ct.metadata.remove(&key) {
            ct.metadata.insert(key.replace("qwen2.", "qwen3."), value);
        }
    }

    let mut model = ModelWeights::from_gguf(ct, &mut file, &Device::Cpu)
        .context("Failed to initialize Qwen3 model from GGUF")?;

    if let Some(adapter) = adapter_path {
        println!("[pki-engine] WARNING: Dynamic LoRA loading from unmerged GGUF is not fully supported in pure Rust yet.");
        println!("[pki-engine] WARNING: In a production environment, you should merge the LoRA using Python before inference.");
        println!("[pki-engine] WARNING: Skipping adapter {} and using Base Model.", adapter);
    }

    // Format prompt using Qwen ChatML
    let prompt = format!(
        "<|im_start|>system\nYou are a helpful local assistant.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
        query
    );

    let tokens = tokenizer.encode(prompt, true)
        .map_err(|e| anyhow::anyhow!("Tokenization error: {}", e))?;
    
    let prompt_tokens = tokens.get_ids().to_vec();

    println!("[pki-engine] Evaluating prompt ({} tokens)...", prompt_tokens.len());
    
    // Very simple generation loop (Greedy)
    print!("\n>>> 🤖 System Reply: ");
    std::io::stdout().flush()?;

    let mut generated_text = String::new();
    let mut current_pos = 0;
    let mut current_input_tokens = prompt_tokens;

    for _ in 0..256 {
        let input = Tensor::new(current_input_tokens.as_slice(), &Device::Cpu)?.unsqueeze(0)?;
        let logits = model.forward(&input, current_pos)?;
        
        // Logits is [batch_size, seq_len, vocab_size]. We want the last token's logits.
        let logits = logits.squeeze(0)?; // [seq_len, vocab_size]
        let seq_len = logits.dim(0)?;
        
        // Extract the last row [vocab_size]
        let last_token_logits = logits.get(seq_len - 1)?; 
        
        // Ensure it's a 1D tensor for to_vec1. If it's 0D, it means something is wrong with dimensions.
        let logits_v: Vec<f32> = if last_token_logits.rank() == 0 {
            logits.to_vec1()?
        } else {
            last_token_logits.to_vec1()?
        };

        let mut next_token = 0;
        let mut max_prob = f32::NEG_INFINITY;
        for (i, &p) in logits_v.iter().enumerate() {
            if p > max_prob {
                max_prob = p;
                next_token = i as u32;
            }
        }

        if next_token == 151645 || next_token == 151643 { // Qwen im_end / eos
            break;
        }

        if let Some(decoded) = tokenizer.decode(&[next_token], false).ok() {
            print!("{}", decoded);
            std::io::stdout().flush()?;
            generated_text.push_str(&decoded);
        }

        current_pos += current_input_tokens.len();
        current_input_tokens = vec![next_token];
    }

    println!("\n\n[pki-engine] Generation complete ({} chars).", generated_text.len());
    Ok(())
}
