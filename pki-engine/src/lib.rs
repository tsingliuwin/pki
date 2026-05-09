use anyhow::{Context, Result};
use candle_core::quantized::gguf_file;
use candle_core::{Device, Tensor};
use candle_transformers::models::quantized_qwen2::ModelWeights;
use std::io::Write;
use tokenizers::Tokenizer;

pub fn chat(model_path: &str, tokenizer_path: &str, adapter_path: Option<&str>, query: &str) -> Result<()> {
    println!("[pki-engine] Loading Tokenizer from {}...", tokenizer_path);
    let tokenizer = Tokenizer::from_file(tokenizer_path)
        .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

    println!("[pki-engine] Loading GGUF Model from {}...", model_path);
    let mut file = std::fs::File::open(model_path)?;
    let model_reader = gguf_file::Content::read(&mut file)
        .context("Failed to read GGUF file")?;
    
    let mut model = ModelWeights::from_gguf(model_reader, &mut file, &Device::Cpu)
        .context("Failed to initialize Qwen2 model from GGUF")?;

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
    
    let mut prompt_tokens = tokens.get_ids().to_vec();

    println!("[pki-engine] Evaluating prompt ({} tokens)...", prompt_tokens.len());
    
    // Very simple generation loop (Greedy)
    print!("\n>>> 🤖 System Reply: ");
    std::io::stdout().flush()?;

    let mut generated_text = String::new();
    
    // For demonstration, we just do a tiny loop. In reality, you'd feed tokens one by one and sample.
    for index in 0..100 {
        let input = Tensor::new(prompt_tokens.as_slice(), &Device::Cpu)?.unsqueeze(0)?;
        let logits = model.forward(&input, index)?;
        
        let logits = logits.squeeze(0)?;
        // Get the argmax token (Greedy sampling)
        // Extract the last token's logits
        let logits_v: Vec<f32> = logits.to_vec1()?;
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

        prompt_tokens.push(next_token);
        
        if let Some(decoded) = tokenizer.decode(&[next_token], false).ok() {
            print!("{}", decoded);
            std::io::stdout().flush()?;
            generated_text.push_str(&decoded);
        }
    }

    println!("\n");
    Ok(())
}
