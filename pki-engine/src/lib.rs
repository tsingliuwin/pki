use anyhow::{Context, Result};
use candle_core::quantized::gguf_file;
use candle_core::{Device, Tensor};
use candle_transformers::models::quantized_qwen3::ModelWeights;
use std::io::Write;
use tokenizers::Tokenizer;

pub struct PkiModel {
    model: ModelWeights,
    tokenizer: Tokenizer,
}

impl PkiModel {
    pub fn new(model_path: &str, tokenizer_path: &str) -> Result<Self> {
        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

        let mut file = std::fs::File::open(model_path)?;
        let mut ct = gguf_file::Content::read(&mut file)
            .context("Failed to read GGUF file")?;

        let old_keys: Vec<String> = ct.metadata.keys().filter(|k| k.starts_with("qwen2.")).cloned().collect();
        for key in old_keys {
            if let Some(value) = ct.metadata.remove(&key) {
                ct.metadata.insert(key.replace("qwen2.", "qwen3."), value);
            }
        }

        let model = ModelWeights::from_gguf(ct, &mut file, &Device::Cpu)
            .context("Failed to initialize Qwen3 model from GGUF")?;

        Ok(Self { model, tokenizer })
    }

    pub fn generate(&mut self, prompt: &str, max_tokens: usize, silent: bool) -> Result<String> {
        let tokens = self.tokenizer.encode(prompt, true)
            .map_err(|e| anyhow::anyhow!("Tokenization error: {}", e))?;
        
        let prompt_tokens = tokens.get_ids().to_vec();
        let mut generated_text = String::new();
        let mut current_pos = 0;
        let mut current_input_tokens = prompt_tokens;

        for _ in 0..max_tokens {
            let input = Tensor::new(current_input_tokens.as_slice(), &Device::Cpu)?.unsqueeze(0)?;
            let logits = self.model.forward(&input, current_pos)?;
            
            let logits = logits.squeeze(0)?;
            let seq_len = logits.dim(0)?;
            let last_token_logits = logits.get(seq_len - 1)?; 
            
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

            if next_token == 151645 || next_token == 151643 { 
                break;
            }

            if let Some(decoded) = self.tokenizer.decode(&[next_token], false).ok() {
                if !silent {
                    print!("{}", decoded);
                    std::io::stdout().flush()?;
                }
                generated_text.push_str(&decoded);
            }

            current_pos += current_input_tokens.len();
            current_input_tokens = vec![next_token];
        }

        Ok(generated_text)
    }
}

pub fn chat(model_path: &str, tokenizer_path: &str, adapter_path: Option<&str>, query: &str) -> Result<()> {
    println!("[pki-engine] Loading Model...");
    let mut pki = PkiModel::new(model_path, tokenizer_path)?;

    if let Some(adapter) = adapter_path {
        println!("[pki-engine] WARNING: Skipping adapter {} (Dynamic loading not yet supported).", adapter);
    }

    let prompt = format!(
        "<|im_start|>system\nYou are a helpful local assistant.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
        query
    );

    print!("\n>>> 🤖 System Reply: ");
    std::io::stdout().flush()?;
    
    let response = pki.generate(&prompt, 256, false)?;
    println!("\n\n[pki-engine] Generation complete ({} chars).", response.len());
    Ok(())
}
