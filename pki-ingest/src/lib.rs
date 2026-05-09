pub mod chunker;
pub mod parser;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::fs::OpenOptions;
use std::io::Write;

#[derive(Serialize, Deserialize, Debug)]
pub struct AlpacaQa {
    pub instruction: String,
    pub input: String,
    pub output: String,
}

pub fn ingest_file(path: &str) -> Result<String> {
    println!("[pki-ingest] Parsing file at `{}`...", path);
    
    // 1. Parse document
    let raw_text = parser::extract_text(path).context("Failed to parse document")?;
    println!("[pki-ingest] Extracted {} characters.", raw_text.len());

    // 2. Chunk text
    println!("[pki-ingest] Splitting into semantic chunks...");
    let chunks = chunker::split_into_chunks(&raw_text);
    println!("[pki-ingest] Created {} chunks (max 1000 chars each).", chunks.len());

    // 3. Generate QA pairs (Mocking the LLM generator for now)
    println!("[pki-ingest] Generating QA pairs with local model (Mocking)...");
    
    let output_path = "mock_qa_dataset.jsonl";
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(output_path)
        .context("Failed to open output jsonl file")?;

    let mut qa_count = 0;
    for (i, chunk) in chunks.iter().enumerate() {
        // We generate 2 mock questions per chunk
        for j in 0..2 {
            let qa = AlpacaQa {
                instruction: format!("根据下文回答问题 (Mock QA #{}-{})", i + 1, j + 1),
                input: chunk.trim().to_string(),
                output: "这是基于上下文的模拟回答。在未来的版本中，这将由 Qwen 模型真实生成。".to_string(),
            };
            
            let json_line = serde_json::to_string(&qa).context("Failed to serialize JSON")?;
            writeln!(file, "{}", json_line).context("Failed to write to file")?;
            qa_count += 1;
        }
    }

    println!("[pki-ingest] Successfully generated {} QA pairs and saved to `{}`", qa_count, output_path);
    
    Ok(output_path.to_string())
}
