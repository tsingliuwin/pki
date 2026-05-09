pub mod chunker;
pub mod parser;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::Path;

#[derive(Serialize, Deserialize, Debug)]
pub struct AlpacaQa {
    pub instruction: String,
    pub input: String,
    pub output: String,
}

pub fn ingest_file<F>(path: &str, mut generator: Option<F>) -> Result<String> 
where 
    F: FnMut(&str) -> Result<String>
{
    println!("[pki-ingest] Parsing file at `{}`...", path);
    
    // 1. Parse document
    let raw_text = parser::extract_text(path).context("Failed to parse document")?;
    println!("[pki-ingest] Extracted {} characters.", raw_text.len());

    // 2. Chunk text
    println!("[pki-ingest] Splitting into semantic chunks...");
    let chunks = chunker::split_into_chunks(&raw_text);
    println!("[pki-ingest] Created {} chunks (max 1000 chars each).", chunks.len());

    // 3. Generate QA pairs
    let data_dir = Path::new("data");
    if !data_dir.exists() {
        fs::create_dir_all(data_dir).context("Failed to create data directory")?;
    }
    let output_path = data_dir.join("pki_qa_dataset.jsonl");

    let mut file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(&output_path)
        .context("Failed to open output jsonl file")?;

    let mut qa_count = 0;
    for (i, chunk) in chunks.iter().enumerate() {
        println!("[pki-ingest] Processing chunk {}/{}...", i + 1, chunks.len());
        
        let (instruction, output) = match generator.as_mut() {
            Some(generator_func) => {
                // Real generation
                let prompt = format!(
                    "请根据以下文本内容，生成一个简短的问题和对应的详细回答。输出格式：\n问题：[问题内容]\n回答：[回答内容]\n\n文本内容：\n{}", 
                    chunk
                );
                let response = generator_func(&prompt)?;
                
                // Simple parsing of "问题：" and "回答："
                let parts: Vec<&str> = response.split("回答：").collect();
                if parts.len() == 2 {
                    let q = parts[0].replace("问题：", "").trim().to_string();
                    let a = parts[1].trim().to_string();
                    (q, a)
                } else {
                    // Fallback if model doesn't follow format exactly
                    ("根据文本总结核心知识点".to_string(), response)
                }
            },
            None => {
                // Mock fallback
                (
                    format!("根据下文回答问题 (Mock QA #{})", i + 1),
                    "这是模拟回答。".to_string()
                )
            }
        };

        let qa = AlpacaQa {
            instruction,
            input: chunk.trim().to_string(),
            output,
        };
        
        let json_line = serde_json::to_string(&qa).context("Failed to serialize JSON")?;
        writeln!(file, "{}", json_line).context("Failed to write to file")?;
        qa_count += 1;
    }

    println!("[pki-ingest] Successfully generated {} QA pairs and saved to `{}`", qa_count, output_path.display());
    
    Ok(output_path.to_string_lossy().to_string())
}
