pub mod env;

use anyhow::{Context, Result};
use std::process::{Command, Stdio};
use std::io::{BufRead, BufReader};
use std::thread;

pub fn train_lora(dataset_path: &str) -> Result<String> {
    let current_dir = std::env::current_dir().context("Failed to get current directory")?;
    
    println!("[pki-trainer] 🛠️  Phase 1: Environment Orchestration");
    env::ensure_env(&current_dir)?;

    let python_exe = env::get_python_exe(&current_dir);
    let script_path = current_dir.join("pki-trainer").join("python_scripts").join("train.py");
    let output_dir = current_dir.join("output").join("lora_adapter_out");
    if !output_dir.exists() {
        std::fs::create_dir_all(&output_dir).context("Failed to create output directory")?;
    }

    println!("[pki-trainer] 🚀 Phase 2: Internalization Training (Starts from BF16 Safetensors)");
    println!("[pki-trainer] Note: This will download ~1.1GB of raw weights if not cached.");
    
    let mut child = Command::new(&python_exe)
        .arg(&script_path)
        .arg("--dataset").arg(dataset_path)
        .arg("--output").arg(&output_dir)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .context("Failed to spawn Python training script")?;

    let stdout = child.stdout.take().expect("Failed to capture stdout");
    let stderr = child.stderr.take().expect("Failed to capture stderr");

    let stdout_handle = thread::spawn(move || {
        let reader = BufReader::new(stdout);
        for line in reader.lines() {
            if let Ok(line) = line {
                println!("[Internalizer] {}", line);
            }
        }
    });

    let stderr_handle = thread::spawn(move || {
        let reader = BufReader::new(stderr);
        for line in reader.lines() {
            if let Ok(line) = line {
                eprintln!("[Internalizer ERR] {}", line);
            }
        }
    });

    let status = child.wait().context("Failed to wait on training process")?;
    
    let _ = stdout_handle.join();
    let _ = stderr_handle.join();

    if !status.success() {
        println!("[pki-trainer] ⚠️  Training process failed or was interrupted.");
        println!("[pki-trainer] Possible reasons: Missing GPU, CUDA, or insufficient RAM (5GB+ required).");
        println!("[pki-trainer] The Pipeline will continue using the BASE model for inference.");
        
        // Return the path to the base model as a fallback, so the pipeline doesn't crash
        let base_gguf = dirs::home_dir().unwrap().join(".pki").join("models").join("Qwen3-0.6B-Q8_0.gguf");
        return Ok(base_gguf.to_string_lossy().to_string());
    }

    // Unsloth's save_pretrained_gguf usually saves the file as 'unsloth.Q8_0.gguf'
    let gguf_path = output_dir.join("unsloth.Q8_0.gguf");
    if !gguf_path.exists() {
        anyhow::bail!("Internalization complete but GGUF model was not found at {:?}", gguf_path);
    }

    println!("[pki-trainer] ✅ Phase 3: Knowledge Internalized & Quantized.");
    Ok(gguf_path.to_string_lossy().to_string())
}
