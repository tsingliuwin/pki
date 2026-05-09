pub mod env;

use anyhow::{Context, Result};
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::thread;

pub fn train_lora(dataset_path: &str) -> Result<String> {
    let current_dir = std::env::current_dir().context("Failed to get current directory")?;
    
    // 1. Ensure the Python environment is set up
    println!("[pki-trainer] Checking environment...");
    env::ensure_env(&current_dir)?;

    let python_exe = env::get_python_exe(&current_dir);
    let script_path = current_dir.join("pki-trainer").join("python_scripts").join("train.py");
    let output_dir = current_dir.join("lora_adapter_out");

    println!("[pki-trainer] Spawning Python training process...");
    
    // 2. Spawn the Python process, piping stdout and stderr
    let mut child = Command::new(&python_exe)
        .arg(&script_path)
        .arg("qwen/Qwen2.5-0.5B-Instruct") // Base model hardcoded for now
        .arg(dataset_path)
        .arg(&output_dir)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .context("Failed to spawn Python training script")?;

    let stdout = child.stdout.take().expect("Failed to capture stdout");
    let stderr = child.stderr.take().expect("Failed to capture stderr");

    // 3. Stream stdout and stderr to the console in real-time
    let stdout_handle = thread::spawn(move || {
        let reader = BufReader::new(stdout);
        for line in reader.lines() {
            if let Ok(line) = line {
                println!("[Unsloth] {}", line);
            }
        }
    });

    let stderr_handle = thread::spawn(move || {
        let reader = BufReader::new(stderr);
        for line in reader.lines() {
            if let Ok(line) = line {
                eprintln!("[Unsloth ERR] {}", line);
            }
        }
    });

    let status = child.wait().context("Failed to wait on Python process")?;
    
    stdout_handle.join().expect("Failed to join stdout thread");
    stderr_handle.join().expect("Failed to join stderr thread");

    if !status.success() {
        anyhow::bail!("Python training script failed with status: {}", status);
    }

    let gguf_path = output_dir.join("lora-adapter.gguf");
    if !gguf_path.exists() {
        anyhow::bail!("Training completed but GGUF adapter was not found at {:?}", gguf_path);
    }

    println!("[pki-trainer] Training pipeline finished successfully.");
    Ok(gguf_path.to_string_lossy().to_string())
}
