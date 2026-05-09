use anyhow::{Context, Result};
use std::path::Path;
use std::process::Command;

/// Checks if nvidia-smi is available to determine if a CUDA GPU is present.
fn has_cuda() -> bool {
    Command::new("nvidia-smi")
        .arg("--query-gpu=name")
        .arg("--format=csv,noheader")
        .output()
        .map(|out| out.status.success())
        .unwrap_or(false)
}

/// Ensures the Python environment is set up via `uv`.
pub fn ensure_env(pki_root: &Path) -> Result<()> {
    // Check if uv is installed
    let uv_check = Command::new("uv").arg("--version").output();
    if uv_check.is_err() || !uv_check.unwrap().status.success() {
        anyhow::bail!("`uv` is not installed or not in PATH. Please install uv (e.g. `pip install uv` or `curl -LsSf https://astral.sh/uv/install.sh | sh`).");
    }

    let venv_dir = pki_root.join(".venv");
    
    if !venv_dir.exists() {
        println!("[pki-trainer] Virtual environment not found. Creating via `uv`...");
        let status = Command::new("uv")
            .arg("venv")
            .current_dir(pki_root)
            .status()
            .context("Failed to run `uv venv`")?;
            
        if !status.success() {
            anyhow::bail!("Failed to create virtual environment.");
        }
        
        let has_gpu = has_cuda();
        if has_gpu {
            println!("[pki-trainer] NVIDIA GPU detected. Installing Unsloth with CUDA support...");
            // Real install would be something like:
            // uv pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" xformers trl peft accelerate jupyter
            // But for this POC/Skeleton, we print the intention. If we actually run this, it takes 5GB.
            println!("[pki-trainer] (Skipping actual 5GB download for development/demonstration)");
        } else {
            println!("[pki-trainer] WARNING: No NVIDIA GPU detected.");
            println!("[pki-trainer] Unsloth requires CUDA. Falling back to CPU PEFT (extremely slow).");
        }
    } else {
        println!("[pki-trainer] Virtual environment already exists at {:?}", venv_dir);
    }
    
    Ok(())
}

/// Helper to get the correct path to the python executable in the virtual environment.
pub fn get_python_exe(pki_root: &Path) -> std::path::PathBuf {
    if cfg!(windows) {
        pki_root.join(".venv").join("Scripts").join("python.exe")
    } else {
        pki_root.join(".venv").join("bin").join("python")
    }
}
