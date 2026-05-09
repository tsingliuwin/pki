use anyhow::{Context, Result};
use std::path::{Path, PathBuf};
use std::process::Command;

pub fn ensure_env(project_root: &Path) -> Result<()> {
    let venv_path = project_root.join(".venv");
    let python_exe = get_python_exe(project_root);
    
    // Check if venv exists AND python exe is actually there
    if !venv_path.exists() || !python_exe.exists() {
        println!("[pki-trainer] Virtual environment not found or broken. Creating with `uv`...");
        
        // Remove broken venv if it exists
        if venv_path.exists() {
            let _ = std::fs::remove_dir_all(&venv_path);
        }

        let status = Command::new("uv")
            .arg("venv")
            .current_dir(project_root)
            .status()
            .context("Failed to run `uv`. Please install it: `pip install uv` or `brew install uv`")?;
        
        if !status.success() {
            anyhow::bail!("`uv venv` failed.");
        }
    }

    // Force an installation if a marker file doesn't exist to ensure requirements are met
    let marker_file = venv_path.join("pki_env_ready.marker");
    if !marker_file.exists() {
        println!("[pki-trainer] Installing Python requirements via `uv`...");
        let req_path = project_root.join("pki-trainer").join("requirements.txt");
        
        // Use `uv pip install` which is much more robust
        let status = Command::new("uv")
            .arg("pip")
            .arg("install")
            .arg("-r")
            .arg(req_path)
            .env("VIRTUAL_ENV", &venv_path) // Explicitly point to our venv
            .stdout(std::process::Stdio::inherit())
            .stderr(std::process::Stdio::inherit())
            .status()
            .context("Failed to run `uv pip install`. Please ensure `uv` is in your PATH.")?;

        if !status.success() {
            anyhow::bail!("Requirement installation failed. You might need to install `torch` and `unsloth` manually in the `.venv`.");
        }

        // Create marker
        std::fs::File::create(marker_file)?;
    }
    
    Ok(())
}

pub fn get_python_exe(project_root: &Path) -> PathBuf {
    if cfg!(target_os = "windows") {
        project_root.join(".venv").join("Scripts").join("python.exe")
    } else {
        project_root.join(".venv").join("bin").join("python")
    }
}

pub fn get_pip_exe(project_root: &Path) -> PathBuf {
    if cfg!(target_os = "windows") {
        project_root.join(".venv").join("Scripts").join("pip.exe")
    } else {
        project_root.join(".venv").join("bin").join("pip")
    }
}
