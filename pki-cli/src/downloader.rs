use anyhow::{Context, Result};
use futures_util::StreamExt;
use indicatif::{ProgressBar, ProgressStyle};
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};

/// Download a file from a URL to a local path with a progress bar.
async fn download_file(url: &str, local_path: &Path, filename: &str) -> Result<()> {
    if local_path.exists() {
        println!("[Downloader] {} already exists at {:?}", filename, local_path);
        return Ok(());
    }

    println!("[Downloader] Downloading {} from ModelScope...", filename);
    
    let client = reqwest::Client::new();
    let res = client.get(url).send().await.context("Failed to send request")?;

    if !res.status().is_success() {
        anyhow::bail!("Failed to download {}: HTTP {}", filename, res.status());
    }

    let total_size = res
        .content_length()
        .context("Failed to get content length from response")?;

    let pb = ProgressBar::new(total_size);
    pb.set_style(ProgressStyle::default_bar()
        .template("{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {bytes}/{total_bytes} ({bytes_per_sec}, {eta})")?
        .progress_chars("#>-"));

    let mut file = File::create(&local_path).context("Failed to create local file")?;
    let mut downloaded: u64 = 0;
    let mut stream = res.bytes_stream();

    while let Some(item) = stream.next().await {
        let chunk = item.context("Error while downloading file")?;
        file.write_all(&chunk).context("Error while writing to file")?;
        let new = std::cmp::min(downloaded + (chunk.len() as u64), total_size);
        downloaded = new;
        pb.set_position(new);
    }

    pb.finish_with_message("Download complete");
    println!("[Downloader] {} saved to {:?}", filename, local_path);
    Ok(())
}

/// Returns (GGUF Path, Tokenizer Path)
pub async fn get_or_download_model(_model_name: &str) -> Result<(PathBuf, PathBuf)> {
    let pki_dir = dirs::home_dir()
        .context("Could not find home directory")?
        .join(".pki")
        .join("models");

    std::fs::create_dir_all(&pki_dir).context("Failed to create models directory")?;

    // Use the requested model repo
    let repo = "Qwen/Qwen3-0.6B-GGUF";
    let base_repo = repo.replace("-GGUF", ""); // Base repo usually contains tokenizer.json
    let gguf_filename = "Qwen3-0.6B-Q8_0.gguf"; // Requested model filename
    let tokenizer_filename = "tokenizer.json";
    
    // We fetch from HuggingFace as it rarely gives 403 for public repos compared to ModelScope API
    let gguf_url = format!("https://huggingface.co/{}/resolve/main/{}", repo, gguf_filename);
    let tokenizer_url = format!("https://huggingface.co/{}/resolve/main/{}", base_repo, tokenizer_filename);

    let gguf_path = pki_dir.join(gguf_filename);
    let tokenizer_path = pki_dir.join(tokenizer_filename);

    // If the files already exist, it will skip downloading.
    // If you don't want to download, you can place the actual files in `~/.pki/models/` manually.
    download_file(&gguf_url, &gguf_path, gguf_filename).await?;
    download_file(&tokenizer_url, &tokenizer_path, tokenizer_filename).await?;

    Ok((gguf_path, tokenizer_path))
}
