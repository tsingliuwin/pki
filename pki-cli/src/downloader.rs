use anyhow::{Context, Result};
use futures_util::StreamExt;
use indicatif::{ProgressBar, ProgressStyle};
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};

/// Resolves the model name to a ModelScope download URL and local filename.
/// We use Qwen2.5-0.5B-Instruct as the practical placeholder for "Qwen3-0.6B-Instruct".
fn resolve_model_info(model_name: &str) -> (String, String) {
    if model_name.contains("Qwen3-0.6B") || model_name.contains("Qwen2.5-0.5B") {
        let repo = "qwen/Qwen2.5-0.5B-Instruct-GGUF";
        let file = "qwen2.5-0.5b-instruct-q4_k_m.gguf";
        let url = format!(
            "https://modelscope.cn/api/v1/models/{}/repo?Revision=master&FilePath={}",
            repo, file
        );
        (url, file.to_string())
    } else {
        // Fallback for custom HF/ModelScope links or local paths
        (String::new(), model_name.to_string())
    }
}

pub async fn get_or_download_model(model_name: &str) -> Result<PathBuf> {
    let pki_dir = dirs::home_dir()
        .context("Could not find home directory")?
        .join(".pki")
        .join("models");

    std::fs::create_dir_all(&pki_dir).context("Failed to create models directory")?;

    let (url, filename) = resolve_model_info(model_name);
    let local_path = pki_dir.join(&filename);

    if local_path.exists() {
        println!("[Downloader] Model {} already exists at {:?}", filename, local_path);
        return Ok(local_path);
    }

    if url.is_empty() {
        anyhow::bail!("Unsupported model name for automatic download. Please provide a local path.");
    }

    println!("[Downloader] Downloading model from ModelScope...");
    println!("[Downloader] Target: {}", url);

    let client = reqwest::Client::new();
    let res = client.get(&url).send().await.context("Failed to send request")?;

    if !res.status().is_success() {
        anyhow::bail!("Failed to download model: HTTP {}", res.status());
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
    println!("[Downloader] Model saved to {:?}", local_path);

    Ok(local_path)
}
