mod downloader;

use clap::{Parser, Subcommand};
use pki_engine::chat;
use pki_ingest::ingest_file;
use pki_trainer::train_lora;
use downloader::get_or_download_model;

#[derive(Parser)]
#[command(name = "pki")]
#[command(about = "Personal Knowledge Internalization System", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Ingest a document and generate QA dataset
    Ingest {
        /// Path to the input file (PDF/Markdown)
        #[arg(short, long)]
        file: String,
    },
    /// Train a LoRA adapter using a QA dataset
    Train {
        /// Path to the input QA dataset (.jsonl)
        #[arg(short, long)]
        dataset: String,
    },
    /// Chat with the model
    Chat {
        /// The model to use (default: Qwen3-0.6B-GGUF)
        #[arg(short, long, default_value = "Qwen3-0.6B-GGUF")]
        model: String,
        /// Path to the LoRA adapter (.gguf)
        #[arg(short, long)]
        adapter: Option<String>,
        /// The query to ask the model
        #[arg(short, long)]
        query: String,
    },
    /// Download a model explicitly
    PullModel {
        /// The model name
        #[arg(short, long, default_value = "Qwen3-0.6B-GGUF")]
        model: String,
    },
    /// Run the full mock pipeline end-to-end
    Pipeline {
        /// Path to the input file
        #[arg(short, long)]
        file: String,
        /// The query to ask at the end
        #[arg(short, long)]
        query: String,
    },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match &cli.command {
        Commands::Ingest { file } => {
            if let Err(e) = ingest_file(file) {
                eprintln!("Ingest failed: {}", e);
            }
        }
        Commands::Train { dataset } => {
            if let Err(e) = train_lora(dataset) {
                eprintln!("Train failed: {}", e);
            }
        }
        Commands::PullModel { model } => {
            get_or_download_model(model).await?;
        }
        Commands::Chat { model, adapter, query } => {
            let (model_path, tokenizer_path) = get_or_download_model(model).await?;
            if let Err(e) = chat(&model_path.to_string_lossy(), &tokenizer_path.to_string_lossy(), adapter.as_deref(), query) {
                eprintln!("Chat failed: {}", e);
            }
        }
        Commands::Pipeline { file, query } => {
            println!("=== Starting E2E Mock Pipeline ===");
            
            println!("\n--- Step 1: Ingest ---");
            let dataset_path = match ingest_file(file) {
                Ok(path) => path,
                Err(e) => {
                    eprintln!("Pipeline aborted at ingest: {}", e);
                    return Ok(());
                }
            };

            println!("\n--- Step 2: Train ---");
            let adapter_path = match train_lora(&dataset_path) {
                Ok(path) => path,
                Err(e) => {
                    eprintln!("Pipeline aborted at train: {}", e);
                    return Ok(());
                }
            };

            println!("\n--- Step 3: Chat ---");
            let model = "Qwen3-0.6B-GGUF";
            let (model_path, tokenizer_path) = get_or_download_model(model).await?;
            if let Err(e) = chat(&model_path.to_string_lossy(), &tokenizer_path.to_string_lossy(), Some(&adapter_path), query) {
                eprintln!("Pipeline aborted at chat: {}", e);
            }
            
            println!("\n=== Pipeline Complete ===");
        }
    }

    Ok(())
}
