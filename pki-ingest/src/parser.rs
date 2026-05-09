use anyhow::{Context, Result};
use pulldown_cmark::{Event, Parser};
use std::fs;
use std::path::Path;

pub trait DocumentParser {
    fn parse(&self, path: &Path) -> Result<String>;
}

pub struct PdfParser;
impl DocumentParser for PdfParser {
    fn parse(&self, path: &Path) -> Result<String> {
        let bytes = fs::read(path).context("Failed to read PDF file")?;
        let text = pdf_extract::extract_text_from_mem(&bytes)
            .context("Failed to extract text from PDF")?;
        Ok(text)
    }
}

pub struct MarkdownParser;
impl DocumentParser for MarkdownParser {
    fn parse(&self, path: &Path) -> Result<String> {
        let markdown_input = fs::read_to_string(path).context("Failed to read Markdown file")?;
        let parser = Parser::new(&markdown_input);
        
        let mut pure_text = String::new();
        for event in parser {
            if let Event::Text(text) = event {
                pure_text.push_str(&text);
                pure_text.push(' ');
            } else if let Event::Code(code) = event {
                pure_text.push_str(&code);
                pure_text.push(' ');
            } else if let Event::SoftBreak | Event::HardBreak = event {
                pure_text.push('\n');
            }
        }
        Ok(pure_text)
    }
}

pub struct PlainTextParser;
impl DocumentParser for PlainTextParser {
    fn parse(&self, path: &Path) -> Result<String> {
        fs::read_to_string(path).context("Failed to read plain text file")
    }
}

/// Automatically selects the parser based on the file extension
pub fn extract_text(path_str: &str) -> Result<String> {
    let path = Path::new(path_str);
    let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("").to_lowercase();

    match ext.as_str() {
        "pdf" => PdfParser.parse(path),
        "md" | "markdown" => MarkdownParser.parse(path),
        _ => PlainTextParser.parse(path), // Fallback to plain text
    }
}
