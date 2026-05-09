use text_splitter::TextSplitter;

/// Splits raw text into semantic chunks
/// We use a chunk size of 1000 characters as requested.
pub fn split_into_chunks(text: &str) -> Vec<String> {
    // text-splitter works by finding semantic boundaries (paragraphs, sentences, words)
    // and splitting up to the capacity.
    let splitter = TextSplitter::new(1000);
    
    // Split and collect into a Vec of owned Strings
    splitter
        .chunks(text)
        .map(|chunk| chunk.to_string())
        .collect()
}
