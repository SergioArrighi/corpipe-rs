use crate::UdpipeDocument;
use anyhow::{Context, Result};
use std::path::Path;
use udpipe_rs::Model;

/// Loaded UDPipe parser whose output can be reused by CorPipe or other consumers.
pub struct UdpipeParser {
    model: Model,
}

impl UdpipeParser {
    pub fn load(model_path: &Path) -> Result<Self> {
        let model = Model::load(model_path)
            .with_context(|| format!("failed to load UDPipe model {}", model_path.display()))?;
        Ok(Self { model })
    }

    pub fn parse(&self, text: &str) -> Result<UdpipeDocument> {
        let words = self
            .model
            .parse(text)
            .with_context(|| "failed to parse text with UDPipe")?;
        Ok(UdpipeDocument::from_words(text, &words))
    }
}

impl std::fmt::Debug for UdpipeParser {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("UdpipeParser")
            .finish_non_exhaustive()
    }
}
