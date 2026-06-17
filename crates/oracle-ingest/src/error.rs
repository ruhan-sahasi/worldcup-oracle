//! Error type for the ingestion layer.

use thiserror::Error;

/// Anything that can go wrong while loading the tournament or streaming events.
#[derive(Debug, Error)]
pub enum IngestError {
    #[error("HTTP request failed: {0}")]
    Http(#[from] reqwest::Error),

    #[error("failed to parse response: {0}")]
    Parse(#[from] serde_json::Error),

    #[error("missing configuration: set the `{0}` environment variable")]
    MissingConfig(&'static str),

    #[error("data error: {0}")]
    Data(String),

    #[error("the event channel was closed by the consumer")]
    ChannelClosed,
}

pub type Result<T> = std::result::Result<T, IngestError>;
