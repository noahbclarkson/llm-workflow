//! Error types for workflow execution.

use thiserror::Error;

/// The main error type for workflow operations.
#[derive(Error, Debug)]
pub enum Error {
    /// A checkpoint was reached, pausing execution.
    #[error("Checkpoint reached at step '{step_name}'")]
    Checkpoint {
        /// The name of the checkpoint step.
        step_name: String,
        /// The data at the checkpoint as a JSON value.
        data: serde_json::Value,
    },

    /// A validation error occurred.
    #[error("Validation error: {0}")]
    Validation(String),

    /// An error occurred during execution.
    #[error("Execution error: {0}")]
    Execution(String),

    /// A JSON serialization/deserialization error.
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// A generic error with a message.
    #[error("{0}")]
    Message(String),
}

impl From<String> for Error {
    fn from(msg: String) -> Self {
        Error::Message(msg)
    }
}

impl From<&str> for Error {
    fn from(msg: &str) -> Self {
        Error::Message(msg.to_string())
    }
}

/// A specialized `Result` type for workflow operations.
pub type Result<T> = std::result::Result<T, Error>;
