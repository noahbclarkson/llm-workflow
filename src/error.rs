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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display_validation() {
        let err = Error::Validation("field is required".to_string());
        assert_eq!(err.to_string(), "Validation error: field is required");
    }

    #[test]
    fn test_error_display_execution() {
        let err = Error::Execution("step timed out".to_string());
        assert_eq!(err.to_string(), "Execution error: step timed out");
    }

    #[test]
    fn test_error_display_message() {
        let err = Error::Message("something went wrong".to_string());
        assert_eq!(err.to_string(), "something went wrong");
    }

    #[test]
    fn test_error_display_checkpoint() {
        let err = Error::Checkpoint {
            step_name: "review".to_string(),
            data: serde_json::json!({"value": 42}),
        };
        assert_eq!(err.to_string(), "Checkpoint reached at step 'review'");
    }

    #[test]
    fn test_from_string() {
        let err: Error = "from string".to_string().into();
        assert!(matches!(err, Error::Message(_)));
        assert_eq!(err.to_string(), "from string");
    }

    #[test]
    fn test_from_str() {
        let err: Error = "from &str".into();
        assert!(matches!(err, Error::Message(_)));
        assert_eq!(err.to_string(), "from &str");
    }

    #[test]
    fn test_from_serde_json_error() {
        let json_err = serde_json::from_str::<serde_json::Value>("{invalid}").unwrap_err();
        let err: Error = json_err.into();
        assert!(matches!(err, Error::Json(_)));
    }

    #[test]
    fn test_error_debug() {
        let err = Error::Validation("test".to_string());
        let debug = format!("{:?}", err);
        assert!(debug.contains("Validation"));
    }
}
