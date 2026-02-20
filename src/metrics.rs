//! Metrics collection for workflow execution.
//!
//! This module provides `WorkflowMetrics` for tracking token usage,
//! execution statistics, and failures.

use serde::{Serialize, Deserialize};

/// Aggregated metrics for a workflow execution.
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct WorkflowMetrics {
    /// Total prompt tokens consumed across all steps.
    pub prompt_token_count: usize,
    /// Total completion tokens generated across all steps.
    pub completion_token_count: usize,
    /// Total tokens (prompt + completion) across all steps.
    pub total_token_count: usize,
    /// Number of workflow steps completed successfully.
    pub steps_completed: usize,
    /// Collected failure messages from the workflow.
    pub failures: Vec<String>,
}

impl WorkflowMetrics {
    /// Record prompt token usage.
    pub fn add_prompt_tokens(&mut self, count: usize) {
        self.prompt_token_count += count;
        self.total_token_count += count;
    }

    /// Record completion token usage.
    pub fn add_completion_tokens(&mut self, count: usize) {
        self.completion_token_count += count;
        self.total_token_count += count;
    }

    /// Record both prompt and completion tokens.
    pub fn add_tokens(&mut self, prompt: usize, completion: usize) {
        self.prompt_token_count += prompt;
        self.completion_token_count += completion;
        self.total_token_count += prompt + completion;
    }

    /// Record a failure message.
    pub fn record_failure(&mut self, error: String) {
        self.failures.push(error);
    }

    /// Increment the steps completed counter.
    pub fn record_step(&mut self) {
        self.steps_completed += 1;
    }

    /// Check if there were any failures.
    pub fn has_failures(&self) -> bool {
        !self.failures.is_empty()
    }

    /// Get the total number of tokens used.
    pub fn total_tokens(&self) -> usize {
        self.total_token_count
    }
}
