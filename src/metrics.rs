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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_default() {
        let metrics = WorkflowMetrics::default();
        assert_eq!(metrics.prompt_token_count, 0);
        assert_eq!(metrics.completion_token_count, 0);
        assert_eq!(metrics.total_token_count, 0);
        assert_eq!(metrics.steps_completed, 0);
        assert!(metrics.failures.is_empty());
        assert!(!metrics.has_failures());
    }

    #[test]
    fn test_add_prompt_tokens() {
        let mut metrics = WorkflowMetrics::default();
        metrics.add_prompt_tokens(100);
        assert_eq!(metrics.prompt_token_count, 100);
        assert_eq!(metrics.total_token_count, 100);
        assert_eq!(metrics.total_tokens(), 100);
    }

    #[test]
    fn test_add_completion_tokens() {
        let mut metrics = WorkflowMetrics::default();
        metrics.add_completion_tokens(50);
        assert_eq!(metrics.completion_token_count, 50);
        assert_eq!(metrics.total_token_count, 50);
        assert_eq!(metrics.total_tokens(), 50);
    }

    #[test]
    fn test_add_tokens_combined() {
        let mut metrics = WorkflowMetrics::default();
        metrics.add_tokens(100, 50);
        assert_eq!(metrics.prompt_token_count, 100);
        assert_eq!(metrics.completion_token_count, 50);
        assert_eq!(metrics.total_token_count, 150);
        assert_eq!(metrics.total_tokens(), 150);
    }

    #[test]
    fn test_record_failure() {
        let mut metrics = WorkflowMetrics::default();
        assert!(!metrics.has_failures());
        
        metrics.record_failure("Error 1".to_string());
        assert!(metrics.has_failures());
        assert_eq!(metrics.failures.len(), 1);
        
        metrics.record_failure("Error 2".to_string());
        assert_eq!(metrics.failures.len(), 2);
        assert_eq!(metrics.failures[0], "Error 1");
        assert_eq!(metrics.failures[1], "Error 2");
    }

    #[test]
    fn test_record_step() {
        let mut metrics = WorkflowMetrics::default();
        assert_eq!(metrics.steps_completed, 0);
        
        metrics.record_step();
        assert_eq!(metrics.steps_completed, 1);
        
        metrics.record_step();
        metrics.record_step();
        assert_eq!(metrics.steps_completed, 3);
    }

    #[test]
    fn test_metrics_accumulation() {
        let mut metrics = WorkflowMetrics::default();
        
        // Simulate multiple LLM calls
        metrics.add_tokens(100, 50);
        metrics.record_step();
        
        metrics.add_tokens(200, 75);
        metrics.record_step();
        
        assert_eq!(metrics.prompt_token_count, 300);
        assert_eq!(metrics.completion_token_count, 125);
        assert_eq!(metrics.total_tokens(), 425);
        assert_eq!(metrics.steps_completed, 2);
    }

    #[test]
    fn test_metrics_serialization() {
        let mut metrics = WorkflowMetrics::default();
        metrics.add_tokens(100, 50);
        metrics.record_step();
        metrics.record_failure("test error".to_string());
        
        let json = serde_json::to_string(&metrics).unwrap();
        let deserialized: WorkflowMetrics = serde_json::from_str(&json).unwrap();
        
        assert_eq!(deserialized.prompt_token_count, 100);
        assert_eq!(deserialized.completion_token_count, 50);
        assert_eq!(deserialized.total_token_count, 150);
        assert_eq!(deserialized.steps_completed, 1);
        assert_eq!(deserialized.failures.len(), 1);
    }
}
