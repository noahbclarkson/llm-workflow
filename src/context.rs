//! Execution context for workflow runs.
//!
//! This module provides the `ExecutionContext` which is passed to every step
//! in a workflow, enabling metrics collection and event tracing.

use std::sync::{Arc, Mutex};

use crate::metrics::WorkflowMetrics;
use crate::events::{TraceEntry, WorkflowEvent};

/// Context passed to every step in the workflow.
///
/// This context is cloneable and thread-safe, allowing it to be shared
/// across parallel step executions. All metric updates are synchronized.
///
/// # Tracing
///
/// The context also maintains a structured trace log of workflow events,
/// enabling detailed observability without relying on unstructured string logs.
///
/// # Example
///
/// ```rust
/// use llm_workflow::{ExecutionContext, WorkflowEvent};
///
/// let ctx = ExecutionContext::new();
/// ctx.emit(WorkflowEvent::StepStart {
///     step_name: "Summarize".to_string(),
///     input_type: "String".to_string(),
/// });
///
/// // Later, get all trace entries
/// let traces = ctx.trace_snapshot();
/// for entry in traces {
///     println!("{:?}", entry);
/// }
/// ```
#[derive(Debug, Clone)]
pub struct ExecutionContext {
    /// Shared metrics accumulator.
    pub metrics: Arc<Mutex<WorkflowMetrics>>,
    /// Shared trace log for structured workflow events.
    pub traces: Arc<Mutex<Vec<TraceEntry>>>,
}

impl Default for ExecutionContext {
    fn default() -> Self {
        Self::new()
    }
}

impl ExecutionContext {
    /// Create a new execution context with empty metrics and traces.
    #[must_use]
    pub fn new() -> Self {
        Self {
            metrics: Arc::new(Mutex::new(WorkflowMetrics::default())),
            traces: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Record prompt token usage.
    pub fn record_prompt_tokens(&self, count: usize) {
        let mut m = self.metrics.lock().unwrap();
        m.prompt_token_count += count;
    }

    /// Record completion token usage.
    pub fn record_completion_tokens(&self, count: usize) {
        let mut m = self.metrics.lock().unwrap();
        m.completion_token_count += count;
    }

    /// Record total token usage (convenience method).
    pub fn record_tokens(&self, prompt: usize, completion: usize) {
        let mut m = self.metrics.lock().unwrap();
        m.prompt_token_count += prompt;
        m.completion_token_count += completion;
        m.total_token_count += prompt + completion;
    }

    /// Increment the steps completed counter.
    pub fn record_step(&self) {
        let mut m = self.metrics.lock().unwrap();
        m.record_step();
    }

    /// Record a failure message.
    pub fn record_failure(&self, error: impl Into<String>) {
        let mut m = self.metrics.lock().unwrap();
        m.record_failure(error.into());
    }

    /// Get a snapshot of the current metrics.
    #[must_use]
    pub fn snapshot(&self) -> WorkflowMetrics {
        let m = self.metrics.lock().unwrap();
        m.clone()
    }

    /// Emit a structured workflow event to the trace log.
    ///
    /// Events are timestamped automatically when emitted.
    ///
    /// # Example
    ///
    /// ```rust
    /// use llm_workflow::{ExecutionContext, WorkflowEvent};
    ///
    /// let ctx = ExecutionContext::new();
    /// ctx.emit(WorkflowEvent::StepStart {
    ///     step_name: "Summarize".to_string(),
    ///     input_type: "Article".to_string(),
    /// });
    /// ```
    pub fn emit(&self, event: WorkflowEvent) {
        let entry = TraceEntry::new(event);
        self.traces.lock().unwrap().push(entry);
    }

    /// Emit an artifact event with automatic JSON serialization.
    ///
    /// This is a convenience method for recording intermediate outputs
    /// from workflow steps.
    ///
    /// # Example
    ///
    /// ```rust
    /// use llm_workflow::ExecutionContext;
    /// use serde::Serialize;
    ///
    /// #[derive(Serialize)]
    /// struct Summary { text: String, word_count: usize }
    ///
    /// let ctx = ExecutionContext::new();
    /// let summary = Summary { text: "...".to_string(), word_count: 150 };
    /// ctx.emit_artifact("Summarize", "output", &summary);
    /// ```
    pub fn emit_artifact<T: serde::Serialize>(&self, step_name: &str, key: &str, data: &T) {
        let json_data = serde_json::to_value(data)
            .unwrap_or_else(|_| serde_json::json!("<serialization_error>"));
        self.emit(WorkflowEvent::Artifact {
            step_name: step_name.to_string(),
            key: key.to_string(),
            data: json_data,
        });
    }

    /// Get a snapshot of the current trace log.
    ///
    /// Returns all trace entries recorded so far. Useful for debugging
    /// or exporting execution traces.
    #[must_use]
    pub fn trace_snapshot(&self) -> Vec<TraceEntry> {
        self.traces.lock().unwrap().clone()
    }

    /// Clear all trace entries.
    ///
    /// This can be useful when reusing a context across multiple workflow runs.
    pub fn clear_traces(&self) {
        self.traces.lock().unwrap().clear();
    }
}
