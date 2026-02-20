//! High-level workflow container with automatic metrics collection.

use crate::{ExecutionContext, Result, WorkflowMetrics, step::Step};

/// A high-level workflow wrapper that runs a step and collects execution metrics.
///
/// `Workflow` owns a step, creates a fresh [`ExecutionContext`] for each run,
/// and returns both the result and the accumulated [`WorkflowMetrics`].
///
/// # Example
///
/// ```rust
/// use llm_workflow::{LambdaStep, Workflow, BoxedStepExt};
///
/// # tokio_test::block_on(async {
/// let step = LambdaStep::new(|x: i32| async move { Ok::<i32, llm_workflow::Error>(x * 2) });
/// let workflow = Workflow::new(step).with_name("Double");
///
/// let (result, metrics) = workflow.run(5i32).await.unwrap();
/// assert_eq!(result, 10);
/// # });
/// ```
pub struct Workflow<S> {
    step: S,
    name: String,
}

impl<S: Step> Workflow<S> {
    /// Create a new workflow wrapping the given step.
    pub fn new(step: S) -> Self {
        Self {
            step,
            name: "workflow".to_string(),
        }
    }

    /// Set a human-readable name for this workflow.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Returns the name of this workflow.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Run the workflow, returning the result along with the collected metrics.
    ///
    /// A fresh [`ExecutionContext`] is created for each invocation.
    /// One step is automatically recorded in metrics on successful completion.
    pub async fn run(&self, input: S::Input) -> Result<(S::Output, WorkflowMetrics)> {
        let ctx = ExecutionContext::new();
        let result = self.step.run(&ctx, input).await?;
        ctx.record_step();
        let metrics = ctx.snapshot();
        Ok((result, metrics))
    }

    /// Run the workflow with a caller-provided execution context.
    ///
    /// Useful when you want to share a context across multiple workflow runs
    /// to accumulate metrics.
    pub async fn run_with_ctx(
        &self,
        ctx: &ExecutionContext,
        input: S::Input,
    ) -> Result<S::Output> {
        self.step.run(ctx, input).await
    }

    /// Access the inner step.
    pub fn inner(&self) -> &S {
        &self.step
    }

    /// Consume the workflow, returning the inner step.
    pub fn into_inner(self) -> S {
        self.step
    }
}
