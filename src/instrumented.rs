//! Instrumented step wrapper for automatic tracing and metrics.

use async_trait::async_trait;
use std::time::Instant;

use crate::{ExecutionContext, Result, WorkflowEvent, step::Step};

/// Wraps any step with automatic event emission and metric recording.
///
/// For each execution, `InstrumentedStep` emits:
/// - A [`WorkflowEvent::StepStart`] before the inner step runs
/// - A [`WorkflowEvent::StepEnd`] with elapsed milliseconds on success
/// - A [`WorkflowEvent::Error`] and failure metric on error
///
/// # Example
///
/// ```rust
/// use llm_workflow::{LambdaStep, InstrumentedStep, Step, ExecutionContext};
///
/// # tokio_test::block_on(async {
/// let step = LambdaStep::new(|x: i32| async move { Ok::<i32, llm_workflow::Error>(x + 1) });
/// let instrumented = InstrumentedStep::new(step, "Increment");
///
/// let ctx = ExecutionContext::new();
/// let result = instrumented.run(&ctx, 5i32).await.unwrap();
/// assert_eq!(result, 6);
///
/// let traces = ctx.trace_snapshot();
/// assert_eq!(traces.len(), 2); // StepStart + StepEnd
/// # });
/// ```
pub struct InstrumentedStep<S> {
    inner: S,
    name: String,
}

impl<S: Step> InstrumentedStep<S> {
    /// Wrap `inner` with instrumentation, labelling it `name`.
    pub fn new(inner: S, name: impl Into<String>) -> Self {
        Self {
            inner,
            name: name.into(),
        }
    }

    /// Access the inner step.
    pub fn inner(&self) -> &S {
        &self.inner
    }
}

#[async_trait]
impl<S> Step for InstrumentedStep<S>
where
    S: Step,
    S::Input: 'static,
    S::Output: 'static,
{
    type Input = S::Input;
    type Output = S::Output;

    async fn run(&self, ctx: &ExecutionContext, input: S::Input) -> Result<S::Output> {
        ctx.emit(WorkflowEvent::StepStart {
            step_name: self.name.clone(),
            input_type: std::any::type_name::<S::Input>().to_string(),
        });

        let start = Instant::now();
        let result = self.inner.run(ctx, input).await;
        let duration_ms = start.elapsed().as_millis();

        match &result {
            Ok(_) => {
                ctx.record_step();
                ctx.emit(WorkflowEvent::StepEnd {
                    step_name: self.name.clone(),
                    duration_ms,
                });
            }
            Err(e) => {
                ctx.record_failure(e.to_string());
                ctx.emit(WorkflowEvent::Error {
                    step_name: self.name.clone(),
                    message: e.to_string(),
                });
            }
        }

        result
    }

    fn name(&self) -> &str {
        &self.name
    }
}
