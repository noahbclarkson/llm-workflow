//! Tap step for side-effect inspection.

use async_trait::async_trait;

use crate::{ExecutionContext, Result};
use super::Step;

/// A step that runs a side-effect closure on the output without modifying it.
///
/// Useful for logging, metrics, or debugging intermediate results in a pipeline.
/// Constructed via [`BoxedStepExt::tap`](crate::BoxedStepExt::tap).
pub struct TapStep<S, F> {
    step: S,
    f: F,
}

impl<S, F> TapStep<S, F> {
    /// Create a new tap step.
    pub fn new(step: S, f: F) -> Self {
        Self { step, f }
    }
}

#[async_trait]
impl<S, F> Step for TapStep<S, F>
where
    S: Step,
    F: Fn(&S::Output) + Send + Sync + 'static,
    S::Output: 'static,
{
    type Input = S::Input;
    type Output = S::Output;

    async fn run(&self, ctx: &ExecutionContext, input: S::Input) -> Result<S::Output> {
        let output = self.step.run(ctx, input).await?;
        (self.f)(&output);
        Ok(output)
    }
}
