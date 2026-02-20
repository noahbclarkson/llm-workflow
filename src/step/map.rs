//! Map step for synchronous output transformation.

use async_trait::async_trait;

use crate::{ExecutionContext, Result};
use super::Step;

/// A step that applies a synchronous function to the output of an inner step.
///
/// Constructed via [`BoxedStepExt::map`](crate::BoxedStepExt::map).
pub struct MapStep<S, F> {
    step: S,
    f: F,
}

impl<S, F> MapStep<S, F> {
    /// Create a new map step.
    pub fn new(step: S, f: F) -> Self {
        Self { step, f }
    }
}

#[async_trait]
impl<S, F, O> Step for MapStep<S, F>
where
    S: Step,
    F: Fn(S::Output) -> O + Send + Sync + 'static,
    S::Output: 'static,
    O: Send + 'static,
{
    type Input = S::Input;
    type Output = O;

    async fn run(&self, ctx: &ExecutionContext, input: S::Input) -> Result<O> {
        let output = self.step.run(ctx, input).await?;
        Ok((self.f)(output))
    }
}
