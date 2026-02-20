//! Parallel step execution over collections.

use async_trait::async_trait;
use futures::future::join_all;
use std::sync::Arc;

use crate::{ExecutionContext, Result};
use super::Step;

/// A step that applies an inner step to each element of a `Vec` concurrently.
///
/// All items are processed in parallel using `join_all`. If any step fails,
/// the first error is returned.
pub struct ParallelMapStep<S> {
    step: Arc<S>,
}

impl<S> ParallelMapStep<S> {
    /// Create a new parallel map step wrapping the given step.
    pub fn new(step: S) -> Self {
        Self {
            step: Arc::new(step),
        }
    }

    /// Access the inner step.
    pub fn inner(&self) -> &S {
        &self.step
    }
}

#[async_trait]
impl<S> Step for ParallelMapStep<S>
where
    S: Step + 'static,
    S::Input: 'static,
    S::Output: 'static,
{
    type Input = Vec<S::Input>;
    type Output = Vec<S::Output>;

    async fn run(&self, ctx: &ExecutionContext, input: Vec<S::Input>) -> Result<Vec<S::Output>> {
        let futures = input.into_iter().map(|item| {
            let step = Arc::clone(&self.step);
            let ctx = ctx.clone();
            async move { step.run(&ctx, item).await }
        });

        let results = join_all(futures).await;
        results.into_iter().collect()
    }
}

/// Builder for configuring and constructing a [`ParallelMapStep`].
///
/// # Example
///
/// ```rust
/// use llm_workflow::{LambdaStep, step::parallel::ParallelMapBuilder};
///
/// let step = LambdaStep::new(|x: i32| async move { Ok::<i32, llm_workflow::Error>(x * 2) });
/// let parallel = ParallelMapBuilder::new(step).build();
/// ```
pub struct ParallelMapBuilder<S> {
    step: S,
}

impl<S: Step> ParallelMapBuilder<S> {
    /// Create a builder for the given step.
    pub fn new(step: S) -> Self {
        Self { step }
    }

    /// Build the [`ParallelMapStep`].
    pub fn build(self) -> ParallelMapStep<S> {
        ParallelMapStep::new(self.step)
    }
}
