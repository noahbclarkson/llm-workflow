//! Batch processing steps.

use async_trait::async_trait;

use crate::{ExecutionContext, Result};
use super::Step;

/// A step that adapts a single-item step into a batch step processing `Vec<Input>`.
///
/// Items are processed sequentially. For parallel processing, use
/// [`ParallelMapStep`](crate::ParallelMapStep) instead.
pub struct SingleItemAdapter<S> {
    step: S,
}

impl<S> SingleItemAdapter<S> {
    /// Wrap a single-item step for batch use.
    pub fn new(step: S) -> Self {
        Self { step }
    }
}

#[async_trait]
impl<S> Step for SingleItemAdapter<S>
where
    S: Step + 'static,
    S::Input: 'static,
    S::Output: 'static,
{
    type Input = Vec<S::Input>;
    type Output = Vec<S::Output>;

    async fn run(&self, ctx: &ExecutionContext, input: Vec<S::Input>) -> Result<Vec<S::Output>> {
        let mut results = Vec::with_capacity(input.len());
        for item in input {
            results.push(self.step.run(ctx, item).await?);
        }
        Ok(results)
    }
}

/// A step that processes a `Vec<I>` in fixed-size batches using an inner batch step.
///
/// The inner step must accept `Vec<I>` and return `Vec<O>`. Large inputs are split
/// into chunks of `batch_size` and each chunk is processed in sequence.
pub struct BatchStep<S> {
    step: S,
    batch_size: usize,
}

impl<S> BatchStep<S> {
    /// Create a new batch step with the given chunk size.
    ///
    /// # Panics
    ///
    /// Panics if `batch_size` is zero.
    pub fn new(step: S, batch_size: usize) -> Self {
        assert!(batch_size > 0, "batch_size must be greater than zero");
        Self { step, batch_size }
    }
}

#[async_trait]
impl<S, I, O> Step for BatchStep<S>
where
    S: Step<Input = Vec<I>, Output = Vec<O>>,
    I: Send + 'static,
    O: Send + 'static,
{
    type Input = Vec<I>;
    type Output = Vec<O>;

    async fn run(&self, ctx: &ExecutionContext, input: Vec<I>) -> Result<Vec<O>> {
        let mut all_outputs = Vec::new();
        let mut remaining = input;

        while !remaining.is_empty() {
            let batch_size = self.batch_size.min(remaining.len());
            let batch: Vec<I> = remaining.drain(..batch_size).collect();
            let outputs = self.step.run(ctx, batch).await?;
            all_outputs.extend(outputs);
        }

        Ok(all_outputs)
    }
}
