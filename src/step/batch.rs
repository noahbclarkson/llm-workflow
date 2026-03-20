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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{LambdaStep, ExecutionContext, Error};

    fn ctx() -> ExecutionContext {
        ExecutionContext::default()
    }

    #[tokio::test]
    async fn test_single_item_adapter_processes_each() {
        let step = LambdaStep::new(|x: i32| async move { Ok(x * 2) });
        let adapter = SingleItemAdapter::new(step);
        let result = adapter.run(&ctx(), vec![1, 2, 3, 4]).await.unwrap();
        assert_eq!(result, vec![2, 4, 6, 8]);
    }

    #[tokio::test]
    async fn test_single_item_adapter_empty_input() {
        let step = LambdaStep::new(|x: i32| async move { Ok(x) });
        let adapter = SingleItemAdapter::new(step);
        let result = adapter.run(&ctx(), vec![]).await.unwrap();
        assert!(result.is_empty());
    }

    #[tokio::test]
    async fn test_batch_step_processes_in_chunks() {
        // Inner step receives chunks and doubles each element
        let inner = LambdaStep::new(|v: Vec<i32>| async move {
            Ok(v.into_iter().map(|x| x * 2).collect::<Vec<_>>())
        });
        let batch = BatchStep::new(inner, 2);
        let result = batch.run(&ctx(), vec![1, 2, 3, 4, 5]).await.unwrap();
        assert_eq!(result, vec![2, 4, 6, 8, 10]);
    }

    #[tokio::test]
    async fn test_batch_step_single_batch() {
        let inner = LambdaStep::new(|v: Vec<i32>| async move {
            Ok(v.into_iter().map(|x| x + 1).collect::<Vec<_>>())
        });
        let batch = BatchStep::new(inner, 10);
        let result = batch.run(&ctx(), vec![1, 2, 3]).await.unwrap();
        assert_eq!(result, vec![2, 3, 4]);
    }

    #[test]
    #[should_panic(expected = "batch_size must be greater than zero")]
    fn test_batch_step_panics_on_zero_size() {
        let step = LambdaStep::new(|v: Vec<i32>| async move { Ok(v) });
        let _ = BatchStep::new(step, 0);
    }

    #[tokio::test]
    async fn test_batch_step_propagates_error() {
        let inner = LambdaStep::new(|_v: Vec<i32>| async move {
            Err::<Vec<i32>, _>(Error::Execution("batch err".to_string()))
        });
        let batch = BatchStep::new(inner, 2);
        assert!(batch.run(&ctx(), vec![1, 2, 3]).await.is_err());
    }
}
