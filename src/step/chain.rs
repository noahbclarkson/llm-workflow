//! Sequential step composition.

use async_trait::async_trait;

use crate::{ExecutionContext, Result};
use super::Step;

/// Two steps composed sequentially: the output of `A` feeds into `B`.
///
/// Constructed via [`BoxedStepExt::then`](crate::BoxedStepExt::then).
pub struct ChainStep<A, B> {
    first: A,
    second: B,
}

impl<A, B> ChainStep<A, B> {
    /// Create a new chained step.
    pub fn new(first: A, second: B) -> Self {
        Self { first, second }
    }
}

#[async_trait]
impl<A, B> Step for ChainStep<A, B>
where
    A: Step,
    B: Step<Input = A::Output>,
    A::Output: 'static,
{
    type Input = A::Input;
    type Output = B::Output;

    async fn run(&self, ctx: &ExecutionContext, input: A::Input) -> Result<B::Output> {
        let intermediate = self.first.run(ctx, input).await?;
        self.second.run(ctx, intermediate).await
    }
}

/// A step that fans out a single input to two independent steps and returns both outputs.
///
/// Both steps receive a clone of the input and execute sequentially.
/// For parallel fan-out, wrap each step in a [`ParallelMapStep`](crate::ParallelMapStep).
pub struct ChainTupleStep<A, B> {
    first: A,
    second: B,
}

impl<A, B> ChainTupleStep<A, B> {
    /// Create a new tuple step from two independent steps.
    pub fn new(first: A, second: B) -> Self {
        Self { first, second }
    }
}

#[async_trait]
impl<A, B, I> Step for ChainTupleStep<A, B>
where
    A: Step<Input = I>,
    B: Step<Input = I>,
    I: Clone + Send + 'static,
    A::Output: Send + 'static,
    B::Output: Send + 'static,
{
    type Input = I;
    type Output = (A::Output, B::Output);

    async fn run(&self, ctx: &ExecutionContext, input: I) -> Result<(A::Output, B::Output)> {
        let a = self.first.run(ctx, input.clone()).await?;
        let b = self.second.run(ctx, input).await?;
        Ok((a, b))
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
    async fn test_chain_step_passes_output_to_next() {
        let a = LambdaStep::new(|x: i32| async move { Ok(x + 10) });
        let b = LambdaStep::new(|x: i32| async move { Ok(x * 2) });
        let chain = ChainStep::new(a, b);
        let result = chain.run(&ctx(), 5).await.unwrap();
        assert_eq!(result, 30, "(5 + 10) * 2 = 30");
    }

    #[tokio::test]
    async fn test_chain_step_propagates_error_from_first() {
        let a = LambdaStep::new(|_x: i32| async move {
            Err::<i32, _>(Error::Execution("first failed".to_string()))
        });
        let b = LambdaStep::new(|x: i32| async move { Ok(x) });
        let chain = ChainStep::new(a, b);
        assert!(chain.run(&ctx(), 5).await.is_err());
    }

    #[tokio::test]
    async fn test_chain_step_propagates_error_from_second() {
        let a = LambdaStep::new(|x: i32| async move { Ok(x) });
        let b = LambdaStep::new(|_x: i32| async move {
            Err::<i32, _>(Error::Execution("second failed".to_string()))
        });
        let chain = ChainStep::new(a, b);
        assert!(chain.run(&ctx(), 5).await.is_err());
    }

    #[tokio::test]
    async fn test_chain_tuple_step_runs_both() {
        let a = LambdaStep::new(|x: i32| async move { Ok(x + 1) });
        let b = LambdaStep::new(|x: i32| async move { Ok(x * 2) });
        let pair = ChainTupleStep::new(a, b);
        let (left, right) = pair.run(&ctx(), 5).await.unwrap();
        assert_eq!(left, 6, "a: 5 + 1 = 6");
        assert_eq!(right, 10, "b: 5 * 2 = 10");
    }
}
