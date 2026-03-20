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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{LambdaStep, ExecutionContext};

    fn ctx() -> ExecutionContext {
        ExecutionContext::default()
    }

    #[tokio::test]
    async fn test_map_step_transforms_output() {
        let inner = LambdaStep::new(|x: i32| async move { Ok(x + 1) });
        let mapped = MapStep::new(inner, |v: i32| v.to_string());
        let result = mapped.run(&ctx(), 5).await.unwrap();
        assert_eq!(result, "6");
    }

    #[tokio::test]
    async fn test_map_step_with_identity() {
        let inner = LambdaStep::new(|x: i32| async move { Ok(x * 3) });
        let mapped = MapStep::new(inner, |v: i32| v);
        let result = mapped.run(&ctx(), 4).await.unwrap();
        assert_eq!(result, 12);
    }

    #[tokio::test]
    async fn test_map_step_propagates_inner_error() {
        use crate::Error;
        let inner = LambdaStep::new(|_x: i32| async move {
            Err::<i32, _>(Error::Execution("inner err".to_string()))
        });
        let mapped = MapStep::new(inner, |v: i32| v * 2);
        assert!(mapped.run(&ctx(), 5).await.is_err());
    }
}
