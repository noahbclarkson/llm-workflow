//! Reduce step for aggregating collections into a single value.

use async_trait::async_trait;
use std::marker::PhantomData;

use crate::{ExecutionContext, Result};
use super::Step;

/// A step that reduces a `Vec<I>` into a single value `O` using a synchronous function.
///
/// Useful for aggregating results from a parallel or batch step.
///
/// # Example
///
/// ```rust
/// use llm_workflow::step::reduce::ReduceStep;
/// use llm_workflow::Step;
///
/// let summer = ReduceStep::<_, i32, i32>::new(|items: Vec<i32>| items.into_iter().sum::<i32>());
/// ```
pub struct ReduceStep<F, I, O> {
    f: F,
    _phantom: PhantomData<fn(Vec<I>) -> O>,
}

impl<F, I, O> ReduceStep<F, I, O>
where
    F: Fn(Vec<I>) -> O + Send + Sync + 'static,
    I: Send + 'static,
    O: Send + 'static,
{
    /// Create a new reduce step from the given aggregation function.
    pub fn new(f: F) -> Self {
        Self {
            f,
            _phantom: PhantomData,
        }
    }
}

#[async_trait]
impl<F, I, O> Step for ReduceStep<F, I, O>
where
    F: Fn(Vec<I>) -> O + Send + Sync + 'static,
    I: Send + 'static,
    O: Send + 'static,
{
    type Input = Vec<I>;
    type Output = O;

    async fn run(&self, _ctx: &ExecutionContext, input: Vec<I>) -> Result<O> {
        Ok((self.f)(input))
    }
}
