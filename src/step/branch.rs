//! Conditional step routing.

use async_trait::async_trait;

use crate::{ExecutionContext, Result};
use super::Step;

/// A step that routes to one of two steps based on a predicate over the input.
///
/// If the predicate returns `true`, `left` is executed; otherwise `right` is.
/// Both branches must have the same input and output types.
///
/// # Example
///
/// ```rust
/// use llm_workflow::{LambdaStep, step::branch::BranchStep, Step};
///
/// let branch = BranchStep::new(
///     |x: &i32| *x > 0,
///     LambdaStep::new(|x: i32| async move { Ok::<String, llm_workflow::Error>(format!("positive: {x}")) }),
///     LambdaStep::new(|x: i32| async move { Ok::<String, llm_workflow::Error>(format!("non-positive: {x}")) }),
/// );
/// ```
pub struct BranchStep<P, L, R> {
    predicate: P,
    left: L,
    right: R,
}

impl<P, L, R> BranchStep<P, L, R> {
    /// Create a new branch step.
    ///
    /// - `predicate`: Determines which branch to take (borrow of input).
    /// - `left`: Executed when predicate is `true`.
    /// - `right`: Executed when predicate is `false`.
    pub fn new(predicate: P, left: L, right: R) -> Self {
        Self {
            predicate,
            left,
            right,
        }
    }
}

#[async_trait]
impl<P, L, R, I, O> Step for BranchStep<P, L, R>
where
    P: Fn(&I) -> bool + Send + Sync + 'static,
    L: Step<Input = I, Output = O>,
    R: Step<Input = I, Output = O>,
    I: Send + 'static,
    O: Send + 'static,
{
    type Input = I;
    type Output = O;

    async fn run(&self, ctx: &ExecutionContext, input: I) -> Result<O> {
        if (self.predicate)(&input) {
            self.left.run(ctx, input).await
        } else {
            self.right.run(ctx, input).await
        }
    }
}
