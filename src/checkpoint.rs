//! Checkpoint steps for human-in-the-loop workflow pausing.
//!
//! Checkpoints emit a [`Error::Checkpoint`](crate::Error::Checkpoint) error,
//! which callers can catch to pause execution, review the current state,
//! and decide whether to continue or abort.

use async_trait::async_trait;

use crate::{Error, ExecutionContext, Result, step::Step};

/// A step that always pauses execution by emitting a checkpoint error.
///
/// The current input is serialized to JSON and embedded in the error,
/// allowing callers to inspect the workflow state at the checkpoint.
///
/// # Example
///
/// ```rust
/// use llm_workflow::{CheckpointStep, Step, ExecutionContext, Error};
///
/// # tokio_test::block_on(async {
/// let cp = CheckpointStep::<i32>::new("review");
/// let ctx = ExecutionContext::new();
/// let err = cp.run(&ctx, 42).await.unwrap_err();
///
/// match err {
///     Error::Checkpoint { step_name, .. } => assert_eq!(step_name, "review"),
///     _ => panic!("expected Checkpoint error"),
/// }
/// # });
/// ```
pub struct CheckpointStep<I> {
    step_name: String,
    _phantom: std::marker::PhantomData<fn(I)>,
}

impl<I> CheckpointStep<I> {
    /// Create a checkpoint step with the given name.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            step_name: name.into(),
            _phantom: std::marker::PhantomData,
        }
    }
}

#[async_trait]
impl<I> Step for CheckpointStep<I>
where
    I: Send + serde::Serialize + 'static,
{
    type Input = I;
    type Output = I;

    async fn run(&self, _ctx: &ExecutionContext, input: I) -> Result<I> {
        let data = serde_json::to_value(&input)
            .unwrap_or_else(|_| serde_json::json!("<serialization_error>"));
        Err(Error::Checkpoint {
            step_name: self.step_name.clone(),
            data,
        })
    }

    fn name(&self) -> &str {
        &self.step_name
    }
}

/// A step that conditionally pauses execution based on a predicate.
///
/// When the predicate returns `true`, a checkpoint error is emitted.
/// When it returns `false`, the input passes through unchanged.
///
/// # Example
///
/// ```rust
/// use llm_workflow::{ConditionalCheckpointStep, Step, ExecutionContext, Error};
///
/// # tokio_test::block_on(async {
/// let cp = ConditionalCheckpointStep::<i32, _>::new("high-value", |x: &i32| *x > 100);
/// let ctx = ExecutionContext::new();
///
/// // Does not checkpoint when predicate is false
/// let result = cp.run(&ctx, 50).await.unwrap();
/// assert_eq!(result, 50);
///
/// // Checkpoints when predicate is true
/// let err = cp.run(&ctx, 200).await.unwrap_err();
/// assert!(matches!(err, Error::Checkpoint { .. }));
/// # });
/// ```
pub struct ConditionalCheckpointStep<I, F> {
    step_name: String,
    predicate: F,
    _phantom: std::marker::PhantomData<fn(I)>,
}

impl<I, F> ConditionalCheckpointStep<I, F>
where
    F: Fn(&I) -> bool + Send + Sync,
{
    /// Create a conditional checkpoint step.
    pub fn new(name: impl Into<String>, predicate: F) -> Self {
        Self {
            step_name: name.into(),
            predicate,
            _phantom: std::marker::PhantomData,
        }
    }
}

#[async_trait]
impl<I, F> Step for ConditionalCheckpointStep<I, F>
where
    I: Send + serde::Serialize + 'static,
    F: Fn(&I) -> bool + Send + Sync + 'static,
{
    type Input = I;
    type Output = I;

    async fn run(&self, _ctx: &ExecutionContext, input: I) -> Result<I> {
        if (self.predicate)(&input) {
            let data = serde_json::to_value(&input)
                .unwrap_or_else(|_| serde_json::json!("<serialization_error>"));
            Err(Error::Checkpoint {
                step_name: self.step_name.clone(),
                data,
            })
        } else {
            Ok(input)
        }
    }

    fn name(&self) -> &str {
        &self.step_name
    }
}
