//! Core step trait and fundamental step types.
//!
//! This module defines the [`Step`] trait — the fundamental building block
//! of all workflows — along with [`LambdaStep`] for closure-based steps
//! and [`BoxedStepExt`] for fluent step composition.

use async_trait::async_trait;
use std::future::Future;
use std::marker::PhantomData;

use crate::{ExecutionContext, Result};

pub mod batch;
pub mod branch;
pub mod chain;
pub mod map;
pub mod parallel;
pub mod reduce;
pub mod tap;

pub use map::MapStep;

/// The fundamental trait for composable, async workflow steps.
///
/// Each step receives shared execution context (for metrics/tracing) and typed
/// input, and produces typed output or an error.
///
/// # Example
///
/// ```rust
/// use llm_workflow::{Step, ExecutionContext, LambdaStep};
///
/// let double = LambdaStep::new(|x: i32| async move {
///     Ok::<i32, llm_workflow::Error>(x * 2)
/// });
/// ```
#[async_trait]
pub trait Step: Send + Sync {
    /// The input type for this step.
    type Input: Send;
    /// The output type produced by this step.
    type Output: Send;

    /// Execute this step with the provided context and input.
    async fn run(&self, ctx: &ExecutionContext, input: Self::Input) -> Result<Self::Output>;

    /// Returns a human-readable name for this step. Defaults to the type name.
    fn name(&self) -> &str {
        std::any::type_name::<Self>()
    }
}

/// A step constructed from a closure or function pointer.
///
/// The type parameters `I` and `O` encode the input and output types,
/// while `F` is the concrete closure type.
///
/// # Example
///
/// ```rust
/// use llm_workflow::{LambdaStep, Step, ExecutionContext};
///
/// let step = LambdaStep::new(|x: i32| async move {
///     Ok::<i32, llm_workflow::Error>(x * 2)
/// });
/// ```
pub struct LambdaStep<I, O, F> {
    /// The underlying closure.
    pub f: F,
    _phantom: PhantomData<fn(I) -> O>,
}

impl<I, O, F, Fut> LambdaStep<I, O, F>
where
    F: Fn(I) -> Fut + Send + Sync + 'static,
    I: Send + 'static,
    O: Send + 'static,
    Fut: Future<Output = Result<O>> + Send + 'static,
{
    /// Create a new `LambdaStep` from the given closure.
    pub fn new(f: F) -> Self {
        Self {
            f,
            _phantom: PhantomData,
        }
    }
}

#[async_trait]
impl<F, I, O, Fut> Step for LambdaStep<I, O, F>
where
    F: Fn(I) -> Fut + Send + Sync + 'static,
    I: Send + 'static,
    O: Send + 'static,
    Fut: Future<Output = Result<O>> + Send + 'static,
{
    type Input = I;
    type Output = O;

    async fn run(&self, _ctx: &ExecutionContext, input: I) -> Result<O> {
        (self.f)(input).await
    }
}

/// Extension trait providing fluent composition methods for all [`Step`] implementors.
///
/// This trait is automatically implemented for every type that implements [`Step`].
///
/// # Methods
///
/// - [`BoxedStepExt::then`]: Chain two steps sequentially
/// - [`BoxedStepExt::map`]: Transform the output with a closure
/// - [`BoxedStepExt::tap`]: Inspect the output without modifying it
/// - [`BoxedStepExt::boxed`]: Erase the concrete type behind a `Box<dyn Step<...>>`
pub trait BoxedStepExt: Step + Sized {
    /// Chain this step with another, feeding this step's output into `next`.
    ///
    /// # Example
    ///
    /// ```rust
    /// use llm_workflow::{LambdaStep, BoxedStepExt};
    ///
    /// let pipeline = LambdaStep::new(|x: i32| async move { Ok::<i32, llm_workflow::Error>(x + 1) })
    ///     .then(LambdaStep::new(|x: i32| async move { Ok::<i32, llm_workflow::Error>(x * 3) }));
    /// ```
    fn then<S>(self, next: S) -> chain::ChainStep<Self, S>
    where
        S: Step<Input = Self::Output>,
        Self::Output: 'static,
    {
        chain::ChainStep::new(self, next)
    }

    /// Apply a synchronous transformation to this step's output.
    ///
    /// # Example
    ///
    /// ```rust
    /// use llm_workflow::{LambdaStep, BoxedStepExt};
    ///
    /// let step = LambdaStep::new(|x: i32| async move { Ok::<i32, llm_workflow::Error>(x) })
    ///     .map(|x: i32| x.to_string());
    /// ```
    fn map<F, O>(self, f: F) -> map::MapStep<Self, F>
    where
        F: Fn(Self::Output) -> O + Send + Sync + 'static,
        O: Send + 'static,
        Self::Output: 'static,
    {
        map::MapStep::new(self, f)
    }

    /// Inspect this step's output via a side-effect closure, passing it through unchanged.
    ///
    /// # Example
    ///
    /// ```rust
    /// use llm_workflow::{LambdaStep, BoxedStepExt};
    ///
    /// let step = LambdaStep::new(|x: i32| async move { Ok::<i32, llm_workflow::Error>(x) })
    ///     .tap(|x: &i32| println!("output: {x}"));
    /// ```
    fn tap<F>(self, f: F) -> tap::TapStep<Self, F>
    where
        F: Fn(&Self::Output) + Send + Sync + 'static,
        Self::Output: 'static,
    {
        tap::TapStep::new(self, f)
    }

    /// Erase the concrete step type, returning a trait object.
    fn boxed(self) -> Box<dyn Step<Input = Self::Input, Output = Self::Output> + Send + Sync>
    where
        Self: 'static,
    {
        Box::new(self)
    }
}

impl<T: Step + Sized> BoxedStepExt for T {}

// Implement Step for boxed steps so they can be used in chains.
#[async_trait]
impl<I, O> Step for Box<dyn Step<Input = I, Output = O> + Send + Sync>
where
    I: Send + 'static,
    O: Send + 'static,
{
    type Input = I;
    type Output = O;

    async fn run(&self, ctx: &ExecutionContext, input: I) -> Result<O> {
        (**self).run(ctx, input).await
    }
}
