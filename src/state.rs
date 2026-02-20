//! Stateful workflow steps.
//!
//! This module provides primitives for steps that carry and update state across
//! invocations — useful for conversational agents, accumulators, or any workflow
//! that needs to remember previous results.

use async_trait::async_trait;
use std::future::Future;
use std::marker::PhantomData;
use std::sync::{Arc, Mutex};

use crate::{ExecutionContext, Result, step::Step};

/// A step that carries mutable state, returning both an output and the updated state.
///
/// Unlike [`Step`], `StateStep` takes ownership of the state on each call and
/// returns the new state alongside the output.
pub trait StateStep: Send + Sync {
    /// The input type consumed by each invocation.
    type Input: Send;
    /// The output type produced by each invocation.
    type Output: Send;
    /// The state type threaded through invocations.
    type State: Send;

    /// Execute with the given state and input, returning output and new state.
    #[allow(clippy::type_complexity)]
    fn run<'life0, 'async_trait>(
        &'life0 self,
        ctx: &'life0 ExecutionContext,
        state: Self::State,
        input: Self::Input,
    ) -> std::pin::Pin<
        Box<dyn Future<Output = Result<(Self::Output, Self::State)>> + Send + 'async_trait>,
    >
    where
        'life0: 'async_trait,
        Self: 'async_trait;
}

/// A stateful step constructed from a closure.
///
/// The type parameters `S` (state), `O` (output), `I` (input), `F` (function type),
/// and `Fut` (future type) are inferred from the closure.
///
/// Use [`LambdaStateStep::new`] to construct one.
///
/// # Example
///
/// ```rust
/// use llm_workflow::state::{LambdaStateStep, StateStep};
/// use llm_workflow::ExecutionContext;
///
/// # tokio_test::block_on(async {
/// let step = LambdaStateStep::new(|count: u32, x: i32| async move {
///     Ok::<(String, u32), llm_workflow::Error>((format!("#{count}: {x}"), count + 1))
/// });
///
/// let ctx = ExecutionContext::new();
/// let (out, new_count) = step.run(&ctx, 0u32, 42i32).await.unwrap();
/// assert_eq!(out, "#0: 42");
/// assert_eq!(new_count, 1);
/// # });
/// ```
pub struct LambdaStateStep<S, O, I, F, Fut> {
    f: F,
    _phantom: PhantomData<fn(S, I) -> Fut>,
    _out: PhantomData<fn() -> O>,
}

impl<S, O, I, F, Fut> LambdaStateStep<S, O, I, F, Fut>
where
    F: Fn(S, I) -> Fut + Send + Sync + 'static,
    I: Send + 'static,
    O: Send + 'static,
    S: Send + 'static,
    Fut: Future<Output = Result<(O, S)>> + Send + 'static,
{
    /// Create a new stateful lambda step from the given closure.
    pub fn new(f: F) -> Self {
        Self {
            f,
            _phantom: PhantomData,
            _out: PhantomData,
        }
    }
}

impl<F, I, O, S, Fut> StateStep for LambdaStateStep<S, O, I, F, Fut>
where
    F: Fn(S, I) -> Fut + Send + Sync + 'static,
    I: Send + 'static,
    O: Send + 'static,
    S: Send + 'static,
    Fut: Future<Output = Result<(O, S)>> + Send + 'static,
{
    type Input = I;
    type Output = O;
    type State = S;

    #[allow(clippy::type_complexity)]
    fn run<'life0, 'async_trait>(
        &'life0 self,
        _ctx: &'life0 ExecutionContext,
        state: S,
        input: I,
    ) -> std::pin::Pin<Box<dyn Future<Output = Result<(O, S)>> + Send + 'async_trait>>
    where
        'life0: 'async_trait,
        Self: 'async_trait,
    {
        Box::pin((self.f)(state, input))
    }
}

/// A high-level wrapper that runs a [`StateStep`] and collects execution metrics.
///
/// Unlike [`Workflow`](crate::Workflow), `StateWorkflow` requires the caller to
/// supply and receive state on each invocation.
pub struct StateWorkflow<SS> {
    step: SS,
    name: String,
}

impl<SS: StateStep> StateWorkflow<SS> {
    /// Create a new stateful workflow wrapping the given step.
    pub fn new(step: SS) -> Self {
        Self {
            step,
            name: "state_workflow".to_string(),
        }
    }

    /// Set a human-readable name for this workflow.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Returns the workflow name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Run one step of the stateful workflow.
    ///
    /// Returns the output and the updated state.
    pub async fn run(
        &self,
        state: SS::State,
        input: SS::Input,
    ) -> Result<(SS::Output, SS::State)> {
        let ctx = ExecutionContext::new();
        self.step.run(&ctx, state, input).await
    }

    /// Run one step using a caller-provided execution context.
    pub async fn run_with_ctx(
        &self,
        ctx: &ExecutionContext,
        state: SS::State,
        input: SS::Input,
    ) -> Result<(SS::Output, SS::State)> {
        self.step.run(ctx, state, input).await
    }
}

/// Adapts a [`StateStep`] into a regular [`Step`] by storing state internally.
///
/// State is initialised from [`Default`] and updated after each call.
/// Thread-safe: the internal state is protected by a `Mutex`.
///
/// # Example
///
/// ```rust
/// use llm_workflow::state::{LambdaStateStep, StepAdapter};
/// use llm_workflow::{Step, ExecutionContext};
///
/// # tokio_test::block_on(async {
/// let state_step = LambdaStateStep::new(|count: u32, x: i32| async move {
///     Ok::<(String, u32), llm_workflow::Error>((format!("#{count}: {x}"), count + 1))
/// });
///
/// let adapter = StepAdapter::new(state_step);
/// let ctx = ExecutionContext::new();
///
/// let out1 = adapter.run(&ctx, 10i32).await.unwrap();
/// let out2 = adapter.run(&ctx, 20i32).await.unwrap();
/// assert_eq!(out1, "#0: 10");
/// assert_eq!(out2, "#1: 20");
/// # });
/// ```
pub struct StepAdapter<SS>
where
    SS: StateStep,
    SS::State: Default,
{
    step: SS,
    state: Arc<Mutex<SS::State>>,
}

impl<SS> StepAdapter<SS>
where
    SS: StateStep,
    SS::State: Default,
{
    /// Create a new adapter, initialising state with `Default::default()`.
    pub fn new(step: SS) -> Self {
        Self {
            step,
            state: Arc::new(Mutex::new(SS::State::default())),
        }
    }

    /// Reset the internal state to its default value.
    pub fn reset(&self) {
        let mut guard = self.state.lock().unwrap();
        *guard = SS::State::default();
    }
}

#[async_trait]
impl<SS> Step for StepAdapter<SS>
where
    SS: StateStep + 'static,
    SS::Input: 'static,
    SS::Output: 'static,
    SS::State: Default + Clone + Send + 'static,
{
    type Input = SS::Input;
    type Output = SS::Output;

    async fn run(&self, ctx: &ExecutionContext, input: SS::Input) -> Result<SS::Output> {
        let state = {
            let guard = self.state.lock().unwrap();
            guard.clone()
        };

        let (output, new_state) = self.step.run(ctx, state, input).await?;

        {
            let mut guard = self.state.lock().unwrap();
            *guard = new_state;
        }

        Ok(output)
    }
}
