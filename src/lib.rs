//! # llm-workflow
//!
//! Type-safe, async workflow primitives for LLM pipelines in Rust.
//!
//! This crate provides building blocks for creating complex workflows
//! with composable steps, parallel execution, aggregation, and
//! observability through execution metrics.
//!
//! ## Core Concepts
//!
//! - **Step**: The fundamental trait for workflow units
//! - **ExecutionContext**: Shared context for metrics collection
//! - **WorkflowMetrics**: Aggregated usage and execution statistics
//! - **ChainStep**: Sequential composition of steps
//! - **MapStep**: Inline transformations between steps
//! - **ParallelMapStep**: Apply a step to multiple inputs concurrently
//! - **BranchStep**: Conditional routing based on predicates
//! - **TapStep**: Side-effect inspection without modifying output
//! - **CheckpointStep**: Human-in-the-loop pausing
//! - **Workflow**: High-level container with automatic metrics collection
//!
//! ## Example: Fluent Pipeline with Metrics
//!
//! ```rust
//! use llm_workflow::{Step, Workflow, ExecutionContext, LambdaStep, BoxedStepExt};
//!
//! # tokio_test::block_on(async {
//! // Create steps using LambdaStep::new
//! let step_a = LambdaStep::new(|x: i32| async move { Ok::<i32, llm_workflow::Error>(x * 2) });
//! let step_b = LambdaStep::new(|x: i32| async move { Ok::<i32, llm_workflow::Error>(x + 10) });
//!
//! // Chain steps fluently
//! let pipeline = step_a.then(step_b);
//!
//! // Run with automatic metrics collection
//! let workflow = Workflow::new(pipeline).with_name("DoubleAndAdd");
//! let (result, metrics) = workflow.run(5).await.unwrap();
//!
//! assert_eq!(result, 20); // (5 * 2) + 10 = 20
//! assert_eq!(metrics.steps_completed, 1);
//! # });
//! ```

pub mod error;
pub mod context;
pub mod metrics;
pub mod events;
pub mod workflow;
pub mod step;
pub mod checkpoint;
pub mod instrumented;
pub mod state;

pub use error::{Error, Result};
pub use context::ExecutionContext;
pub use metrics::WorkflowMetrics;
pub use events::{TraceEntry, WorkflowEvent};
pub use workflow::Workflow;
pub use checkpoint::{CheckpointStep, ConditionalCheckpointStep};
pub use instrumented::InstrumentedStep;
pub use state::{StateStep, StateWorkflow, LambdaStateStep, StepAdapter};

// Re-export step types
pub use step::{Step, LambdaStep, MapStep, BoxedStepExt};
pub use step::chain::{ChainStep, ChainTupleStep};
pub use step::map::MapStep as MapStepType;
pub use step::tap::TapStep;
pub use step::parallel::{ParallelMapStep, ParallelMapBuilder};
pub use step::reduce::ReduceStep;
pub use step::batch::{BatchStep, SingleItemAdapter};
pub use step::branch::BranchStep;
