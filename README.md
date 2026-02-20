# llm-workflow

[![Crates.io](https://img.shields.io/crates/v/llm-workflow)](https://crates.io/crates/llm-workflow)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue)](LICENSE-MIT)

Type-safe, async workflow primitives for LLM pipelines in Rust.

## Overview

`llm-workflow` provides composable building blocks for constructing complex LLM
pipelines with full type safety, async support, and observability via execution
metrics and structured event tracing.

## Features

- **`Step` trait** — the fundamental async, composable unit
- **`LambdaStep`** — wrap any closure as a step
- **`ChainStep`** — sequential composition (`step_a.then(step_b)`)
- **`MapStep`** — synchronous output transformation (`.map(|x| ...)`)
- **`TapStep`** — side-effect inspection without modifying output (`.tap(|x| ...)`)
- **`ParallelMapStep`** — fan-out a step over `Vec<Input>` concurrently
- **`BranchStep`** — conditional routing based on a predicate
- **`BatchStep` / `SingleItemAdapter`** — batch processing utilities
- **`CheckpointStep`** — human-in-the-loop pausing
- **`InstrumentedStep`** — automatic timing, event tracing, and metric recording
- **`StateStep` / `StepAdapter`** — stateful workflows (e.g. conversation history)
- **`Workflow`** — high-level container with automatic metrics collection

## Quick Start

```toml
[dependencies]
llm-workflow = "0.1"
```

```rust
use llm_workflow::{LambdaStep, Workflow, BoxedStepExt};

#[tokio::main]
async fn main() -> llm_workflow::Result<()> {
    let step_a = LambdaStep::new(|x: i32| async move { Ok(x * 2) });
    let step_b = LambdaStep::new(|x: i32| async move { Ok(x + 10) });

    let pipeline = step_a.then(step_b);
    let workflow = Workflow::new(pipeline).with_name("DoubleAndAdd");

    let (result, metrics) = workflow.run(5i32).await?;
    println!("Result: {result}"); // 20
    println!("Steps completed: {}", metrics.steps_completed);

    Ok(())
}
```

## License

Licensed under either of [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE) at your option.
