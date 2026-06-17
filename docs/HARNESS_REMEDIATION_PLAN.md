# Harness Remediation Plan

Persistent Agent must use the `oh-my-harness` framework repositories referenced by
[`oh-my-harness/llm-harness-skills`](https://github.com/oh-my-harness/llm-harness-skills).

## Allowed Framework Sources

The only approved agent framework sources are:

- `llm_adapter` from `https://github.com/oh-my-harness/llm-api-adapter.git`
- `llm_harness_core` from `https://github.com/oh-my-harness/llm-harness-core.git`
- `llm_harness_runtime` from `https://github.com/oh-my-harness/llm-harness-runtime.git`

Do not use crates or repositories from other publishers as substitutes for the provider adapter,
agent loop, runtime, tool registry, hooks, sub-agent support, or skill system.

## Immediate Remediation

1. Remove the non-approved `llm_runtime` dependency. Done.
2. Change `llm_adapter` to the approved `oh-my-harness/llm-api-adapter` git dependency. Done.
3. Remove the temporary `HarnessWorker` built on the non-approved runtime loop. Done.
4. Restore the worker path to either:
   - `StubWorker` when no model key is configured, or
   - `OhMyHarnessWorker`, which executes through `AgentHarness` from `llm-harness-core` and calls DeepSeek through the approved `llm-api-adapter`.
5. Add `oh-my-harness` core/runtime dependencies only after their public API is inspected and the worker can use them directly. Core is wired; runtime is still pending repository availability.

## Target Worker Shape

The final worker path should be:

```text
Scheduler -> TaskWorker -> OhMyHarnessWorker -> AgentHarness -> WorkerResult
```

`OhMyHarnessWorker` must use:

- `AgentHarness` from `llm-harness-core` (wired)
- provider calls through `llm-api-adapter` (wired)
- runtime tool registry, hooks, audit, and sub-agent services from `llm-harness-runtime` (pending repository availability)

## Product Context To Inject

The harness worker must receive:

- task title, description, type, priority, requested skills, and matched skills
- approved memories selected for the task
- task notes
- recent task conversation
- active product skill metadata and resource paths
- allowed product tools derived from active skills

## Product Tools To Register First

The first harness-backed product tools are:

- `complete_task` (wired)
- `block_task` (wired)
- `remember` (wired)
- `record_artifact` (wired)
- `create_follow_up_task` (wired)

External tools such as shell, browser, git, and GitHub must be added later through the approved
runtime tool policy and audit path.

## Verification Gates

Each remediation step must pass:

```powershell
cargo test
cd apps/web
npm run build
```

The dependency gate is:

- `Cargo.toml` must not contain non-approved runtime/agent framework dependencies.
- `Cargo.lock` must resolve the framework crates from approved `oh-my-harness` git repositories.
