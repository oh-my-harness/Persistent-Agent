# Harness Remediation Plan

Persistent Agent must use the `oh-my-harness` framework repositories referenced by
[`oh-my-harness/llm-harness-skills`](https://github.com/oh-my-harness/llm-harness-skills).

## Allowed Framework Sources

The only approved agent framework sources are:

- `llm_adapter` from `https://github.com/oh-my-harness/llm-api-adapter.git`
- `llm_harness_core` from `https://github.com/oh-my-harness/llm-harness-core.git`
- `llm_harness_runtime` from `https://github.com/oh-my-harness/llm-harness-runtime.git`

Runtime v0.2 publishes `llm-harness-agent`, `llm-harness-loop`, `llm-harness-runtime`, and
runtime extension crates from one workspace. Product code should keep those crates on the same git
source to avoid duplicate `Tool`, `AgentEvent`, and `ExecutionEnv` trait instances.

Do not use crates or repositories from other publishers as substitutes for the provider adapter,
agent loop, runtime, tool registry, hooks, sub-agent support, or skill system.

## Immediate Remediation

1. Remove the non-approved `llm_runtime` dependency. Done.
2. Change `llm_adapter` to the approved `oh-my-harness/llm-api-adapter` git dependency. Done.
3. Remove the temporary `HarnessWorker` built on the non-approved runtime loop. Done.
4. Restore the worker path to either:
   - `StubWorker` when no model key is configured, or
   - `OhMyHarnessWorker`, which executes through `AgentHarness` from the approved runtime v0.2 `llm-harness-agent` package and calls DeepSeek through the approved `llm-api-adapter`. Done.
5. Add `oh-my-harness` core/runtime dependencies only after their public API is inspected and the worker can use them directly. Done for `llm-harness-agent`, `llm-harness-runtime`, and `llm-harness-runtime-sandbox-os`.

## Target Worker Shape

The final worker path should be:

```text
Scheduler -> TaskWorker -> OhMyHarnessWorker -> AgentHarness -> WorkerResult
```

The main-agent conversation path should also use the approved harness where LLM assistance is
enabled:

```text
User message -> deterministic MainAgent task operation -> OhMyHarnessMainAgentAdvisor -> AgentHarness -> conversational reply
```

The deterministic product operation remains the authority for task state changes. The advisor does
not receive state-changing tools; it composes the final user-facing response from the verified action
context and falls back to the deterministic reply on empty output or LLM failure.

The deterministic main-agent operation layer includes skill definition management, so users can
create, list, update, and delete skills through the main conversation while worker execution still
receives skills through the approved harness context path.

`OhMyHarnessWorker` must use:

- `AgentHarness` from the approved runtime v0.2 `llm-harness-agent` package (wired)
- provider calls through `llm-api-adapter` (wired)
- runtime `InMemoryToolRegistry` from `llm-harness-runtime` (wired)
- runtime `OsEnvSandbox` from `llm-harness-runtime-sandbox-os` (wired)
- runtime hooks, audit, sub-agent, auth/approval, and budget services (available as dependencies; product policies still need dedicated feature work before they should affect live task execution)

## Product Context To Inject

The harness worker must receive:

- task title, description, type, priority, requested skills, and matched skills
- approved memories selected for the task
- task notes
- recent task conversation
- active product skill metadata, resource paths, and loaded workspace-relative skill instructions
- allowed product tools derived from active skills

Skill `resource_path` values are loaded by the scheduler before worker execution. A directory path
resolves to `SKILL.md`; a file path is loaded directly. Paths must stay inside the workspace. Load
errors are injected into context and attempt events so the worker can proceed while operators can
see why a skill resource was unavailable.

## Product Tools To Register First

The first harness-backed product tools are:

- `complete_task` (wired)
- `block_task` (wired)
- `remember` (wired)
- `record_artifact` (wired)
- `create_follow_up_task` (wired)
- `read_file` / `write_file` / `append_file` / `list_dir` (wired through runtime `ExecutionEnv`)
- `shell` (wired through runtime `ExecutionEnv`)
- `http_fetch` (wired for HTTP(S) lookups)
- `github_list_issues` (wired for read-only GitHub issue discovery)

Specialized tools such as browser automation, first-class GitHub issue/PR operations, and richer git
workflows should be added later through the approved runtime tool policy and audit path.

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
