# Persistent Agent

Persistent Agent is a task-pool driven agent system that can continuously pick up user-defined work, execute it, ask for help when blocked, and preserve useful experience as long-term memory.

The first milestone focuses on reliable serial execution. The architecture should keep a clean path toward future parallel execution.

## Documents

- [Technical Selection](docs/TECH_SELECTION.md)

## Current Implementation

The repository now contains the first executable skeleton:

- Rust workspace backend under `crates/` and `apps/server`.
- SQLite migrations and repositories for task lifecycle state.
- Main-agent task-management service for creating, updating, pausing, resuming, cancelling, reordering, and summarizing tasks.
- Main-agent conversation commands for pause, resume, cancel, reprioritize, reorder, create, and summarize.
- Web task pool controls for priority and queue-position changes.
- Serial scheduler loop with a shared execution lock, manual tick endpoint, and stub worker that exercises task claiming, attempts, completion, blockers, and event emission.
- Task execution history API and UI for attempts and auditable task actions.
- Skill management with automatic matching, explicit task selection, and active-skill context for workers.
- Axum API with REST endpoints and an SSE event stream.
- Vite React Web UI under `apps/web`.

Run the backend:

```powershell
cargo run -p persistent-agent-server
```

Enable the DeepSeek LLM worker by setting `DEEPSEEK_API_KEY` in your local environment. Do not commit real API keys.

The server scans the task pool every 30 seconds by default. Set `SCHEDULER_INTERVAL_SECONDS=0` to disable the background scheduler loop, or set another positive value to adjust the polling interval.

Run the Web UI:

```powershell
cd apps/web
npm install
npm run dev
```

The Web UI defaults to <http://127.0.0.1:5173> and proxies API calls to <http://127.0.0.1:8080>.

## Product Goals

1. Let users create, prioritize, inspect, and discuss tasks through both conversation and structured UI controls.
2. Let a main agent manage the task pool conversationally, periodically scan for runnable work, and coordinate execution.
3. Let worker agents execute concrete tasks and report results, blockers, and memory candidates.
4. Let blocked tasks request user input through a conversation thread, then resume when enough context is available.
5. Support one-off tasks and recurring tasks.
6. Support user-defined skills that can be matched automatically or specified explicitly.
7. Maintain long-term memory for preferences, pitfalls, useful decisions, and project-specific conventions.
8. Provide a clean Web UI for task creation, conversations, queue state, execution state, and history.

## Core Concepts

### Task

A task is the durable unit of work in the system.

Recommended fields:

- `id`: stable task identifier.
- `title`: short task name.
- `description`: full user request or recurring instruction.
- `type`: `one_off` or `recurring`.
- `status`: current lifecycle status.
- `priority`: ordering weight.
- `queue_position`: explicit serial queue order.
- `created_by`: user or system.
- `assigned_agent`: current executor, if any.
- `requested_skills`: skills explicitly selected by the user.
- `matched_skills`: skills matched by the system.
- `schedule`: optional recurring or polling configuration.
- `conversation_id`: linked user-agent discussion thread.
- `attempt_count`: execution attempts.
- `last_run_at`: last execution timestamp.
- `next_run_at`: next eligible execution timestamp.
- `blocked_reason`: reason the task needs user input.
- `result_summary`: latest outcome summary.
- `memory_candidates`: summaries proposed by worker agents.

### Task Creation Channels

Tasks can enter the system through three channels:

1. Conversational creation through the main agent.
2. Structured creation through Web UI forms and queue controls.
3. System-created follow-up tasks from recurring jobs, worker summaries, or main-agent planning.

The conversational path should be a first-class product surface. Users can ask the main agent to create tasks, split vague goals into multiple tasks, reprioritize work, pause recurring tasks, resume blocked tasks, or summarize the queue.

Structured UI controls remain important for exact edits, batch operations, and reviewable state changes.

### Task Types

`one_off` tasks are completed once execution succeeds.

`recurring` tasks are re-enqueued after each run. A recurring task should not be duplicated blindly; instead, the scheduler should create a new runnable queue entry or update `next_run_at` while preserving one canonical task definition.

Example recurring task:

> Check issues in a GitHub repository. If new issues exist, investigate and resolve them.

### Task Status

Suggested lifecycle:

- `draft`: created but not ready for execution.
- `queued`: ready to be picked up.
- `running`: currently being executed.
- `waiting_for_user`: blocked and needs user input.
- `waiting_for_schedule`: recurring task waiting for the next eligible run.
- `completed`: finished successfully.
- `failed`: failed permanently or exceeded policy.
- `cancelled`: stopped by the user.

## Execution Model

### MVP: Serial Execution

The main agent owns a single execution loop:

1. Wake on a timer or explicit user action.
2. Find the first runnable task by queue order and priority.
3. Resolve applicable skills.
4. Start one worker agent for that task.
5. Track logs, state, artifacts, and conversation messages.
6. On success:
   - Mark one-off tasks as `completed`.
   - Move recurring tasks to `waiting_for_schedule` or the tail of the queue.
7. On blocker:
   - Mark task as `waiting_for_user`.
   - Post a concise request in the linked conversation.
8. Review worker memory candidates and optionally promote useful ones into long-term memory.

### Future: Parallel Execution

Prepare for parallel execution by separating:

- queue selection from task execution,
- task locks from task records,
- agent orchestration from worker implementation,
- worker capacity from scheduler policy.

Future scheduler policies can include:

- max global concurrency,
- per-project concurrency,
- per-skill concurrency,
- exclusive locks for repositories, files, or external services,
- dependency-aware execution.

## Agent Roles

### Main Agent

The main agent is the conversational task manager, scheduler, and worker coordinator. It should:

- discuss goals with the user and turn them into task changes,
- create, update, pause, resume, cancel, reorder, and reprioritize tasks through explicit task-management tools,
- summarize the task pool and explain current execution state,
- perform lightweight local inspection when useful for task planning,
- scan the task pool,
- select runnable tasks,
- resolve skills,
- spawn worker agents,
- monitor execution,
- handle task state transitions,
- ask the user for help when needed,
- decide whether worker memory candidates should be stored.

The main agent should not mutate task state through hidden database writes. It should use auditable tools with clear arguments and recorded outcomes.

For substantial task execution, code changes, long-running operations, or risky local actions, the main agent should delegate to a worker agent. For lightweight planning and inspection, it may use local tools directly.

### Worker Agent

A worker agent executes one task attempt. It should:

- understand the task and available context,
- use matched or requested skills,
- perform the work,
- produce artifacts or code changes,
- summarize outcome and verification,
- explain blockers clearly,
- propose memory candidates.

Workers should be replaceable so future implementations can support local tools, remote sandboxes, browser agents, code agents, or specialized domain agents.

## Skill System

Skills are user-defined capability packages. A skill can include:

- name and description,
- trigger rules,
- required tools,
- prompts or instructions,
- scripts or templates,
- examples,
- safety constraints.

Skill activation should support two paths:

1. Explicit activation: the user attaches skills when creating a task.
2. Automatic activation: the system matches skills against task title, description, metadata, and historical usage.

When both exist, explicit user selection should take precedence.

## Long-Term Memory

Memory should be curated, not a raw log dump.

Good memory candidates include:

- stable user preferences,
- project conventions,
- recurring external constraints,
- failed approaches to avoid,
- useful commands or workflows,
- repository-specific setup details.

Suggested flow:

1. Worker submits `memory_candidates` after execution.
2. Main agent reviews candidates.
3. Approved memories are stored with scope, source task, timestamp, and confidence.
4. Future tasks retrieve relevant memories by project, skill, repository, or semantic match.

Memory scopes:

- global,
- user,
- project,
- repository,
- skill,
- task family.

## Web UI

The Web UI should prioritize operational clarity over a marketing-style interface.

Core views:

- task pool: create tasks, reorder queue, filter status, inspect priority;
- task detail: description, metadata, run history, artifacts, memory candidates;
- main conversation: talk to the main agent to create, update, reorder, explain, or discuss tasks;
- task conversation: user-agent discussion for blocked or active tasks;
- execution monitor: current running task, logs, state transitions;
- skills: manage user-defined skills and see activation rules;
- memory: inspect and edit approved long-term memories.

MVP UI actions:

- create or change tasks by chatting with the main agent,
- create one-off task,
- create recurring task,
- edit task priority/order,
- pause/resume/cancel task,
- reply to blocked task,
- manually trigger scheduler scan,
- view execution history.

## Suggested MVP Scope

Build the first version in this order:

1. Data model and task lifecycle.
2. Task-management tools for the main agent.
3. Main-agent conversation flow for creating and changing tasks.
4. Serial scheduler loop.
5. Main-agent to worker-agent execution contract.
6. Basic one-off task execution.
7. Blocked task conversation flow.
8. Recurring task requeue behavior.
9. Skill metadata and explicit skill selection.
10. Basic automatic skill matching.
11. Long-term memory candidate review.
12. Web UI for task pool, task detail, main conversation, task conversation, and execution status.

## Design Principles

- Make every state transition explicit and auditable.
- Keep the scheduler deterministic before adding parallelism.
- Treat recurring tasks as durable definitions with repeated runs.
- Separate human conversation from execution logs.
- Store summaries and decisions, not unbounded raw context.
- Prefer small, inspectable agent contracts over hidden orchestration.
- Design locks and capacity now, even if MVP uses only one worker.
- Let the main agent manage task state through explicit tools, not implicit database mutation.
