# Technical Selection

This document records the first technology choices for Persistent Agent.

The goal is to build a durable product around an existing agent framework instead of reimplementing the low-level agent loop.

## Recommended Stack

### Agent And Runtime Foundation

Use the `oh-my-harness` framework family as the agent foundation:

- `llm-api-adapter`: provider adapter layer.
- `llm-harness-core`: core agent loop, `Agent`, `AgentHarness`, tools, hooks, sessions, compaction, skills, and events.
- `llm-harness-runtime`: platform services such as task lifecycle, sandbox, tool registry, MCP, resource injection, sub-agents, tracing, audit, auth, budget, and approval.

Product code should prefer `AgentHarness` as the product-facing agent entry point.

Use runtime services where they already match product concerns:

- `ToolRegistry` and `ToolSource` for tool discovery and tool subsets.
- `TaskRunner` as the execution bridge when its lifecycle fits our task attempts.
- `SubAgentSpawner` for worker-agent creation.
- `HarnessHooks` for approval, budget, memory injection, tracing, audit, and phase control.
- `HumanApprovalWrapper` for risky tools and future user approval flows.
- `AuditSink` and `TraceExporter` for durable execution records and observability.

Do not write a separate provider client, tool protocol, hook system, or low-level agent loop in product code unless the framework is missing a capability and the gap is explicitly recorded.

### Product Backend

Use Rust for the backend.

Recommended crates:

- `tokio` for async runtime.
- `axum` for HTTP API, SSE, WebSocket-ready routing, and middleware.
- `sqlx` for database access.
- `serde` / `serde_json` for API and persistence serialization.
- `utoipa` later if OpenAPI generation becomes useful.
- `tracing` for structured backend logs.

Why Rust:

- The agent framework is Rust-native.
- Tools, hooks, runtime services, and scheduler logic can share types directly.
- Long-running task execution benefits from explicit cancellation, structured concurrency, and predictable resource ownership.

Why `axum`:

- It fits the `tokio` ecosystem naturally.
- It is small enough for a product API and streaming events.
- It does not force a heavy application framework over the agent runtime.

### Persistence

Use SQL as the durable source of truth.

MVP:

- SQLite through `sqlx`.
- Store tasks, queue entries, task attempts, conversations, messages, skills, memories, locks, audit events, and scheduler state.

Production path:

- PostgreSQL through `sqlx`.
- Add `pgvector` when semantic memory retrieval becomes important at scale.

This keeps local development simple while preserving a clean path to multi-user or hosted deployment.

### Scheduler

Build a product-level scheduler around a database-backed queue.

MVP behavior:

- serial worker capacity: `1`;
- deterministic queue ordering by `priority`, `queue_position`, and creation time;
- dependency-aware claiming, so a queued task is skipped until its prerequisites are satisfied;
- resource-lock-aware claiming, so tasks can declare exclusive repositories, files, or services before future parallel workers are enabled;
- task lease before execution;
- heartbeat while running;
- explicit state transitions;
- recurring tasks move to `waiting_for_schedule`, then re-enter the queue when eligible.

Future parallelism:

- increase global worker capacity;
- add per-skill and capacity-based resource policies;
- broaden repository/workspace locks beyond exclusive mode when needed;
- add scheduler policies without changing task records.

The scheduler should not be hidden inside the UI or a single process-local memory queue. The database is the coordination boundary.

### API Shape

Use REST plus server-sent events for the MVP.

REST:

- task CRUD;
- queue reorder;
- task dependency management;
- task resource lock management;
- task note creation and listing;
- pause, resume, cancel;
- conversation message creation;
- main-agent conversation message creation;
- task-management action records;
- skill CRUD;
- memory review and editing;
- manual scheduler trigger.
- scheduler state summary.

SSE:

- task status updates;
- current run events;
- tool-call events;
- assistant text deltas;
- main-agent task-management decisions;
- scheduler heartbeat.

Prefer SSE for the first version because most real-time data is server-to-client. Add WebSocket later only when the UI needs richer bidirectional live collaboration.

### Web UI

Use a Vite React SPA.

Recommended frontend stack:

- React + TypeScript.
- Vite for development and build.
- TanStack Router for typed client routes and URL search state.
- TanStack Query for server state, cache invalidation, polling fallback, and mutations.
- shadcn/ui as the component starting point.
- Radix Primitives for accessible low-level UI behavior.
- Tailwind CSS v4 for styling.
- lucide-react for icons.
- Zustand only for small local UI state that is not server state.

Why this over Next.js:

- Persistent Agent is an authenticated operational dashboard, not a public content site.
- SEO and SSR are not central requirements.
- A Rust backend already owns APIs, auth, task execution, streaming, and persistence.
- Avoiding React Server Components and Server Actions keeps the frontend simpler for agent-generated changes.

Why this over TanStack Start or React Router framework mode:

- They are good options for full-stack React apps, but this product already has a Rust backend.
- The MVP benefits more from a focused SPA than from adding a second server framework.

Why shadcn/ui:

- It gives editable component code instead of a locked component package.
- It pairs well with Radix and Tailwind.
- It is a good fit for dense operational UI: tables, dialogs, forms, tabs, command menus, sheets, and status badges.

### Desktop App Path

Do not start with a desktop wrapper.

Keep the app as a local or hosted web service first. If a desktop distribution becomes important, wrap the existing frontend and Rust backend with Tauri later.

Tauri is a good future fit because it supports web frontends with Rust application logic, but adding it at the MVP stage would increase packaging and update complexity before the product workflow is proven.

## Core Domain Modules

Suggested Rust workspace layout:

```text
crates/
  persistent-agent-domain/
  persistent-agent-db/
  persistent-agent-scheduler/
  persistent-agent-agent/
  persistent-agent-api/
  persistent-agent-memory/
  persistent-agent-skill/
apps/
  server/
  web/
```

Module responsibilities:

- `domain`: task, queue, attempt, conversation, skill, memory, and event types.
- `db`: migrations, repositories, transaction helpers.
- `scheduler`: queue scan, leases, recurring task eligibility, serial/parallel policy.
- `agent`: main-agent conversation, task-management tools, worker-agent adapter, harness wiring.
- `api`: HTTP routes, SSE stream, DTOs.
- `memory`: memory candidate review, storage, retrieval, injection policy.
- `skill`: skill metadata, explicit activation, automatic matching, framework resource loading.
- `server`: executable composition.
- `web`: React UI.

## Agent Design Decisions

### Main Agent

The main agent is the conversational task manager, scheduler, and worker coordinator. It is product orchestration, not the low-level LLM loop.

It should:

- talk with the user about goals, tasks, queue state, blockers, and priorities;
- show recent global audit actions for non-task-specific tool calls;
- create, update, pause, resume, cancel, reorder, and reprioritize tasks through explicit tools;
- split vague user requests into concrete task proposals;
- summarize the task pool and explain why work is or is not running;
- perform lightweight local inspection when that helps task planning;
- scan runnable tasks through scheduled ticks, manual API calls, or main-agent conversation requests;
- claim a task lease;
- resolve requested and matched skills;
- construct or request a worker agent;
- subscribe to worker events;
- persist task events and summaries;
- mark the task outcome;
- review memory candidates.

The main agent may inspect local state, such as repository files or git status, when planning or clarifying tasks. The first implemented local operations are read-only: workspace status inspection reports the process working directory and `git status --short --branch`, while workspace file inspection previews a relative file inside the current workspace with an output cap. They record `inspect_workspace_status` and `inspect_workspace_file` actions. Substantial execution, code changes, long-running operations, and risky local actions should be delegated to worker agents.

Task mutations should flow through task-management tools. The main agent should not update database records through hidden side effects.

### Worker Agent

A worker agent should be one task attempt.

It should:

- run through `AgentHarness`;
- receive task context, selected skills, memories, and allowed tools;
- receive active product skill metadata, tool subsets, and resource paths;
- derive the allowed-tool list from active skill tool subsets and emit it in attempt events;
- emit framework events into the product event stream;
- produce a final task result;
- report blockers in a structured form;
- propose memory candidates.

### Tools

Implement product/domain tools using the framework `Tool` contract.

Rules:

- stable tool names;
- JSON Schema parameters;
- use `ExecutionEnv` for filesystem and shell work;
- put UI/audit metadata in `ToolResult.details`;
- use sequential execution mode for mutating or unsafe tools;
- keep provider/model selection out of tools.

Main-agent task-management tools should be separate from worker execution tools.

Recommended main-agent tools:

- `create_task`;
- `update_task`;
- `reprioritize_task`;
- `reorder_queue`;
- `pause_task`;
- `resume_task`;
- `cancel_task`;
- `convert_task_type`;
- `add_requested_skill`;
- `remove_requested_skill`;
- `add_task_dependency`;
- `remove_task_dependency`;
- `add_resource_lock`;
- `remove_resource_lock`;
- `add_task_note`;
- `list_tasks`;
- `summarize_task_pool`;
- `inspect_workspace_status`;
- `inspect_workspace_file`;
- `request_user_clarification`.

These tools should write action records so the UI can explain when a task changed because of user instruction, scheduler policy, or main-agent planning.

### Skills

Use two layers:

1. Framework skill resources loaded by `AgentHarness`.
2. Product skill records stored in the database for UI management, matching, and user selection.

Product skill records can point to skill directories, prompt resources, tool subsets, or runtime prompt sources.

Activation order:

1. user-requested skills;
2. task-type default skills, expressed initially as deterministic rules such as `type:recurring` or `task_type:one_off`;
3. automatic matcher results;
4. main-agent adjustments.

### Hooks

Use hooks for cross-cutting behavior:

- `transform_context`: inject relevant memories and task context;
- `before_tool_call`: approval, policy, active-tool checks;
- `after_tool_call`: audit and result normalization;
- `after_provider_response`: cost and usage accounting;
- `before_compact`: product-specific compaction summaries;
- `prepare_next_turn` / `should_stop`: phase control and replan behavior.

## Web UI Information Architecture

MVP screens:

- task pool: queue, status filters, create task, reorder, pause/resume;
- task detail: metadata, attempt history, logs, artifacts, linked conversation;
- main conversation: user talks with the main agent to create, update, prioritize, pause, resume, or discuss tasks;
- main-agent audit: recent global action records for non-task-specific tool calls;
- task conversation: user replies for blocked tasks and active task discussion;
- execution monitor: current run, next queued task, events, tool calls, output stream;
- skills: list, create/edit, activation rules, tool subsets;
- memory: pending candidates, approved memories, edit/delete.

Design style:

- dense but calm operational UI;
- left navigation plus main work area;
- make the main conversation and task pool feel like two views of the same system, not separate products;
- queue and task status should be visible without drilling into every task;
- avoid marketing-style hero sections;
- prefer tables, split panes, tabs, drawers, command menus, and compact cards for repeated items only.

## Open Questions

- Should the MVP target local single-user only, or hosted multi-user from day one?
- Should recurring tasks create child run records only, or visible child tasks for each execution?
- Should skill matching start as keyword/rules only, or include embedding-based semantic matching immediately?
- Should memory approval be manual by default, automatic by confidence, or both?
- Which LLM provider should be the default development provider?
- How much sandboxing is required for the first coding-task tools?

## Initial Recommendation

Start with:

- Rust workspace backend.
- `AgentHarness` + runtime services for agent execution.
- SQLite + `sqlx` for MVP persistence.
- DB-backed serial scheduler with lease records.
- REST + SSE API.
- Vite React SPA with TanStack Router, TanStack Query, shadcn/ui, Radix, Tailwind v4, and lucide-react.

This is the smallest stack that respects the existing agent framework, gives the UI enough real-time feedback, keeps the scheduler durable, and leaves a clear path to parallel workers and PostgreSQL-backed hosted deployment.

## Research Notes

- Next.js App Router is powerful and supports Server Components, but it is more useful when the React framework owns full-stack routing and data access.
- React Router framework mode and TanStack Start are strong full-stack React options, but they overlap with the Rust backend role in this project.
- TanStack Query is a strong fit for server-state caching, mutation flows, invalidation, and polling fallback.
- Tailwind CSS v4 and shadcn/ui are current enough to use as the UI foundation, with Radix covering accessible primitives.
- Tauri remains a good packaging option if a desktop app becomes a product requirement.

Primary references:

- LLM Harness Skills: <https://github.com/oh-my-harness/llm-harness-skills>
- Next.js App Router docs: <https://nextjs.org/docs/app>
- React Router framework docs: <https://reactrouter.com/start/framework/routing>
- TanStack Router docs: <https://tanstack.com/router/latest/docs/framework/react/overview>
- TanStack Query docs: <https://tanstack.com/query/latest/docs/framework/react/overview>
- shadcn/ui docs: <https://ui.shadcn.com/docs>
- Radix Primitives docs: <https://www.radix-ui.com/primitives/docs/overview/introduction>
- Tailwind CSS docs: <https://tailwindcss.com/docs>
- axum docs: <https://docs.rs/axum/latest/axum/>
- SQLx docs: <https://docs.rs/sqlx/latest/sqlx/>
- Tauri docs: <https://tauri.app/start/>
