use async_trait::async_trait;
use llm_adapter::deepseek;
use llm_harness_agent::prelude::{
    AgentHarness, AgentHarnessEvent, AgentHarnessOptions, ContentBlock, Tool, ToolContext,
    ToolError, ToolExecutionMode, ToolResult,
};
use llm_harness_loop::LlmClient;
use llm_harness_runtime::{
    InMemoryToolRegistry, ResourceLimits, Sandbox, SandboxConfig, ToolRegistry,
};
use llm_harness_runtime_sandbox_os::OsEnvSandbox;
use persistent_agent_db::Db;
use persistent_agent_domain::{
    ConversationMessage, CreateMemory, CreateTask, Memory, MemoryStatus, Skill, Task, TaskNote,
    TaskStatus, TaskType,
};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::{collections::BTreeSet, sync::Arc};
use tokio::sync::Mutex;
use tokio::time::{Instant, sleep};

#[derive(Clone)]
pub struct Scheduler<W> {
    db: Db,
    worker: W,
    lease_owner: String,
    policy: SchedulerPolicy,
    serial_lock: Arc<Mutex<()>>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct SchedulerPolicy {
    pub worker_capacity: usize,
    pub lease_seconds: i64,
    pub max_attempts: i64,
    pub memory_auto_approve_confidence: Option<f64>,
}

impl SchedulerPolicy {
    pub fn serial() -> Self {
        Self {
            worker_capacity: 1,
            lease_seconds: 300,
            max_attempts: 1,
            memory_auto_approve_confidence: None,
        }
    }

    pub fn new(worker_capacity: usize, lease_seconds: i64) -> Self {
        Self {
            worker_capacity: worker_capacity.max(1),
            lease_seconds: lease_seconds.max(1),
            max_attempts: 1,
            memory_auto_approve_confidence: None,
        }
    }

    pub fn with_max_attempts(mut self, max_attempts: i64) -> Self {
        self.max_attempts = max_attempts.max(1);
        self
    }

    pub fn with_memory_auto_approve_confidence(mut self, confidence: Option<f64>) -> Self {
        self.memory_auto_approve_confidence =
            confidence.map(|confidence| confidence.clamp(0.0, 1.0));
        self
    }
}

impl Default for SchedulerPolicy {
    fn default() -> Self {
        Self::serial()
    }
}

impl<W> Scheduler<W>
where
    W: TaskWorker,
{
    pub fn new(db: Db, worker: W) -> Self {
        Self {
            db,
            worker,
            lease_owner: "persistent-agent-scheduler".to_owned(),
            policy: SchedulerPolicy::serial(),
            serial_lock: Arc::new(Mutex::new(())),
        }
    }

    pub fn with_policy(db: Db, worker: W, policy: SchedulerPolicy) -> Self {
        Self {
            db,
            worker,
            lease_owner: "persistent-agent-scheduler".to_owned(),
            policy,
            serial_lock: Arc::new(Mutex::new(())),
        }
    }

    pub fn policy(&self) -> SchedulerPolicy {
        self.policy
    }

    pub async fn tick(&self) -> anyhow::Result<SchedulerTick> {
        let _guard = self.serial_lock.lock().await;
        self.tick_inner().await
    }

    async fn tick_inner(&self) -> anyhow::Result<SchedulerTick> {
        let recovered_tasks = self.db.recover_expired_running_tasks("scheduler").await?;
        let requeued_tasks = self.db.requeue_due_recurring_tasks("scheduler").await?;
        let Some(task) = self
            .db
            .claim_next_runnable(&self.lease_owner, self.policy.lease_seconds)
            .await?
        else {
            return Ok(SchedulerTick {
                recovered_tasks,
                requeued_tasks,
                claimed_task: None,
                outcome: SchedulerOutcome::Idle,
            });
        };

        tracing::info!(task_id = %task.id, title = %task.title, "claimed task");
        let attempt = self
            .db
            .create_attempt(task.id, TaskStatus::Running, Some("worker started"))
            .await?;

        let skills = self
            .db
            .list_skills_by_names(&active_skill_names(&task))
            .await?;
        let allowed_tools = allowed_tools_for_skills(&skills);
        let context = WorkerContext {
            memories: select_relevant_memories(&task, self.db.list_approved_memories(20).await?, 5),
            skills,
            allowed_tools,
            notes: self.db.list_task_notes(task.id).await?,
            conversation_messages: self.db.list_task_conversation_messages(task.id, 20).await?,
        };
        self.db
            .record_attempt_event(
                attempt.id,
                task.id,
                "worker_context_prepared",
                "Prepared worker context.",
                json!({
                    "memory_count": context.memories.len(),
                    "skill_count": context.skills.len(),
                    "allowed_tools": &context.allowed_tools,
                    "note_count": context.notes.len(),
                    "conversation_message_count": context.conversation_messages.len(),
                    "requested_skills": &task.requested_skills,
                    "matched_skills": &task.matched_skills,
                }),
            )
            .await?;
        let result = match self
            .execute_worker_with_heartbeat(task.clone(), context, attempt.id)
            .await
        {
            Ok(result) => result,
            Err(error) => {
                let error = error.to_string();
                self.db
                    .record_attempt_event(
                        attempt.id,
                        task.id,
                        "worker_failed",
                        "Worker execution failed.",
                        json!({ "error": error }),
                    )
                    .await?;
                if let Some(outcome) = self
                    .supersede_if_task_no_longer_running(task.id, attempt.id, "failed")
                    .await?
                {
                    return Ok(SchedulerTick {
                        recovered_tasks,
                        requeued_tasks,
                        claimed_task: Some(task),
                        outcome,
                    });
                }
                self.db
                    .create_attempt(task.id, TaskStatus::Failed, Some(&error))
                    .await?;
                if task.attempt_count < self.policy.max_attempts {
                    let next_attempt = task.attempt_count + 1;
                    self.db
                        .requeue_task_after_failure(task.id, &error, "worker")
                        .await?;
                    self.db
                        .record_attempt_event(
                            attempt.id,
                            task.id,
                            "worker_retry_scheduled",
                            "Worker failed; task was requeued for another attempt.",
                            json!({
                                "error": error,
                                "next_attempt": next_attempt,
                                "max_attempts": self.policy.max_attempts,
                            }),
                        )
                        .await?;

                    return Ok(SchedulerTick {
                        recovered_tasks,
                        requeued_tasks,
                        claimed_task: Some(task),
                        outcome: SchedulerOutcome::RetryScheduled {
                            error,
                            next_attempt,
                            max_attempts: self.policy.max_attempts,
                        },
                    });
                }
                self.db.fail_task(task.id, &error, "worker").await?;

                return Ok(SchedulerTick {
                    recovered_tasks,
                    requeued_tasks,
                    claimed_task: Some(task),
                    outcome: SchedulerOutcome::Failed { error },
                });
            }
        };
        match result {
            WorkerResult::Completed {
                summary,
                memory_candidates,
                artifacts,
                follow_up_tasks,
            } => {
                self.db
                    .record_attempt_event(
                        attempt.id,
                        task.id,
                        "worker_completed",
                        "Worker completed the task.",
                        json!({
                            "summary": summary,
                            "memory_candidate_count": memory_candidates.len(),
                            "artifact_count": artifacts.len(),
                            "follow_up_task_count": follow_up_tasks.len(),
                        }),
                    )
                    .await?;
                if let Some(outcome) = self
                    .supersede_if_task_no_longer_running(task.id, attempt.id, "completed")
                    .await?
                {
                    return Ok(SchedulerTick {
                        recovered_tasks,
                        requeued_tasks,
                        claimed_task: Some(task),
                        outcome,
                    });
                }
                self.db
                    .create_attempt(task.id, TaskStatus::Completed, Some(&summary))
                    .await?;
                for artifact in artifacts {
                    self.db
                        .record_task_artifact(
                            task.id,
                            Some(attempt.id),
                            &artifact.name,
                            &artifact.artifact_type,
                            &artifact.uri,
                            artifact.summary.as_deref(),
                        )
                        .await?;
                }
                let mut created_memory_candidates = Vec::new();
                for candidate in memory_candidate_contents(&task, &summary, &memory_candidates) {
                    let confidence = 0.6;
                    let status = match self.policy.memory_auto_approve_confidence {
                        Some(threshold) if confidence >= threshold => MemoryStatus::Approved,
                        _ => MemoryStatus::Pending,
                    };
                    let memory = self
                        .db
                        .create_memory(
                            CreateMemory {
                                scope: "task".to_owned(),
                                content: candidate,
                                source_task_id: Some(task.id),
                                status,
                                confidence,
                            },
                            "worker",
                        )
                        .await?;
                    created_memory_candidates.push(memory);
                }
                let created_follow_up_tasks = self
                    .create_follow_up_tasks(task.id, attempt.id, &follow_up_tasks)
                    .await?;
                self.db.complete_task(task.id, &summary, "worker").await?;
                Ok(SchedulerTick {
                    recovered_tasks,
                    requeued_tasks,
                    claimed_task: Some(task),
                    outcome: SchedulerOutcome::Completed {
                        summary,
                        follow_up_tasks: created_follow_up_tasks,
                        memory_candidates: created_memory_candidates,
                    },
                })
            }
            WorkerResult::Blocked { reason } => {
                self.db
                    .record_attempt_event(
                        attempt.id,
                        task.id,
                        "worker_blocked",
                        "Worker needs user input.",
                        json!({ "reason": reason }),
                    )
                    .await?;
                if let Some(outcome) = self
                    .supersede_if_task_no_longer_running(task.id, attempt.id, "blocked")
                    .await?
                {
                    return Ok(SchedulerTick {
                        recovered_tasks,
                        requeued_tasks,
                        claimed_task: Some(task),
                        outcome,
                    });
                }
                self.db
                    .create_attempt(task.id, TaskStatus::WaitingForUser, Some(&reason))
                    .await?;
                if let Some(conversation_id) = task.conversation_id {
                    self.db
                        .add_conversation_message(
                            conversation_id,
                            Some(task.id),
                            "assistant",
                            &reason,
                        )
                        .await?;
                }
                self.db
                    .set_task_status(task.id, TaskStatus::WaitingForUser, "worker", Some(&reason))
                    .await?;
                Ok(SchedulerTick {
                    recovered_tasks,
                    requeued_tasks,
                    claimed_task: Some(task),
                    outcome: SchedulerOutcome::Blocked { reason },
                })
            }
        }
    }

    async fn supersede_if_task_no_longer_running(
        &self,
        task_id: persistent_agent_domain::TaskId,
        attempt_id: persistent_agent_domain::TaskAttemptId,
        worker_result: &str,
    ) -> anyhow::Result<Option<SchedulerOutcome>> {
        let current = self.db.get_task(task_id).await?;
        if current.status == TaskStatus::Running {
            return Ok(None);
        }

        let reason = format!(
            "Worker result '{worker_result}' was ignored because task status is now '{}'.",
            current.status
        );
        self.db
            .record_attempt_event(
                attempt_id,
                task_id,
                "worker_outcome_superseded",
                "Worker result ignored because task state changed during execution.",
                json!({
                    "worker_result": worker_result,
                    "current_status": current.status,
                }),
            )
            .await?;
        self.db
            .create_attempt(task_id, current.status, Some(&reason))
            .await?;

        Ok(Some(SchedulerOutcome::Superseded {
            status: current.status,
            reason,
        }))
    }

    async fn create_follow_up_tasks(
        &self,
        source_task_id: persistent_agent_domain::TaskId,
        attempt_id: persistent_agent_domain::TaskAttemptId,
        follow_up_tasks: &[WorkerFollowUpTask],
    ) -> anyhow::Result<Vec<Task>> {
        let mut created_tasks = Vec::new();
        for follow_up in follow_up_tasks {
            let created = self
                .db
                .create_task(
                    CreateTask {
                        title: follow_up.title.clone(),
                        description: follow_up.description.clone(),
                        task_type: TaskType::OneOff,
                        priority: follow_up.priority,
                        requested_skills: follow_up.requested_skills.clone(),
                        schedule: None,
                        created_by: "worker".to_owned(),
                    },
                    "worker",
                )
                .await?;
            self.db
                .record_action(
                    Some(source_task_id),
                    "worker",
                    "create_follow_up_task",
                    json!({
                        "follow_up_task_id": created.id,
                        "title": created.title,
                    }),
                )
                .await?;
            created_tasks.push(created);
        }

        if !created_tasks.is_empty() {
            self.db
                .record_attempt_event(
                    attempt_id,
                    source_task_id,
                    "follow_up_tasks_created",
                    "Created follow-up tasks requested by worker result.",
                    json!({
                        "count": created_tasks.len(),
                        "task_ids": created_tasks.iter().map(|task| task.id).collect::<Vec<_>>(),
                    }),
                )
                .await?;
        }

        Ok(created_tasks)
    }

    async fn execute_worker_with_heartbeat(
        &self,
        task: Task,
        context: WorkerContext,
        attempt_id: persistent_agent_domain::TaskAttemptId,
    ) -> anyhow::Result<WorkerResult> {
        let task_id = task.id;
        let worker = self.worker.clone();
        let mut worker_task = tokio::spawn(async move { worker.execute(task, context).await });

        let heartbeat_every = heartbeat_interval(self.policy.lease_seconds);
        let heartbeat_delay = sleep(heartbeat_every);
        tokio::pin!(heartbeat_delay);

        loop {
            tokio::select! {
                result = &mut worker_task => {
                    return result?;
                }
                _ = &mut heartbeat_delay => {
                    match self
                        .db
                        .heartbeat_task_lease(task_id, &self.lease_owner, self.policy.lease_seconds)
                        .await
                    {
                        Ok(Some(lease_expires_at)) => {
                            if let Err(error) = self.db
                                .record_attempt_event(
                                    attempt_id,
                                    task_id,
                                    "worker_heartbeat",
                                    "Refreshed running task lease.",
                                    json!({ "lease_owner": self.lease_owner, "lease_expires_at": lease_expires_at }),
                                )
                                .await
                            {
                                tracing::warn!(task_id = %task_id, %error, "failed to record worker heartbeat event");
                            }
                        }
                        Ok(None) => {
                            if let Err(error) = self.db
                                .record_attempt_event(
                                    attempt_id,
                                    task_id,
                                    "worker_lease_lost",
                                    "Stopped worker because task lease is no longer active.",
                                    json!({ "lease_owner": self.lease_owner }),
                                )
                                .await
                            {
                                tracing::warn!(task_id = %task_id, %error, "failed to record worker lease loss event");
                            }
                            worker_task.abort();
                            anyhow::bail!("task lease lost before worker completed");
                        }
                        Err(error) => {
                            tracing::warn!(task_id = %task_id, %error, "failed to refresh running task lease");
                        }
                    }
                    heartbeat_delay
                        .as_mut()
                        .reset(Instant::now() + heartbeat_every);
                }
            }
        }
    }
}

#[async_trait]
pub trait TaskWorker: Clone + Send + Sync + 'static {
    async fn execute(&self, task: Task, context: WorkerContext) -> anyhow::Result<WorkerResult>;
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct WorkerContext {
    pub memories: Vec<Memory>,
    pub skills: Vec<Skill>,
    pub allowed_tools: Vec<String>,
    pub notes: Vec<TaskNote>,
    pub conversation_messages: Vec<ConversationMessage>,
}

fn heartbeat_interval(lease_seconds: i64) -> std::time::Duration {
    let seconds = (lease_seconds.max(1) / 3).clamp(1, 60) as u64;
    std::time::Duration::from_secs(seconds)
}

#[derive(Debug, Clone)]
pub struct StubWorker;

#[async_trait]
impl TaskWorker for StubWorker {
    async fn execute(&self, task: Task, context: WorkerContext) -> anyhow::Result<WorkerResult> {
        let marker_text = format!("{} {}", task.title, task.description).to_lowercase();
        if marker_text.contains("blocked")
            || marker_text.contains("needs_user")
            || marker_text.contains("need_user")
        {
            if context
                .conversation_messages
                .iter()
                .any(|message| message.role == "user")
            {
                return Ok(WorkerResult::Completed {
                    summary: format!(
                        "Stub worker resumed task '{}' using the latest user conversation context.",
                        task.title
                    ),
                    memory_candidates: Vec::new(),
                    artifacts: Vec::new(),
                    follow_up_tasks: Vec::new(),
                });
            }

            return Ok(WorkerResult::Blocked {
                reason: format!(
                    "I need more user input before continuing task '{}'. Please add the missing context in this task conversation.",
                    task.title
                ),
            });
        }

        Ok(WorkerResult::Completed {
            summary: format!(
                "Stub worker accepted task '{}' and completed the lifecycle placeholder.",
                task.title
            ),
            memory_candidates: Vec::new(),
            artifacts: Vec::new(),
            follow_up_tasks: Vec::new(),
        })
    }
}

#[derive(Debug, Clone)]
pub enum WorkerBackend {
    Stub(StubWorker),
    Harness(OhMyHarnessWorker),
}

#[async_trait]
impl TaskWorker for WorkerBackend {
    async fn execute(&self, task: Task, context: WorkerContext) -> anyhow::Result<WorkerResult> {
        match self {
            Self::Stub(worker) => worker.execute(task, context).await,
            Self::Harness(worker) => worker.execute(task, context).await,
        }
    }
}

#[derive(Debug, Clone)]
pub struct OhMyHarnessWorker {
    config: LlmWorkerConfig,
}

pub type LlmWorker = OhMyHarnessWorker;

impl OhMyHarnessWorker {
    pub fn new(config: LlmWorkerConfig) -> Self {
        Self { config }
    }
}

#[derive(Debug, Clone)]
pub struct LlmWorkerConfig {
    pub api_key: String,
    pub model: String,
    pub base_url: String,
    pub timeout_ms: u64,
}

impl LlmWorkerConfig {
    pub fn deepseek(api_key: String, model: String) -> Self {
        Self {
            api_key,
            model,
            base_url: "https://api.deepseek.com".to_owned(),
            timeout_ms: 60_000,
        }
    }
}

#[async_trait]
impl TaskWorker for OhMyHarnessWorker {
    async fn execute(&self, task: Task, context: WorkerContext) -> anyhow::Result<WorkerResult> {
        let config = self.config.clone();
        let prompt = task_prompt(
            &task,
            &context.memories,
            &context.skills,
            &context.allowed_tools,
            &context.notes,
            &context.conversation_messages,
        );
        dispatch_task_with_harness(config, prompt).await
    }
}

async fn dispatch_task_with_harness(
    config: LlmWorkerConfig,
    prompt: String,
) -> anyhow::Result<WorkerResult> {
    let client = Arc::new(deepseek::client(config.api_key)) as Arc<dyn LlmClient>;
    let state = Arc::new(Mutex::new(HarnessWorkerState::default()));
    let tool_registry = worker_tool_registry(state.clone());
    let sandbox = OsEnvSandbox::new(SandboxConfig {
        fs_allowlist: Vec::new(),
        fs_denylist: Vec::new(),
        net_allowlist: Vec::new(),
        resource_limits: ResourceLimits {
            max_cpus: None,
            max_memory_mb: None,
            max_disk_mb: None,
            timeout: None,
        },
        work_dir: std::env::current_dir().ok(),
    });
    sandbox.start().await?;

    let mut opts = AgentHarnessOptions::new(config.model);
    opts.max_tokens = 1200;
    opts.system_prompt = Some(HARNESS_WORKER_SYSTEM_PROMPT.to_owned());
    opts.tools = tool_registry.subset(&[
        "remember",
        "record_artifact",
        "create_follow_up_task",
        "complete_task",
        "block_task",
    ]);

    let harness = AgentHarness::new_in_memory(client, sandbox.env(), opts).await;
    let mut events = harness.subscribe();
    harness.prompt(prompt).await?;
    harness.wait_for_idle().await;

    let mut assistant_text = String::new();
    while let Ok(event) = events.try_recv() {
        if let AgentHarnessEvent::Agent(llm_harness_agent::prelude::AgentEvent::AgentEnd {
            new_messages,
        }) = event.as_ref()
        {
            assistant_text.push_str(&assistant_text_from_messages(new_messages));
        }
    }

    let mut state = state.lock().await.clone();
    let result = state
        .result
        .take()
        .unwrap_or_else(|| parse_worker_result_text(&assistant_text));
    Ok(state.merge_into_result(result))
}

const HARNESS_WORKER_SYSTEM_PROMPT: &str = "You are a worker agent inside Persistent Agent. Execute the assigned task as far as possible using the provided context. You must call complete_task when the task is genuinely done, or block_task when user input is required. Before complete_task, call remember only for durable preferences, pitfalls, project conventions, or reusable task learnings. Call record_artifact for durable outputs or references. Call create_follow_up_task only for concrete one-off work that should be queued after this task succeeds. Do not mark a task completed if you only explained what should be done.";

#[derive(Debug, Clone, Default)]
struct HarnessWorkerState {
    result: Option<WorkerResult>,
    memory_candidates: Vec<String>,
    artifacts: Vec<WorkerArtifact>,
    follow_up_tasks: Vec<WorkerFollowUpTask>,
}

impl HarnessWorkerState {
    fn merge_into_result(&mut self, result: WorkerResult) -> WorkerResult {
        match result {
            WorkerResult::Completed {
                summary,
                mut memory_candidates,
                mut artifacts,
                mut follow_up_tasks,
            } => {
                memory_candidates.extend(std::mem::take(&mut self.memory_candidates));
                artifacts.extend(std::mem::take(&mut self.artifacts));
                follow_up_tasks.extend(std::mem::take(&mut self.follow_up_tasks));
                WorkerResult::Completed {
                    summary,
                    memory_candidates: dedupe_strings(memory_candidates),
                    artifacts: dedupe_artifacts(artifacts),
                    follow_up_tasks: dedupe_follow_up_tasks(follow_up_tasks),
                }
            }
            WorkerResult::Blocked { reason } => WorkerResult::Blocked { reason },
        }
    }
}

fn product_lifecycle_tools(state: Arc<Mutex<HarnessWorkerState>>) -> Vec<Arc<dyn Tool>> {
    vec![
        Arc::new(RememberTool::new(state.clone())),
        Arc::new(RecordArtifactTool::new(state.clone())),
        Arc::new(CreateFollowUpTaskTool::new(state.clone())),
        Arc::new(CompleteTaskTool::new(state.clone())),
        Arc::new(BlockTaskTool::new(state)),
    ]
}

fn worker_tool_registry(state: Arc<Mutex<HarnessWorkerState>>) -> Arc<InMemoryToolRegistry> {
    let registry = Arc::new(InMemoryToolRegistry::new());
    for tool in product_lifecycle_tools(state) {
        registry.register(tool);
    }
    registry
}

struct CompleteTaskTool {
    state: Arc<Mutex<HarnessWorkerState>>,
    schema: serde_json::Value,
}

impl CompleteTaskTool {
    fn new(state: Arc<Mutex<HarnessWorkerState>>) -> Self {
        Self {
            state,
            schema: json!({
                "type": "object",
                "properties": {
                    "summary": { "type": "string" },
                    "memory_candidates": { "type": "array", "items": { "type": "string" } },
                    "artifacts": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": { "type": "string" },
                                "artifact_type": { "type": "string" },
                                "uri": { "type": "string" },
                                "summary": { "type": "string" }
                            },
                            "required": ["name", "artifact_type", "uri"]
                        }
                    },
                    "follow_up_tasks": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "title": { "type": "string" },
                                "description": { "type": "string" },
                                "priority": { "type": "integer" },
                                "requested_skills": { "type": "array", "items": { "type": "string" } }
                            },
                            "required": ["title"]
                        }
                    }
                },
                "required": ["summary"]
            }),
        }
    }
}

impl Tool for CompleteTaskTool {
    fn name(&self) -> &str {
        "complete_task"
    }

    fn description(&self) -> &str {
        "Mark the assigned task as completed and provide the final durable result."
    }

    fn parameters_schema(&self) -> &serde_json::Value {
        &self.schema
    }

    fn execution_mode(&self) -> ToolExecutionMode {
        ToolExecutionMode::Sequential
    }

    fn execute<'a>(
        &'a self,
        args: serde_json::Value,
        _ctx: &'a ToolContext,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<ToolResult, ToolError>> + Send + 'a>,
    > {
        Box::pin(async move {
            let summary = required_string(&args, "summary")?;
            let parsed = serde_json::from_value::<StructuredWorkerResult>(json!({
                "status": "completed",
                "summary": summary,
                "memory_candidates": args.get("memory_candidates").cloned().unwrap_or_else(|| json!([])),
                "artifacts": args.get("artifacts").cloned().unwrap_or_else(|| json!([])),
                "follow_up_tasks": args.get("follow_up_tasks").cloned().unwrap_or_else(|| json!([])),
            }))
            .map_err(|error| ToolError::InvalidArguments(error.to_string()))?;
            let result = parse_structured_worker_result(parsed);
            self.state.lock().await.result = Some(result);
            Ok(tool_text_result("task marked completed", true))
        })
    }
}

struct BlockTaskTool {
    state: Arc<Mutex<HarnessWorkerState>>,
    schema: serde_json::Value,
}

impl BlockTaskTool {
    fn new(state: Arc<Mutex<HarnessWorkerState>>) -> Self {
        Self {
            state,
            schema: json!({
                "type": "object",
                "properties": {
                    "reason": { "type": "string" }
                },
                "required": ["reason"]
            }),
        }
    }
}

impl Tool for BlockTaskTool {
    fn name(&self) -> &str {
        "block_task"
    }

    fn description(&self) -> &str {
        "Mark the assigned task as blocked because user input or external context is required."
    }

    fn parameters_schema(&self) -> &serde_json::Value {
        &self.schema
    }

    fn execution_mode(&self) -> ToolExecutionMode {
        ToolExecutionMode::Sequential
    }

    fn execute<'a>(
        &'a self,
        args: serde_json::Value,
        _ctx: &'a ToolContext,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<ToolResult, ToolError>> + Send + 'a>,
    > {
        Box::pin(async move {
            let reason = required_string(&args, "reason")?;
            self.state.lock().await.result = Some(WorkerResult::Blocked { reason });
            Ok(tool_text_result("task marked blocked", true))
        })
    }
}

struct RememberTool {
    state: Arc<Mutex<HarnessWorkerState>>,
    schema: serde_json::Value,
}

impl RememberTool {
    fn new(state: Arc<Mutex<HarnessWorkerState>>) -> Self {
        Self {
            state,
            schema: json!({
                "type": "object",
                "properties": {
                    "content": { "type": "string" }
                },
                "required": ["content"]
            }),
        }
    }
}

impl Tool for RememberTool {
    fn name(&self) -> &str {
        "remember"
    }

    fn description(&self) -> &str {
        "Record one durable memory candidate learned while executing this task."
    }

    fn parameters_schema(&self) -> &serde_json::Value {
        &self.schema
    }

    fn execute<'a>(
        &'a self,
        args: serde_json::Value,
        _ctx: &'a ToolContext,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<ToolResult, ToolError>> + Send + 'a>,
    > {
        Box::pin(async move {
            let content = required_string(&args, "content")?;
            self.state.lock().await.memory_candidates.push(content);
            Ok(tool_text_result("memory candidate recorded", false))
        })
    }
}

struct RecordArtifactTool {
    state: Arc<Mutex<HarnessWorkerState>>,
    schema: serde_json::Value,
}

impl RecordArtifactTool {
    fn new(state: Arc<Mutex<HarnessWorkerState>>) -> Self {
        Self {
            state,
            schema: json!({
                "type": "object",
                "properties": {
                    "name": { "type": "string" },
                    "artifact_type": { "type": "string" },
                    "uri": { "type": "string" },
                    "summary": { "type": "string" }
                },
                "required": ["name", "artifact_type", "uri"]
            }),
        }
    }
}

impl Tool for RecordArtifactTool {
    fn name(&self) -> &str {
        "record_artifact"
    }

    fn description(&self) -> &str {
        "Record a durable artifact or external reference produced by the task."
    }

    fn parameters_schema(&self) -> &serde_json::Value {
        &self.schema
    }

    fn execute<'a>(
        &'a self,
        args: serde_json::Value,
        _ctx: &'a ToolContext,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<ToolResult, ToolError>> + Send + 'a>,
    > {
        Box::pin(async move {
            let artifact = WorkerArtifact {
                name: required_string(&args, "name")?,
                artifact_type: required_string(&args, "artifact_type")?,
                uri: required_string(&args, "uri")?,
                summary: args
                    .get("summary")
                    .and_then(|value| value.as_str())
                    .map(str::trim)
                    .filter(|value| !value.is_empty())
                    .map(ToOwned::to_owned),
            };
            self.state.lock().await.artifacts.push(artifact);
            Ok(tool_text_result("artifact recorded", false))
        })
    }
}

struct CreateFollowUpTaskTool {
    state: Arc<Mutex<HarnessWorkerState>>,
    schema: serde_json::Value,
}

impl CreateFollowUpTaskTool {
    fn new(state: Arc<Mutex<HarnessWorkerState>>) -> Self {
        Self {
            state,
            schema: json!({
                "type": "object",
                "properties": {
                    "title": { "type": "string" },
                    "description": { "type": "string" },
                    "priority": { "type": "integer" },
                    "requested_skills": { "type": "array", "items": { "type": "string" } }
                },
                "required": ["title"]
            }),
        }
    }
}

impl Tool for CreateFollowUpTaskTool {
    fn name(&self) -> &str {
        "create_follow_up_task"
    }

    fn description(&self) -> &str {
        "Queue a concrete one-off follow-up task after the current task succeeds."
    }

    fn parameters_schema(&self) -> &serde_json::Value {
        &self.schema
    }

    fn execute<'a>(
        &'a self,
        args: serde_json::Value,
        _ctx: &'a ToolContext,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<ToolResult, ToolError>> + Send + 'a>,
    > {
        Box::pin(async move {
            let follow_up = WorkerFollowUpTask {
                title: required_string(&args, "title")?,
                description: args
                    .get("description")
                    .and_then(|value| value.as_str())
                    .map(str::trim)
                    .unwrap_or_default()
                    .to_owned(),
                priority: args
                    .get("priority")
                    .and_then(|value| value.as_i64())
                    .unwrap_or(0),
                requested_skills: args
                    .get("requested_skills")
                    .and_then(|value| value.as_array())
                    .map(|values| {
                        values
                            .iter()
                            .filter_map(|value| value.as_str())
                            .map(str::trim)
                            .filter(|value| !value.is_empty())
                            .map(ToOwned::to_owned)
                            .collect()
                    })
                    .unwrap_or_default(),
            };
            self.state.lock().await.follow_up_tasks.push(follow_up);
            Ok(tool_text_result("follow-up task recorded", false))
        })
    }
}

fn tool_text_result(text: &str, terminate: bool) -> ToolResult {
    ToolResult {
        content: vec![ContentBlock::Text {
            text: text.to_owned(),
        }],
        details: json!({ "message": text }),
        terminate,
    }
}

fn required_string(args: &serde_json::Value, field: &str) -> Result<String, ToolError> {
    args.get(field)
        .and_then(|value| value.as_str())
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
        .ok_or_else(|| ToolError::InvalidArguments(format!("{field} is required")))
}

fn assistant_text_from_messages(messages: &[llm_harness_agent::prelude::AgentMessage]) -> String {
    let mut output = String::new();
    for message in messages {
        let llm_harness_agent::prelude::AgentMessage::Assistant(assistant) = message else {
            continue;
        };
        for block in &assistant.content {
            if let ContentBlock::Text { text } = block {
                output.push_str(text);
            }
        }
    }
    output
}

fn task_prompt(
    task: &Task,
    memories: &[Memory],
    skills: &[Skill],
    allowed_tools: &[String],
    notes: &[TaskNote],
    conversation_messages: &[ConversationMessage],
) -> String {
    format!(
        "Task title: {}\nTask type: {}\nPriority: {}\nRequested skills: {}\nMatched skills: {}\nActive skills: {}\nAllowed tools: {}\n\nActive skill resources:\n{}\n\nRelevant approved memories:\n{}\n\nTask notes:\n{}\n\nRecent task conversation:\n{}\n\nTask description:\n{}",
        task.title,
        task.task_type,
        task.priority,
        if task.requested_skills.is_empty() {
            "none".to_owned()
        } else {
            task.requested_skills.join(", ")
        },
        if task.matched_skills.is_empty() {
            "none".to_owned()
        } else {
            task.matched_skills.join(", ")
        },
        format_active_skill_names(task),
        format_allowed_tools(allowed_tools),
        format_skills(skills),
        format_memories(memories),
        format_notes(notes),
        format_conversation(conversation_messages),
        task.description
    )
}

fn format_skills(skills: &[Skill]) -> String {
    if skills.is_empty() {
        return "none".to_owned();
    }

    skills
        .iter()
        .map(|skill| {
            format!(
                "- {}: {}\n  tools: {}\n  resource_path: {}",
                skill.name,
                if skill.description.trim().is_empty() {
                    "no description"
                } else {
                    skill.description.trim()
                },
                if skill.tool_subset.is_empty() {
                    "none".to_owned()
                } else {
                    skill.tool_subset.join(", ")
                },
                skill.resource_path.as_deref().unwrap_or("none")
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn allowed_tools_for_skills(skills: &[Skill]) -> Vec<String> {
    skills
        .iter()
        .flat_map(|skill| skill.tool_subset.iter())
        .map(|tool| tool.trim())
        .filter(|tool| !tool.is_empty())
        .map(ToOwned::to_owned)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

fn format_allowed_tools(allowed_tools: &[String]) -> String {
    if allowed_tools.is_empty() {
        "none".to_owned()
    } else {
        allowed_tools.join(", ")
    }
}

fn format_memories(memories: &[Memory]) -> String {
    if memories.is_empty() {
        return "none".to_owned();
    }

    memories
        .iter()
        .map(|memory| {
            format!(
                "- [{} confidence {:.2}] {}",
                memory.scope, memory.confidence, memory.content
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn format_notes(notes: &[TaskNote]) -> String {
    if notes.is_empty() {
        return "none".to_owned();
    }

    notes
        .iter()
        .map(|note| format!("- {}: {}", note.actor, note.content))
        .collect::<Vec<_>>()
        .join("\n")
}

fn format_conversation(messages: &[ConversationMessage]) -> String {
    if messages.is_empty() {
        return "none".to_owned();
    }

    messages
        .iter()
        .map(|message| format!("- {}: {}", message.role, message.content))
        .collect::<Vec<_>>()
        .join("\n")
}

fn select_relevant_memories(task: &Task, memories: Vec<Memory>, limit: usize) -> Vec<Memory> {
    let tokens = task_memory_tokens(task);
    let mut scored = memories
        .into_iter()
        .filter_map(|memory| {
            let haystack = format!("{} {}", memory.scope, memory.content).to_lowercase();
            let score = tokens
                .iter()
                .filter(|token| haystack.contains(token.as_str()))
                .count();
            (score > 0).then_some((score, memory))
        })
        .collect::<Vec<_>>();

    scored.sort_by(|(left_score, left_memory), (right_score, right_memory)| {
        right_score
            .cmp(left_score)
            .then_with(|| right_memory.confidence.total_cmp(&left_memory.confidence))
            .then_with(|| right_memory.created_at.cmp(&left_memory.created_at))
    });

    scored
        .into_iter()
        .take(limit)
        .map(|(_, memory)| memory)
        .collect()
}

fn task_memory_tokens(task: &Task) -> Vec<String> {
    let text = format!(
        "{} {} {}",
        task.title,
        task.description,
        active_skill_names(task).join(" ")
    )
    .to_lowercase();
    let mut tokens = Vec::new();
    let mut current = String::new();

    for ch in text.chars() {
        if ch.is_alphanumeric() || ch == '-' || ch == '_' {
            current.push(ch);
            continue;
        }

        if is_memory_token(&current) && !tokens.contains(&current) {
            tokens.push(current.clone());
        }
        current.clear();
    }

    if is_memory_token(&current) && !tokens.contains(&current) {
        tokens.push(current);
    }

    tokens
}

fn is_memory_token(token: &str) -> bool {
    token.chars().count() >= 3
        && !matches!(
            token,
            "and"
                | "for"
                | "the"
                | "use"
                | "with"
                | "task"
                | "run"
                | "this"
                | "that"
                | "from"
                | "into"
        )
}

fn active_skill_names(task: &Task) -> Vec<String> {
    let mut skills = Vec::new();
    for skill in task
        .requested_skills
        .iter()
        .chain(task.matched_skills.iter())
    {
        if !skills.contains(skill) {
            skills.push(skill.clone());
        }
    }

    skills
}

fn format_active_skill_names(task: &Task) -> String {
    let skills = active_skill_names(task);
    if skills.is_empty() {
        "none".to_owned()
    } else {
        skills.join(", ")
    }
}

fn memory_candidate_contents(
    task: &Task,
    summary: &str,
    memory_candidates: &[String],
) -> Vec<String> {
    let cleaned = memory_candidates
        .iter()
        .map(|candidate| candidate.trim())
        .filter(|candidate| !candidate.is_empty())
        .map(ToOwned::to_owned)
        .collect::<Vec<_>>();

    if cleaned.is_empty() {
        return vec![format!(
            "Task '{}' completed with summary: {}",
            task.title,
            summary.trim()
        )];
    }

    cleaned
}

#[derive(Debug, Deserialize)]
struct StructuredWorkerResult {
    status: String,
    summary: Option<String>,
    reason: Option<String>,
    memory_candidates: Option<Vec<String>>,
    artifacts: Option<Vec<WorkerArtifact>>,
    follow_up_tasks: Option<Vec<WorkerFollowUpTask>>,
}

fn parse_worker_result_text(text: &str) -> WorkerResult {
    let trimmed = trim_json_code_fence(text.trim());
    if let Ok(parsed) = serde_json::from_str::<StructuredWorkerResult>(trimmed) {
        return parse_structured_worker_result(parsed);
    }

    fallback_worker_result(text)
}

fn parse_structured_worker_result(parsed: StructuredWorkerResult) -> WorkerResult {
    match parsed.status.to_lowercase().as_str() {
        "blocked" => WorkerResult::Blocked {
            reason: parsed
                .reason
                .or(parsed.summary)
                .filter(|value| !value.trim().is_empty())
                .unwrap_or_else(|| "Worker is blocked and needs user input.".to_owned()),
        },
        "completed" => WorkerResult::Completed {
            summary: parsed
                .summary
                .or(parsed.reason)
                .filter(|value| !value.trim().is_empty())
                .unwrap_or_else(|| "Worker completed the task.".to_owned()),
            memory_candidates: clean_memory_candidates(parsed.memory_candidates),
            artifacts: clean_artifacts(parsed.artifacts),
            follow_up_tasks: clean_follow_up_tasks(parsed.follow_up_tasks),
        },
        _ => fallback_worker_result(""),
    }
}

fn trim_json_code_fence(text: &str) -> &str {
    let Some(without_opening) = text
        .strip_prefix("```json")
        .or_else(|| text.strip_prefix("```"))
    else {
        return text;
    };

    without_opening
        .strip_suffix("```")
        .unwrap_or(without_opening)
        .trim()
}

fn clean_memory_candidates(candidates: Option<Vec<String>>) -> Vec<String> {
    candidates
        .unwrap_or_default()
        .into_iter()
        .map(|candidate| candidate.trim().to_owned())
        .filter(|candidate| !candidate.is_empty())
        .collect()
}

fn clean_artifacts(artifacts: Option<Vec<WorkerArtifact>>) -> Vec<WorkerArtifact> {
    artifacts
        .unwrap_or_default()
        .into_iter()
        .filter_map(|artifact| {
            let name = artifact.name.trim().to_owned();
            let artifact_type = artifact.artifact_type.trim().to_owned();
            let uri = artifact.uri.trim().to_owned();
            if name.is_empty() || artifact_type.is_empty() || uri.is_empty() {
                return None;
            }

            Some(WorkerArtifact {
                name,
                artifact_type,
                uri,
                summary: artifact
                    .summary
                    .map(|summary| summary.trim().to_owned())
                    .filter(|summary| !summary.is_empty()),
            })
        })
        .collect()
}

fn clean_follow_up_tasks(tasks: Option<Vec<WorkerFollowUpTask>>) -> Vec<WorkerFollowUpTask> {
    tasks
        .unwrap_or_default()
        .into_iter()
        .take(10)
        .filter_map(|task| {
            let title = task.title.trim().to_owned();
            if title.is_empty() {
                return None;
            }

            let description = task.description.trim().to_owned();
            let requested_skills = task
                .requested_skills
                .into_iter()
                .map(|skill| skill.trim().to_owned())
                .filter(|skill| !skill.is_empty())
                .collect();

            Some(WorkerFollowUpTask {
                title: title.clone(),
                description: if description.is_empty() {
                    title
                } else {
                    description
                },
                priority: task.priority.clamp(-100, 100),
                requested_skills,
            })
        })
        .collect()
}

fn dedupe_strings(values: Vec<String>) -> Vec<String> {
    let mut seen = BTreeSet::new();
    values
        .into_iter()
        .filter(|value| seen.insert(value.clone()))
        .collect()
}

fn dedupe_artifacts(artifacts: Vec<WorkerArtifact>) -> Vec<WorkerArtifact> {
    let mut seen = BTreeSet::new();
    artifacts
        .into_iter()
        .filter(|artifact| {
            seen.insert(format!(
                "{}\u{1f}{}\u{1f}{}",
                artifact.name, artifact.artifact_type, artifact.uri
            ))
        })
        .collect()
}

fn dedupe_follow_up_tasks(tasks: Vec<WorkerFollowUpTask>) -> Vec<WorkerFollowUpTask> {
    let mut seen = BTreeSet::new();
    tasks
        .into_iter()
        .filter(|task| seen.insert(format!("{}\u{1f}{}", task.title, task.description)))
        .collect()
}

fn fallback_worker_result(text: &str) -> WorkerResult {
    let trimmed = text.trim();
    let normalized = text.to_lowercase();
    if [
        "need more user input",
        "need user input",
        "please provide",
        "missing context",
        "cannot complete",
        "can't complete",
        "unable to complete",
        "i need",
    ]
    .iter()
    .any(|marker| normalized.contains(marker))
    {
        WorkerResult::Blocked {
            reason: if trimmed.is_empty() {
                "Worker is blocked and needs user input.".to_owned()
            } else {
                trimmed.to_owned()
            },
        }
    } else {
        WorkerResult::Completed {
            summary: if trimmed.is_empty() {
                "Worker completed the task.".to_owned()
            } else {
                trimmed.to_owned()
            },
            memory_candidates: Vec::new(),
            artifacts: Vec::new(),
            follow_up_tasks: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "status")]
pub enum WorkerResult {
    Completed {
        summary: String,
        memory_candidates: Vec<String>,
        artifacts: Vec<WorkerArtifact>,
        follow_up_tasks: Vec<WorkerFollowUpTask>,
    },
    Blocked {
        reason: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct WorkerArtifact {
    pub name: String,
    pub artifact_type: String,
    pub uri: String,
    pub summary: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct WorkerFollowUpTask {
    pub title: String,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub priority: i64,
    #[serde(default)]
    pub requested_skills: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulerTick {
    pub recovered_tasks: Vec<Task>,
    pub requeued_tasks: Vec<Task>,
    pub claimed_task: Option<Task>,
    pub outcome: SchedulerOutcome,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "type")]
pub enum SchedulerOutcome {
    Idle,
    Completed {
        summary: String,
        follow_up_tasks: Vec<Task>,
        memory_candidates: Vec<Memory>,
    },
    Blocked {
        reason: String,
    },
    Failed {
        error: String,
    },
    RetryScheduled {
        error: String,
        next_attempt: i64,
        max_attempts: i64,
    },
    Superseded {
        status: TaskStatus,
        reason: String,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use persistent_agent_db::Db;
    use persistent_agent_domain::{CreateSkill, CreateTask, TaskType};
    use std::{
        sync::{
            Arc, Mutex as StdMutex,
            atomic::{AtomicUsize, Ordering},
        },
        time::Duration,
    };
    use uuid::Uuid;

    #[test]
    fn scheduler_policy_clamps_capacity_and_lease() {
        assert_eq!(SchedulerPolicy::serial().worker_capacity, 1);
        assert_eq!(SchedulerPolicy::serial().max_attempts, 1);
        assert_eq!(
            SchedulerPolicy::new(0, 0).with_max_attempts(0),
            SchedulerPolicy {
                worker_capacity: 1,
                lease_seconds: 1,
                max_attempts: 1,
                memory_auto_approve_confidence: None,
            }
        );
    }

    #[tokio::test]
    async fn scheduler_can_be_constructed_with_policy() -> anyhow::Result<()> {
        let db = Db::connect("sqlite::memory:").await?;
        let scheduler = Scheduler::with_policy(db, StubWorker, SchedulerPolicy::new(3, 45));

        assert_eq!(
            scheduler.policy(),
            SchedulerPolicy {
                worker_capacity: 3,
                lease_seconds: 45,
                max_attempts: 1,
                memory_auto_approve_confidence: None,
            }
        );

        Ok(())
    }

    #[tokio::test]
    async fn harness_worker_registers_product_lifecycle_tools() {
        let state = Arc::new(Mutex::new(HarnessWorkerState::default()));
        let registry = worker_tool_registry(state);
        let tools = registry.subset(&[
            "remember",
            "record_artifact",
            "create_follow_up_task",
            "complete_task",
            "block_task",
        ]);
        let names = tools.iter().map(|tool| tool.name()).collect::<Vec<_>>();

        assert_eq!(
            names,
            vec![
                "remember",
                "record_artifact",
                "create_follow_up_task",
                "complete_task",
                "block_task"
            ]
        );
    }

    #[test]
    fn harness_worker_merges_tool_outputs_into_completion() {
        let mut state = HarnessWorkerState {
            result: None,
            memory_candidates: vec!["Prefer focused regression checks.".to_owned()],
            artifacts: vec![WorkerArtifact {
                name: "report.md".to_owned(),
                artifact_type: "file".to_owned(),
                uri: "file://report.md".to_owned(),
                summary: Some("Run report".to_owned()),
            }],
            follow_up_tasks: vec![WorkerFollowUpTask {
                title: "Run regression tests".to_owned(),
                description: "Validate the worker change".to_owned(),
                priority: 1,
                requested_skills: vec!["testing".to_owned()],
            }],
        };

        let result = state.merge_into_result(WorkerResult::Completed {
            summary: "Harness completed the task.".to_owned(),
            memory_candidates: vec!["Prefer focused regression checks.".to_owned()],
            artifacts: Vec::new(),
            follow_up_tasks: Vec::new(),
        });

        assert!(matches!(
            result,
            WorkerResult::Completed {
                ref summary,
                ref memory_candidates,
                ref artifacts,
                ref follow_up_tasks,
            } if summary == "Harness completed the task."
                && memory_candidates == &vec!["Prefer focused regression checks.".to_owned()]
                && artifacts.len() == 1
                && follow_up_tasks.len() == 1
        ));
    }

    #[tokio::test]
    async fn scheduler_recovers_expired_running_task_before_claiming_work() -> anyhow::Result<()> {
        let db = Db::connect("sqlite::memory:").await?;
        let task = db
            .create_task(
                CreateTask {
                    title: "Resume stale work".to_owned(),
                    description: "Recover before executing".to_owned(),
                    task_type: TaskType::OneOff,
                    priority: 0,
                    requested_skills: Vec::new(),
                    schedule: None,
                    created_by: "test".to_owned(),
                },
                "test",
            )
            .await?;
        db.claim_next_runnable("previous-worker", 1).await?;
        tokio::time::sleep(Duration::from_millis(1_100)).await;

        let scheduler = Scheduler::new(db.clone(), StubWorker);
        let tick = scheduler.tick().await?;

        assert_eq!(tick.recovered_tasks.len(), 1);
        assert_eq!(tick.recovered_tasks[0].id, task.id);
        assert_eq!(
            tick.claimed_task.as_ref().map(|task| task.id),
            Some(task.id)
        );
        assert!(matches!(tick.outcome, SchedulerOutcome::Completed { .. }));

        Ok(())
    }

    #[test]
    fn parses_structured_completed_worker_result() {
        let result = parse_worker_result_text(
            r#"{"status":"completed","summary":"Updated the README and ran tests."}"#,
        );

        assert!(matches!(
            result,
            WorkerResult::Completed { ref summary, .. } if summary.contains("Updated the README")
        ));
    }

    #[test]
    fn parses_structured_worker_memory_candidates() {
        let result = parse_worker_result_text(
            r#"{"status":"completed","summary":"Finished.","memory_candidates":["Prefer cargo test --workspace.","  ","Avoid flaky selectors."]}"#,
        );

        assert!(matches!(
            result,
            WorkerResult::Completed { ref memory_candidates, .. }
                if memory_candidates == &vec![
                    "Prefer cargo test --workspace.".to_owned(),
                    "Avoid flaky selectors.".to_owned()
                ]
        ));
    }

    #[test]
    fn parses_structured_worker_artifacts() {
        let result = parse_worker_result_text(
            r#"{"status":"completed","summary":"Finished.","artifacts":[{"name":"report.md","artifact_type":"file","uri":"file://report.md","summary":"Run report"},{"name":"","artifact_type":"file","uri":"file://ignored"}]}"#,
        );

        assert!(matches!(
            result,
            WorkerResult::Completed { ref artifacts, .. }
                if artifacts == &vec![WorkerArtifact {
                    name: "report.md".to_owned(),
                    artifact_type: "file".to_owned(),
                    uri: "file://report.md".to_owned(),
                    summary: Some("Run report".to_owned()),
                }]
        ));
    }

    #[test]
    fn parses_structured_worker_follow_up_tasks() {
        let result = parse_worker_result_text(
            r#"{"status":"completed","summary":"Finished.","follow_up_tasks":[{"title":"Run regression tests","description":"Run the regression test suite","priority":7,"requested_skills":[" testing ",""]},{"title":" ","description":"ignored","priority":1},{"title":"Write release note","description":"","priority":999,"requested_skills":[]}]}"#,
        );

        assert!(matches!(
            result,
            WorkerResult::Completed { ref follow_up_tasks, .. }
                if follow_up_tasks == &vec![
                    WorkerFollowUpTask {
                        title: "Run regression tests".to_owned(),
                        description: "Run the regression test suite".to_owned(),
                        priority: 7,
                        requested_skills: vec!["testing".to_owned()],
                    },
                    WorkerFollowUpTask {
                        title: "Write release note".to_owned(),
                        description: "Write release note".to_owned(),
                        priority: 100,
                        requested_skills: Vec::new(),
                    },
                ]
        ));
    }

    #[test]
    fn parses_structured_blocked_worker_result() {
        let result = parse_worker_result_text(
            r#"{"status":"blocked","reason":"I need the target repository URL."}"#,
        );

        assert!(matches!(
            result,
            WorkerResult::Blocked { ref reason } if reason.contains("target repository")
        ));
    }

    #[test]
    fn parses_fenced_structured_worker_result() {
        let result = parse_worker_result_text(
            "```json\n{\"status\":\"completed\",\"summary\":\"Finished the task.\"}\n```",
        );

        assert!(matches!(
            result,
            WorkerResult::Completed { ref summary, .. } if summary == "Finished the task."
        ));
    }

    #[test]
    fn falls_back_to_blocked_for_missing_context_text() {
        let result = parse_worker_result_text(
            "I cannot complete this because the task is missing context. Please provide the repository.",
        );

        assert!(matches!(
            result,
            WorkerResult::Blocked { ref reason } if reason.contains("missing context")
        ));
    }

    #[test]
    fn builds_task_prompt_with_skills() {
        let now = Utc::now();
        let task = Task {
            id: Uuid::now_v7(),
            title: "Check issues".to_owned(),
            description: "Look for repository issues.".to_owned(),
            task_type: TaskType::Recurring,
            status: TaskStatus::Queued,
            priority: 3,
            queue_position: 1,
            created_by: "user".to_owned(),
            conversation_id: None,
            requested_skills: vec!["github".to_owned()],
            matched_skills: vec!["rust".to_owned(), "github".to_owned()],
            schedule: None,
            attempt_count: 0,
            lease_owner: None,
            lease_expires_at: None,
            last_run_at: None,
            next_run_at: None,
            blocked_reason: None,
            result_summary: None,
            created_at: now,
            updated_at: now,
        };

        let skill = Skill {
            id: Uuid::now_v7(),
            name: "github".to_owned(),
            description: "Inspect GitHub repositories and issues.".to_owned(),
            trigger_rules: vec!["github".to_owned()],
            tool_subset: vec!["github_search".to_owned(), "shell".to_owned()],
            resource_path: Some("skills/github".to_owned()),
            created_at: now,
            updated_at: now,
        };

        let prompt = task_prompt(
            &task,
            &[],
            &[skill],
            &["github_search".to_owned(), "shell".to_owned()],
            &[],
            &[],
        );

        assert!(prompt.contains("Check issues"));
        assert!(prompt.contains("recurring"));
        assert!(prompt.contains("github"));
        assert!(prompt.contains("Matched skills: rust, github"));
        assert!(prompt.contains("Active skills: github, rust"));
        assert!(prompt.contains("Allowed tools: github_search, shell"));
        assert!(prompt.contains("Active skill resources"));
        assert!(prompt.contains("Inspect GitHub repositories"));
        assert!(prompt.contains("tools: github_search, shell"));
        assert!(prompt.contains("resource_path: skills/github"));
    }

    #[test]
    fn injects_relevant_approved_memories_into_prompt() {
        let now = Utc::now();
        let task = Task {
            id: Uuid::now_v7(),
            title: "Run cargo tests".to_owned(),
            description: "Use the repository workflow.".to_owned(),
            task_type: TaskType::OneOff,
            status: TaskStatus::Queued,
            priority: 0,
            queue_position: 0,
            created_by: "user".to_owned(),
            conversation_id: None,
            requested_skills: vec!["rust".to_owned()],
            matched_skills: Vec::new(),
            schedule: None,
            attempt_count: 0,
            lease_owner: None,
            lease_expires_at: None,
            last_run_at: None,
            next_run_at: None,
            blocked_reason: None,
            result_summary: None,
            created_at: now,
            updated_at: now,
        };
        let relevant = Memory {
            id: Uuid::now_v7(),
            scope: "project".to_owned(),
            content: "Prefer cargo test --workspace before committing.".to_owned(),
            source_task_id: None,
            status: MemoryStatus::Approved,
            confidence: 0.9,
            created_at: now,
        };
        let unrelated = Memory {
            id: Uuid::now_v7(),
            scope: "project".to_owned(),
            content: "Use design screenshots for frontend polish.".to_owned(),
            source_task_id: None,
            status: MemoryStatus::Approved,
            confidence: 1.0,
            created_at: now,
        };

        let memories = select_relevant_memories(&task, vec![unrelated, relevant], 5);
        let prompt = task_prompt(&task, &memories, &[], &[], &[], &[]);

        assert_eq!(memories.len(), 1);
        assert!(prompt.contains("Prefer cargo test --workspace"));
        assert!(!prompt.contains("design screenshots"));
    }

    #[test]
    fn injects_recent_task_conversation_into_prompt() {
        let now = Utc::now();
        let conversation_id = Uuid::now_v7();
        let task_id = Uuid::now_v7();
        let task = Task {
            id: task_id,
            title: "Use user context".to_owned(),
            description: "Continue after clarification.".to_owned(),
            task_type: TaskType::OneOff,
            status: TaskStatus::Queued,
            priority: 0,
            queue_position: 0,
            created_by: "user".to_owned(),
            conversation_id: Some(conversation_id),
            requested_skills: Vec::new(),
            matched_skills: Vec::new(),
            schedule: None,
            attempt_count: 0,
            lease_owner: None,
            lease_expires_at: None,
            last_run_at: None,
            next_run_at: None,
            blocked_reason: None,
            result_summary: None,
            created_at: now,
            updated_at: now,
        };
        let messages = vec![ConversationMessage {
            id: Uuid::now_v7(),
            conversation_id,
            task_id: Some(task_id),
            role: "user".to_owned(),
            content: "The target repository is oh-my-harness/Persistent-Agent.".to_owned(),
            created_at: now,
        }];

        let prompt = task_prompt(&task, &[], &[], &[], &[], &messages);

        assert!(prompt.contains("Recent task conversation"));
        assert!(prompt.contains("user: The target repository"));
    }

    #[test]
    fn injects_task_notes_into_prompt() {
        let now = Utc::now();
        let task = Task {
            id: Uuid::now_v7(),
            title: "Deploy release".to_owned(),
            description: "Deploy after approvals.".to_owned(),
            task_type: TaskType::OneOff,
            status: TaskStatus::Queued,
            priority: 0,
            queue_position: 0,
            created_by: "user".to_owned(),
            conversation_id: None,
            requested_skills: Vec::new(),
            matched_skills: Vec::new(),
            schedule: None,
            attempt_count: 0,
            lease_owner: None,
            lease_expires_at: None,
            last_run_at: None,
            next_run_at: None,
            blocked_reason: None,
            result_summary: None,
            created_at: now,
            updated_at: now,
        };
        let notes = vec![TaskNote {
            id: Uuid::now_v7(),
            task_id: task.id,
            actor: "main_agent".to_owned(),
            content: "Wait for staging approval.".to_owned(),
            created_at: now,
        }];

        let prompt = task_prompt(&task, &[], &[], &[], &notes, &[]);

        assert!(prompt.contains("Task notes"));
        assert!(prompt.contains("main_agent: Wait for staging approval."));
    }

    #[test]
    fn builds_memory_candidate_from_summary() {
        let now = Utc::now();
        let task = Task {
            id: Uuid::now_v7(),
            title: "Remember setup".to_owned(),
            description: "Capture setup details.".to_owned(),
            task_type: TaskType::OneOff,
            status: TaskStatus::Queued,
            priority: 0,
            queue_position: 0,
            created_by: "user".to_owned(),
            conversation_id: None,
            requested_skills: Vec::new(),
            matched_skills: Vec::new(),
            schedule: None,
            attempt_count: 0,
            lease_owner: None,
            lease_expires_at: None,
            last_run_at: None,
            next_run_at: None,
            blocked_reason: None,
            result_summary: None,
            created_at: now,
            updated_at: now,
        };

        let candidate = memory_candidate_contents(&task, "Use cargo test.", &Vec::new()).remove(0);

        assert!(candidate.contains("Remember setup"));
        assert!(candidate.contains("Use cargo test."));
    }

    #[tokio::test]
    async fn blocked_task_records_conversation_message() -> anyhow::Result<()> {
        let db = Db::connect("sqlite::memory:").await?;
        let task = db
            .create_task(
                CreateTask {
                    title: "Blocked smoke".to_owned(),
                    description: "BLOCKED until the user provides a target repository.".to_owned(),
                    task_type: TaskType::OneOff,
                    priority: 0,
                    requested_skills: Vec::new(),
                    schedule: None,
                    created_by: "test".to_owned(),
                },
                "test",
            )
            .await?;
        let scheduler = Scheduler::new(db.clone(), StubWorker);

        let tick = scheduler.tick().await?;
        let updated = db.get_task(task.id).await?;
        let messages = db.list_task_conversation_messages(task.id, 10).await?;

        assert!(matches!(tick.outcome, SchedulerOutcome::Blocked { .. }));
        assert_eq!(updated.status, TaskStatus::WaitingForUser);
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].role, "assistant");
        assert!(messages[0].content.contains("more user input"));

        Ok(())
    }

    #[tokio::test]
    async fn blocked_task_can_resume_with_user_conversation_context() -> anyhow::Result<()> {
        let db = Db::connect("sqlite::memory:").await?;
        let task = db
            .create_task(
                CreateTask {
                    title: "Blocked resume smoke".to_owned(),
                    description: "BLOCKED until the user provides a target repository.".to_owned(),
                    task_type: TaskType::OneOff,
                    priority: 0,
                    requested_skills: Vec::new(),
                    schedule: None,
                    created_by: "test".to_owned(),
                },
                "test",
            )
            .await?;
        let scheduler = Scheduler::new(db.clone(), StubWorker);

        let first_tick = scheduler.tick().await?;
        assert!(matches!(
            first_tick.outcome,
            SchedulerOutcome::Blocked { .. }
        ));

        let blocked = db.get_task(task.id).await?;
        let conversation_id = blocked
            .conversation_id
            .expect("created tasks should have a conversation");
        db.add_conversation_message(
            conversation_id,
            Some(task.id),
            "user",
            "Use repository oh-my-harness/Persistent-Agent.",
        )
        .await?;
        db.set_task_status(task.id, TaskStatus::Queued, "test", None)
            .await?;

        let second_tick = scheduler.tick().await?;
        let updated = db.get_task(task.id).await?;

        assert!(matches!(
            second_tick.outcome,
            SchedulerOutcome::Completed { .. }
        ));
        assert_eq!(updated.status, TaskStatus::Completed);
        assert!(
            updated
                .result_summary
                .as_deref()
                .unwrap_or_default()
                .contains("latest user conversation context")
        );

        Ok(())
    }

    #[derive(Clone)]
    struct SlowCountingWorker {
        active: Arc<AtomicUsize>,
        max_active: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl TaskWorker for SlowCountingWorker {
        async fn execute(
            &self,
            task: Task,
            _context: WorkerContext,
        ) -> anyhow::Result<WorkerResult> {
            let active = self.active.fetch_add(1, Ordering::SeqCst) + 1;
            self.max_active.fetch_max(active, Ordering::SeqCst);
            tokio::time::sleep(Duration::from_millis(25)).await;
            self.active.fetch_sub(1, Ordering::SeqCst);

            Ok(WorkerResult::Completed {
                summary: format!("completed {}", task.title),
                memory_candidates: Vec::new(),
                artifacts: Vec::new(),
                follow_up_tasks: Vec::new(),
            })
        }
    }

    #[tokio::test]
    async fn concurrent_ticks_share_serial_execution_lock() -> anyhow::Result<()> {
        let db = Db::connect("sqlite::memory:").await?;
        for title in ["First", "Second"] {
            db.create_task(
                CreateTask {
                    title: title.to_owned(),
                    description: "Run slowly".to_owned(),
                    task_type: TaskType::OneOff,
                    priority: 0,
                    requested_skills: Vec::new(),
                    schedule: None,
                    created_by: "test".to_owned(),
                },
                "test",
            )
            .await?;
        }

        let worker = SlowCountingWorker {
            active: Arc::new(AtomicUsize::new(0)),
            max_active: Arc::new(AtomicUsize::new(0)),
        };
        let max_active = worker.max_active.clone();
        let scheduler = Scheduler::new(db, worker);
        let scheduler_clone = scheduler.clone();

        let (first, second) = tokio::join!(scheduler.tick(), scheduler_clone.tick());

        first?;
        second?;
        assert_eq!(max_active.load(Ordering::SeqCst), 1);

        Ok(())
    }

    #[derive(Clone)]
    struct DelayedWorker {
        delay: Duration,
    }

    #[async_trait]
    impl TaskWorker for DelayedWorker {
        async fn execute(
            &self,
            task: Task,
            _context: WorkerContext,
        ) -> anyhow::Result<WorkerResult> {
            tokio::time::sleep(self.delay).await;
            Ok(WorkerResult::Completed {
                summary: format!("completed {}", task.title),
                memory_candidates: Vec::new(),
                artifacts: Vec::new(),
                follow_up_tasks: Vec::new(),
            })
        }
    }

    #[tokio::test]
    async fn scheduler_refreshes_running_task_lease() -> anyhow::Result<()> {
        let db = Db::connect("sqlite::memory:").await?;
        let task = db
            .create_task(
                CreateTask {
                    title: "Long running work".to_owned(),
                    description: "Run long enough for heartbeat".to_owned(),
                    task_type: TaskType::OneOff,
                    priority: 0,
                    requested_skills: Vec::new(),
                    schedule: None,
                    created_by: "test".to_owned(),
                },
                "test",
            )
            .await?;

        let scheduler = Scheduler::with_policy(
            db.clone(),
            DelayedWorker {
                delay: Duration::from_millis(1_100),
            },
            SchedulerPolicy::new(1, 1),
        );

        let tick = scheduler.tick().await?;
        let events = db.list_task_attempt_events(task.id).await?;

        assert!(matches!(tick.outcome, SchedulerOutcome::Completed { .. }));
        assert!(
            events
                .iter()
                .any(|event| event.event_type == "worker_heartbeat")
        );

        Ok(())
    }

    #[tokio::test]
    async fn scheduler_does_not_overwrite_cancelled_running_task() -> anyhow::Result<()> {
        let db = Db::connect("sqlite::memory:").await?;
        let task = db
            .create_task(
                CreateTask {
                    title: "Cancelable work".to_owned(),
                    description: "Worker result should not override cancellation".to_owned(),
                    task_type: TaskType::OneOff,
                    priority: 0,
                    requested_skills: Vec::new(),
                    schedule: None,
                    created_by: "test".to_owned(),
                },
                "test",
            )
            .await?;

        let scheduler = Scheduler::with_policy(
            db.clone(),
            DelayedWorker {
                delay: Duration::from_millis(100),
            },
            SchedulerPolicy::new(1, 60),
        );
        let tick_handle = tokio::spawn(async move { scheduler.tick().await });
        tokio::time::sleep(Duration::from_millis(25)).await;
        db.set_task_status(task.id, TaskStatus::Cancelled, "test", None)
            .await?;

        let tick = tick_handle.await??;
        let final_task = db.get_task(task.id).await?;
        let events = db.list_task_attempt_events(task.id).await?;

        assert!(matches!(
            tick.outcome,
            SchedulerOutcome::Superseded {
                status: TaskStatus::Cancelled,
                ..
            }
        ));
        assert_eq!(final_task.status, TaskStatus::Cancelled);
        assert!(
            events
                .iter()
                .any(|event| event.event_type == "worker_outcome_superseded")
        );

        Ok(())
    }

    #[derive(Clone)]
    struct AbortRecordingWorker {
        finished: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl TaskWorker for AbortRecordingWorker {
        async fn execute(
            &self,
            task: Task,
            _context: WorkerContext,
        ) -> anyhow::Result<WorkerResult> {
            tokio::time::sleep(Duration::from_secs(5)).await;
            self.finished.fetch_add(1, Ordering::SeqCst);

            Ok(WorkerResult::Completed {
                summary: format!("completed {}", task.title),
                memory_candidates: Vec::new(),
                artifacts: Vec::new(),
                follow_up_tasks: Vec::new(),
            })
        }
    }

    #[tokio::test]
    async fn scheduler_aborts_worker_when_running_task_is_cancelled() -> anyhow::Result<()> {
        let db = Db::connect("sqlite::memory:").await?;
        let task = db
            .create_task(
                CreateTask {
                    title: "Abortable work".to_owned(),
                    description: "Worker should stop after cancellation".to_owned(),
                    task_type: TaskType::OneOff,
                    priority: 0,
                    requested_skills: Vec::new(),
                    schedule: None,
                    created_by: "test".to_owned(),
                },
                "test",
            )
            .await?;
        let finished = Arc::new(AtomicUsize::new(0));
        let scheduler = Scheduler::with_policy(
            db.clone(),
            AbortRecordingWorker {
                finished: finished.clone(),
            },
            SchedulerPolicy::new(1, 1),
        );
        let tick_handle = tokio::spawn(async move { scheduler.tick().await });
        tokio::time::sleep(Duration::from_millis(50)).await;
        db.set_task_status(task.id, TaskStatus::Cancelled, "test", None)
            .await?;

        let tick = tick_handle.await??;
        let final_task = db.get_task(task.id).await?;
        let events = db.list_task_attempt_events(task.id).await?;

        assert!(matches!(
            tick.outcome,
            SchedulerOutcome::Superseded {
                status: TaskStatus::Cancelled,
                ..
            }
        ));
        assert_eq!(final_task.status, TaskStatus::Cancelled);
        assert_eq!(finished.load(Ordering::SeqCst), 0);
        assert!(
            events
                .iter()
                .any(|event| event.event_type == "worker_lease_lost")
        );

        Ok(())
    }

    #[derive(Clone)]
    struct ContextRecordingWorker {
        memory_counts: Arc<StdMutex<Vec<usize>>>,
        skill_names: Arc<StdMutex<Vec<Vec<String>>>>,
        allowed_tools: Arc<StdMutex<Vec<Vec<String>>>>,
    }

    #[async_trait]
    impl TaskWorker for ContextRecordingWorker {
        async fn execute(
            &self,
            task: Task,
            context: WorkerContext,
        ) -> anyhow::Result<WorkerResult> {
            self.memory_counts
                .lock()
                .expect("memory count lock")
                .push(context.memories.len());
            self.skill_names.lock().expect("skill names lock").push(
                context
                    .skills
                    .iter()
                    .map(|skill| skill.name.clone())
                    .collect(),
            );
            self.allowed_tools
                .lock()
                .expect("allowed tools lock")
                .push(context.allowed_tools);

            Ok(WorkerResult::Completed {
                summary: format!("completed {}", task.title),
                memory_candidates: Vec::new(),
                artifacts: Vec::new(),
                follow_up_tasks: Vec::new(),
            })
        }
    }

    #[tokio::test]
    async fn scheduler_passes_relevant_approved_memories_to_worker() -> anyhow::Result<()> {
        let db = Db::connect("sqlite::memory:").await?;
        db.create_memory(
            CreateMemory {
                scope: "project".to_owned(),
                content: "For cargo work, run cargo test --workspace.".to_owned(),
                source_task_id: None,
                status: MemoryStatus::Approved,
                confidence: 0.9,
            },
            "test",
        )
        .await?;
        db.create_memory(
            CreateMemory {
                scope: "project".to_owned(),
                content: "Frontend visual checks use browser screenshots.".to_owned(),
                source_task_id: None,
                status: MemoryStatus::Pending,
                confidence: 0.9,
            },
            "test",
        )
        .await?;
        db.create_task(
            CreateTask {
                title: "Run cargo checks".to_owned(),
                description: "Validate the Rust workspace.".to_owned(),
                task_type: TaskType::OneOff,
                priority: 0,
                requested_skills: vec!["rust".to_owned()],
                schedule: None,
                created_by: "test".to_owned(),
            },
            "test",
        )
        .await?;
        let memory_counts = Arc::new(StdMutex::new(Vec::new()));
        let skill_names = Arc::new(StdMutex::new(Vec::new()));
        let allowed_tools = Arc::new(StdMutex::new(Vec::new()));
        let scheduler = Scheduler::new(
            db,
            ContextRecordingWorker {
                memory_counts: memory_counts.clone(),
                skill_names,
                allowed_tools,
            },
        );

        scheduler.tick().await?;

        assert_eq!(*memory_counts.lock().expect("memory count lock"), vec![1]);

        Ok(())
    }

    #[tokio::test]
    async fn scheduler_passes_active_skill_resources_to_worker() -> anyhow::Result<()> {
        let db = Db::connect("sqlite::memory:").await?;
        db.create_skill(
            CreateSkill {
                name: "rust".to_owned(),
                description: "Rust workspace maintenance.".to_owned(),
                trigger_rules: vec!["cargo".to_owned()],
                tool_subset: vec!["shell".to_owned()],
                resource_path: Some("skills/rust".to_owned()),
            },
            "test",
        )
        .await?;
        db.create_skill(
            CreateSkill {
                name: "github".to_owned(),
                description: "GitHub issue workflow.".to_owned(),
                trigger_rules: vec!["issue".to_owned()],
                tool_subset: vec!["github".to_owned()],
                resource_path: Some("skills/github".to_owned()),
            },
            "test",
        )
        .await?;
        db.create_task(
            CreateTask {
                title: "Run cargo checks".to_owned(),
                description: "Validate the Rust workspace.".to_owned(),
                task_type: TaskType::OneOff,
                priority: 0,
                requested_skills: vec!["github".to_owned()],
                schedule: None,
                created_by: "test".to_owned(),
            },
            "test",
        )
        .await?;
        let memory_counts = Arc::new(StdMutex::new(Vec::new()));
        let skill_names = Arc::new(StdMutex::new(Vec::new()));
        let allowed_tools = Arc::new(StdMutex::new(Vec::new()));
        let scheduler = Scheduler::new(
            db.clone(),
            ContextRecordingWorker {
                memory_counts,
                skill_names: skill_names.clone(),
                allowed_tools: allowed_tools.clone(),
            },
        );

        scheduler.tick().await?;
        let tasks = db.list_tasks().await?;
        let task = tasks
            .iter()
            .find(|task| task.title == "Run cargo checks")
            .expect("task exists");
        let events = db.list_task_attempt_events(task.id).await?;

        assert_eq!(
            *skill_names.lock().expect("skill names lock"),
            vec![vec!["github".to_owned(), "rust".to_owned()]]
        );
        assert_eq!(
            *allowed_tools.lock().expect("allowed tools lock"),
            vec![vec!["github".to_owned(), "shell".to_owned()]]
        );
        assert!(events.iter().any(|event| {
            event.event_type == "worker_context_prepared"
                && event.details["allowed_tools"] == json!(["github", "shell"])
        }));

        Ok(())
    }

    #[derive(Clone)]
    struct MemoryCandidateWorker;

    #[async_trait]
    impl TaskWorker for MemoryCandidateWorker {
        async fn execute(
            &self,
            _task: Task,
            _context: WorkerContext,
        ) -> anyhow::Result<WorkerResult> {
            Ok(WorkerResult::Completed {
                summary: "finished with reusable lessons".to_owned(),
                memory_candidates: vec![
                    "Prefer cargo test --workspace before committing.".to_owned(),
                    "Avoid broad UI selectors in browser checks.".to_owned(),
                ],
                artifacts: Vec::new(),
                follow_up_tasks: Vec::new(),
            })
        }
    }

    #[tokio::test]
    async fn scheduler_stores_worker_memory_candidates() -> anyhow::Result<()> {
        let db = Db::connect("sqlite::memory:").await?;
        db.create_task(
            CreateTask {
                title: "Capture worker memories".to_owned(),
                description: "Worker should suggest memories".to_owned(),
                task_type: TaskType::OneOff,
                priority: 0,
                requested_skills: Vec::new(),
                schedule: None,
                created_by: "test".to_owned(),
            },
            "test",
        )
        .await?;
        let scheduler = Scheduler::new(db.clone(), MemoryCandidateWorker);

        let tick = scheduler.tick().await?;
        let memories = db.list_memories().await?;

        assert_eq!(memories.len(), 2);
        assert!(matches!(
            tick.outcome,
            SchedulerOutcome::Completed {
                ref memory_candidates,
                ..
            } if memory_candidates.len() == 2
        ));
        assert!(
            memories
                .iter()
                .all(|memory| memory.status == MemoryStatus::Pending)
        );
        assert!(
            memories
                .iter()
                .any(|memory| memory.content.contains("cargo test --workspace"))
        );
        assert!(
            memories
                .iter()
                .any(|memory| memory.content.contains("broad UI selectors"))
        );

        Ok(())
    }

    #[tokio::test]
    async fn scheduler_can_auto_approve_high_confidence_memory_candidates() -> anyhow::Result<()> {
        let db = Db::connect("sqlite::memory:").await?;
        db.create_task(
            CreateTask {
                title: "Capture approved memories".to_owned(),
                description: "Worker should suggest memories".to_owned(),
                task_type: TaskType::OneOff,
                priority: 0,
                requested_skills: Vec::new(),
                schedule: None,
                created_by: "test".to_owned(),
            },
            "test",
        )
        .await?;
        let scheduler = Scheduler::with_policy(
            db.clone(),
            MemoryCandidateWorker,
            SchedulerPolicy::serial().with_memory_auto_approve_confidence(Some(0.6)),
        );

        scheduler.tick().await?;
        let memories = db.list_memories().await?;

        assert_eq!(memories.len(), 2);
        assert!(
            memories
                .iter()
                .all(|memory| memory.status == MemoryStatus::Approved)
        );

        Ok(())
    }

    #[derive(Clone)]
    struct ArtifactWorker;

    #[async_trait]
    impl TaskWorker for ArtifactWorker {
        async fn execute(
            &self,
            _task: Task,
            _context: WorkerContext,
        ) -> anyhow::Result<WorkerResult> {
            Ok(WorkerResult::Completed {
                summary: "created a report".to_owned(),
                memory_candidates: Vec::new(),
                artifacts: vec![WorkerArtifact {
                    name: "report.md".to_owned(),
                    artifact_type: "file".to_owned(),
                    uri: "file://report.md".to_owned(),
                    summary: Some("Run report".to_owned()),
                }],
                follow_up_tasks: Vec::new(),
            })
        }
    }

    #[tokio::test]
    async fn scheduler_stores_worker_artifacts() -> anyhow::Result<()> {
        let db = Db::connect("sqlite::memory:").await?;
        let task = db
            .create_task(
                CreateTask {
                    title: "Capture worker artifacts".to_owned(),
                    description: "Worker should report artifacts".to_owned(),
                    task_type: TaskType::OneOff,
                    priority: 0,
                    requested_skills: Vec::new(),
                    schedule: None,
                    created_by: "test".to_owned(),
                },
                "test",
            )
            .await?;
        let scheduler = Scheduler::new(db.clone(), ArtifactWorker);

        scheduler.tick().await?;
        let artifacts = db.list_task_artifacts(task.id).await?;
        let events = db.list_task_attempt_events(task.id).await?;

        assert_eq!(artifacts.len(), 1);
        assert_eq!(artifacts[0].name, "report.md");
        assert_eq!(artifacts[0].artifact_type, "file");
        assert_eq!(artifacts[0].summary.as_deref(), Some("Run report"));
        assert!(events.iter().any(|event| {
            event.event_type == "worker_completed" && event.details["artifact_count"] == 1
        }));

        Ok(())
    }

    #[derive(Clone)]
    struct FollowUpWorker;

    #[async_trait]
    impl TaskWorker for FollowUpWorker {
        async fn execute(
            &self,
            _task: Task,
            _context: WorkerContext,
        ) -> anyhow::Result<WorkerResult> {
            Ok(WorkerResult::Completed {
                summary: "finished and found next work".to_owned(),
                memory_candidates: Vec::new(),
                artifacts: Vec::new(),
                follow_up_tasks: vec![WorkerFollowUpTask {
                    title: "Run regression tests".to_owned(),
                    description: "Validate the follow-up fix".to_owned(),
                    priority: 5,
                    requested_skills: vec!["testing".to_owned()],
                }],
            })
        }
    }

    #[tokio::test]
    async fn scheduler_creates_worker_follow_up_tasks() -> anyhow::Result<()> {
        let db = Db::connect("sqlite::memory:").await?;
        let task = db
            .create_task(
                CreateTask {
                    title: "Investigate issue".to_owned(),
                    description: "Worker should find a follow-up task".to_owned(),
                    task_type: TaskType::OneOff,
                    priority: 0,
                    requested_skills: Vec::new(),
                    schedule: None,
                    created_by: "test".to_owned(),
                },
                "test",
            )
            .await?;
        let scheduler = Scheduler::new(db.clone(), FollowUpWorker);

        let tick = scheduler.tick().await?;
        let tasks = db.list_tasks().await?;
        let actions = db.list_task_actions(task.id).await?;
        let events = db.list_task_attempt_events(task.id).await?;
        let follow_up = tasks
            .iter()
            .find(|task| task.title == "Run regression tests")
            .expect("follow-up task");

        assert_eq!(follow_up.status, TaskStatus::Queued);
        assert_eq!(follow_up.created_by, "worker");
        assert_eq!(follow_up.requested_skills, vec!["testing".to_owned()]);
        assert!(matches!(
            tick.outcome,
            SchedulerOutcome::Completed {
                ref follow_up_tasks,
                ..
            } if follow_up_tasks.len() == 1
        ));
        assert!(
            actions
                .iter()
                .any(|action| action.action_type == "create_follow_up_task")
        );
        assert!(events.iter().any(|event| {
            event.event_type == "follow_up_tasks_created" && event.details["count"] == 1
        }));

        Ok(())
    }

    #[tokio::test]
    async fn scheduler_records_worker_attempt_events() -> anyhow::Result<()> {
        let db = Db::connect("sqlite::memory:").await?;
        let task = db
            .create_task(
                CreateTask {
                    title: "Observable task".to_owned(),
                    description: "Record worker events".to_owned(),
                    task_type: TaskType::OneOff,
                    priority: 0,
                    requested_skills: Vec::new(),
                    schedule: None,
                    created_by: "test".to_owned(),
                },
                "test",
            )
            .await?;
        let scheduler = Scheduler::new(db.clone(), StubWorker);

        scheduler.tick().await?;

        let events = db.list_task_attempt_events(task.id).await?;
        assert!(
            events
                .iter()
                .any(|event| event.event_type == "worker_context_prepared")
        );
        assert!(
            events
                .iter()
                .any(|event| event.event_type == "worker_completed")
        );

        Ok(())
    }

    #[derive(Clone)]
    struct FailingWorker;

    #[async_trait]
    impl TaskWorker for FailingWorker {
        async fn execute(
            &self,
            _task: Task,
            _context: WorkerContext,
        ) -> anyhow::Result<WorkerResult> {
            anyhow::bail!("simulated worker failure")
        }
    }

    #[tokio::test]
    async fn scheduler_persists_worker_failures() -> anyhow::Result<()> {
        let db = Db::connect("sqlite::memory:").await?;
        let task = db
            .create_task(
                CreateTask {
                    title: "Failure task".to_owned(),
                    description: "Worker will fail".to_owned(),
                    task_type: TaskType::OneOff,
                    priority: 0,
                    requested_skills: Vec::new(),
                    schedule: None,
                    created_by: "test".to_owned(),
                },
                "test",
            )
            .await?;
        let scheduler = Scheduler::new(db.clone(), FailingWorker);

        let tick = scheduler.tick().await?;
        let updated = db.get_task(task.id).await?;
        let events = db.list_task_attempt_events(task.id).await?;

        assert!(matches!(tick.outcome, SchedulerOutcome::Failed { .. }));
        assert_eq!(updated.status, TaskStatus::Failed);
        assert!(
            updated
                .result_summary
                .as_deref()
                .unwrap_or_default()
                .contains("simulated worker failure")
        );
        assert!(
            events
                .iter()
                .any(|event| event.event_type == "worker_failed")
        );

        Ok(())
    }

    #[tokio::test]
    async fn scheduler_retries_worker_failures_until_policy_is_exhausted() -> anyhow::Result<()> {
        let db = Db::connect("sqlite::memory:").await?;
        let task = db
            .create_task(
                CreateTask {
                    title: "Retry failure task".to_owned(),
                    description: "Worker will fail twice".to_owned(),
                    task_type: TaskType::OneOff,
                    priority: 0,
                    requested_skills: Vec::new(),
                    schedule: None,
                    created_by: "test".to_owned(),
                },
                "test",
            )
            .await?;
        let scheduler = Scheduler::with_policy(
            db.clone(),
            FailingWorker,
            SchedulerPolicy::new(1, 60).with_max_attempts(2),
        );

        let first_tick = scheduler.tick().await?;
        let requeued = db.get_task(task.id).await?;
        let actions = db.list_task_actions(task.id).await?;
        let first_events = db.list_task_attempt_events(task.id).await?;

        assert!(matches!(
            first_tick.outcome,
            SchedulerOutcome::RetryScheduled {
                next_attempt: 2,
                max_attempts: 2,
                ..
            }
        ));
        assert_eq!(requeued.status, TaskStatus::Queued);
        assert_eq!(requeued.attempt_count, 1);
        assert!(
            actions
                .iter()
                .any(|action| action.action_type == "requeue_task_after_failure")
        );
        assert!(
            first_events
                .iter()
                .any(|event| event.event_type == "worker_retry_scheduled")
        );

        let second_tick = scheduler.tick().await?;
        let failed = db.get_task(task.id).await?;

        assert!(matches!(
            second_tick.outcome,
            SchedulerOutcome::Failed { .. }
        ));
        assert_eq!(failed.status, TaskStatus::Failed);
        assert_eq!(failed.attempt_count, 2);

        Ok(())
    }
}
