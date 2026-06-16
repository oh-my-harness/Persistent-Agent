use async_trait::async_trait;
use llm_adapter::{
    backend::{
        BackendConfig, BackendProtocol, BackendRequestLayer, ReqwestHttpClient, dispatch_request,
    },
    core::{CoreContent, CoreMessage, CoreRequest, CoreResponse, CoreRole},
};
use persistent_agent_db::Db;
use persistent_agent_domain::{
    ConversationMessage, CreateMemory, Memory, MemoryStatus, Skill, Task, TaskNote, TaskStatus,
};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::{collections::BTreeMap, sync::Arc};
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

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub struct SchedulerPolicy {
    pub worker_capacity: usize,
    pub lease_seconds: i64,
}

impl SchedulerPolicy {
    pub fn serial() -> Self {
        Self {
            worker_capacity: 1,
            lease_seconds: 300,
        }
    }

    pub fn new(worker_capacity: usize, lease_seconds: i64) -> Self {
        Self {
            worker_capacity: worker_capacity.max(1),
            lease_seconds: lease_seconds.max(1),
        }
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

        let context = WorkerContext {
            memories: select_relevant_memories(&task, self.db.list_approved_memories(20).await?, 5),
            skills: self
                .db
                .list_skills_by_names(&active_skill_names(&task))
                .await?,
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
                for candidate in memory_candidate_contents(&task, &summary, &memory_candidates) {
                    self.db
                        .create_memory(
                            CreateMemory {
                                scope: "task".to_owned(),
                                content: candidate,
                                source_task_id: Some(task.id),
                                status: MemoryStatus::Pending,
                                confidence: 0.6,
                            },
                            "worker",
                        )
                        .await?;
                }
                self.db.complete_task(task.id, &summary, "worker").await?;
                Ok(SchedulerTick {
                    recovered_tasks,
                    requeued_tasks,
                    claimed_task: Some(task),
                    outcome: SchedulerOutcome::Completed { summary },
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

    async fn execute_worker_with_heartbeat(
        &self,
        task: Task,
        context: WorkerContext,
        attempt_id: persistent_agent_domain::TaskAttemptId,
    ) -> anyhow::Result<WorkerResult> {
        let task_id = task.id;
        let worker = self.worker.clone();
        let worker_task = tokio::spawn(async move { worker.execute(task, context).await });
        tokio::pin!(worker_task);

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
                        Ok(None) => {}
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
        })
    }
}

#[derive(Debug, Clone)]
pub enum WorkerBackend {
    Stub(StubWorker),
    Llm(LlmWorker),
}

#[async_trait]
impl TaskWorker for WorkerBackend {
    async fn execute(&self, task: Task, context: WorkerContext) -> anyhow::Result<WorkerResult> {
        match self {
            Self::Stub(worker) => worker.execute(task, context).await,
            Self::Llm(worker) => worker.execute(task, context).await,
        }
    }
}

#[derive(Debug, Clone)]
pub struct LlmWorker {
    config: LlmWorkerConfig,
}

impl LlmWorker {
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
impl TaskWorker for LlmWorker {
    async fn execute(&self, task: Task, context: WorkerContext) -> anyhow::Result<WorkerResult> {
        let config = self.config.clone();
        let prompt = task_prompt(
            &task,
            &context.memories,
            &context.skills,
            &context.notes,
            &context.conversation_messages,
        );
        let result =
            tokio::task::spawn_blocking(move || dispatch_task_prompt(config, prompt)).await??;

        Ok(result)
    }
}

fn dispatch_task_prompt(config: LlmWorkerConfig, prompt: String) -> anyhow::Result<WorkerResult> {
    let backend_config = BackendConfig {
        base_url: config.base_url,
        auth_token: config.api_key,
        request_layer: Some(BackendRequestLayer::ChatCompletions),
        headers: BTreeMap::new(),
        no_streaming: true,
        timeout_ms: Some(config.timeout_ms),
    };
    let request = CoreRequest {
        model: config.model,
        messages: vec![
            CoreMessage {
                role: CoreRole::System,
                content: vec![CoreContent::Text {
                    text: "You are a worker agent inside Persistent Agent. Execute the assigned task as far as possible. Return only JSON with this shape: {\"status\":\"completed\",\"summary\":\"...\",\"memory_candidates\":[\"...\"],\"artifacts\":[{\"name\":\"...\",\"artifact_type\":\"file|url|note\",\"uri\":\"...\",\"summary\":\"...\"}]} or {\"status\":\"blocked\",\"reason\":\"...\"}. Use blocked when you cannot truly complete the task from the provided context and need user input. Put only durable preferences, pitfalls, project conventions, or reusable task learnings in memory_candidates. Put durable outputs or references in artifacts.".to_owned(),
                }],
            },
            CoreMessage {
                role: CoreRole::User,
                content: vec![CoreContent::Text { text: prompt }],
            },
        ],
        stream: false,
        max_tokens: Some(700),
        temperature: Some(0.2),
        tools: Vec::new(),
        tool_choice: None,
        include: None,
        reasoning: None,
        response_schema: None,
    };

    let client = ReqwestHttpClient::default();
    let response = dispatch_request(
        &client,
        &backend_config,
        BackendProtocol::OpenaiChatCompletions,
        &request,
    )?;
    Ok(parse_worker_result_text(&extract_response_text(&response)))
}

fn task_prompt(
    task: &Task,
    memories: &[Memory],
    skills: &[Skill],
    notes: &[TaskNote],
    conversation_messages: &[ConversationMessage],
) -> String {
    format!(
        "Task title: {}\nTask type: {}\nPriority: {}\nRequested skills: {}\nMatched skills: {}\nActive skills: {}\n\nActive skill resources:\n{}\n\nRelevant approved memories:\n{}\n\nTask notes:\n{}\n\nRecent task conversation:\n{}\n\nTask description:\n{}",
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

fn extract_response_text(response: &CoreResponse) -> String {
    let text = response
        .message
        .content
        .iter()
        .filter_map(|content| match content {
            CoreContent::Text { text } => Some(text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
        .trim()
        .to_owned();

    if text.is_empty() {
        format!(
            "LLM worker returned no text. Finish reason: {}. Usage: {} total tokens.",
            response.finish_reason, response.usage.total_tokens
        )
    } else {
        text
    }
}

#[derive(Debug, Deserialize)]
struct StructuredWorkerResult {
    status: String,
    summary: Option<String>,
    reason: Option<String>,
    memory_candidates: Option<Vec<String>>,
    artifacts: Option<Vec<WorkerArtifact>>,
}

fn parse_worker_result_text(text: &str) -> WorkerResult {
    let trimmed = trim_json_code_fence(text.trim());
    if let Ok(parsed) = serde_json::from_str::<StructuredWorkerResult>(trimmed) {
        return match parsed.status.to_lowercase().as_str() {
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
            },
            _ => fallback_worker_result(text),
        };
    }

    fallback_worker_result(text)
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
    Completed { summary: String },
    Blocked { reason: String },
    Failed { error: String },
    Superseded { status: TaskStatus, reason: String },
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
        assert_eq!(
            SchedulerPolicy::new(0, 0),
            SchedulerPolicy {
                worker_capacity: 1,
                lease_seconds: 1,
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
            }
        );

        Ok(())
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
    fn extracts_text_from_core_response() {
        let response = CoreResponse {
            id: "r1".to_owned(),
            model: "deepseek-chat".to_owned(),
            message: CoreMessage {
                role: CoreRole::Assistant,
                content: vec![CoreContent::Text {
                    text: "done".to_owned(),
                }],
            },
            usage: Default::default(),
            finish_reason: "stop".to_owned(),
            reasoning_details: None,
        };

        assert_eq!(extract_response_text(&response), "done");
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

        let prompt = task_prompt(&task, &[], &[skill], &[], &[]);

        assert!(prompt.contains("Check issues"));
        assert!(prompt.contains("recurring"));
        assert!(prompt.contains("github"));
        assert!(prompt.contains("Matched skills: rust, github"));
        assert!(prompt.contains("Active skills: github, rust"));
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
        let prompt = task_prompt(&task, &memories, &[], &[], &[]);

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

        let prompt = task_prompt(&task, &[], &[], &[], &messages);

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

        let prompt = task_prompt(&task, &[], &[], &notes, &[]);

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
    struct ContextRecordingWorker {
        memory_counts: Arc<StdMutex<Vec<usize>>>,
        skill_names: Arc<StdMutex<Vec<Vec<String>>>>,
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

            Ok(WorkerResult::Completed {
                summary: format!("completed {}", task.title),
                memory_candidates: Vec::new(),
                artifacts: Vec::new(),
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
        let scheduler = Scheduler::new(
            db,
            ContextRecordingWorker {
                memory_counts: memory_counts.clone(),
                skill_names,
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
        let scheduler = Scheduler::new(
            db,
            ContextRecordingWorker {
                memory_counts,
                skill_names: skill_names.clone(),
            },
        );

        scheduler.tick().await?;

        assert_eq!(
            *skill_names.lock().expect("skill names lock"),
            vec![vec!["github".to_owned(), "rust".to_owned()]]
        );

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

        scheduler.tick().await?;
        let memories = db.list_memories().await?;

        assert_eq!(memories.len(), 2);
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
}
