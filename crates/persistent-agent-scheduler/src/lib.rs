use async_trait::async_trait;
use llm_adapter::{
    backend::{
        BackendConfig, BackendProtocol, BackendRequestLayer, ReqwestHttpClient, dispatch_request,
    },
    core::{CoreContent, CoreMessage, CoreRequest, CoreResponse, CoreRole},
};
use persistent_agent_db::Db;
use persistent_agent_domain::{Task, TaskStatus};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Clone)]
pub struct Scheduler<W> {
    db: Db,
    worker: W,
    lease_owner: String,
    lease_seconds: i64,
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
            lease_seconds: 300,
        }
    }

    pub async fn tick(&self) -> anyhow::Result<SchedulerTick> {
        let requeued_tasks = self.db.requeue_due_recurring_tasks("scheduler").await?;
        let Some(task) = self
            .db
            .claim_next_runnable(&self.lease_owner, self.lease_seconds)
            .await?
        else {
            return Ok(SchedulerTick {
                requeued_tasks,
                claimed_task: None,
                outcome: SchedulerOutcome::Idle,
            });
        };

        tracing::info!(task_id = %task.id, title = %task.title, "claimed task");
        self.db
            .create_attempt(task.id, TaskStatus::Running, Some("worker started"))
            .await?;

        let result = self.worker.execute(task.clone()).await?;
        match result {
            WorkerResult::Completed { summary } => {
                self.db
                    .create_attempt(task.id, TaskStatus::Completed, Some(&summary))
                    .await?;
                self.db.complete_task(task.id, &summary, "worker").await?;
                Ok(SchedulerTick {
                    requeued_tasks,
                    claimed_task: Some(task),
                    outcome: SchedulerOutcome::Completed { summary },
                })
            }
            WorkerResult::Blocked { reason } => {
                self.db
                    .create_attempt(task.id, TaskStatus::WaitingForUser, Some(&reason))
                    .await?;
                self.db
                    .set_task_status(task.id, TaskStatus::WaitingForUser, "worker", Some(&reason))
                    .await?;
                Ok(SchedulerTick {
                    requeued_tasks,
                    claimed_task: Some(task),
                    outcome: SchedulerOutcome::Blocked { reason },
                })
            }
        }
    }
}

#[async_trait]
pub trait TaskWorker: Clone + Send + Sync + 'static {
    async fn execute(&self, task: Task) -> anyhow::Result<WorkerResult>;
}

#[derive(Debug, Clone)]
pub struct StubWorker;

#[async_trait]
impl TaskWorker for StubWorker {
    async fn execute(&self, task: Task) -> anyhow::Result<WorkerResult> {
        Ok(WorkerResult::Completed {
            summary: format!(
                "Stub worker accepted task '{}' and completed the lifecycle placeholder.",
                task.title
            ),
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
    async fn execute(&self, task: Task) -> anyhow::Result<WorkerResult> {
        match self {
            Self::Stub(worker) => worker.execute(task).await,
            Self::Llm(worker) => worker.execute(task).await,
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
    async fn execute(&self, task: Task) -> anyhow::Result<WorkerResult> {
        let config = self.config.clone();
        let prompt = task_prompt(&task);
        let summary =
            tokio::task::spawn_blocking(move || dispatch_task_prompt(config, prompt)).await??;

        Ok(WorkerResult::Completed { summary })
    }
}

fn dispatch_task_prompt(config: LlmWorkerConfig, prompt: String) -> anyhow::Result<String> {
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
                    text: "You are a worker agent inside Persistent Agent. Execute the assigned task as far as possible and return a concise result summary. If the task cannot truly be completed from the provided context, clearly explain what is missing.".to_owned(),
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
    Ok(extract_response_text(&response))
}

fn task_prompt(task: &Task) -> String {
    format!(
        "Task title: {}\nTask type: {}\nPriority: {}\nRequested skills: {}\n\nTask description:\n{}",
        task.title,
        task.task_type,
        task.priority,
        if task.requested_skills.is_empty() {
            "none".to_owned()
        } else {
            task.requested_skills.join(", ")
        },
        task.description
    )
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

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "status")]
pub enum WorkerResult {
    Completed { summary: String },
    Blocked { reason: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulerTick {
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use persistent_agent_domain::TaskType;
    use uuid::Uuid;

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

        let prompt = task_prompt(&task);

        assert!(prompt.contains("Check issues"));
        assert!(prompt.contains("recurring"));
        assert!(prompt.contains("github"));
    }
}
