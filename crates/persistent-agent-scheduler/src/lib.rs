use async_trait::async_trait;
use persistent_agent_db::Db;
use persistent_agent_domain::{Task, TaskStatus};
use serde::{Deserialize, Serialize};

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
        let Some(task) = self
            .db
            .claim_next_runnable(&self.lease_owner, self.lease_seconds)
            .await?
        else {
            return Ok(SchedulerTick {
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

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "status")]
pub enum WorkerResult {
    Completed { summary: String },
    Blocked { reason: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulerTick {
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
