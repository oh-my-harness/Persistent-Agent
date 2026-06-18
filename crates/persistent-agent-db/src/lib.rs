use chrono::{DateTime, Duration, Utc};
use persistent_agent_domain::{
    Conversation, ConversationId, ConversationMessage, CreateMemory, CreateSkill, CreateTask,
    Memory, MemoryId, MemoryStatus, Skill, SkillId, Task, TaskAction, TaskArtifact, TaskAttempt,
    TaskAttemptEvent, TaskAttemptId, TaskDependency, TaskId, TaskNote, TaskResourceLock,
    TaskStatus, TaskType, UpdateMemory, UpdateSkill, UpdateTask,
};
use serde_json::{Value, json};
use sqlx::{Row, SqlitePool, sqlite::SqlitePoolOptions};
use uuid::Uuid;

#[derive(Clone)]
pub struct Db {
    pool: SqlitePool,
}

impl Db {
    pub async fn connect(database_url: &str) -> anyhow::Result<Self> {
        let pool = SqlitePoolOptions::new()
            .max_connections(8)
            .connect(database_url)
            .await?;
        sqlx::query("PRAGMA foreign_keys = ON")
            .execute(&pool)
            .await?;
        sqlx::migrate!("./migrations").run(&pool).await?;
        Ok(Self { pool })
    }

    pub fn pool(&self) -> &SqlitePool {
        &self.pool
    }

    pub async fn create_task(&self, input: CreateTask, actor: &str) -> anyhow::Result<Task> {
        let now = Utc::now();
        let id = Uuid::now_v7();
        let conversation_id = Uuid::now_v7();
        let queue_position = self.next_queue_position().await?;
        let requested_skills = serde_json::to_string(&input.requested_skills)?;
        let matched_skill_names = self
            .match_skills(&input.title, &input.description, input.task_type)
            .await?;
        let matched_skills = serde_json::to_string(&matched_skill_names)?;
        let schedule = input
            .schedule
            .as_ref()
            .map(serde_json::to_string)
            .transpose()?;

        sqlx::query(
            r#"
            INSERT INTO tasks (
              id, title, description, task_type, status, priority, queue_position, created_by,
              conversation_id, requested_skills, matched_skills, schedule, attempt_count,
              created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?)
            "#,
        )
        .bind(id.to_string())
        .bind(&input.title)
        .bind(&input.description)
        .bind(input.task_type.to_string())
        .bind(TaskStatus::Queued.to_string())
        .bind(input.priority)
        .bind(queue_position)
        .bind(&input.created_by)
        .bind(conversation_id.to_string())
        .bind(requested_skills)
        .bind(matched_skills)
        .bind(schedule)
        .bind(now)
        .bind(now)
        .execute(&self.pool)
        .await?;

        sqlx::query(
            "INSERT INTO conversations (id, task_id, title, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
        )
        .bind(conversation_id.to_string())
        .bind(id.to_string())
        .bind(&input.title)
        .bind(now)
        .bind(now)
        .execute(&self.pool)
        .await?;

        self.record_action(
            Some(id),
            actor,
            "create_task",
            json!({ "title": input.title, "task_type": input.task_type }),
        )
        .await?;

        self.get_task(id).await
    }

    pub async fn list_tasks(&self) -> anyhow::Result<Vec<Task>> {
        let rows = sqlx::query(
            r#"
            SELECT * FROM tasks
            ORDER BY
              CASE status
                WHEN 'running' THEN 0
                WHEN 'waiting_for_user' THEN 1
                WHEN 'queued' THEN 2
                WHEN 'waiting_for_schedule' THEN 3
                ELSE 4
              END,
              priority DESC,
              queue_position ASC,
              created_at ASC
            "#,
        )
        .fetch_all(&self.pool)
        .await?;

        rows.into_iter().map(row_to_task).collect()
    }

    pub async fn get_task(&self, id: TaskId) -> anyhow::Result<Task> {
        let row = sqlx::query("SELECT * FROM tasks WHERE id = ?")
            .bind(id.to_string())
            .fetch_one(&self.pool)
            .await?;
        row_to_task(row)
    }

    pub async fn update_task(
        &self,
        id: TaskId,
        input: UpdateTask,
        actor: &str,
    ) -> anyhow::Result<Task> {
        let current = self.get_task(id).await?;
        let title = input.title.unwrap_or(current.title);
        let description = input.description.unwrap_or(current.description);
        let priority = input.priority.unwrap_or(current.priority);
        let requested_skills = input.requested_skills.unwrap_or(current.requested_skills);
        let matched_skills = self
            .match_skills(&title, &description, current.task_type)
            .await?;
        let schedule = input.schedule.or(current.schedule);
        let now = Utc::now();

        sqlx::query(
            r#"
            UPDATE tasks
            SET title = ?, description = ?, priority = ?, requested_skills = ?, matched_skills = ?, schedule = ?, updated_at = ?
            WHERE id = ?
            "#,
        )
        .bind(&title)
        .bind(&description)
        .bind(priority)
        .bind(serde_json::to_string(&requested_skills)?)
        .bind(serde_json::to_string(&matched_skills)?)
        .bind(schedule.as_ref().map(serde_json::to_string).transpose()?)
        .bind(now)
        .bind(id.to_string())
        .execute(&self.pool)
        .await?;

        self.record_action(
            Some(id),
            actor,
            "update_task",
            json!({ "title": title, "priority": priority, "requested_skills": requested_skills, "matched_skills": matched_skills }),
        )
        .await?;

        self.get_task(id).await
    }

    pub async fn delete_task(&self, id: TaskId, actor: &str) -> anyhow::Result<Task> {
        let task = self.get_task(id).await?;
        sqlx::query("DELETE FROM tasks WHERE id = ?")
            .bind(id.to_string())
            .execute(&self.pool)
            .await?;

        self.record_action(
            None,
            actor,
            "delete_task",
            json!({
                "task_id": id,
                "title": task.title,
                "status": task.status,
                "task_type": task.task_type
            }),
        )
        .await?;

        Ok(task)
    }

    pub async fn update_task_schedule(
        &self,
        id: TaskId,
        schedule: Value,
        actor: &str,
    ) -> anyhow::Result<Task> {
        let current = self.get_task(id).await?;
        let now = Utc::now();
        let next_run_at = if current.task_type == TaskType::Recurring
            && current.status == TaskStatus::WaitingForSchedule
        {
            Some(now + Duration::seconds(recurring_interval_seconds(Some(&schedule))))
        } else {
            current.next_run_at
        };

        sqlx::query(
            r#"
            UPDATE tasks
            SET schedule = ?, next_run_at = ?, updated_at = ?
            WHERE id = ?
            "#,
        )
        .bind(serde_json::to_string(&schedule)?)
        .bind(next_run_at)
        .bind(now)
        .bind(id.to_string())
        .execute(&self.pool)
        .await?;

        self.record_action(
            Some(id),
            actor,
            "update_task_schedule",
            json!({ "schedule": schedule, "next_run_at": next_run_at }),
        )
        .await?;

        self.get_task(id).await
    }

    pub async fn set_task_status(
        &self,
        id: TaskId,
        status: TaskStatus,
        actor: &str,
        reason: Option<&str>,
    ) -> anyhow::Result<Task> {
        let now = Utc::now();
        sqlx::query(
            r#"
            UPDATE tasks
            SET status = ?, blocked_reason = ?, lease_owner = NULL, lease_expires_at = NULL,
                updated_at = ?
            WHERE id = ?
            "#,
        )
        .bind(status.to_string())
        .bind(reason)
        .bind(now)
        .bind(id.to_string())
        .execute(&self.pool)
        .await?;

        self.record_action(
            Some(id),
            actor,
            "set_task_status",
            json!({ "status": status, "reason": reason }),
        )
        .await?;

        self.get_task(id).await
    }

    pub async fn reorder_task(
        &self,
        id: TaskId,
        queue_position: i64,
        actor: &str,
    ) -> anyhow::Result<Task> {
        let now = Utc::now();
        sqlx::query("UPDATE tasks SET queue_position = ?, updated_at = ? WHERE id = ?")
            .bind(queue_position)
            .bind(now)
            .bind(id.to_string())
            .execute(&self.pool)
            .await?;
        self.record_action(
            Some(id),
            actor,
            "reorder_task",
            json!({ "queue_position": queue_position }),
        )
        .await?;
        self.get_task(id).await
    }

    pub async fn convert_task_type(
        &self,
        id: TaskId,
        task_type: TaskType,
        schedule: Option<Value>,
        actor: &str,
    ) -> anyhow::Result<Task> {
        let current = self.get_task(id).await?;
        let now = Utc::now();
        let schedule = match task_type {
            TaskType::OneOff => None,
            TaskType::Recurring => {
                Some(schedule.unwrap_or_else(|| json!({ "interval_seconds": 300 })))
            }
        };
        let next_run_at = match task_type {
            TaskType::OneOff => None,
            TaskType::Recurring => {
                if matches!(
                    current.status,
                    TaskStatus::Completed | TaskStatus::WaitingForSchedule
                ) {
                    Some(now + Duration::seconds(recurring_interval_seconds(schedule.as_ref())))
                } else {
                    current.next_run_at
                }
            }
        };
        let status = match (task_type, current.status) {
            (TaskType::OneOff, TaskStatus::WaitingForSchedule) => TaskStatus::Completed,
            (TaskType::Recurring, TaskStatus::Completed) => TaskStatus::WaitingForSchedule,
            (_, status) => status,
        };
        let matched_skills = self
            .match_skills(&current.title, &current.description, task_type)
            .await?;

        sqlx::query(
            r#"
            UPDATE tasks
            SET task_type = ?, status = ?, matched_skills = ?, schedule = ?, next_run_at = ?, updated_at = ?
            WHERE id = ?
            "#,
        )
        .bind(task_type.to_string())
        .bind(status.to_string())
        .bind(serde_json::to_string(&matched_skills)?)
        .bind(schedule.as_ref().map(serde_json::to_string).transpose()?)
        .bind(next_run_at)
        .bind(now)
        .bind(id.to_string())
        .execute(&self.pool)
        .await?;

        self.record_action(
            Some(id),
            actor,
            "convert_task_type",
            json!({ "task_type": task_type, "schedule": schedule, "matched_skills": matched_skills }),
        )
        .await?;

        self.get_task(id).await
    }

    pub async fn claim_next_runnable(
        &self,
        lease_owner: &str,
        lease_seconds: i64,
    ) -> anyhow::Result<Option<Task>> {
        let now = Utc::now();
        let lease_expires_at = now + Duration::seconds(lease_seconds);
        let mut tx = self.pool.begin().await?;

        let row = sqlx::query(
            r#"
            SELECT * FROM tasks
            WHERE status = 'queued'
              AND (next_run_at IS NULL OR next_run_at <= ?)
              AND (lease_expires_at IS NULL OR lease_expires_at <= ?)
              AND NOT EXISTS (
                SELECT 1
                FROM task_dependencies
                JOIN tasks dependency_tasks ON dependency_tasks.id = task_dependencies.depends_on_task_id
                WHERE task_dependencies.task_id = tasks.id
                  AND dependency_tasks.status NOT IN ('completed', 'waiting_for_schedule')
              )
              AND NOT EXISTS (
                SELECT 1
                FROM task_resource_locks candidate_locks
                JOIN task_resource_locks running_locks
                  ON running_locks.resource_key = candidate_locks.resource_key
                 AND running_locks.lock_mode = 'exclusive'
                JOIN tasks running_tasks ON running_tasks.id = running_locks.task_id
                WHERE candidate_locks.task_id = tasks.id
                  AND candidate_locks.lock_mode = 'exclusive'
                  AND running_tasks.status = 'running'
                  AND running_tasks.id <> tasks.id
                  AND (running_tasks.lease_expires_at IS NULL OR running_tasks.lease_expires_at > ?)
              )
            ORDER BY priority DESC, queue_position ASC, created_at ASC
            LIMIT 1
            "#,
        )
        .bind(now)
        .bind(now)
        .bind(now)
        .fetch_optional(&mut *tx)
        .await?;

        let Some(row) = row else {
            tx.commit().await?;
            return Ok(None);
        };

        let task = row_to_task(row)?;
        sqlx::query(
            r#"
            UPDATE tasks
            SET status = 'running', lease_owner = ?, lease_expires_at = ?, attempt_count = attempt_count + 1,
                last_run_at = ?, updated_at = ?
            WHERE id = ? AND status = 'queued'
            "#,
        )
        .bind(lease_owner)
        .bind(lease_expires_at)
        .bind(now)
        .bind(now)
        .bind(task.id.to_string())
        .execute(&mut *tx)
        .await?;
        tx.commit().await?;

        self.record_action(
            Some(task.id),
            "scheduler",
            "claim_task",
            json!({ "lease_owner": lease_owner, "lease_expires_at": lease_expires_at }),
        )
        .await?;

        self.get_task(task.id).await.map(Some)
    }

    pub async fn peek_next_runnable(&self) -> anyhow::Result<Option<Task>> {
        let now = Utc::now();
        let row = sqlx::query(
            r#"
            SELECT * FROM tasks
            WHERE status = 'queued'
              AND (next_run_at IS NULL OR next_run_at <= ?)
              AND (lease_expires_at IS NULL OR lease_expires_at <= ?)
              AND NOT EXISTS (
                SELECT 1
                FROM task_dependencies
                JOIN tasks dependency_tasks ON dependency_tasks.id = task_dependencies.depends_on_task_id
                WHERE task_dependencies.task_id = tasks.id
                  AND dependency_tasks.status NOT IN ('completed', 'waiting_for_schedule')
              )
              AND NOT EXISTS (
                SELECT 1
                FROM task_resource_locks candidate_locks
                JOIN task_resource_locks running_locks
                  ON running_locks.resource_key = candidate_locks.resource_key
                 AND running_locks.lock_mode = 'exclusive'
                JOIN tasks running_tasks ON running_tasks.id = running_locks.task_id
                WHERE candidate_locks.task_id = tasks.id
                  AND candidate_locks.lock_mode = 'exclusive'
                  AND running_tasks.status = 'running'
                  AND running_tasks.id <> tasks.id
                  AND (running_tasks.lease_expires_at IS NULL OR running_tasks.lease_expires_at > ?)
              )
            ORDER BY priority DESC, queue_position ASC, created_at ASC
            LIMIT 1
            "#,
        )
        .bind(now)
        .bind(now)
        .bind(now)
        .fetch_optional(&self.pool)
        .await?;

        row.map(row_to_task).transpose()
    }

    pub async fn heartbeat_task_lease(
        &self,
        id: TaskId,
        lease_owner: &str,
        lease_seconds: i64,
    ) -> anyhow::Result<Option<DateTime<Utc>>> {
        let now = Utc::now();
        let lease_expires_at = now + Duration::seconds(lease_seconds.max(1));
        let result = sqlx::query(
            r#"
            UPDATE tasks
            SET lease_expires_at = ?, updated_at = ?
            WHERE id = ? AND status = 'running' AND lease_owner = ?
            "#,
        )
        .bind(lease_expires_at)
        .bind(now)
        .bind(id.to_string())
        .bind(lease_owner)
        .execute(&self.pool)
        .await?;

        if result.rows_affected() == 0 {
            Ok(None)
        } else {
            Ok(Some(lease_expires_at))
        }
    }

    pub async fn recover_expired_running_tasks(&self, actor: &str) -> anyhow::Result<Vec<Task>> {
        let now = Utc::now();
        let rows = sqlx::query(
            r#"
            SELECT id, lease_owner, lease_expires_at FROM tasks
            WHERE status = 'running'
              AND lease_expires_at IS NOT NULL
              AND lease_expires_at <= ?
            ORDER BY lease_expires_at ASC, priority DESC, queue_position ASC
            "#,
        )
        .bind(now)
        .fetch_all(&self.pool)
        .await?;

        let mut recovered = Vec::new();
        for row in rows {
            let task_id = parse_uuid(row.try_get::<String, _>("id")?)?;
            let previous_owner: Option<String> = row.try_get("lease_owner")?;
            let previous_lease_expires_at: Option<DateTime<Utc>> =
                row.try_get("lease_expires_at")?;
            let queue_position = self.next_queue_position().await?;

            let result = sqlx::query(
                r#"
                UPDATE tasks
                SET status = 'queued', queue_position = ?, blocked_reason = NULL,
                    lease_owner = NULL, lease_expires_at = NULL, updated_at = ?
                WHERE id = ? AND status = 'running'
                  AND lease_expires_at IS NOT NULL
                  AND lease_expires_at <= ?
                "#,
            )
            .bind(queue_position)
            .bind(now)
            .bind(task_id.to_string())
            .bind(now)
            .execute(&self.pool)
            .await?;

            if result.rows_affected() == 0 {
                continue;
            }

            self.record_action(
                Some(task_id),
                actor,
                "recover_expired_running_task",
                json!({
                    "queue_position": queue_position,
                    "previous_lease_owner": previous_owner,
                    "previous_lease_expires_at": previous_lease_expires_at,
                }),
            )
            .await?;

            recovered.push(self.get_task(task_id).await?);
        }

        Ok(recovered)
    }

    pub async fn requeue_due_recurring_tasks(&self, actor: &str) -> anyhow::Result<Vec<Task>> {
        let now = Utc::now();
        let rows = sqlx::query(
            r#"
            SELECT id FROM tasks
            WHERE task_type = 'recurring'
              AND status = 'waiting_for_schedule'
              AND (next_run_at IS NULL OR next_run_at <= ?)
            ORDER BY next_run_at ASC, priority DESC, queue_position ASC
            "#,
        )
        .bind(now)
        .fetch_all(&self.pool)
        .await?;

        let mut requeued = Vec::new();
        for row in rows {
            let task_id = parse_uuid(row.try_get::<String, _>("id")?)?;
            let queue_position = self.next_queue_position().await?;
            sqlx::query(
                r#"
                UPDATE tasks
                SET status = 'queued', queue_position = ?, next_run_at = NULL,
                    lease_owner = NULL, lease_expires_at = NULL, updated_at = ?
                WHERE id = ? AND status = 'waiting_for_schedule'
                "#,
            )
            .bind(queue_position)
            .bind(now)
            .bind(task_id.to_string())
            .execute(&self.pool)
            .await?;

            self.record_action(
                Some(task_id),
                actor,
                "requeue_recurring_task",
                json!({ "queue_position": queue_position }),
            )
            .await?;

            requeued.push(self.get_task(task_id).await?);
        }

        Ok(requeued)
    }

    pub async fn complete_task(
        &self,
        id: TaskId,
        summary: &str,
        actor: &str,
    ) -> anyhow::Result<Task> {
        let now = Utc::now();
        let task = self.get_task(id).await?;
        let next_status = match task.task_type {
            TaskType::OneOff => TaskStatus::Completed,
            TaskType::Recurring => TaskStatus::WaitingForSchedule,
        };
        let next_run_at = match task.task_type {
            TaskType::OneOff => None,
            TaskType::Recurring => {
                Some(now + Duration::seconds(recurring_interval_seconds(task.schedule.as_ref())))
            }
        };

        sqlx::query(
            r#"
            UPDATE tasks
            SET status = ?, result_summary = ?, next_run_at = ?,
                lease_owner = NULL, lease_expires_at = NULL, updated_at = ?
            WHERE id = ?
            "#,
        )
        .bind(next_status.to_string())
        .bind(summary)
        .bind(next_run_at)
        .bind(now)
        .bind(id.to_string())
        .execute(&self.pool)
        .await?;

        self.record_action(
            Some(id),
            actor,
            "complete_task",
            json!({ "summary": summary, "next_status": next_status }),
        )
        .await?;

        self.get_task(id).await
    }

    pub async fn fail_task(&self, id: TaskId, error: &str, actor: &str) -> anyhow::Result<Task> {
        let now = Utc::now();
        sqlx::query(
            r#"
            UPDATE tasks
            SET status = ?, result_summary = ?, blocked_reason = NULL,
                lease_owner = NULL, lease_expires_at = NULL, updated_at = ?
            WHERE id = ?
            "#,
        )
        .bind(TaskStatus::Failed.to_string())
        .bind(error)
        .bind(now)
        .bind(id.to_string())
        .execute(&self.pool)
        .await?;

        self.record_action(Some(id), actor, "fail_task", json!({ "error": error }))
            .await?;

        self.get_task(id).await
    }

    pub async fn requeue_task_after_failure(
        &self,
        id: TaskId,
        error: &str,
        actor: &str,
    ) -> anyhow::Result<Task> {
        let now = Utc::now();
        let queue_position = self.next_queue_position().await?;
        sqlx::query(
            r#"
            UPDATE tasks
            SET status = ?, queue_position = ?, result_summary = ?, blocked_reason = NULL,
                lease_owner = NULL, lease_expires_at = NULL, updated_at = ?
            WHERE id = ?
            "#,
        )
        .bind(TaskStatus::Queued.to_string())
        .bind(queue_position)
        .bind(error)
        .bind(now)
        .bind(id.to_string())
        .execute(&self.pool)
        .await?;

        self.record_action(
            Some(id),
            actor,
            "requeue_task_after_failure",
            json!({ "error": error, "queue_position": queue_position }),
        )
        .await?;

        self.get_task(id).await
    }

    pub async fn create_attempt(
        &self,
        task_id: TaskId,
        status: TaskStatus,
        summary: Option<&str>,
    ) -> anyhow::Result<TaskAttempt> {
        let id = Uuid::now_v7();
        let now = Utc::now();
        sqlx::query(
            "INSERT INTO task_attempts (id, task_id, status, summary, started_at, finished_at) VALUES (?, ?, ?, ?, ?, ?)",
        )
        .bind(id.to_string())
        .bind(task_id.to_string())
        .bind(status.to_string())
        .bind(summary)
        .bind(now)
        .bind(if matches!(status, TaskStatus::Running) { None } else { Some(now) })
        .execute(&self.pool)
        .await?;

        Ok(TaskAttempt {
            id,
            task_id,
            status,
            summary: summary.map(ToOwned::to_owned),
            started_at: now,
            finished_at: if matches!(status, TaskStatus::Running) {
                None
            } else {
                Some(now)
            },
        })
    }

    pub async fn list_task_attempts(&self, task_id: TaskId) -> anyhow::Result<Vec<TaskAttempt>> {
        let rows = sqlx::query(
            r#"
            SELECT * FROM task_attempts
            WHERE task_id = ?
            ORDER BY started_at ASC
            "#,
        )
        .bind(task_id.to_string())
        .fetch_all(&self.pool)
        .await?;

        rows.into_iter().map(row_to_task_attempt).collect()
    }

    pub async fn record_attempt_event(
        &self,
        attempt_id: TaskAttemptId,
        task_id: TaskId,
        event_type: &str,
        message: &str,
        details: serde_json::Value,
    ) -> anyhow::Result<TaskAttemptEvent> {
        let event = TaskAttemptEvent {
            id: Uuid::now_v7(),
            attempt_id,
            task_id,
            event_type: event_type.to_owned(),
            message: message.to_owned(),
            details,
            created_at: Utc::now(),
        };

        sqlx::query(
            "INSERT INTO task_attempt_events (id, attempt_id, task_id, event_type, message, details, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
        )
        .bind(event.id.to_string())
        .bind(event.attempt_id.to_string())
        .bind(event.task_id.to_string())
        .bind(&event.event_type)
        .bind(&event.message)
        .bind(serde_json::to_string(&event.details)?)
        .bind(event.created_at)
        .execute(&self.pool)
        .await?;

        Ok(event)
    }

    pub async fn list_task_attempt_events(
        &self,
        task_id: TaskId,
    ) -> anyhow::Result<Vec<TaskAttemptEvent>> {
        let rows = sqlx::query(
            r#"
            SELECT * FROM task_attempt_events
            WHERE task_id = ?
            ORDER BY created_at ASC
            "#,
        )
        .bind(task_id.to_string())
        .fetch_all(&self.pool)
        .await?;

        rows.into_iter().map(row_to_task_attempt_event).collect()
    }

    pub async fn list_attempt_events(
        &self,
        attempt_id: TaskAttemptId,
    ) -> anyhow::Result<Vec<TaskAttemptEvent>> {
        let rows = sqlx::query(
            r#"
            SELECT * FROM task_attempt_events
            WHERE attempt_id = ?
            ORDER BY created_at ASC
            "#,
        )
        .bind(attempt_id.to_string())
        .fetch_all(&self.pool)
        .await?;

        rows.into_iter().map(row_to_task_attempt_event).collect()
    }

    pub async fn record_task_artifact(
        &self,
        task_id: TaskId,
        attempt_id: Option<TaskAttemptId>,
        name: &str,
        artifact_type: &str,
        uri: &str,
        summary: Option<&str>,
    ) -> anyhow::Result<TaskArtifact> {
        let artifact = TaskArtifact {
            id: Uuid::now_v7(),
            task_id,
            attempt_id,
            name: name.to_owned(),
            artifact_type: artifact_type.to_owned(),
            uri: uri.to_owned(),
            summary: summary.map(ToOwned::to_owned),
            created_at: Utc::now(),
        };

        sqlx::query(
            "INSERT INTO task_artifacts (id, task_id, attempt_id, name, artifact_type, uri, summary, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        )
        .bind(artifact.id.to_string())
        .bind(artifact.task_id.to_string())
        .bind(artifact.attempt_id.map(|id| id.to_string()))
        .bind(&artifact.name)
        .bind(&artifact.artifact_type)
        .bind(&artifact.uri)
        .bind(&artifact.summary)
        .bind(artifact.created_at)
        .execute(&self.pool)
        .await?;

        Ok(artifact)
    }

    pub async fn list_task_artifacts(&self, task_id: TaskId) -> anyhow::Result<Vec<TaskArtifact>> {
        let rows = sqlx::query(
            r#"
            SELECT * FROM task_artifacts
            WHERE task_id = ?
            ORDER BY created_at ASC
            "#,
        )
        .bind(task_id.to_string())
        .fetch_all(&self.pool)
        .await?;

        rows.into_iter().map(row_to_task_artifact).collect()
    }

    pub async fn add_task_dependency(
        &self,
        task_id: TaskId,
        depends_on_task_id: TaskId,
        actor: &str,
    ) -> anyhow::Result<TaskDependency> {
        if task_id == depends_on_task_id {
            anyhow::bail!("task cannot depend on itself");
        }

        self.get_task(task_id).await?;
        self.get_task(depends_on_task_id).await?;

        if self
            .dependency_would_create_cycle(task_id, depends_on_task_id)
            .await?
        {
            anyhow::bail!("task dependency would create a cycle");
        }

        let dependency = TaskDependency {
            task_id,
            depends_on_task_id,
            created_at: Utc::now(),
        };

        sqlx::query(
            "INSERT OR IGNORE INTO task_dependencies (task_id, depends_on_task_id, created_at) VALUES (?, ?, ?)",
        )
        .bind(dependency.task_id.to_string())
        .bind(dependency.depends_on_task_id.to_string())
        .bind(dependency.created_at)
        .execute(&self.pool)
        .await?;

        self.record_action(
            Some(task_id),
            actor,
            "add_task_dependency",
            json!({ "depends_on_task_id": depends_on_task_id }),
        )
        .await?;

        self.get_task_dependency(task_id, depends_on_task_id).await
    }

    pub async fn remove_task_dependency(
        &self,
        task_id: TaskId,
        depends_on_task_id: TaskId,
        actor: &str,
    ) -> anyhow::Result<TaskDependency> {
        let dependency = self
            .get_task_dependency(task_id, depends_on_task_id)
            .await?;

        sqlx::query("DELETE FROM task_dependencies WHERE task_id = ? AND depends_on_task_id = ?")
            .bind(task_id.to_string())
            .bind(depends_on_task_id.to_string())
            .execute(&self.pool)
            .await?;

        self.record_action(
            Some(task_id),
            actor,
            "remove_task_dependency",
            json!({ "depends_on_task_id": depends_on_task_id }),
        )
        .await?;

        Ok(dependency)
    }

    pub async fn get_task_dependency(
        &self,
        task_id: TaskId,
        depends_on_task_id: TaskId,
    ) -> anyhow::Result<TaskDependency> {
        let row = sqlx::query(
            "SELECT * FROM task_dependencies WHERE task_id = ? AND depends_on_task_id = ?",
        )
        .bind(task_id.to_string())
        .bind(depends_on_task_id.to_string())
        .fetch_one(&self.pool)
        .await?;

        row_to_task_dependency(row)
    }

    pub async fn list_task_dependencies(
        &self,
        task_id: TaskId,
    ) -> anyhow::Result<Vec<TaskDependency>> {
        let rows = sqlx::query(
            r#"
            SELECT * FROM task_dependencies
            WHERE task_id = ?
            ORDER BY created_at ASC
            "#,
        )
        .bind(task_id.to_string())
        .fetch_all(&self.pool)
        .await?;

        rows.into_iter().map(row_to_task_dependency).collect()
    }

    pub async fn add_task_resource_lock(
        &self,
        task_id: TaskId,
        resource_key: &str,
        actor: &str,
    ) -> anyhow::Result<TaskResourceLock> {
        self.get_task(task_id).await?;
        let resource_key = normalize_resource_key(resource_key)?;
        let resource_lock = TaskResourceLock {
            task_id,
            resource_key,
            lock_mode: "exclusive".to_owned(),
            created_at: Utc::now(),
        };

        sqlx::query(
            "INSERT OR IGNORE INTO task_resource_locks (task_id, resource_key, lock_mode, created_at) VALUES (?, ?, ?, ?)",
        )
        .bind(resource_lock.task_id.to_string())
        .bind(&resource_lock.resource_key)
        .bind(&resource_lock.lock_mode)
        .bind(resource_lock.created_at)
        .execute(&self.pool)
        .await?;

        self.record_action(
            Some(task_id),
            actor,
            "add_task_resource_lock",
            json!({ "resource_key": resource_lock.resource_key, "lock_mode": resource_lock.lock_mode }),
        )
        .await?;

        self.get_task_resource_lock(task_id, &resource_lock.resource_key)
            .await
    }

    pub async fn remove_task_resource_lock(
        &self,
        task_id: TaskId,
        resource_key: &str,
        actor: &str,
    ) -> anyhow::Result<TaskResourceLock> {
        let resource_key = normalize_resource_key(resource_key)?;
        let resource_lock = self.get_task_resource_lock(task_id, &resource_key).await?;

        sqlx::query("DELETE FROM task_resource_locks WHERE task_id = ? AND resource_key = ?")
            .bind(task_id.to_string())
            .bind(&resource_key)
            .execute(&self.pool)
            .await?;

        self.record_action(
            Some(task_id),
            actor,
            "remove_task_resource_lock",
            json!({ "resource_key": resource_key }),
        )
        .await?;

        Ok(resource_lock)
    }

    pub async fn get_task_resource_lock(
        &self,
        task_id: TaskId,
        resource_key: &str,
    ) -> anyhow::Result<TaskResourceLock> {
        let resource_key = normalize_resource_key(resource_key)?;
        let row =
            sqlx::query("SELECT * FROM task_resource_locks WHERE task_id = ? AND resource_key = ?")
                .bind(task_id.to_string())
                .bind(resource_key)
                .fetch_one(&self.pool)
                .await?;

        row_to_task_resource_lock(row)
    }

    pub async fn list_task_resource_locks(
        &self,
        task_id: TaskId,
    ) -> anyhow::Result<Vec<TaskResourceLock>> {
        let rows = sqlx::query(
            r#"
            SELECT * FROM task_resource_locks
            WHERE task_id = ?
            ORDER BY created_at ASC
            "#,
        )
        .bind(task_id.to_string())
        .fetch_all(&self.pool)
        .await?;

        rows.into_iter().map(row_to_task_resource_lock).collect()
    }

    pub async fn add_task_note(
        &self,
        task_id: TaskId,
        content: &str,
        actor: &str,
    ) -> anyhow::Result<TaskNote> {
        self.get_task(task_id).await?;
        let note = TaskNote {
            id: Uuid::now_v7(),
            task_id,
            actor: actor.to_owned(),
            content: content.trim().to_owned(),
            created_at: Utc::now(),
        };

        if note.content.is_empty() {
            anyhow::bail!("task note content cannot be empty");
        }

        sqlx::query(
            "INSERT INTO task_notes (id, task_id, actor, content, created_at) VALUES (?, ?, ?, ?, ?)",
        )
        .bind(note.id.to_string())
        .bind(note.task_id.to_string())
        .bind(&note.actor)
        .bind(&note.content)
        .bind(note.created_at)
        .execute(&self.pool)
        .await?;

        self.record_action(
            Some(task_id),
            actor,
            "add_task_note",
            json!({ "note_id": note.id, "content": note.content }),
        )
        .await?;

        Ok(note)
    }

    pub async fn list_task_notes(&self, task_id: TaskId) -> anyhow::Result<Vec<TaskNote>> {
        let rows = sqlx::query(
            r#"
            SELECT * FROM task_notes
            WHERE task_id = ?
            ORDER BY created_at ASC
            "#,
        )
        .bind(task_id.to_string())
        .fetch_all(&self.pool)
        .await?;

        rows.into_iter().map(row_to_task_note).collect()
    }

    pub async fn record_action(
        &self,
        task_id: Option<TaskId>,
        actor: &str,
        action_type: &str,
        details: serde_json::Value,
    ) -> anyhow::Result<TaskAction> {
        let action = TaskAction {
            id: Uuid::now_v7(),
            task_id,
            actor: actor.to_owned(),
            action_type: action_type.to_owned(),
            details,
            created_at: Utc::now(),
        };

        sqlx::query(
            "INSERT INTO task_actions (id, task_id, actor, action_type, details, created_at) VALUES (?, ?, ?, ?, ?, ?)",
        )
        .bind(action.id.to_string())
        .bind(action.task_id.map(|id| id.to_string()))
        .bind(&action.actor)
        .bind(&action.action_type)
        .bind(serde_json::to_string(&action.details)?)
        .bind(action.created_at)
        .execute(&self.pool)
        .await?;

        Ok(action)
    }

    pub async fn list_task_actions(&self, task_id: TaskId) -> anyhow::Result<Vec<TaskAction>> {
        let rows = sqlx::query(
            r#"
            SELECT * FROM task_actions
            WHERE task_id = ?
            ORDER BY created_at ASC
            "#,
        )
        .bind(task_id.to_string())
        .fetch_all(&self.pool)
        .await?;

        rows.into_iter().map(row_to_task_action).collect()
    }

    pub async fn list_global_actions(&self) -> anyhow::Result<Vec<TaskAction>> {
        let rows = sqlx::query(
            r#"
            SELECT * FROM task_actions
            WHERE task_id IS NULL
            ORDER BY created_at ASC
            "#,
        )
        .fetch_all(&self.pool)
        .await?;

        rows.into_iter().map(row_to_task_action).collect()
    }

    pub async fn get_or_create_main_conversation(&self) -> anyhow::Result<Conversation> {
        if let Some(conversation) = self.get_main_conversation().await? {
            return Ok(conversation);
        }

        let now = Utc::now();
        let id = Uuid::now_v7();
        sqlx::query(
            "INSERT INTO conversations (id, task_id, title, created_at, updated_at) VALUES (?, NULL, ?, ?, ?)",
        )
        .bind(id.to_string())
        .bind("Main Agent")
        .bind(now)
        .bind(now)
        .execute(&self.pool)
        .await?;

        self.get_conversation(id).await
    }

    pub async fn get_main_conversation(&self) -> anyhow::Result<Option<Conversation>> {
        let row = sqlx::query(
            "SELECT * FROM conversations WHERE task_id IS NULL AND title = 'Main Agent' ORDER BY created_at ASC LIMIT 1",
        )
        .fetch_optional(&self.pool)
        .await?;

        row.map(row_to_conversation).transpose()
    }

    pub async fn get_conversation(&self, id: ConversationId) -> anyhow::Result<Conversation> {
        let row = sqlx::query("SELECT * FROM conversations WHERE id = ?")
            .bind(id.to_string())
            .fetch_one(&self.pool)
            .await?;
        row_to_conversation(row)
    }

    pub async fn add_conversation_message(
        &self,
        conversation_id: ConversationId,
        task_id: Option<TaskId>,
        role: &str,
        content: &str,
    ) -> anyhow::Result<ConversationMessage> {
        let message = ConversationMessage {
            id: Uuid::now_v7(),
            conversation_id,
            task_id,
            role: role.to_owned(),
            content: content.to_owned(),
            created_at: Utc::now(),
        };

        sqlx::query(
            "INSERT INTO conversation_messages (id, conversation_id, task_id, role, content, created_at) VALUES (?, ?, ?, ?, ?, ?)",
        )
        .bind(message.id.to_string())
        .bind(message.conversation_id.to_string())
        .bind(message.task_id.map(|id| id.to_string()))
        .bind(&message.role)
        .bind(&message.content)
        .bind(message.created_at)
        .execute(&self.pool)
        .await?;

        sqlx::query("UPDATE conversations SET updated_at = ? WHERE id = ?")
            .bind(message.created_at)
            .bind(message.conversation_id.to_string())
            .execute(&self.pool)
            .await?;

        Ok(message)
    }

    pub async fn list_conversation_messages(
        &self,
        conversation_id: ConversationId,
        limit: i64,
    ) -> anyhow::Result<Vec<ConversationMessage>> {
        let rows = sqlx::query(
            r#"
            SELECT * FROM conversation_messages
            WHERE conversation_id = ?
            ORDER BY created_at DESC
            LIMIT ?
            "#,
        )
        .bind(conversation_id.to_string())
        .bind(limit)
        .fetch_all(&self.pool)
        .await?;

        let mut messages = rows
            .into_iter()
            .map(row_to_conversation_message)
            .collect::<anyhow::Result<Vec<_>>>()?;
        messages.reverse();
        Ok(messages)
    }

    pub async fn list_task_conversation_messages(
        &self,
        task_id: TaskId,
        limit: i64,
    ) -> anyhow::Result<Vec<ConversationMessage>> {
        let task = self.get_task(task_id).await?;
        let Some(conversation_id) = task.conversation_id else {
            return Ok(Vec::new());
        };

        self.list_conversation_messages(conversation_id, limit)
            .await
    }

    pub async fn create_memory(&self, input: CreateMemory, actor: &str) -> anyhow::Result<Memory> {
        let memory = Memory {
            id: Uuid::now_v7(),
            scope: input.scope,
            content: input.content,
            source_task_id: input.source_task_id,
            status: input.status,
            confidence: input.confidence,
            created_at: Utc::now(),
        };

        sqlx::query(
            "INSERT INTO memories (id, scope, content, source_task_id, status, confidence, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
        )
        .bind(memory.id.to_string())
        .bind(&memory.scope)
        .bind(&memory.content)
        .bind(memory.source_task_id.map(|id| id.to_string()))
        .bind(memory.status.to_string())
        .bind(memory.confidence)
        .bind(memory.created_at)
        .execute(&self.pool)
        .await?;

        self.record_action(
            memory.source_task_id,
            actor,
            "create_memory",
            json!({ "memory_id": memory.id, "status": memory.status, "scope": memory.scope }),
        )
        .await?;

        Ok(memory)
    }

    pub async fn list_memories(&self) -> anyhow::Result<Vec<Memory>> {
        let rows = sqlx::query(
            r#"
            SELECT * FROM memories
            ORDER BY
              CASE status
                WHEN 'pending' THEN 0
                WHEN 'approved' THEN 1
                ELSE 2
              END,
              created_at DESC
            "#,
        )
        .fetch_all(&self.pool)
        .await?;

        rows.into_iter().map(row_to_memory).collect()
    }

    pub async fn list_approved_memories(&self, limit: i64) -> anyhow::Result<Vec<Memory>> {
        let rows = sqlx::query(
            r#"
            SELECT * FROM memories
            WHERE status = 'approved'
            ORDER BY confidence DESC, created_at DESC
            LIMIT ?
            "#,
        )
        .bind(limit)
        .fetch_all(&self.pool)
        .await?;

        rows.into_iter().map(row_to_memory).collect()
    }

    pub async fn list_task_memories(&self, task_id: TaskId) -> anyhow::Result<Vec<Memory>> {
        let rows = sqlx::query(
            r#"
            SELECT * FROM memories
            WHERE source_task_id = ?
            ORDER BY
              CASE status
                WHEN 'pending' THEN 0
                WHEN 'approved' THEN 1
                ELSE 2
              END,
              confidence DESC,
              created_at DESC
            "#,
        )
        .bind(task_id.to_string())
        .fetch_all(&self.pool)
        .await?;

        rows.into_iter().map(row_to_memory).collect()
    }

    pub async fn set_memory_status(
        &self,
        id: MemoryId,
        status: MemoryStatus,
        actor: &str,
    ) -> anyhow::Result<Memory> {
        sqlx::query("UPDATE memories SET status = ? WHERE id = ?")
            .bind(status.to_string())
            .bind(id.to_string())
            .execute(&self.pool)
            .await?;

        let memory = self.get_memory(id).await?;
        self.record_action(
            memory.source_task_id,
            actor,
            "set_memory_status",
            json!({ "memory_id": id, "status": status }),
        )
        .await?;
        Ok(memory)
    }

    pub async fn update_memory(
        &self,
        id: MemoryId,
        input: UpdateMemory,
        actor: &str,
    ) -> anyhow::Result<Memory> {
        let current = self.get_memory(id).await?;
        let scope = input.scope.unwrap_or(current.scope);
        let content = input.content.unwrap_or(current.content);
        let confidence = input.confidence.unwrap_or(current.confidence);

        sqlx::query("UPDATE memories SET scope = ?, content = ?, confidence = ? WHERE id = ?")
            .bind(&scope)
            .bind(&content)
            .bind(confidence)
            .bind(id.to_string())
            .execute(&self.pool)
            .await?;

        let memory = self.get_memory(id).await?;
        self.record_action(
            memory.source_task_id,
            actor,
            "update_memory",
            json!({ "memory_id": id, "scope": scope, "confidence": confidence }),
        )
        .await?;
        Ok(memory)
    }

    pub async fn delete_memory(&self, id: MemoryId, actor: &str) -> anyhow::Result<Memory> {
        let memory = self.get_memory(id).await?;
        sqlx::query("DELETE FROM memories WHERE id = ?")
            .bind(id.to_string())
            .execute(&self.pool)
            .await?;

        self.record_action(
            memory.source_task_id,
            actor,
            "delete_memory",
            json!({ "memory_id": id, "scope": memory.scope, "status": memory.status }),
        )
        .await?;
        Ok(memory)
    }

    pub async fn get_memory(&self, id: MemoryId) -> anyhow::Result<Memory> {
        let row = sqlx::query("SELECT * FROM memories WHERE id = ?")
            .bind(id.to_string())
            .fetch_one(&self.pool)
            .await?;
        row_to_memory(row)
    }

    pub async fn create_skill(&self, input: CreateSkill, actor: &str) -> anyhow::Result<Skill> {
        let now = Utc::now();
        let skill = Skill {
            id: Uuid::now_v7(),
            name: input.name,
            description: input.description,
            trigger_rules: input.trigger_rules,
            tool_subset: input.tool_subset,
            resource_path: input.resource_path,
            created_at: now,
            updated_at: now,
        };

        sqlx::query(
            r#"
            INSERT INTO skills (id, name, description, trigger_rules, tool_subset, resource_path, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            "#,
        )
        .bind(skill.id.to_string())
        .bind(&skill.name)
        .bind(&skill.description)
        .bind(serde_json::to_string(&skill.trigger_rules)?)
        .bind(serde_json::to_string(&skill.tool_subset)?)
        .bind(&skill.resource_path)
        .bind(skill.created_at)
        .bind(skill.updated_at)
        .execute(&self.pool)
        .await?;

        self.record_action(
            None,
            actor,
            "create_skill",
            json!({ "skill_id": skill.id, "name": skill.name }),
        )
        .await?;

        Ok(skill)
    }

    pub async fn list_skills(&self) -> anyhow::Result<Vec<Skill>> {
        let rows = sqlx::query("SELECT * FROM skills ORDER BY name ASC")
            .fetch_all(&self.pool)
            .await?;

        rows.into_iter().map(row_to_skill).collect()
    }

    pub async fn list_skills_by_names(&self, names: &[String]) -> anyhow::Result<Vec<Skill>> {
        if names.is_empty() {
            return Ok(Vec::new());
        }

        let skills = self.list_skills().await?;
        let mut selected = Vec::new();
        for name in names {
            if selected.iter().any(|skill: &Skill| skill.name == *name) {
                continue;
            }
            if let Some(skill) = skills.iter().find(|skill| skill.name == *name) {
                selected.push(skill.clone());
            }
        }

        Ok(selected)
    }

    pub async fn get_skill(&self, id: SkillId) -> anyhow::Result<Skill> {
        let row = sqlx::query("SELECT * FROM skills WHERE id = ?")
            .bind(id.to_string())
            .fetch_one(&self.pool)
            .await?;

        row_to_skill(row)
    }

    pub async fn update_skill(
        &self,
        id: SkillId,
        input: UpdateSkill,
        actor: &str,
    ) -> anyhow::Result<Skill> {
        let current = self.get_skill(id).await?;
        let name = input.name.unwrap_or(current.name);
        let description = input.description.unwrap_or(current.description);
        let trigger_rules = input.trigger_rules.unwrap_or(current.trigger_rules);
        let tool_subset = input.tool_subset.unwrap_or(current.tool_subset);
        let resource_path = match input.resource_path {
            Some(resource_path) => resource_path.and_then(|path| {
                let path = path.trim().to_owned();
                (!path.is_empty()).then_some(path)
            }),
            None => current.resource_path,
        };
        let now = Utc::now();

        sqlx::query(
            r#"
            UPDATE skills
            SET name = ?, description = ?, trigger_rules = ?, tool_subset = ?,
                resource_path = ?, updated_at = ?
            WHERE id = ?
            "#,
        )
        .bind(&name)
        .bind(&description)
        .bind(serde_json::to_string(&trigger_rules)?)
        .bind(serde_json::to_string(&tool_subset)?)
        .bind(&resource_path)
        .bind(now)
        .bind(id.to_string())
        .execute(&self.pool)
        .await?;

        self.record_action(
            None,
            actor,
            "update_skill",
            json!({ "skill_id": id, "name": name, "trigger_rules": trigger_rules, "tool_subset": tool_subset, "resource_path": resource_path }),
        )
        .await?;
        self.refresh_all_task_skill_matches(actor).await?;

        self.get_skill(id).await
    }

    pub async fn delete_skill(&self, id: SkillId, actor: &str) -> anyhow::Result<Skill> {
        let skill = self.get_skill(id).await?;

        sqlx::query("DELETE FROM skills WHERE id = ?")
            .bind(id.to_string())
            .execute(&self.pool)
            .await?;

        self.record_action(
            None,
            actor,
            "delete_skill",
            json!({ "skill_id": id, "name": skill.name }),
        )
        .await?;
        self.refresh_all_task_skill_matches(actor).await?;

        Ok(skill)
    }

    async fn match_skills(
        &self,
        title: &str,
        description: &str,
        task_type: TaskType,
    ) -> anyhow::Result<Vec<String>> {
        let skills = self.list_skills().await?;
        let haystack = format!("{title}\n{description}").to_lowercase();
        let matched = skills
            .into_iter()
            .filter(|skill| skill_matches_task(skill, &haystack, task_type))
            .map(|skill| skill.name)
            .collect();

        Ok(matched)
    }

    async fn refresh_all_task_skill_matches(&self, actor: &str) -> anyhow::Result<()> {
        let rows = sqlx::query("SELECT * FROM tasks")
            .fetch_all(&self.pool)
            .await?;
        for row in rows {
            let task = row_to_task(row)?;
            let matched_skills = self
                .match_skills(&task.title, &task.description, task.task_type)
                .await?;
            if matched_skills == task.matched_skills {
                continue;
            }

            sqlx::query("UPDATE tasks SET matched_skills = ?, updated_at = ? WHERE id = ?")
                .bind(serde_json::to_string(&matched_skills)?)
                .bind(Utc::now())
                .bind(task.id.to_string())
                .execute(&self.pool)
                .await?;

            self.record_action(
                Some(task.id),
                actor,
                "refresh_matched_skills",
                json!({ "matched_skills": matched_skills }),
            )
            .await?;
        }

        Ok(())
    }

    async fn next_queue_position(&self) -> anyhow::Result<i64> {
        let value: Option<i64> = sqlx::query_scalar("SELECT MAX(queue_position) + 1 FROM tasks")
            .fetch_one(&self.pool)
            .await?;
        Ok(value.unwrap_or(0))
    }

    async fn dependency_would_create_cycle(
        &self,
        task_id: TaskId,
        depends_on_task_id: TaskId,
    ) -> anyhow::Result<bool> {
        let mut stack = vec![depends_on_task_id];
        let mut visited = std::collections::HashSet::new();

        while let Some(current_id) = stack.pop() {
            if current_id == task_id {
                return Ok(true);
            }

            if !visited.insert(current_id) {
                continue;
            }

            let rows =
                sqlx::query("SELECT depends_on_task_id FROM task_dependencies WHERE task_id = ?")
                    .bind(current_id.to_string())
                    .fetch_all(&self.pool)
                    .await?;

            for row in rows {
                stack.push(parse_uuid(row.try_get::<String, _>("depends_on_task_id")?)?);
            }
        }

        Ok(false)
    }
}

fn recurring_interval_seconds(schedule: Option<&Value>) -> i64 {
    let seconds = schedule
        .and_then(|value| {
            value
                .get("interval_seconds")
                .or_else(|| value.get("every_seconds"))
                .or_else(|| value.get("seconds"))
                .and_then(Value::as_i64)
        })
        .unwrap_or(300);

    seconds.max(0)
}

fn skill_matches_task(skill: &Skill, haystack: &str, task_type: TaskType) -> bool {
    skill.trigger_rules.iter().any(|rule| {
        let rule = rule.trim().to_lowercase();
        if rule.is_empty() {
            return false;
        }

        if let Some(expected_task_type) = parse_task_type_rule(&rule) {
            return expected_task_type == task_type;
        }

        haystack.contains(&rule)
    })
}

fn parse_task_type_rule(rule: &str) -> Option<TaskType> {
    for prefix in [
        "type:",
        "type=",
        "task_type:",
        "task_type=",
        "task-type:",
        "task-type=",
    ] {
        if let Some(value) = rule.strip_prefix(prefix) {
            return parse_task_type_rule_value(value);
        }
    }

    None
}

fn parse_task_type_rule_value(value: &str) -> Option<TaskType> {
    match value.trim().replace('-', "_").as_str() {
        "one_off" | "oneoff" | "once" => Some(TaskType::OneOff),
        "recurring" | "repeat" | "scheduled" => Some(TaskType::Recurring),
        _ => None,
    }
}

fn row_to_memory(row: sqlx::sqlite::SqliteRow) -> anyhow::Result<Memory> {
    let source_task_id: Option<String> = row.try_get("source_task_id")?;
    let status: String = row.try_get("status")?;
    Ok(Memory {
        id: parse_uuid(row.try_get::<String, _>("id")?)?,
        scope: row.try_get("scope")?,
        content: row.try_get("content")?,
        source_task_id: source_task_id.map(parse_uuid).transpose()?,
        status: status.parse()?,
        confidence: row.try_get("confidence")?,
        created_at: row.try_get::<DateTime<Utc>, _>("created_at")?,
    })
}

fn row_to_skill(row: sqlx::sqlite::SqliteRow) -> anyhow::Result<Skill> {
    let trigger_rules: String = row.try_get("trigger_rules")?;
    let tool_subset: String = row.try_get("tool_subset")?;
    Ok(Skill {
        id: parse_uuid(row.try_get::<String, _>("id")?)?,
        name: row.try_get("name")?,
        description: row.try_get("description")?,
        trigger_rules: serde_json::from_str(&trigger_rules)?,
        tool_subset: serde_json::from_str(&tool_subset)?,
        resource_path: row.try_get("resource_path")?,
        created_at: row.try_get::<DateTime<Utc>, _>("created_at")?,
        updated_at: row.try_get::<DateTime<Utc>, _>("updated_at")?,
    })
}

fn row_to_conversation(row: sqlx::sqlite::SqliteRow) -> anyhow::Result<Conversation> {
    let task_id: Option<String> = row.try_get("task_id")?;
    Ok(Conversation {
        id: parse_uuid(row.try_get::<String, _>("id")?)?,
        task_id: task_id.map(parse_uuid).transpose()?,
        title: row.try_get("title")?,
        created_at: row.try_get::<DateTime<Utc>, _>("created_at")?,
        updated_at: row.try_get::<DateTime<Utc>, _>("updated_at")?,
    })
}

fn row_to_conversation_message(
    row: sqlx::sqlite::SqliteRow,
) -> anyhow::Result<ConversationMessage> {
    let task_id: Option<String> = row.try_get("task_id")?;
    Ok(ConversationMessage {
        id: parse_uuid(row.try_get::<String, _>("id")?)?,
        conversation_id: parse_uuid(row.try_get::<String, _>("conversation_id")?)?,
        task_id: task_id.map(parse_uuid).transpose()?,
        role: row.try_get("role")?,
        content: row.try_get("content")?,
        created_at: row.try_get::<DateTime<Utc>, _>("created_at")?,
    })
}

fn row_to_task_attempt(row: sqlx::sqlite::SqliteRow) -> anyhow::Result<TaskAttempt> {
    let status: String = row.try_get("status")?;
    Ok(TaskAttempt {
        id: parse_uuid(row.try_get::<String, _>("id")?)?,
        task_id: parse_uuid(row.try_get::<String, _>("task_id")?)?,
        status: status.parse()?,
        summary: row.try_get("summary")?,
        started_at: row.try_get::<DateTime<Utc>, _>("started_at")?,
        finished_at: row.try_get("finished_at")?,
    })
}

fn row_to_task_attempt_event(row: sqlx::sqlite::SqliteRow) -> anyhow::Result<TaskAttemptEvent> {
    let details: String = row.try_get("details")?;
    Ok(TaskAttemptEvent {
        id: parse_uuid(row.try_get::<String, _>("id")?)?,
        attempt_id: parse_uuid(row.try_get::<String, _>("attempt_id")?)?,
        task_id: parse_uuid(row.try_get::<String, _>("task_id")?)?,
        event_type: row.try_get("event_type")?,
        message: row.try_get("message")?,
        details: serde_json::from_str(&details)?,
        created_at: row.try_get::<DateTime<Utc>, _>("created_at")?,
    })
}

fn row_to_task_artifact(row: sqlx::sqlite::SqliteRow) -> anyhow::Result<TaskArtifact> {
    let attempt_id: Option<String> = row.try_get("attempt_id")?;
    Ok(TaskArtifact {
        id: parse_uuid(row.try_get::<String, _>("id")?)?,
        task_id: parse_uuid(row.try_get::<String, _>("task_id")?)?,
        attempt_id: attempt_id.map(parse_uuid).transpose()?,
        name: row.try_get("name")?,
        artifact_type: row.try_get("artifact_type")?,
        uri: row.try_get("uri")?,
        summary: row.try_get("summary")?,
        created_at: row.try_get::<DateTime<Utc>, _>("created_at")?,
    })
}

fn row_to_task_dependency(row: sqlx::sqlite::SqliteRow) -> anyhow::Result<TaskDependency> {
    Ok(TaskDependency {
        task_id: parse_uuid(row.try_get::<String, _>("task_id")?)?,
        depends_on_task_id: parse_uuid(row.try_get::<String, _>("depends_on_task_id")?)?,
        created_at: row.try_get::<DateTime<Utc>, _>("created_at")?,
    })
}

fn row_to_task_resource_lock(row: sqlx::sqlite::SqliteRow) -> anyhow::Result<TaskResourceLock> {
    Ok(TaskResourceLock {
        task_id: parse_uuid(row.try_get::<String, _>("task_id")?)?,
        resource_key: row.try_get("resource_key")?,
        lock_mode: row.try_get("lock_mode")?,
        created_at: row.try_get::<DateTime<Utc>, _>("created_at")?,
    })
}

fn row_to_task_note(row: sqlx::sqlite::SqliteRow) -> anyhow::Result<TaskNote> {
    Ok(TaskNote {
        id: parse_uuid(row.try_get::<String, _>("id")?)?,
        task_id: parse_uuid(row.try_get::<String, _>("task_id")?)?,
        actor: row.try_get("actor")?,
        content: row.try_get("content")?,
        created_at: row.try_get::<DateTime<Utc>, _>("created_at")?,
    })
}

fn row_to_task_action(row: sqlx::sqlite::SqliteRow) -> anyhow::Result<TaskAction> {
    let task_id: Option<String> = row.try_get("task_id")?;
    let details: String = row.try_get("details")?;
    Ok(TaskAction {
        id: parse_uuid(row.try_get::<String, _>("id")?)?,
        task_id: task_id.map(parse_uuid).transpose()?,
        actor: row.try_get("actor")?,
        action_type: row.try_get("action_type")?,
        details: serde_json::from_str(&details)?,
        created_at: row.try_get::<DateTime<Utc>, _>("created_at")?,
    })
}

fn row_to_task(row: sqlx::sqlite::SqliteRow) -> anyhow::Result<Task> {
    let requested_skills: String = row.try_get("requested_skills")?;
    let matched_skills: String = row.try_get("matched_skills")?;
    let schedule: Option<String> = row.try_get("schedule")?;
    let task_type: String = row.try_get("task_type")?;
    let status: String = row.try_get("status")?;
    let conversation_id: Option<String> = row.try_get("conversation_id")?;

    Ok(Task {
        id: parse_uuid(row.try_get::<String, _>("id")?)?,
        title: row.try_get("title")?,
        description: row.try_get("description")?,
        task_type: task_type.parse()?,
        status: status.parse()?,
        priority: row.try_get("priority")?,
        queue_position: row.try_get("queue_position")?,
        created_by: row.try_get("created_by")?,
        conversation_id: conversation_id.map(parse_uuid).transpose()?,
        requested_skills: serde_json::from_str(&requested_skills)?,
        matched_skills: serde_json::from_str(&matched_skills)?,
        schedule: schedule
            .map(|value| serde_json::from_str(&value))
            .transpose()?,
        attempt_count: row.try_get("attempt_count")?,
        lease_owner: row.try_get("lease_owner")?,
        lease_expires_at: row.try_get("lease_expires_at")?,
        last_run_at: row.try_get("last_run_at")?,
        next_run_at: row.try_get("next_run_at")?,
        blocked_reason: row.try_get("blocked_reason")?,
        result_summary: row.try_get("result_summary")?,
        created_at: row.try_get::<DateTime<Utc>, _>("created_at")?,
        updated_at: row.try_get::<DateTime<Utc>, _>("updated_at")?,
    })
}

fn parse_uuid(value: String) -> anyhow::Result<Uuid> {
    Ok(Uuid::parse_str(&value)?)
}

fn normalize_resource_key(resource_key: &str) -> anyhow::Result<String> {
    let resource_key = resource_key.trim().to_owned();
    if resource_key.is_empty() {
        anyhow::bail!("resource key cannot be empty");
    }
    Ok(resource_key)
}

#[cfg(test)]
mod tests {
    use super::*;
    use persistent_agent_domain::TaskType;

    #[tokio::test]
    async fn recurring_task_waits_then_requeues_when_due() -> anyhow::Result<()> {
        let db = Db::connect("sqlite::memory:").await?;
        let task = db
            .create_task(
                CreateTask {
                    title: "Recurring check".to_owned(),
                    description: "Run repeatedly".to_owned(),
                    task_type: TaskType::Recurring,
                    priority: 1,
                    requested_skills: Vec::new(),
                    schedule: Some(json!({ "interval_seconds": 0 })),
                    created_by: "test".to_owned(),
                },
                "test",
            )
            .await?;

        let completed = db.complete_task(task.id, "first run", "test").await?;

        assert_eq!(completed.status, TaskStatus::WaitingForSchedule);
        assert!(completed.next_run_at.is_some());

        let requeued = db.requeue_due_recurring_tasks("test").await?;

        assert_eq!(requeued.len(), 1);
        assert_eq!(requeued[0].status, TaskStatus::Queued);
        assert!(requeued[0].next_run_at.is_none());

        Ok(())
    }

    #[test]
    fn parses_recurring_interval_seconds() {
        assert_eq!(
            recurring_interval_seconds(Some(&json!({ "interval_seconds": 12 }))),
            12
        );
        assert_eq!(
            recurring_interval_seconds(Some(&json!({ "every_seconds": 3 }))),
            3
        );
        assert_eq!(
            recurring_interval_seconds(Some(&json!({ "seconds": -5 }))),
            0
        );
        assert_eq!(recurring_interval_seconds(None), 300);
    }

    #[tokio::test]
    async fn memory_candidate_can_be_approved() -> anyhow::Result<()> {
        let db = Db::connect("sqlite::memory:").await?;
        let memory = db
            .create_memory(
                CreateMemory {
                    scope: "project".to_owned(),
                    content: "Prefer focused tests.".to_owned(),
                    source_task_id: None,
                    status: MemoryStatus::Pending,
                    confidence: 0.7,
                },
                "test",
            )
            .await?;

        let approved = db
            .set_memory_status(memory.id, MemoryStatus::Approved, "test")
            .await?;
        let memories = db.list_memories().await?;

        assert_eq!(approved.status, MemoryStatus::Approved);
        assert_eq!(memories.len(), 1);
        assert_eq!(memories[0].content, "Prefer focused tests.");

        Ok(())
    }

    #[tokio::test]
    async fn approved_memories_are_listed_for_injection() -> anyhow::Result<()> {
        let db = Db::connect("sqlite::memory:").await?;
        let pending = db
            .create_memory(
                CreateMemory {
                    scope: "project".to_owned(),
                    content: "Pending memory".to_owned(),
                    source_task_id: None,
                    status: MemoryStatus::Pending,
                    confidence: 0.4,
                },
                "test",
            )
            .await?;
        db.create_memory(
            CreateMemory {
                scope: "project".to_owned(),
                content: "Approved memory".to_owned(),
                source_task_id: None,
                status: MemoryStatus::Approved,
                confidence: 0.9,
            },
            "test",
        )
        .await?;

        let approved = db.list_approved_memories(10).await?;

        assert_eq!(approved.len(), 1);
        assert_eq!(approved[0].content, "Approved memory");
        assert_ne!(approved[0].id, pending.id);

        Ok(())
    }

    #[tokio::test]
    async fn task_memories_are_listed_by_source_task() -> anyhow::Result<()> {
        let db = Db::connect("sqlite::memory:").await?;
        let task = db
            .create_task(
                CreateTask {
                    title: "Memory source".to_owned(),
                    description: "Record candidates".to_owned(),
                    task_type: TaskType::OneOff,
                    priority: 0,
                    requested_skills: Vec::new(),
                    schedule: None,
                    created_by: "test".to_owned(),
                },
                "test",
            )
            .await?;
        let other_task = db
            .create_task(
                CreateTask {
                    title: "Other source".to_owned(),
                    description: "Should not appear".to_owned(),
                    task_type: TaskType::OneOff,
                    priority: 0,
                    requested_skills: Vec::new(),
                    schedule: None,
                    created_by: "test".to_owned(),
                },
                "test",
            )
            .await?;
        db.create_memory(
            CreateMemory {
                scope: "task".to_owned(),
                content: "Pending candidate".to_owned(),
                source_task_id: Some(task.id),
                status: MemoryStatus::Pending,
                confidence: 0.5,
            },
            "test",
        )
        .await?;
        db.create_memory(
            CreateMemory {
                scope: "task".to_owned(),
                content: "Approved candidate".to_owned(),
                source_task_id: Some(task.id),
                status: MemoryStatus::Approved,
                confidence: 0.9,
            },
            "test",
        )
        .await?;
        db.create_memory(
            CreateMemory {
                scope: "task".to_owned(),
                content: "Other candidate".to_owned(),
                source_task_id: Some(other_task.id),
                status: MemoryStatus::Pending,
                confidence: 1.0,
            },
            "test",
        )
        .await?;

        let memories = db.list_task_memories(task.id).await?;

        assert_eq!(memories.len(), 2);
        assert_eq!(memories[0].content, "Pending candidate");
        assert_eq!(memories[1].content, "Approved candidate");

        Ok(())
    }

    #[tokio::test]
    async fn memory_can_be_updated_and_deleted() -> anyhow::Result<()> {
        let db = Db::connect("sqlite::memory:").await?;
        let memory = db
            .create_memory(
                CreateMemory {
                    scope: "project".to_owned(),
                    content: "Old content".to_owned(),
                    source_task_id: None,
                    status: MemoryStatus::Approved,
                    confidence: 0.4,
                },
                "test",
            )
            .await?;

        let updated = db
            .update_memory(
                memory.id,
                UpdateMemory {
                    scope: Some("repository".to_owned()),
                    content: Some("New content".to_owned()),
                    confidence: Some(0.8),
                },
                "test",
            )
            .await?;
        let deleted = db.delete_memory(memory.id, "test").await?;
        let memories = db.list_memories().await?;

        assert_eq!(updated.scope, "repository");
        assert_eq!(updated.content, "New content");
        assert_eq!(updated.confidence, 0.8);
        assert_eq!(deleted.id, memory.id);
        assert!(memories.is_empty());

        Ok(())
    }

    #[tokio::test]
    async fn task_creation_matches_skills_by_trigger_rules() -> anyhow::Result<()> {
        let db = Db::connect("sqlite::memory:").await?;
        db.create_skill(
            CreateSkill {
                name: "github".to_owned(),
                description: "GitHub repository work".to_owned(),
                trigger_rules: vec!["github".to_owned(), "issue".to_owned()],
                tool_subset: vec!["github_search".to_owned()],
                resource_path: None,
            },
            "test",
        )
        .await?;

        let task = db
            .create_task(
                CreateTask {
                    title: "Check GitHub issue".to_owned(),
                    description: "Look for open issues".to_owned(),
                    task_type: TaskType::OneOff,
                    priority: 0,
                    requested_skills: Vec::new(),
                    schedule: None,
                    created_by: "test".to_owned(),
                },
                "test",
            )
            .await?;

        assert_eq!(task.matched_skills, vec!["github"]);

        Ok(())
    }

    #[tokio::test]
    async fn task_creation_matches_skills_by_task_type_rules() -> anyhow::Result<()> {
        let db = Db::connect("sqlite::memory:").await?;
        db.create_skill(
            CreateSkill {
                name: "recurring-monitor".to_owned(),
                description: "Default recurring monitor skill".to_owned(),
                trigger_rules: vec!["type:recurring".to_owned()],
                tool_subset: vec!["scheduler".to_owned()],
                resource_path: None,
            },
            "test",
        )
        .await?;
        db.create_skill(
            CreateSkill {
                name: "one-off-review".to_owned(),
                description: "Default one-off review skill".to_owned(),
                trigger_rules: vec!["task_type:one_off".to_owned()],
                tool_subset: Vec::new(),
                resource_path: None,
            },
            "test",
        )
        .await?;

        let recurring = db
            .create_task(
                CreateTask {
                    title: "Check repository issues".to_owned(),
                    description: "Poll for updates".to_owned(),
                    task_type: TaskType::Recurring,
                    priority: 0,
                    requested_skills: Vec::new(),
                    schedule: Some(json!({ "interval_seconds": 60 })),
                    created_by: "test".to_owned(),
                },
                "test",
            )
            .await?;
        let one_off = db
            .create_task(
                CreateTask {
                    title: "Write release notes".to_owned(),
                    description: "One time writing task".to_owned(),
                    task_type: TaskType::OneOff,
                    priority: 0,
                    requested_skills: Vec::new(),
                    schedule: None,
                    created_by: "test".to_owned(),
                },
                "test",
            )
            .await?;

        assert_eq!(recurring.matched_skills, vec!["recurring-monitor"]);
        assert_eq!(one_off.matched_skills, vec!["one-off-review"]);

        Ok(())
    }

    #[tokio::test]
    async fn task_update_refreshes_matched_skills() -> anyhow::Result<()> {
        let db = Db::connect("sqlite::memory:").await?;
        db.create_skill(
            CreateSkill {
                name: "github".to_owned(),
                description: "GitHub repository work".to_owned(),
                trigger_rules: vec!["github".to_owned()],
                tool_subset: Vec::new(),
                resource_path: None,
            },
            "test",
        )
        .await?;
        let task = db
            .create_task(
                CreateTask {
                    title: "Plain task".to_owned(),
                    description: "No repository trigger yet".to_owned(),
                    task_type: TaskType::OneOff,
                    priority: 0,
                    requested_skills: Vec::new(),
                    schedule: None,
                    created_by: "test".to_owned(),
                },
                "test",
            )
            .await?;

        let updated = db
            .update_task(
                task.id,
                UpdateTask {
                    title: Some("Check GitHub issue".to_owned()),
                    description: None,
                    priority: None,
                    requested_skills: None,
                    schedule: None,
                },
                "test",
            )
            .await?;

        assert_eq!(updated.matched_skills, vec!["github"]);

        Ok(())
    }

    #[tokio::test]
    async fn skill_can_be_updated_and_deleted() -> anyhow::Result<()> {
        let db = Db::connect("sqlite::memory:").await?;
        let skill = db
            .create_skill(
                CreateSkill {
                    name: "repo".to_owned(),
                    description: "Repository work".to_owned(),
                    trigger_rules: vec!["repository".to_owned()],
                    tool_subset: vec!["shell".to_owned()],
                    resource_path: None,
                },
                "test",
            )
            .await?;
        let task = db
            .create_task(
                CreateTask {
                    title: "Check GitHub issue".to_owned(),
                    description: "Look for open issues".to_owned(),
                    task_type: TaskType::OneOff,
                    priority: 0,
                    requested_skills: Vec::new(),
                    schedule: None,
                    created_by: "test".to_owned(),
                },
                "test",
            )
            .await?;
        assert!(task.matched_skills.is_empty());

        let updated = db
            .update_skill(
                skill.id,
                UpdateSkill {
                    name: Some("github".to_owned()),
                    description: Some("GitHub issue work".to_owned()),
                    trigger_rules: Some(vec!["github".to_owned(), "issue".to_owned()]),
                    tool_subset: Some(vec!["github_search".to_owned()]),
                    resource_path: Some(Some("skills/github".to_owned())),
                },
                "test",
            )
            .await?;
        let matched = db.get_task(task.id).await?;
        let cleared = db
            .update_skill(
                skill.id,
                UpdateSkill {
                    name: None,
                    description: None,
                    trigger_rules: None,
                    tool_subset: None,
                    resource_path: Some(None),
                },
                "test",
            )
            .await?;
        let deleted = db.delete_skill(skill.id, "test").await?;
        let unmatched = db.get_task(task.id).await?;

        assert_eq!(updated.name, "github");
        assert_eq!(updated.trigger_rules, vec!["github", "issue"]);
        assert_eq!(updated.tool_subset, vec!["github_search"]);
        assert_eq!(updated.resource_path.as_deref(), Some("skills/github"));
        assert_eq!(cleared.resource_path, None);
        assert_eq!(matched.matched_skills, vec!["github"]);
        assert_eq!(deleted.id, skill.id);
        assert!(unmatched.matched_skills.is_empty());

        Ok(())
    }

    #[tokio::test]
    async fn task_history_lists_attempts_and_actions() -> anyhow::Result<()> {
        let db = Db::connect("sqlite::memory:").await?;
        let task = db
            .create_task(
                CreateTask {
                    title: "History check".to_owned(),
                    description: "Record history".to_owned(),
                    task_type: TaskType::OneOff,
                    priority: 0,
                    requested_skills: Vec::new(),
                    schedule: None,
                    created_by: "test".to_owned(),
                },
                "test",
            )
            .await?;
        let first_attempt = db
            .create_attempt(task.id, TaskStatus::Running, Some("started"))
            .await?;
        db.record_attempt_event(
            first_attempt.id,
            task.id,
            "worker_context_prepared",
            "Prepared worker context.",
            json!({ "memory_count": 0 }),
        )
        .await?;
        let attempt = db
            .create_attempt(task.id, TaskStatus::Completed, Some("finished"))
            .await?;
        db.record_attempt_event(
            attempt.id,
            task.id,
            "worker_completed",
            "Worker completed the task.",
            json!({ "summary": "finished" }),
        )
        .await?;
        db.set_task_status(task.id, TaskStatus::Paused, "test", None)
            .await?;

        let attempts = db.list_task_attempts(task.id).await?;
        let events = db.list_task_attempt_events(task.id).await?;
        let second_attempt_events = db.list_attempt_events(attempt.id).await?;
        let actions = db.list_task_actions(task.id).await?;

        assert_eq!(attempts.len(), 2);
        assert_eq!(attempts[0].summary.as_deref(), Some("started"));
        assert_eq!(events.len(), 2);
        assert_eq!(events[1].event_type, "worker_completed");
        assert_eq!(events[1].details["summary"], "finished");
        assert_eq!(second_attempt_events.len(), 1);
        assert_eq!(second_attempt_events[0].event_type, "worker_completed");
        assert!(
            actions
                .iter()
                .any(|action| action.action_type == "create_task")
        );
        assert!(
            actions
                .iter()
                .any(|action| action.action_type == "set_task_status")
        );

        Ok(())
    }

    #[tokio::test]
    async fn failed_task_records_summary_and_action() -> anyhow::Result<()> {
        let db = Db::connect("sqlite::memory:").await?;
        let task = db
            .create_task(
                CreateTask {
                    title: "Fail check".to_owned(),
                    description: "Record failure".to_owned(),
                    task_type: TaskType::OneOff,
                    priority: 0,
                    requested_skills: Vec::new(),
                    schedule: None,
                    created_by: "test".to_owned(),
                },
                "test",
            )
            .await?;
        db.claim_next_runnable("test-worker", 60).await?;

        let failed = db.fail_task(task.id, "tool crashed", "worker").await?;
        let actions = db.list_task_actions(task.id).await?;

        assert_eq!(failed.status, TaskStatus::Failed);
        assert_eq!(failed.result_summary.as_deref(), Some("tool crashed"));
        assert!(failed.blocked_reason.is_none());
        assert!(
            actions
                .iter()
                .any(|action| action.action_type == "fail_task")
        );

        Ok(())
    }

    #[tokio::test]
    async fn task_can_be_deleted_with_global_audit_action() -> anyhow::Result<()> {
        let db = Db::connect("sqlite::memory:").await?;
        let task = db
            .create_task(
                CreateTask {
                    title: "Remove obsolete task".to_owned(),
                    description: "This task should be deleted".to_owned(),
                    task_type: TaskType::OneOff,
                    priority: 0,
                    requested_skills: Vec::new(),
                    schedule: None,
                    created_by: "test".to_owned(),
                },
                "test",
            )
            .await?;

        let deleted = db.delete_task(task.id, "test").await?;
        let tasks = db.list_tasks().await?;
        let actions = db.list_global_actions().await?;

        assert_eq!(deleted.id, task.id);
        assert!(tasks.is_empty());
        assert!(db.get_task(task.id).await.is_err());
        assert!(actions.iter().any(|action| {
            action.action_type == "delete_task"
                && action.details["task_id"] == task.id.to_string()
                && action.details["title"] == "Remove obsolete task"
        }));

        Ok(())
    }

    #[tokio::test]
    async fn failed_attempt_can_be_requeued_with_audit_action() -> anyhow::Result<()> {
        let db = Db::connect("sqlite::memory:").await?;
        let task = db
            .create_task(
                CreateTask {
                    title: "Retry check".to_owned(),
                    description: "Requeue after transient failure".to_owned(),
                    task_type: TaskType::OneOff,
                    priority: 0,
                    requested_skills: Vec::new(),
                    schedule: None,
                    created_by: "test".to_owned(),
                },
                "test",
            )
            .await?;
        db.claim_next_runnable("test-worker", 60).await?;

        let requeued = db
            .requeue_task_after_failure(task.id, "transient error", "worker")
            .await?;
        let actions = db.list_task_actions(task.id).await?;

        assert_eq!(requeued.status, TaskStatus::Queued);
        assert_eq!(requeued.result_summary.as_deref(), Some("transient error"));
        assert!(requeued.blocked_reason.is_none());
        assert!(
            actions
                .iter()
                .any(|action| action.action_type == "requeue_task_after_failure")
        );

        Ok(())
    }

    #[tokio::test]
    async fn running_task_lease_can_be_refreshed_by_owner() -> anyhow::Result<()> {
        let db = Db::connect("sqlite::memory:").await?;
        let task = db
            .create_task(
                CreateTask {
                    title: "Heartbeat check".to_owned(),
                    description: "Refresh running lease".to_owned(),
                    task_type: TaskType::OneOff,
                    priority: 0,
                    requested_skills: Vec::new(),
                    schedule: None,
                    created_by: "test".to_owned(),
                },
                "test",
            )
            .await?;
        db.claim_next_runnable("worker-a", 1).await?;

        let wrong_owner = db.heartbeat_task_lease(task.id, "worker-b", 60).await?;
        let refreshed = db.heartbeat_task_lease(task.id, "worker-a", 60).await?;

        assert!(wrong_owner.is_none());
        assert!(refreshed.is_some());

        db.complete_task(task.id, "done", "worker-a").await?;

        let completed = db.heartbeat_task_lease(task.id, "worker-a", 60).await?;
        assert!(completed.is_none());

        Ok(())
    }

    #[tokio::test]
    async fn expired_running_task_can_be_recovered_to_queue() -> anyhow::Result<()> {
        let db = Db::connect("sqlite::memory:").await?;
        let task = db
            .create_task(
                CreateTask {
                    title: "Recover stale lease".to_owned(),
                    description: "Move expired running work back to the queue".to_owned(),
                    task_type: TaskType::OneOff,
                    priority: 0,
                    requested_skills: Vec::new(),
                    schedule: None,
                    created_by: "test".to_owned(),
                },
                "test",
            )
            .await?;
        db.claim_next_runnable("worker-a", 60).await?;

        let expired_at = Utc::now() - Duration::seconds(5);
        sqlx::query("UPDATE tasks SET lease_expires_at = ? WHERE id = ?")
            .bind(expired_at)
            .bind(task.id.to_string())
            .execute(db.pool())
            .await?;

        let recovered = db.recover_expired_running_tasks("scheduler").await?;
        let lease_row = sqlx::query("SELECT lease_owner, lease_expires_at FROM tasks WHERE id = ?")
            .bind(task.id.to_string())
            .fetch_one(db.pool())
            .await?;
        let actions = db.list_task_actions(task.id).await?;

        assert_eq!(recovered.len(), 1);
        assert_eq!(recovered[0].status, TaskStatus::Queued);
        assert!(
            lease_row
                .try_get::<Option<String>, _>("lease_owner")?
                .is_none()
        );
        assert!(
            lease_row
                .try_get::<Option<DateTime<Utc>>, _>("lease_expires_at")?
                .is_none()
        );
        assert!(
            actions
                .iter()
                .any(|action| action.action_type == "recover_expired_running_task")
        );

        let claimed_again = db.claim_next_runnable("worker-b", 60).await?;
        assert_eq!(claimed_again.map(|task| task.id), Some(task.id));

        Ok(())
    }

    #[tokio::test]
    async fn task_type_can_be_converted_with_audit_action() -> anyhow::Result<()> {
        let db = Db::connect("sqlite::memory:").await?;
        db.create_skill(
            CreateSkill {
                name: "recurring-monitor".to_owned(),
                description: "Default recurring monitor skill".to_owned(),
                trigger_rules: vec!["type:recurring".to_owned()],
                tool_subset: Vec::new(),
                resource_path: None,
            },
            "test",
        )
        .await?;
        let task = db
            .create_task(
                CreateTask {
                    title: "Convert check".to_owned(),
                    description: "Change type later".to_owned(),
                    task_type: TaskType::OneOff,
                    priority: 0,
                    requested_skills: Vec::new(),
                    schedule: None,
                    created_by: "test".to_owned(),
                },
                "test",
            )
            .await?;

        let recurring = db
            .convert_task_type(
                task.id,
                TaskType::Recurring,
                Some(json!({ "interval_seconds": 12 })),
                "test",
            )
            .await?;
        let one_off = db
            .convert_task_type(task.id, TaskType::OneOff, None, "test")
            .await?;
        let actions = db.list_task_actions(task.id).await?;

        assert_eq!(recurring.task_type, TaskType::Recurring);
        assert_eq!(recurring.schedule, Some(json!({ "interval_seconds": 12 })));
        assert_eq!(recurring.matched_skills, vec!["recurring-monitor"]);
        assert_eq!(one_off.task_type, TaskType::OneOff);
        assert!(one_off.matched_skills.is_empty());
        assert!(one_off.schedule.is_none());
        assert!(one_off.next_run_at.is_none());
        assert!(
            actions
                .iter()
                .any(|action| action.action_type == "convert_task_type")
        );

        Ok(())
    }

    #[tokio::test]
    async fn recurring_task_schedule_can_be_updated_with_audit_action() -> anyhow::Result<()> {
        let db = Db::connect("sqlite::memory:").await?;
        let task = db
            .create_task(
                CreateTask {
                    title: "Check issues".to_owned(),
                    description: "Check repository issues".to_owned(),
                    task_type: TaskType::Recurring,
                    priority: 0,
                    requested_skills: Vec::new(),
                    schedule: Some(json!({ "interval_seconds": 60 })),
                    created_by: "test".to_owned(),
                },
                "test",
            )
            .await?;
        db.complete_task(task.id, "scheduled", "test").await?;

        let updated = db
            .update_task_schedule(task.id, json!({ "interval_seconds": 600 }), "test")
            .await?;
        let actions = db.list_task_actions(task.id).await?;

        assert_eq!(updated.schedule, Some(json!({ "interval_seconds": 600 })));
        assert_eq!(updated.status, TaskStatus::WaitingForSchedule);
        assert!(updated.next_run_at.is_some());
        assert!(
            actions
                .iter()
                .any(|action| action.action_type == "update_task_schedule")
        );

        Ok(())
    }

    #[tokio::test]
    async fn task_dependency_blocks_claim_until_dependency_is_satisfied() -> anyhow::Result<()> {
        let db = Db::connect("sqlite::memory:").await?;
        let dependency = db
            .create_task(
                CreateTask {
                    title: "Prepare context".to_owned(),
                    description: "Must run first".to_owned(),
                    task_type: TaskType::OneOff,
                    priority: 1,
                    requested_skills: Vec::new(),
                    schedule: None,
                    created_by: "test".to_owned(),
                },
                "test",
            )
            .await?;
        let dependent = db
            .create_task(
                CreateTask {
                    title: "Use prepared context".to_owned(),
                    description: "Should wait for dependency".to_owned(),
                    task_type: TaskType::OneOff,
                    priority: 100,
                    requested_skills: Vec::new(),
                    schedule: None,
                    created_by: "test".to_owned(),
                },
                "test",
            )
            .await?;

        db.add_task_dependency(dependent.id, dependency.id, "test")
            .await?;

        let claimed_first = db.claim_next_runnable("worker", 60).await?;
        assert_eq!(claimed_first.map(|task| task.id), Some(dependency.id));

        db.complete_task(dependency.id, "done", "worker").await?;

        let claimed_second = db.claim_next_runnable("worker", 60).await?;
        assert_eq!(claimed_second.map(|task| task.id), Some(dependent.id));

        Ok(())
    }

    #[tokio::test]
    async fn next_runnable_peek_skips_blocked_queued_tasks_without_claiming() -> anyhow::Result<()>
    {
        let db = Db::connect("sqlite::memory:").await?;
        let dependency = db
            .create_task(
                CreateTask {
                    title: "Build package".to_owned(),
                    description: "Build first".to_owned(),
                    task_type: TaskType::OneOff,
                    priority: 0,
                    requested_skills: Vec::new(),
                    schedule: None,
                    created_by: "test".to_owned(),
                },
                "test",
            )
            .await?;
        let dependent = db
            .create_task(
                CreateTask {
                    title: "Deploy release".to_owned(),
                    description: "Should wait for dependency".to_owned(),
                    task_type: TaskType::OneOff,
                    priority: 20,
                    requested_skills: Vec::new(),
                    schedule: None,
                    created_by: "test".to_owned(),
                },
                "test",
            )
            .await?;
        let runnable = db
            .create_task(
                CreateTask {
                    title: "Write notes".to_owned(),
                    description: "Can run now".to_owned(),
                    task_type: TaskType::OneOff,
                    priority: 10,
                    requested_skills: Vec::new(),
                    schedule: None,
                    created_by: "test".to_owned(),
                },
                "test",
            )
            .await?;
        db.add_task_dependency(dependent.id, dependency.id, "test")
            .await?;

        let peeked = db.peek_next_runnable().await?;
        let dependent_after_peek = db.get_task(dependent.id).await?;
        let runnable_after_peek = db.get_task(runnable.id).await?;

        assert_eq!(peeked.map(|task| task.id), Some(runnable.id));
        assert_eq!(dependent_after_peek.status, TaskStatus::Queued);
        assert_eq!(runnable_after_peek.status, TaskStatus::Queued);

        Ok(())
    }

    #[tokio::test]
    async fn task_dependency_rejects_self_and_cycles() -> anyhow::Result<()> {
        let db = Db::connect("sqlite::memory:").await?;
        let first = db
            .create_task(
                CreateTask {
                    title: "First".to_owned(),
                    description: "First task".to_owned(),
                    task_type: TaskType::OneOff,
                    priority: 0,
                    requested_skills: Vec::new(),
                    schedule: None,
                    created_by: "test".to_owned(),
                },
                "test",
            )
            .await?;
        let second = db
            .create_task(
                CreateTask {
                    title: "Second".to_owned(),
                    description: "Second task".to_owned(),
                    task_type: TaskType::OneOff,
                    priority: 0,
                    requested_skills: Vec::new(),
                    schedule: None,
                    created_by: "test".to_owned(),
                },
                "test",
            )
            .await?;

        assert!(
            db.add_task_dependency(first.id, first.id, "test")
                .await
                .is_err()
        );

        db.add_task_dependency(first.id, second.id, "test").await?;

        assert!(
            db.add_task_dependency(second.id, first.id, "test")
                .await
                .is_err()
        );

        Ok(())
    }

    #[tokio::test]
    async fn task_resource_lock_blocks_conflicting_claims() -> anyhow::Result<()> {
        let db = Db::connect("sqlite::memory:").await?;
        let first = db
            .create_task(
                CreateTask {
                    title: "First repo job".to_owned(),
                    description: "Use the shared repository".to_owned(),
                    task_type: TaskType::OneOff,
                    priority: 10,
                    requested_skills: Vec::new(),
                    schedule: None,
                    created_by: "test".to_owned(),
                },
                "test",
            )
            .await?;
        let second = db
            .create_task(
                CreateTask {
                    title: "Second repo job".to_owned(),
                    description: "Should wait for the shared repository".to_owned(),
                    task_type: TaskType::OneOff,
                    priority: 10,
                    requested_skills: Vec::new(),
                    schedule: None,
                    created_by: "test".to_owned(),
                },
                "test",
            )
            .await?;

        db.add_task_resource_lock(first.id, "repo:persistent-agent", "test")
            .await?;
        db.add_task_resource_lock(second.id, " repo:persistent-agent ", "test")
            .await?;

        let claimed_first = db.claim_next_runnable("worker-a", 60).await?;
        assert_eq!(claimed_first.map(|task| task.id), Some(first.id));

        let blocked_by_lock = db.claim_next_runnable("worker-b", 60).await?;
        assert!(blocked_by_lock.is_none());

        db.complete_task(first.id, "done", "worker-a").await?;

        let claimed_second = db.claim_next_runnable("worker-b", 60).await?;
        assert_eq!(claimed_second.map(|task| task.id), Some(second.id));

        let locks = db.list_task_resource_locks(second.id).await?;
        let actions = db.list_task_actions(second.id).await?;

        assert_eq!(locks.len(), 1);
        assert_eq!(locks[0].resource_key, "repo:persistent-agent");
        assert!(actions.iter().any(|action| {
            action.action_type == "add_task_resource_lock"
                && action.details["resource_key"] == "repo:persistent-agent"
        }));

        Ok(())
    }

    #[tokio::test]
    async fn task_resource_locks_can_be_removed() -> anyhow::Result<()> {
        let db = Db::connect("sqlite::memory:").await?;
        let task = db
            .create_task(
                CreateTask {
                    title: "Lock cleanup".to_owned(),
                    description: "Remove lock metadata".to_owned(),
                    task_type: TaskType::OneOff,
                    priority: 0,
                    requested_skills: Vec::new(),
                    schedule: None,
                    created_by: "test".to_owned(),
                },
                "test",
            )
            .await?;

        db.add_task_resource_lock(task.id, "repo:persistent-agent", "test")
            .await?;
        let removed = db
            .remove_task_resource_lock(task.id, "repo:persistent-agent", "test")
            .await?;
        let locks = db.list_task_resource_locks(task.id).await?;

        assert_eq!(removed.resource_key, "repo:persistent-agent");
        assert!(locks.is_empty());

        Ok(())
    }

    #[tokio::test]
    async fn task_notes_can_be_recorded_and_listed() -> anyhow::Result<()> {
        let db = Db::connect("sqlite::memory:").await?;
        let task = db
            .create_task(
                CreateTask {
                    title: "Note check".to_owned(),
                    description: "Record task notes".to_owned(),
                    task_type: TaskType::OneOff,
                    priority: 0,
                    requested_skills: Vec::new(),
                    schedule: None,
                    created_by: "test".to_owned(),
                },
                "test",
            )
            .await?;

        let note = db
            .add_task_note(task.id, "Remember staging approval.", "test")
            .await?;
        let notes = db.list_task_notes(task.id).await?;
        let actions = db.list_task_actions(task.id).await?;

        assert_eq!(note.content, "Remember staging approval.");
        assert_eq!(notes.len(), 1);
        assert_eq!(notes[0].id, note.id);
        assert!(actions.iter().any(|action| {
            action.action_type == "add_task_note"
                && action.details["note_id"] == note.id.to_string()
        }));

        Ok(())
    }

    #[tokio::test]
    async fn task_artifacts_can_be_recorded_and_listed() -> anyhow::Result<()> {
        let db = Db::connect("sqlite::memory:").await?;
        let task = db
            .create_task(
                CreateTask {
                    title: "Artifact check".to_owned(),
                    description: "Record artifact metadata".to_owned(),
                    task_type: TaskType::OneOff,
                    priority: 0,
                    requested_skills: Vec::new(),
                    schedule: None,
                    created_by: "test".to_owned(),
                },
                "test",
            )
            .await?;
        let attempt = db
            .create_attempt(task.id, TaskStatus::Completed, Some("done"))
            .await?;

        db.record_task_artifact(
            task.id,
            Some(attempt.id),
            "summary.md",
            "file",
            "file://summary.md",
            Some("Run summary"),
        )
        .await?;

        let artifacts = db.list_task_artifacts(task.id).await?;

        assert_eq!(artifacts.len(), 1);
        assert_eq!(artifacts[0].attempt_id, Some(attempt.id));
        assert_eq!(artifacts[0].name, "summary.md");
        assert_eq!(artifacts[0].artifact_type, "file");
        assert_eq!(artifacts[0].summary.as_deref(), Some("Run summary"));

        Ok(())
    }
}
