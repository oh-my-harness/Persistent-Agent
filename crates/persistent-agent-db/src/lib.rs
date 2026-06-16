use chrono::{DateTime, Duration, Utc};
use persistent_agent_domain::{
    CreateTask, Task, TaskAction, TaskAttempt, TaskId, TaskStatus, TaskType, UpdateTask,
};
use serde_json::json;
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
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, '[]', ?, 0, ?, ?)
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
        let schedule = input.schedule.or(current.schedule);
        let now = Utc::now();

        sqlx::query(
            r#"
            UPDATE tasks
            SET title = ?, description = ?, priority = ?, requested_skills = ?, schedule = ?, updated_at = ?
            WHERE id = ?
            "#,
        )
        .bind(&title)
        .bind(&description)
        .bind(priority)
        .bind(serde_json::to_string(&requested_skills)?)
        .bind(schedule.as_ref().map(serde_json::to_string).transpose()?)
        .bind(now)
        .bind(id.to_string())
        .execute(&self.pool)
        .await?;

        self.record_action(
            Some(id),
            actor,
            "update_task",
            json!({ "title": title, "priority": priority }),
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
        sqlx::query("UPDATE tasks SET status = ?, blocked_reason = ?, updated_at = ? WHERE id = ?")
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
            ORDER BY priority DESC, queue_position ASC, created_at ASC
            LIMIT 1
            "#,
        )
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

        sqlx::query(
            r#"
            UPDATE tasks
            SET status = ?, result_summary = ?, lease_owner = NULL, lease_expires_at = NULL, updated_at = ?
            WHERE id = ?
            "#,
        )
        .bind(next_status.to_string())
        .bind(summary)
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

    async fn next_queue_position(&self) -> anyhow::Result<i64> {
        let value: Option<i64> = sqlx::query_scalar("SELECT MAX(queue_position) + 1 FROM tasks")
            .fetch_one(&self.pool)
            .await?;
        Ok(value.unwrap_or(0))
    }
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
