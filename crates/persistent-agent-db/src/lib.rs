use chrono::{DateTime, Duration, Utc};
use persistent_agent_domain::{
    Conversation, ConversationId, ConversationMessage, CreateMemory, CreateSkill, CreateTask,
    Memory, MemoryId, MemoryStatus, Skill, Task, TaskAction, TaskAttempt, TaskId, TaskStatus,
    TaskType, UpdateTask,
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
        let matched_skill_names = self.match_skills(&input.title, &input.description).await?;
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

    async fn match_skills(&self, title: &str, description: &str) -> anyhow::Result<Vec<String>> {
        let skills = self.list_skills().await?;
        let haystack = format!("{title}\n{description}").to_lowercase();
        let matched = skills
            .into_iter()
            .filter(|skill| {
                skill.trigger_rules.iter().any(|rule| {
                    let rule = rule.trim().to_lowercase();
                    !rule.is_empty() && haystack.contains(&rule)
                })
            })
            .map(|skill| skill.name)
            .collect();

        Ok(matched)
    }

    async fn next_queue_position(&self) -> anyhow::Result<i64> {
        let value: Option<i64> = sqlx::query_scalar("SELECT MAX(queue_position) + 1 FROM tasks")
            .fetch_one(&self.pool)
            .await?;
        Ok(value.unwrap_or(0))
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
}
