use std::{env, fs, path::Path, process::Command, sync::Arc};

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
    ConversationId, ConversationMessage, CreateMemory, CreateSkill, CreateTask, Memory, MemoryId,
    MemoryStatus, Skill, Task, TaskAction, TaskArtifact, TaskAttempt, TaskAttemptEvent, TaskId,
    TaskNote, TaskResourceLock, TaskStatus, TaskType, UpdateMemory, UpdateSkill, UpdateTask,
};
use serde::{Deserialize, Serialize};

const ZH_SUMMARY: &str = "\u{603b}\u{7ed3}";
const ZH_OVERVIEW: &str = "\u{6982}\u{89c8}";
const ZH_TASK_POOL: &str = "\u{4efb}\u{52a1}\u{6c60}";
const ZH_LIST: &str = "\u{5217}\u{51fa}";
const ZH_CREATE: &str = "\u{521b}\u{5efa}";
const ZH_NEW: &str = "\u{65b0}\u{5efa}";
const ZH_ADD: &str = "\u{6dfb}\u{52a0}";
const ZH_ADD_ONE: &str = "\u{52a0}\u{4e00}\u{4e2a}";
const ZH_SPLIT: &str = "\u{62c6}\u{5206}";
const ZH_DECOMPOSE: &str = "\u{5206}\u{89e3}";
const ZH_RECURRING: &str = "\u{5faa}\u{73af}";
const ZH_SCHEDULED: &str = "\u{5b9a}\u{671f}";
const ZH_ONE_OFF: &str = "\u{4e00}\u{6b21}\u{6027}";
const ZH_PRIORITY: &str = "\u{4f18}\u{5148}\u{7ea7}";
const ZH_PAUSE_TASK: &str = "\u{6682}\u{505c}\u{4efb}\u{52a1}";
const ZH_RESUME_TASK: &str = "\u{6062}\u{590d}\u{4efb}\u{52a1}";
const ZH_CANCEL_TASK: &str = "\u{53d6}\u{6d88}\u{4efb}\u{52a1}";
const ZH_DELETE_TASK: &str = "\u{5220}\u{9664}\u{4efb}\u{52a1}";
const ZH_QUEUE: &str = "\u{961f}\u{5217}";
const ZH_SORT: &str = "\u{6392}\u{5e8f}";
const ZH_MOVE: &str = "\u{79fb}\u{52a8}";
const ZH_POSITION: &str = "\u{4f4d}\u{7f6e}";
const ZH_TASK: &str = "\u{4efb}\u{52a1}";
const ZH_PAUSE: &str = "\u{6682}\u{505c}";
const ZH_RESUME: &str = "\u{6062}\u{590d}";
const ZH_CANCEL: &str = "\u{53d6}\u{6d88}";
const ZH_ADJUST: &str = "\u{8c03}\u{6574}";
const ZH_EXPLAIN: &str = "\u{89e3}\u{91ca}";
const ZH_WHY: &str = "\u{4e3a}\u{4ec0}\u{4e48}";
const ZH_STATE: &str = "\u{72b6}\u{6001}";
const ZH_CLARIFY: &str = "\u{6f84}\u{6e05}";
const ZH_QUESTION: &str = "\u{95ee}\u{9898}";
const ZH_NEED: &str = "\u{9700}\u{8981}";
const ZH_CONVERT_TASK: &str = "\u{8f6c}\u{6362}\u{4efb}\u{52a1}";
const ZH_CHANGE_TO: &str = "\u{6539}\u{6210}";
const ZH_CHANGE_AS: &str = "\u{6539}\u{4e3a}";
const ZH_SET_AS: &str = "\u{8bbe}\u{4e3a}";
const ZH_TO: &str = "\u{5230}";
const ZH_AS: &str = "\u{4e3a}";
const ZH_EVERY: &str = "\u{6bcf}";
const ZH_INTERVAL: &str = "\u{95f4}\u{9694}";
const ZH_DEPEND_ON: &str = "\u{4f9d}\u{8d56}";
const ZH_REMOVE_DEPENDENCY: &str = "\u{53d6}\u{6d88}\u{4f9d}\u{8d56}";
const ZH_NOTE: &str = "\u{5907}\u{6ce8}";
const ZH_RESOURCE_LOCK: &str = "\u{8d44}\u{6e90}\u{9501}";
const ZH_RESOURCE: &str = "\u{8d44}\u{6e90}";
const ZH_MEMORY: &str = "\u{8bb0}\u{5fc6}";
const ZH_LONG_TERM_MEMORY: &str = "\u{957f}\u{671f}\u{8bb0}\u{5fc6}";
const ZH_APPROVE: &str = "\u{91c7}\u{7eb3}";
const ZH_ACCEPT: &str = "\u{63a5}\u{53d7}";
const ZH_REJECT: &str = "\u{62d2}\u{7edd}";
const ZH_SKILL: &str = "\u{6280}\u{80fd}";
const ZH_SCAN: &str = "\u{626b}\u{63cf}";
const ZH_RUN: &str = "\u{8fd0}\u{884c}";
const ZH_SCHEDULER: &str = "\u{8c03}\u{5ea6}";
const ZH_ARTIFACT: &str = "\u{4ea7}\u{7269}";
const ZH_RESULT: &str = "\u{6210}\u{679c}";

#[derive(Clone)]
pub struct MainAgent {
    db: Db,
    advisor: Option<Arc<dyn MainAgentAdvisor>>,
    planner: Option<Arc<dyn MainAgentPlanner>>,
}

impl MainAgent {
    pub fn new(db: Db) -> Self {
        Self {
            db,
            advisor: None,
            planner: None,
        }
    }

    pub fn new_with_advisor(db: Db, advisor: Arc<dyn MainAgentAdvisor>) -> Self {
        Self {
            db,
            advisor: Some(advisor),
            planner: None,
        }
    }

    pub fn with_advisor(mut self, advisor: Arc<dyn MainAgentAdvisor>) -> Self {
        self.advisor = Some(advisor);
        self
    }

    pub fn with_planner(mut self, planner: Arc<dyn MainAgentPlanner>) -> Self {
        self.planner = Some(planner);
        self
    }

    pub async fn create_task(&self, input: CreateTask) -> anyhow::Result<Task> {
        self.db.create_task(input, "main_agent").await
    }

    pub async fn update_task(&self, id: TaskId, input: UpdateTask) -> anyhow::Result<Task> {
        self.db.update_task(id, input, "main_agent").await
    }

    pub async fn delete_task(&self, id: TaskId) -> anyhow::Result<Task> {
        self.db.delete_task(id, "main_agent").await
    }

    pub async fn reprioritize_task(&self, id: TaskId, priority: i64) -> anyhow::Result<Task> {
        self.db
            .update_task(
                id,
                UpdateTask {
                    title: None,
                    description: None,
                    priority: Some(priority),
                    requested_skills: None,
                    schedule: None,
                },
                "main_agent",
            )
            .await
    }

    pub async fn reorder_task(&self, id: TaskId, queue_position: i64) -> anyhow::Result<Task> {
        self.db.reorder_task(id, queue_position, "main_agent").await
    }

    pub async fn convert_task_type(
        &self,
        id: TaskId,
        task_type: TaskType,
        interval_seconds: Option<i64>,
    ) -> anyhow::Result<Task> {
        let schedule = match task_type {
            TaskType::OneOff => None,
            TaskType::Recurring => {
                Some(serde_json::json!({ "interval_seconds": interval_seconds.unwrap_or(300) }))
            }
        };
        self.db
            .convert_task_type(id, task_type, schedule, "main_agent")
            .await
    }

    pub async fn update_task_schedule(
        &self,
        id: TaskId,
        interval_seconds: i64,
    ) -> anyhow::Result<Task> {
        self.db
            .update_task_schedule(
                id,
                serde_json::json!({ "interval_seconds": interval_seconds }),
                "main_agent",
            )
            .await
    }

    pub async fn pause_task(&self, id: TaskId) -> anyhow::Result<Task> {
        self.db
            .set_task_status(id, TaskStatus::Paused, "main_agent", None)
            .await
    }

    pub async fn resume_task(&self, id: TaskId) -> anyhow::Result<Task> {
        self.db
            .set_task_status(id, TaskStatus::Queued, "main_agent", None)
            .await
    }

    pub async fn cancel_task(&self, id: TaskId) -> anyhow::Result<Task> {
        self.db
            .set_task_status(id, TaskStatus::Cancelled, "main_agent", None)
            .await
    }

    pub async fn complete_task(&self, id: TaskId, summary: &str) -> anyhow::Result<Task> {
        self.db.complete_task(id, summary, "main_agent").await
    }

    pub async fn fail_task(&self, id: TaskId, error: &str) -> anyhow::Result<Task> {
        self.db.fail_task(id, error, "main_agent").await
    }

    pub async fn retry_task(&self, id: TaskId, reason: &str) -> anyhow::Result<Task> {
        self.db
            .requeue_task_after_failure(id, reason, "main_agent")
            .await
    }

    pub async fn prepare_task_for_immediate_run(
        &self,
        id: TaskId,
    ) -> anyhow::Result<Result<Task, String>> {
        let mut task = self.db.get_task(id).await?;
        match task.status {
            TaskStatus::Queued => {}
            TaskStatus::Draft | TaskStatus::Paused => {
                task = self.resume_task(id).await?;
            }
            TaskStatus::Failed => {
                task = self
                    .retry_task(id, "Immediate run requested by user.")
                    .await?;
            }
            TaskStatus::Running => {
                return Ok(Err(format!("Task '{}' is already running.", task.title)));
            }
            TaskStatus::WaitingForUser => {
                return Ok(Err(format!(
                    "Task '{}' is waiting for user input before it can run.",
                    task.title
                )));
            }
            TaskStatus::WaitingForSchedule => {
                task = self.resume_task(id).await?;
            }
            TaskStatus::Completed => {
                return Ok(Err(format!(
                    "Task '{}' is already completed. Retry or create follow-up work if it needs another pass.",
                    task.title
                )));
            }
            TaskStatus::Cancelled => {
                return Ok(Err(format!(
                    "Task '{}' is cancelled. Resume or recreate it before running.",
                    task.title
                )));
            }
        }

        let max_priority = self
            .db
            .list_tasks()
            .await?
            .into_iter()
            .map(|task| task.priority)
            .max()
            .unwrap_or(task.priority);
        if task.priority <= max_priority {
            self.reprioritize_task(id, max_priority + 1).await?;
        }
        task = self.reorder_task(id, -1).await?;

        self.db
            .record_action(
                Some(id),
                "main_agent",
                "request_task_run_now",
                serde_json::json!({
                    "priority": task.priority,
                    "queue_position": task.queue_position,
                }),
            )
            .await?;

        Ok(Ok(task))
    }

    pub async fn prepare_next_task_for_run(&self) -> anyhow::Result<Result<Task, String>> {
        let Some(task) = self.db.peek_next_runnable().await? else {
            self.db
                .record_action(
                    None,
                    "main_agent",
                    "request_next_task_run",
                    serde_json::json!({ "selected_task_id": null }),
                )
                .await?;
            return Ok(Err(
                "No queued task is runnable right now. Check scheduler state for blockers."
                    .to_owned(),
            ));
        };

        let prepared = self.prepare_task_for_immediate_run(task.id).await?;
        if let Ok(prepared_task) = &prepared {
            self.db
                .record_action(
                    Some(prepared_task.id),
                    "main_agent",
                    "request_next_task_run",
                    serde_json::json!({
                        "selected_task_id": prepared_task.id,
                        "title": prepared_task.title,
                        "priority": prepared_task.priority,
                        "queue_position": prepared_task.queue_position,
                    }),
                )
                .await?;
        }

        Ok(prepared)
    }

    pub async fn add_task_dependency(
        &self,
        task_id: TaskId,
        depends_on_task_id: TaskId,
    ) -> anyhow::Result<Task> {
        self.db
            .add_task_dependency(task_id, depends_on_task_id, "main_agent")
            .await?;
        self.db.get_task(task_id).await
    }

    pub async fn remove_task_dependency(
        &self,
        task_id: TaskId,
        depends_on_task_id: TaskId,
    ) -> anyhow::Result<Task> {
        self.db
            .remove_task_dependency(task_id, depends_on_task_id, "main_agent")
            .await?;
        self.db.get_task(task_id).await
    }

    pub async fn add_task_note(&self, task_id: TaskId, content: &str) -> anyhow::Result<TaskNote> {
        self.db.add_task_note(task_id, content, "main_agent").await
    }

    pub async fn list_task_notes(&self, task_id: TaskId) -> anyhow::Result<String> {
        let task = self.db.get_task(task_id).await?;
        let notes = self.db.list_task_notes(task_id).await?;
        self.db
            .record_action(
                Some(task_id),
                "main_agent",
                "list_task_notes",
                serde_json::json!({ "note_count": notes.len() }),
            )
            .await?;

        Ok(format_task_notes(&task, &notes))
    }

    pub async fn add_requested_skills(
        &self,
        task_id: TaskId,
        skill_names: Vec<String>,
    ) -> anyhow::Result<Task> {
        let task = self.db.get_task(task_id).await?;
        let mut requested_skills = task.requested_skills;
        for skill_name in normalize_skill_names(skill_names) {
            if !requested_skills
                .iter()
                .any(|existing| existing.eq_ignore_ascii_case(&skill_name))
            {
                requested_skills.push(skill_name);
            }
        }

        self.db
            .update_task(
                task_id,
                UpdateTask {
                    title: None,
                    description: None,
                    priority: None,
                    requested_skills: Some(requested_skills),
                    schedule: None,
                },
                "main_agent",
            )
            .await
    }

    pub async fn remove_requested_skills(
        &self,
        task_id: TaskId,
        skill_names: Vec<String>,
    ) -> anyhow::Result<Task> {
        let task = self.db.get_task(task_id).await?;
        let skill_names = normalize_skill_names(skill_names);
        let requested_skills = task
            .requested_skills
            .into_iter()
            .filter(|existing| {
                !skill_names
                    .iter()
                    .any(|skill_name| existing.eq_ignore_ascii_case(skill_name))
            })
            .collect();

        self.db
            .update_task(
                task_id,
                UpdateTask {
                    title: None,
                    description: None,
                    priority: None,
                    requested_skills: Some(requested_skills),
                    schedule: None,
                },
                "main_agent",
            )
            .await
    }

    pub async fn create_skill_definition(&self, input: CreateSkill) -> anyhow::Result<Skill> {
        self.db.create_skill(input, "main_agent").await
    }

    pub async fn list_skill_definitions(&self) -> anyhow::Result<Vec<Skill>> {
        let skills = self.db.list_skills().await?;
        self.db
            .record_action(
                None,
                "main_agent",
                "list_skills",
                serde_json::json!({ "count": skills.len() }),
            )
            .await?;
        Ok(skills)
    }

    pub async fn update_skill_definition(
        &self,
        selector: &str,
        input: UpdateSkill,
    ) -> anyhow::Result<Result<Skill, String>> {
        match self.find_skill(selector).await? {
            Ok(skill) => Ok(Ok(self
                .db
                .update_skill(skill.id, input, "main_agent")
                .await?)),
            Err(reply) => Ok(Err(reply)),
        }
    }

    pub async fn delete_skill_definition(
        &self,
        selector: &str,
    ) -> anyhow::Result<Result<Skill, String>> {
        match self.find_skill(selector).await? {
            Ok(skill) => Ok(Ok(self.db.delete_skill(skill.id, "main_agent").await?)),
            Err(reply) => Ok(Err(reply)),
        }
    }

    pub async fn add_task_resource_lock(
        &self,
        task_id: TaskId,
        resource_key: &str,
    ) -> anyhow::Result<TaskResourceLock> {
        self.db
            .add_task_resource_lock(task_id, resource_key, "main_agent")
            .await
    }

    pub async fn remove_task_resource_lock(
        &self,
        task_id: TaskId,
        resource_key: &str,
    ) -> anyhow::Result<TaskResourceLock> {
        self.db
            .remove_task_resource_lock(task_id, resource_key, "main_agent")
            .await
    }

    pub async fn request_user_clarification(
        &self,
        task_id: TaskId,
        question: &str,
    ) -> anyhow::Result<Task> {
        let task = self.db.get_task(task_id).await?;
        let question = question.trim();
        if question.is_empty() {
            anyhow::bail!("clarification question cannot be empty");
        }

        if let Some(conversation_id) = task.conversation_id {
            self.db
                .add_conversation_message(conversation_id, Some(task_id), "assistant", question)
                .await?;
        }

        self.db
            .record_action(
                Some(task_id),
                "main_agent",
                "request_user_clarification",
                serde_json::json!({ "question": question }),
            )
            .await?;
        self.db
            .set_task_status(
                task_id,
                TaskStatus::WaitingForUser,
                "main_agent",
                Some(question),
            )
            .await
    }

    pub async fn reply_to_task(
        &self,
        task_id: TaskId,
        content: &str,
    ) -> anyhow::Result<(Task, ConversationMessage, Option<ConversationMessage>)> {
        let task = self.db.get_task(task_id).await?;
        let content = content.trim();
        if content.is_empty() {
            anyhow::bail!("task reply cannot be empty");
        }
        let Some(conversation_id) = task.conversation_id else {
            anyhow::bail!("task has no conversation");
        };

        let user_message = self
            .db
            .add_conversation_message(conversation_id, Some(task_id), "user", content)
            .await?;
        self.db
            .record_action(
                Some(task_id),
                "main_agent",
                "reply_to_task",
                serde_json::json!({
                    "content_preview": bounded_preview(content, 200),
                    "resumed": task.status == TaskStatus::WaitingForUser,
                }),
            )
            .await?;

        if task.status == TaskStatus::WaitingForUser {
            let resumed = self.resume_task(task_id).await?;
            let assistant_message = self
                .db
                .add_conversation_message(
                    conversation_id,
                    Some(task_id),
                    "assistant",
                    "Thanks, I have the extra context and moved this task back to the queue.",
                )
                .await?;
            Ok((resumed, user_message, Some(assistant_message)))
        } else {
            Ok((task, user_message, None))
        }
    }

    pub async fn list_task_conversation(&self, task_id: TaskId) -> anyhow::Result<String> {
        let task = self.db.get_task(task_id).await?;
        let messages = self.db.list_task_conversation_messages(task_id, 20).await?;
        self.db
            .record_action(
                Some(task_id),
                "main_agent",
                "list_task_conversation",
                serde_json::json!({ "message_count": messages.len() }),
            )
            .await?;

        Ok(format_task_conversation(&task, &messages))
    }

    pub async fn summarize_task_pool(&self) -> anyhow::Result<TaskPoolSummary> {
        let tasks = self.db.list_tasks().await?;
        let mut summary = TaskPoolSummary {
            total: tasks.len(),
            ..Default::default()
        };

        for task in tasks {
            match task.status {
                TaskStatus::Draft => summary.draft += 1,
                TaskStatus::Queued => summary.queued += 1,
                TaskStatus::Running => summary.running += 1,
                TaskStatus::WaitingForUser => summary.waiting_for_user += 1,
                TaskStatus::WaitingForSchedule => summary.waiting_for_schedule += 1,
                TaskStatus::Completed => summary.completed += 1,
                TaskStatus::Failed => summary.failed += 1,
                TaskStatus::Cancelled => summary.cancelled += 1,
                TaskStatus::Paused => summary.paused += 1,
            }
        }

        Ok(summary)
    }

    pub async fn list_task_pool(&self) -> anyhow::Result<Vec<Task>> {
        let tasks = self.db.list_tasks().await?;
        self.db
            .record_action(
                None,
                "main_agent",
                "list_tasks",
                serde_json::json!({ "count": tasks.len() }),
            )
            .await?;
        Ok(tasks)
    }

    pub async fn list_task_pool_by_status(&self, status: TaskStatus) -> anyhow::Result<Vec<Task>> {
        let tasks = self
            .db
            .list_tasks()
            .await?
            .into_iter()
            .filter(|task| task.status == status)
            .collect::<Vec<_>>();
        self.db
            .record_action(
                None,
                "main_agent",
                "list_tasks_by_status",
                serde_json::json!({ "status": status, "count": tasks.len() }),
            )
            .await?;
        Ok(tasks)
    }

    pub async fn list_waiting_for_user_tasks(&self) -> anyhow::Result<String> {
        let tasks = self
            .db
            .list_tasks()
            .await?
            .into_iter()
            .filter(|task| task.status == TaskStatus::WaitingForUser)
            .collect::<Vec<_>>();
        let mut summaries = Vec::new();
        for task in tasks {
            let latest_question = if let Some(conversation_id) = task.conversation_id {
                self.db
                    .list_conversation_messages(conversation_id, 20)
                    .await?
                    .into_iter()
                    .filter(|message| message.role == "assistant")
                    .max_by_key(|message| message.created_at)
                    .map(|message| message.content)
            } else {
                None
            };
            summaries.push((task, latest_question));
        }

        self.db
            .record_action(
                None,
                "main_agent",
                "list_waiting_for_user_tasks",
                serde_json::json!({ "count": summaries.len() }),
            )
            .await?;

        Ok(format_waiting_for_user_tasks(&summaries))
    }

    pub async fn list_waiting_for_schedule_tasks(&self) -> anyhow::Result<String> {
        let mut tasks = self
            .db
            .list_tasks()
            .await?
            .into_iter()
            .filter(|task| task.status == TaskStatus::WaitingForSchedule)
            .collect::<Vec<_>>();
        tasks.sort_by(|left, right| {
            left.next_run_at
                .cmp(&right.next_run_at)
                .then_with(|| right.priority.cmp(&left.priority))
                .then_with(|| left.queue_position.cmp(&right.queue_position))
        });

        self.db
            .record_action(
                None,
                "main_agent",
                "list_waiting_for_schedule_tasks",
                serde_json::json!({ "count": tasks.len() }),
            )
            .await?;

        Ok(format_waiting_for_schedule_tasks(&tasks))
    }

    pub async fn explain_task_pool_state(&self) -> anyhow::Result<String> {
        let tasks = self.db.list_tasks().await?;
        self.db
            .record_action(
                None,
                "main_agent",
                "explain_task_pool_state",
                serde_json::json!({ "count": tasks.len() }),
            )
            .await?;

        Ok(format_task_pool_explanation(&tasks))
    }

    pub async fn recommend_next_action(&self) -> anyhow::Result<String> {
        let tasks = self.db.list_tasks().await?;
        let next_runnable = self.db.peek_next_runnable().await?;
        self.db
            .record_action(
                None,
                "main_agent",
                "recommend_next_action",
                serde_json::json!({
                    "task_count": tasks.len(),
                    "next_runnable_task_id": next_runnable.as_ref().map(|task| task.id),
                }),
            )
            .await?;

        Ok(format_next_action_recommendation(
            &tasks,
            next_runnable.as_ref(),
        ))
    }

    pub async fn list_main_agent_actions(&self) -> anyhow::Result<String> {
        self.db
            .record_action(
                None,
                "main_agent",
                "list_global_actions",
                serde_json::json!({ "limit": 10 }),
            )
            .await?;
        let actions = self.db.list_global_actions().await?;
        Ok(format_global_action_list(&actions))
    }

    pub async fn explain_task_state(&self, id: TaskId) -> anyhow::Result<String> {
        let task = self.db.get_task(id).await?;
        let dependencies = self.db.list_task_dependencies(id).await?;
        let mut dependency_states = Vec::new();
        for dependency in dependencies {
            let dependency_task = self.db.get_task(dependency.depends_on_task_id).await?;
            dependency_states.push(dependency_task);
        }
        let resource_lock_conflicts = self.resource_lock_conflicts(&task).await?;

        self.db
            .record_action(
                Some(id),
                "main_agent",
                "explain_task_state",
                serde_json::json!({
                    "status": task.status,
                    "dependency_count": dependency_states.len(),
                    "resource_lock_conflict_count": resource_lock_conflicts.len(),
                    "requested_skill_count": task.requested_skills.len(),
                    "matched_skill_count": task.matched_skills.len(),
                    "active_skills": active_skill_names_for_task(&task),
                }),
            )
            .await?;

        Ok(format_task_explanation(
            &task,
            &dependency_states,
            &resource_lock_conflicts,
        ))
    }

    pub async fn list_task_constraints(&self, id: TaskId) -> anyhow::Result<String> {
        let task = self.db.get_task(id).await?;
        let dependencies = self.db.list_task_dependencies(id).await?;
        let mut dependency_states = Vec::new();
        for dependency in dependencies {
            dependency_states.push(self.db.get_task(dependency.depends_on_task_id).await?);
        }
        let resource_locks = self.db.list_task_resource_locks(id).await?;
        let resource_lock_conflicts = self.resource_lock_conflicts(&task).await?;

        self.db
            .record_action(
                Some(id),
                "main_agent",
                "list_task_constraints",
                serde_json::json!({
                    "dependency_count": dependency_states.len(),
                    "resource_lock_count": resource_locks.len(),
                    "resource_lock_conflict_count": resource_lock_conflicts.len(),
                }),
            )
            .await?;

        Ok(format_task_constraints(
            &task,
            &dependency_states,
            &resource_locks,
            &resource_lock_conflicts,
        ))
    }

    async fn resource_lock_conflicts(
        &self,
        task: &Task,
    ) -> anyhow::Result<Vec<ResourceLockConflict>> {
        let candidate_locks = self.db.list_task_resource_locks(task.id).await?;
        let exclusive_resources = candidate_locks
            .iter()
            .filter(|lock| lock.lock_mode == "exclusive")
            .map(|lock| lock.resource_key.as_str())
            .collect::<Vec<_>>();
        if exclusive_resources.is_empty() {
            return Ok(Vec::new());
        }

        let mut conflicts = Vec::new();
        for running_task in
            self.db.list_tasks().await?.into_iter().filter(|candidate| {
                candidate.status == TaskStatus::Running && candidate.id != task.id
            })
        {
            for lock in self.db.list_task_resource_locks(running_task.id).await? {
                if lock.lock_mode == "exclusive"
                    && exclusive_resources
                        .iter()
                        .any(|resource| *resource == lock.resource_key)
                {
                    conflicts.push(ResourceLockConflict {
                        resource_key: lock.resource_key,
                        running_task: running_task.clone(),
                    });
                }
            }
        }

        Ok(conflicts)
    }

    pub async fn list_task_artifacts(&self, id: TaskId) -> anyhow::Result<String> {
        let task = self.db.get_task(id).await?;
        let artifacts = self.db.list_task_artifacts(id).await?;
        self.db
            .record_action(
                Some(id),
                "main_agent",
                "list_task_artifacts",
                serde_json::json!({ "artifact_count": artifacts.len() }),
            )
            .await?;

        Ok(format_task_artifacts(&task, &artifacts))
    }

    pub async fn list_task_history(&self, id: TaskId) -> anyhow::Result<String> {
        let task = self.db.get_task(id).await?;
        let attempts = self.db.list_task_attempts(id).await?;
        let attempt_events = self.db.list_task_attempt_events(id).await?;
        let actions = self.db.list_task_actions(id).await?;
        self.db
            .record_action(
                Some(id),
                "main_agent",
                "list_task_history",
                serde_json::json!({
                    "attempt_count": attempts.len(),
                    "event_count": attempt_events.len(),
                    "action_count": actions.len(),
                }),
            )
            .await?;

        Ok(format_task_history(
            &task,
            &attempts,
            &attempt_events,
            &actions,
        ))
    }

    pub async fn show_task_latest_result(&self, id: TaskId) -> anyhow::Result<String> {
        let task = self.db.get_task(id).await?;
        let attempts = self.db.list_task_attempts(id).await?;
        self.db
            .record_action(
                Some(id),
                "main_agent",
                "show_task_latest_result",
                serde_json::json!({
                    "status": task.status,
                    "has_result_summary": task.result_summary.is_some(),
                    "attempt_count": attempts.len(),
                }),
            )
            .await?;

        Ok(format_task_latest_result(&task, &attempts))
    }

    pub async fn list_task_follow_ups(&self, id: TaskId) -> anyhow::Result<String> {
        let task = self.db.get_task(id).await?;
        let actions = self.db.list_task_actions(id).await?;
        let mut follow_ups = Vec::new();
        for action in actions
            .iter()
            .filter(|action| action.action_type == "create_follow_up_task")
        {
            let Some(id_text) = action.details["follow_up_task_id"].as_str() else {
                continue;
            };
            let Ok(follow_up_id) = id_text.parse::<TaskId>() else {
                continue;
            };
            if let Ok(follow_up) = self.db.get_task(follow_up_id).await {
                follow_ups.push(follow_up);
            }
        }

        self.db
            .record_action(
                Some(id),
                "main_agent",
                "list_task_follow_ups",
                serde_json::json!({ "follow_up_count": follow_ups.len() }),
            )
            .await?;

        Ok(format_task_follow_ups(&task, &follow_ups))
    }

    pub async fn inspect_workspace_status(&self) -> anyhow::Result<String> {
        let cwd = env::current_dir()?;
        let git_status = Command::new("git")
            .args(["status", "--short", "--branch"])
            .current_dir(&cwd)
            .output();

        let (status_text, success) = match git_status {
            Ok(output) if output.status.success() => (
                String::from_utf8_lossy(&output.stdout).trim().to_owned(),
                true,
            ),
            Ok(output) => (
                String::from_utf8_lossy(&output.stderr).trim().to_owned(),
                false,
            ),
            Err(error) => (error.to_string(), false),
        };

        self.db
            .record_action(
                None,
                "main_agent",
                "inspect_workspace_status",
                serde_json::json!({
                    "cwd": cwd.display().to_string(),
                    "git_status_success": success,
                }),
            )
            .await?;

        let status_text = if status_text.is_empty() {
            "no output".to_owned()
        } else {
            status_text
        };

        Ok(format!(
            "Workspace: {}\nGit status:\n{}",
            cwd.display(),
            status_text
        ))
    }

    pub async fn inspect_workspace_file(&self, relative_path: &str) -> anyhow::Result<String> {
        let cwd = env::current_dir()?;
        let path_text = clean_workspace_path(relative_path);
        if path_text.is_empty() {
            self.record_workspace_file_inspection_failure(
                &path_text,
                "workspace file path cannot be empty",
            )
            .await?;
            anyhow::bail!("workspace file path cannot be empty");
        }

        let requested_path = Path::new(&path_text);
        if requested_path.is_absolute() {
            self.record_workspace_file_inspection_failure(
                &path_text,
                "workspace file path must be relative",
            )
            .await?;
            anyhow::bail!("workspace file path must be relative");
        }

        let workspace_root = cwd.canonicalize()?;
        let file_path = cwd.join(requested_path);
        let canonical_file_path = match file_path.canonicalize() {
            Ok(path) => path,
            Err(error) => {
                self.record_workspace_file_inspection_failure(&path_text, &error.to_string())
                    .await?;
                return Err(error.into());
            }
        };
        if !canonical_file_path.starts_with(&workspace_root) {
            self.record_workspace_file_inspection_failure(
                &path_text,
                "workspace file path must stay inside the current workspace",
            )
            .await?;
            anyhow::bail!("workspace file path must stay inside the current workspace");
        }

        let metadata = match fs::metadata(&canonical_file_path) {
            Ok(metadata) => metadata,
            Err(error) => {
                self.record_workspace_file_inspection_failure(&path_text, &error.to_string())
                    .await?;
                return Err(error.into());
            }
        };
        if !metadata.is_file() {
            self.record_workspace_file_inspection_failure(
                &path_text,
                "workspace path is not a file",
            )
            .await?;
            anyhow::bail!("workspace path is not a file");
        }

        const MAX_FILE_PREVIEW_BYTES: usize = 8 * 1024;
        let bytes = match fs::read(&canonical_file_path) {
            Ok(bytes) => bytes,
            Err(error) => {
                self.record_workspace_file_inspection_failure(&path_text, &error.to_string())
                    .await?;
                return Err(error.into());
            }
        };
        let truncated = bytes.len() > MAX_FILE_PREVIEW_BYTES;
        let preview_len = bytes.len().min(MAX_FILE_PREVIEW_BYTES);
        let preview = String::from_utf8_lossy(&bytes[..preview_len]);

        self.db
            .record_action(
                None,
                "main_agent",
                "inspect_workspace_file",
                serde_json::json!({
                    "path": path_text,
                    "bytes": bytes.len(),
                    "preview_bytes": preview_len,
                    "truncated": truncated,
                }),
            )
            .await?;

        let truncation_note = if truncated { " (truncated)" } else { "" };

        Ok(format!(
            "File: {}\nBytes shown: {} of {}{}\n{}",
            path_text,
            preview_len,
            bytes.len(),
            truncation_note,
            preview
        ))
    }

    pub async fn inspect_workspace_directory(&self, relative_path: &str) -> anyhow::Result<String> {
        let cwd = env::current_dir()?;
        let path_text = clean_workspace_path(relative_path);
        let display_path = if path_text.is_empty() {
            ".".to_owned()
        } else {
            path_text.clone()
        };
        let requested_path = Path::new(&display_path);
        if requested_path.is_absolute() {
            self.record_workspace_directory_inspection_failure(
                &display_path,
                "workspace directory path must be relative",
            )
            .await?;
            anyhow::bail!("workspace directory path must be relative");
        }

        let workspace_root = cwd.canonicalize()?;
        let directory_path = cwd.join(requested_path);
        let canonical_directory_path = match directory_path.canonicalize() {
            Ok(path) => path,
            Err(error) => {
                self.record_workspace_directory_inspection_failure(
                    &display_path,
                    &error.to_string(),
                )
                .await?;
                return Err(error.into());
            }
        };
        if !canonical_directory_path.starts_with(&workspace_root) {
            self.record_workspace_directory_inspection_failure(
                &display_path,
                "workspace directory path must stay inside the current workspace",
            )
            .await?;
            anyhow::bail!("workspace directory path must stay inside the current workspace");
        }

        let metadata = match fs::metadata(&canonical_directory_path) {
            Ok(metadata) => metadata,
            Err(error) => {
                self.record_workspace_directory_inspection_failure(
                    &display_path,
                    &error.to_string(),
                )
                .await?;
                return Err(error.into());
            }
        };
        if !metadata.is_dir() {
            self.record_workspace_directory_inspection_failure(
                &display_path,
                "workspace path is not a directory",
            )
            .await?;
            anyhow::bail!("workspace path is not a directory");
        }

        let mut entries = Vec::new();
        let read_dir = match fs::read_dir(&canonical_directory_path) {
            Ok(read_dir) => read_dir,
            Err(error) => {
                self.record_workspace_directory_inspection_failure(
                    &display_path,
                    &error.to_string(),
                )
                .await?;
                return Err(error.into());
            }
        };
        for entry in read_dir {
            let entry = match entry {
                Ok(entry) => entry,
                Err(error) => {
                    self.record_workspace_directory_inspection_failure(
                        &display_path,
                        &error.to_string(),
                    )
                    .await?;
                    return Err(error.into());
                }
            };
            let name = entry.file_name().to_string_lossy().to_string();
            let metadata = match entry.metadata() {
                Ok(metadata) => metadata,
                Err(error) => {
                    self.record_workspace_directory_inspection_failure(
                        &display_path,
                        &error.to_string(),
                    )
                    .await?;
                    return Err(error.into());
                }
            };
            let kind = if metadata.is_dir() {
                "dir"
            } else if metadata.is_file() {
                "file"
            } else if metadata.file_type().is_symlink() {
                "symlink"
            } else {
                "other"
            };
            entries.push((kind.to_owned(), name, metadata.len()));
        }
        entries.sort_by(|left, right| {
            entry_kind_rank(&left.0)
                .cmp(&entry_kind_rank(&right.0))
                .then_with(|| left.1.to_lowercase().cmp(&right.1.to_lowercase()))
        });

        const MAX_DIRECTORY_ENTRIES: usize = 50;
        let shown = entries.len().min(MAX_DIRECTORY_ENTRIES);
        let truncated = entries.len() > shown;
        self.db
            .record_action(
                None,
                "main_agent",
                "inspect_workspace_directory",
                serde_json::json!({
                    "path": display_path,
                    "entry_count": entries.len(),
                    "shown": shown,
                    "truncated": truncated,
                }),
            )
            .await?;

        Ok(format_workspace_directory(
            &display_path,
            &entries,
            shown,
            truncated,
        ))
    }

    async fn record_workspace_file_inspection_failure(
        &self,
        path: &str,
        error: &str,
    ) -> anyhow::Result<()> {
        self.db
            .record_action(
                None,
                "main_agent",
                "inspect_workspace_file_failed",
                serde_json::json!({
                    "path": path,
                    "error": error,
                }),
            )
            .await?;
        Ok(())
    }

    async fn record_workspace_directory_inspection_failure(
        &self,
        path: &str,
        error: &str,
    ) -> anyhow::Result<()> {
        self.db
            .record_action(
                None,
                "main_agent",
                "inspect_workspace_directory_failed",
                serde_json::json!({
                    "path": path,
                    "error": error,
                }),
            )
            .await?;
        Ok(())
    }

    pub async fn create_memory(&self, scope: String, content: String) -> anyhow::Result<Memory> {
        self.db
            .create_memory(
                CreateMemory {
                    scope,
                    content,
                    source_task_id: None,
                    status: MemoryStatus::Approved,
                    confidence: 1.0,
                },
                "main_agent",
            )
            .await
    }

    pub async fn approve_memory(&self, id: MemoryId) -> anyhow::Result<Memory> {
        self.db
            .set_memory_status(id, MemoryStatus::Approved, "main_agent")
            .await
    }

    pub async fn reject_memory(&self, id: MemoryId) -> anyhow::Result<Memory> {
        self.db
            .set_memory_status(id, MemoryStatus::Rejected, "main_agent")
            .await
    }

    pub async fn set_all_pending_memories_status(
        &self,
        status: MemoryStatus,
    ) -> anyhow::Result<Vec<Memory>> {
        if status == MemoryStatus::Pending {
            anyhow::bail!("bulk memory review requires a terminal memory status");
        }

        let pending = self
            .db
            .list_memories()
            .await?
            .into_iter()
            .filter(|memory| memory.status == MemoryStatus::Pending)
            .collect::<Vec<_>>();
        let mut updated = Vec::with_capacity(pending.len());
        for memory in pending {
            updated.push(
                self.db
                    .set_memory_status(memory.id, status, "main_agent")
                    .await?,
            );
        }
        self.db
            .record_action(
                None,
                "main_agent",
                "bulk_set_memory_status",
                serde_json::json!({
                    "status": status,
                    "count": updated.len(),
                }),
            )
            .await?;

        Ok(updated)
    }

    pub async fn set_task_pending_memories_status(
        &self,
        task_id: TaskId,
        status: MemoryStatus,
    ) -> anyhow::Result<Vec<Memory>> {
        if status == MemoryStatus::Pending {
            anyhow::bail!("task memory review requires a terminal memory status");
        }

        let pending = self
            .db
            .list_task_memories(task_id)
            .await?
            .into_iter()
            .filter(|memory| memory.status == MemoryStatus::Pending)
            .collect::<Vec<_>>();
        let mut updated = Vec::with_capacity(pending.len());
        for memory in pending {
            updated.push(
                self.db
                    .set_memory_status(memory.id, status, "main_agent")
                    .await?,
            );
        }
        self.db
            .record_action(
                Some(task_id),
                "main_agent",
                "bulk_set_task_memory_status",
                serde_json::json!({
                    "status": status,
                    "count": updated.len(),
                }),
            )
            .await?;

        Ok(updated)
    }

    pub async fn update_memory(&self, id: MemoryId, input: UpdateMemory) -> anyhow::Result<Memory> {
        self.db.update_memory(id, input, "main_agent").await
    }

    pub async fn delete_memory(&self, id: MemoryId) -> anyhow::Result<Memory> {
        self.db.delete_memory(id, "main_agent").await
    }

    pub async fn list_memories_for_review(
        &self,
        filter: MemoryListFilter,
    ) -> anyhow::Result<String> {
        let memories = self.db.list_memories().await?;
        let filtered = memories
            .into_iter()
            .filter(|memory| filter.matches(memory.status))
            .collect::<Vec<_>>();
        self.db
            .record_action(
                None,
                "main_agent",
                "list_memories",
                serde_json::json!({
                    "filter": filter.as_str(),
                    "count": filtered.len(),
                }),
            )
            .await?;

        Ok(format_memory_list(filter, &filtered))
    }

    pub async fn list_task_memories(&self, task_id: TaskId) -> anyhow::Result<String> {
        let task = self.db.get_task(task_id).await?;
        let memories = self.db.list_task_memories(task_id).await?;
        self.db
            .record_action(
                Some(task_id),
                "main_agent",
                "list_task_memories",
                serde_json::json!({ "memory_count": memories.len() }),
            )
            .await?;

        Ok(format_task_memories(&task, &memories))
    }

    pub async fn inspect_scheduler_state(&self) -> anyhow::Result<String> {
        let tasks = self.db.list_tasks().await?;
        let running_tasks = tasks
            .iter()
            .filter(|task| task.status == TaskStatus::Running)
            .cloned()
            .collect::<Vec<_>>();
        let waiting_for_user_tasks = tasks
            .iter()
            .filter(|task| task.status == TaskStatus::WaitingForUser)
            .cloned()
            .collect::<Vec<_>>();
        let queued_count = tasks
            .iter()
            .filter(|task| task.status == TaskStatus::Queued)
            .count();
        let waiting_for_schedule_count = tasks
            .iter()
            .filter(|task| task.status == TaskStatus::WaitingForSchedule)
            .count();
        let next_queued_task = tasks
            .iter()
            .filter(|task| task.status == TaskStatus::Queued)
            .min_by(|left, right| {
                right
                    .priority
                    .cmp(&left.priority)
                    .then_with(|| left.queue_position.cmp(&right.queue_position))
                    .then_with(|| left.created_at.cmp(&right.created_at))
            })
            .cloned();
        let next_runnable_task = self.db.peek_next_runnable().await?;

        self.db
            .record_action(
                None,
                "main_agent",
                "inspect_scheduler_state",
                serde_json::json!({
                    "running_count": running_tasks.len(),
                    "queued_count": queued_count,
                    "waiting_for_user_count": waiting_for_user_tasks.len(),
                    "waiting_for_schedule_count": waiting_for_schedule_count,
                    "next_queued_task_id": next_queued_task.as_ref().map(|task| task.id),
                    "next_runnable_task_id": next_runnable_task.as_ref().map(|task| task.id),
                }),
            )
            .await?;

        Ok(format_scheduler_state(
            &running_tasks,
            next_queued_task.as_ref(),
            next_runnable_task.as_ref(),
            queued_count,
            &waiting_for_user_tasks,
            waiting_for_schedule_count,
        ))
    }

    pub async fn main_conversation_messages(
        &self,
        limit: i64,
    ) -> anyhow::Result<Vec<ConversationMessage>> {
        let conversation = self.db.get_or_create_main_conversation().await?;
        self.db
            .list_conversation_messages(conversation.id, limit)
            .await
    }

    pub async fn handle_user_message(
        &self,
        input: MainAgentMessageInput,
    ) -> anyhow::Result<MainAgentMessageResponse> {
        let conversation = self.db.get_or_create_main_conversation().await?;
        let user_message = self
            .db
            .add_conversation_message(conversation.id, None, "user", &input.content)
            .await?;

        let intent = self.resolve_intent(&input.content).await?;
        let mut scheduler_tick_requested = false;
        let mut changed_tasks = Vec::new();
        let deterministic_reply = match intent {
            MainAgentIntent::SplitTasks { titles } => {
                for title in titles {
                    let task = self
                        .create_task(CreateTask {
                            title: title.clone(),
                            description: format!("Split from user request: {}", input.content),
                            task_type: TaskType::OneOff,
                            priority: 0,
                            requested_skills: Vec::new(),
                            schedule: None,
                            created_by: "user".to_owned(),
                        })
                        .await?;
                    changed_tasks.push(task);
                }
                format!("Created {} split task(s).", changed_tasks.len())
            }
            MainAgentIntent::CreateTask {
                title,
                description,
                task_type,
                priority,
                interval_seconds,
                requested_skills,
            } => {
                let requested_skills =
                    enrich_requested_skills_for_task_text(requested_skills, &description);
                let requested_skills_for_reply = requested_skills.clone();
                let inferred_resource_locks = infer_resource_locks_for_task_text(&description);
                let schedule = match task_type {
                    TaskType::OneOff => None,
                    TaskType::Recurring => Some(
                        serde_json::json!({ "interval_seconds": interval_seconds.unwrap_or(300) }),
                    ),
                };
                let task = self
                    .create_task(CreateTask {
                        title,
                        description,
                        task_type,
                        priority,
                        requested_skills,
                        schedule,
                        created_by: "user".to_owned(),
                    })
                    .await?;
                for resource_key in &inferred_resource_locks {
                    self.add_task_resource_lock(task.id, resource_key).await?;
                }
                let mut reply = format!(
                    "Created task '{}'. Status: {}, priority: {}.",
                    task.title, task.status, task.priority
                );
                if !requested_skills_for_reply.is_empty() {
                    reply.push_str(&format!(
                        " Requested skills: {}.",
                        format_skill_list(&requested_skills_for_reply)
                    ));
                }
                if !inferred_resource_locks.is_empty() {
                    reply.push_str(&format!(
                        " Resource locks: {}.",
                        inferred_resource_locks.join(", ")
                    ));
                }
                changed_tasks.push(task);
                reply
            }
            MainAgentIntent::PauseTask { selector } => match self.find_task(&selector).await? {
                Ok(task) => {
                    let task = self.pause_task(task.id).await?;
                    let reply = format!("Paused task '{}'.", task.title);
                    changed_tasks.push(task);
                    reply
                }
                Err(reply) => reply,
            },
            MainAgentIntent::ResumeTask { selector } => match self.find_task(&selector).await? {
                Ok(task) => {
                    let task = self.resume_task(task.id).await?;
                    let reply = format!("Resumed task '{}'.", task.title);
                    changed_tasks.push(task);
                    reply
                }
                Err(reply) => reply,
            },
            MainAgentIntent::CancelTask { selector } => match self.find_task(&selector).await? {
                Ok(task) => {
                    let task = self.cancel_task(task.id).await?;
                    let reply = format!("Cancelled task '{}'.", task.title);
                    changed_tasks.push(task);
                    reply
                }
                Err(reply) => reply,
            },
            MainAgentIntent::DeleteTask { selector } => match self.find_task(&selector).await? {
                Ok(task) => {
                    let task = self.delete_task(task.id).await?;
                    let reply = format!("Deleted task '{}'.", task.title);
                    changed_tasks.push(task);
                    reply
                }
                Err(reply) => reply,
            },
            MainAgentIntent::CompleteTask { selector, summary } => {
                match self.find_task(&selector).await? {
                    Ok(task) => {
                        let task = self.complete_task(task.id, &summary).await?;
                        let reply = format!(
                            "Marked task '{}' as {}. Summary: {}",
                            task.title, task.status, summary
                        );
                        changed_tasks.push(task);
                        reply
                    }
                    Err(reply) => reply,
                }
            }
            MainAgentIntent::FailTask { selector, error } => match self.find_task(&selector).await?
            {
                Ok(task) => {
                    let task = self.fail_task(task.id, &error).await?;
                    let reply = format!("Marked task '{}' as failed. Reason: {}", task.title, error);
                    changed_tasks.push(task);
                    reply
                }
                Err(reply) => reply,
            },
            MainAgentIntent::RetryTask { selector, reason } => {
                match self.find_task(&selector).await? {
                    Ok(task) => {
                        let task = self.retry_task(task.id, &reason).await?;
                        let reply = format!(
                            "Requeued task '{}' for retry at queue position {}. Reason: {}",
                            task.title, task.queue_position, reason
                        );
                        changed_tasks.push(task);
                        reply
                    }
                    Err(reply) => reply,
                }
            }
            MainAgentIntent::RunTaskNow { selector } => match self.find_task(&selector).await? {
                Ok(task) => match self.prepare_task_for_immediate_run(task.id).await? {
                    Ok(task) => {
                        scheduler_tick_requested = true;
                        let reply = format!(
                            "Moved task '{}' to the front of runnable work and requested a scheduler scan.",
                            task.title
                        );
                        changed_tasks.push(task);
                        reply
                    }
                    Err(reply) => reply,
                },
                Err(reply) => reply,
            },
            MainAgentIntent::RunNextTask => match self.prepare_next_task_for_run().await? {
                Ok(task) => {
                    scheduler_tick_requested = true;
                    let reply = format!(
                        "Selected next runnable task '{}' and requested a scheduler scan.",
                        task.title
                    );
                    changed_tasks.push(task);
                    reply
                }
                Err(reply) => reply,
            },
            MainAgentIntent::UpdateTaskDetails {
                selector,
                title,
                description,
            } => match self.find_task(&selector).await? {
                Ok(task) => {
                    let previous_title = task.title.clone();
                    let task = self
                        .update_task(
                            task.id,
                            UpdateTask {
                                title,
                                description,
                                priority: None,
                                requested_skills: None,
                                schedule: None,
                            },
                        )
                        .await?;
                    let reply = format!(
                        "Updated task '{}'. Title: '{}'. Description: {}",
                        previous_title, task.title, task.description
                    );
                    changed_tasks.push(task);
                    reply
                }
                Err(reply) => reply,
            },
            MainAgentIntent::ReprioritizeTask { selector, priority } => {
                match self.find_task(&selector).await? {
                    Ok(task) => {
                        let task = self.reprioritize_task(task.id, priority).await?;
                        let reply =
                            format!("Set task '{}' priority to {}.", task.title, task.priority);
                        changed_tasks.push(task);
                        reply
                    }
                    Err(reply) => reply,
                }
            }
            MainAgentIntent::ReorderTask {
                selector,
                queue_position,
            } => match self.find_task(&selector).await? {
                Ok(task) => {
                    let task = self.reorder_task(task.id, queue_position).await?;
                    let reply = format!(
                        "Moved task '{}' to queue position {}.",
                        task.title, task.queue_position
                    );
                    changed_tasks.push(task);
                    reply
                }
                Err(reply) => reply,
            },
            MainAgentIntent::ConvertTaskType {
                selector,
                task_type,
                interval_seconds,
            } => match self.find_task(&selector).await? {
                Ok(task) => {
                    let task = self
                        .convert_task_type(task.id, task_type, interval_seconds)
                        .await?;
                    let reply = format!("Converted task '{}' to {}.", task.title, task.task_type);
                    changed_tasks.push(task);
                    reply
                }
                Err(reply) => reply,
            },
            MainAgentIntent::UpdateTaskSchedule {
                selector,
                interval_seconds,
            } => match self.find_task(&selector).await? {
                Ok(task) if task.task_type != TaskType::Recurring => {
                    format!("Task '{}' is one-off. Convert it to recurring before setting a recurring interval.", task.title)
                }
                Ok(task) => {
                    let task = self.update_task_schedule(task.id, interval_seconds).await?;
                    let reply = format!(
                        "Updated task '{}' recurring interval to {} seconds.",
                        task.title, interval_seconds
                    );
                    changed_tasks.push(task);
                    reply
                }
                Err(reply) => reply,
            },
            MainAgentIntent::AddTaskDependency {
                selector,
                depends_on_selector,
            } => {
                let task = self.find_task(&selector).await?;
                let depends_on_task = self.find_task(&depends_on_selector).await?;
                match (task, depends_on_task) {
                    (Ok(task), Ok(depends_on_task)) => {
                        let task = self
                            .add_task_dependency(task.id, depends_on_task.id)
                            .await?;
                        let reply = format!(
                            "Added dependency: '{}' now waits for '{}'.",
                            task.title, depends_on_task.title
                        );
                        changed_tasks.push(task);
                        reply
                    }
                    (Err(reply), _) | (_, Err(reply)) => reply,
                }
            }
            MainAgentIntent::RemoveTaskDependency {
                selector,
                depends_on_selector,
            } => {
                let task = self.find_task(&selector).await?;
                let depends_on_task = self.find_task(&depends_on_selector).await?;
                match (task, depends_on_task) {
                    (Ok(task), Ok(depends_on_task)) => {
                        let task = self
                            .remove_task_dependency(task.id, depends_on_task.id)
                            .await?;
                        let reply = format!(
                            "Removed dependency: '{}' no longer waits for '{}'.",
                            task.title, depends_on_task.title
                        );
                        changed_tasks.push(task);
                        reply
                    }
                    (Err(reply), _) | (_, Err(reply)) => reply,
                }
            }
            MainAgentIntent::AddTaskNote { selector, content } => {
                match self.find_task(&selector).await? {
                    Ok(task) => {
                        let note = self.add_task_note(task.id, &content).await?;
                        changed_tasks.push(task.clone());
                        format!("Added note to '{}': {}", task.title, note.content)
                    }
                    Err(reply) => reply,
                }
            }
            MainAgentIntent::ListTaskNotes { selector } => match self.find_task(&selector).await? {
                Ok(task) => self.list_task_notes(task.id).await?,
                Err(reply) => reply,
            },
            MainAgentIntent::AddRequestedSkills {
                selector,
                skill_names,
            } => match self.find_task(&selector).await? {
                Ok(task) => {
                    let task = self.add_requested_skills(task.id, skill_names).await?;
                    let reply = format!(
                        "Updated requested skills for '{}': {}.",
                        task.title,
                        format_skill_list(&task.requested_skills)
                    );
                    changed_tasks.push(task);
                    reply
                }
                Err(reply) => reply,
            },
            MainAgentIntent::RemoveRequestedSkills {
                selector,
                skill_names,
            } => match self.find_task(&selector).await? {
                Ok(task) => {
                    let task = self.remove_requested_skills(task.id, skill_names).await?;
                    let reply = format!(
                        "Updated requested skills for '{}': {}.",
                        task.title,
                        format_skill_list(&task.requested_skills)
                    );
                    changed_tasks.push(task);
                    reply
                }
                Err(reply) => reply,
            },
            MainAgentIntent::AddResourceLock {
                selector,
                resource_key,
            } => match self.find_task(&selector).await? {
                Ok(task) => {
                    let resource_lock = self
                        .add_task_resource_lock(task.id, &resource_key)
                        .await?;
                    changed_tasks.push(task.clone());
                    format!(
                        "Added resource lock to '{}': {}.",
                        task.title, resource_lock.resource_key
                    )
                }
                Err(reply) => reply,
            },
            MainAgentIntent::RemoveResourceLock {
                selector,
                resource_key,
            } => match self.find_task(&selector).await? {
                Ok(task) => {
                    let resource_lock = self
                        .remove_task_resource_lock(task.id, &resource_key)
                        .await?;
                    changed_tasks.push(task.clone());
                    format!(
                        "Removed resource lock from '{}': {}.",
                        task.title, resource_lock.resource_key
                    )
                }
                Err(reply) => reply,
            },
            MainAgentIntent::RequestClarification { selector, question } => {
                match self.find_task(&selector).await? {
                    Ok(task) => {
                        let task = self.request_user_clarification(task.id, &question).await?;
                        changed_tasks.push(task.clone());
                        format!(
                            "Requested user clarification for '{}': {}",
                            task.title, question
                        )
                    }
                    Err(reply) => reply,
                }
            }
            MainAgentIntent::ReplyToTask { selector, content } => {
                match self.find_task(&selector).await? {
                    Ok(task) => {
                        let (task, _, resumed) = self.reply_to_task(task.id, &content).await?;
                        let reply = if resumed.is_some() {
                            scheduler_tick_requested = true;
                            format!(
                                "Sent your reply to '{}' and moved it back to the queue, then requested a scheduler scan.",
                                task.title
                            )
                        } else {
                            format!("Sent your reply to '{}'.", task.title)
                        };
                        changed_tasks.push(task);
                        reply
                    }
                    Err(reply) => reply,
                }
            }
            MainAgentIntent::ListTaskConversation { selector } => {
                match self.find_task(&selector).await? {
                    Ok(task) => self.list_task_conversation(task.id).await?,
                    Err(reply) => reply,
                }
            }
            MainAgentIntent::Summarize => {
                let summary = self.summarize_task_pool().await?;
                format!(
                    "Task pool: {} total, {} queued, {} running, {} waiting for user, {} waiting for schedule, {} completed, {} failed, {} paused.",
                    summary.total,
                    summary.queued,
                    summary.running,
                    summary.waiting_for_user,
                    summary.waiting_for_schedule,
                    summary.completed,
                    summary.failed,
                    summary.paused
                )
            }
            MainAgentIntent::ListTasks => {
                let tasks = self.list_task_pool().await?;
                format_task_list(&tasks)
            }
            MainAgentIntent::ListTasksByStatus { status } => {
                let tasks = self.list_task_pool_by_status(status).await?;
                format_task_list_by_status(status, &tasks)
            }
            MainAgentIntent::ListWaitingForUserTasks => self.list_waiting_for_user_tasks().await?,
            MainAgentIntent::ListWaitingForScheduleTasks => {
                self.list_waiting_for_schedule_tasks().await?
            }
            MainAgentIntent::ShowSchedulerState => self.inspect_scheduler_state().await?,
            MainAgentIntent::ListGlobalActions => self.list_main_agent_actions().await?,
            MainAgentIntent::ExplainTaskPool => self.explain_task_pool_state().await?,
            MainAgentIntent::RecommendNextAction => self.recommend_next_action().await?,
            MainAgentIntent::ExplainTask { selector } => match self.find_task(&selector).await? {
                Ok(task) => self.explain_task_state(task.id).await?,
                Err(reply) => reply,
            },
            MainAgentIntent::ListTaskConstraints { selector } => {
                match self.find_task(&selector).await? {
                    Ok(task) => self.list_task_constraints(task.id).await?,
                    Err(reply) => reply,
                }
            }
            MainAgentIntent::ListTaskArtifacts { selector } => {
                match self.find_task(&selector).await? {
                    Ok(task) => self.list_task_artifacts(task.id).await?,
                    Err(reply) => reply,
                }
            }
            MainAgentIntent::ListTaskHistory { selector } => match self.find_task(&selector).await?
            {
                Ok(task) => self.list_task_history(task.id).await?,
                Err(reply) => reply,
            },
            MainAgentIntent::ShowTaskLatestResult { selector } => {
                match self.find_task(&selector).await? {
                    Ok(task) => self.show_task_latest_result(task.id).await?,
                    Err(reply) => reply,
                }
            }
            MainAgentIntent::ListTaskFollowUps { selector } => {
                match self.find_task(&selector).await? {
                    Ok(task) => self.list_task_follow_ups(task.id).await?,
                    Err(reply) => reply,
                }
            }
            MainAgentIntent::InspectWorkspace => match self.inspect_workspace_status().await {
                Ok(reply) => reply,
                Err(error) => format!("I could not inspect the workspace status: {error}"),
            },
            MainAgentIntent::InspectWorkspaceFile { path } => {
                match self.inspect_workspace_file(&path).await {
                    Ok(reply) => reply,
                    Err(error) => {
                        format!("I could not inspect workspace file '{}': {error}", path)
                    }
                }
            }
            MainAgentIntent::InspectWorkspaceDirectory { path } => {
                match self.inspect_workspace_directory(&path).await {
                    Ok(reply) => reply,
                    Err(error) => {
                        format!("I could not inspect workspace directory '{}': {error}", path)
                    }
                }
            }
            MainAgentIntent::CreateMemory { scope, content } => {
                let memory = self.create_memory(scope, content).await?;
                format!(
                    "Remembered [{}]: {}",
                    memory.scope,
                    bounded_preview(&memory.content, 240)
                )
            }
            MainAgentIntent::ApproveMemory { selector } => match self.find_memory(&selector).await?
            {
                Ok(memory) => {
                    let memory = self.approve_memory(memory.id).await?;
                    format!(
                        "Approved memory '{}': {}",
                        memory.id.to_string().chars().take(8).collect::<String>(),
                        memory.content
                    )
                }
                Err(reply) => reply,
            },
            MainAgentIntent::RejectMemory { selector } => match self.find_memory(&selector).await? {
                Ok(memory) => {
                    let memory = self.reject_memory(memory.id).await?;
                    format!(
                        "Rejected memory '{}': {}",
                        memory.id.to_string().chars().take(8).collect::<String>(),
                        memory.content
                    )
                }
                Err(reply) => reply,
            },
            MainAgentIntent::BulkReviewMemories { status } => {
                let memories = self.set_all_pending_memories_status(status).await?;
                format!(
                    "{} {} pending memory candidate(s).",
                    match status {
                        MemoryStatus::Approved => "Approved",
                        MemoryStatus::Rejected => "Rejected",
                        MemoryStatus::Pending => "Updated",
                    },
                    memories.len()
                )
            }
            MainAgentIntent::BulkReviewTaskMemories { selector, status } => {
                match self.find_task(&selector).await? {
                    Ok(task) => {
                        let memories = self.set_task_pending_memories_status(task.id, status).await?;
                        format!(
                            "{} {} pending memory candidate(s) from '{}'.",
                            match status {
                                MemoryStatus::Approved => "Approved",
                                MemoryStatus::Rejected => "Rejected",
                                MemoryStatus::Pending => "Updated",
                            },
                            memories.len(),
                            task.title
                        )
                    }
                    Err(reply) => reply,
                }
            }
            MainAgentIntent::UpdateMemory { selector, input } => match self.find_memory(&selector).await? {
                Ok(memory) => {
                    let memory = self.update_memory(memory.id, input).await?;
                    format!(
                        "Updated memory '{}': {}",
                        memory.id.to_string().chars().take(8).collect::<String>(),
                        memory.content
                    )
                }
                Err(reply) => reply,
            },
            MainAgentIntent::DeleteMemory { selector } => match self.find_memory(&selector).await? {
                Ok(memory) => {
                    let memory = self.delete_memory(memory.id).await?;
                    format!(
                        "Deleted memory '{}': {}",
                        memory.id.to_string().chars().take(8).collect::<String>(),
                        memory.content
                    )
                }
                Err(reply) => reply,
            },
            MainAgentIntent::ListMemories { filter } => self.list_memories_for_review(filter).await?,
            MainAgentIntent::ListTaskMemories { selector } => {
                match self.find_task(&selector).await? {
                    Ok(task) => self.list_task_memories(task.id).await?,
                    Err(reply) => reply,
                }
            }
            MainAgentIntent::CreateSkillDefinition { input } => {
                let skill = self.create_skill_definition(input).await?;
                format!(
                    "Created skill '{}'. Triggers: {}. Tools: {}. Resource: {}.",
                    skill.name,
                    format_skill_list(&skill.trigger_rules),
                    format_skill_list(&skill.tool_subset),
                    skill.resource_path.as_deref().unwrap_or("none")
                )
            }
            MainAgentIntent::ListSkillDefinitions => {
                let skills = self.list_skill_definitions().await?;
                format_skill_definitions(&skills)
            }
            MainAgentIntent::UpdateSkillDefinition { selector, input } => {
                match self.update_skill_definition(&selector, input).await? {
                    Ok(skill) => format!(
                        "Updated skill '{}'. Triggers: {}. Tools: {}. Resource: {}.",
                        skill.name,
                        format_skill_list(&skill.trigger_rules),
                        format_skill_list(&skill.tool_subset),
                        skill.resource_path.as_deref().unwrap_or("none")
                    ),
                    Err(reply) => reply,
                }
            }
            MainAgentIntent::DeleteSkillDefinition { selector } => {
                match self.delete_skill_definition(&selector).await? {
                    Ok(skill) => format!("Deleted skill '{}'.", skill.name),
                    Err(reply) => reply,
                }
            }
            MainAgentIntent::RunSchedulerTick => {
                scheduler_tick_requested = true;
                "Scheduler scan requested. I will ask the scheduler to check the task pool now."
                    .to_owned()
            }
            MainAgentIntent::Help => {
                "I can create tasks, split goals into tasks, list tasks by status, list tasks waiting for user input or schedule, create/list/delete skills, show task artifacts, show task history, show task latest result, show task follow-up tasks, show task notes, show task constraints, show task memories, show task conversation, remember durable preferences, show memory candidates, show main-agent audit actions, explain task state, recommend the next action, inspect workspace status, preview workspace files, list workspace directories, request user clarification, reply to blocked tasks, run the next runnable task, run a selected task now, pause/resume/cancel/delete/complete/fail/retry tasks, update task title/description, set priority, reorder the queue, add notes, add/remove requested skills, add/remove task dependencies, add/remove resource locks, list/approve/reject/update/delete memory candidates, approve/reject all pending memories from one task, run a scheduler scan, convert tasks between one-off and recurring, or summarize the task pool. Example: split goal: investigate issue; write fix; run tests.".to_owned()
            }
        };

        let reply = self
            .maybe_advise_reply(
                &input.content,
                deterministic_reply,
                scheduler_tick_requested,
                &changed_tasks,
            )
            .await?;

        let assistant_message = self
            .db
            .add_conversation_message(conversation.id, None, "assistant", &reply)
            .await?;

        Ok(MainAgentMessageResponse {
            conversation_id: conversation.id,
            user_message,
            assistant_message,
            changed_tasks,
            scheduler_tick_requested,
        })
    }

    async fn resolve_intent(&self, content: &str) -> anyhow::Result<MainAgentIntent> {
        let deterministic_intent = parse_intent(content);
        if !matches!(deterministic_intent, MainAgentIntent::Help)
            || is_explicit_help_request(content)
        {
            return Ok(deterministic_intent);
        }

        let Some(planner) = &self.planner else {
            return Ok(deterministic_intent);
        };
        let conversation = self.db.get_or_create_main_conversation().await?;
        let context = MainAgentPlanContext {
            user_message: content.to_owned(),
            task_pool_summary: self.summarize_task_pool().await?,
            task_snapshot: self.db.list_tasks().await?,
            recent_messages: self
                .db
                .list_conversation_messages(conversation.id, 8)
                .await?,
        };

        match planner.plan(context).await {
            Ok(Some(plan)) => {
                self.db
                    .record_action(
                        None,
                        "main_agent",
                        "llm_planner_intent",
                        serde_json::json!({ "plan": plan }),
                    )
                    .await?;
                Ok(plan.into_intent())
            }
            Ok(None) => {
                self.db
                    .record_action(
                        None,
                        "main_agent",
                        "llm_planner_no_intent",
                        serde_json::json!({ "fallback": true }),
                    )
                    .await?;
                Ok(deterministic_intent)
            }
            Err(error) => {
                self.db
                    .record_action(
                        None,
                        "main_agent",
                        "llm_planner_failed",
                        serde_json::json!({
                            "error": error.to_string(),
                            "fallback": true,
                        }),
                    )
                    .await?;
                Ok(deterministic_intent)
            }
        }
    }

    async fn maybe_advise_reply(
        &self,
        user_message: &str,
        deterministic_reply: String,
        scheduler_tick_requested: bool,
        changed_tasks: &[Task],
    ) -> anyhow::Result<String> {
        let Some(advisor) = &self.advisor else {
            return Ok(deterministic_reply);
        };

        let conversation = self.db.get_or_create_main_conversation().await?;
        let context = MainAgentAdviceContext {
            user_message: user_message.to_owned(),
            deterministic_reply: deterministic_reply.clone(),
            scheduler_tick_requested,
            changed_tasks: changed_tasks.to_vec(),
            task_pool_summary: self.summarize_task_pool().await?,
            recent_messages: self
                .db
                .list_conversation_messages(conversation.id, 12)
                .await?,
        };

        match advisor.advise(context).await {
            Ok(reply) if !reply.trim().is_empty() => {
                let reply = reply.trim().to_owned();
                self.db
                    .record_action(
                        None,
                        "main_agent",
                        "llm_advisor_reply",
                        serde_json::json!({
                            "used": true,
                            "deterministic_reply": deterministic_reply,
                        }),
                    )
                    .await?;
                Ok(reply)
            }
            Ok(_) => {
                self.db
                    .record_action(
                        None,
                        "main_agent",
                        "llm_advisor_empty_reply",
                        serde_json::json!({ "fallback": true }),
                    )
                    .await?;
                Ok(deterministic_reply)
            }
            Err(error) => {
                self.db
                    .record_action(
                        None,
                        "main_agent",
                        "llm_advisor_failed",
                        serde_json::json!({
                            "error": error.to_string(),
                            "fallback": true,
                        }),
                    )
                    .await?;
                Ok(deterministic_reply)
            }
        }
    }

    async fn find_task(&self, selector: &str) -> anyhow::Result<Result<Task, String>> {
        let needle = selector.trim().to_lowercase();
        if needle.is_empty() {
            return Ok(Err(
                "Please tell me which task to change by title fragment or task id.".to_owned(),
            ));
        }

        let tasks = self.db.list_tasks().await?;
        if let Some(status) = task_selector_status(&needle) {
            let matches = tasks
                .iter()
                .filter(|task| task.status == status)
                .cloned()
                .collect::<Vec<_>>();
            return match matches.as_slice() {
                [task] => Ok(Ok(task.clone())),
                [] => Ok(Err(format!("No {status} task is available."))),
                _ => Ok(Err(format!(
                    "Found {} {status} tasks. Please use a title fragment or task id.",
                    matches.len()
                ))),
            };
        }

        let matches = tasks
            .into_iter()
            .filter(|task| {
                task.id.to_string().starts_with(&needle)
                    || task.title.to_lowercase().contains(&needle)
            })
            .collect::<Vec<_>>();

        match matches.as_slice() {
            [task] => Ok(Ok(task.clone())),
            [] => Ok(Err(format!("No task matched '{selector}'."))),
            _ => Ok(Err(format!(
                "Found {} tasks matching '{selector}'. Please use a more specific title or id.",
                matches.len()
            ))),
        }
    }

    async fn find_memory(&self, selector: &str) -> anyhow::Result<Result<Memory, String>> {
        let needle = selector.trim().to_lowercase();
        if needle.is_empty() {
            return Ok(Err(
                "Please tell me which memory to review by content fragment or memory id."
                    .to_owned(),
            ));
        }

        let matches = self
            .db
            .list_memories()
            .await?
            .into_iter()
            .filter(|memory| {
                memory.id.to_string().starts_with(&needle)
                    || memory.content.to_lowercase().contains(&needle)
                    || memory.scope.to_lowercase().contains(&needle)
            })
            .collect::<Vec<_>>();

        match matches.as_slice() {
            [memory] => Ok(Ok(memory.clone())),
            [] => Ok(Err(format!("No memory matched '{selector}'."))),
            _ => Ok(Err(format!(
                "Found {} memories matching '{selector}'. Please use a more specific content fragment or id.",
                matches.len()
            ))),
        }
    }

    async fn find_skill(&self, selector: &str) -> anyhow::Result<Result<Skill, String>> {
        let needle = selector.trim().to_lowercase();
        if needle.is_empty() {
            return Ok(Err(
                "Please tell me which skill to manage by name or skill id.".to_owned(),
            ));
        }

        let matches = self
            .db
            .list_skills()
            .await?
            .into_iter()
            .filter(|skill| {
                skill.id.to_string().starts_with(&needle)
                    || skill.name.to_lowercase().contains(&needle)
            })
            .collect::<Vec<_>>();

        match matches.as_slice() {
            [skill] => Ok(Ok(skill.clone())),
            [] => Ok(Err(format!("No skill matched '{selector}'."))),
            _ => Ok(Err(format!(
                "Found {} skills matching '{selector}'. Please use a more specific name or id.",
                matches.len()
            ))),
        }
    }
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct TaskPoolSummary {
    pub total: usize,
    pub draft: usize,
    pub queued: usize,
    pub running: usize,
    pub waiting_for_user: usize,
    pub waiting_for_schedule: usize,
    pub completed: usize,
    pub failed: usize,
    pub cancelled: usize,
    pub paused: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MainAgentMessageInput {
    pub content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MainAgentMessageResponse {
    pub conversation_id: ConversationId,
    pub user_message: ConversationMessage,
    pub assistant_message: ConversationMessage,
    pub changed_tasks: Vec<Task>,
    pub scheduler_tick_requested: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MainAgentAdviceContext {
    pub user_message: String,
    pub deterministic_reply: String,
    pub scheduler_tick_requested: bool,
    pub changed_tasks: Vec<Task>,
    pub task_pool_summary: TaskPoolSummary,
    pub recent_messages: Vec<ConversationMessage>,
}

#[async_trait]
pub trait MainAgentAdvisor: Send + Sync {
    async fn advise(&self, context: MainAgentAdviceContext) -> anyhow::Result<String>;
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MainAgentPlanContext {
    pub user_message: String,
    pub task_pool_summary: TaskPoolSummary,
    pub task_snapshot: Vec<Task>,
    pub recent_messages: Vec<ConversationMessage>,
}

#[derive(Debug, Clone)]
struct ResourceLockConflict {
    resource_key: String,
    running_task: Task,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case", tag = "type")]
pub enum MainAgentPlan {
    CreateTask {
        title: String,
        description: String,
        task_type: TaskType,
        priority: i64,
        interval_seconds: Option<i64>,
        requested_skills: Vec<String>,
    },
    SplitTasks {
        titles: Vec<String>,
    },
    PauseTask {
        selector: String,
    },
    ResumeTask {
        selector: String,
    },
    CancelTask {
        selector: String,
    },
    DeleteTask {
        selector: String,
    },
    CompleteTask {
        selector: String,
        summary: String,
    },
    FailTask {
        selector: String,
        error: String,
    },
    RetryTask {
        selector: String,
        reason: String,
    },
    RunTaskNow {
        selector: String,
    },
    RunNextTask,
    UpdateTaskDetails {
        selector: String,
        title: Option<String>,
        description: Option<String>,
    },
    ReprioritizeTask {
        selector: String,
        priority: i64,
    },
    ReorderTask {
        selector: String,
        queue_position: i64,
    },
    ConvertTaskType {
        selector: String,
        task_type: TaskType,
        interval_seconds: Option<i64>,
    },
    UpdateTaskSchedule {
        selector: String,
        interval_seconds: i64,
    },
    AddTaskDependency {
        selector: String,
        depends_on_selector: String,
    },
    RemoveTaskDependency {
        selector: String,
        depends_on_selector: String,
    },
    AddTaskNote {
        selector: String,
        content: String,
    },
    ListTaskNotes {
        selector: String,
    },
    AddRequestedSkills {
        selector: String,
        skill_names: Vec<String>,
    },
    RemoveRequestedSkills {
        selector: String,
        skill_names: Vec<String>,
    },
    CreateSkillDefinition {
        input: CreateSkill,
    },
    UpdateSkillDefinition {
        selector: String,
        input: UpdateSkill,
    },
    DeleteSkillDefinition {
        selector: String,
    },
    AddResourceLock {
        selector: String,
        resource_key: String,
    },
    RemoveResourceLock {
        selector: String,
        resource_key: String,
    },
    RequestClarification {
        selector: String,
        question: String,
    },
    ReplyToTask {
        selector: String,
        content: String,
    },
    ListGlobalActions,
    ListWaitingForUserTasks,
    ListWaitingForScheduleTasks,
    ListTasksByStatus {
        status: TaskStatus,
    },
    ExplainTaskPool,
    RecommendNextAction,
    ExplainTask {
        selector: String,
    },
    ListTaskConstraints {
        selector: String,
    },
    ListTaskMemories {
        selector: String,
    },
    ListTaskArtifacts {
        selector: String,
    },
    ListTaskHistory {
        selector: String,
    },
    ShowTaskLatestResult {
        selector: String,
    },
    ListTaskFollowUps {
        selector: String,
    },
    ListTaskConversation {
        selector: String,
    },
    InspectWorkspace,
    InspectWorkspaceFile {
        path: String,
    },
    InspectWorkspaceDirectory {
        path: String,
    },
    CreateMemory {
        scope: String,
        content: String,
    },
    ApproveMemory {
        selector: String,
    },
    RejectMemory {
        selector: String,
    },
    BulkReviewMemories {
        status: MemoryStatus,
    },
    BulkReviewTaskMemories {
        selector: String,
        status: MemoryStatus,
    },
    UpdateMemory {
        selector: String,
        input: UpdateMemory,
    },
    DeleteMemory {
        selector: String,
    },
    ListMemories {
        filter: MemoryListFilter,
    },
    ShowSchedulerState,
    ListSkillDefinitions,
    ListTasks,
    Summarize,
    RunSchedulerTick,
}

impl MainAgentPlan {
    fn into_intent(self) -> MainAgentIntent {
        match self {
            Self::CreateTask {
                title,
                description,
                task_type,
                priority,
                interval_seconds,
                requested_skills,
            } => MainAgentIntent::CreateTask {
                title,
                description,
                task_type,
                priority,
                interval_seconds,
                requested_skills,
            },
            Self::SplitTasks { titles } => MainAgentIntent::SplitTasks { titles },
            Self::PauseTask { selector } => MainAgentIntent::PauseTask { selector },
            Self::ResumeTask { selector } => MainAgentIntent::ResumeTask { selector },
            Self::CancelTask { selector } => MainAgentIntent::CancelTask { selector },
            Self::DeleteTask { selector } => MainAgentIntent::DeleteTask { selector },
            Self::CompleteTask { selector, summary } => {
                MainAgentIntent::CompleteTask { selector, summary }
            }
            Self::FailTask { selector, error } => MainAgentIntent::FailTask { selector, error },
            Self::RetryTask { selector, reason } => MainAgentIntent::RetryTask { selector, reason },
            Self::RunTaskNow { selector } => MainAgentIntent::RunTaskNow { selector },
            Self::RunNextTask => MainAgentIntent::RunNextTask,
            Self::UpdateTaskDetails {
                selector,
                title,
                description,
            } => MainAgentIntent::UpdateTaskDetails {
                selector,
                title,
                description,
            },
            Self::ReprioritizeTask { selector, priority } => {
                MainAgentIntent::ReprioritizeTask { selector, priority }
            }
            Self::ReorderTask {
                selector,
                queue_position,
            } => MainAgentIntent::ReorderTask {
                selector,
                queue_position,
            },
            Self::ConvertTaskType {
                selector,
                task_type,
                interval_seconds,
            } => MainAgentIntent::ConvertTaskType {
                selector,
                task_type,
                interval_seconds,
            },
            Self::UpdateTaskSchedule {
                selector,
                interval_seconds,
            } => MainAgentIntent::UpdateTaskSchedule {
                selector,
                interval_seconds,
            },
            Self::AddTaskDependency {
                selector,
                depends_on_selector,
            } => MainAgentIntent::AddTaskDependency {
                selector,
                depends_on_selector,
            },
            Self::RemoveTaskDependency {
                selector,
                depends_on_selector,
            } => MainAgentIntent::RemoveTaskDependency {
                selector,
                depends_on_selector,
            },
            Self::AddTaskNote { selector, content } => {
                MainAgentIntent::AddTaskNote { selector, content }
            }
            Self::ListTaskNotes { selector } => MainAgentIntent::ListTaskNotes { selector },
            Self::AddRequestedSkills {
                selector,
                skill_names,
            } => MainAgentIntent::AddRequestedSkills {
                selector,
                skill_names,
            },
            Self::RemoveRequestedSkills {
                selector,
                skill_names,
            } => MainAgentIntent::RemoveRequestedSkills {
                selector,
                skill_names,
            },
            Self::CreateSkillDefinition { input } => {
                MainAgentIntent::CreateSkillDefinition { input }
            }
            Self::UpdateSkillDefinition { selector, input } => {
                MainAgentIntent::UpdateSkillDefinition { selector, input }
            }
            Self::DeleteSkillDefinition { selector } => {
                MainAgentIntent::DeleteSkillDefinition { selector }
            }
            Self::AddResourceLock {
                selector,
                resource_key,
            } => MainAgentIntent::AddResourceLock {
                selector,
                resource_key,
            },
            Self::RemoveResourceLock {
                selector,
                resource_key,
            } => MainAgentIntent::RemoveResourceLock {
                selector,
                resource_key,
            },
            Self::RequestClarification { selector, question } => {
                MainAgentIntent::RequestClarification { selector, question }
            }
            Self::ReplyToTask { selector, content } => {
                MainAgentIntent::ReplyToTask { selector, content }
            }
            Self::ListGlobalActions => MainAgentIntent::ListGlobalActions,
            Self::ListWaitingForUserTasks => MainAgentIntent::ListWaitingForUserTasks,
            Self::ListWaitingForScheduleTasks => MainAgentIntent::ListWaitingForScheduleTasks,
            Self::ListTasksByStatus { status } => MainAgentIntent::ListTasksByStatus { status },
            Self::ExplainTaskPool => MainAgentIntent::ExplainTaskPool,
            Self::RecommendNextAction => MainAgentIntent::RecommendNextAction,
            Self::ExplainTask { selector } => MainAgentIntent::ExplainTask { selector },
            Self::ListTaskConstraints { selector } => {
                MainAgentIntent::ListTaskConstraints { selector }
            }
            Self::ListTaskMemories { selector } => MainAgentIntent::ListTaskMemories { selector },
            Self::ListTaskArtifacts { selector } => MainAgentIntent::ListTaskArtifacts { selector },
            Self::ListTaskHistory { selector } => MainAgentIntent::ListTaskHistory { selector },
            Self::ShowTaskLatestResult { selector } => {
                MainAgentIntent::ShowTaskLatestResult { selector }
            }
            Self::ListTaskFollowUps { selector } => MainAgentIntent::ListTaskFollowUps { selector },
            Self::ListTaskConversation { selector } => {
                MainAgentIntent::ListTaskConversation { selector }
            }
            Self::InspectWorkspace => MainAgentIntent::InspectWorkspace,
            Self::InspectWorkspaceFile { path } => MainAgentIntent::InspectWorkspaceFile { path },
            Self::InspectWorkspaceDirectory { path } => {
                MainAgentIntent::InspectWorkspaceDirectory { path }
            }
            Self::CreateMemory { scope, content } => {
                MainAgentIntent::CreateMemory { scope, content }
            }
            Self::ApproveMemory { selector } => MainAgentIntent::ApproveMemory { selector },
            Self::RejectMemory { selector } => MainAgentIntent::RejectMemory { selector },
            Self::BulkReviewMemories { status } => MainAgentIntent::BulkReviewMemories { status },
            Self::BulkReviewTaskMemories { selector, status } => {
                MainAgentIntent::BulkReviewTaskMemories { selector, status }
            }
            Self::UpdateMemory { selector, input } => {
                MainAgentIntent::UpdateMemory { selector, input }
            }
            Self::DeleteMemory { selector } => MainAgentIntent::DeleteMemory { selector },
            Self::ListMemories { filter } => MainAgentIntent::ListMemories { filter },
            Self::ShowSchedulerState => MainAgentIntent::ShowSchedulerState,
            Self::ListSkillDefinitions => MainAgentIntent::ListSkillDefinitions,
            Self::ListTasks => MainAgentIntent::ListTasks,
            Self::Summarize => MainAgentIntent::Summarize,
            Self::RunSchedulerTick => MainAgentIntent::RunSchedulerTick,
        }
    }
}

#[async_trait]
pub trait MainAgentPlanner: Send + Sync {
    async fn plan(&self, context: MainAgentPlanContext) -> anyhow::Result<Option<MainAgentPlan>>;
}

#[derive(Debug, Clone)]
pub struct MainAgentLlmConfig {
    pub api_key: String,
    pub model: String,
    pub timeout_ms: u64,
}

impl MainAgentLlmConfig {
    pub fn deepseek(api_key: String, model: String) -> Self {
        Self {
            api_key,
            model,
            timeout_ms: 30_000,
        }
    }
}

#[derive(Debug, Clone)]
pub struct OhMyHarnessMainAgentAdvisor {
    config: MainAgentLlmConfig,
}

impl OhMyHarnessMainAgentAdvisor {
    pub fn new(config: MainAgentLlmConfig) -> Self {
        Self { config }
    }
}

#[async_trait]
impl MainAgentAdvisor for OhMyHarnessMainAgentAdvisor {
    async fn advise(&self, context: MainAgentAdviceContext) -> anyhow::Result<String> {
        dispatch_main_agent_advisor(self.config.clone(), main_agent_advisor_prompt(&context)).await
    }
}

#[async_trait]
impl MainAgentPlanner for OhMyHarnessMainAgentAdvisor {
    async fn plan(&self, context: MainAgentPlanContext) -> anyhow::Result<Option<MainAgentPlan>> {
        dispatch_main_agent_planner(self.config.clone(), main_agent_planner_prompt(&context)).await
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryListFilter {
    Pending,
    Approved,
    Rejected,
    All,
}

impl MemoryListFilter {
    fn matches(self, status: MemoryStatus) -> bool {
        match self {
            Self::Pending => status == MemoryStatus::Pending,
            Self::Approved => status == MemoryStatus::Approved,
            Self::Rejected => status == MemoryStatus::Rejected,
            Self::All => true,
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Pending => "pending",
            Self::Approved => "approved",
            Self::Rejected => "rejected",
            Self::All => "all",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
enum MainAgentIntent {
    SplitTasks {
        titles: Vec<String>,
    },
    CreateTask {
        title: String,
        description: String,
        task_type: TaskType,
        priority: i64,
        interval_seconds: Option<i64>,
        requested_skills: Vec<String>,
    },
    PauseTask {
        selector: String,
    },
    ResumeTask {
        selector: String,
    },
    CancelTask {
        selector: String,
    },
    DeleteTask {
        selector: String,
    },
    CompleteTask {
        selector: String,
        summary: String,
    },
    FailTask {
        selector: String,
        error: String,
    },
    RetryTask {
        selector: String,
        reason: String,
    },
    RunTaskNow {
        selector: String,
    },
    RunNextTask,
    UpdateTaskDetails {
        selector: String,
        title: Option<String>,
        description: Option<String>,
    },
    ReprioritizeTask {
        selector: String,
        priority: i64,
    },
    ReorderTask {
        selector: String,
        queue_position: i64,
    },
    ConvertTaskType {
        selector: String,
        task_type: TaskType,
        interval_seconds: Option<i64>,
    },
    UpdateTaskSchedule {
        selector: String,
        interval_seconds: i64,
    },
    AddTaskDependency {
        selector: String,
        depends_on_selector: String,
    },
    RemoveTaskDependency {
        selector: String,
        depends_on_selector: String,
    },
    AddTaskNote {
        selector: String,
        content: String,
    },
    ListTaskNotes {
        selector: String,
    },
    AddRequestedSkills {
        selector: String,
        skill_names: Vec<String>,
    },
    RemoveRequestedSkills {
        selector: String,
        skill_names: Vec<String>,
    },
    AddResourceLock {
        selector: String,
        resource_key: String,
    },
    RemoveResourceLock {
        selector: String,
        resource_key: String,
    },
    RequestClarification {
        selector: String,
        question: String,
    },
    ReplyToTask {
        selector: String,
        content: String,
    },
    ListTasks,
    ListTasksByStatus {
        status: TaskStatus,
    },
    ListWaitingForUserTasks,
    ListWaitingForScheduleTasks,
    ListGlobalActions,
    ExplainTaskPool,
    RecommendNextAction,
    ExplainTask {
        selector: String,
    },
    ListTaskConstraints {
        selector: String,
    },
    ListTaskMemories {
        selector: String,
    },
    ListTaskArtifacts {
        selector: String,
    },
    ListTaskHistory {
        selector: String,
    },
    ShowTaskLatestResult {
        selector: String,
    },
    ListTaskFollowUps {
        selector: String,
    },
    ListTaskConversation {
        selector: String,
    },
    InspectWorkspace,
    InspectWorkspaceFile {
        path: String,
    },
    InspectWorkspaceDirectory {
        path: String,
    },
    CreateMemory {
        scope: String,
        content: String,
    },
    ApproveMemory {
        selector: String,
    },
    RejectMemory {
        selector: String,
    },
    BulkReviewMemories {
        status: MemoryStatus,
    },
    BulkReviewTaskMemories {
        selector: String,
        status: MemoryStatus,
    },
    UpdateMemory {
        selector: String,
        input: UpdateMemory,
    },
    DeleteMemory {
        selector: String,
    },
    ListMemories {
        filter: MemoryListFilter,
    },
    ShowSchedulerState,
    CreateSkillDefinition {
        input: CreateSkill,
    },
    ListSkillDefinitions,
    UpdateSkillDefinition {
        selector: String,
        input: UpdateSkill,
    },
    DeleteSkillDefinition {
        selector: String,
    },
    RunSchedulerTick,
    Summarize,
    Help,
}

fn parse_intent(content: &str) -> MainAgentIntent {
    let trimmed = content.trim();
    let normalized = trimmed.to_lowercase();

    if is_run_next_task_request(&normalized) {
        return MainAgentIntent::RunNextTask;
    }

    if is_scheduler_state_request(&normalized) {
        return MainAgentIntent::ShowSchedulerState;
    }

    if is_next_action_recommendation_request(&normalized) {
        return MainAgentIntent::RecommendNextAction;
    }

    if let Some(intent) = parse_explain_intent(trimmed, &normalized) {
        return intent;
    }

    if let Some(intent) = parse_split_intent(trimmed, &normalized) {
        return intent;
    }

    if let Some(path) = extract_workspace_file_inspection_path(trimmed, &normalized) {
        return MainAgentIntent::InspectWorkspaceFile { path };
    }

    if let Some(path) = extract_workspace_directory_inspection_path(trimmed, &normalized) {
        return MainAgentIntent::InspectWorkspaceDirectory { path };
    }

    if is_workspace_inspection_request(&normalized) {
        return MainAgentIntent::InspectWorkspace;
    }

    if let Some(intent) = parse_run_task_now_intent(trimmed, &normalized) {
        return intent;
    }

    if is_scheduler_scan_request(&normalized) {
        return MainAgentIntent::RunSchedulerTick;
    }

    if is_waiting_for_user_list_request(&normalized) {
        return MainAgentIntent::ListWaitingForUserTasks;
    }

    if is_waiting_for_schedule_list_request(&normalized) {
        return MainAgentIntent::ListWaitingForScheduleTasks;
    }

    if let Some(intent) = parse_task_status_list_intent(&normalized) {
        return intent;
    }

    if let Some(intent) = parse_task_retry_intent(trimmed, &normalized) {
        return intent;
    }

    if let Some(intent) = parse_task_finish_intent(trimmed, &normalized) {
        return intent;
    }

    if let Some(intent) = parse_task_detail_update_intent(trimmed, &normalized) {
        return intent;
    }

    if let Some(intent) = parse_task_schedule_update_intent(trimmed, &normalized) {
        return intent;
    }

    if let Some(intent) = parse_skill_definition_intent(trimmed, &normalized) {
        return intent;
    }

    if let Some(selector) = extract_task_result_selector(trimmed, &normalized) {
        return MainAgentIntent::ShowTaskLatestResult { selector };
    }

    if contains_any(
        &normalized,
        &[
            ZH_SUMMARY,
            ZH_OVERVIEW,
            ZH_TASK_POOL,
            "summary",
            "summarize",
        ],
    ) {
        return MainAgentIntent::Summarize;
    }

    if let Some(intent) = parse_task_memory_review_intent(trimmed, &normalized) {
        return intent;
    }

    if let Some(selector) = extract_task_memories_selector(trimmed, &normalized) {
        return MainAgentIntent::ListTaskMemories { selector };
    }

    if let Some(intent) = parse_create_memory_intent(trimmed, &normalized) {
        return intent;
    }

    if let Some(intent) = parse_memory_review_intent(trimmed, &normalized) {
        return intent;
    }

    if is_list_tasks_request(&normalized) {
        return MainAgentIntent::ListTasks;
    }

    if is_global_action_list_request(&normalized) {
        return MainAgentIntent::ListGlobalActions;
    }

    if let Some(selector) = extract_task_artifacts_selector(trimmed, &normalized) {
        return MainAgentIntent::ListTaskArtifacts { selector };
    }

    if let Some(selector) = extract_task_history_selector(trimmed, &normalized) {
        return MainAgentIntent::ListTaskHistory { selector };
    }

    if let Some(selector) = extract_task_follow_up_selector(trimmed, &normalized) {
        return MainAgentIntent::ListTaskFollowUps { selector };
    }

    if let Some(selector) = extract_task_notes_selector(trimmed, &normalized) {
        return MainAgentIntent::ListTaskNotes { selector };
    }

    if let Some(selector) = extract_task_constraints_selector(trimmed, &normalized) {
        return MainAgentIntent::ListTaskConstraints { selector };
    }

    if let Some(selector) = extract_task_conversation_selector(trimmed, &normalized) {
        return MainAgentIntent::ListTaskConversation { selector };
    }

    if let Some(intent) = parse_dependency_intent(trimmed, &normalized) {
        return intent;
    }

    if let Some(intent) = parse_note_intent(trimmed, &normalized) {
        return intent;
    }

    if let Some(intent) = parse_requested_skill_intent(trimmed, &normalized) {
        return intent;
    }

    if let Some(intent) = parse_resource_lock_intent(trimmed, &normalized) {
        return intent;
    }

    if let Some(intent) = parse_reply_to_task_intent(trimmed, &normalized) {
        return intent;
    }

    if let Some(intent) = parse_clarification_intent(trimmed, &normalized) {
        return intent;
    }

    if let Some(priority) = extract_priority(&normalized) {
        if !is_create_request(&normalized)
            && contains_any(
                &normalized,
                &[
                    "reprioritize",
                    "set priority",
                    "change priority",
                    ZH_PRIORITY,
                ],
            )
        {
            return MainAgentIntent::ReprioritizeTask {
                selector: extract_task_selector(
                    trimmed,
                    &["priority", "to", ZH_PRIORITY, ZH_AS, ZH_TO],
                ),
                priority,
            };
        }
    }

    if let Some(queue_position) = extract_queue_position(&normalized) {
        if !is_create_request(&normalized)
            && contains_any(
                &normalized,
                &[
                    "reorder",
                    "queue position",
                    "move task",
                    ZH_QUEUE,
                    ZH_SORT,
                    ZH_MOVE,
                ],
            )
        {
            return MainAgentIntent::ReorderTask {
                selector: extract_task_selector(
                    trimmed,
                    &[
                        "queue",
                        "position",
                        "to",
                        ZH_QUEUE,
                        ZH_SORT,
                        ZH_POSITION,
                        ZH_AS,
                        ZH_TO,
                    ],
                ),
                queue_position,
            };
        }
    }

    if is_convert_request(&normalized) {
        if contains_any(
            &normalized,
            &["recurring", "repeat", ZH_RECURRING, ZH_SCHEDULED],
        ) {
            return MainAgentIntent::ConvertTaskType {
                selector: extract_task_selector(
                    trimmed,
                    &[
                        "to",
                        "recurring",
                        "repeat",
                        "every",
                        "interval",
                        "seconds",
                        "second",
                        ZH_RECURRING,
                        ZH_SCHEDULED,
                        ZH_EVERY,
                    ],
                ),
                task_type: TaskType::Recurring,
                interval_seconds: extract_interval_seconds(&normalized),
            };
        }

        if contains_any(&normalized, &["one-off", "one_off", "one off", ZH_ONE_OFF]) {
            return MainAgentIntent::ConvertTaskType {
                selector: extract_task_selector(
                    trimmed,
                    &["to", "one-off", "one_off", "one off", ZH_ONE_OFF],
                ),
                task_type: TaskType::OneOff,
                interval_seconds: None,
            };
        }
    }

    if contains_any(&normalized, &["pause task", ZH_PAUSE_TASK]) || normalized.starts_with("pause ")
    {
        return MainAgentIntent::PauseTask {
            selector: extract_task_selector(trimmed, &[]),
        };
    }

    if contains_any(
        &normalized,
        &["resume task", "unpause task", ZH_RESUME_TASK],
    ) || normalized.starts_with("resume ")
    {
        return MainAgentIntent::ResumeTask {
            selector: extract_task_selector(trimmed, &[]),
        };
    }

    if contains_any(
        &normalized,
        &[
            "delete task",
            "remove task",
            "delete the task",
            "remove the task",
            ZH_DELETE_TASK,
        ],
    ) || normalized.starts_with("delete ")
        || normalized.starts_with("remove task ")
    {
        return MainAgentIntent::DeleteTask {
            selector: extract_task_selector(trimmed, &[]),
        };
    }

    if contains_any(&normalized, &["cancel task", ZH_CANCEL_TASK])
        || normalized.starts_with("cancel ")
    {
        return MainAgentIntent::CancelTask {
            selector: extract_task_selector(trimmed, &[]),
        };
    }

    if is_create_request(&normalized) {
        let task_type = if contains_any(
            &normalized,
            &[ZH_RECURRING, ZH_SCHEDULED, "recurring", "repeat"],
        ) {
            TaskType::Recurring
        } else {
            TaskType::OneOff
        };
        let priority = extract_priority(&normalized).unwrap_or(0);
        let interval_seconds = (task_type == TaskType::Recurring)
            .then(|| extract_interval_seconds(&normalized))
            .flatten();
        let requested_skills = extract_create_requested_skills(trimmed, &normalized);
        let title = extract_title(trimmed);

        return MainAgentIntent::CreateTask {
            title,
            description: trimmed.to_owned(),
            task_type,
            priority,
            interval_seconds,
            requested_skills,
        };
    }

    MainAgentIntent::Help
}

fn is_explicit_help_request(content: &str) -> bool {
    let normalized = content.trim().to_lowercase();
    matches!(normalized.as_str(), "help" | "?" | "帮助" | "用法")
        || normalized.contains("what can you do")
        || normalized.contains("how can you help")
        || normalized.contains("你能做什么")
}

fn enrich_requested_skills_for_task_text(
    mut requested_skills: Vec<String>,
    task_text: &str,
) -> Vec<String> {
    if is_github_task_text(task_text)
        && !requested_skills
            .iter()
            .any(|skill| skill.eq_ignore_ascii_case("github"))
    {
        requested_skills.push("github".to_owned());
    }

    requested_skills
}

fn infer_resource_locks_for_task_text(task_text: &str) -> Vec<String> {
    if !is_github_task_text(task_text) {
        return Vec::new();
    }

    dedupe_strings(
        extract_github_repository_refs(task_text)
            .into_iter()
            .map(|repository| format!("repo:{repository}"))
            .collect(),
    )
}

fn is_github_task_text(task_text: &str) -> bool {
    let normalized = task_text.to_ascii_lowercase();
    normalized.contains("github")
        || normalized.contains("github.com/")
        || normalized.contains("仓库")
        || normalized.contains("代码仓")
        || !extract_github_repository_refs(task_text).is_empty()
}

fn extract_github_repository_refs(task_text: &str) -> Vec<String> {
    let tokens = task_text
        .split(|ch: char| {
            ch.is_whitespace()
                || matches!(
                    ch,
                    ',' | ';' | '"' | '\'' | '`' | '<' | '>' | '(' | ')' | '[' | ']'
                )
        })
        .filter_map(normalize_github_repository_token)
        .collect::<Vec<_>>();
    dedupe_strings(tokens)
}

fn normalize_github_repository_token(token: &str) -> Option<String> {
    let mut token = token
        .trim()
        .trim_matches(|ch: char| matches!(ch, '.' | ':' | '/' | '#'));
    if token.is_empty() {
        return None;
    }

    if let Some(rest) = token.strip_prefix("https://github.com/") {
        token = rest;
    } else if let Some(rest) = token.strip_prefix("http://github.com/") {
        token = rest;
    } else if let Some(rest) = token.strip_prefix("github.com/") {
        token = rest;
    }

    let mut parts = token.split('/');
    let owner = parts.next()?.trim();
    let repo = parts.next()?.trim().trim_end_matches(".git");
    if owner.is_empty() || repo.is_empty() {
        return None;
    }
    if !is_safe_github_repo_part(owner) || !is_safe_github_repo_part(repo) {
        return None;
    }

    Some(format!("{owner}/{repo}"))
}

fn is_safe_github_repo_part(value: &str) -> bool {
    value
        .chars()
        .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '.' | '_' | '-'))
}

fn dedupe_strings(values: Vec<String>) -> Vec<String> {
    let mut deduped = Vec::new();
    for value in values {
        if !deduped.iter().any(|existing| existing == &value) {
            deduped.push(value);
        }
    }

    deduped
}

const MAIN_AGENT_ADVISOR_SYSTEM_PROMPT: &str = "You are the main agent for Persistent Agent. The deterministic task-state operation has already been performed by product code. Write a concise, helpful user-facing reply in the user's language. Preserve the facts from the deterministic reply. Do not claim that you changed task state unless the deterministic reply says it happened. Mention useful next steps only when they directly follow from the current context.";
const MAIN_AGENT_PLANNER_SYSTEM_PROMPT: &str = "You are the planning layer for the Persistent Agent main agent. Choose exactly one available planning tool when the user's message clearly asks to manage the task pool. Do not write free-form task state changes. If the request is ambiguous or unsupported, answer briefly without calling a tool.";

#[derive(Debug, Default)]
struct MainAgentPlannerState {
    plan: Option<MainAgentPlan>,
}

async fn dispatch_main_agent_advisor(
    config: MainAgentLlmConfig,
    prompt: String,
) -> anyhow::Result<String> {
    let client = Arc::new(deepseek::client(config.api_key)) as Arc<dyn LlmClient>;
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
    opts.max_tokens = 500;
    opts.system_prompt = Some(MAIN_AGENT_ADVISOR_SYSTEM_PROMPT.to_owned());
    opts.tools = Vec::new();

    let harness = AgentHarness::new_in_memory(client, sandbox.env(), opts).await;
    let mut events = harness.subscribe();
    harness.prompt(prompt).await?;
    let wait = harness.wait_for_idle();
    tokio::time::timeout(std::time::Duration::from_millis(config.timeout_ms), wait).await?;

    let mut assistant_text = String::new();
    while let Ok(event) = events.try_recv() {
        if let AgentHarnessEvent::Agent(llm_harness_agent::prelude::AgentEvent::AgentEnd {
            new_messages,
        }) = event.as_ref()
        {
            assistant_text.push_str(&assistant_text_from_messages(new_messages));
        }
    }

    Ok(assistant_text)
}

async fn dispatch_main_agent_planner(
    config: MainAgentLlmConfig,
    prompt: String,
) -> anyhow::Result<Option<MainAgentPlan>> {
    let client = Arc::new(deepseek::client(config.api_key)) as Arc<dyn LlmClient>;
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

    let state = Arc::new(tokio::sync::Mutex::new(MainAgentPlannerState::default()));
    let registry = main_agent_planner_tool_registry(state.clone());
    let mut opts = AgentHarnessOptions::new(config.model);
    opts.max_tokens = 500;
    opts.system_prompt = Some(MAIN_AGENT_PLANNER_SYSTEM_PROMPT.to_owned());
    opts.tools = registry.subset(&[
        "plan_create_task",
        "plan_split_tasks",
        "plan_pause_task",
        "plan_resume_task",
        "plan_cancel_task",
        "plan_delete_task",
        "plan_complete_task",
        "plan_fail_task",
        "plan_retry_task",
        "plan_run_task_now",
        "plan_run_next_task",
        "plan_update_task_details",
        "plan_reprioritize_task",
        "plan_reorder_task",
        "plan_convert_task_type",
        "plan_update_task_schedule",
        "plan_add_task_dependency",
        "plan_remove_task_dependency",
        "plan_add_task_note",
        "plan_list_task_notes",
        "plan_add_requested_skills",
        "plan_remove_requested_skills",
        "plan_create_skill_definition",
        "plan_update_skill_definition",
        "plan_delete_skill_definition",
        "plan_add_resource_lock",
        "plan_remove_resource_lock",
        "plan_request_clarification",
        "plan_reply_to_task",
        "plan_list_main_agent_actions",
        "plan_list_waiting_for_user_tasks",
        "plan_list_waiting_for_schedule_tasks",
        "plan_explain_task_pool",
        "plan_recommend_next_action",
        "plan_explain_task",
        "plan_list_task_constraints",
        "plan_list_task_memories",
        "plan_list_task_artifacts",
        "plan_list_task_history",
        "plan_show_task_latest_result",
        "plan_list_task_follow_ups",
        "plan_list_task_conversation",
        "plan_inspect_workspace",
        "plan_inspect_workspace_file",
        "plan_inspect_workspace_directory",
        "plan_create_memory",
        "plan_approve_memory",
        "plan_reject_memory",
        "plan_approve_all_pending_memories",
        "plan_reject_all_pending_memories",
        "plan_review_task_memories",
        "plan_update_memory",
        "plan_delete_memory",
        "plan_list_memories",
        "plan_show_scheduler_state",
        "plan_list_skill_definitions",
        "plan_list_tasks",
        "plan_list_tasks_by_status",
        "plan_summarize_task_pool",
        "plan_scheduler_scan",
    ]);

    let harness = AgentHarness::new_in_memory(client, sandbox.env(), opts).await;
    harness.prompt(prompt).await?;
    let wait = harness.wait_for_idle();
    tokio::time::timeout(std::time::Duration::from_millis(config.timeout_ms), wait).await?;

    Ok(state.lock().await.plan.clone())
}

fn main_agent_planner_tool_registry(
    state: Arc<tokio::sync::Mutex<MainAgentPlannerState>>,
) -> Arc<InMemoryToolRegistry> {
    let registry = Arc::new(InMemoryToolRegistry::new());
    registry.register(Arc::new(PlanCreateTaskTool::new(state.clone())));
    registry.register(Arc::new(PlanSplitTasksTool::new(state.clone())));
    registry.register(Arc::new(PlanTaskSelectorTool::new(
        state.clone(),
        "plan_pause_task",
        "Plan to pause a task selected by id or title fragment.",
        PlannedTaskSelectorAction::Pause,
    )));
    registry.register(Arc::new(PlanTaskSelectorTool::new(
        state.clone(),
        "plan_resume_task",
        "Plan to resume a paused, blocked, or waiting task selected by id or title fragment.",
        PlannedTaskSelectorAction::Resume,
    )));
    registry.register(Arc::new(PlanTaskSelectorTool::new(
        state.clone(),
        "plan_cancel_task",
        "Plan to cancel a task selected by id or title fragment.",
        PlannedTaskSelectorAction::Cancel,
    )));
    registry.register(Arc::new(PlanTaskSelectorTool::new(
        state.clone(),
        "plan_delete_task",
        "Plan to permanently delete a task selected by id or title fragment.",
        PlannedTaskSelectorAction::Delete,
    )));
    registry.register(Arc::new(PlanTaskFinishTool::new(
        state.clone(),
        "plan_complete_task",
        "Plan to manually mark one task complete with a result summary.",
        PlannedTaskFinishAction::Complete,
    )));
    registry.register(Arc::new(PlanTaskFinishTool::new(
        state.clone(),
        "plan_fail_task",
        "Plan to manually mark one task failed with an error reason.",
        PlannedTaskFinishAction::Fail,
    )));
    registry.register(Arc::new(PlanTaskFinishTool::new(
        state.clone(),
        "plan_retry_task",
        "Plan to requeue a failed task for another attempt.",
        PlannedTaskFinishAction::Retry,
    )));
    registry.register(Arc::new(PlanTaskSelectorTool::new(
        state.clone(),
        "plan_run_task_now",
        "Plan to move one task to the front of runnable work and request a scheduler scan.",
        PlannedTaskSelectorAction::RunNow,
    )));
    registry.register(Arc::new(PlanSimpleIntentTool::new(
        state.clone(),
        "plan_run_next_task",
        "Plan to select the next runnable task and request a scheduler scan.",
        MainAgentPlan::RunNextTask,
    )));
    registry.register(Arc::new(PlanUpdateTaskDetailsTool::new(state.clone())));
    registry.register(Arc::new(PlanReprioritizeTaskTool::new(state.clone())));
    registry.register(Arc::new(PlanReorderTaskTool::new(state.clone())));
    registry.register(Arc::new(PlanConvertTaskTypeTool::new(state.clone())));
    registry.register(Arc::new(PlanUpdateTaskScheduleTool::new(state.clone())));
    registry.register(Arc::new(PlanTaskDependencyTool::new(
        state.clone(),
        "plan_add_task_dependency",
        "Plan to make one task depend on another task.",
        PlannedDependencyAction::Add,
    )));
    registry.register(Arc::new(PlanTaskDependencyTool::new(
        state.clone(),
        "plan_remove_task_dependency",
        "Plan to remove a dependency between two tasks.",
        PlannedDependencyAction::Remove,
    )));
    registry.register(Arc::new(PlanTaskNoteTool::new(state.clone())));
    registry.register(Arc::new(PlanTaskReadTool::new(
        state.clone(),
        "plan_list_task_notes",
        "Plan to list notes attached to one task.",
        PlannedTaskReadAction::ListNotes,
    )));
    registry.register(Arc::new(PlanRequestedSkillsTool::new(
        state.clone(),
        "plan_add_requested_skills",
        "Plan to add requested skills to an existing task.",
        PlannedRequestedSkillsAction::Add,
    )));
    registry.register(Arc::new(PlanRequestedSkillsTool::new(
        state.clone(),
        "plan_remove_requested_skills",
        "Plan to remove requested skills from an existing task.",
        PlannedRequestedSkillsAction::Remove,
    )));
    registry.register(Arc::new(PlanCreateSkillDefinitionTool::new(state.clone())));
    registry.register(Arc::new(PlanUpdateSkillDefinitionTool::new(state.clone())));
    registry.register(Arc::new(PlanDeleteSkillDefinitionTool::new(state.clone())));
    registry.register(Arc::new(PlanResourceLockTool::new(
        state.clone(),
        "plan_add_resource_lock",
        "Plan to add a resource lock to a task.",
        PlannedResourceLockAction::Add,
    )));
    registry.register(Arc::new(PlanResourceLockTool::new(
        state.clone(),
        "plan_remove_resource_lock",
        "Plan to remove a resource lock from a task.",
        PlannedResourceLockAction::Remove,
    )));
    registry.register(Arc::new(PlanRequestClarificationTool::new(state.clone())));
    registry.register(Arc::new(PlanReplyToTaskTool::new(state.clone())));
    registry.register(Arc::new(PlanSimpleIntentTool::new(
        state.clone(),
        "plan_list_main_agent_actions",
        "Plan to list recent main-agent audit actions.",
        MainAgentPlan::ListGlobalActions,
    )));
    registry.register(Arc::new(PlanSimpleIntentTool::new(
        state.clone(),
        "plan_list_waiting_for_user_tasks",
        "Plan to list tasks that are waiting for user input.",
        MainAgentPlan::ListWaitingForUserTasks,
    )));
    registry.register(Arc::new(PlanSimpleIntentTool::new(
        state.clone(),
        "plan_list_waiting_for_schedule_tasks",
        "Plan to list tasks that are waiting for their next schedule.",
        MainAgentPlan::ListWaitingForScheduleTasks,
    )));
    registry.register(Arc::new(PlanSimpleIntentTool::new(
        state.clone(),
        "plan_explain_task_pool",
        "Plan to explain why the task pool is in its current state.",
        MainAgentPlan::ExplainTaskPool,
    )));
    registry.register(Arc::new(PlanSimpleIntentTool::new(
        state.clone(),
        "plan_recommend_next_action",
        "Plan to recommend the next operator action for the task pool.",
        MainAgentPlan::RecommendNextAction,
    )));
    registry.register(Arc::new(PlanTaskReadTool::new(
        state.clone(),
        "plan_explain_task",
        "Plan to explain one task's current state.",
        PlannedTaskReadAction::Explain,
    )));
    registry.register(Arc::new(PlanTaskReadTool::new(
        state.clone(),
        "plan_list_task_constraints",
        "Plan to list one task's dependencies and resource locks.",
        PlannedTaskReadAction::ListConstraints,
    )));
    registry.register(Arc::new(PlanTaskReadTool::new(
        state.clone(),
        "plan_list_task_memories",
        "Plan to list memory candidates proposed by one task.",
        PlannedTaskReadAction::ListMemories,
    )));
    registry.register(Arc::new(PlanTaskReadTool::new(
        state.clone(),
        "plan_list_task_artifacts",
        "Plan to list artifacts reported for one task.",
        PlannedTaskReadAction::ListArtifacts,
    )));
    registry.register(Arc::new(PlanTaskReadTool::new(
        state.clone(),
        "plan_list_task_history",
        "Plan to list attempts, worker events, and audit actions for one task.",
        PlannedTaskReadAction::ListHistory,
    )));
    registry.register(Arc::new(PlanTaskReadTool::new(
        state.clone(),
        "plan_show_task_latest_result",
        "Plan to show one task's latest result summary.",
        PlannedTaskReadAction::ShowLatestResult,
    )));
    registry.register(Arc::new(PlanTaskReadTool::new(
        state.clone(),
        "plan_list_task_follow_ups",
        "Plan to list follow-up tasks created from one task.",
        PlannedTaskReadAction::ListFollowUps,
    )));
    registry.register(Arc::new(PlanTaskReadTool::new(
        state.clone(),
        "plan_list_task_conversation",
        "Plan to list recent conversation messages for one task.",
        PlannedTaskReadAction::ListConversation,
    )));
    registry.register(Arc::new(PlanSimpleIntentTool::new(
        state.clone(),
        "plan_inspect_workspace",
        "Plan to inspect the current workspace status.",
        MainAgentPlan::InspectWorkspace,
    )));
    registry.register(Arc::new(PlanInspectWorkspaceFileTool::new(state.clone())));
    registry.register(Arc::new(PlanInspectWorkspaceDirectoryTool::new(
        state.clone(),
    )));
    registry.register(Arc::new(PlanCreateMemoryTool::new(state.clone())));
    registry.register(Arc::new(PlanMemoryReviewTool::new(
        state.clone(),
        "plan_approve_memory",
        "Plan to approve a memory candidate selected by id, scope, or content fragment.",
        PlannedMemoryReviewAction::Approve,
    )));
    registry.register(Arc::new(PlanMemoryReviewTool::new(
        state.clone(),
        "plan_reject_memory",
        "Plan to reject a memory candidate selected by id, scope, or content fragment.",
        PlannedMemoryReviewAction::Reject,
    )));
    registry.register(Arc::new(PlanSimpleIntentTool::new(
        state.clone(),
        "plan_approve_all_pending_memories",
        "Plan to approve all pending memory candidates.",
        MainAgentPlan::BulkReviewMemories {
            status: MemoryStatus::Approved,
        },
    )));
    registry.register(Arc::new(PlanSimpleIntentTool::new(
        state.clone(),
        "plan_reject_all_pending_memories",
        "Plan to reject all pending memory candidates.",
        MainAgentPlan::BulkReviewMemories {
            status: MemoryStatus::Rejected,
        },
    )));
    registry.register(Arc::new(PlanTaskMemoryReviewTool::new(state.clone())));
    registry.register(Arc::new(PlanUpdateMemoryTool::new(state.clone())));
    registry.register(Arc::new(PlanMemoryReviewTool::new(
        state.clone(),
        "plan_delete_memory",
        "Plan to delete a memory selected by id, scope, or content fragment.",
        PlannedMemoryReviewAction::Delete,
    )));
    registry.register(Arc::new(PlanListMemoriesTool::new(state.clone())));
    registry.register(Arc::new(PlanSimpleIntentTool::new(
        state.clone(),
        "plan_show_scheduler_state",
        "Plan to show current scheduler execution state without running a scan.",
        MainAgentPlan::ShowSchedulerState,
    )));
    registry.register(Arc::new(PlanSimpleIntentTool::new(
        state.clone(),
        "plan_list_skill_definitions",
        "Plan to list existing skill definitions.",
        MainAgentPlan::ListSkillDefinitions,
    )));
    registry.register(Arc::new(PlanSimpleIntentTool::new(
        state.clone(),
        "plan_list_tasks",
        "Plan to list the current task pool.",
        MainAgentPlan::ListTasks,
    )));
    registry.register(Arc::new(PlanListTasksByStatusTool::new(state.clone())));
    registry.register(Arc::new(PlanSimpleIntentTool::new(
        state.clone(),
        "plan_summarize_task_pool",
        "Plan to summarize the current task pool.",
        MainAgentPlan::Summarize,
    )));
    registry.register(Arc::new(PlanSimpleIntentTool::new(
        state,
        "plan_scheduler_scan",
        "Plan to request one scheduler scan of the task pool.",
        MainAgentPlan::RunSchedulerTick,
    )));
    registry
}

struct PlanCreateTaskTool {
    state: Arc<tokio::sync::Mutex<MainAgentPlannerState>>,
    schema: serde_json::Value,
}

impl PlanCreateTaskTool {
    fn new(state: Arc<tokio::sync::Mutex<MainAgentPlannerState>>) -> Self {
        Self {
            state,
            schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "title": { "type": "string" },
                    "description": { "type": "string" },
                    "task_type": { "type": "string", "enum": ["one_off", "recurring"] },
                    "priority": { "type": "integer" },
                    "interval_seconds": { "type": "integer", "minimum": 1 },
                    "requested_skills": { "type": "array", "items": { "type": "string" } }
                },
                "required": ["title"]
            }),
        }
    }
}

impl Tool for PlanCreateTaskTool {
    fn name(&self) -> &str {
        "plan_create_task"
    }

    fn description(&self) -> &str {
        "Plan one task creation from a natural-language user request."
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
            let title = planner_required_string(&args, "title")?;
            let description = args
                .get("description")
                .and_then(|value| value.as_str())
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(ToOwned::to_owned)
                .unwrap_or_else(|| title.clone());
            let task_type = planner_task_type(&args)?;
            let interval_seconds = (task_type == TaskType::Recurring)
                .then(|| {
                    args.get("interval_seconds")
                        .and_then(|value| value.as_i64())
                })
                .flatten();
            let requested_skills = planner_string_array(&args, "requested_skills");
            let priority = args
                .get("priority")
                .and_then(|value| value.as_i64())
                .unwrap_or(0);
            self.state.lock().await.plan = Some(MainAgentPlan::CreateTask {
                title,
                description,
                task_type,
                priority,
                interval_seconds,
                requested_skills,
            });
            Ok(planner_tool_result("planned task creation"))
        })
    }
}

struct PlanSplitTasksTool {
    state: Arc<tokio::sync::Mutex<MainAgentPlannerState>>,
    schema: serde_json::Value,
}

impl PlanSplitTasksTool {
    fn new(state: Arc<tokio::sync::Mutex<MainAgentPlannerState>>) -> Self {
        Self {
            state,
            schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "titles": {
                        "type": "array",
                        "items": { "type": "string" },
                        "minItems": 2
                    }
                },
                "required": ["titles"]
            }),
        }
    }
}

impl Tool for PlanSplitTasksTool {
    fn name(&self) -> &str {
        "plan_split_tasks"
    }

    fn description(&self) -> &str {
        "Plan multiple one-off tasks when the user asks to break a goal into steps."
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
            let titles = planner_string_array(&args, "titles");
            if titles.len() < 2 {
                return Err(ToolError::InvalidArguments(
                    "titles must contain at least two items".to_owned(),
                ));
            }
            self.state.lock().await.plan = Some(MainAgentPlan::SplitTasks { titles });
            Ok(planner_tool_result("planned split tasks"))
        })
    }
}

#[derive(Clone, Copy)]
enum PlannedTaskSelectorAction {
    Pause,
    Resume,
    Cancel,
    Delete,
    RunNow,
}

struct PlanTaskSelectorTool {
    state: Arc<tokio::sync::Mutex<MainAgentPlannerState>>,
    name: &'static str,
    description: &'static str,
    action: PlannedTaskSelectorAction,
    schema: serde_json::Value,
}

impl PlanTaskSelectorTool {
    fn new(
        state: Arc<tokio::sync::Mutex<MainAgentPlannerState>>,
        name: &'static str,
        description: &'static str,
        action: PlannedTaskSelectorAction,
    ) -> Self {
        Self {
            state,
            name,
            description,
            action,
            schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "selector": {
                        "type": "string",
                        "description": "Task id or title fragment identifying the task."
                    }
                },
                "required": ["selector"]
            }),
        }
    }
}

impl Tool for PlanTaskSelectorTool {
    fn name(&self) -> &str {
        self.name
    }

    fn description(&self) -> &str {
        self.description
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
            let selector = planner_required_string(&args, "selector")?;
            let plan = match self.action {
                PlannedTaskSelectorAction::Pause => MainAgentPlan::PauseTask { selector },
                PlannedTaskSelectorAction::Resume => MainAgentPlan::ResumeTask { selector },
                PlannedTaskSelectorAction::Cancel => MainAgentPlan::CancelTask { selector },
                PlannedTaskSelectorAction::Delete => MainAgentPlan::DeleteTask { selector },
                PlannedTaskSelectorAction::RunNow => MainAgentPlan::RunTaskNow { selector },
            };
            self.state.lock().await.plan = Some(plan);
            Ok(planner_tool_result("planned task state change"))
        })
    }
}

#[derive(Clone, Copy)]
enum PlannedTaskFinishAction {
    Complete,
    Fail,
    Retry,
}

struct PlanTaskFinishTool {
    state: Arc<tokio::sync::Mutex<MainAgentPlannerState>>,
    name: &'static str,
    description: &'static str,
    action: PlannedTaskFinishAction,
    schema: serde_json::Value,
}

impl PlanTaskFinishTool {
    fn new(
        state: Arc<tokio::sync::Mutex<MainAgentPlannerState>>,
        name: &'static str,
        description: &'static str,
        action: PlannedTaskFinishAction,
    ) -> Self {
        Self {
            state,
            name,
            description,
            action,
            schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "selector": {
                        "type": "string",
                        "description": "Task id or title fragment identifying the task."
                    },
                    "summary": {
                        "type": "string",
                        "description": "Completion summary or failure reason to store on the task."
                    }
                },
                "required": ["selector", "summary"]
            }),
        }
    }
}

impl Tool for PlanTaskFinishTool {
    fn name(&self) -> &str {
        self.name
    }

    fn description(&self) -> &str {
        self.description
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
            let selector = planner_required_string(&args, "selector")?;
            let summary = planner_required_string(&args, "summary")?;
            let plan = match self.action {
                PlannedTaskFinishAction::Complete => {
                    MainAgentPlan::CompleteTask { selector, summary }
                }
                PlannedTaskFinishAction::Fail => MainAgentPlan::FailTask {
                    selector,
                    error: summary,
                },
                PlannedTaskFinishAction::Retry => MainAgentPlan::RetryTask {
                    selector,
                    reason: summary,
                },
            };
            self.state.lock().await.plan = Some(plan);
            Ok(planner_tool_result("planned task finish state change"))
        })
    }
}

struct PlanUpdateTaskDetailsTool {
    state: Arc<tokio::sync::Mutex<MainAgentPlannerState>>,
    schema: serde_json::Value,
}

impl PlanUpdateTaskDetailsTool {
    fn new(state: Arc<tokio::sync::Mutex<MainAgentPlannerState>>) -> Self {
        Self {
            state,
            schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "selector": {
                        "type": "string",
                        "description": "Task id or title fragment identifying the task."
                    },
                    "title": {
                        "type": "string",
                        "description": "New task title, if the user asked to rename it."
                    },
                    "description": {
                        "type": "string",
                        "description": "New task description, if the user asked to change task details."
                    }
                },
                "required": ["selector"]
            }),
        }
    }
}

impl Tool for PlanUpdateTaskDetailsTool {
    fn name(&self) -> &str {
        "plan_update_task_details"
    }

    fn description(&self) -> &str {
        "Plan to update an existing task's title and/or description."
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
            let selector = planner_required_string(&args, "selector")?;
            let title = args
                .get("title")
                .and_then(|value| value.as_str())
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(ToOwned::to_owned);
            let description = args
                .get("description")
                .and_then(|value| value.as_str())
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(ToOwned::to_owned);
            if title.is_none() && description.is_none() {
                return Err(ToolError::InvalidArguments(
                    "plan_update_task_details requires title or description".to_owned(),
                ));
            }
            self.state.lock().await.plan = Some(MainAgentPlan::UpdateTaskDetails {
                selector,
                title,
                description,
            });
            Ok(planner_tool_result("planned task detail update"))
        })
    }
}

struct PlanReprioritizeTaskTool {
    state: Arc<tokio::sync::Mutex<MainAgentPlannerState>>,
    schema: serde_json::Value,
}

impl PlanReprioritizeTaskTool {
    fn new(state: Arc<tokio::sync::Mutex<MainAgentPlannerState>>) -> Self {
        Self {
            state,
            schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "selector": {
                        "type": "string",
                        "description": "Task id or title fragment identifying the task."
                    },
                    "priority": { "type": "integer" }
                },
                "required": ["selector", "priority"]
            }),
        }
    }
}

impl Tool for PlanReprioritizeTaskTool {
    fn name(&self) -> &str {
        "plan_reprioritize_task"
    }

    fn description(&self) -> &str {
        "Plan to change the priority of a task selected by id or title fragment."
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
            let selector = planner_required_string(&args, "selector")?;
            let priority = args
                .get("priority")
                .and_then(|value| value.as_i64())
                .ok_or_else(|| ToolError::InvalidArguments("priority is required".to_owned()))?;
            self.state.lock().await.plan =
                Some(MainAgentPlan::ReprioritizeTask { selector, priority });
            Ok(planner_tool_result("planned task priority change"))
        })
    }
}

struct PlanReorderTaskTool {
    state: Arc<tokio::sync::Mutex<MainAgentPlannerState>>,
    schema: serde_json::Value,
}

impl PlanReorderTaskTool {
    fn new(state: Arc<tokio::sync::Mutex<MainAgentPlannerState>>) -> Self {
        Self {
            state,
            schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "selector": {
                        "type": "string",
                        "description": "Task id or title fragment identifying the task."
                    },
                    "queue_position": { "type": "integer", "minimum": 0 }
                },
                "required": ["selector", "queue_position"]
            }),
        }
    }
}

impl Tool for PlanReorderTaskTool {
    fn name(&self) -> &str {
        "plan_reorder_task"
    }

    fn description(&self) -> &str {
        "Plan to move a task to a specific queue position."
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
            let selector = planner_required_string(&args, "selector")?;
            let queue_position = args
                .get("queue_position")
                .and_then(|value| value.as_i64())
                .ok_or_else(|| {
                    ToolError::InvalidArguments("queue_position is required".to_owned())
                })?;
            self.state.lock().await.plan = Some(MainAgentPlan::ReorderTask {
                selector,
                queue_position,
            });
            Ok(planner_tool_result("planned task reorder"))
        })
    }
}

#[derive(Clone, Copy)]
enum PlannedDependencyAction {
    Add,
    Remove,
}

struct PlanTaskDependencyTool {
    state: Arc<tokio::sync::Mutex<MainAgentPlannerState>>,
    name: &'static str,
    description: &'static str,
    action: PlannedDependencyAction,
    schema: serde_json::Value,
}

impl PlanTaskDependencyTool {
    fn new(
        state: Arc<tokio::sync::Mutex<MainAgentPlannerState>>,
        name: &'static str,
        description: &'static str,
        action: PlannedDependencyAction,
    ) -> Self {
        Self {
            state,
            name,
            description,
            action,
            schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "selector": {
                        "type": "string",
                        "description": "Dependent task id or title fragment."
                    },
                    "depends_on_selector": {
                        "type": "string",
                        "description": "Prerequisite task id or title fragment."
                    }
                },
                "required": ["selector", "depends_on_selector"]
            }),
        }
    }
}

impl Tool for PlanTaskDependencyTool {
    fn name(&self) -> &str {
        self.name
    }

    fn description(&self) -> &str {
        self.description
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
            let selector = planner_required_string(&args, "selector")?;
            let depends_on_selector = planner_required_string(&args, "depends_on_selector")?;
            let plan = match self.action {
                PlannedDependencyAction::Add => MainAgentPlan::AddTaskDependency {
                    selector,
                    depends_on_selector,
                },
                PlannedDependencyAction::Remove => MainAgentPlan::RemoveTaskDependency {
                    selector,
                    depends_on_selector,
                },
            };
            self.state.lock().await.plan = Some(plan);
            Ok(planner_tool_result("planned task dependency change"))
        })
    }
}

struct PlanTaskNoteTool {
    state: Arc<tokio::sync::Mutex<MainAgentPlannerState>>,
    schema: serde_json::Value,
}

impl PlanTaskNoteTool {
    fn new(state: Arc<tokio::sync::Mutex<MainAgentPlannerState>>) -> Self {
        Self {
            state,
            schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "selector": {
                        "type": "string",
                        "description": "Task id or title fragment identifying the task."
                    },
                    "content": { "type": "string" }
                },
                "required": ["selector", "content"]
            }),
        }
    }
}

impl Tool for PlanTaskNoteTool {
    fn name(&self) -> &str {
        "plan_add_task_note"
    }

    fn description(&self) -> &str {
        "Plan to add a note to an existing task."
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
            self.state.lock().await.plan = Some(MainAgentPlan::AddTaskNote {
                selector: planner_required_string(&args, "selector")?,
                content: planner_required_string(&args, "content")?,
            });
            Ok(planner_tool_result("planned task note"))
        })
    }
}

#[derive(Clone, Copy)]
enum PlannedRequestedSkillsAction {
    Add,
    Remove,
}

struct PlanRequestedSkillsTool {
    state: Arc<tokio::sync::Mutex<MainAgentPlannerState>>,
    name: &'static str,
    description: &'static str,
    action: PlannedRequestedSkillsAction,
    schema: serde_json::Value,
}

impl PlanRequestedSkillsTool {
    fn new(
        state: Arc<tokio::sync::Mutex<MainAgentPlannerState>>,
        name: &'static str,
        description: &'static str,
        action: PlannedRequestedSkillsAction,
    ) -> Self {
        Self {
            state,
            name,
            description,
            action,
            schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "selector": {
                        "type": "string",
                        "description": "Task id or title fragment identifying the task."
                    },
                    "skill_names": {
                        "type": "array",
                        "items": { "type": "string" },
                        "minItems": 1
                    }
                },
                "required": ["selector", "skill_names"]
            }),
        }
    }
}

impl Tool for PlanRequestedSkillsTool {
    fn name(&self) -> &str {
        self.name
    }

    fn description(&self) -> &str {
        self.description
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
            let selector = planner_required_string(&args, "selector")?;
            let skill_names = planner_string_array(&args, "skill_names");
            if skill_names.is_empty() {
                return Err(ToolError::InvalidArguments(
                    "skill_names must contain at least one item".to_owned(),
                ));
            }
            let plan = match self.action {
                PlannedRequestedSkillsAction::Add => MainAgentPlan::AddRequestedSkills {
                    selector,
                    skill_names,
                },
                PlannedRequestedSkillsAction::Remove => MainAgentPlan::RemoveRequestedSkills {
                    selector,
                    skill_names,
                },
            };
            self.state.lock().await.plan = Some(plan);
            Ok(planner_tool_result("planned requested skill change"))
        })
    }
}

struct PlanCreateSkillDefinitionTool {
    state: Arc<tokio::sync::Mutex<MainAgentPlannerState>>,
    schema: serde_json::Value,
}

impl PlanCreateSkillDefinitionTool {
    fn new(state: Arc<tokio::sync::Mutex<MainAgentPlannerState>>) -> Self {
        Self {
            state,
            schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "name": { "type": "string" },
                    "description": { "type": "string" },
                    "trigger_rules": {
                        "type": "array",
                        "items": { "type": "string" }
                    },
                    "tool_subset": {
                        "type": "array",
                        "items": { "type": "string" }
                    },
                    "resource_path": { "type": "string" }
                },
                "required": ["name"]
            }),
        }
    }
}

impl Tool for PlanCreateSkillDefinitionTool {
    fn name(&self) -> &str {
        "plan_create_skill_definition"
    }

    fn description(&self) -> &str {
        "Plan to create a reusable skill definition with triggers, allowed tool aliases, and optional resource path."
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
            let name = planner_required_string(&args, "name")?;
            let description = args
                .get("description")
                .and_then(|value| value.as_str())
                .map(str::trim)
                .unwrap_or_default()
                .to_owned();
            let mut trigger_rules = planner_string_array(&args, "trigger_rules");
            if trigger_rules.is_empty() {
                trigger_rules.push(name.clone());
            }
            let resource_path = args
                .get("resource_path")
                .and_then(|value| value.as_str())
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(ToOwned::to_owned);
            self.state.lock().await.plan = Some(MainAgentPlan::CreateSkillDefinition {
                input: CreateSkill {
                    name,
                    description,
                    trigger_rules,
                    tool_subset: planner_string_array(&args, "tool_subset"),
                    resource_path,
                },
            });
            Ok(planner_tool_result("planned skill definition creation"))
        })
    }
}

struct PlanUpdateSkillDefinitionTool {
    state: Arc<tokio::sync::Mutex<MainAgentPlannerState>>,
    schema: serde_json::Value,
}

impl PlanUpdateSkillDefinitionTool {
    fn new(state: Arc<tokio::sync::Mutex<MainAgentPlannerState>>) -> Self {
        Self {
            state,
            schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "selector": {
                        "type": "string",
                        "description": "Skill id, name, or name fragment identifying the skill."
                    },
                    "name": { "type": "string" },
                    "description": { "type": "string" },
                    "trigger_rules": {
                        "type": "array",
                        "items": { "type": "string" }
                    },
                    "tool_subset": {
                        "type": "array",
                        "items": { "type": "string" }
                    },
                    "resource_path": { "type": "string" },
                    "clear_resource_path": { "type": "boolean" }
                },
                "required": ["selector"]
            }),
        }
    }
}

impl Tool for PlanUpdateSkillDefinitionTool {
    fn name(&self) -> &str {
        "plan_update_skill_definition"
    }

    fn description(&self) -> &str {
        "Plan to update fields on an existing skill definition."
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
            let selector = planner_required_string(&args, "selector")?;
            let resource_path = if args
                .get("clear_resource_path")
                .and_then(|value| value.as_bool())
                .unwrap_or(false)
            {
                Some(None)
            } else {
                args.get("resource_path")
                    .and_then(|value| value.as_str())
                    .map(str::trim)
                    .filter(|value| !value.is_empty())
                    .map(ToOwned::to_owned)
                    .map(Some)
            };
            self.state.lock().await.plan = Some(MainAgentPlan::UpdateSkillDefinition {
                selector,
                input: UpdateSkill {
                    name: planner_optional_string(&args, "name"),
                    description: planner_optional_string(&args, "description"),
                    trigger_rules: planner_optional_string_array(&args, "trigger_rules"),
                    tool_subset: planner_optional_string_array(&args, "tool_subset"),
                    resource_path,
                },
            });
            Ok(planner_tool_result("planned skill definition update"))
        })
    }
}

struct PlanDeleteSkillDefinitionTool {
    state: Arc<tokio::sync::Mutex<MainAgentPlannerState>>,
    schema: serde_json::Value,
}

impl PlanDeleteSkillDefinitionTool {
    fn new(state: Arc<tokio::sync::Mutex<MainAgentPlannerState>>) -> Self {
        Self {
            state,
            schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "selector": {
                        "type": "string",
                        "description": "Skill id, name, or name fragment identifying the skill."
                    }
                },
                "required": ["selector"]
            }),
        }
    }
}

impl Tool for PlanDeleteSkillDefinitionTool {
    fn name(&self) -> &str {
        "plan_delete_skill_definition"
    }

    fn description(&self) -> &str {
        "Plan to delete an existing skill definition."
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
            let selector = planner_required_string(&args, "selector")?;
            self.state.lock().await.plan = Some(MainAgentPlan::DeleteSkillDefinition { selector });
            Ok(planner_tool_result("planned skill definition deletion"))
        })
    }
}

#[derive(Clone, Copy)]
enum PlannedResourceLockAction {
    Add,
    Remove,
}

struct PlanResourceLockTool {
    state: Arc<tokio::sync::Mutex<MainAgentPlannerState>>,
    name: &'static str,
    description: &'static str,
    action: PlannedResourceLockAction,
    schema: serde_json::Value,
}

impl PlanResourceLockTool {
    fn new(
        state: Arc<tokio::sync::Mutex<MainAgentPlannerState>>,
        name: &'static str,
        description: &'static str,
        action: PlannedResourceLockAction,
    ) -> Self {
        Self {
            state,
            name,
            description,
            action,
            schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "selector": {
                        "type": "string",
                        "description": "Task id or title fragment identifying the task."
                    },
                    "resource_key": {
                        "type": "string",
                        "description": "Resource key such as repo:owner/name."
                    }
                },
                "required": ["selector", "resource_key"]
            }),
        }
    }
}

impl Tool for PlanResourceLockTool {
    fn name(&self) -> &str {
        self.name
    }

    fn description(&self) -> &str {
        self.description
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
            let selector = planner_required_string(&args, "selector")?;
            let resource_key = planner_required_string(&args, "resource_key")?;
            let plan = match self.action {
                PlannedResourceLockAction::Add => MainAgentPlan::AddResourceLock {
                    selector,
                    resource_key,
                },
                PlannedResourceLockAction::Remove => MainAgentPlan::RemoveResourceLock {
                    selector,
                    resource_key,
                },
            };
            self.state.lock().await.plan = Some(plan);
            Ok(planner_tool_result("planned task resource lock change"))
        })
    }
}

struct PlanRequestClarificationTool {
    state: Arc<tokio::sync::Mutex<MainAgentPlannerState>>,
    schema: serde_json::Value,
}

impl PlanRequestClarificationTool {
    fn new(state: Arc<tokio::sync::Mutex<MainAgentPlannerState>>) -> Self {
        Self {
            state,
            schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "selector": {
                        "type": "string",
                        "description": "Task id or title fragment identifying the task."
                    },
                    "question": { "type": "string" }
                },
                "required": ["selector", "question"]
            }),
        }
    }
}

impl Tool for PlanRequestClarificationTool {
    fn name(&self) -> &str {
        "plan_request_clarification"
    }

    fn description(&self) -> &str {
        "Plan to ask the user a clarification question for one task."
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
            self.state.lock().await.plan = Some(MainAgentPlan::RequestClarification {
                selector: planner_required_string(&args, "selector")?,
                question: planner_required_string(&args, "question")?,
            });
            Ok(planner_tool_result("planned user clarification request"))
        })
    }
}

struct PlanReplyToTaskTool {
    state: Arc<tokio::sync::Mutex<MainAgentPlannerState>>,
    schema: serde_json::Value,
}

impl PlanReplyToTaskTool {
    fn new(state: Arc<tokio::sync::Mutex<MainAgentPlannerState>>) -> Self {
        Self {
            state,
            schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "selector": {
                        "type": "string",
                        "description": "Task id or title fragment identifying the task."
                    },
                    "content": {
                        "type": "string",
                        "description": "User reply to append to the task conversation."
                    }
                },
                "required": ["selector", "content"]
            }),
        }
    }
}

impl Tool for PlanReplyToTaskTool {
    fn name(&self) -> &str {
        "plan_reply_to_task"
    }

    fn description(&self) -> &str {
        "Plan to append a user reply to a task conversation and resume it if it is waiting for user input."
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
            self.state.lock().await.plan = Some(MainAgentPlan::ReplyToTask {
                selector: planner_required_string(&args, "selector")?,
                content: planner_required_string(&args, "content")?,
            });
            Ok(planner_tool_result("planned task conversation reply"))
        })
    }
}

struct PlanConvertTaskTypeTool {
    state: Arc<tokio::sync::Mutex<MainAgentPlannerState>>,
    schema: serde_json::Value,
}

impl PlanConvertTaskTypeTool {
    fn new(state: Arc<tokio::sync::Mutex<MainAgentPlannerState>>) -> Self {
        Self {
            state,
            schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "selector": {
                        "type": "string",
                        "description": "Task id or title fragment identifying the task."
                    },
                    "task_type": {
                        "type": "string",
                        "enum": ["one_off", "recurring"]
                    },
                    "interval_seconds": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "Required when converting to recurring unless the user left the cadence implicit."
                    }
                },
                "required": ["selector", "task_type"]
            }),
        }
    }
}

impl Tool for PlanConvertTaskTypeTool {
    fn name(&self) -> &str {
        "plan_convert_task_type"
    }

    fn description(&self) -> &str {
        "Plan to convert an existing task between one-off and recurring."
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
            let selector = planner_required_string(&args, "selector")?;
            let task_type = planner_task_type(&args)?;
            let interval_seconds = args
                .get("interval_seconds")
                .and_then(|value| value.as_i64());
            self.state.lock().await.plan = Some(MainAgentPlan::ConvertTaskType {
                selector,
                task_type,
                interval_seconds,
            });
            Ok(planner_tool_result("planned task type conversion"))
        })
    }
}

struct PlanUpdateTaskScheduleTool {
    state: Arc<tokio::sync::Mutex<MainAgentPlannerState>>,
    schema: serde_json::Value,
}

impl PlanUpdateTaskScheduleTool {
    fn new(state: Arc<tokio::sync::Mutex<MainAgentPlannerState>>) -> Self {
        Self {
            state,
            schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "selector": {
                        "type": "string",
                        "description": "Task id or title fragment identifying the recurring task."
                    },
                    "interval_seconds": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "New recurring interval in seconds."
                    }
                },
                "required": ["selector", "interval_seconds"]
            }),
        }
    }
}

impl Tool for PlanUpdateTaskScheduleTool {
    fn name(&self) -> &str {
        "plan_update_task_schedule"
    }

    fn description(&self) -> &str {
        "Plan to update the recurring schedule interval for an existing task."
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
            let selector = planner_required_string(&args, "selector")?;
            let interval_seconds = args
                .get("interval_seconds")
                .and_then(|value| value.as_i64())
                .filter(|value| *value > 0)
                .ok_or_else(|| {
                    ToolError::InvalidArguments(
                        "interval_seconds must be a positive integer".to_owned(),
                    )
                })?;
            self.state.lock().await.plan = Some(MainAgentPlan::UpdateTaskSchedule {
                selector,
                interval_seconds,
            });
            Ok(planner_tool_result("planned task schedule update"))
        })
    }
}

#[derive(Clone, Copy)]
enum PlannedTaskReadAction {
    Explain,
    ListConstraints,
    ListMemories,
    ListArtifacts,
    ListHistory,
    ShowLatestResult,
    ListFollowUps,
    ListNotes,
    ListConversation,
}

struct PlanTaskReadTool {
    state: Arc<tokio::sync::Mutex<MainAgentPlannerState>>,
    name: &'static str,
    description: &'static str,
    action: PlannedTaskReadAction,
    schema: serde_json::Value,
}

impl PlanTaskReadTool {
    fn new(
        state: Arc<tokio::sync::Mutex<MainAgentPlannerState>>,
        name: &'static str,
        description: &'static str,
        action: PlannedTaskReadAction,
    ) -> Self {
        Self {
            state,
            name,
            description,
            action,
            schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "selector": {
                        "type": "string",
                        "description": "Task id or title fragment identifying the task."
                    }
                },
                "required": ["selector"]
            }),
        }
    }
}

impl Tool for PlanTaskReadTool {
    fn name(&self) -> &str {
        self.name
    }

    fn description(&self) -> &str {
        self.description
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
            let selector = planner_required_string(&args, "selector")?;
            let plan = match self.action {
                PlannedTaskReadAction::Explain => MainAgentPlan::ExplainTask { selector },
                PlannedTaskReadAction::ListConstraints => {
                    MainAgentPlan::ListTaskConstraints { selector }
                }
                PlannedTaskReadAction::ListMemories => MainAgentPlan::ListTaskMemories { selector },
                PlannedTaskReadAction::ListArtifacts => {
                    MainAgentPlan::ListTaskArtifacts { selector }
                }
                PlannedTaskReadAction::ListHistory => MainAgentPlan::ListTaskHistory { selector },
                PlannedTaskReadAction::ShowLatestResult => {
                    MainAgentPlan::ShowTaskLatestResult { selector }
                }
                PlannedTaskReadAction::ListFollowUps => {
                    MainAgentPlan::ListTaskFollowUps { selector }
                }
                PlannedTaskReadAction::ListNotes => MainAgentPlan::ListTaskNotes { selector },
                PlannedTaskReadAction::ListConversation => {
                    MainAgentPlan::ListTaskConversation { selector }
                }
            };
            self.state.lock().await.plan = Some(plan);
            Ok(planner_tool_result("planned task read action"))
        })
    }
}

struct PlanInspectWorkspaceFileTool {
    state: Arc<tokio::sync::Mutex<MainAgentPlannerState>>,
    schema: serde_json::Value,
}

impl PlanInspectWorkspaceFileTool {
    fn new(state: Arc<tokio::sync::Mutex<MainAgentPlannerState>>) -> Self {
        Self {
            state,
            schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Workspace-relative file path to preview."
                    }
                },
                "required": ["path"]
            }),
        }
    }
}

impl Tool for PlanInspectWorkspaceFileTool {
    fn name(&self) -> &str {
        "plan_inspect_workspace_file"
    }

    fn description(&self) -> &str {
        "Plan to preview a workspace-relative file."
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
            self.state.lock().await.plan = Some(MainAgentPlan::InspectWorkspaceFile {
                path: planner_required_string(&args, "path")?,
            });
            Ok(planner_tool_result("planned workspace file inspection"))
        })
    }
}

struct PlanInspectWorkspaceDirectoryTool {
    state: Arc<tokio::sync::Mutex<MainAgentPlannerState>>,
    schema: serde_json::Value,
}

impl PlanInspectWorkspaceDirectoryTool {
    fn new(state: Arc<tokio::sync::Mutex<MainAgentPlannerState>>) -> Self {
        Self {
            state,
            schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Workspace-relative directory path to list. Use . for the workspace root."
                    }
                },
                "required": ["path"]
            }),
        }
    }
}

impl Tool for PlanInspectWorkspaceDirectoryTool {
    fn name(&self) -> &str {
        "plan_inspect_workspace_directory"
    }

    fn description(&self) -> &str {
        "Plan to list a workspace-relative directory."
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
            self.state.lock().await.plan = Some(MainAgentPlan::InspectWorkspaceDirectory {
                path: planner_required_string(&args, "path")?,
            });
            Ok(planner_tool_result(
                "planned workspace directory inspection",
            ))
        })
    }
}

struct PlanCreateMemoryTool {
    state: Arc<tokio::sync::Mutex<MainAgentPlannerState>>,
    schema: serde_json::Value,
}

impl PlanCreateMemoryTool {
    fn new(state: Arc<tokio::sync::Mutex<MainAgentPlannerState>>) -> Self {
        Self {
            state,
            schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "scope": {
                        "type": "string",
                        "description": "Short memory scope such as repo, project, skill:rust, or github."
                    },
                    "content": {
                        "type": "string",
                        "description": "User preference, pitfall, or convention to remember for future tasks."
                    }
                },
                "required": ["content"]
            }),
        }
    }
}

impl Tool for PlanCreateMemoryTool {
    fn name(&self) -> &str {
        "plan_create_memory"
    }

    fn description(&self) -> &str {
        "Plan to store a user-provided long-term memory for future task context."
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
            let content = planner_required_string(&args, "content")?;
            let scope = args
                .get("scope")
                .and_then(|value| value.as_str())
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .unwrap_or("repo")
                .to_owned();
            self.state.lock().await.plan = Some(MainAgentPlan::CreateMemory { scope, content });
            Ok(planner_tool_result("planned memory creation"))
        })
    }
}

#[derive(Clone, Copy)]
enum PlannedMemoryReviewAction {
    Approve,
    Reject,
    Delete,
}

struct PlanMemoryReviewTool {
    state: Arc<tokio::sync::Mutex<MainAgentPlannerState>>,
    name: &'static str,
    description: &'static str,
    action: PlannedMemoryReviewAction,
    schema: serde_json::Value,
}

impl PlanMemoryReviewTool {
    fn new(
        state: Arc<tokio::sync::Mutex<MainAgentPlannerState>>,
        name: &'static str,
        description: &'static str,
        action: PlannedMemoryReviewAction,
    ) -> Self {
        Self {
            state,
            name,
            description,
            action,
            schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "selector": {
                        "type": "string",
                        "description": "Memory id, scope, or content fragment identifying the memory candidate."
                    }
                },
                "required": ["selector"]
            }),
        }
    }
}

impl Tool for PlanMemoryReviewTool {
    fn name(&self) -> &str {
        self.name
    }

    fn description(&self) -> &str {
        self.description
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
            let selector = planner_required_string(&args, "selector")?;
            let plan = match self.action {
                PlannedMemoryReviewAction::Approve => MainAgentPlan::ApproveMemory { selector },
                PlannedMemoryReviewAction::Reject => MainAgentPlan::RejectMemory { selector },
                PlannedMemoryReviewAction::Delete => MainAgentPlan::DeleteMemory { selector },
            };
            self.state.lock().await.plan = Some(plan);
            Ok(planner_tool_result("planned memory review action"))
        })
    }
}

struct PlanTaskMemoryReviewTool {
    state: Arc<tokio::sync::Mutex<MainAgentPlannerState>>,
    schema: serde_json::Value,
}

impl PlanTaskMemoryReviewTool {
    fn new(state: Arc<tokio::sync::Mutex<MainAgentPlannerState>>) -> Self {
        Self {
            state,
            schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "selector": {
                        "type": "string",
                        "description": "Task id or title fragment identifying the source task."
                    },
                    "status": {
                        "type": "string",
                        "enum": ["approved", "rejected"],
                        "description": "Terminal review status to apply to pending memory candidates from the selected task."
                    }
                },
                "required": ["selector", "status"]
            }),
        }
    }
}

impl Tool for PlanTaskMemoryReviewTool {
    fn name(&self) -> &str {
        "plan_review_task_memories"
    }

    fn description(&self) -> &str {
        "Plan to approve or reject all pending memory candidates proposed by one task."
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
            let selector = planner_required_string(&args, "selector")?;
            let status_text = planner_required_string(&args, "status")?;
            let status = match status_text.to_ascii_lowercase().as_str() {
                "approved" | "approve" | "accepted" | "accept" => MemoryStatus::Approved,
                "rejected" | "reject" | "discarded" | "discard" => MemoryStatus::Rejected,
                _ => {
                    return Err(ToolError::InvalidArguments(
                        "status must be approved or rejected".to_owned(),
                    ));
                }
            };
            self.state.lock().await.plan =
                Some(MainAgentPlan::BulkReviewTaskMemories { selector, status });
            Ok(planner_tool_result("planned task memory review"))
        })
    }
}

struct PlanUpdateMemoryTool {
    state: Arc<tokio::sync::Mutex<MainAgentPlannerState>>,
    schema: serde_json::Value,
}

impl PlanUpdateMemoryTool {
    fn new(state: Arc<tokio::sync::Mutex<MainAgentPlannerState>>) -> Self {
        Self {
            state,
            schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "selector": {
                        "type": "string",
                        "description": "Memory id, scope, or content fragment identifying the memory."
                    },
                    "scope": {
                        "type": "string",
                        "description": "Optional replacement scope."
                    },
                    "content": {
                        "type": "string",
                        "description": "Optional replacement memory content."
                    },
                    "confidence": {
                        "type": "number",
                        "description": "Optional replacement confidence from 0.0 to 1.0."
                    }
                },
                "required": ["selector"]
            }),
        }
    }
}

impl Tool for PlanUpdateMemoryTool {
    fn name(&self) -> &str {
        "plan_update_memory"
    }

    fn description(&self) -> &str {
        "Plan to update an existing memory's scope, content, or confidence."
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
            let selector = planner_required_string(&args, "selector")?;
            let input = UpdateMemory {
                scope: planner_optional_string(&args, "scope"),
                content: planner_optional_string(&args, "content"),
                confidence: args.get("confidence").and_then(|value| value.as_f64()),
            };
            if input.scope.is_none() && input.content.is_none() && input.confidence.is_none() {
                return Err(ToolError::InvalidArguments(
                    "plan_update_memory requires at least one of scope, content, or confidence"
                        .to_owned(),
                ));
            }
            self.state.lock().await.plan = Some(MainAgentPlan::UpdateMemory { selector, input });
            Ok(planner_tool_result("planned memory update"))
        })
    }
}

struct PlanListMemoriesTool {
    state: Arc<tokio::sync::Mutex<MainAgentPlannerState>>,
    schema: serde_json::Value,
}

impl PlanListMemoriesTool {
    fn new(state: Arc<tokio::sync::Mutex<MainAgentPlannerState>>) -> Self {
        Self {
            state,
            schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "filter": {
                        "type": "string",
                        "enum": ["pending", "approved", "rejected", "all"],
                        "description": "Memory status filter; defaults to pending."
                    }
                }
            }),
        }
    }
}

impl Tool for PlanListMemoriesTool {
    fn name(&self) -> &str {
        "plan_list_memories"
    }

    fn description(&self) -> &str {
        "Plan to list memory candidates for review."
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
            self.state.lock().await.plan = Some(MainAgentPlan::ListMemories {
                filter: planner_memory_filter(&args)?,
            });
            Ok(planner_tool_result("planned memory list"))
        })
    }
}

struct PlanListTasksByStatusTool {
    state: Arc<tokio::sync::Mutex<MainAgentPlannerState>>,
    schema: serde_json::Value,
}

impl PlanListTasksByStatusTool {
    fn new(state: Arc<tokio::sync::Mutex<MainAgentPlannerState>>) -> Self {
        Self {
            state,
            schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "enum": [
                            "draft",
                            "queued",
                            "running",
                            "waiting_for_user",
                            "waiting_for_schedule",
                            "completed",
                            "failed",
                            "cancelled",
                            "paused"
                        ]
                    }
                },
                "required": ["status"]
            }),
        }
    }
}

impl Tool for PlanListTasksByStatusTool {
    fn name(&self) -> &str {
        "plan_list_tasks_by_status"
    }

    fn description(&self) -> &str {
        "Plan to list tasks filtered by lifecycle status."
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
            self.state.lock().await.plan = Some(MainAgentPlan::ListTasksByStatus {
                status: planner_task_status(&args)?,
            });
            Ok(planner_tool_result("planned task status list"))
        })
    }
}

struct PlanSimpleIntentTool {
    state: Arc<tokio::sync::Mutex<MainAgentPlannerState>>,
    name: &'static str,
    description: &'static str,
    plan: MainAgentPlan,
    schema: serde_json::Value,
}

impl PlanSimpleIntentTool {
    fn new(
        state: Arc<tokio::sync::Mutex<MainAgentPlannerState>>,
        name: &'static str,
        description: &'static str,
        plan: MainAgentPlan,
    ) -> Self {
        Self {
            state,
            name,
            description,
            plan,
            schema: serde_json::json!({ "type": "object", "properties": {} }),
        }
    }
}

impl Tool for PlanSimpleIntentTool {
    fn name(&self) -> &str {
        self.name
    }

    fn description(&self) -> &str {
        self.description
    }

    fn parameters_schema(&self) -> &serde_json::Value {
        &self.schema
    }

    fn execution_mode(&self) -> ToolExecutionMode {
        ToolExecutionMode::Sequential
    }

    fn execute<'a>(
        &'a self,
        _args: serde_json::Value,
        _ctx: &'a ToolContext,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<ToolResult, ToolError>> + Send + 'a>,
    > {
        Box::pin(async move {
            self.state.lock().await.plan = Some(self.plan.clone());
            Ok(planner_tool_result("planned main-agent action"))
        })
    }
}

fn planner_required_string(args: &serde_json::Value, field: &str) -> Result<String, ToolError> {
    args.get(field)
        .and_then(|value| value.as_str())
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
        .ok_or_else(|| ToolError::InvalidArguments(format!("{field} is required")))
}

fn planner_optional_string(args: &serde_json::Value, field: &str) -> Option<String> {
    args.get(field)
        .and_then(|value| value.as_str())
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
}

fn planner_task_type(args: &serde_json::Value) -> Result<TaskType, ToolError> {
    match args
        .get("task_type")
        .and_then(|value| value.as_str())
        .unwrap_or("one_off")
    {
        "one_off" => Ok(TaskType::OneOff),
        "recurring" => Ok(TaskType::Recurring),
        value => Err(ToolError::InvalidArguments(format!(
            "unsupported task_type: {value}"
        ))),
    }
}

fn planner_task_status(args: &serde_json::Value) -> Result<TaskStatus, ToolError> {
    let value = planner_required_string(args, "status")?;
    value
        .parse::<TaskStatus>()
        .map_err(|_| ToolError::InvalidArguments(format!("unsupported task status: {value}")))
}

fn planner_memory_filter(args: &serde_json::Value) -> Result<MemoryListFilter, ToolError> {
    match args
        .get("filter")
        .and_then(|value| value.as_str())
        .unwrap_or("pending")
    {
        "pending" => Ok(MemoryListFilter::Pending),
        "approved" => Ok(MemoryListFilter::Approved),
        "rejected" => Ok(MemoryListFilter::Rejected),
        "all" => Ok(MemoryListFilter::All),
        value => Err(ToolError::InvalidArguments(format!(
            "unsupported memory filter: {value}"
        ))),
    }
}

fn planner_string_array(args: &serde_json::Value, field: &str) -> Vec<String> {
    args.get(field)
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
        .unwrap_or_default()
}

fn planner_optional_string_array(args: &serde_json::Value, field: &str) -> Option<Vec<String>> {
    args.get(field).map(|_| planner_string_array(args, field))
}

fn planner_tool_result(message: &str) -> ToolResult {
    ToolResult {
        content: vec![ContentBlock::Text {
            text: message.to_owned(),
        }],
        details: serde_json::json!({ "message": message }),
        terminate: true,
    }
}

fn main_agent_planner_prompt(context: &MainAgentPlanContext) -> String {
    format!(
        "User message:\n{}\n\nTask pool summary:\n{}\n\nTask snapshot:\n{}\n\nRecent main conversation:\n{}\n\nSupported planning tools:\n- plan_create_task: create one one-off or recurring task.\n- plan_split_tasks: split a goal into multiple one-off tasks.\n- plan_pause_task: pause one existing task by id or title fragment.\n- plan_resume_task: resume a paused, blocked, or waiting task selected by id or title fragment.\n- plan_cancel_task: cancel one existing task by id or title fragment.\n- plan_delete_task: permanently delete one existing task by id or title fragment.\n- plan_complete_task: manually mark one existing task complete with a result summary.\n- plan_fail_task: manually mark one existing task failed with an error reason.\n- plan_retry_task: requeue a failed task for another attempt.\n- plan_run_task_now: move one selected task to the front of runnable work and request a scheduler scan.\n- plan_run_next_task: select the next runnable task by scheduler order and request a scheduler scan.\n- plan_update_task_details: update one existing task's title and/or description.\n- plan_reprioritize_task: set one existing task's priority.\n- plan_reorder_task: move one task to a queue position.\n- plan_convert_task_type: convert one task between one-off and recurring.\n- plan_update_task_schedule: update one recurring task's interval in seconds.\n- plan_add_task_dependency: make one task wait for another task.\n- plan_remove_task_dependency: remove a task dependency.\n- plan_add_task_note: add a note to one task.\n- plan_list_task_notes: list notes attached to one task.\n- plan_add_requested_skills: add one or more requested skills to an existing task.\n- plan_remove_requested_skills: remove one or more requested skills from an existing task.\n- plan_create_skill_definition: create a reusable skill definition with triggers, tools, and optional resource path.\n- plan_update_skill_definition: update an existing skill definition's metadata, triggers, tools, or resource path.\n- plan_delete_skill_definition: delete an existing skill definition.\n- plan_add_resource_lock: add a resource lock to one task.\n- plan_remove_resource_lock: remove a resource lock from one task.\n- plan_request_clarification: ask the user a clarification question for one task.\n- plan_reply_to_task: append the user's reply to a task conversation and resume it if it is waiting for input.\n- plan_list_main_agent_actions: list recent main-agent audit actions.\n- plan_list_waiting_for_user_tasks: list tasks waiting for user input.\n- plan_list_waiting_for_schedule_tasks: list tasks waiting for their next schedule.\n- plan_explain_task_pool: explain current task pool state.\n- plan_recommend_next_action: recommend the next operator action for the task pool.\n- plan_explain_task: explain one task's current state.\n- plan_list_task_artifacts: list artifacts for one task.\n- plan_list_task_history: list attempts, worker events, and audit actions for one task.\n- plan_show_task_latest_result: show one task's latest result summary.\n- plan_list_task_follow_ups: list follow-up tasks created from one task.\n- plan_inspect_workspace: inspect workspace status.\n- plan_inspect_workspace_file: preview a workspace-relative file.\n- plan_inspect_workspace_directory: list a workspace-relative directory.\n- plan_approve_memory: approve one memory candidate.\n- plan_reject_memory: reject one memory candidate.\n- plan_approve_all_pending_memories: approve all pending memory candidates.\n- plan_reject_all_pending_memories: reject all pending memory candidates.\n- plan_list_memories: list memory candidates by status.\n- plan_list_skill_definitions: list skill definitions.\n- plan_list_tasks: list tasks.\n- plan_summarize_task_pool: summarize task pool state.\n- plan_scheduler_scan: run one scheduler scan.\n\nCall one tool only when the user intent is clear.",
        context.user_message,
        format_advisor_summary(&context.task_pool_summary),
        format_planner_task_snapshot(&context.task_snapshot),
        format_advisor_recent_messages(&context.recent_messages),
    )
    .replace(
        "- plan_list_memories: list memory candidates by status.",
        "- plan_create_memory: store a user-provided long-term memory for future tasks.\n- plan_update_memory: update an existing memory's scope, content, or confidence.\n- plan_delete_memory: delete an existing memory.\n- plan_list_memories: list memory candidates by status.\n- plan_show_scheduler_state: show current scheduler execution state without running a scan.",
    )
    .replace(
        "- plan_reject_all_pending_memories: reject all pending memory candidates.",
        "- plan_reject_all_pending_memories: reject all pending memory candidates.\n- plan_review_task_memories: approve or reject all pending memory candidates proposed by one task.",
    )
    .replace(
        "- plan_list_task_artifacts: list artifacts for one task.",
        "- plan_list_task_artifacts: list artifacts for one task.\n- plan_list_task_history: list attempts, worker events, and audit actions for one task.",
    )
    .replace(
        "- plan_explain_task: explain one task's current state.",
        "- plan_explain_task: explain one task's current state.\n- plan_list_task_constraints: list one task's dependencies and resource locks.",
    )
    .replace(
        "- plan_list_task_artifacts: list artifacts for one task.",
        "- plan_list_task_memories: list memory candidates proposed by one task.\n- plan_list_task_artifacts: list artifacts for one task.",
    )
    .replace(
        "- plan_reply_to_task: append the user's reply to a task conversation and resume it if it is waiting for input.",
        "- plan_reply_to_task: append the user's reply to a task conversation and resume it if it is waiting for input.\n- plan_list_task_conversation: list recent conversation messages for one task.",
    )
    .replace(
        "- plan_list_tasks: list tasks.",
        "- plan_list_tasks: list tasks.\n- plan_list_tasks_by_status: list tasks filtered by lifecycle status.",
    )
}

fn main_agent_advisor_prompt(context: &MainAgentAdviceContext) -> String {
    format!(
        "User message:\n{}\n\nDeterministic reply already produced by the task manager:\n{}\n\nScheduler scan requested: {}\n\nChanged tasks:\n{}\n\nTask pool summary:\n{}\n\nRecent main conversation:\n{}",
        context.user_message,
        context.deterministic_reply,
        context.scheduler_tick_requested,
        format_advisor_changed_tasks(&context.changed_tasks),
        format_advisor_summary(&context.task_pool_summary),
        format_advisor_recent_messages(&context.recent_messages),
    )
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

fn format_advisor_changed_tasks(tasks: &[Task]) -> String {
    if tasks.is_empty() {
        return "none".to_owned();
    }

    tasks
        .iter()
        .map(|task| {
            format!(
                "- {} [{}] priority {} queue {}",
                task.title, task.status, task.priority, task.queue_position
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn format_planner_task_snapshot(tasks: &[Task]) -> String {
    if tasks.is_empty() {
        return "none".to_owned();
    }

    let mut lines = Vec::new();
    for task in tasks.iter().take(20) {
        let blocked = task
            .blocked_reason
            .as_deref()
            .map(|reason| format!(" blocked_reason={}", bounded_preview(reason, 120)))
            .unwrap_or_default();
        lines.push(format!(
            "- id={} title={} status={} type={} priority={} queue={}{}",
            task.id.to_string().chars().take(8).collect::<String>(),
            bounded_preview(&task.title, 120),
            task.status,
            task.task_type,
            task.priority,
            task.queue_position,
            blocked
        ));
    }
    if tasks.len() > 20 {
        lines.push(format!("- ... and {} more", tasks.len() - 20));
    }
    lines.join("\n")
}

fn format_advisor_summary(summary: &TaskPoolSummary) -> String {
    format!(
        "{} total, {} queued, {} running, {} waiting for user, {} waiting for schedule, {} completed, {} failed, {} cancelled, {} paused",
        summary.total,
        summary.queued,
        summary.running,
        summary.waiting_for_user,
        summary.waiting_for_schedule,
        summary.completed,
        summary.failed,
        summary.cancelled,
        summary.paused
    )
}

fn format_advisor_recent_messages(messages: &[ConversationMessage]) -> String {
    if messages.is_empty() {
        return "none".to_owned();
    }

    messages
        .iter()
        .map(|message| format!("- {}: {}", message.role, message.content))
        .collect::<Vec<_>>()
        .join("\n")
}

fn contains_any(haystack: &str, needles: &[&str]) -> bool {
    needles.iter().any(|needle| haystack.contains(needle))
}

fn replace_case_insensitive(value: &str, needle: &str, replacement: &str) -> String {
    let needle = needle.to_lowercase();
    let mut result = value.to_owned();

    while let Some(index) = normalized_slice(&result).find(&needle) {
        let end = index + needle.len();
        result.replace_range(index..end, replacement);
    }

    result
}

fn normalized_slice(value: &str) -> String {
    value.to_lowercase()
}

fn bounded_preview(value: &str, max_chars: usize) -> String {
    let mut output = value.chars().take(max_chars).collect::<String>();
    if value.chars().count() > max_chars {
        output.push_str("...");
    }
    output
}

fn task_selector_status(selector: &str) -> Option<TaskStatus> {
    let selector = selector.trim();
    match selector {
        "blocked"
        | "blocked task"
        | "waiting"
        | "waiting task"
        | "waiting for user"
        | "waiting for user task" => Some(TaskStatus::WaitingForUser),
        "running" | "running task" => Some(TaskStatus::Running),
        "queued" | "queued task" | "next task" => Some(TaskStatus::Queued),
        "paused" | "paused task" => Some(TaskStatus::Paused),
        "failed" | "failed task" => Some(TaskStatus::Failed),
        _ => None,
    }
}

fn is_create_request(normalized: &str) -> bool {
    contains_any(
        normalized,
        &[ZH_CREATE, ZH_NEW, ZH_ADD, ZH_ADD_ONE, "create", "add task"],
    )
}

fn parse_split_intent(content: &str, normalized: &str) -> Option<MainAgentIntent> {
    if !contains_any(
        normalized,
        &[
            "split goal",
            "split task",
            "split into tasks",
            "decompose goal",
            "break down",
            ZH_SPLIT,
            ZH_DECOMPOSE,
        ],
    ) {
        return None;
    }

    let tail = split_tail(content)?;
    let titles = extract_split_titles(tail);
    (titles.len() >= 2).then_some(MainAgentIntent::SplitTasks { titles })
}

fn parse_skill_definition_intent(content: &str, normalized: &str) -> Option<MainAgentIntent> {
    if is_skill_definition_list_request(normalized) {
        return Some(MainAgentIntent::ListSkillDefinitions);
    }

    if let Some(selector) = extract_delete_skill_definition_selector(content, normalized) {
        return Some(MainAgentIntent::DeleteSkillDefinition { selector });
    }

    if let Some((selector, input)) = extract_update_skill_definition(content, normalized) {
        return Some(MainAgentIntent::UpdateSkillDefinition { selector, input });
    }

    extract_create_skill_definition(content, normalized)
        .map(|input| MainAgentIntent::CreateSkillDefinition { input })
}

fn is_skill_definition_list_request(normalized: &str) -> bool {
    contains_any(
        normalized,
        &[
            "list skills",
            "show skills",
            "list skill definitions",
            "show skill definitions",
        ],
    ) || (contains_any(normalized, &[ZH_LIST]) && contains_any(normalized, &[ZH_SKILL]))
}

fn extract_delete_skill_definition_selector(content: &str, normalized: &str) -> Option<String> {
    if contains_any(normalized, &[" from task", " to task", " requested skill"]) {
        return None;
    }

    for prefix in ["delete skill definition", "delete skill", "remove skill"] {
        if normalized.starts_with(prefix) {
            let selector = content[prefix.len()..]
                .trim()
                .trim_matches([':', '\u{ff1a}', '"', '\''])
                .trim()
                .to_owned();
            return (!selector.is_empty()).then_some(selector);
        }
    }

    None
}

fn extract_create_skill_definition(content: &str, normalized: &str) -> Option<CreateSkill> {
    let prefix = if normalized.starts_with("create skill definition") {
        "create skill definition"
    } else if normalized.starts_with("create skill") {
        "create skill"
    } else if normalized.starts_with("add skill definition") {
        "add skill definition"
    } else {
        return None;
    };

    let body = content[prefix.len()..].trim();
    if body.is_empty() {
        return None;
    }

    let mut sections = body
        .split([';', '\n'])
        .map(str::trim)
        .filter(|section| !section.is_empty());
    let head = sections.next()?;
    let (name, mut description) = split_skill_name_description(head);
    if name.is_empty() {
        return None;
    }

    let mut trigger_rules = Vec::new();
    let mut tool_subset = Vec::new();
    let mut resource_path = None;

    for section in sections {
        let normalized_section = section.to_lowercase();
        if let Some(value) =
            strip_skill_clause_value(section, &normalized_section, &["description", "desc"])
        {
            description = value;
            continue;
        }
        if let Some(value) = strip_skill_clause_value(
            section,
            &normalized_section,
            &["trigger rules", "triggers", "trigger", "rules"],
        ) {
            trigger_rules = parse_skill_names(&value);
            continue;
        }
        if let Some(value) = strip_skill_clause_value(
            section,
            &normalized_section,
            &["tool subset", "tools", "tool"],
        ) {
            tool_subset = parse_skill_names(&value);
            continue;
        }
        if let Some(value) = strip_skill_clause_value(
            section,
            &normalized_section,
            &["resource path", "resource_path", "resource"],
        ) {
            let value = value.trim().to_owned();
            if !value.is_empty() {
                resource_path = Some(value);
            }
        }
    }

    if trigger_rules.is_empty() {
        trigger_rules.push(name.clone());
    }

    Some(CreateSkill {
        name,
        description,
        trigger_rules,
        tool_subset,
        resource_path,
    })
}

fn extract_update_skill_definition(
    content: &str,
    normalized: &str,
) -> Option<(String, UpdateSkill)> {
    let prefix = if normalized.starts_with("update skill definition") {
        "update skill definition"
    } else if normalized.starts_with("update skill") {
        "update skill"
    } else if normalized.starts_with("edit skill definition") {
        "edit skill definition"
    } else if normalized.starts_with("edit skill") {
        "edit skill"
    } else if normalized.starts_with("set skill") {
        "set skill"
    } else {
        return None;
    };

    let body = content[prefix.len()..].trim();
    if body.is_empty() {
        return None;
    }

    let mut sections = body
        .split([';', '\n'])
        .map(str::trim)
        .filter(|section| !section.is_empty());
    let selector = clean_skill_field(sections.next()?);
    if selector.is_empty() {
        return None;
    }

    let mut input = UpdateSkill::default();
    for section in sections {
        let normalized_section = section.to_lowercase();
        if let Some(value) =
            strip_skill_clause_value(section, &normalized_section, &["name", "rename"])
        {
            let value = clean_skill_field(&value);
            if !value.is_empty() {
                input.name = Some(value);
            }
            continue;
        }
        if let Some(value) =
            strip_skill_clause_value(section, &normalized_section, &["description", "desc"])
        {
            input.description = Some(value);
            continue;
        }
        if let Some(value) = strip_skill_clause_value(
            section,
            &normalized_section,
            &["trigger rules", "triggers", "trigger", "rules"],
        ) {
            input.trigger_rules = Some(parse_skill_names(&value));
            continue;
        }
        if let Some(value) = strip_skill_clause_value(
            section,
            &normalized_section,
            &["tool subset", "tools", "tool"],
        ) {
            input.tool_subset = Some(parse_skill_names(&value));
            continue;
        }
        if let Some(value) = strip_skill_clause_value(
            section,
            &normalized_section,
            &["resource path", "resource_path", "resource"],
        ) {
            let value = value.trim().to_owned();
            if value.is_empty()
                || matches!(
                    value.to_ascii_lowercase().as_str(),
                    "none" | "null" | "clear" | "remove" | "delete"
                )
            {
                input.resource_path = Some(None);
            } else {
                input.resource_path = Some(Some(value));
            }
        }
    }

    has_skill_update(&input).then_some((selector, input))
}

fn has_skill_update(input: &UpdateSkill) -> bool {
    input.name.is_some()
        || input.description.is_some()
        || input.trigger_rules.is_some()
        || input.tool_subset.is_some()
        || input.resource_path.is_some()
}

fn split_skill_name_description(head: &str) -> (String, String) {
    let head = clean_skill_field(head);
    for separator in [":", "\u{ff1a}", " - "] {
        if let Some((name, description)) = head.split_once(separator) {
            let name = clean_skill_field(name);
            let description = clean_skill_field(description);
            return (name, description);
        }
    }

    (head.clone(), head)
}

fn strip_skill_clause_value(
    section: &str,
    normalized_section: &str,
    labels: &[&str],
) -> Option<String> {
    for label in labels {
        if normalized_section.starts_with(label) {
            let value = section[label.len()..]
                .trim()
                .trim_matches([':', '\u{ff1a}', '=', '"', '\''])
                .trim()
                .to_owned();
            return Some(value);
        }
    }

    None
}

fn clean_skill_field(value: &str) -> String {
    value
        .trim()
        .trim_matches([
            ':', '\u{ff1a}', ',', '\u{ff0c}', '.', '\u{3002}', '"', '\'', '`',
        ])
        .trim()
        .to_owned()
}

fn split_tail(content: &str) -> Option<&str> {
    for separator in ["\u{ff1a}", ":", "\n"] {
        if let Some((_, tail)) = content.split_once(separator) {
            let tail = tail.trim();
            if !tail.is_empty() {
                return Some(tail);
            }
        }
    }

    None
}

fn extract_split_titles(content: &str) -> Vec<String> {
    content
        .split(['\n', ';', '\u{ff1b}'])
        .flat_map(|part| part.split("、"))
        .map(clean_split_title)
        .filter(|title| !title.is_empty())
        .map(|title| clamp_title(&title))
        .collect()
}

fn clean_split_title(value: &str) -> String {
    let value = value.trim();
    let value = value
        .trim_start_matches(|ch: char| {
            ch.is_ascii_whitespace()
                || ch == '-'
                || ch == '*'
                || ch == '\u{2022}'
                || ch == '.'
                || ch == ')'
                || ch == '\u{3001}'
        })
        .trim();
    let value = value
        .trim_start_matches(|ch: char| ch.is_ascii_digit())
        .trim_start_matches(['.', ')', '\u{3001}'])
        .trim();

    value.to_owned()
}

fn is_convert_request(normalized: &str) -> bool {
    contains_any(
        normalized,
        &[
            "convert task",
            "change task type",
            "make task",
            "task type",
            ZH_CONVERT_TASK,
            ZH_CHANGE_TO,
            ZH_CHANGE_AS,
            ZH_SET_AS,
        ],
    )
}

fn is_list_tasks_request(normalized: &str) -> bool {
    contains_any(
        normalized,
        &[
            "list tasks",
            "show tasks",
            "show task list",
            "task list",
            "queue list",
            ZH_LIST,
            "\u{4efb}\u{52a1}\u{5217}\u{8868}",
        ],
    ) && !is_create_request(normalized)
}

fn is_waiting_for_user_list_request(normalized: &str) -> bool {
    contains_any(
        normalized,
        &[
            "waiting for user",
            "waiting for my input",
            "need my input",
            "needs my input",
            "need user input",
            "needs user input",
            "blocked tasks",
            "blocked task",
            "what needs me",
            "\u{7b49}\u{5f85}\u{7528}\u{6237}",
            "\u{9700}\u{8981}\u{6211}",
            "\u{963b}\u{585e}\u{4efb}\u{52a1}",
        ],
    ) && !is_create_request(normalized)
        && !contains_any(
            normalized,
            &[
                "reply",
                "answer",
                "respond",
                "message",
                "\u{56de}\u{590d}",
                "\u{56de}\u{7b54}",
            ],
        )
        && contains_any(
            normalized,
            &[
                "list",
                "show",
                "what",
                "which",
                "tasks",
                "task",
                ZH_LIST,
                "\u{67e5}\u{770b}",
                ZH_TASK,
            ],
        )
}

fn is_waiting_for_schedule_list_request(normalized: &str) -> bool {
    contains_any(
        normalized,
        &[
            "waiting for schedule",
            "waiting for its schedule",
            "waiting for next schedule",
            "scheduled tasks",
            "scheduled task",
            "next scheduled",
            "recurring tasks waiting",
            "recurring task waiting",
            "waiting recurring",
            "recurring tasks are waiting",
            "recurring task is waiting",
            "\u{7b49}\u{5f85}\u{8c03}\u{5ea6}",
            "\u{7b49}\u{5f85}\u{65e5}\u{7a0b}",
            "\u{5faa}\u{73af}\u{4efb}\u{52a1}",
        ],
    ) && !is_create_request(normalized)
        && contains_any(
            normalized,
            &[
                "list",
                "show",
                "what",
                "which",
                "tasks",
                "task",
                "scheduled",
                "recurring",
                ZH_LIST,
                "\u{67e5}\u{770b}",
                ZH_TASK,
            ],
        )
}

fn parse_task_status_list_intent(normalized: &str) -> Option<MainAgentIntent> {
    if is_create_request(normalized)
        || !contains_any(
            normalized,
            &["list", "show", "which", "what", ZH_LIST, "\u{67e5}\u{770b}"],
        )
        || !contains_any(normalized, &["task", "tasks", ZH_TASK])
    {
        return None;
    }

    task_status_from_list_request(normalized)
        .map(|status| MainAgentIntent::ListTasksByStatus { status })
}

fn task_status_from_list_request(normalized: &str) -> Option<TaskStatus> {
    if contains_any(normalized, &["waiting_for_user", "waiting for user"]) {
        Some(TaskStatus::WaitingForUser)
    } else if contains_any(
        normalized,
        &[
            "waiting_for_schedule",
            "waiting for schedule",
            "waiting for next schedule",
            "scheduled",
        ],
    ) {
        Some(TaskStatus::WaitingForSchedule)
    } else if contains_any(normalized, &["queued", "queue", "\u{961f}\u{5217}"]) {
        Some(TaskStatus::Queued)
    } else if contains_any(normalized, &["running", "in progress", "\u{8fd0}\u{884c}"]) {
        Some(TaskStatus::Running)
    } else if contains_any(
        normalized,
        &["completed", "complete", "done", "\u{5b8c}\u{6210}"],
    ) {
        Some(TaskStatus::Completed)
    } else if contains_any(
        normalized,
        &["failed", "failure", "errored", "\u{5931}\u{8d25}"],
    ) {
        Some(TaskStatus::Failed)
    } else if contains_any(
        normalized,
        &["cancelled", "canceled", "cancelled", "\u{53d6}\u{6d88}"],
    ) {
        Some(TaskStatus::Cancelled)
    } else if contains_any(normalized, &["paused", "pause", "\u{6682}\u{505c}"]) {
        Some(TaskStatus::Paused)
    } else if contains_any(normalized, &["draft", "\u{8349}\u{7a3f}"]) {
        Some(TaskStatus::Draft)
    } else {
        None
    }
}

fn is_global_action_list_request(normalized: &str) -> bool {
    contains_any(
        normalized,
        &[
            "main agent audit",
            "main-agent audit",
            "global actions",
            "global action",
            "recent actions",
            "audit actions",
            "audit log",
            "tool calls",
            "\u{5ba1}\u{8ba1}\u{8bb0}\u{5f55}",
            "\u{5168}\u{5c40}\u{52a8}\u{4f5c}",
            "\u{5de5}\u{5177}\u{8c03}\u{7528}",
        ],
    ) && !is_create_request(normalized)
}

fn extract_task_result_selector(content: &str, normalized: &str) -> Option<String> {
    if !contains_any(
        normalized,
        &[
            "result",
            "latest result",
            "outcome",
            "latest outcome",
            "summary",
            ZH_RESULT,
            "\u{7ed3}\u{679c}",
            "\u{6700}\u{65b0}\u{7ed3}\u{679c}",
        ],
    ) || !contains_any(normalized, &["task", ZH_TASK])
        || contains_any(
            normalized,
            &["artifact", "artifacts", "output", "outputs", ZH_ARTIFACT],
        )
        || is_create_request(normalized)
    {
        return None;
    }

    for prefix in [
        "show result for task",
        "show latest result for task",
        "show outcome for task",
        "latest result for task",
        "result for task",
        "task result",
        "task latest result",
        "task outcome",
    ] {
        if let Some(index) = normalized.find(prefix) {
            let selector = content[index + prefix.len()..]
                .trim()
                .trim_matches([':', '\u{ff1a}', '?', '\u{ff1f}', '"', '\'']);
            if !selector.is_empty() {
                return Some(selector.to_owned());
            }
        }
    }

    let selector = extract_task_selector(
        content,
        &[
            "show",
            "latest",
            "result",
            "outcome",
            "summary",
            ZH_RESULT,
            "\u{7ed3}\u{679c}",
            "\u{6700}\u{65b0}",
            "\u{67e5}\u{770b}",
        ],
    );
    (!selector.is_empty() && selector != content).then_some(selector)
}

fn extract_task_artifacts_selector(content: &str, normalized: &str) -> Option<String> {
    if !contains_any(
        normalized,
        &[
            "artifact",
            "artifacts",
            "outputs",
            "output",
            ZH_ARTIFACT,
            ZH_RESULT,
        ],
    ) || !contains_any(normalized, &["task", ZH_TASK])
        || is_create_request(normalized)
    {
        return None;
    }

    for prefix in [
        "show artifacts for task",
        "list artifacts for task",
        "show outputs for task",
        "list outputs for task",
        "task artifacts",
        "task output",
        "task outputs",
    ] {
        if let Some(index) = normalized.find(prefix) {
            let selector = content[index + prefix.len()..]
                .trim()
                .trim_matches([':', '\u{ff1a}', '?', '\u{ff1f}', '"', '\'']);
            if !selector.is_empty() {
                return Some(selector.to_owned());
            }
        }
    }

    let selector = extract_task_selector(
        content,
        &[
            "show",
            "list",
            "artifacts",
            "artifact",
            "outputs",
            "output",
            ZH_LIST,
            "\u{67e5}\u{770b}",
            ZH_ARTIFACT,
            ZH_RESULT,
        ],
    );
    (!selector.is_empty() && selector != content).then_some(selector)
}

fn extract_task_history_selector(content: &str, normalized: &str) -> Option<String> {
    if !contains_any(
        normalized,
        &[
            "history",
            "execution history",
            "attempt",
            "attempts",
            "worker event",
            "worker events",
            "run evidence",
            "audit trail",
            "\u{5386}\u{53f2}",
            "\u{6267}\u{884c}\u{8bb0}\u{5f55}",
            "\u{5c1d}\u{8bd5}",
        ],
    ) || !contains_any(normalized, &["task", ZH_TASK])
        || is_create_request(normalized)
    {
        return None;
    }

    for prefix in [
        "show history for task",
        "list history for task",
        "show execution history for task",
        "list execution history for task",
        "show attempts for task",
        "list attempts for task",
        "show worker events for task",
        "list worker events for task",
        "task history",
        "task attempts",
        "task worker events",
    ] {
        if let Some(index) = normalized.find(prefix) {
            let selector = content[index + prefix.len()..]
                .trim()
                .trim_matches([':', '\u{ff1a}', '?', '\u{ff1f}', '"', '\'']);
            if !selector.is_empty() {
                return Some(selector.to_owned());
            }
        }
    }

    let selector = extract_task_selector(
        content,
        &[
            "show",
            "list",
            "history",
            "execution",
            "attempts",
            "attempt",
            "worker",
            "events",
            "event",
            "run",
            "evidence",
            "audit",
            "trail",
            ZH_LIST,
            "\u{67e5}\u{770b}",
            "\u{5386}\u{53f2}",
            "\u{6267}\u{884c}",
            "\u{8bb0}\u{5f55}",
            "\u{5c1d}\u{8bd5}",
        ],
    );
    (!selector.is_empty() && selector != content).then_some(selector)
}

fn extract_task_follow_up_selector(content: &str, normalized: &str) -> Option<String> {
    if !contains_any(
        normalized,
        &[
            "follow-up",
            "follow up",
            "followups",
            "follow-up task",
            "follow up task",
            "follow-up tasks",
            "follow up tasks",
            "\u{540e}\u{7eed}\u{4efb}\u{52a1}",
        ],
    ) || !contains_any(normalized, &["task", ZH_TASK])
        || is_create_request(normalized)
    {
        return None;
    }

    for prefix in [
        "show follow-up tasks for task",
        "list follow-up tasks for task",
        "show follow up tasks for task",
        "list follow up tasks for task",
        "show follow-ups for task",
        "list follow-ups for task",
        "task follow-up tasks",
        "task follow up tasks",
        "task follow-ups",
    ] {
        if let Some(index) = normalized.find(prefix) {
            let selector = content[index + prefix.len()..]
                .trim()
                .trim_matches([':', '\u{ff1a}', '?', '\u{ff1f}', '"', '\'']);
            if !selector.is_empty() {
                return Some(selector.to_owned());
            }
        }
    }

    let selector = extract_task_selector(
        content,
        &[
            "show",
            "list",
            "follow-up",
            "follow",
            "up",
            "followups",
            "tasks",
            "task",
            "\u{67e5}\u{770b}",
            ZH_LIST,
            "\u{540e}\u{7eed}\u{4efb}\u{52a1}",
        ],
    );
    (!selector.is_empty() && selector != content).then_some(selector)
}

fn extract_task_notes_selector(content: &str, normalized: &str) -> Option<String> {
    if !contains_any(normalized, &["note", "notes", ZH_NOTE])
        || !contains_any(normalized, &["task", ZH_TASK])
        || is_create_request(normalized)
        || contains_any(
            normalized,
            &["add note", "note to task", "note task", ZH_ADD],
        )
        || !contains_any(
            normalized,
            &[
                "show",
                "list",
                "view",
                "what",
                "which",
                "\u{67e5}\u{770b}",
                ZH_LIST,
            ],
        )
    {
        return None;
    }

    let selector = extract_task_notes_selector_text(content);
    (!selector.is_empty() && selector != content).then_some(selector)
}

fn extract_task_notes_selector_text(content: &str) -> String {
    let mut selector = content.to_owned();
    for word in [
        "show notes for task",
        "list notes for task",
        "view notes for task",
        "show task notes",
        "list task notes",
        "view task notes",
        "task notes",
        "notes",
        "note",
        "show",
        "list",
        "view",
        "what",
        "which",
        "task",
        ZH_TASK,
        "\u{67e5}\u{770b}",
        ZH_LIST,
        ZH_NOTE,
    ] {
        selector = replace_case_insensitive(&selector, word, "");
    }

    selector
        .trim()
        .trim_matches([':', '\u{ff1a}', '?', '\u{ff1f}', '"', '\''])
        .trim()
        .to_owned()
}

fn extract_task_constraints_selector(content: &str, normalized: &str) -> Option<String> {
    if !contains_any(
        normalized,
        &[
            "constraint",
            "constraints",
            "dependency",
            "dependencies",
            "resource lock",
            "resource locks",
            "\u{7ea6}\u{675f}",
            "\u{4f9d}\u{8d56}",
            ZH_RESOURCE_LOCK,
        ],
    ) || !contains_any(normalized, &["task", ZH_TASK])
        || is_create_request(normalized)
        || contains_any(
            normalized,
            &[
                "add dependency",
                "set dependency",
                "remove dependency",
                "delete dependency",
                "clear dependency",
                "add resource lock",
                "remove resource lock",
                "delete resource lock",
                "clear resource lock",
                ZH_ADD,
                "\u{79fb}\u{9664}",
                "\u{5220}\u{9664}",
            ],
        )
        || !contains_any(
            normalized,
            &[
                "show",
                "list",
                "view",
                "what",
                "which",
                "\u{67e5}\u{770b}",
                ZH_LIST,
            ],
        )
    {
        return None;
    }

    let selector = extract_task_constraints_selector_text(content);
    (!selector.is_empty() && selector != content).then_some(selector)
}

fn extract_task_constraints_selector_text(content: &str) -> String {
    let mut selector = content.to_owned();
    for word in [
        "show constraints for task",
        "list constraints for task",
        "view constraints for task",
        "show dependencies for task",
        "list dependencies for task",
        "show resource locks for task",
        "list resource locks for task",
        "show task constraints",
        "list task constraints",
        "task constraints",
        "resource locks",
        "resource lock",
        "dependencies",
        "dependency",
        "constraints",
        "constraint",
        "show",
        "list",
        "view",
        "what",
        "which",
        "task",
        ZH_TASK,
        "\u{67e5}\u{770b}",
        ZH_LIST,
        "\u{7ea6}\u{675f}",
        "\u{4f9d}\u{8d56}",
        ZH_RESOURCE_LOCK,
    ] {
        selector = replace_case_insensitive(&selector, word, "");
    }

    selector
        .trim()
        .trim_matches([':', '\u{ff1a}', '?', '\u{ff1f}', '"', '\''])
        .trim()
        .to_owned()
}

fn extract_task_memories_selector(content: &str, normalized: &str) -> Option<String> {
    if !contains_any(
        normalized,
        &[
            "memory",
            "memories",
            "memory candidates",
            "\u{8bb0}\u{5fc6}",
            "\u{8bb0}\u{5fc6}\u{5019}\u{9009}",
        ],
    ) || !contains_any(normalized, &["task", ZH_TASK])
        || is_create_request(normalized)
        || contains_any(
            normalized,
            &[
                "remember",
                "approve",
                "reject",
                "delete",
                "update",
                "edit",
                ZH_APPROVE,
                ZH_REJECT,
                "\u{5220}\u{9664}",
                "\u{66f4}\u{65b0}",
            ],
        )
        || !contains_any(
            normalized,
            &[
                "show",
                "list",
                "view",
                "what",
                "which",
                "\u{67e5}\u{770b}",
                ZH_LIST,
            ],
        )
    {
        return None;
    }

    let selector = extract_task_memories_selector_text(content);
    (!selector.is_empty() && selector != content).then_some(selector)
}

fn extract_task_memories_selector_text(content: &str) -> String {
    let mut selector = content.to_owned();
    for word in [
        "show memory candidates for task",
        "list memory candidates for task",
        "view memory candidates for task",
        "show memories for task",
        "list memories for task",
        "view memories for task",
        "show task memory candidates",
        "list task memory candidates",
        "show task memories",
        "list task memories",
        "task memory candidates",
        "task memories",
        "memory candidates",
        "memory candidate",
        "memories",
        "memory",
        "show",
        "list",
        "view",
        "what",
        "which",
        "task",
        ZH_TASK,
        "\u{67e5}\u{770b}",
        ZH_LIST,
        "\u{8bb0}\u{5fc6}\u{5019}\u{9009}",
        "\u{8bb0}\u{5fc6}",
    ] {
        selector = replace_case_insensitive(&selector, word, "");
    }

    selector
        .trim()
        .trim_matches([':', '\u{ff1a}', '?', '\u{ff1f}', '"', '\''])
        .trim()
        .to_owned()
}

fn parse_task_memory_review_intent(content: &str, normalized: &str) -> Option<MainAgentIntent> {
    if !contains_any(normalized, &["task", ZH_TASK])
        || !contains_any(
            normalized,
            &[
                "memory",
                "memories",
                "memory candidate",
                "memory candidates",
                ZH_MEMORY,
                ZH_LONG_TERM_MEMORY,
            ],
        )
        || !is_bulk_memory_review_request(normalized)
    {
        return None;
    }

    let status = if contains_any(
        normalized,
        &[
            "approve",
            "accept",
            "approve memory",
            "accept memory",
            ZH_APPROVE,
            ZH_ACCEPT,
        ],
    ) {
        MemoryStatus::Approved
    } else if contains_any(
        normalized,
        &[
            "reject",
            "discard",
            "reject memory",
            "discard memory",
            ZH_REJECT,
        ],
    ) {
        MemoryStatus::Rejected
    } else {
        return None;
    };

    let selector = extract_task_memory_review_selector_text(content);
    (!selector.is_empty() && selector != content)
        .then_some(MainAgentIntent::BulkReviewTaskMemories { selector, status })
}

fn extract_task_memory_review_selector_text(content: &str) -> String {
    let mut selector = content.to_owned();
    for word in [
        "approve all pending memory candidates for task",
        "approve all memory candidates for task",
        "approve pending memory candidates for task",
        "approve all pending memories for task",
        "approve all memories for task",
        "approve pending memories for task",
        "accept all pending memory candidates for task",
        "accept all memory candidates for task",
        "accept pending memory candidates for task",
        "accept all pending memories for task",
        "accept all memories for task",
        "accept pending memories for task",
        "reject all pending memory candidates for task",
        "reject all memory candidates for task",
        "reject pending memory candidates for task",
        "reject all pending memories for task",
        "reject all memories for task",
        "reject pending memories for task",
        "discard all pending memory candidates for task",
        "discard all memory candidates for task",
        "discard pending memory candidates for task",
        "discard all pending memories for task",
        "discard all memories for task",
        "discard pending memories for task",
        "approve task memory candidates",
        "accept task memory candidates",
        "reject task memory candidates",
        "discard task memory candidates",
        "approve task memories",
        "accept task memories",
        "reject task memories",
        "discard task memories",
        "memory candidates",
        "memory candidate",
        "memories",
        "memory",
        "approve",
        "accept",
        "reject",
        "discard",
        "all pending",
        "pending",
        "all",
        "every",
        "task",
        ZH_TASK,
        ZH_APPROVE,
        ZH_ACCEPT,
        ZH_REJECT,
        "\u{5168}\u{90e8}",
        "\u{6240}\u{6709}",
        "\u{5f85}\u{5ba1}\u{6838}",
        "\u{8bb0}\u{5fc6}\u{5019}\u{9009}",
        ZH_LONG_TERM_MEMORY,
        ZH_MEMORY,
    ] {
        selector = replace_case_insensitive(&selector, word, "");
    }

    selector
        .trim()
        .trim_matches([':', '\u{ff1a}', '?', '\u{ff1f}', '"', '\''])
        .trim()
        .to_owned()
}

fn extract_task_conversation_selector(content: &str, normalized: &str) -> Option<String> {
    if !contains_any(
        normalized,
        &[
            "conversation",
            "messages",
            "message thread",
            "chat",
            "\u{4f1a}\u{8bdd}",
            "\u{5bf9}\u{8bdd}",
            "\u{6d88}\u{606f}",
        ],
    ) || !contains_any(normalized, &["task", ZH_TASK])
        || is_create_request(normalized)
    {
        return None;
    }

    for prefix in [
        "show conversation for task",
        "list conversation for task",
        "show messages for task",
        "list messages for task",
        "show task conversation",
        "list task conversation",
        "task conversation",
        "task messages",
    ] {
        if let Some(index) = normalized.find(prefix) {
            let selector = content[index + prefix.len()..]
                .trim()
                .trim_matches([':', '\u{ff1a}', '?', '\u{ff1f}', '"', '\'']);
            if !selector.is_empty() {
                return Some(selector.to_owned());
            }
        }
    }

    let selector = extract_task_selector(
        content,
        &[
            "show",
            "list",
            "conversation",
            "messages",
            "message",
            "thread",
            "chat",
            ZH_LIST,
            "\u{67e5}\u{770b}",
            "\u{4f1a}\u{8bdd}",
            "\u{5bf9}\u{8bdd}",
            "\u{6d88}\u{606f}",
        ],
    );
    (!selector.is_empty() && selector != content).then_some(selector)
}

fn is_scheduler_scan_request(normalized: &str) -> bool {
    contains_any(
        normalized,
        &[
            "run scheduler",
            "scheduler tick",
            "scheduler scan",
            "scan task pool",
            "scan tasks",
            "check task pool now",
            "run task pool",
            ZH_SCAN,
            ZH_RUN,
            ZH_SCHEDULER,
        ],
    ) && contains_any(
        normalized,
        &[
            "scheduler",
            "task pool",
            "tasks",
            ZH_SCHEDULER,
            ZH_TASK_POOL,
            ZH_TASK,
        ],
    ) && !is_create_request(normalized)
}

fn is_run_next_task_request(normalized: &str) -> bool {
    if is_create_request(normalized) {
        return false;
    }

    contains_any(
        normalized,
        &[
            "run next task",
            "run the next task",
            "execute next task",
            "execute the next task",
            "start next task",
            "start the next task",
            "run next runnable task",
            "execute next runnable task",
            "continue with next task",
            "continue work",
            "keep working",
            "\u{7ee7}\u{7eed}\u{6267}\u{884c}",
            "\u{7ee7}\u{7eed}\u{4efb}\u{52a1}",
            "\u{6267}\u{884c}\u{4e0b}\u{4e00}\u{4e2a}\u{4efb}\u{52a1}",
            "\u{8fd0}\u{884c}\u{4e0b}\u{4e00}\u{4e2a}\u{4efb}\u{52a1}",
        ],
    ) || (contains_any(
        normalized,
        &["next task", "\u{4e0b}\u{4e00}\u{4e2a}\u{4efb}\u{52a1}"],
    ) && contains_any(
        normalized,
        &["run", "execute", "start", ZH_RUN, "\u{6267}\u{884c}"],
    ))
}

fn is_next_action_recommendation_request(normalized: &str) -> bool {
    contains_any(
        normalized,
        &[
            "what should i do next",
            "what should we do next",
            "what is next",
            "next action",
            "recommended next action",
            "recommend next action",
            "suggest next action",
            "what should the agent do next",
            "how should we proceed",
            "\u{63a5}\u{4e0b}\u{6765}\u{505a}\u{4ec0}\u{4e48}",
            "\u{4e0b}\u{4e00}\u{6b65}\u{505a}\u{4ec0}\u{4e48}",
            "\u{63a8}\u{8350}\u{4e0b}\u{4e00}\u{6b65}",
        ],
    ) && !is_create_request(normalized)
}

fn is_scheduler_state_request(normalized: &str) -> bool {
    contains_any(
        normalized,
        &[
            "scheduler state",
            "scheduler status",
            "execution state",
            "execution status",
            "agent state",
            "agent status",
            "what is running",
            "what is the agent doing",
            "next runnable",
            "next task",
            "\u{8c03}\u{5ea6}\u{72b6}\u{6001}",
            "\u{6267}\u{884c}\u{72b6}\u{6001}",
            "\u{4e0b}\u{4e00}\u{4e2a}\u{4efb}\u{52a1}",
        ],
    ) && !is_create_request(normalized)
}

fn is_workspace_inspection_request(normalized: &str) -> bool {
    contains_any(
        normalized,
        &[
            "workspace status",
            "project status",
            "repo status",
            "repository status",
            "git status",
            "inspect workspace",
            "inspect project",
            "\u{9879}\u{76ee}\u{72b6}\u{6001}",
            "\u{4ed3}\u{5e93}\u{72b6}\u{6001}",
            "\u{67e5}\u{770b}\u{9879}\u{76ee}",
            "\u{68c0}\u{67e5}\u{9879}\u{76ee}",
        ],
    ) && !is_create_request(normalized)
}

fn extract_workspace_file_inspection_path(content: &str, normalized: &str) -> Option<String> {
    for marker in [
        "read file ",
        "show file ",
        "inspect file ",
        "open file ",
        "cat ",
        "\u{8bfb}\u{53d6}\u{6587}\u{4ef6}",
        "\u{67e5}\u{770b}\u{6587}\u{4ef6}",
        "\u{68c0}\u{67e5}\u{6587}\u{4ef6}",
    ] {
        if let Some(index) = normalized.find(marker) {
            let path = clean_workspace_path(&content[index + marker.len()..]);
            if !path.is_empty() {
                return Some(path);
            }
        }
    }

    None
}

fn extract_workspace_directory_inspection_path(content: &str, normalized: &str) -> Option<String> {
    if is_create_request(normalized) {
        return None;
    }

    if normalized == "ls" {
        return Some(".".to_owned());
    }
    if normalized.starts_with("ls ") {
        let path = clean_workspace_path(&content[3..]);
        return Some(if path.is_empty() {
            ".".to_owned()
        } else {
            path
        });
    }

    for marker in [
        "list directory ",
        "show directory ",
        "inspect directory ",
        "list folder ",
        "show folder ",
        "inspect folder ",
        "list dir ",
        "\u{5217}\u{51fa}\u{76ee}\u{5f55}",
        "\u{67e5}\u{770b}\u{76ee}\u{5f55}",
        "\u{68c0}\u{67e5}\u{76ee}\u{5f55}",
    ] {
        if let Some(index) = normalized.find(marker) {
            let path = clean_workspace_path(&content[index + marker.len()..]);
            return Some(if path.is_empty() {
                ".".to_owned()
            } else {
                path
            });
        }
    }

    if contains_any(
        normalized,
        &[
            "list workspace files",
            "show workspace files",
            "list project files",
            "show project files",
            "list root files",
            "show root files",
            "workspace tree",
            "\u{5217}\u{51fa}\u{9879}\u{76ee}\u{6587}\u{4ef6}",
            "\u{67e5}\u{770b}\u{9879}\u{76ee}\u{6587}\u{4ef6}",
        ],
    ) {
        return Some(".".to_owned());
    }

    None
}

fn clean_workspace_path(value: &str) -> String {
    value
        .trim()
        .trim_start_matches([':', '\u{ff1a}'])
        .trim()
        .trim_matches(['`', '"', '\''])
        .trim()
        .to_owned()
}

fn entry_kind_rank(kind: &str) -> u8 {
    match kind {
        "dir" => 0,
        "file" => 1,
        "symlink" => 2,
        _ => 3,
    }
}

fn format_workspace_directory(
    path: &str,
    entries: &[(String, String, u64)],
    shown: usize,
    truncated: bool,
) -> String {
    if entries.is_empty() {
        return format!("Directory: {path}\nNo entries.");
    }

    let truncation_note = if truncated { " (truncated)" } else { "" };
    let mut lines = vec![format!(
        "Directory: {path}\nEntries shown: {} of {}{}",
        shown,
        entries.len(),
        truncation_note
    )];
    for (kind, name, bytes) in entries.iter().take(shown) {
        let suffix = if kind == "dir" { "/" } else { "" };
        let size = if kind == "file" {
            format!(" {bytes} bytes")
        } else {
            String::new()
        };
        lines.push(format!("- {name}{suffix} [{kind}{size}]"));
    }

    lines.join("\n")
}

fn format_task_list(tasks: &[Task]) -> String {
    format_task_list_with_title("Task pool", tasks)
}

fn format_task_list_by_status(status: TaskStatus, tasks: &[Task]) -> String {
    format_task_list_with_title(&format!("{} tasks", status), tasks)
}

fn format_task_list_with_title(title: &str, tasks: &[Task]) -> String {
    if tasks.is_empty() {
        return format!("{title} is empty.");
    }

    let mut lines = vec![format!("{title} has {} task(s):", tasks.len())];
    for task in tasks.iter().take(10) {
        lines.push(format!(
            "- {} [{} {} priority {} queue {}] {}",
            task.id.to_string().chars().take(8).collect::<String>(),
            task.status,
            task.task_type,
            task.priority,
            task.queue_position,
            task.title
        ));
    }
    if tasks.len() > 10 {
        lines.push(format!("- ... and {} more", tasks.len() - 10));
    }

    lines.join("\n")
}

fn format_waiting_for_user_tasks(tasks: &[(Task, Option<String>)]) -> String {
    if tasks.is_empty() {
        return "No tasks are waiting for user input.".to_owned();
    }

    let mut lines = vec![format!(
        "{} task(s) are waiting for user input:",
        tasks.len()
    )];
    for (task, latest_question) in tasks.iter().take(10) {
        let reason = task.blocked_reason.as_deref().unwrap_or("needs input");
        lines.push(format!("- {} ({})", format_task_brief(task), reason));
        if let Some(question) = latest_question {
            lines.push(format!(
                "  Latest question: {}",
                bounded_preview(question, 240)
            ));
        }
    }
    if tasks.len() > 10 {
        lines.push(format!("- ... and {} more", tasks.len() - 10));
    }
    lines.push("Reply with: reply to task <title>: <your answer>".to_owned());

    lines.join("\n")
}

fn format_waiting_for_schedule_tasks(tasks: &[Task]) -> String {
    if tasks.is_empty() {
        return "No tasks are waiting for schedule.".to_owned();
    }

    let mut lines = vec![format!("{} task(s) are waiting for schedule:", tasks.len())];
    for task in tasks.iter().take(10) {
        let next_run = task
            .next_run_at
            .map(|time| time.to_string())
            .unwrap_or_else(|| "not set".to_owned());
        lines.push(format!(
            "- {} next_run_at {}",
            format_task_brief(task),
            next_run
        ));
    }
    if tasks.len() > 10 {
        lines.push(format!("- ... and {} more", tasks.len() - 10));
    }
    lines.push("Run a due scan with: run scheduler scan".to_owned());

    lines.join("\n")
}

fn format_scheduler_state(
    running_tasks: &[Task],
    next_queued_task: Option<&Task>,
    next_runnable_task: Option<&Task>,
    queued_count: usize,
    waiting_for_user_tasks: &[Task],
    waiting_for_schedule_count: usize,
) -> String {
    let mut lines = vec![format!(
        "Execution state: {} running, {} queued, {} waiting for user, {} waiting for schedule.",
        running_tasks.len(),
        queued_count,
        waiting_for_user_tasks.len(),
        waiting_for_schedule_count
    )];

    if running_tasks.is_empty() {
        lines.push("Running: none.".to_owned());
    } else {
        lines.push("Running:".to_owned());
        for task in running_tasks.iter().take(5) {
            lines.push(format!("- {}", format_task_brief(task)));
        }
    }

    lines.push(match next_queued_task {
        Some(task) => format!("Next queued: {}", format_task_brief(task)),
        None => "Next queued: none.".to_owned(),
    });
    lines.push(match next_runnable_task {
        Some(task) => format!("Next runnable: {}", format_task_brief(task)),
        None => "Next runnable: none.".to_owned(),
    });

    if !waiting_for_user_tasks.is_empty() {
        lines.push("Waiting for user:".to_owned());
        for task in waiting_for_user_tasks.iter().take(5) {
            let reason = task.blocked_reason.as_deref().unwrap_or("needs input");
            lines.push(format!("- {} ({})", format_task_brief(task), reason));
        }
    }

    lines.join("\n")
}

fn format_task_brief(task: &Task) -> String {
    format!(
        "{} [{} priority {} queue {}] {}",
        task.id.to_string().chars().take(8).collect::<String>(),
        task.status,
        task.priority,
        task.queue_position,
        task.title
    )
}

fn format_global_action_list(actions: &[TaskAction]) -> String {
    if actions.is_empty() {
        return "No global main-agent actions yet.".to_owned();
    }

    let mut lines = vec![format!(
        "Recent global main-agent actions ({} shown):",
        actions.len().min(10)
    )];
    for action in actions.iter().take(10) {
        lines.push(format!(
            "- {} [{}] {} {}",
            action.id.to_string().chars().take(8).collect::<String>(),
            action.actor,
            action.action_type,
            action.details
        ));
    }

    lines.join("\n")
}

fn format_task_artifacts(task: &Task, artifacts: &[TaskArtifact]) -> String {
    if artifacts.is_empty() {
        return format!("Task '{}' has no reported artifacts yet.", task.title);
    }

    let mut lines = vec![format!(
        "Task '{}' has {} artifact(s):",
        task.title,
        artifacts.len()
    )];
    for artifact in artifacts.iter().take(10) {
        let summary = artifact
            .summary
            .as_deref()
            .filter(|summary| !summary.trim().is_empty())
            .map(|summary| format!(" - {summary}"))
            .unwrap_or_default();
        lines.push(format!(
            "- {} [{}] {}{}",
            artifact.name, artifact.artifact_type, artifact.uri, summary
        ));
    }
    if artifacts.len() > 10 {
        lines.push(format!("- ... and {} more", artifacts.len() - 10));
    }

    lines.join("\n")
}

fn format_task_history(
    task: &Task,
    attempts: &[TaskAttempt],
    events: &[TaskAttemptEvent],
    actions: &[TaskAction],
) -> String {
    let mut lines = vec![format!(
        "Task '{}' history: {} attempt(s), {} worker event(s), {} action(s).",
        task.title,
        attempts.len(),
        events.len(),
        actions.len()
    )];

    if attempts.is_empty() {
        lines.push("Attempts: none.".to_owned());
    } else {
        lines.push("Recent attempts:".to_owned());
        for attempt in attempts.iter().rev().take(5) {
            lines.push(format!(
                "- {} [{}] {}",
                attempt.id.to_string().chars().take(8).collect::<String>(),
                attempt.status,
                attempt.summary.as_deref().unwrap_or("no summary")
            ));
        }
    }

    if !events.is_empty() {
        lines.push("Recent worker events:".to_owned());
        for event in events.iter().rev().take(8) {
            lines.push(format!(
                "- {} [{}] {}",
                event.id.to_string().chars().take(8).collect::<String>(),
                event.event_type,
                event.message
            ));
        }
    }

    if !actions.is_empty() {
        lines.push("Recent task actions:".to_owned());
        for action in actions.iter().rev().take(5) {
            lines.push(format!(
                "- {} [{}] {}",
                action.id.to_string().chars().take(8).collect::<String>(),
                action.actor,
                action.action_type
            ));
        }
    }

    lines.join("\n")
}

fn format_task_latest_result(task: &Task, attempts: &[TaskAttempt]) -> String {
    let mut lines = vec![format!(
        "Task '{}' latest result: status {}.",
        task.title, task.status
    )];

    match task.result_summary.as_deref() {
        Some(summary) if !summary.trim().is_empty() => {
            lines.push(format!("Result summary: {}", bounded_preview(summary, 500)));
        }
        _ => lines.push("Result summary: none recorded yet.".to_owned()),
    }

    if let Some(attempt) = attempts.iter().max_by_key(|attempt| attempt.started_at) {
        lines.push(format!(
            "Latest attempt: {} [{}] {}",
            attempt.id.to_string().chars().take(8).collect::<String>(),
            attempt.status,
            attempt
                .summary
                .as_deref()
                .filter(|summary| !summary.trim().is_empty())
                .map(|summary| bounded_preview(summary, 300))
                .unwrap_or_else(|| "no attempt summary".to_owned())
        ));
    } else {
        lines.push("Latest attempt: none.".to_owned());
    }

    if matches!(task.status, TaskStatus::WaitingForUser) {
        if let Some(reason) = task.blocked_reason.as_deref() {
            lines.push(format!(
                "Needs user input: {}",
                bounded_preview(reason, 300)
            ));
        }
    }

    lines.join("\n")
}

fn format_task_follow_ups(task: &Task, follow_ups: &[Task]) -> String {
    if follow_ups.is_empty() {
        return format!("Task '{}' has no recorded follow-up tasks.", task.title);
    }

    let mut lines = vec![format!(
        "Task '{}' created {} follow-up task(s):",
        task.title,
        follow_ups.len()
    )];
    for follow_up in follow_ups.iter().take(10) {
        lines.push(format!(
            "- {} [{} {} priority {} queue {}] {}",
            follow_up.id.to_string().chars().take(8).collect::<String>(),
            follow_up.status,
            follow_up.task_type,
            follow_up.priority,
            follow_up.queue_position,
            follow_up.title
        ));
    }
    if follow_ups.len() > 10 {
        lines.push(format!("- ... and {} more", follow_ups.len() - 10));
    }

    lines.join("\n")
}

fn format_task_notes(task: &Task, notes: &[TaskNote]) -> String {
    if notes.is_empty() {
        return format!("Task '{}' has no notes yet.", task.title);
    }

    let mut lines = vec![format!(
        "Task '{}' notes ({} shown):",
        task.title,
        notes.len().min(10)
    )];
    for note in notes.iter().rev().take(10) {
        lines.push(format!(
            "- {} [{}] {}",
            note.id.to_string().chars().take(8).collect::<String>(),
            note.actor,
            bounded_preview(&note.content, 240)
        ));
    }
    if notes.len() > 10 {
        lines.push(format!("- ... and {} more", notes.len() - 10));
    }

    lines.join("\n")
}

fn format_task_constraints(
    task: &Task,
    dependencies: &[Task],
    resource_locks: &[TaskResourceLock],
    resource_lock_conflicts: &[ResourceLockConflict],
) -> String {
    let mut lines = vec![format!("Task '{}' constraints:", task.title)];

    if dependencies.is_empty() {
        lines.push("Dependencies: none.".to_owned());
    } else {
        lines.push(format!("Dependencies ({}):", dependencies.len()));
        for dependency in dependencies.iter().take(10) {
            let state = if matches!(
                dependency.status,
                TaskStatus::Completed | TaskStatus::WaitingForSchedule
            ) {
                "satisfied"
            } else {
                "blocking"
            };
            lines.push(format!(
                "- {} [{} {}] {}",
                dependency
                    .id
                    .to_string()
                    .chars()
                    .take(8)
                    .collect::<String>(),
                dependency.status,
                state,
                dependency.title
            ));
        }
        if dependencies.len() > 10 {
            lines.push(format!("- ... and {} more", dependencies.len() - 10));
        }
    }

    if resource_locks.is_empty() {
        lines.push("Resource locks: none.".to_owned());
    } else {
        lines.push(format!("Resource locks ({}):", resource_locks.len()));
        for lock in resource_locks.iter().take(10) {
            lines.push(format!("- {} [{}]", lock.resource_key, lock.lock_mode));
        }
        if resource_locks.len() > 10 {
            lines.push(format!("- ... and {} more", resource_locks.len() - 10));
        }
    }

    if resource_lock_conflicts.is_empty() {
        lines.push("Active resource lock conflicts: none.".to_owned());
    } else {
        lines.push(format!(
            "Active resource lock conflicts ({}):",
            resource_lock_conflicts.len()
        ));
        for conflict in resource_lock_conflicts.iter().take(10) {
            lines.push(format!(
                "- {} held by running task '{}'",
                conflict.resource_key, conflict.running_task.title
            ));
        }
        if resource_lock_conflicts.len() > 10 {
            lines.push(format!(
                "- ... and {} more",
                resource_lock_conflicts.len() - 10
            ));
        }
    }

    lines.join("\n")
}

fn format_task_memories(task: &Task, memories: &[Memory]) -> String {
    if memories.is_empty() {
        return format!("Task '{}' has no memory candidates yet.", task.title);
    }

    let mut lines = vec![format!(
        "Task '{}' memory candidates ({} shown):",
        task.title,
        memories.len().min(10)
    )];
    for memory in memories.iter().take(10) {
        lines.push(format!(
            "- {} [{} {} confidence {:.2}] {}",
            memory.id.to_string().chars().take(8).collect::<String>(),
            memory.status,
            memory.scope,
            memory.confidence,
            memory.content
        ));
    }
    if memories.len() > 10 {
        lines.push(format!("- ... and {} more", memories.len() - 10));
    }

    lines.join("\n")
}

fn format_task_conversation(task: &Task, messages: &[ConversationMessage]) -> String {
    if messages.is_empty() {
        return format!("Task '{}' has no conversation messages yet.", task.title);
    }

    let mut lines = vec![format!(
        "Task '{}' conversation ({} shown):",
        task.title,
        messages.len().min(20)
    )];
    for message in messages.iter().take(20) {
        lines.push(format!(
            "- {}: {}",
            message.role,
            bounded_preview(&message.content, 240)
        ));
    }

    lines.join("\n")
}

fn format_memory_list(filter: MemoryListFilter, memories: &[Memory]) -> String {
    if memories.is_empty() {
        return format!("No {} memories found.", filter.as_str());
    }

    let mut lines = vec![format!(
        "{} memories ({} shown):",
        capitalize_ascii(filter.as_str()),
        memories.len().min(10)
    )];
    for memory in memories.iter().take(10) {
        let source = memory
            .source_task_id
            .map(|id| {
                format!(
                    " source {}",
                    id.to_string().chars().take(8).collect::<String>()
                )
            })
            .unwrap_or_default();
        lines.push(format!(
            "- {} [{} {} confidence {:.2}{}] {}",
            memory.id.to_string().chars().take(8).collect::<String>(),
            memory.status,
            memory.scope,
            memory.confidence,
            source,
            memory.content
        ));
    }
    if memories.len() > 10 {
        lines.push(format!("- ... and {} more", memories.len() - 10));
    }

    lines.join("\n")
}

fn capitalize_ascii(value: &str) -> String {
    let mut chars = value.chars();
    match chars.next() {
        Some(first) => format!("{}{}", first.to_ascii_uppercase(), chars.as_str()),
        None => String::new(),
    }
}

fn parse_explain_intent(content: &str, normalized: &str) -> Option<MainAgentIntent> {
    if !contains_any(
        normalized,
        &[
            "explain",
            "why",
            "why is",
            "why isn't",
            "status",
            "state",
            ZH_EXPLAIN,
            ZH_WHY,
            ZH_STATE,
        ],
    ) {
        return None;
    }

    if contains_any(
        normalized,
        &[
            "task pool",
            "queue",
            "current execution",
            "execution state",
            ZH_TASK_POOL,
            ZH_QUEUE,
        ],
    ) {
        return Some(MainAgentIntent::ExplainTaskPool);
    }

    extract_explain_task_selector(content, normalized)
        .map(|selector| MainAgentIntent::ExplainTask { selector })
}

fn extract_explain_task_selector(content: &str, normalized: &str) -> Option<String> {
    for prefix in [
        "why is task",
        "why isn't task",
        "why is the task",
        "explain task",
        "task status",
        "status of task",
        "state of task",
    ] {
        if let Some(index) = normalized.find(prefix) {
            let selector = content[index + prefix.len()..]
                .trim()
                .trim_matches([':', '\u{ff1a}', '?', '\u{ff1f}', '"', '\'']);
            return clean_explain_selector(selector);
        }
    }

    if contains_any(normalized, &[ZH_TASK, ZH_EXPLAIN, ZH_WHY, ZH_STATE]) {
        let selector = extract_task_selector(
            content,
            &[
                "not running",
                "running",
                "status",
                "state",
                "why",
                "explain",
                ZH_EXPLAIN,
                ZH_WHY,
                ZH_STATE,
            ],
        );
        if !selector.is_empty() && selector != content {
            return Some(selector);
        }
    }

    None
}

fn clean_explain_selector(selector: &str) -> Option<String> {
    let mut value = selector.to_owned();
    for suffix in [
        "not running",
        "running",
        "blocked",
        "waiting",
        "?",
        "\u{ff1f}",
    ] {
        if let Some(index) = value.to_lowercase().find(suffix) {
            value.truncate(index);
        }
    }
    let value = value
        .trim()
        .trim_matches([':', '\u{ff1a}', ',', '\u{ff0c}', '.', '\u{3002}', '"', '\''])
        .trim()
        .to_owned();
    (!value.is_empty()).then_some(value)
}

fn format_task_pool_explanation(tasks: &[Task]) -> String {
    if tasks.is_empty() {
        return "No work is running because the task pool is empty.".to_owned();
    }

    let running = tasks
        .iter()
        .filter(|task| task.status == TaskStatus::Running)
        .count();
    let queued = tasks
        .iter()
        .filter(|task| task.status == TaskStatus::Queued)
        .count();
    let waiting_for_user = tasks
        .iter()
        .filter(|task| task.status == TaskStatus::WaitingForUser)
        .count();
    let waiting_for_schedule = tasks
        .iter()
        .filter(|task| task.status == TaskStatus::WaitingForSchedule)
        .count();

    let mut lines = vec![format!(
        "Execution state: {running} running, {queued} queued, {waiting_for_user} waiting for user, {waiting_for_schedule} waiting for schedule."
    )];

    if let Some(task) = tasks.iter().find(|task| task.status == TaskStatus::Running) {
        lines.push(format!("Currently running: '{}'.", task.title));
    } else if let Some(task) = tasks.iter().find(|task| task.status == TaskStatus::Queued) {
        lines.push(format!(
            "No task is running right now. The next queued candidate appears to be '{}' by priority and queue order.",
            task.title
        ));
    } else if waiting_for_user > 0 {
        lines.push(
            "No queued task is runnable because at least one task is waiting for user input."
                .to_owned(),
        );
    } else if waiting_for_schedule > 0 {
        lines.push(
            "No queued task is runnable because recurring work is waiting for its next schedule."
                .to_owned(),
        );
    } else {
        lines.push("No task is currently runnable.".to_owned());
    }

    lines.join("\n")
}

fn format_next_action_recommendation(tasks: &[Task], next_runnable: Option<&Task>) -> String {
    if tasks.is_empty() {
        return "Recommended next action: create the first task. The task pool is empty."
            .to_owned();
    }

    if let Some(task) = tasks.iter().find(|task| task.status == TaskStatus::Running) {
        return format!(
            "Recommended next action: monitor the running task '{}'. Use: show history for task {}",
            task.title, task.title
        );
    }

    if let Some(task) = next_runnable {
        return format!(
            "Recommended next action: run the next runnable task '{}'. Use: run next task",
            task.title
        );
    }

    if let Some(task) = tasks
        .iter()
        .find(|task| task.status == TaskStatus::WaitingForUser)
    {
        let reason = task.blocked_reason.as_deref().unwrap_or("needs input");
        return format!(
            "Recommended next action: answer the blocked task '{}'. Reason: {}. Use: reply to task {}: <your answer>",
            task.title,
            bounded_preview(reason, 180),
            task.title
        );
    }

    if let Some(task) = tasks
        .iter()
        .filter(|task| task.status == TaskStatus::WaitingForSchedule)
        .min_by_key(|task| task.next_run_at)
    {
        let next_run = task
            .next_run_at
            .map(|time| time.to_string())
            .unwrap_or_else(|| "not set".to_owned());
        return format!(
            "Recommended next action: wait for recurring task '{}' to become due at {}. Use: show scheduler status",
            task.title, next_run
        );
    }

    if tasks.iter().any(|task| task.status == TaskStatus::Failed) {
        return "Recommended next action: inspect failed work. Use: show failed tasks".to_owned();
    }

    "Recommended next action: review the task pool. Use: list tasks".to_owned()
}

fn format_task_explanation(
    task: &Task,
    dependencies: &[Task],
    resource_lock_conflicts: &[ResourceLockConflict],
) -> String {
    let mut lines = vec![format!(
        "Task '{}' is currently {}.",
        task.title, task.status
    )];

    match task.status {
        TaskStatus::Queued => {
            let unsatisfied = dependencies
                .iter()
                .filter(|dependency| {
                    !matches!(
                        dependency.status,
                        TaskStatus::Completed | TaskStatus::WaitingForSchedule
                    )
                })
                .collect::<Vec<_>>();
            if unsatisfied.is_empty() && resource_lock_conflicts.is_empty() {
                lines.push("It is queued and eligible for the scheduler once it reaches the front of priority and queue order.".to_owned());
            } else {
                if !unsatisfied.is_empty() {
                    lines.push(format!(
                        "It is queued but blocked by {} unfinished dependenc{}:",
                        unsatisfied.len(),
                        if unsatisfied.len() == 1 { "y" } else { "ies" }
                    ));
                    for dependency in unsatisfied {
                        lines.push(format!("- '{}' is {}", dependency.title, dependency.status));
                    }
                }
                if !resource_lock_conflicts.is_empty() {
                    lines.push(format!(
                        "It is also blocked by {} active resource lock conflict{}:",
                        resource_lock_conflicts.len(),
                        if resource_lock_conflicts.len() == 1 {
                            ""
                        } else {
                            "s"
                        }
                    ));
                    for conflict in resource_lock_conflicts {
                        lines.push(format!(
                            "- '{}' is held by running task '{}'.",
                            conflict.resource_key, conflict.running_task.title
                        ));
                    }
                }
            }
        }
        TaskStatus::Running => {
            lines.push("A worker has claimed it and is executing it now.".to_owned())
        }
        TaskStatus::WaitingForUser => {
            lines.push(
                task.blocked_reason
                    .as_ref()
                    .map(|reason| format!("It needs user input: {reason}"))
                    .unwrap_or_else(|| "It needs user input before it can continue.".to_owned()),
            );
        }
        TaskStatus::WaitingForSchedule => {
            lines.push(
                task.next_run_at
                    .map(|time| format!("It is a recurring task waiting until {time}."))
                    .unwrap_or_else(|| {
                        "It is a recurring task waiting for its next schedule.".to_owned()
                    }),
            );
        }
        TaskStatus::Completed => lines.push("It has completed successfully.".to_owned()),
        TaskStatus::Failed => lines.push(
            task.result_summary
                .as_ref()
                .map(|summary| format!("It failed with summary: {summary}"))
                .unwrap_or_else(|| "It failed without a detailed summary.".to_owned()),
        ),
        TaskStatus::Cancelled => {
            lines.push("It was cancelled and will not be scheduled.".to_owned())
        }
        TaskStatus::Paused => {
            lines.push("It is paused and will not be scheduled until resumed.".to_owned())
        }
        TaskStatus::Draft => {
            lines.push("It is a draft and is not ready for scheduling.".to_owned())
        }
    }

    if !dependencies.is_empty() {
        lines.push(format!("Dependencies: {}.", dependencies.len()));
    }
    if !resource_lock_conflicts.is_empty() {
        lines.push(format!(
            "Resource lock conflicts: {}.",
            resource_lock_conflicts.len()
        ));
    }
    lines.push(format!(
        "Requested skills: {}.",
        format_skill_list(&task.requested_skills)
    ));
    lines.push(format!(
        "Matched skills: {}.",
        format_skill_list(&task.matched_skills)
    ));
    lines.push(format!(
        "Active skills for the next worker run: {}.",
        format_skill_list(&active_skill_names_for_task(task))
    ));

    lines.join("\n")
}

fn parse_dependency_intent(content: &str, normalized: &str) -> Option<MainAgentIntent> {
    if contains_any(
        normalized,
        &[
            "remove dependency",
            "delete dependency",
            "clear dependency",
            ZH_REMOVE_DEPENDENCY,
        ],
    ) || (normalized.contains(ZH_CANCEL) && normalized.contains(ZH_DEPEND_ON))
    {
        return split_dependency_selectors(
            content,
            normalized,
            &["depends on", "depend on", " on ", ZH_DEPEND_ON],
        )
        .map(
            |(selector, depends_on_selector)| MainAgentIntent::RemoveTaskDependency {
                selector,
                depends_on_selector,
            },
        );
    }

    if contains_any(
        normalized,
        &[
            "depends on",
            "depend on",
            "add dependency",
            "set dependency",
            ZH_DEPEND_ON,
        ],
    ) && !is_create_request(normalized)
    {
        return split_dependency_selectors(
            content,
            normalized,
            &["depends on", "depend on", " on ", ZH_DEPEND_ON],
        )
        .map(
            |(selector, depends_on_selector)| MainAgentIntent::AddTaskDependency {
                selector,
                depends_on_selector,
            },
        );
    }

    None
}

fn parse_note_intent(content: &str, normalized: &str) -> Option<MainAgentIntent> {
    if !contains_any(
        normalized,
        &[
            "add note",
            "note task",
            "note to task",
            "add note to task",
            ZH_NOTE,
        ],
    ) {
        return None;
    }

    split_note_selector_and_content(content)
        .map(|(selector, content)| MainAgentIntent::AddTaskNote { selector, content })
}

fn parse_requested_skill_intent(content: &str, normalized: &str) -> Option<MainAgentIntent> {
    if !contains_any(
        normalized,
        &[
            "requested skill",
            "requested skills",
            "skill",
            "skills",
            ZH_SKILL,
        ],
    ) || contains_any(normalized, &["skill management", "skills page"])
    {
        return None;
    }

    let is_remove = contains_any(
        normalized,
        &[
            "remove skill",
            "remove requested skill",
            "delete skill",
            "clear skill",
            "detach skill",
            ZH_CANCEL,
            ZH_REJECT,
        ],
    );

    let (selector, skill_names) = split_skill_selector_and_names(content, normalized, is_remove)?;
    if skill_names.is_empty() {
        return None;
    }

    if is_remove {
        Some(MainAgentIntent::RemoveRequestedSkills {
            selector,
            skill_names,
        })
    } else {
        Some(MainAgentIntent::AddRequestedSkills {
            selector,
            skill_names,
        })
    }
}

fn parse_resource_lock_intent(content: &str, normalized: &str) -> Option<MainAgentIntent> {
    if !contains_any(
        normalized,
        &[
            "resource lock",
            "resource-lock",
            "lock resource",
            ZH_RESOURCE_LOCK,
        ],
    ) {
        return None;
    }

    if contains_any(
        normalized,
        &[
            "remove resource lock",
            "delete resource lock",
            "clear resource lock",
            "unlock resource",
            ZH_CANCEL,
        ],
    ) {
        return split_resource_lock_selector_and_key(content).map(|(selector, resource_key)| {
            MainAgentIntent::RemoveResourceLock {
                selector,
                resource_key,
            }
        });
    }

    split_resource_lock_selector_and_key(content).map(|(selector, resource_key)| {
        MainAgentIntent::AddResourceLock {
            selector,
            resource_key,
        }
    })
}

fn parse_clarification_intent(content: &str, normalized: &str) -> Option<MainAgentIntent> {
    if !contains_any(
        normalized,
        &[
            "ask clarification",
            "request clarification",
            "ask user",
            "need clarification",
            ZH_CLARIFY,
        ],
    ) {
        return None;
    }

    split_clarification_selector_and_question(content)
        .map(|(selector, question)| MainAgentIntent::RequestClarification { selector, question })
}

fn parse_reply_to_task_intent(content: &str, normalized: &str) -> Option<MainAgentIntent> {
    if !contains_any(
        normalized,
        &[
            "reply to task",
            "reply to blocked task",
            "reply to waiting task",
            "reply for task",
            "reply for blocked task",
            "reply for waiting task",
            "answer task",
            "answer for task",
            "answer blocked task",
            "answer waiting task",
            "respond to task",
            "respond for task",
            "respond to blocked task",
            "respond to waiting task",
            "task reply",
        ],
    ) {
        return None;
    }

    split_reply_selector_and_content(content)
        .map(|(selector, content)| MainAgentIntent::ReplyToTask { selector, content })
}

fn parse_task_finish_intent(content: &str, normalized: &str) -> Option<MainAgentIntent> {
    if is_create_request(normalized) || !contains_any(normalized, &["task", ZH_TASK]) {
        return None;
    }

    let is_failure = contains_any(
        normalized,
        &[
            "fail task",
            "failed task",
            "mark task",
            "mark the task",
            "task failed",
            "task as failed",
        ],
    ) && contains_any(normalized, &["fail", "failed"]);
    if is_failure {
        let (selector, error) = split_task_finish_selector_and_summary(content, true)?;
        return Some(MainAgentIntent::FailTask { selector, error });
    }

    let is_completion = contains_any(
        normalized,
        &[
            "complete task",
            "completed task",
            "finish task",
            "finished task",
            "mark task",
            "mark the task",
            "task complete",
            "task completed",
            "task as complete",
            "task as completed",
        ],
    ) && contains_any(
        normalized,
        &["complete", "completed", "finish", "finished", "done"],
    );
    if is_completion {
        let (selector, summary) = split_task_finish_selector_and_summary(content, false)?;
        return Some(MainAgentIntent::CompleteTask { selector, summary });
    }

    None
}

fn parse_task_retry_intent(content: &str, normalized: &str) -> Option<MainAgentIntent> {
    if is_create_request(normalized)
        || is_convert_request(normalized)
        || !contains_any(normalized, &["task", ZH_TASK])
        || !contains_any(
            normalized,
            &[
                "retry task",
                "retry failed task",
                "requeue task",
                "requeue failed task",
                "try task again",
                "run task again",
            ],
        )
    {
        return None;
    }

    let (selector, reason) = split_task_retry_selector_and_reason(content)?;
    Some(MainAgentIntent::RetryTask { selector, reason })
}

fn parse_run_task_now_intent(content: &str, normalized: &str) -> Option<MainAgentIntent> {
    if is_create_request(normalized)
        || !contains_any(normalized, &["task", ZH_TASK])
        || contains_any(normalized, &["task pool", ZH_TASK_POOL, "scheduler"])
        || !contains_any(
            normalized,
            &[
                "run task",
                "run the task",
                "execute task",
                "execute the task",
                "start task",
                "start the task",
                "run now",
                "execute now",
                "start now",
            ],
        )
    {
        return None;
    }

    let selector = extract_task_selector(
        content,
        &[
            "now",
            "immediately",
            "please",
            "run",
            "execute",
            "start",
            "task",
        ],
    );
    (!selector.is_empty() && selector != content)
        .then_some(MainAgentIntent::RunTaskNow { selector })
}

fn parse_task_detail_update_intent(content: &str, normalized: &str) -> Option<MainAgentIntent> {
    if is_create_request(normalized) {
        return None;
    }

    if let Some((selector, title)) =
        split_selector_and_value_after_phrase(content, normalized, "rename task", " to ")
    {
        return Some(MainAgentIntent::UpdateTaskDetails {
            selector,
            title: Some(title),
            description: None,
        });
    }

    if let Some((selector, title)) = split_selector_and_value_after_marker(
        content,
        normalized,
        &[
            "set task title",
            "update task title",
            "change task title",
            "rename task",
        ],
        &[" to ", ":", "\u{ff1a}"],
    ) {
        return Some(MainAgentIntent::UpdateTaskDetails {
            selector,
            title: Some(title),
            description: None,
        });
    }

    if let Some((selector, description)) = split_selector_and_value_after_marker(
        content,
        normalized,
        &[
            "set task description",
            "update task description",
            "change task description",
            "\u{66f4}\u{65b0}\u{4efb}\u{52a1}\u{63cf}\u{8ff0}",
            "\u{4fee}\u{6539}\u{4efb}\u{52a1}\u{63cf}\u{8ff0}",
        ],
        &[" to ", ":", "\u{ff1a}"],
    ) {
        return Some(MainAgentIntent::UpdateTaskDetails {
            selector,
            title: None,
            description: Some(description),
        });
    }

    if !contains_any(
        normalized,
        &[
            "update task",
            "change task",
            "edit task",
            "\u{66f4}\u{65b0}\u{4efb}\u{52a1}",
            "\u{4fee}\u{6539}\u{4efb}\u{52a1}",
        ],
    ) || contains_any(
        normalized,
        &[
            "priority",
            "queue",
            "note",
            "skill",
            "resource lock",
            "dependency",
            ZH_PRIORITY,
            ZH_SKILL,
        ],
    ) {
        return None;
    }

    let selector = extract_task_selector(
        content,
        &[
            "update",
            "change",
            "edit",
            "task",
            "title",
            "description",
            "desc",
            "\u{66f4}\u{65b0}",
            "\u{4fee}\u{6539}",
            ZH_TASK,
            "\u{6807}\u{9898}",
            "\u{63cf}\u{8ff0}",
            "\u{5185}\u{5bb9}",
        ],
    );
    let title = extract_labeled_value(content, normalized, &["title", "\u{6807}\u{9898}"]);
    let description = extract_labeled_value(
        content,
        normalized,
        &[
            "description",
            "desc",
            "\u{63cf}\u{8ff0}",
            "\u{5185}\u{5bb9}",
        ],
    );

    if selector.is_empty() || selector == content || (title.is_none() && description.is_none()) {
        return None;
    }

    Some(MainAgentIntent::UpdateTaskDetails {
        selector,
        title,
        description,
    })
}

fn parse_task_schedule_update_intent(content: &str, normalized: &str) -> Option<MainAgentIntent> {
    if is_create_request(normalized)
        || is_convert_request(normalized)
        || !contains_any(normalized, &["task", ZH_TASK])
        || !contains_any(
            normalized,
            &[
                "schedule",
                "interval",
                "cadence",
                "every",
                ZH_INTERVAL,
                ZH_EVERY,
                "\u{8c03}\u{5ea6}",
            ],
        )
    {
        return None;
    }

    let interval_seconds = extract_interval_seconds(normalized)?;
    if interval_seconds <= 0 {
        return None;
    }
    let selector = extract_task_selector(
        content,
        &[
            "schedule",
            "interval",
            "cadence",
            "every",
            "to",
            "seconds",
            "second",
            "minutes",
            "minute",
            "hours",
            "hour",
            "days",
            "day",
            ZH_INTERVAL,
            ZH_EVERY,
            ZH_TO,
        ],
    );

    (!selector.is_empty()).then_some(MainAgentIntent::UpdateTaskSchedule {
        selector,
        interval_seconds,
    })
}

fn parse_create_memory_intent(content: &str, normalized: &str) -> Option<MainAgentIntent> {
    if !contains_any(
        normalized,
        &[
            "remember",
            "record memory",
            "save memory",
            "store memory",
            "add memory",
            "\u{8bb0}\u{4f4f}",
            "\u{8bb0}\u{5fc6}",
        ],
    ) || contains_any(
        normalized,
        &[
            "list memory",
            "show memory",
            "approve memory",
            "reject memory",
            "update memory",
            "delete memory",
            "remembered",
            "remembering",
            "memory candidate",
            "memory candidates",
            ZH_LIST,
            ZH_APPROVE,
            ZH_REJECT,
        ],
    ) {
        return None;
    }

    let mut content_part = content;
    for marker in [
        "remember",
        "record memory",
        "save memory",
        "store memory",
        "add memory",
        "\u{8bb0}\u{4f4f}",
        "\u{8bb0}\u{5fc6}",
    ] {
        if let Some(index) = normalized.find(marker) {
            content_part = &content[index + marker.len()..];
            break;
        }
    }

    let content_part = content_part
        .trim()
        .trim_start_matches([':', '\u{ff1a}', '-', '\u{2014}'])
        .trim();
    let (scope, memory_content) = split_memory_scope_and_content(content_part);
    if memory_content.is_empty() {
        return None;
    }

    Some(MainAgentIntent::CreateMemory {
        scope,
        content: memory_content,
    })
}

fn parse_memory_review_intent(content: &str, normalized: &str) -> Option<MainAgentIntent> {
    if !contains_any(
        normalized,
        &[
            "memory",
            "memories",
            "memory candidate",
            "memory candidates",
            "long-term memory",
            "long-term memories",
            ZH_MEMORY,
            ZH_LONG_TERM_MEMORY,
        ],
    ) {
        return None;
    }

    if is_bulk_memory_review_request(normalized)
        && contains_any(
            normalized,
            &[
                "approve",
                "accept",
                "approve memory",
                "accept memory",
                ZH_APPROVE,
                ZH_ACCEPT,
            ],
        )
    {
        return Some(MainAgentIntent::BulkReviewMemories {
            status: MemoryStatus::Approved,
        });
    }

    if is_bulk_memory_review_request(normalized)
        && contains_any(
            normalized,
            &[
                "reject",
                "discard",
                "reject memory",
                "discard memory",
                ZH_REJECT,
            ],
        )
    {
        return Some(MainAgentIntent::BulkReviewMemories {
            status: MemoryStatus::Rejected,
        });
    }

    if contains_any(
        normalized,
        &[
            "list memory",
            "list memories",
            "show memory",
            "show memories",
            "memory list",
            "memory candidates",
            "pending memories",
            "approved memories",
            "rejected memories",
            ZH_LIST,
            "\u{67e5}\u{770b}",
        ],
    ) {
        let filter = if contains_any(normalized, &["approved", ZH_APPROVE, ZH_ACCEPT]) {
            MemoryListFilter::Approved
        } else if contains_any(normalized, &["rejected", "discarded", ZH_REJECT]) {
            MemoryListFilter::Rejected
        } else if contains_any(normalized, &["all", "every", "\u{5168}\u{90e8}"]) {
            MemoryListFilter::All
        } else {
            MemoryListFilter::Pending
        };
        return Some(MainAgentIntent::ListMemories { filter });
    }

    if contains_any(
        normalized,
        &[
            "approve memory",
            "approve memory candidate",
            "accept memory",
            "accept memory candidate",
            ZH_APPROVE,
            ZH_ACCEPT,
        ],
    ) {
        if is_bulk_memory_review_request(normalized) {
            return Some(MainAgentIntent::BulkReviewMemories {
                status: MemoryStatus::Approved,
            });
        }
        let selector = extract_memory_selector(content);
        return (!selector.is_empty()).then_some(MainAgentIntent::ApproveMemory { selector });
    }

    if contains_any(
        normalized,
        &[
            "update memory",
            "update memory candidate",
            "edit memory",
            "edit memory candidate",
            "revise memory",
            "\u{66f4}\u{65b0}\u{8bb0}\u{5fc6}",
            "\u{4fee}\u{6539}\u{8bb0}\u{5fc6}",
            "\u{7f16}\u{8f91}\u{8bb0}\u{5fc6}",
        ],
    ) {
        return split_memory_update(content)
            .map(|(selector, input)| MainAgentIntent::UpdateMemory { selector, input });
    }

    if contains_any(
        normalized,
        &[
            "delete memory",
            "delete memory candidate",
            "remove memory",
            "remove memory candidate",
            "forget memory",
            "\u{5220}\u{9664}\u{8bb0}\u{5fc6}",
            "\u{79fb}\u{9664}\u{8bb0}\u{5fc6}",
        ],
    ) {
        let selector = extract_memory_selector(content);
        return (!selector.is_empty()).then_some(MainAgentIntent::DeleteMemory { selector });
    }

    if contains_any(
        normalized,
        &[
            "reject memory",
            "reject memory candidate",
            "discard memory",
            "discard memory candidate",
            ZH_REJECT,
        ],
    ) {
        if is_bulk_memory_review_request(normalized) {
            return Some(MainAgentIntent::BulkReviewMemories {
                status: MemoryStatus::Rejected,
            });
        }
        let selector = extract_memory_selector(content);
        return (!selector.is_empty()).then_some(MainAgentIntent::RejectMemory { selector });
    }

    None
}

fn is_bulk_memory_review_request(normalized: &str) -> bool {
    contains_any(
        normalized,
        &[
            "all pending",
            "all memory",
            "all memories",
            "all candidates",
            "every memory",
            "every candidate",
            "\u{5168}\u{90e8}",
            "\u{6240}\u{6709}",
        ],
    )
}

fn split_memory_update(content: &str) -> Option<(String, UpdateMemory)> {
    for separator in ["\u{ff1a}", ":", "\n"] {
        if let Some((head, tail)) = content.split_once(separator) {
            let selector = extract_memory_selector(head);
            let content = tail.trim().trim_matches(['"', '\'']).trim();
            if !selector.is_empty() && !content.is_empty() {
                return Some((
                    selector,
                    UpdateMemory {
                        scope: None,
                        content: Some(content.to_owned()),
                        confidence: None,
                    },
                ));
            }
        }
    }

    None
}

fn split_memory_scope_and_content(content: &str) -> (String, String) {
    for separator in ["\u{ff1a}", ":", "\n"] {
        if let Some((head, tail)) = content.split_once(separator) {
            let scope = head.trim().trim_matches(['"', '\'']).trim().to_lowercase();
            let memory_content = tail.trim().trim_matches(['"', '\'']).trim();
            if !scope.is_empty()
                && !memory_content.is_empty()
                && scope.chars().count() <= 40
                && !scope.contains(' ')
            {
                return (scope, memory_content.to_owned());
            }
        }
    }

    (
        "repo".to_owned(),
        content.trim().trim_matches(['"', '\'']).trim().to_owned(),
    )
}

fn split_reply_selector_and_content(content: &str) -> Option<(String, String)> {
    for separator in ["\u{ff1a}", ":", "\n"] {
        if let Some((head, tail)) = content.split_once(separator) {
            let selector = semantic_task_selector(head).unwrap_or_else(|| {
                extract_task_selector(
                    head,
                    &[
                        "reply", "answer", "respond", "task", "to", "for", "with", "message",
                    ],
                )
            });
            let content = tail.trim();
            if !selector.is_empty() && !content.is_empty() {
                return Some((selector, content.to_owned()));
            }
        }
    }

    None
}

fn split_task_finish_selector_and_summary(
    content: &str,
    is_failure: bool,
) -> Option<(String, String)> {
    for separator in ["\u{ff1a}", ":", "\n"] {
        if let Some((head, tail)) = content.split_once(separator) {
            let selector = extract_task_selector(
                head,
                &[
                    "complete",
                    "completed",
                    "finish",
                    "finished",
                    "done",
                    "fail",
                    "failed",
                    "summary",
                    "reason",
                    "error",
                ],
            );
            let summary = tail.trim().trim_matches(['"', '\'']).trim();
            if !selector.is_empty() && !summary.is_empty() {
                return Some((selector, summary.to_owned()));
            }
        }
    }

    let selector = extract_task_selector(
        content,
        &[
            "complete",
            "completed",
            "finish",
            "finished",
            "done",
            "fail",
            "failed",
            "summary",
            "reason",
            "error",
        ],
    );
    if selector.is_empty() || selector == content {
        return None;
    }

    let summary = if is_failure {
        "Marked failed by user."
    } else {
        "Marked complete by user."
    };
    Some((selector, summary.to_owned()))
}

fn split_task_retry_selector_and_reason(content: &str) -> Option<(String, String)> {
    for separator in ["\u{ff1a}", ":", "\n"] {
        if let Some((head, tail)) = content.split_once(separator) {
            let selector = extract_task_selector(head, &["reason"]);
            let reason = tail.trim().trim_matches(['"', '\'']).trim();
            if !selector.is_empty() && !reason.is_empty() {
                return Some((selector, reason.to_owned()));
            }
        }
    }

    let selector = extract_task_selector(content, &["reason"]);
    if selector.is_empty() || selector == content {
        return None;
    }

    Some((selector, "Retry requested by user.".to_owned()))
}

fn split_selector_and_value_after_phrase(
    content: &str,
    normalized: &str,
    phrase: &str,
    separator: &str,
) -> Option<(String, String)> {
    let phrase_index = normalized.find(phrase)?;
    let after_phrase_start = phrase_index + phrase.len();
    let after_phrase = &content[after_phrase_start..];
    let after_phrase_normalized = &normalized[after_phrase_start..];
    let separator_index = after_phrase_normalized.find(separator)?;
    let selector = after_phrase[..separator_index]
        .trim()
        .trim_matches([':', '\u{ff1a}', '"', '\''])
        .trim();
    let value = after_phrase[separator_index + separator.len()..]
        .trim()
        .trim_matches(['"', '\''])
        .trim();

    (!selector.is_empty() && !value.is_empty()).then(|| (selector.to_owned(), value.to_owned()))
}

fn split_selector_and_value_after_marker(
    content: &str,
    normalized: &str,
    prefixes: &[&str],
    separators: &[&str],
) -> Option<(String, String)> {
    for prefix in prefixes {
        if let Some(prefix_index) = normalized.find(prefix) {
            let after_prefix_start = prefix_index + prefix.len();
            let after_prefix = &content[after_prefix_start..];
            let after_prefix_normalized = &normalized[after_prefix_start..];
            for separator in separators {
                if let Some(separator_index) = after_prefix_normalized.find(separator) {
                    let selector = after_prefix[..separator_index]
                        .trim()
                        .trim_matches([':', '\u{ff1a}', '"', '\''])
                        .trim();
                    let value = after_prefix[separator_index + separator.len()..]
                        .trim()
                        .trim_matches(['"', '\''])
                        .trim();
                    if !selector.is_empty() && !value.is_empty() {
                        return Some((selector.to_owned(), value.to_owned()));
                    }
                }
            }
        }
    }

    None
}

fn extract_labeled_value(content: &str, normalized: &str, labels: &[&str]) -> Option<String> {
    let mut earliest: Option<(usize, usize)> = None;
    for label in labels {
        for suffix in [":", "\u{ff1a}"] {
            let marker = format!("{label}{suffix}");
            if let Some(index) = normalized.find(&marker) {
                let value_start = index + marker.len();
                if earliest
                    .map(|(existing_index, _)| index < existing_index)
                    .unwrap_or(true)
                {
                    earliest = Some((index, value_start));
                }
            }
        }
    }

    let (_, value_start) = earliest?;
    let mut value_end = content.len();
    let value_tail_normalized = &normalized[value_start..];
    for label in [
        " title:",
        " description:",
        " desc:",
        " \u{6807}\u{9898}\u{ff1a}",
        " \u{63cf}\u{8ff0}\u{ff1a}",
        " \u{5185}\u{5bb9}\u{ff1a}",
    ] {
        if let Some(index) = value_tail_normalized.find(label) {
            value_end = value_end.min(value_start + index);
        }
    }

    let value = content[value_start..value_end]
        .trim()
        .trim_matches(['"', '\''])
        .trim();
    (!value.is_empty()).then(|| value.to_owned())
}

fn semantic_task_selector(value: &str) -> Option<String> {
    let normalized = value.to_lowercase();
    if contains_any(
        &normalized,
        &[
            "blocked task",
            "blocked",
            "waiting task",
            "waiting for user",
        ],
    ) {
        Some("blocked".to_owned())
    } else if contains_any(&normalized, &["running task", "running"]) {
        Some("running".to_owned())
    } else if contains_any(&normalized, &["queued task", "queued", "next task"]) {
        Some("queued".to_owned())
    } else if contains_any(&normalized, &["paused task", "paused"]) {
        Some("paused".to_owned())
    } else if contains_any(&normalized, &["failed task", "failed"]) {
        Some("failed".to_owned())
    } else {
        None
    }
}

fn split_clarification_selector_and_question(content: &str) -> Option<(String, String)> {
    for separator in ["\u{ff1a}", ":", "\n"] {
        if let Some((head, tail)) = content.split_once(separator) {
            let selector_head = if head.contains(ZH_NEED) {
                head.split(ZH_NEED).next().unwrap_or(head)
            } else if head.contains(ZH_CLARIFY) {
                head.split(ZH_CLARIFY).next().unwrap_or(head)
            } else {
                head
            };
            let selector = extract_task_selector(
                selector_head,
                &[
                    "ask",
                    "request",
                    "clarification",
                    "user",
                    "for",
                    "about",
                    ZH_CLARIFY,
                    ZH_QUESTION,
                    ZH_TO,
                ],
            );
            let question = tail.trim();
            if !selector.is_empty() && !question.is_empty() {
                return Some((selector, question.to_owned()));
            }
        }
    }

    None
}

fn split_note_selector_and_content(content: &str) -> Option<(String, String)> {
    for separator in ["\u{ff1a}", ":", "\n"] {
        if let Some((head, tail)) = content.split_once(separator) {
            let selector_head = if head.contains(ZH_NOTE) {
                head.split(ZH_NOTE).next().unwrap_or(head)
            } else {
                head
            };
            let selector = extract_task_selector(
                selector_head,
                &["note", "notes", "add", "to", "for", ZH_ADD, ZH_NOTE, ZH_TO],
            );
            let note = tail.trim();
            if !selector.is_empty() && !note.is_empty() {
                return Some((selector, note.to_owned()));
            }
        }
    }

    for marker in [" note ", " note: ", ZH_NOTE] {
        if let Some((head, tail)) = content.split_once(marker) {
            let selector = extract_task_selector(head, &[]);
            let note = tail.trim().trim_matches([':', '\u{ff1a}']).trim();
            if !selector.is_empty() && !note.is_empty() {
                return Some((selector, note.to_owned()));
            }
        }
    }

    None
}

fn split_resource_lock_selector_and_key(content: &str) -> Option<(String, String)> {
    for separator in ["\u{ff1a}", ":", "\n"] {
        if let Some((head, tail)) = content.split_once(separator) {
            let selector = extract_task_selector(
                head,
                &[
                    "resource",
                    "lock",
                    "resource lock",
                    "resource-lock",
                    "to",
                    "for",
                    "from",
                    ZH_ADD,
                    ZH_RESOURCE,
                    ZH_RESOURCE_LOCK,
                    ZH_TO,
                ],
            );
            let resource_key = tail.trim();
            if !selector.is_empty() && !resource_key.is_empty() {
                return Some((selector, resource_key.to_owned()));
            }
        }
    }

    None
}

fn split_skill_selector_and_names(
    content: &str,
    normalized: &str,
    is_remove: bool,
) -> Option<(String, Vec<String>)> {
    for separator in ["\u{ff1a}", ":", "\n"] {
        if let Some((head, tail)) = content.split_once(separator) {
            let selector_head = if head.contains(ZH_SKILL) && head.contains(ZH_TASK) {
                let before_skill = head.split(ZH_SKILL).next().unwrap_or(head);
                before_skill.split(ZH_ADD).next().unwrap_or(before_skill)
            } else {
                head
            };
            let selector = extract_task_selector(
                selector_head,
                &[
                    "requested",
                    "skill",
                    "skills",
                    "to",
                    "for",
                    "from",
                    ZH_ADD,
                    ZH_SKILL,
                    ZH_TO,
                ],
            );
            let skill_names = parse_skill_names(tail);
            if !selector.is_empty() && !skill_names.is_empty() {
                return Some((selector, skill_names));
            }
        }
    }

    if is_remove {
        split_inline_skill_selector_and_names(
            content,
            normalized,
            &[
                " from task ",
                " from ",
                " for task ",
                " for ",
                " task ",
                ZH_TO,
                ZH_TASK,
            ],
            &[
                "remove requested skills",
                "remove requested skill",
                "remove skills",
                "remove skill",
                "delete skill",
                "detach skill",
                "clear skill",
                ZH_CANCEL,
                ZH_REJECT,
                ZH_SKILL,
            ],
        )
    } else {
        split_inline_skill_selector_and_names(
            content,
            normalized,
            &[
                " to task ",
                " to ",
                " for task ",
                " for ",
                " task ",
                ZH_TO,
                ZH_TASK,
            ],
            &[
                "add requested skills",
                "add requested skill",
                "add skills",
                "add skill",
                "attach skill",
                "use skill",
                ZH_ADD,
                ZH_SKILL,
            ],
        )
    }
}

fn split_inline_skill_selector_and_names(
    content: &str,
    normalized: &str,
    relation_markers: &[&str],
    action_words: &[&str],
) -> Option<(String, Vec<String>)> {
    relation_markers
        .iter()
        .filter_map(|marker| normalized.find(marker).map(|index| (index, *marker)))
        .min_by_key(|(index, _)| *index)
        .and_then(|(index, marker)| {
            let skill_part = content[..index].trim();
            let selector_part = content[index + marker.len()..].trim();
            let mut skill_text = skill_part.to_owned();
            for action_word in action_words {
                skill_text = replace_case_insensitive(&skill_text, action_word, "");
            }

            let selector = extract_task_selector(selector_part, &[]);
            let skill_names = parse_skill_names(&skill_text);
            (!selector.is_empty() && !skill_names.is_empty()).then_some((selector, skill_names))
        })
}

fn parse_skill_names(content: &str) -> Vec<String> {
    normalize_skill_names(
        content
            .split([',', ';', '\n', '\u{ff0c}', '\u{ff1b}', '\u{3001}'])
            .flat_map(|part| part.split(" and "))
            .flat_map(|part| part.split(" with "))
            .map(|part| part.trim().trim_matches(['"', '\'', '`']).to_owned())
            .collect(),
    )
}

fn normalize_skill_names(skill_names: Vec<String>) -> Vec<String> {
    let mut normalized = Vec::new();
    for skill_name in skill_names {
        let skill_name = skill_name
            .trim()
            .trim_matches([
                ':', '\u{ff1a}', ',', '\u{ff0c}', '.', '\u{3002}', '"', '\'', '`',
            ])
            .trim()
            .to_owned();
        if skill_name.is_empty()
            || normalized
                .iter()
                .any(|existing: &String| existing.eq_ignore_ascii_case(&skill_name))
        {
            continue;
        }
        normalized.push(skill_name);
    }

    normalized
}

fn format_skill_list(skill_names: &[String]) -> String {
    if skill_names.is_empty() {
        "none".to_owned()
    } else {
        skill_names.join(", ")
    }
}

fn active_skill_names_for_task(task: &Task) -> Vec<String> {
    let mut skills = Vec::new();
    for skill in task
        .requested_skills
        .iter()
        .chain(task.matched_skills.iter())
    {
        if !skills
            .iter()
            .any(|existing: &String| existing.eq_ignore_ascii_case(skill))
        {
            skills.push(skill.clone());
        }
    }
    skills
}

fn format_skill_definitions(skills: &[Skill]) -> String {
    if skills.is_empty() {
        return "No skills are defined yet.".to_owned();
    }

    skills
        .iter()
        .map(|skill| {
            format!(
                "- {}: {}\n  triggers: {}\n  tools: {}\n  resource: {}",
                skill.name,
                if skill.description.trim().is_empty() {
                    "no description"
                } else {
                    skill.description.trim()
                },
                format_skill_list(&skill.trigger_rules),
                format_skill_list(&skill.tool_subset),
                skill.resource_path.as_deref().unwrap_or("none")
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn extract_memory_selector(content: &str) -> String {
    for separator in ["\u{ff1a}", ":", "\n"] {
        if let Some((_, tail)) = content.split_once(separator) {
            let tail = tail.trim();
            if !tail.is_empty() {
                return tail.trim_matches(['"', '\'']).trim().to_owned();
            }
        }
    }

    let mut selector = content.to_owned();
    for word in [
        "approve memory candidate",
        "approve memory",
        "accept memory candidate",
        "accept memory",
        "reject memory candidate",
        "reject memory",
        "discard memory candidate",
        "discard memory",
        "update memory candidate",
        "update memory",
        "edit memory candidate",
        "edit memory",
        "revise memory candidate",
        "revise memory",
        "delete memory candidate",
        "delete memory",
        "remove memory candidate",
        "remove memory",
        "forget memory",
        "memory candidate",
        "long-term memory",
        "approve",
        "accept",
        "reject",
        "discard",
        "update",
        "edit",
        "revise",
        "delete",
        "remove",
        "forget",
        "memory",
        ZH_LONG_TERM_MEMORY,
        ZH_MEMORY,
        ZH_APPROVE,
        ZH_ACCEPT,
        ZH_REJECT,
        "\u{66f4}\u{65b0}",
        "\u{4fee}\u{6539}",
        "\u{7f16}\u{8f91}",
        "\u{5220}\u{9664}",
        "\u{79fb}\u{9664}",
    ] {
        selector = replace_case_insensitive(&selector, word, "");
    }

    selector
        .trim()
        .trim_matches([':', '\u{ff1a}', ',', '\u{ff0c}', '.', '\u{3002}', '"', '\''])
        .trim()
        .to_owned()
}

fn split_dependency_selectors(
    content: &str,
    normalized: &str,
    markers: &[&str],
) -> Option<(String, String)> {
    markers
        .iter()
        .filter_map(|marker| normalized.find(marker).map(|index| (index, *marker)))
        .min_by_key(|(index, _)| *index)
        .and_then(|(index, marker)| {
            let left = content[..index].trim();
            let right = content[index + marker.len()..].trim();
            let selector = extract_task_selector(left, &[]);
            let depends_on_selector = extract_task_selector(right, &[]);

            (!selector.is_empty() && !depends_on_selector.is_empty())
                .then_some((selector, depends_on_selector))
        })
}

fn extract_title(content: &str) -> String {
    let content = strip_create_skill_clause(content);
    for separator in ["\u{ff1a}", ":", "\u{ff0c}", ",", "\n"] {
        if let Some((_, tail)) = content.split_once(separator) {
            let title = tail.trim();
            if !title.is_empty() {
                return clamp_title(title);
            }
        }
    }

    let title = content
        .replace(&format!("{ZH_CREATE}{ZH_RECURRING}{ZH_TASK}"), "")
        .replace(&format!("{ZH_CREATE}{ZH_ONE_OFF}{ZH_TASK}"), "")
        .replace(&format!("{ZH_CREATE}{ZH_TASK}"), "")
        .replace(&format!("{ZH_NEW}{ZH_TASK}"), "")
        .replace(&format!("{ZH_ADD}{ZH_TASK}"), "")
        .replace("create task", "")
        .replace("add task", "")
        .trim()
        .to_owned();

    if title.is_empty() {
        "Untitled task".to_owned()
    } else {
        clamp_title(&title)
    }
}

fn extract_create_requested_skills(content: &str, normalized: &str) -> Vec<String> {
    for marker in [
        " with requested skills ",
        " with requested skill ",
        " with skills ",
        " with skill ",
        " using skills ",
        " using skill ",
        " use skills ",
        " use skill ",
        "\u{4f7f}\u{7528}\u{6280}\u{80fd}",
        "\u{6307}\u{5b9a}\u{6280}\u{80fd}",
    ] {
        if let Some(index) = normalized.find(marker) {
            let after = &content[index + marker.len()..];
            return parse_skill_names(after);
        }
    }

    Vec::new()
}

fn strip_create_skill_clause(content: &str) -> String {
    let normalized = content.to_lowercase();
    let mut end = content.len();
    for marker in [
        " with requested skills ",
        " with requested skill ",
        " with skills ",
        " with skill ",
        " using skills ",
        " using skill ",
        " use skills ",
        " use skill ",
        "\u{4f7f}\u{7528}\u{6280}\u{80fd}",
        "\u{6307}\u{5b9a}\u{6280}\u{80fd}",
    ] {
        if let Some(index) = normalized.find(marker) {
            end = end.min(index);
        }
    }

    content[..end].trim().to_owned()
}

fn extract_task_selector(content: &str, stop_words: &[&str]) -> String {
    let normalized = content.to_lowercase();
    let action_words = [
        "convert task",
        "change task type",
        "make task",
        "add dependency",
        "set dependency",
        "remove dependency",
        "delete dependency",
        "clear dependency",
        "add resource lock to task",
        "add resource lock",
        "remove resource lock from task",
        "remove resource lock",
        "delete resource lock",
        "clear resource lock",
        "lock resource",
        "unlock resource",
        "add note to task",
        "add note",
        "note to task",
        "note task",
        "note",
        "add requested skills to task",
        "add requested skill to task",
        "add requested skills",
        "add requested skill",
        "add skills to task",
        "add skill to task",
        "add skills",
        "add skill",
        "remove requested skills from task",
        "remove requested skill from task",
        "remove requested skills",
        "remove requested skill",
        "remove skills from task",
        "remove skill from task",
        "remove skills",
        "remove skill",
        "attach skill",
        "detach skill",
        "use skill",
        "ask clarification for task",
        "request clarification for task",
        "ask user for task",
        "ask clarification",
        "request clarification",
        "ask user",
        "reprioritize",
        "set priority",
        "change priority",
        "reorder",
        "move task",
        "pause task",
        "pause",
        "resume task",
        "resume",
        "unpause task",
        "cancel task",
        "cancel",
        "delete the task",
        "delete task",
        "delete",
        "remove the task",
        "remove task",
        "complete task",
        "completed task",
        "finish task",
        "finished task",
        "fail task",
        "failed task",
        "retry failed task",
        "retry task",
        "requeue failed task",
        "requeue task",
        "try task again",
        "run task again",
        "run the task",
        "run task",
        "execute the task",
        "execute task",
        "start the task",
        "start task",
        "mark the task",
        "mark task",
        "update task title",
        "update task description",
        "update task schedule",
        "update task interval",
        "update task",
        "change task title",
        "change task description",
        "change task schedule",
        "change task interval",
        "change task",
        "set task schedule",
        "set task interval",
        "set schedule",
        "set interval",
        "edit task",
        "rename task",
        "task",
        ZH_CONVERT_TASK,
        ZH_TASK,
        ZH_PAUSE,
        ZH_RESUME,
        ZH_CANCEL,
        ZH_DELETE_TASK,
        ZH_NOTE,
        ZH_SKILL,
        ZH_CLARIFY,
        ZH_QUESTION,
        ZH_NEED,
        ZH_ADJUST,
        ZH_MOVE,
        ZH_CHANGE_TO,
        ZH_CHANGE_AS,
        ZH_SET_AS,
    ];

    let mut start = 0;
    for word in action_words {
        if let Some(index) = normalized.find(word) {
            start = (index + word.len()).max(start);
        }
    }

    let mut selector = content[start..].trim().trim_matches(['"', '\'']).to_owned();
    let selector_lower = selector.to_lowercase();
    let mut end = selector.len();
    for word in stop_words {
        if let Some(index) = selector_lower.find(word) {
            end = end.min(index);
        }
    }
    selector.truncate(end);

    selector
        .trim()
        .trim_matches([':', '\u{ff1a}', ',', '\u{ff0c}', '.', '\u{3002}', '"', '\''])
        .trim()
        .to_owned()
}

fn clamp_title(title: &str) -> String {
    let mut chars = title.chars();
    let clipped: String = chars.by_ref().take(80).collect();
    if chars.next().is_some() {
        format!("{clipped}...")
    } else {
        clipped
    }
}

fn extract_priority(normalized: &str) -> Option<i64> {
    extract_number_after_any(normalized, &["priority", ZH_PRIORITY])
}

fn extract_queue_position(normalized: &str) -> Option<i64> {
    extract_number_after_any(
        normalized,
        &[
            "queue position",
            "queue",
            "position",
            ZH_QUEUE,
            ZH_SORT,
            ZH_POSITION,
        ],
    )
}

fn extract_interval_seconds(normalized: &str) -> Option<i64> {
    extract_duration_after_last_any(normalized, &["every", "interval", ZH_INTERVAL, ZH_EVERY])
}

fn extract_duration_after_last_any(normalized: &str, markers: &[&str]) -> Option<i64> {
    markers
        .iter()
        .filter_map(|marker| normalized.rfind(marker).map(|index| (index, *marker)))
        .max_by_key(|(index, _)| *index)
        .and_then(|(index, marker)| {
            let after = &normalized[index + marker.len()..];
            let digit_start = after
                .char_indices()
                .find(|(_, ch)| ch.is_ascii_digit() || *ch == '-')
                .map(|(index, _)| index)?;
            let digits = after[digit_start..]
                .char_indices()
                .take_while(|(_, ch)| ch.is_ascii_digit() || *ch == '-')
                .last()
                .map(|(index, ch)| digit_start + index + ch.len_utf8())?;
            let value = after[digit_start..digits].parse::<i64>().ok()?;
            let unit_tail = after[digits..].trim_start();
            Some(value * interval_unit_multiplier(unit_tail))
        })
}

fn interval_unit_multiplier(unit_tail: &str) -> i64 {
    if unit_tail.starts_with("day")
        || unit_tail.starts_with("days")
        || unit_tail.starts_with("\u{5929}")
    {
        86_400
    } else if unit_tail.starts_with("hour")
        || unit_tail.starts_with("hours")
        || unit_tail.starts_with("hr")
        || unit_tail.starts_with("\u{5c0f}\u{65f6}")
    {
        3_600
    } else if unit_tail.starts_with("minute")
        || unit_tail.starts_with("minutes")
        || unit_tail.starts_with("min")
        || unit_tail.starts_with("\u{5206}\u{949f}")
    {
        60
    } else {
        1
    }
}

fn extract_number_after_any(normalized: &str, markers: &[&str]) -> Option<i64> {
    for marker in markers {
        if let Some(index) = normalized.find(marker) {
            let after = &normalized[index + marker.len()..];
            let digits: String = after
                .chars()
                .skip_while(|ch| !ch.is_ascii_digit() && *ch != '-')
                .take_while(|ch| ch.is_ascii_digit() || *ch == '-')
                .collect();
            if let Ok(value) = digits.parse() {
                return Some(value);
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use persistent_agent_db::Db;
    use persistent_agent_domain::CreateMemory;
    use std::sync::Mutex as StdMutex;

    struct FixedAdvisor {
        reply: String,
        contexts: Arc<StdMutex<Vec<MainAgentAdviceContext>>>,
    }

    #[async_trait]
    impl MainAgentAdvisor for FixedAdvisor {
        async fn advise(&self, context: MainAgentAdviceContext) -> anyhow::Result<String> {
            self.contexts
                .lock()
                .expect("advisor contexts lock")
                .push(context);
            Ok(self.reply.clone())
        }
    }

    struct FixedPlanner {
        plan: Option<MainAgentPlan>,
        contexts: Arc<StdMutex<Vec<MainAgentPlanContext>>>,
    }

    #[async_trait]
    impl MainAgentPlanner for FixedPlanner {
        async fn plan(
            &self,
            context: MainAgentPlanContext,
        ) -> anyhow::Result<Option<MainAgentPlan>> {
            self.contexts
                .lock()
                .expect("planner contexts lock")
                .push(context);
            Ok(self.plan.clone())
        }
    }

    #[test]
    fn parses_create_task_with_priority() {
        let intent = parse_intent("create task: Check GitHub issues priority 7");

        assert_eq!(
            intent,
            MainAgentIntent::CreateTask {
                title: "Check GitHub issues priority 7".to_owned(),
                description: "create task: Check GitHub issues priority 7".to_owned(),
                task_type: TaskType::OneOff,
                priority: 7,
                interval_seconds: None,
                requested_skills: Vec::new(),
            }
        );

        assert_eq!(
            parse_intent("create task: Check GitHub issues with skills github, shell"),
            MainAgentIntent::CreateTask {
                title: "Check GitHub issues".to_owned(),
                description: "create task: Check GitHub issues with skills github, shell"
                    .to_owned(),
                task_type: TaskType::OneOff,
                priority: 0,
                interval_seconds: None,
                requested_skills: vec!["github".to_owned(), "shell".to_owned()],
            }
        );
    }

    #[test]
    fn parses_workspace_inspection_intents() {
        assert_eq!(
            parse_intent("check project status"),
            MainAgentIntent::InspectWorkspace
        );
        assert_eq!(
            parse_intent("git status"),
            MainAgentIntent::InspectWorkspace
        );
        assert_eq!(
            parse_intent("read file README.md"),
            MainAgentIntent::InspectWorkspaceFile {
                path: "README.md".to_owned(),
            }
        );
        assert_eq!(
            parse_intent("list directory crates"),
            MainAgentIntent::InspectWorkspaceDirectory {
                path: "crates".to_owned(),
            }
        );
        assert_eq!(
            parse_intent("list workspace files"),
            MainAgentIntent::InspectWorkspaceDirectory {
                path: ".".to_owned(),
            }
        );
        assert_eq!(
            parse_intent("\u{67e5}\u{770b}\u{6587}\u{4ef6}\u{ff1a}Cargo.toml"),
            MainAgentIntent::InspectWorkspaceFile {
                path: "Cargo.toml".to_owned(),
            }
        );
    }

    #[test]
    fn parses_split_tasks_intents() {
        assert_eq!(
            parse_intent("split goal: investigate issue; write fix; run tests"),
            MainAgentIntent::SplitTasks {
                titles: vec![
                    "investigate issue".to_owned(),
                    "write fix".to_owned(),
                    "run tests".to_owned(),
                ],
            }
        );
        assert_eq!(
            parse_intent(
                "\u{62c6}\u{5206}\u{76ee}\u{6807}\u{ff1a}\u{5206}\u{6790} issue\u{ff1b}\u{4fee}\u{590d}\u{4ee3}\u{7801}\u{ff1b}\u{8fd0}\u{884c}\u{6d4b}\u{8bd5}"
            ),
            MainAgentIntent::SplitTasks {
                titles: vec![
                    "\u{5206}\u{6790} issue".to_owned(),
                    "\u{4fee}\u{590d}\u{4ee3}\u{7801}".to_owned(),
                    "\u{8fd0}\u{884c}\u{6d4b}\u{8bd5}".to_owned(),
                ],
            }
        );
    }

    #[test]
    fn parses_recurring_chinese_create_task() {
        let content = "\u{521b}\u{5efa}\u{5faa}\u{73af}\u{4efb}\u{52a1}\u{ff1a}\u{6bcf}\u{5929}\u{68c0}\u{67e5}\u{4ed3}\u{5e93} issue \u{4f18}\u{5148}\u{7ea7} 3 \u{6bcf} 60 \u{79d2}";
        let intent = parse_intent(content);

        assert_eq!(
            intent,
            MainAgentIntent::CreateTask {
                title: "\u{6bcf}\u{5929}\u{68c0}\u{67e5}\u{4ed3}\u{5e93} issue \u{4f18}\u{5148}\u{7ea7} 3 \u{6bcf} 60 \u{79d2}".to_owned(),
                description: content.to_owned(),
                task_type: TaskType::Recurring,
                priority: 3,
                interval_seconds: Some(60),
                requested_skills: Vec::new(),
            }
        );
    }

    #[test]
    fn parses_task_pool_summary() {
        assert_eq!(
            parse_intent("\u{603b}\u{7ed3}\u{4efb}\u{52a1}\u{6c60}"),
            MainAgentIntent::Summarize
        );
    }

    #[test]
    fn parses_task_list_intents() {
        assert_eq!(parse_intent("list tasks"), MainAgentIntent::ListTasks);
        assert_eq!(
            parse_intent("\u{5217}\u{51fa}\u{4efb}\u{52a1}"),
            MainAgentIntent::ListTasks
        );
        assert_eq!(
            parse_intent("show tasks waiting for my input"),
            MainAgentIntent::ListWaitingForUserTasks
        );
        assert_eq!(
            parse_intent("what needs user input?"),
            MainAgentIntent::ListWaitingForUserTasks
        );
        assert_eq!(
            parse_intent("show tasks waiting for schedule"),
            MainAgentIntent::ListWaitingForScheduleTasks
        );
        assert_eq!(
            parse_intent("which recurring tasks are waiting?"),
            MainAgentIntent::ListWaitingForScheduleTasks
        );
        assert_eq!(
            parse_intent("list failed tasks"),
            MainAgentIntent::ListTasksByStatus {
                status: TaskStatus::Failed
            }
        );
        assert_eq!(
            parse_intent("show queued tasks"),
            MainAgentIntent::ListTasksByStatus {
                status: TaskStatus::Queued
            }
        );
        assert_eq!(
            parse_intent("\u{67e5}\u{770b}\u{5931}\u{8d25}\u{4efb}\u{52a1}"),
            MainAgentIntent::ListTasksByStatus {
                status: TaskStatus::Failed
            }
        );
    }

    #[test]
    fn parses_global_action_list_intents() {
        assert_eq!(
            parse_intent("show main agent audit"),
            MainAgentIntent::ListGlobalActions
        );
        assert_eq!(
            parse_intent("list recent tool calls"),
            MainAgentIntent::ListGlobalActions
        );
    }

    #[test]
    fn parses_task_artifact_list_intents() {
        assert_eq!(
            parse_intent("show result for task Deploy release"),
            MainAgentIntent::ShowTaskLatestResult {
                selector: "Deploy release".to_owned(),
            }
        );
        assert_eq!(
            parse_intent("show summary for task Deploy release"),
            MainAgentIntent::ShowTaskLatestResult {
                selector: "Deploy release".to_owned(),
            }
        );
        assert_eq!(
            parse_intent(
                "\u{67e5}\u{770b}\u{4efb}\u{52a1} \u{53d1}\u{5e03}\u{7248}\u{672c} \u{7ed3}\u{679c}"
            ),
            MainAgentIntent::ShowTaskLatestResult {
                selector: "\u{53d1}\u{5e03}\u{7248}\u{672c}".to_owned(),
            }
        );
        assert_eq!(
            parse_intent("show artifacts for task Deploy release"),
            MainAgentIntent::ListTaskArtifacts {
                selector: "Deploy release".to_owned(),
            }
        );
        assert_eq!(
            parse_intent("list task Deploy release outputs"),
            MainAgentIntent::ListTaskArtifacts {
                selector: "Deploy release".to_owned(),
            }
        );
        assert_eq!(
            parse_intent(
                "\u{67e5}\u{770b}\u{4efb}\u{52a1} \u{53d1}\u{5e03}\u{7248}\u{672c} \u{4ea7}\u{7269}"
            ),
            MainAgentIntent::ListTaskArtifacts {
                selector: "\u{53d1}\u{5e03}\u{7248}\u{672c}".to_owned(),
            }
        );
    }

    #[test]
    fn parses_task_history_intents() {
        assert_eq!(
            parse_intent("show history for task Deploy release"),
            MainAgentIntent::ListTaskHistory {
                selector: "Deploy release".to_owned(),
            }
        );
        assert_eq!(
            parse_intent("list task Deploy release worker events"),
            MainAgentIntent::ListTaskHistory {
                selector: "Deploy release".to_owned(),
            }
        );
        assert_eq!(
            parse_intent(
                "\u{67e5}\u{770b}\u{4efb}\u{52a1} \u{53d1}\u{5e03}\u{7248}\u{672c} \u{6267}\u{884c}\u{8bb0}\u{5f55}"
            ),
            MainAgentIntent::ListTaskHistory {
                selector: "\u{53d1}\u{5e03}\u{7248}\u{672c}".to_owned(),
            }
        );
    }

    #[test]
    fn parses_task_follow_up_intents() {
        assert_eq!(
            parse_intent("show follow-up tasks for task Deploy release"),
            MainAgentIntent::ListTaskFollowUps {
                selector: "Deploy release".to_owned(),
            }
        );
        assert_eq!(
            parse_intent("list task Deploy release follow up tasks"),
            MainAgentIntent::ListTaskFollowUps {
                selector: "Deploy release".to_owned(),
            }
        );
        assert_eq!(
            parse_intent(
                "\u{67e5}\u{770b}\u{4efb}\u{52a1} \u{53d1}\u{5e03}\u{7248}\u{672c} \u{540e}\u{7eed}\u{4efb}\u{52a1}"
            ),
            MainAgentIntent::ListTaskFollowUps {
                selector: "\u{53d1}\u{5e03}\u{7248}\u{672c}".to_owned(),
            }
        );
    }

    #[test]
    fn parses_task_conversation_intents() {
        assert_eq!(
            parse_intent("show conversation for task Deploy release"),
            MainAgentIntent::ListTaskConversation {
                selector: "Deploy release".to_owned(),
            }
        );
        assert_eq!(
            parse_intent("list task Deploy release messages"),
            MainAgentIntent::ListTaskConversation {
                selector: "Deploy release".to_owned(),
            }
        );
        assert_eq!(
            parse_intent(
                "\u{67e5}\u{770b}\u{4efb}\u{52a1} \u{53d1}\u{5e03}\u{7248}\u{672c} \u{5bf9}\u{8bdd}"
            ),
            MainAgentIntent::ListTaskConversation {
                selector: "\u{53d1}\u{5e03}\u{7248}\u{672c}".to_owned(),
            }
        );
    }

    #[test]
    fn parses_scheduler_scan_intents() {
        assert_eq!(
            parse_intent("run scheduler tick"),
            MainAgentIntent::RunSchedulerTick
        );
        assert_eq!(
            parse_intent("\u{626b}\u{63cf}\u{4efb}\u{52a1}\u{6c60}"),
            MainAgentIntent::RunSchedulerTick
        );
    }

    #[test]
    fn parses_scheduler_state_intents() {
        assert_eq!(
            parse_intent("show scheduler status"),
            MainAgentIntent::ShowSchedulerState
        );
        assert_eq!(
            parse_intent("what is the agent doing?"),
            MainAgentIntent::ShowSchedulerState
        );
        assert_eq!(
            parse_intent("\u{8c03}\u{5ea6}\u{72b6}\u{6001}"),
            MainAgentIntent::ShowSchedulerState
        );
    }

    #[test]
    fn parses_explain_intents() {
        assert_eq!(
            parse_intent("explain task pool state"),
            MainAgentIntent::ExplainTaskPool
        );
        assert_eq!(
            parse_intent("\u{89e3}\u{91ca}\u{4efb}\u{52a1}\u{6c60}\u{72b6}\u{6001}"),
            MainAgentIntent::ExplainTaskPool
        );
        assert_eq!(
            parse_intent("why is task Deploy release not running?"),
            MainAgentIntent::ExplainTask {
                selector: "Deploy release".to_owned()
            }
        );
        assert_eq!(
            parse_intent("what should I do next?"),
            MainAgentIntent::RecommendNextAction
        );
        assert_eq!(
            parse_intent("\u{4e0b}\u{4e00}\u{6b65}\u{505a}\u{4ec0}\u{4e48}"),
            MainAgentIntent::RecommendNextAction
        );
    }

    #[test]
    fn parses_task_management_intents() {
        assert_eq!(
            parse_intent("pause task Check GitHub issues"),
            MainAgentIntent::PauseTask {
                selector: "Check GitHub issues".to_owned()
            }
        );
        assert_eq!(
            parse_intent("resume task Check GitHub issues"),
            MainAgentIntent::ResumeTask {
                selector: "Check GitHub issues".to_owned()
            }
        );
        assert_eq!(
            parse_intent("cancel task Check GitHub issues"),
            MainAgentIntent::CancelTask {
                selector: "Check GitHub issues".to_owned()
            }
        );
        assert_eq!(
            parse_intent("delete task Check GitHub issues"),
            MainAgentIntent::DeleteTask {
                selector: "Check GitHub issues".to_owned()
            }
        );
        assert_eq!(
            parse_intent("remove task Check GitHub issues"),
            MainAgentIntent::DeleteTask {
                selector: "Check GitHub issues".to_owned()
            }
        );
        assert_eq!(
            parse_intent("set priority task Check GitHub issues to 8"),
            MainAgentIntent::ReprioritizeTask {
                selector: "Check GitHub issues".to_owned(),
                priority: 8,
            }
        );
        assert_eq!(
            parse_intent("reorder task Check GitHub issues queue 4"),
            MainAgentIntent::ReorderTask {
                selector: "Check GitHub issues".to_owned(),
                queue_position: 4,
            }
        );
        assert_eq!(
            parse_intent("run task Check GitHub issues now"),
            MainAgentIntent::RunTaskNow {
                selector: "Check GitHub issues".to_owned(),
            }
        );
        assert_eq!(
            parse_intent("execute the task Deploy release immediately"),
            MainAgentIntent::RunTaskNow {
                selector: "Deploy release".to_owned(),
            }
        );
        assert_eq!(parse_intent("run next task"), MainAgentIntent::RunNextTask);
        assert_eq!(
            parse_intent("\u{6267}\u{884c}\u{4e0b}\u{4e00}\u{4e2a}\u{4efb}\u{52a1}"),
            MainAgentIntent::RunNextTask
        );
    }

    #[test]
    fn parses_task_finish_intents() {
        assert_eq!(
            parse_intent("complete task Deploy release: deployed to production"),
            MainAgentIntent::CompleteTask {
                selector: "Deploy release".to_owned(),
                summary: "deployed to production".to_owned(),
            }
        );
        assert_eq!(
            parse_intent("mark task Deploy release complete"),
            MainAgentIntent::CompleteTask {
                selector: "Deploy release".to_owned(),
                summary: "Marked complete by user.".to_owned(),
            }
        );
        assert_eq!(
            parse_intent("fail task Deploy release: deployment token expired"),
            MainAgentIntent::FailTask {
                selector: "Deploy release".to_owned(),
                error: "deployment token expired".to_owned(),
            }
        );
        assert_eq!(
            parse_intent("retry task Deploy release: credentials have been refreshed"),
            MainAgentIntent::RetryTask {
                selector: "Deploy release".to_owned(),
                reason: "credentials have been refreshed".to_owned(),
            }
        );
        assert_eq!(
            parse_intent("requeue failed task Deploy release"),
            MainAgentIntent::RetryTask {
                selector: "Deploy release".to_owned(),
                reason: "Retry requested by user.".to_owned(),
            }
        );
    }

    #[test]
    fn parses_chinese_task_management_intents() {
        assert_eq!(
            parse_intent("\u{6682}\u{505c}\u{4efb}\u{52a1} \u{68c0}\u{67e5} GitHub issues"),
            MainAgentIntent::PauseTask {
                selector: "\u{68c0}\u{67e5} GitHub issues".to_owned()
            }
        );
        assert_eq!(
            parse_intent("\u{6062}\u{590d}\u{4efb}\u{52a1} \u{68c0}\u{67e5} GitHub issues"),
            MainAgentIntent::ResumeTask {
                selector: "\u{68c0}\u{67e5} GitHub issues".to_owned()
            }
        );
        assert_eq!(
            parse_intent("\u{53d6}\u{6d88}\u{4efb}\u{52a1} \u{68c0}\u{67e5} GitHub issues"),
            MainAgentIntent::CancelTask {
                selector: "\u{68c0}\u{67e5} GitHub issues".to_owned()
            }
        );
        assert_eq!(
            parse_intent("\u{5220}\u{9664}\u{4efb}\u{52a1} \u{68c0}\u{67e5} GitHub issues"),
            MainAgentIntent::DeleteTask {
                selector: "\u{68c0}\u{67e5} GitHub issues".to_owned()
            }
        );
        assert_eq!(
            parse_intent(
                "\u{8c03}\u{6574}\u{4efb}\u{52a1} \u{68c0}\u{67e5} GitHub issues \u{4f18}\u{5148}\u{7ea7}\u{4e3a} 8"
            ),
            MainAgentIntent::ReprioritizeTask {
                selector: "\u{68c0}\u{67e5} GitHub issues".to_owned(),
                priority: 8,
            }
        );
        assert_eq!(
            parse_intent(
                "\u{79fb}\u{52a8}\u{4efb}\u{52a1} \u{68c0}\u{67e5} GitHub issues \u{5230}\u{961f}\u{5217}\u{4f4d}\u{7f6e} 4"
            ),
            MainAgentIntent::ReorderTask {
                selector: "\u{68c0}\u{67e5} GitHub issues".to_owned(),
                queue_position: 4,
            }
        );
    }

    #[test]
    fn parses_task_detail_update_intents() {
        assert_eq!(
            parse_intent("rename task Deploy release to Ship release"),
            MainAgentIntent::UpdateTaskDetails {
                selector: "Deploy release".to_owned(),
                title: Some("Ship release".to_owned()),
                description: None,
            }
        );
        assert_eq!(
            parse_intent("update task Deploy release title: Ship release"),
            MainAgentIntent::UpdateTaskDetails {
                selector: "Deploy release".to_owned(),
                title: Some("Ship release".to_owned()),
                description: None,
            }
        );
        assert_eq!(
            parse_intent("update task Deploy release description: Deploy to production"),
            MainAgentIntent::UpdateTaskDetails {
                selector: "Deploy release".to_owned(),
                title: None,
                description: Some("Deploy to production".to_owned()),
            }
        );
        assert_eq!(
            parse_intent(
                "\u{66f4}\u{65b0}\u{4efb}\u{52a1} \u{53d1}\u{5e03}\u{7248}\u{672c} \u{63cf}\u{8ff0}\u{ff1a}\u{53d1}\u{5e03}\u{5230}\u{751f}\u{4ea7}"
            ),
            MainAgentIntent::UpdateTaskDetails {
                selector: "\u{53d1}\u{5e03}\u{7248}\u{672c}".to_owned(),
                title: None,
                description: Some("\u{53d1}\u{5e03}\u{5230}\u{751f}\u{4ea7}".to_owned()),
            }
        );
    }

    #[test]
    fn parses_convert_task_type_intents() {
        assert_eq!(
            parse_intent("convert task Check GitHub issues to recurring every 60 seconds"),
            MainAgentIntent::ConvertTaskType {
                selector: "Check GitHub issues".to_owned(),
                task_type: TaskType::Recurring,
                interval_seconds: Some(60),
            }
        );
        assert_eq!(
            parse_intent("convert task Check GitHub issues to one-off"),
            MainAgentIntent::ConvertTaskType {
                selector: "Check GitHub issues".to_owned(),
                task_type: TaskType::OneOff,
                interval_seconds: None,
            }
        );
        assert_eq!(
            parse_intent("set task Check GitHub issues interval to 5 minutes"),
            MainAgentIntent::UpdateTaskSchedule {
                selector: "Check GitHub issues".to_owned(),
                interval_seconds: 300,
            }
        );
        assert_eq!(
            parse_intent("update task Check GitHub issues schedule every 2 hours"),
            MainAgentIntent::UpdateTaskSchedule {
                selector: "Check GitHub issues".to_owned(),
                interval_seconds: 7200,
            }
        );
        assert_eq!(
            parse_intent(
                "\u{66f4}\u{65b0}\u{4efb}\u{52a1} \u{68c0}\u{67e5} GitHub issues \u{95f4}\u{9694}\u{4e3a} 10\u{5206}\u{949f}"
            ),
            MainAgentIntent::UpdateTaskSchedule {
                selector: "\u{68c0}\u{67e5} GitHub issues".to_owned(),
                interval_seconds: 600,
            }
        );
    }

    #[test]
    fn parses_task_dependency_intents() {
        assert_eq!(
            parse_intent("show constraints for task Deploy release"),
            MainAgentIntent::ListTaskConstraints {
                selector: "Deploy release".to_owned(),
            }
        );
        assert_eq!(
            parse_intent("list task Deploy release dependencies"),
            MainAgentIntent::ListTaskConstraints {
                selector: "Deploy release".to_owned(),
            }
        );
        assert_eq!(
            parse_intent(
                "\u{67e5}\u{770b}\u{4efb}\u{52a1} \u{53d1}\u{5e03}\u{7248}\u{672c} \u{7ea6}\u{675f}"
            ),
            MainAgentIntent::ListTaskConstraints {
                selector: "\u{53d1}\u{5e03}\u{7248}\u{672c}".to_owned(),
            }
        );
        assert_eq!(
            parse_intent("make task Deploy release depend on Build package"),
            MainAgentIntent::AddTaskDependency {
                selector: "Deploy release".to_owned(),
                depends_on_selector: "Build package".to_owned(),
            }
        );
        assert_eq!(
            parse_intent("remove dependency task Deploy release on Build package"),
            MainAgentIntent::RemoveTaskDependency {
                selector: "Deploy release".to_owned(),
                depends_on_selector: "Build package".to_owned(),
            }
        );
        assert_eq!(
            parse_intent(
                "\u{8bbe}\u{7f6e}\u{4efb}\u{52a1} \u{53d1}\u{5e03}\u{7248}\u{672c} \u{4f9d}\u{8d56} \u{6784}\u{5efa}\u{5305}"
            ),
            MainAgentIntent::AddTaskDependency {
                selector: "\u{53d1}\u{5e03}\u{7248}\u{672c}".to_owned(),
                depends_on_selector: "\u{6784}\u{5efa}\u{5305}".to_owned(),
            }
        );
        assert_eq!(
            parse_intent(
                "\u{53d6}\u{6d88}\u{4efb}\u{52a1} \u{53d1}\u{5e03}\u{7248}\u{672c} \u{4f9d}\u{8d56} \u{6784}\u{5efa}\u{5305}"
            ),
            MainAgentIntent::RemoveTaskDependency {
                selector: "\u{53d1}\u{5e03}\u{7248}\u{672c}".to_owned(),
                depends_on_selector: "\u{6784}\u{5efa}\u{5305}".to_owned(),
            }
        );
    }

    #[test]
    fn parses_task_note_intents() {
        assert_eq!(
            parse_intent("show notes for task Deploy release"),
            MainAgentIntent::ListTaskNotes {
                selector: "Deploy release".to_owned(),
            }
        );
        assert_eq!(
            parse_intent("list task Deploy release notes"),
            MainAgentIntent::ListTaskNotes {
                selector: "Deploy release".to_owned(),
            }
        );
        assert_eq!(
            parse_intent(
                "\u{67e5}\u{770b}\u{4efb}\u{52a1} \u{53d1}\u{5e03}\u{7248}\u{672c} \u{5907}\u{6ce8}"
            ),
            MainAgentIntent::ListTaskNotes {
                selector: "\u{53d1}\u{5e03}\u{7248}\u{672c}".to_owned(),
            }
        );
        assert_eq!(
            parse_intent("add note to task Deploy release: wait for staging approval"),
            MainAgentIntent::AddTaskNote {
                selector: "Deploy release".to_owned(),
                content: "wait for staging approval".to_owned(),
            }
        );
        assert_eq!(
            parse_intent(
                "\u{7ed9}\u{4efb}\u{52a1} \u{53d1}\u{5e03}\u{7248}\u{672c} \u{6dfb}\u{52a0}\u{5907}\u{6ce8}\u{ff1a}\u{7b49}\u{5f85} staging \u{5ba1}\u{6279}"
            ),
            MainAgentIntent::AddTaskNote {
                selector: "\u{53d1}\u{5e03}\u{7248}\u{672c}".to_owned(),
                content: "\u{7b49}\u{5f85} staging \u{5ba1}\u{6279}".to_owned(),
            }
        );
    }

    #[test]
    fn parses_requested_skill_intents() {
        assert_eq!(
            parse_intent("add skill to task Deploy release: github, browser"),
            MainAgentIntent::AddRequestedSkills {
                selector: "Deploy release".to_owned(),
                skill_names: vec!["github".to_owned(), "browser".to_owned()],
            }
        );
        assert_eq!(
            parse_intent("remove skill browser from task Deploy release"),
            MainAgentIntent::RemoveRequestedSkills {
                selector: "Deploy release".to_owned(),
                skill_names: vec!["browser".to_owned()],
            }
        );
        assert_eq!(
            parse_intent(
                "\u{7ed9}\u{4efb}\u{52a1} \u{53d1}\u{5e03}\u{7248}\u{672c} \u{6dfb}\u{52a0}\u{6280}\u{80fd}\u{ff1a}github"
            ),
            MainAgentIntent::AddRequestedSkills {
                selector: "\u{53d1}\u{5e03}\u{7248}\u{672c}".to_owned(),
                skill_names: vec!["github".to_owned()],
            }
        );
    }

    #[test]
    fn parses_skill_definition_intents() {
        assert_eq!(
            parse_intent(
                "create skill rust: Rust workspace maintenance; triggers cargo,rust; tools shell,filesystem; resource skills/rust"
            ),
            MainAgentIntent::CreateSkillDefinition {
                input: CreateSkill {
                    name: "rust".to_owned(),
                    description: "Rust workspace maintenance".to_owned(),
                    trigger_rules: vec!["cargo".to_owned(), "rust".to_owned()],
                    tool_subset: vec!["shell".to_owned(), "filesystem".to_owned()],
                    resource_path: Some("skills/rust".to_owned()),
                },
            }
        );
        assert_eq!(
            parse_intent("list skills"),
            MainAgentIntent::ListSkillDefinitions
        );
        assert_eq!(
            parse_intent(
                "update skill rust; triggers cargo,clippy; tools shell; resource skills/rust-v2"
            ),
            MainAgentIntent::UpdateSkillDefinition {
                selector: "rust".to_owned(),
                input: UpdateSkill {
                    name: None,
                    description: None,
                    trigger_rules: Some(vec!["cargo".to_owned(), "clippy".to_owned()]),
                    tool_subset: Some(vec!["shell".to_owned()]),
                    resource_path: Some(Some("skills/rust-v2".to_owned())),
                },
            }
        );
        assert_eq!(
            parse_intent("update skill rust; resource none"),
            MainAgentIntent::UpdateSkillDefinition {
                selector: "rust".to_owned(),
                input: UpdateSkill {
                    name: None,
                    description: None,
                    trigger_rules: None,
                    tool_subset: None,
                    resource_path: Some(None),
                },
            }
        );
        assert_eq!(
            parse_intent("delete skill rust"),
            MainAgentIntent::DeleteSkillDefinition {
                selector: "rust".to_owned()
            }
        );
        assert_eq!(
            parse_intent("remove skill browser from task Deploy release"),
            MainAgentIntent::RemoveRequestedSkills {
                selector: "Deploy release".to_owned(),
                skill_names: vec!["browser".to_owned()],
            }
        );
    }

    #[test]
    fn parses_task_resource_lock_intents() {
        assert_eq!(
            parse_intent("add resource lock to task Deploy release: repo:persistent-agent"),
            MainAgentIntent::AddResourceLock {
                selector: "Deploy release".to_owned(),
                resource_key: "repo:persistent-agent".to_owned(),
            }
        );
        assert_eq!(
            parse_intent("remove resource lock from task Deploy release: repo:persistent-agent"),
            MainAgentIntent::RemoveResourceLock {
                selector: "Deploy release".to_owned(),
                resource_key: "repo:persistent-agent".to_owned(),
            }
        );
        assert_eq!(
            parse_intent(
                "\u{7ed9}\u{4efb}\u{52a1} \u{53d1}\u{5e03}\u{7248}\u{672c} \u{6dfb}\u{52a0}\u{8d44}\u{6e90}\u{9501}\u{ff1a}repo:persistent-agent"
            ),
            MainAgentIntent::AddResourceLock {
                selector: "\u{53d1}\u{5e03}\u{7248}\u{672c}".to_owned(),
                resource_key: "repo:persistent-agent".to_owned(),
            }
        );
    }

    #[test]
    fn parses_clarification_intents() {
        assert_eq!(
            parse_intent("ask clarification for task Deploy release: Which environment?"),
            MainAgentIntent::RequestClarification {
                selector: "Deploy release".to_owned(),
                question: "Which environment?".to_owned(),
            }
        );
        assert_eq!(
            parse_intent(
                "\u{4efb}\u{52a1} \u{53d1}\u{5e03}\u{7248}\u{672c} \u{9700}\u{8981}\u{6f84}\u{6e05}\u{ff1a}\u{8981}\u{53d1}\u{5e03}\u{5230}\u{54ea}\u{4e2a}\u{73af}\u{5883}\u{ff1f}"
            ),
            MainAgentIntent::RequestClarification {
                selector: "\u{53d1}\u{5e03}\u{7248}\u{672c}".to_owned(),
                question:
                    "\u{8981}\u{53d1}\u{5e03}\u{5230}\u{54ea}\u{4e2a}\u{73af}\u{5883}\u{ff1f}"
                        .to_owned(),
            }
        );
    }

    #[test]
    fn parses_task_reply_intents() {
        assert_eq!(
            parse_intent("reply to task Deploy release: use production"),
            MainAgentIntent::ReplyToTask {
                selector: "Deploy release".to_owned(),
                content: "use production".to_owned(),
            }
        );
        assert_eq!(
            parse_intent("reply to blocked task: use production"),
            MainAgentIntent::ReplyToTask {
                selector: "blocked".to_owned(),
                content: "use production".to_owned(),
            }
        );
        assert_eq!(
            parse_intent(
                "answer for task Check release: the repo is oh-my-harness/Persistent-Agent"
            ),
            MainAgentIntent::ReplyToTask {
                selector: "Check release".to_owned(),
                content: "the repo is oh-my-harness/Persistent-Agent".to_owned(),
            }
        );
    }

    #[test]
    fn parses_memory_review_intents() {
        assert_eq!(
            parse_intent("remember repo: prefer cargo test before push"),
            MainAgentIntent::CreateMemory {
                scope: "repo".to_owned(),
                content: "prefer cargo test before push".to_owned(),
            }
        );
        assert_eq!(
            parse_intent("\u{8bb0}\u{4f4f}\u{ff1a}\u{4f18}\u{5148}\u{8fd0}\u{884c} cargo test"),
            MainAgentIntent::CreateMemory {
                scope: "repo".to_owned(),
                content: "\u{4f18}\u{5148}\u{8fd0}\u{884c} cargo test".to_owned(),
            }
        );
        assert_eq!(
            parse_intent("show memory candidates"),
            MainAgentIntent::ListMemories {
                filter: MemoryListFilter::Pending,
            }
        );
        assert_eq!(
            parse_intent("show memory candidates for task Deploy release"),
            MainAgentIntent::ListTaskMemories {
                selector: "Deploy release".to_owned(),
            }
        );
        assert_eq!(
            parse_intent("list task Deploy release memories"),
            MainAgentIntent::ListTaskMemories {
                selector: "Deploy release".to_owned(),
            }
        );
        assert_eq!(
            parse_intent(
                "\u{67e5}\u{770b}\u{4efb}\u{52a1} \u{53d1}\u{5e03}\u{7248}\u{672c} \u{8bb0}\u{5fc6}\u{5019}\u{9009}"
            ),
            MainAgentIntent::ListTaskMemories {
                selector: "\u{53d1}\u{5e03}\u{7248}\u{672c}".to_owned(),
            }
        );
        assert_eq!(
            parse_intent("list approved memories"),
            MainAgentIntent::ListMemories {
                filter: MemoryListFilter::Approved,
            }
        );
        assert_eq!(
            parse_intent("\u{5217}\u{51fa}\u{957f}\u{671f}\u{8bb0}\u{5fc6}\u{5168}\u{90e8}"),
            MainAgentIntent::ListMemories {
                filter: MemoryListFilter::All,
            }
        );
        assert_eq!(
            parse_intent("approve memory candidate: prefer cargo test before push"),
            MainAgentIntent::ApproveMemory {
                selector: "prefer cargo test before push".to_owned(),
            }
        );
        assert_eq!(
            parse_intent("approve all pending memories"),
            MainAgentIntent::BulkReviewMemories {
                status: MemoryStatus::Approved,
            }
        );
        assert_eq!(
            parse_intent("approve all memory candidates for task Deploy release"),
            MainAgentIntent::BulkReviewTaskMemories {
                selector: "Deploy release".to_owned(),
                status: MemoryStatus::Approved,
            }
        );
        assert_eq!(
            parse_intent("reject memory candidate noisy temporary note"),
            MainAgentIntent::RejectMemory {
                selector: "noisy temporary note".to_owned(),
            }
        );
        assert_eq!(
            parse_intent("reject all memory candidates"),
            MainAgentIntent::BulkReviewMemories {
                status: MemoryStatus::Rejected,
            }
        );
        assert_eq!(
            parse_intent("reject all memories for task Deploy release"),
            MainAgentIntent::BulkReviewTaskMemories {
                selector: "Deploy release".to_owned(),
                status: MemoryStatus::Rejected,
            }
        );
        assert_eq!(
            parse_intent("update memory candidate prefer cargo: prefer cargo test before push"),
            MainAgentIntent::UpdateMemory {
                selector: "prefer cargo".to_owned(),
                input: UpdateMemory {
                    scope: None,
                    content: Some("prefer cargo test before push".to_owned()),
                    confidence: None,
                },
            }
        );
        assert_eq!(
            parse_intent("delete memory candidate noisy temporary note"),
            MainAgentIntent::DeleteMemory {
                selector: "noisy temporary note".to_owned(),
            }
        );
        assert_eq!(
            parse_intent(
                "\u{91c7}\u{7eb3}\u{957f}\u{671f}\u{8bb0}\u{5fc6}\u{ff1a}\u{4f18}\u{5148}\u{8fd0}\u{884c} cargo test"
            ),
            MainAgentIntent::ApproveMemory {
                selector: "\u{4f18}\u{5148}\u{8fd0}\u{884c} cargo test".to_owned(),
            }
        );
    }

    #[tokio::test]
    async fn main_agent_can_create_task_with_requested_skills_by_conversation() -> anyhow::Result<()>
    {
        let db = Db::connect("sqlite::memory:").await?;
        let agent = MainAgent::new(db.clone());

        let response = agent
            .handle_user_message(MainAgentMessageInput {
                content: "create task: Check GitHub issues with skills github, shell".to_owned(),
            })
            .await?;
        let tasks = db.list_tasks().await?;
        let task = tasks
            .iter()
            .find(|task| task.title == "Check GitHub issues")
            .expect("created task");

        assert_eq!(response.changed_tasks.len(), 1);
        assert_eq!(task.requested_skills, vec!["github", "shell"]);
        assert!(response.assistant_message.content.contains("Created task"));

        Ok(())
    }

    #[tokio::test]
    async fn main_agent_enriches_github_issue_tasks_with_skill_and_resource_lock()
    -> anyhow::Result<()> {
        let db = Db::connect("sqlite::memory:").await?;
        let agent = MainAgent::new(db.clone());

        let response = agent
            .handle_user_message(MainAgentMessageInput {
                content: "create recurring task: check GitHub repo oh-my-harness/Persistent-Agent issues every 60 seconds".to_owned(),
            })
            .await?;
        let task = response.changed_tasks.first().expect("created task");
        let locks = db.list_task_resource_locks(task.id).await?;

        assert_eq!(task.requested_skills, vec!["github"]);
        assert_eq!(locks.len(), 1);
        assert_eq!(locks[0].resource_key, "repo:oh-my-harness/Persistent-Agent");
        assert!(
            response
                .assistant_message
                .content
                .contains("Requested skills: github")
        );
        assert!(
            response
                .assistant_message
                .content
                .contains("repo:oh-my-harness/Persistent-Agent")
        );

        Ok(())
    }

    #[test]
    fn infers_github_repository_refs_from_urls_and_chinese_text() {
        assert_eq!(
            extract_github_repository_refs(
                "\u{67e5}\u{770b} https://github.com/oh-my-harness/Persistent-Agent/issues \u{4ed3}\u{5e93} issue"
            ),
            vec!["oh-my-harness/Persistent-Agent"]
        );
        assert_eq!(
            infer_resource_locks_for_task_text(
                "\u{521b}\u{5efa}\u{5faa}\u{73af}\u{4efb}\u{52a1}\u{ff1a}\u{68c0}\u{67e5} oh-my-harness/Persistent-Agent \u{4ed3}\u{5e93} issue"
            ),
            vec!["repo:oh-my-harness/Persistent-Agent"]
        );
    }

    #[tokio::test]
    async fn main_agent_can_manage_skill_definitions_by_conversation() -> anyhow::Result<()> {
        let db = Db::connect("sqlite::memory:").await?;
        let agent = MainAgent::new(db.clone());

        let created = agent
            .handle_user_message(MainAgentMessageInput {
                content: "create skill rust: Rust workspace maintenance; triggers cargo,rust; tools shell,filesystem; resource skills/rust".to_owned(),
            })
            .await?;
        let skills = db.list_skills().await?;

        assert!(
            created
                .assistant_message
                .content
                .contains("Created skill 'rust'")
        );
        assert_eq!(skills.len(), 1);
        assert_eq!(skills[0].name, "rust");
        assert_eq!(skills[0].trigger_rules, vec!["cargo", "rust"]);
        assert_eq!(skills[0].tool_subset, vec!["shell", "filesystem"]);
        assert_eq!(skills[0].resource_path.as_deref(), Some("skills/rust"));

        let listed = agent
            .handle_user_message(MainAgentMessageInput {
                content: "list skills".to_owned(),
            })
            .await?;
        assert!(
            listed
                .assistant_message
                .content
                .contains("Rust workspace maintenance")
        );
        assert!(
            listed
                .assistant_message
                .content
                .contains("resource: skills/rust")
        );

        let updated = agent
            .handle_user_message(MainAgentMessageInput {
                content:
                    "update skill rust; triggers cargo,clippy; tools shell; resource skills/rust-v2"
                        .to_owned(),
            })
            .await?;
        let skills = db.list_skills().await?;

        assert!(
            updated
                .assistant_message
                .content
                .contains("Updated skill 'rust'")
        );
        assert_eq!(skills.len(), 1);
        assert_eq!(skills[0].trigger_rules, vec!["cargo", "clippy"]);
        assert_eq!(skills[0].tool_subset, vec!["shell"]);
        assert_eq!(skills[0].resource_path.as_deref(), Some("skills/rust-v2"));

        let cleared = agent
            .handle_user_message(MainAgentMessageInput {
                content: "update skill rust; resource none".to_owned(),
            })
            .await?;
        let skills = db.list_skills().await?;

        assert!(
            cleared
                .assistant_message
                .content
                .contains("Updated skill 'rust'")
        );
        assert_eq!(skills.len(), 1);
        assert_eq!(skills[0].resource_path, None);

        let deleted = agent
            .handle_user_message(MainAgentMessageInput {
                content: "delete skill rust".to_owned(),
            })
            .await?;
        let skills = db.list_skills().await?;

        assert!(
            deleted
                .assistant_message
                .content
                .contains("Deleted skill 'rust'")
        );
        assert!(skills.is_empty());

        Ok(())
    }

    #[tokio::test]
    async fn main_agent_uses_advisor_reply_when_configured() -> anyhow::Result<()> {
        let db = Db::connect("sqlite::memory:").await?;
        let contexts = Arc::new(StdMutex::new(Vec::new()));
        let agent = MainAgent::new_with_advisor(
            db.clone(),
            Arc::new(FixedAdvisor {
                reply: "Advisor: created and queued the task.".to_owned(),
                contexts: contexts.clone(),
            }),
        );

        let response = agent
            .handle_user_message(MainAgentMessageInput {
                content: "create task: Check GitHub issues".to_owned(),
            })
            .await?;
        let tasks = db.list_tasks().await?;
        let actions = db.list_global_actions().await?;

        assert_eq!(
            response.assistant_message.content,
            "Advisor: created and queued the task."
        );
        assert_eq!(tasks.len(), 1);
        assert_eq!(response.changed_tasks.len(), 1);
        let captured = contexts.lock().expect("advisor contexts lock");
        assert_eq!(captured.len(), 1);
        assert!(captured[0].deterministic_reply.contains("Created task"));
        assert_eq!(captured[0].changed_tasks.len(), 1);
        assert!(
            actions
                .iter()
                .any(|action| action.action_type == "llm_advisor_reply")
        );

        Ok(())
    }

    #[tokio::test]
    async fn main_agent_uses_llm_planner_for_unparsed_task_requests() -> anyhow::Result<()> {
        let db = Db::connect("sqlite::memory:").await?;
        let contexts = Arc::new(StdMutex::new(Vec::new()));
        let agent = MainAgent::new(db.clone()).with_planner(Arc::new(FixedPlanner {
            plan: Some(MainAgentPlan::CreateTask {
                title: "Watch release notes".to_owned(),
                description: "Monitor upstream release notes every two minutes.".to_owned(),
                task_type: TaskType::Recurring,
                priority: 4,
                interval_seconds: Some(120),
                requested_skills: vec!["network".to_owned()],
            }),
            contexts: contexts.clone(),
        }));

        let response = agent
            .handle_user_message(MainAgentMessageInput {
                content: "Could you keep an eye on upstream release notes every couple minutes?"
                    .to_owned(),
            })
            .await?;
        let tasks = db.list_tasks().await?;
        let actions = db.list_global_actions().await?;

        assert_eq!(tasks.len(), 1);
        assert_eq!(tasks[0].title, "Watch release notes");
        assert_eq!(tasks[0].task_type, TaskType::Recurring);
        assert_eq!(tasks[0].priority, 4);
        assert_eq!(tasks[0].requested_skills, vec!["network".to_owned()]);
        assert_eq!(response.changed_tasks.len(), 1);
        assert!(response.assistant_message.content.contains("Created task"));
        assert_eq!(contexts.lock().expect("planner contexts lock").len(), 1);
        assert!(
            actions
                .iter()
                .any(|action| action.action_type == "llm_planner_intent")
        );

        Ok(())
    }

    #[tokio::test]
    #[ignore = "requires DEEPSEEK_API_KEY and calls the real LLM through oh-my-harness AgentHarness"]
    async fn llm_harness_main_agent_smoke_plans_and_advises() -> anyhow::Result<()> {
        let Ok(api_key) = std::env::var("DEEPSEEK_API_KEY") else {
            eprintln!("Skipping smoke: DEEPSEEK_API_KEY is not set.");
            return Ok(());
        };
        if api_key.trim().is_empty() {
            eprintln!("Skipping smoke: DEEPSEEK_API_KEY is empty.");
            return Ok(());
        }

        let model = std::env::var("DEEPSEEK_MODEL").unwrap_or_else(|_| "deepseek-chat".to_owned());
        let mut config = MainAgentLlmConfig::deepseek(api_key, model);
        config.timeout_ms = 60_000;
        let harness_main_agent = Arc::new(OhMyHarnessMainAgentAdvisor::new(config));
        let db = Db::connect("sqlite::memory:").await?;
        let agent = MainAgent::new(db.clone())
            .with_planner(harness_main_agent.clone())
            .with_advisor(harness_main_agent);

        let response = agent
            .handle_user_message(MainAgentMessageInput {
                content: "The backlog should hold a work item titled exactly 'Main agent LLM smoke' with description 'from the main-agent harness smoke test'.".to_owned(),
            })
            .await?;
        let tasks = db.list_tasks().await?;
        let actions = db.list_global_actions().await?;

        assert_eq!(tasks.len(), 1);
        assert!(tasks[0].title.contains("Main agent LLM smoke"));
        assert_eq!(response.changed_tasks.len(), 1);
        assert!(
            actions
                .iter()
                .any(|action| action.action_type == "llm_planner_intent")
        );
        assert!(
            actions
                .iter()
                .any(|action| action.action_type == "llm_advisor_reply")
        );

        Ok(())
    }

    #[tokio::test]
    async fn main_agent_uses_llm_planner_for_task_state_changes() -> anyhow::Result<()> {
        let db = Db::connect("sqlite::memory:").await?;
        let task = db
            .create_task(
                CreateTask {
                    title: "Watch GitHub issues".to_owned(),
                    description: "Monitor repository issues".to_owned(),
                    task_type: TaskType::Recurring,
                    priority: 0,
                    requested_skills: Vec::new(),
                    schedule: Some(serde_json::json!({ "interval_seconds": 300 })),
                    created_by: "test".to_owned(),
                },
                "test",
            )
            .await?;
        let contexts = Arc::new(StdMutex::new(Vec::new()));
        let agent = MainAgent::new(db.clone()).with_planner(Arc::new(FixedPlanner {
            plan: Some(MainAgentPlan::PauseTask {
                selector: "Watch GitHub issues".to_owned(),
            }),
            contexts: contexts.clone(),
        }));

        let response = agent
            .handle_user_message(MainAgentMessageInput {
                content: "Could you put the GitHub issue watcher on hold for now?".to_owned(),
            })
            .await?;
        let updated = db.get_task(task.id).await?;

        assert_eq!(updated.status, TaskStatus::Paused);
        assert_eq!(response.changed_tasks.len(), 1);
        assert!(response.assistant_message.content.contains("Paused task"));
        let captured = contexts.lock().expect("planner contexts lock");
        assert_eq!(captured.len(), 1);
        assert_eq!(captured[0].task_snapshot.len(), 1);
        assert_eq!(captured[0].task_snapshot[0].title, "Watch GitHub issues");

        let delete_task = db
            .create_task(
                CreateTask {
                    title: "Remove stale work".to_owned(),
                    description: "No longer useful".to_owned(),
                    task_type: TaskType::OneOff,
                    priority: 0,
                    requested_skills: Vec::new(),
                    schedule: None,
                    created_by: "test".to_owned(),
                },
                "test",
            )
            .await?;
        let agent = MainAgent::new(db.clone()).with_planner(Arc::new(FixedPlanner {
            plan: Some(MainAgentPlan::DeleteTask {
                selector: "Remove stale work".to_owned(),
            }),
            contexts: Arc::new(StdMutex::new(Vec::new())),
        }));

        let response = agent
            .handle_user_message(MainAgentMessageInput {
                content: "Could you remove that stale item from the backlog?".to_owned(),
            })
            .await?;

        assert!(db.get_task(delete_task.id).await.is_err());
        assert_eq!(response.changed_tasks.len(), 1);
        assert!(response.assistant_message.content.contains("Deleted task"));

        Ok(())
    }

    #[tokio::test]
    async fn main_agent_uses_llm_planner_for_task_finish_states() -> anyhow::Result<()> {
        let db = Db::connect("sqlite::memory:").await?;
        let complete_task = db
            .create_task(
                CreateTask {
                    title: "Deploy release".to_owned(),
                    description: "Deploy the release".to_owned(),
                    task_type: TaskType::OneOff,
                    priority: 0,
                    requested_skills: Vec::new(),
                    schedule: None,
                    created_by: "test".to_owned(),
                },
                "test",
            )
            .await?;
        let fail_task = db
            .create_task(
                CreateTask {
                    title: "Publish package".to_owned(),
                    description: "Publish the package".to_owned(),
                    task_type: TaskType::OneOff,
                    priority: 0,
                    requested_skills: Vec::new(),
                    schedule: None,
                    created_by: "test".to_owned(),
                },
                "test",
            )
            .await?;
        let retry_task = db
            .create_task(
                CreateTask {
                    title: "Refresh deployment".to_owned(),
                    description: "Refresh the failed deployment".to_owned(),
                    task_type: TaskType::OneOff,
                    priority: 0,
                    requested_skills: Vec::new(),
                    schedule: None,
                    created_by: "test".to_owned(),
                },
                "test",
            )
            .await?;
        db.fail_task(retry_task.id, "Expired credentials", "test")
            .await?;

        let agent = MainAgent::new(db.clone()).with_planner(Arc::new(FixedPlanner {
            plan: Some(MainAgentPlan::CompleteTask {
                selector: "Deploy release".to_owned(),
                summary: "Deployed to production.".to_owned(),
            }),
            contexts: Arc::new(StdMutex::new(Vec::new())),
        }));
        let response = agent
            .handle_user_message(MainAgentMessageInput {
                content: "That deployment task is done.".to_owned(),
            })
            .await?;
        let completed = db.get_task(complete_task.id).await?;
        let actions = db.list_task_actions(complete_task.id).await?;

        assert_eq!(completed.status, TaskStatus::Completed);
        assert_eq!(
            completed.result_summary.as_deref(),
            Some("Deployed to production.")
        );
        assert!(response.assistant_message.content.contains("Marked task"));
        assert!(
            actions
                .iter()
                .any(|action| action.action_type == "complete_task")
        );

        let agent = MainAgent::new(db.clone()).with_planner(Arc::new(FixedPlanner {
            plan: Some(MainAgentPlan::FailTask {
                selector: "Publish package".to_owned(),
                error: "Registry rejected the token.".to_owned(),
            }),
            contexts: Arc::new(StdMutex::new(Vec::new())),
        }));
        let response = agent
            .handle_user_message(MainAgentMessageInput {
                content: "The package publish cannot be completed.".to_owned(),
            })
            .await?;
        let failed = db.get_task(fail_task.id).await?;
        let actions = db.list_task_actions(fail_task.id).await?;

        assert_eq!(failed.status, TaskStatus::Failed);
        assert_eq!(
            failed.result_summary.as_deref(),
            Some("Registry rejected the token.")
        );
        assert!(response.assistant_message.content.contains("failed"));
        assert!(
            actions
                .iter()
                .any(|action| action.action_type == "fail_task")
        );

        let agent = MainAgent::new(db.clone()).with_planner(Arc::new(FixedPlanner {
            plan: Some(MainAgentPlan::RetryTask {
                selector: "Refresh deployment".to_owned(),
                reason: "Credentials refreshed.".to_owned(),
            }),
            contexts: Arc::new(StdMutex::new(Vec::new())),
        }));
        let response = agent
            .handle_user_message(MainAgentMessageInput {
                content: "Try the failed deployment again.".to_owned(),
            })
            .await?;
        let retried = db.get_task(retry_task.id).await?;
        let actions = db.list_task_actions(retry_task.id).await?;

        assert_eq!(retried.status, TaskStatus::Queued);
        assert_eq!(
            retried.result_summary.as_deref(),
            Some("Credentials refreshed.")
        );
        assert!(response.assistant_message.content.contains("Requeued task"));
        assert!(actions.iter().any(|action| {
            action.action_type == "requeue_task_after_failure"
                && action.details["error"] == "Credentials refreshed."
        }));

        Ok(())
    }

    #[tokio::test]
    async fn main_agent_uses_llm_planner_for_task_detail_updates() -> anyhow::Result<()> {
        let db = Db::connect("sqlite::memory:").await?;
        let task = db
            .create_task(
                CreateTask {
                    title: "Deploy release".to_owned(),
                    description: "Old deployment instructions".to_owned(),
                    task_type: TaskType::OneOff,
                    priority: 0,
                    requested_skills: Vec::new(),
                    schedule: None,
                    created_by: "test".to_owned(),
                },
                "test",
            )
            .await?;
        let agent = MainAgent::new(db.clone()).with_planner(Arc::new(FixedPlanner {
            plan: Some(MainAgentPlan::UpdateTaskDetails {
                selector: "Deploy release".to_owned(),
                title: Some("Ship production release".to_owned()),
                description: Some("Deploy to production after staging approval.".to_owned()),
            }),
            contexts: Arc::new(StdMutex::new(Vec::new())),
        }));

        let response = agent
            .handle_user_message(MainAgentMessageInput {
                content: "Please clean up the deployment task name and details.".to_owned(),
            })
            .await?;
        let updated = db.get_task(task.id).await?;
        let actions = db.list_task_actions(task.id).await?;

        assert_eq!(updated.title, "Ship production release");
        assert_eq!(
            updated.description,
            "Deploy to production after staging approval."
        );
        assert_eq!(response.changed_tasks.len(), 1);
        assert!(response.assistant_message.content.contains("Updated task"));
        assert!(
            actions
                .iter()
                .any(|action| action.action_type == "update_task")
        );

        Ok(())
    }

    #[tokio::test]
    async fn main_agent_uses_llm_planner_for_priority_changes() -> anyhow::Result<()> {
        let db = Db::connect("sqlite::memory:").await?;
        let task = db
            .create_task(
                CreateTask {
                    title: "Prepare release notes".to_owned(),
                    description: "Draft release notes".to_owned(),
                    task_type: TaskType::OneOff,
                    priority: 0,
                    requested_skills: Vec::new(),
                    schedule: None,
                    created_by: "test".to_owned(),
                },
                "test",
            )
            .await?;
        let agent = MainAgent::new(db.clone()).with_planner(Arc::new(FixedPlanner {
            plan: Some(MainAgentPlan::ReprioritizeTask {
                selector: "Prepare release notes".to_owned(),
                priority: 9,
            }),
            contexts: Arc::new(StdMutex::new(Vec::new())),
        }));

        let response = agent
            .handle_user_message(MainAgentMessageInput {
                content: "Please make the release notes work much more urgent.".to_owned(),
            })
            .await?;
        let updated = db.get_task(task.id).await?;

        assert_eq!(updated.priority, 9);
        assert_eq!(response.changed_tasks.len(), 1);
        assert!(response.assistant_message.content.contains("priority to 9"));

        Ok(())
    }

    #[tokio::test]
    async fn main_agent_uses_llm_planner_for_queue_reordering() -> anyhow::Result<()> {
        let db = Db::connect("sqlite::memory:").await?;
        let task = db
            .create_task(
                CreateTask {
                    title: "Prepare changelog".to_owned(),
                    description: "Write changelog".to_owned(),
                    task_type: TaskType::OneOff,
                    priority: 0,
                    requested_skills: Vec::new(),
                    schedule: None,
                    created_by: "test".to_owned(),
                },
                "test",
            )
            .await?;
        let agent = MainAgent::new(db.clone()).with_planner(Arc::new(FixedPlanner {
            plan: Some(MainAgentPlan::ReorderTask {
                selector: "Prepare changelog".to_owned(),
                queue_position: 8,
            }),
            contexts: Arc::new(StdMutex::new(Vec::new())),
        }));

        let response = agent
            .handle_user_message(MainAgentMessageInput {
                content: "Could you move the changelog work later in the queue?".to_owned(),
            })
            .await?;
        let updated = db.get_task(task.id).await?;

        assert_eq!(updated.queue_position, 8);
        assert_eq!(response.changed_tasks.len(), 1);
        assert!(
            response
                .assistant_message
                .content
                .contains("queue position 8")
        );

        Ok(())
    }

    #[tokio::test]
    async fn main_agent_uses_llm_planner_for_run_task_now() -> anyhow::Result<()> {
        let db = Db::connect("sqlite::memory:").await?;
        let task = db
            .create_task(
                CreateTask {
                    title: "Deploy release".to_owned(),
                    description: "Deploy the release".to_owned(),
                    task_type: TaskType::OneOff,
                    priority: 0,
                    requested_skills: Vec::new(),
                    schedule: None,
                    created_by: "test".to_owned(),
                },
                "test",
            )
            .await?;
        let agent = MainAgent::new(db.clone()).with_planner(Arc::new(FixedPlanner {
            plan: Some(MainAgentPlan::RunTaskNow {
                selector: "Deploy release".to_owned(),
            }),
            contexts: Arc::new(StdMutex::new(Vec::new())),
        }));

        let response = agent
            .handle_user_message(MainAgentMessageInput {
                content: "Can you kick off the deployment work now?".to_owned(),
            })
            .await?;
        let updated = db.get_task(task.id).await?;

        assert!(response.scheduler_tick_requested);
        assert_eq!(response.changed_tasks.len(), 1);
        assert_eq!(updated.status, TaskStatus::Queued);
        assert_eq!(updated.queue_position, -1);
        assert!(updated.priority > task.priority);

        Ok(())
    }

    #[tokio::test]
    async fn main_agent_uses_llm_planner_for_run_next_task() -> anyhow::Result<()> {
        let db = Db::connect("sqlite::memory:").await?;
        let low_priority = db
            .create_task(
                CreateTask {
                    title: "Write notes".to_owned(),
                    description: "Write project notes".to_owned(),
                    task_type: TaskType::OneOff,
                    priority: 0,
                    requested_skills: Vec::new(),
                    schedule: None,
                    created_by: "test".to_owned(),
                },
                "test",
            )
            .await?;
        let high_priority = db
            .create_task(
                CreateTask {
                    title: "Fix release blocker".to_owned(),
                    description: "Fix the urgent blocker".to_owned(),
                    task_type: TaskType::OneOff,
                    priority: 5,
                    requested_skills: Vec::new(),
                    schedule: None,
                    created_by: "test".to_owned(),
                },
                "test",
            )
            .await?;
        let agent = MainAgent::new(db.clone()).with_planner(Arc::new(FixedPlanner {
            plan: Some(MainAgentPlan::RunNextTask),
            contexts: Arc::new(StdMutex::new(Vec::new())),
        }));

        let response = agent
            .handle_user_message(MainAgentMessageInput {
                content: "Please make progress on the queue.".to_owned(),
            })
            .await?;
        let updated_low_priority = db.get_task(low_priority.id).await?;
        let updated_high_priority = db.get_task(high_priority.id).await?;

        assert!(response.scheduler_tick_requested);
        assert_eq!(response.changed_tasks.len(), 1);
        assert_eq!(response.changed_tasks[0].id, high_priority.id);
        assert_eq!(updated_high_priority.queue_position, -1);
        assert!(updated_high_priority.priority > high_priority.priority);
        assert_eq!(
            updated_low_priority.queue_position,
            low_priority.queue_position
        );

        Ok(())
    }

    #[tokio::test]
    async fn main_agent_uses_llm_planner_for_task_type_conversion() -> anyhow::Result<()> {
        let db = Db::connect("sqlite::memory:").await?;
        let task = db
            .create_task(
                CreateTask {
                    title: "Check release feed".to_owned(),
                    description: "Check the release feed".to_owned(),
                    task_type: TaskType::OneOff,
                    priority: 0,
                    requested_skills: Vec::new(),
                    schedule: None,
                    created_by: "test".to_owned(),
                },
                "test",
            )
            .await?;
        let agent = MainAgent::new(db.clone()).with_planner(Arc::new(FixedPlanner {
            plan: Some(MainAgentPlan::ConvertTaskType {
                selector: "Check release feed".to_owned(),
                task_type: TaskType::Recurring,
                interval_seconds: Some(180),
            }),
            contexts: Arc::new(StdMutex::new(Vec::new())),
        }));

        let response = agent
            .handle_user_message(MainAgentMessageInput {
                content: "Please make that release feed work happen repeatedly.".to_owned(),
            })
            .await?;
        let updated = db.get_task(task.id).await?;

        assert_eq!(updated.task_type, TaskType::Recurring);
        assert_eq!(
            updated.schedule,
            Some(serde_json::json!({ "interval_seconds": 180 }))
        );
        assert_eq!(response.changed_tasks.len(), 1);
        assert!(
            response
                .assistant_message
                .content
                .contains("Converted task")
        );

        let agent = MainAgent::new(db.clone()).with_planner(Arc::new(FixedPlanner {
            plan: Some(MainAgentPlan::UpdateTaskSchedule {
                selector: "Check release feed".to_owned(),
                interval_seconds: 900,
            }),
            contexts: Arc::new(StdMutex::new(Vec::new())),
        }));

        let response = agent
            .handle_user_message(MainAgentMessageInput {
                content: "Please slow down that recurring release feed check.".to_owned(),
            })
            .await?;
        let updated = db.get_task(task.id).await?;

        assert_eq!(
            updated.schedule,
            Some(serde_json::json!({ "interval_seconds": 900 }))
        );
        assert_eq!(response.changed_tasks.len(), 1);
        assert!(
            response
                .assistant_message
                .content
                .contains("recurring interval to 900 seconds")
        );

        Ok(())
    }

    #[tokio::test]
    async fn main_agent_uses_llm_planner_for_task_dependencies() -> anyhow::Result<()> {
        let db = Db::connect("sqlite::memory:").await?;
        db.create_task(
            CreateTask {
                title: "Build package".to_owned(),
                description: "Build the package".to_owned(),
                task_type: TaskType::OneOff,
                priority: 0,
                requested_skills: Vec::new(),
                schedule: None,
                created_by: "test".to_owned(),
            },
            "test",
        )
        .await?;
        let deploy = db
            .create_task(
                CreateTask {
                    title: "Deploy package".to_owned(),
                    description: "Deploy after build".to_owned(),
                    task_type: TaskType::OneOff,
                    priority: 0,
                    requested_skills: Vec::new(),
                    schedule: None,
                    created_by: "test".to_owned(),
                },
                "test",
            )
            .await?;
        let agent = MainAgent::new(db.clone()).with_planner(Arc::new(FixedPlanner {
            plan: Some(MainAgentPlan::AddTaskDependency {
                selector: "Deploy package".to_owned(),
                depends_on_selector: "Build package".to_owned(),
            }),
            contexts: Arc::new(StdMutex::new(Vec::new())),
        }));

        let response = agent
            .handle_user_message(MainAgentMessageInput {
                content: "Deployment should wait until the build is done.".to_owned(),
            })
            .await?;
        let dependencies = db.list_task_dependencies(deploy.id).await?;

        assert_eq!(dependencies.len(), 1);
        assert!(
            response
                .assistant_message
                .content
                .contains("Added dependency")
        );

        Ok(())
    }

    #[tokio::test]
    async fn main_agent_uses_llm_planner_for_task_notes() -> anyhow::Result<()> {
        let db = Db::connect("sqlite::memory:").await?;
        let task = db
            .create_task(
                CreateTask {
                    title: "Deploy release".to_owned(),
                    description: "Deploy the release".to_owned(),
                    task_type: TaskType::OneOff,
                    priority: 0,
                    requested_skills: Vec::new(),
                    schedule: None,
                    created_by: "test".to_owned(),
                },
                "test",
            )
            .await?;
        let agent = MainAgent::new(db.clone()).with_planner(Arc::new(FixedPlanner {
            plan: Some(MainAgentPlan::AddTaskNote {
                selector: "Deploy release".to_owned(),
                content: "Wait for staging approval".to_owned(),
            }),
            contexts: Arc::new(StdMutex::new(Vec::new())),
        }));

        let response = agent
            .handle_user_message(MainAgentMessageInput {
                content: "Please keep this coordination caveat with the deployment item."
                    .to_owned(),
            })
            .await?;
        let notes = db.list_task_notes(task.id).await?;

        assert_eq!(notes.len(), 1);
        assert_eq!(notes[0].content, "Wait for staging approval");
        assert_eq!(response.changed_tasks.len(), 1);
        assert!(response.assistant_message.content.contains("Added note"));

        let agent = MainAgent::new(db.clone()).with_planner(Arc::new(FixedPlanner {
            plan: Some(MainAgentPlan::ListTaskNotes {
                selector: "Deploy release".to_owned(),
            }),
            contexts: Arc::new(StdMutex::new(Vec::new())),
        }));

        let response = agent
            .handle_user_message(MainAgentMessageInput {
                content: "What coordination notes do we have for deployment?".to_owned(),
            })
            .await?;
        let actions = db.list_task_actions(task.id).await?;

        assert!(
            response
                .assistant_message
                .content
                .contains("Task 'Deploy release' notes")
        );
        assert!(
            response
                .assistant_message
                .content
                .contains("Wait for staging approval")
        );
        assert!(
            actions
                .iter()
                .any(|action| action.action_type == "list_task_notes")
        );

        Ok(())
    }

    #[tokio::test]
    async fn main_agent_uses_llm_planner_for_resource_locks() -> anyhow::Result<()> {
        let db = Db::connect("sqlite::memory:").await?;
        let task = db
            .create_task(
                CreateTask {
                    title: "Fix repository issue".to_owned(),
                    description: "Work on the repository issue".to_owned(),
                    task_type: TaskType::OneOff,
                    priority: 0,
                    requested_skills: Vec::new(),
                    schedule: None,
                    created_by: "test".to_owned(),
                },
                "test",
            )
            .await?;
        let agent = MainAgent::new(db.clone()).with_planner(Arc::new(FixedPlanner {
            plan: Some(MainAgentPlan::AddResourceLock {
                selector: "Fix repository issue".to_owned(),
                resource_key: "repo:oh-my-harness/Persistent-Agent".to_owned(),
            }),
            contexts: Arc::new(StdMutex::new(Vec::new())),
        }));

        let response = agent
            .handle_user_message(MainAgentMessageInput {
                content: "Please reserve the shared repository context for that work.".to_owned(),
            })
            .await?;
        let locks = db.list_task_resource_locks(task.id).await?;

        assert_eq!(locks.len(), 1);
        assert_eq!(locks[0].resource_key, "repo:oh-my-harness/Persistent-Agent");
        assert_eq!(response.changed_tasks.len(), 1);
        assert!(
            response
                .assistant_message
                .content
                .contains("Added resource lock")
        );

        Ok(())
    }

    #[tokio::test]
    async fn main_agent_uses_llm_planner_for_clarification_requests() -> anyhow::Result<()> {
        let db = Db::connect("sqlite::memory:").await?;
        let task = db
            .create_task(
                CreateTask {
                    title: "Deploy release".to_owned(),
                    description: "Deploy the release".to_owned(),
                    task_type: TaskType::OneOff,
                    priority: 0,
                    requested_skills: Vec::new(),
                    schedule: None,
                    created_by: "test".to_owned(),
                },
                "test",
            )
            .await?;
        let agent = MainAgent::new(db.clone()).with_planner(Arc::new(FixedPlanner {
            plan: Some(MainAgentPlan::RequestClarification {
                selector: "Deploy release".to_owned(),
                question: "Which environment should receive the deployment?".to_owned(),
            }),
            contexts: Arc::new(StdMutex::new(Vec::new())),
        }));

        let response = agent
            .handle_user_message(MainAgentMessageInput {
                content: "Please ask the missing deployment question before proceeding.".to_owned(),
            })
            .await?;
        let updated = db.get_task(task.id).await?;
        let messages = db.list_task_conversation_messages(task.id, 20).await?;

        assert_eq!(updated.status, TaskStatus::WaitingForUser);
        assert_eq!(
            updated.blocked_reason.as_deref(),
            Some("Which environment should receive the deployment?")
        );
        assert!(messages.iter().any(|message| message.role == "assistant"
            && message.content == "Which environment should receive the deployment?"));
        assert_eq!(response.changed_tasks.len(), 1);

        Ok(())
    }

    #[tokio::test]
    async fn main_agent_uses_llm_planner_for_task_replies() -> anyhow::Result<()> {
        let db = Db::connect("sqlite::memory:").await?;
        let task = db
            .create_task(
                CreateTask {
                    title: "Deploy release".to_owned(),
                    description: "Deploy the release".to_owned(),
                    task_type: TaskType::OneOff,
                    priority: 0,
                    requested_skills: Vec::new(),
                    schedule: None,
                    created_by: "test".to_owned(),
                },
                "test",
            )
            .await?;
        db.set_task_status(
            task.id,
            TaskStatus::WaitingForUser,
            "test",
            Some("Which environment?"),
        )
        .await?;
        let agent = MainAgent::new(db.clone()).with_planner(Arc::new(FixedPlanner {
            plan: Some(MainAgentPlan::ReplyToTask {
                selector: "Deploy release".to_owned(),
                content: "Use production.".to_owned(),
            }),
            contexts: Arc::new(StdMutex::new(Vec::new())),
        }));

        let response = agent
            .handle_user_message(MainAgentMessageInput {
                content: "Please send the answer back to the deployment work.".to_owned(),
            })
            .await?;
        let updated = db.get_task(task.id).await?;
        let messages = db.list_task_conversation_messages(task.id, 20).await?;
        let actions = db.list_task_actions(task.id).await?;

        assert_eq!(updated.status, TaskStatus::Queued);
        assert!(updated.blocked_reason.is_none());
        assert_eq!(response.changed_tasks.len(), 1);
        assert!(response.scheduler_tick_requested);
        assert!(
            response
                .assistant_message
                .content
                .contains("requested a scheduler scan")
        );
        assert!(
            messages
                .iter()
                .any(|message| message.role == "user" && message.content == "Use production.")
        );
        assert!(messages.iter().any(|message| {
            message.role == "assistant"
                && message
                    .content
                    .contains("moved this task back to the queue")
        }));
        assert!(
            actions
                .iter()
                .any(|action| action.action_type == "reply_to_task")
        );

        Ok(())
    }

    #[tokio::test]
    async fn main_agent_uses_llm_planner_for_read_oriented_actions() -> anyhow::Result<()> {
        let db = Db::connect("sqlite::memory:").await?;
        let task = db
            .create_task(
                CreateTask {
                    title: "Deploy release".to_owned(),
                    description: "Deploy the release".to_owned(),
                    task_type: TaskType::OneOff,
                    priority: 0,
                    requested_skills: Vec::new(),
                    schedule: None,
                    created_by: "test".to_owned(),
                },
                "test",
            )
            .await?;
        db.record_task_artifact(
            task.id,
            None,
            "release-notes",
            "markdown",
            "artifact://release-notes",
            Some("Generated release notes"),
        )
        .await?;
        let attempt = db
            .create_attempt(
                task.id,
                TaskStatus::Completed,
                Some("published release notes"),
            )
            .await?;
        db.record_attempt_event(
            attempt.id,
            task.id,
            "worker_completed",
            "Worker published release notes.",
            serde_json::json!({ "summary": "published release notes" }),
        )
        .await?;
        db.add_conversation_message(
            task.conversation_id.expect("task conversation"),
            Some(task.id),
            "assistant",
            "Which environment?",
        )
        .await?;

        let agent = MainAgent::new(db.clone()).with_planner(Arc::new(FixedPlanner {
            plan: Some(MainAgentPlan::ListTaskArtifacts {
                selector: "Deploy release".to_owned(),
            }),
            contexts: Arc::new(StdMutex::new(Vec::new())),
        }));

        let response = agent
            .handle_user_message(MainAgentMessageInput {
                content: "Can you show me the deliverables for that deployment work?".to_owned(),
            })
            .await?;

        assert!(response.assistant_message.content.contains("release-notes"));
        assert!(
            response
                .assistant_message
                .content
                .contains("artifact://release-notes")
        );

        let agent = MainAgent::new(db.clone()).with_planner(Arc::new(FixedPlanner {
            plan: Some(MainAgentPlan::InspectWorkspaceFile {
                path: "Cargo.toml".to_owned(),
            }),
            contexts: Arc::new(StdMutex::new(Vec::new())),
        }));

        let response = agent
            .handle_user_message(MainAgentMessageInput {
                content: "Please look at the root manifest for context.".to_owned(),
            })
            .await?;
        let actions = db.list_global_actions().await?;

        assert!(response.assistant_message.content.contains("Cargo.toml"));
        assert!(actions.iter().any(|action| {
            action.action_type == "inspect_workspace_file" && action.details["path"] == "Cargo.toml"
        }));

        let agent = MainAgent::new(db.clone()).with_planner(Arc::new(FixedPlanner {
            plan: Some(MainAgentPlan::InspectWorkspaceDirectory {
                path: ".".to_owned(),
            }),
            contexts: Arc::new(StdMutex::new(Vec::new())),
        }));

        let response = agent
            .handle_user_message(MainAgentMessageInput {
                content: "Please show me the crate layout.".to_owned(),
            })
            .await?;
        let actions = db.list_global_actions().await?;

        assert!(response.assistant_message.content.contains("Directory: ."));
        assert!(actions.iter().any(|action| {
            action.action_type == "inspect_workspace_directory" && action.details["path"] == "."
        }));

        let agent = MainAgent::new(db.clone()).with_planner(Arc::new(FixedPlanner {
            plan: Some(MainAgentPlan::ExplainTaskPool),
            contexts: Arc::new(StdMutex::new(Vec::new())),
        }));

        let response = agent
            .handle_user_message(MainAgentMessageInput {
                content: "Can you explain what is going on overall?".to_owned(),
            })
            .await?;

        assert!(
            response
                .assistant_message
                .content
                .contains("Execution state")
        );

        let agent = MainAgent::new(db.clone()).with_planner(Arc::new(FixedPlanner {
            plan: Some(MainAgentPlan::ShowSchedulerState),
            contexts: Arc::new(StdMutex::new(Vec::new())),
        }));

        let response = agent
            .handle_user_message(MainAgentMessageInput {
                content: "What is the worker scheduler doing right now?".to_owned(),
            })
            .await?;
        let actions = db.list_global_actions().await?;

        assert!(response.assistant_message.content.contains("Next runnable"));
        assert!(
            actions
                .iter()
                .any(|action| action.action_type == "inspect_scheduler_state")
        );

        let agent = MainAgent::new(db.clone()).with_planner(Arc::new(FixedPlanner {
            plan: Some(MainAgentPlan::RecommendNextAction),
            contexts: Arc::new(StdMutex::new(Vec::new())),
        }));

        let response = agent
            .handle_user_message(MainAgentMessageInput {
                content: "How should we proceed?".to_owned(),
            })
            .await?;
        let actions = db.list_global_actions().await?;

        assert!(
            response
                .assistant_message
                .content
                .contains("Recommended next action")
        );
        assert!(
            actions
                .iter()
                .any(|action| action.action_type == "recommend_next_action")
        );

        let dependency = db
            .create_task(
                CreateTask {
                    title: "Build package".to_owned(),
                    description: "Build package before deployment".to_owned(),
                    task_type: TaskType::OneOff,
                    priority: 0,
                    requested_skills: Vec::new(),
                    schedule: None,
                    created_by: "test".to_owned(),
                },
                "test",
            )
            .await?;
        db.add_task_dependency(task.id, dependency.id, "test")
            .await?;
        db.add_task_resource_lock(task.id, "repo:oh-my-harness/Persistent-Agent", "test")
            .await?;
        let agent = MainAgent::new(db.clone()).with_planner(Arc::new(FixedPlanner {
            plan: Some(MainAgentPlan::ListTaskConstraints {
                selector: "Deploy release".to_owned(),
            }),
            contexts: Arc::new(StdMutex::new(Vec::new())),
        }));

        let response = agent
            .handle_user_message(MainAgentMessageInput {
                content: "What dependencies and resource locks constrain deployment?".to_owned(),
            })
            .await?;
        let actions = db.list_task_actions(task.id).await?;

        assert!(
            response
                .assistant_message
                .content
                .contains("Task 'Deploy release' constraints")
        );
        assert!(response.assistant_message.content.contains("Build package"));
        assert!(
            response
                .assistant_message
                .content
                .contains("repo:oh-my-harness/Persistent-Agent")
        );
        assert!(
            actions
                .iter()
                .any(|action| action.action_type == "list_task_constraints")
        );

        db.set_task_status(
            task.id,
            TaskStatus::WaitingForUser,
            "test",
            Some("Need deployment target"),
        )
        .await?;
        let agent = MainAgent::new(db.clone()).with_planner(Arc::new(FixedPlanner {
            plan: Some(MainAgentPlan::ListWaitingForUserTasks),
            contexts: Arc::new(StdMutex::new(Vec::new())),
        }));

        let response = agent
            .handle_user_message(MainAgentMessageInput {
                content: "What work is waiting on me?".to_owned(),
            })
            .await?;
        let actions = db.list_global_actions().await?;

        assert!(
            response
                .assistant_message
                .content
                .contains("Deploy release")
        );
        assert!(
            response
                .assistant_message
                .content
                .contains("Which environment?")
        );
        assert!(
            actions
                .iter()
                .any(|action| action.action_type == "list_waiting_for_user_tasks")
        );

        db.convert_task_type(
            task.id,
            TaskType::Recurring,
            Some(serde_json::json!({ "interval_seconds": 300 })),
            "test",
        )
        .await?;
        db.complete_task(task.id, "scheduled next run", "test")
            .await?;
        let agent = MainAgent::new(db.clone()).with_planner(Arc::new(FixedPlanner {
            plan: Some(MainAgentPlan::ListWaitingForScheduleTasks),
            contexts: Arc::new(StdMutex::new(Vec::new())),
        }));

        let response = agent
            .handle_user_message(MainAgentMessageInput {
                content: "Which recurring work is waiting for schedule?".to_owned(),
            })
            .await?;
        let actions = db.list_global_actions().await?;

        assert!(
            response
                .assistant_message
                .content
                .contains("Deploy release")
        );
        assert!(response.assistant_message.content.contains("next_run_at"));
        assert!(
            actions
                .iter()
                .any(|action| action.action_type == "list_waiting_for_schedule_tasks")
        );

        let agent = MainAgent::new(db.clone()).with_planner(Arc::new(FixedPlanner {
            plan: Some(MainAgentPlan::ListTasksByStatus {
                status: TaskStatus::WaitingForSchedule,
            }),
            contexts: Arc::new(StdMutex::new(Vec::new())),
        }));

        let response = agent
            .handle_user_message(MainAgentMessageInput {
                content: "Could you use the lifecycle status filter?".to_owned(),
            })
            .await?;
        let actions = db.list_global_actions().await?;

        assert!(
            response
                .assistant_message
                .content
                .contains("Deploy release")
        );
        assert!(
            actions
                .iter()
                .any(|action| action.action_type == "list_tasks_by_status")
        );

        let agent = MainAgent::new(db.clone()).with_planner(Arc::new(FixedPlanner {
            plan: Some(MainAgentPlan::ListTaskHistory {
                selector: "Deploy release".to_owned(),
            }),
            contexts: Arc::new(StdMutex::new(Vec::new())),
        }));

        let response = agent
            .handle_user_message(MainAgentMessageInput {
                content: "Show me what happened during that deployment task.".to_owned(),
            })
            .await?;
        let actions = db.list_task_actions(task.id).await?;

        assert!(
            response
                .assistant_message
                .content
                .contains("worker_completed")
        );
        assert!(
            actions
                .iter()
                .any(|action| action.action_type == "list_task_history")
        );

        let agent = MainAgent::new(db.clone()).with_planner(Arc::new(FixedPlanner {
            plan: Some(MainAgentPlan::ShowTaskLatestResult {
                selector: "Deploy release".to_owned(),
            }),
            contexts: Arc::new(StdMutex::new(Vec::new())),
        }));

        let response = agent
            .handle_user_message(MainAgentMessageInput {
                content: "What was the latest deployment outcome?".to_owned(),
            })
            .await?;
        let actions = db.list_task_actions(task.id).await?;

        assert!(response.assistant_message.content.contains("latest result"));
        assert!(
            actions
                .iter()
                .any(|action| action.action_type == "show_task_latest_result")
        );

        let follow_up = db
            .create_task(
                CreateTask {
                    title: "Verify production deploy".to_owned(),
                    description: "Check production after deployment".to_owned(),
                    task_type: TaskType::OneOff,
                    priority: task.priority,
                    requested_skills: Vec::new(),
                    schedule: None,
                    created_by: "worker".to_owned(),
                },
                "worker",
            )
            .await?;
        db.record_action(
            Some(task.id),
            "worker",
            "create_follow_up_task",
            serde_json::json!({
                "follow_up_task_id": follow_up.id,
                "title": follow_up.title,
            }),
        )
        .await?;
        let agent = MainAgent::new(db.clone()).with_planner(Arc::new(FixedPlanner {
            plan: Some(MainAgentPlan::ListTaskFollowUps {
                selector: "Deploy release".to_owned(),
            }),
            contexts: Arc::new(StdMutex::new(Vec::new())),
        }));

        let response = agent
            .handle_user_message(MainAgentMessageInput {
                content: "What follow-up work came out of deployment?".to_owned(),
            })
            .await?;
        let actions = db.list_task_actions(task.id).await?;

        assert!(
            response
                .assistant_message
                .content
                .contains("Verify production deploy")
        );
        assert!(
            actions
                .iter()
                .any(|action| action.action_type == "list_task_follow_ups")
        );

        let agent = MainAgent::new(db.clone()).with_planner(Arc::new(FixedPlanner {
            plan: Some(MainAgentPlan::ListTaskConversation {
                selector: "Deploy release".to_owned(),
            }),
            contexts: Arc::new(StdMutex::new(Vec::new())),
        }));

        let response = agent
            .handle_user_message(MainAgentMessageInput {
                content: "Show me the conversation thread for deployment.".to_owned(),
            })
            .await?;
        let actions = db.list_task_actions(task.id).await?;

        assert!(
            response
                .assistant_message
                .content
                .contains("Which environment?")
        );
        assert!(
            actions
                .iter()
                .any(|action| action.action_type == "list_task_conversation")
        );

        Ok(())
    }

    #[tokio::test]
    async fn main_agent_uses_llm_planner_for_memory_review() -> anyhow::Result<()> {
        let db = Db::connect("sqlite::memory:").await?;
        let memory = db
            .create_memory(
                persistent_agent_domain::CreateMemory {
                    scope: "repo".to_owned(),
                    content: "Prefer cargo test before pushing.".to_owned(),
                    source_task_id: None,
                    status: MemoryStatus::Pending,
                    confidence: 0.91,
                },
                "test",
            )
            .await?;
        let agent = MainAgent::new(db.clone()).with_planner(Arc::new(FixedPlanner {
            plan: Some(MainAgentPlan::ApproveMemory {
                selector: "cargo test before pushing".to_owned(),
            }),
            contexts: Arc::new(StdMutex::new(Vec::new())),
        }));

        let response = agent
            .handle_user_message(MainAgentMessageInput {
                content: "That remembered testing preference looks good.".to_owned(),
            })
            .await?;
        let approved = db.get_memory(memory.id).await?;

        assert_eq!(approved.status, MemoryStatus::Approved);
        assert!(
            response
                .assistant_message
                .content
                .contains("Approved memory")
        );

        let agent = MainAgent::new(db.clone()).with_planner(Arc::new(FixedPlanner {
            plan: Some(MainAgentPlan::ListMemories {
                filter: MemoryListFilter::Approved,
            }),
            contexts: Arc::new(StdMutex::new(Vec::new())),
        }));

        let response = agent
            .handle_user_message(MainAgentMessageInput {
                content: "Show me the useful remembered preferences.".to_owned(),
            })
            .await?;

        assert!(
            response
                .assistant_message
                .content
                .contains("Approved memories")
        );
        assert!(
            response
                .assistant_message
                .content
                .contains("Prefer cargo test before pushing.")
        );

        let task = db
            .create_task(
                CreateTask {
                    title: "Deploy release".to_owned(),
                    description: "Deploy the release".to_owned(),
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
            persistent_agent_domain::CreateMemory {
                scope: "repo".to_owned(),
                content: "Deployment needs release notes checked.".to_owned(),
                source_task_id: Some(task.id),
                status: MemoryStatus::Approved,
                confidence: 0.77,
            },
            "worker",
        )
        .await?;
        let agent = MainAgent::new(db.clone()).with_planner(Arc::new(FixedPlanner {
            plan: Some(MainAgentPlan::ListTaskMemories {
                selector: "Deploy release".to_owned(),
            }),
            contexts: Arc::new(StdMutex::new(Vec::new())),
        }));

        let response = agent
            .handle_user_message(MainAgentMessageInput {
                content: "What reusable lesson came out of that deployment work?".to_owned(),
            })
            .await?;
        let actions = db.list_task_actions(task.id).await?;

        assert!(
            response
                .assistant_message
                .content
                .contains("Task 'Deploy release' memory candidates")
        );
        assert!(
            response
                .assistant_message
                .content
                .contains("Deployment needs release notes checked.")
        );
        assert!(
            actions
                .iter()
                .any(|action| action.action_type == "list_task_memories")
        );

        let agent = MainAgent::new(db.clone()).with_planner(Arc::new(FixedPlanner {
            plan: Some(MainAgentPlan::UpdateMemory {
                selector: "cargo test before pushing".to_owned(),
                input: UpdateMemory {
                    scope: None,
                    content: Some("Prefer cargo test --workspace before pushing.".to_owned()),
                    confidence: None,
                },
            }),
            contexts: Arc::new(StdMutex::new(Vec::new())),
        }));

        let response = agent
            .handle_user_message(MainAgentMessageInput {
                content: "Tighten that remembered testing preference.".to_owned(),
            })
            .await?;
        let updated = db.get_memory(memory.id).await?;

        assert_eq!(
            updated.content,
            "Prefer cargo test --workspace before pushing."
        );
        assert!(
            response
                .assistant_message
                .content
                .contains("Updated memory")
        );

        let batch_memory = db
            .create_memory(
                persistent_agent_domain::CreateMemory {
                    scope: "repo".to_owned(),
                    content: "Prefer small focused commits.".to_owned(),
                    source_task_id: None,
                    status: MemoryStatus::Pending,
                    confidence: 0.82,
                },
                "test",
            )
            .await?;
        let agent = MainAgent::new(db.clone()).with_planner(Arc::new(FixedPlanner {
            plan: Some(MainAgentPlan::BulkReviewMemories {
                status: MemoryStatus::Approved,
            }),
            contexts: Arc::new(StdMutex::new(Vec::new())),
        }));

        let response = agent
            .handle_user_message(MainAgentMessageInput {
                content: "Those suggested lessons all look useful.".to_owned(),
            })
            .await?;
        let batch_memory = db.get_memory(batch_memory.id).await?;

        assert_eq!(batch_memory.status, MemoryStatus::Approved);
        assert!(response.assistant_message.content.contains("Approved 1"));

        let task_memory_a = db
            .create_memory(
                persistent_agent_domain::CreateMemory {
                    scope: "repo".to_owned(),
                    content: "Deployment tasks should run smoke tests first.".to_owned(),
                    source_task_id: Some(task.id),
                    status: MemoryStatus::Pending,
                    confidence: 0.84,
                },
                "worker",
            )
            .await?;
        let task_memory_b = db
            .create_memory(
                persistent_agent_domain::CreateMemory {
                    scope: "repo".to_owned(),
                    content: "Deployment tasks should record rollback steps.".to_owned(),
                    source_task_id: Some(task.id),
                    status: MemoryStatus::Pending,
                    confidence: 0.81,
                },
                "worker",
            )
            .await?;
        let unrelated_memory = db
            .create_memory(
                persistent_agent_domain::CreateMemory {
                    scope: "repo".to_owned(),
                    content: "Unrelated pending candidate should stay pending.".to_owned(),
                    source_task_id: None,
                    status: MemoryStatus::Pending,
                    confidence: 0.8,
                },
                "worker",
            )
            .await?;
        let agent = MainAgent::new(db.clone()).with_planner(Arc::new(FixedPlanner {
            plan: Some(MainAgentPlan::BulkReviewTaskMemories {
                selector: "Deploy release".to_owned(),
                status: MemoryStatus::Approved,
            }),
            contexts: Arc::new(StdMutex::new(Vec::new())),
        }));

        let response = agent
            .handle_user_message(MainAgentMessageInput {
                content: "Adopt the deployment-specific lessons.".to_owned(),
            })
            .await?;
        let task_memory_a = db.get_memory(task_memory_a.id).await?;
        let task_memory_b = db.get_memory(task_memory_b.id).await?;
        let unrelated_memory = db.get_memory(unrelated_memory.id).await?;
        let actions = db.list_task_actions(task.id).await?;

        assert_eq!(task_memory_a.status, MemoryStatus::Approved);
        assert_eq!(task_memory_b.status, MemoryStatus::Approved);
        assert_eq!(unrelated_memory.status, MemoryStatus::Pending);
        assert!(
            response
                .assistant_message
                .content
                .contains("Approved 2 pending memory candidate(s) from 'Deploy release'")
        );
        assert!(actions.iter().any(|action| {
            action.action_type == "bulk_set_task_memory_status"
                && action.details["status"] == "approved"
                && action.details["count"] == 2
        }));

        let agent = MainAgent::new(db.clone()).with_planner(Arc::new(FixedPlanner {
            plan: Some(MainAgentPlan::DeleteMemory {
                selector: "cargo test --workspace".to_owned(),
            }),
            contexts: Arc::new(StdMutex::new(Vec::new())),
        }));

        let response = agent
            .handle_user_message(MainAgentMessageInput {
                content: "Remove that remembered testing preference.".to_owned(),
            })
            .await?;
        let memories = db.list_memories().await?;

        assert!(!memories.iter().any(|candidate| candidate.id == memory.id));
        assert!(
            response
                .assistant_message
                .content
                .contains("Deleted memory")
        );

        Ok(())
    }

    #[tokio::test]
    async fn main_agent_uses_llm_planner_for_memory_creation() -> anyhow::Result<()> {
        let db = Db::connect("sqlite::memory:").await?;
        let agent = MainAgent::new(db.clone()).with_planner(Arc::new(FixedPlanner {
            plan: Some(MainAgentPlan::CreateMemory {
                scope: "repo".to_owned(),
                content: "Prefer cargo test before pushing.".to_owned(),
            }),
            contexts: Arc::new(StdMutex::new(Vec::new())),
        }));

        let response = agent
            .handle_user_message(MainAgentMessageInput {
                content: "Please remember that testing preference for future work.".to_owned(),
            })
            .await?;
        let memories = db.list_memories().await?;

        assert_eq!(memories.len(), 1);
        assert_eq!(memories[0].scope, "repo");
        assert_eq!(memories[0].status, MemoryStatus::Approved);
        assert_eq!(memories[0].confidence, 1.0);
        assert!(
            response
                .assistant_message
                .content
                .contains("Remembered [repo]")
        );

        Ok(())
    }

    #[tokio::test]
    async fn main_agent_uses_llm_planner_for_skill_listing() -> anyhow::Result<()> {
        let db = Db::connect("sqlite::memory:").await?;
        db.create_skill(
            CreateSkill {
                name: "rust".to_owned(),
                description: "Rust workspace maintenance".to_owned(),
                trigger_rules: vec!["cargo".to_owned()],
                tool_subset: vec!["shell".to_owned()],
                resource_path: None,
            },
            "test",
        )
        .await?;
        let agent = MainAgent::new(db.clone()).with_planner(Arc::new(FixedPlanner {
            plan: Some(MainAgentPlan::ListSkillDefinitions),
            contexts: Arc::new(StdMutex::new(Vec::new())),
        }));

        let response = agent
            .handle_user_message(MainAgentMessageInput {
                content: "Which reusable capabilities are available?".to_owned(),
            })
            .await?;

        assert!(response.assistant_message.content.contains("rust"));
        assert!(
            response
                .assistant_message
                .content
                .contains("Rust workspace maintenance")
        );

        Ok(())
    }

    #[tokio::test]
    async fn main_agent_uses_llm_planner_for_requested_skill_changes() -> anyhow::Result<()> {
        let db = Db::connect("sqlite::memory:").await?;
        let task = db
            .create_task(
                CreateTask {
                    title: "Investigate GitHub bug".to_owned(),
                    description: "Inspect a repository issue".to_owned(),
                    task_type: TaskType::OneOff,
                    priority: 0,
                    requested_skills: Vec::new(),
                    schedule: None,
                    created_by: "test".to_owned(),
                },
                "test",
            )
            .await?;
        let agent = MainAgent::new(db.clone()).with_planner(Arc::new(FixedPlanner {
            plan: Some(MainAgentPlan::AddRequestedSkills {
                selector: "Investigate GitHub bug".to_owned(),
                skill_names: vec!["github".to_owned(), "shell".to_owned()],
            }),
            contexts: Arc::new(StdMutex::new(Vec::new())),
        }));

        let response = agent
            .handle_user_message(MainAgentMessageInput {
                content: "Make sure the bug investigation can use GitHub and shell skills."
                    .to_owned(),
            })
            .await?;
        let updated = db.get_task(task.id).await?;

        assert_eq!(
            updated.requested_skills,
            vec!["github".to_owned(), "shell".to_owned()]
        );
        assert_eq!(response.changed_tasks.len(), 1);
        assert!(
            response
                .assistant_message
                .content
                .contains("Updated requested skills")
        );

        Ok(())
    }

    #[tokio::test]
    async fn main_agent_uses_llm_planner_for_skill_definition_creation() -> anyhow::Result<()> {
        let db = Db::connect("sqlite::memory:").await?;
        let agent = MainAgent::new(db.clone()).with_planner(Arc::new(FixedPlanner {
            plan: Some(MainAgentPlan::CreateSkillDefinition {
                input: CreateSkill {
                    name: "github".to_owned(),
                    description: "Work with GitHub issues and pull requests".to_owned(),
                    trigger_rules: vec!["github".to_owned(), "issue".to_owned()],
                    tool_subset: vec!["github".to_owned(), "network".to_owned()],
                    resource_path: Some("skills/github".to_owned()),
                },
            }),
            contexts: Arc::new(StdMutex::new(Vec::new())),
        }));

        let response = agent
            .handle_user_message(MainAgentMessageInput {
                content: "Could you make GitHub reusable for issue and pull request work?"
                    .to_owned(),
            })
            .await?;
        let skills = db.list_skills().await?;
        let actions = db.list_global_actions().await?;

        assert_eq!(skills.len(), 1);
        assert_eq!(skills[0].name, "github");
        assert_eq!(
            skills[0].tool_subset,
            vec!["github".to_owned(), "network".to_owned()]
        );
        assert_eq!(skills[0].resource_path.as_deref(), Some("skills/github"));
        assert!(response.assistant_message.content.contains("Created skill"));
        assert!(
            actions
                .iter()
                .any(|action| action.action_type == "llm_planner_intent")
        );

        Ok(())
    }

    #[tokio::test]
    async fn main_agent_uses_llm_planner_for_skill_definition_updates() -> anyhow::Result<()> {
        let db = Db::connect("sqlite::memory:").await?;
        db.create_skill(
            CreateSkill {
                name: "github".to_owned(),
                description: "Old GitHub skill".to_owned(),
                trigger_rules: vec!["github".to_owned()],
                tool_subset: vec!["network".to_owned()],
                resource_path: Some("skills/github".to_owned()),
            },
            "test",
        )
        .await?;
        let agent = MainAgent::new(db.clone()).with_planner(Arc::new(FixedPlanner {
            plan: Some(MainAgentPlan::UpdateSkillDefinition {
                selector: "github".to_owned(),
                input: UpdateSkill {
                    name: None,
                    description: Some("GitHub issue and pull request work".to_owned()),
                    trigger_rules: Some(vec!["github".to_owned(), "issue".to_owned()]),
                    tool_subset: Some(vec!["github".to_owned(), "network".to_owned()]),
                    resource_path: Some(None),
                },
            }),
            contexts: Arc::new(StdMutex::new(Vec::new())),
        }));

        let response = agent
            .handle_user_message(MainAgentMessageInput {
                content: "Could you tune that reusable capability for issue work?".to_owned(),
            })
            .await?;
        let skills = db.list_skills().await?;

        assert_eq!(skills.len(), 1);
        assert_eq!(skills[0].description, "GitHub issue and pull request work");
        assert_eq!(
            skills[0].trigger_rules,
            vec!["github".to_owned(), "issue".to_owned()]
        );
        assert_eq!(
            skills[0].tool_subset,
            vec!["github".to_owned(), "network".to_owned()]
        );
        assert_eq!(skills[0].resource_path, None);
        assert!(response.assistant_message.content.contains("Updated skill"));

        Ok(())
    }

    #[tokio::test]
    async fn main_agent_uses_llm_planner_for_skill_definition_deletion() -> anyhow::Result<()> {
        let db = Db::connect("sqlite::memory:").await?;
        db.create_skill(
            CreateSkill {
                name: "obsolete".to_owned(),
                description: "No longer useful".to_owned(),
                trigger_rules: vec!["obsolete".to_owned()],
                tool_subset: Vec::new(),
                resource_path: None,
            },
            "test",
        )
        .await?;
        let agent = MainAgent::new(db.clone()).with_planner(Arc::new(FixedPlanner {
            plan: Some(MainAgentPlan::DeleteSkillDefinition {
                selector: "obsolete".to_owned(),
            }),
            contexts: Arc::new(StdMutex::new(Vec::new())),
        }));

        let response = agent
            .handle_user_message(MainAgentMessageInput {
                content: "Could you remove the obsolete reusable skill?".to_owned(),
            })
            .await?;
        let skills = db.list_skills().await?;

        assert!(skills.is_empty());
        assert!(response.assistant_message.content.contains("Deleted skill"));

        Ok(())
    }

    #[tokio::test]
    async fn main_agent_can_pause_task_by_conversation() -> anyhow::Result<()> {
        let db = Db::connect("sqlite::memory:").await?;
        let agent = MainAgent::new(db.clone());
        let task = agent
            .create_task(CreateTask {
                title: "Check GitHub issues".to_owned(),
                description: "Look for open issues".to_owned(),
                task_type: TaskType::OneOff,
                priority: 0,
                requested_skills: Vec::new(),
                schedule: None,
                created_by: "test".to_owned(),
            })
            .await?;

        let response = agent
            .handle_user_message(MainAgentMessageInput {
                content: "pause task Check GitHub issues".to_owned(),
            })
            .await?;
        let updated = db.get_task(task.id).await?;

        assert_eq!(updated.status, TaskStatus::Paused);
        assert_eq!(response.changed_tasks.len(), 1);
        assert!(response.assistant_message.content.contains("Paused task"));

        Ok(())
    }

    #[tokio::test]
    async fn main_agent_can_delete_task_by_conversation() -> anyhow::Result<()> {
        let db = Db::connect("sqlite::memory:").await?;
        let agent = MainAgent::new(db.clone());
        let task = agent
            .create_task(CreateTask {
                title: "Remove stale task".to_owned(),
                description: "No longer needed".to_owned(),
                task_type: TaskType::OneOff,
                priority: 0,
                requested_skills: Vec::new(),
                schedule: None,
                created_by: "test".to_owned(),
            })
            .await?;

        let response = agent
            .handle_user_message(MainAgentMessageInput {
                content: "delete task Remove stale task".to_owned(),
            })
            .await?;
        let tasks = db.list_tasks().await?;
        let actions = db.list_global_actions().await?;

        assert!(tasks.is_empty());
        assert!(db.get_task(task.id).await.is_err());
        assert_eq!(response.changed_tasks.len(), 1);
        assert_eq!(response.changed_tasks[0].id, task.id);
        assert!(response.assistant_message.content.contains("Deleted task"));
        assert!(
            actions
                .iter()
                .any(|action| action.action_type == "delete_task")
        );

        Ok(())
    }

    #[tokio::test]
    async fn main_agent_can_update_task_details_by_conversation() -> anyhow::Result<()> {
        let db = Db::connect("sqlite::memory:").await?;
        let agent = MainAgent::new(db.clone());
        let task = agent
            .create_task(CreateTask {
                title: "Deploy release".to_owned(),
                description: "Old deployment instructions".to_owned(),
                task_type: TaskType::OneOff,
                priority: 0,
                requested_skills: Vec::new(),
                schedule: None,
                created_by: "test".to_owned(),
            })
            .await?;

        let response = agent
            .handle_user_message(MainAgentMessageInput {
                content: "update task Deploy release description: Deploy to production".to_owned(),
            })
            .await?;
        let updated = db.get_task(task.id).await?;
        let actions = db.list_task_actions(task.id).await?;

        assert_eq!(updated.description, "Deploy to production");
        assert_eq!(response.changed_tasks.len(), 1);
        assert!(response.assistant_message.content.contains("Updated task"));
        assert!(
            actions
                .iter()
                .any(|action| action.action_type == "update_task")
        );

        let response = agent
            .handle_user_message(MainAgentMessageInput {
                content: "rename task Deploy release to Ship release".to_owned(),
            })
            .await?;
        let updated = db.get_task(task.id).await?;

        assert_eq!(updated.title, "Ship release");
        assert_eq!(response.changed_tasks.len(), 1);

        Ok(())
    }

    #[tokio::test]
    async fn main_agent_can_request_task_run_now_by_conversation() -> anyhow::Result<()> {
        let db = Db::connect("sqlite::memory:").await?;
        let agent = MainAgent::new(db.clone());
        agent
            .create_task(CreateTask {
                title: "High priority background task".to_owned(),
                description: "Already urgent work".to_owned(),
                task_type: TaskType::OneOff,
                priority: 9,
                requested_skills: Vec::new(),
                schedule: None,
                created_by: "test".to_owned(),
            })
            .await?;
        let task = agent
            .create_task(CreateTask {
                title: "Deploy release".to_owned(),
                description: "Deploy the release".to_owned(),
                task_type: TaskType::OneOff,
                priority: 0,
                requested_skills: Vec::new(),
                schedule: None,
                created_by: "test".to_owned(),
            })
            .await?;

        let response = agent
            .handle_user_message(MainAgentMessageInput {
                content: "run task Deploy release now".to_owned(),
            })
            .await?;
        let updated = db.get_task(task.id).await?;
        let actions = db.list_task_actions(task.id).await?;

        assert!(response.scheduler_tick_requested);
        assert_eq!(response.changed_tasks.len(), 1);
        assert_eq!(updated.status, TaskStatus::Queued);
        assert_eq!(updated.priority, 10);
        assert_eq!(updated.queue_position, -1);
        assert!(
            response
                .assistant_message
                .content
                .contains("requested a scheduler scan")
        );
        assert!(
            actions
                .iter()
                .any(|action| action.action_type == "request_task_run_now")
        );

        let completed_task = agent
            .create_task(CreateTask {
                title: "Completed rollout".to_owned(),
                description: "Already done".to_owned(),
                task_type: TaskType::OneOff,
                priority: 0,
                requested_skills: Vec::new(),
                schedule: None,
                created_by: "test".to_owned(),
            })
            .await?;
        agent
            .complete_task(completed_task.id, "already deployed")
            .await?;

        let response = agent
            .handle_user_message(MainAgentMessageInput {
                content: "run task Completed rollout now".to_owned(),
            })
            .await?;
        assert!(!response.scheduler_tick_requested);
        assert!(
            response
                .assistant_message
                .content
                .contains("already completed")
        );

        Ok(())
    }

    #[tokio::test]
    async fn main_agent_can_request_next_task_run_by_conversation() -> anyhow::Result<()> {
        let db = Db::connect("sqlite::memory:").await?;
        let agent = MainAgent::new(db.clone());
        let low_priority = agent
            .create_task(CreateTask {
                title: "Write implementation notes".to_owned(),
                description: "Document the current implementation".to_owned(),
                task_type: TaskType::OneOff,
                priority: 0,
                requested_skills: Vec::new(),
                schedule: None,
                created_by: "test".to_owned(),
            })
            .await?;
        let high_priority = agent
            .create_task(CreateTask {
                title: "Fix release blocker".to_owned(),
                description: "Resolve the blocker first".to_owned(),
                task_type: TaskType::OneOff,
                priority: 5,
                requested_skills: Vec::new(),
                schedule: None,
                created_by: "test".to_owned(),
            })
            .await?;

        let response = agent
            .handle_user_message(MainAgentMessageInput {
                content: "run next task".to_owned(),
            })
            .await?;
        let updated_low_priority = db.get_task(low_priority.id).await?;
        let updated_high_priority = db.get_task(high_priority.id).await?;
        let actions = db.list_task_actions(high_priority.id).await?;

        assert!(response.scheduler_tick_requested);
        assert_eq!(response.changed_tasks.len(), 1);
        assert_eq!(response.changed_tasks[0].id, high_priority.id);
        assert_eq!(updated_high_priority.status, TaskStatus::Queued);
        assert_eq!(updated_high_priority.queue_position, -1);
        assert!(updated_high_priority.priority > high_priority.priority);
        assert_eq!(
            updated_low_priority.queue_position,
            low_priority.queue_position
        );
        assert!(
            response
                .assistant_message
                .content
                .contains("Selected next runnable task")
        );
        assert!(
            actions
                .iter()
                .any(|action| action.action_type == "request_next_task_run")
        );

        Ok(())
    }

    #[tokio::test]
    async fn main_agent_can_finish_tasks_by_conversation() -> anyhow::Result<()> {
        let db = Db::connect("sqlite::memory:").await?;
        let agent = MainAgent::new(db.clone());
        let complete_task = agent
            .create_task(CreateTask {
                title: "Deploy release".to_owned(),
                description: "Deploy the release".to_owned(),
                task_type: TaskType::OneOff,
                priority: 0,
                requested_skills: Vec::new(),
                schedule: None,
                created_by: "test".to_owned(),
            })
            .await?;
        let fail_task = agent
            .create_task(CreateTask {
                title: "Publish package".to_owned(),
                description: "Publish the package".to_owned(),
                task_type: TaskType::OneOff,
                priority: 0,
                requested_skills: Vec::new(),
                schedule: None,
                created_by: "test".to_owned(),
            })
            .await?;
        let retry_task = agent
            .create_task(CreateTask {
                title: "Refresh deployment".to_owned(),
                description: "Refresh the failed deployment".to_owned(),
                task_type: TaskType::OneOff,
                priority: 0,
                requested_skills: Vec::new(),
                schedule: None,
                created_by: "test".to_owned(),
            })
            .await?;
        db.fail_task(retry_task.id, "expired credentials", "test")
            .await?;

        let response = agent
            .handle_user_message(MainAgentMessageInput {
                content: "complete task Deploy release: deployed to production".to_owned(),
            })
            .await?;
        let completed = db.get_task(complete_task.id).await?;

        assert_eq!(completed.status, TaskStatus::Completed);
        assert_eq!(
            completed.result_summary.as_deref(),
            Some("deployed to production")
        );
        assert_eq!(response.changed_tasks.len(), 1);

        let response = agent
            .handle_user_message(MainAgentMessageInput {
                content: "fail task Publish package: registry token expired".to_owned(),
            })
            .await?;
        let failed = db.get_task(fail_task.id).await?;

        assert_eq!(failed.status, TaskStatus::Failed);
        assert_eq!(
            failed.result_summary.as_deref(),
            Some("registry token expired")
        );
        assert_eq!(response.changed_tasks.len(), 1);

        let response = agent
            .handle_user_message(MainAgentMessageInput {
                content: "retry task Refresh deployment: credentials refreshed".to_owned(),
            })
            .await?;
        let retried = db.get_task(retry_task.id).await?;
        let actions = db.list_task_actions(retry_task.id).await?;

        assert_eq!(retried.status, TaskStatus::Queued);
        assert_eq!(
            retried.result_summary.as_deref(),
            Some("credentials refreshed")
        );
        assert_eq!(response.changed_tasks.len(), 1);
        assert!(
            actions
                .iter()
                .any(|action| action.action_type == "requeue_task_after_failure")
        );

        Ok(())
    }

    #[tokio::test]
    async fn main_agent_can_review_memory_by_conversation() -> anyhow::Result<()> {
        let db = Db::connect("sqlite::memory:").await?;
        let agent = MainAgent::new(db.clone());
        let approved_memory = db
            .create_memory(
                persistent_agent_domain::CreateMemory {
                    scope: "project".to_owned(),
                    content: "Prefer cargo test before push".to_owned(),
                    source_task_id: None,
                    status: MemoryStatus::Pending,
                    confidence: 0.84,
                },
                "worker",
            )
            .await?;
        let rejected_memory = db
            .create_memory(
                persistent_agent_domain::CreateMemory {
                    scope: "project".to_owned(),
                    content: "Noisy temporary note".to_owned(),
                    source_task_id: None,
                    status: MemoryStatus::Pending,
                    confidence: 0.22,
                },
                "worker",
            )
            .await?;

        let response = agent
            .handle_user_message(MainAgentMessageInput {
                content: format!("approve memory candidate {}", approved_memory.id),
            })
            .await?;
        let approved = db.get_memory(approved_memory.id).await?;

        assert_eq!(approved.status, MemoryStatus::Approved);
        assert!(
            response
                .assistant_message
                .content
                .contains("Approved memory")
        );

        let response = agent
            .handle_user_message(MainAgentMessageInput {
                content: "reject memory candidate Noisy temporary note".to_owned(),
            })
            .await?;
        let rejected = db.get_memory(rejected_memory.id).await?;

        assert_eq!(rejected.status, MemoryStatus::Rejected);
        assert!(
            response
                .assistant_message
                .content
                .contains("Rejected memory")
        );

        let response = agent
            .handle_user_message(MainAgentMessageInput {
                content: format!(
                    "update memory candidate {}: Prefer cargo test --workspace before push",
                    approved_memory.id
                ),
            })
            .await?;
        let updated = db.get_memory(approved_memory.id).await?;

        assert_eq!(updated.content, "Prefer cargo test --workspace before push");
        assert!(
            response
                .assistant_message
                .content
                .contains("Updated memory")
        );

        let response = agent
            .handle_user_message(MainAgentMessageInput {
                content: "delete memory candidate Noisy temporary note".to_owned(),
            })
            .await?;
        let memories = db.list_memories().await?;

        assert!(
            !memories
                .iter()
                .any(|memory| memory.id == rejected_memory.id)
        );
        assert!(
            response
                .assistant_message
                .content
                .contains("Deleted memory")
        );

        Ok(())
    }

    #[tokio::test]
    async fn main_agent_can_bulk_review_pending_memories_by_conversation() -> anyhow::Result<()> {
        let db = Db::connect("sqlite::memory:").await?;
        let agent = MainAgent::new(db.clone());
        let first = db
            .create_memory(
                persistent_agent_domain::CreateMemory {
                    scope: "project".to_owned(),
                    content: "Prefer cargo test before push".to_owned(),
                    source_task_id: None,
                    status: MemoryStatus::Pending,
                    confidence: 0.84,
                },
                "worker",
            )
            .await?;
        let second = db
            .create_memory(
                persistent_agent_domain::CreateMemory {
                    scope: "project".to_owned(),
                    content: "Avoid noisy temporary notes".to_owned(),
                    source_task_id: None,
                    status: MemoryStatus::Pending,
                    confidence: 0.72,
                },
                "worker",
            )
            .await?;
        let already_approved = db
            .create_memory(
                persistent_agent_domain::CreateMemory {
                    scope: "project".to_owned(),
                    content: "Already durable".to_owned(),
                    source_task_id: None,
                    status: MemoryStatus::Approved,
                    confidence: 0.95,
                },
                "worker",
            )
            .await?;

        let response = agent
            .handle_user_message(MainAgentMessageInput {
                content: "approve all pending memories".to_owned(),
            })
            .await?;
        let first = db.get_memory(first.id).await?;
        let second = db.get_memory(second.id).await?;
        let already_approved = db.get_memory(already_approved.id).await?;
        let actions = db.list_global_actions().await?;

        assert_eq!(first.status, MemoryStatus::Approved);
        assert_eq!(second.status, MemoryStatus::Approved);
        assert_eq!(already_approved.status, MemoryStatus::Approved);
        assert!(response.assistant_message.content.contains("Approved 2"));
        assert!(actions.iter().any(|action| {
            action.action_type == "bulk_set_memory_status"
                && action.details["status"] == "approved"
                && action.details["count"] == 2
        }));

        let reject_me = db
            .create_memory(
                persistent_agent_domain::CreateMemory {
                    scope: "project".to_owned(),
                    content: "Temporary dead end".to_owned(),
                    source_task_id: None,
                    status: MemoryStatus::Pending,
                    confidence: 0.2,
                },
                "worker",
            )
            .await?;
        let response = agent
            .handle_user_message(MainAgentMessageInput {
                content: "reject all memory candidates".to_owned(),
            })
            .await?;
        let rejected = db.get_memory(reject_me.id).await?;

        assert_eq!(rejected.status, MemoryStatus::Rejected);
        assert!(response.assistant_message.content.contains("Rejected 1"));

        Ok(())
    }

    #[tokio::test]
    async fn main_agent_can_create_memory_by_conversation() -> anyhow::Result<()> {
        let db = Db::connect("sqlite::memory:").await?;
        let agent = MainAgent::new(db.clone());

        let response = agent
            .handle_user_message(MainAgentMessageInput {
                content: "remember repo: prefer cargo test before push".to_owned(),
            })
            .await?;
        let memories = db.list_memories().await?;
        let actions = db.list_global_actions().await?;

        assert_eq!(memories.len(), 1);
        assert_eq!(memories[0].scope, "repo");
        assert_eq!(memories[0].content, "prefer cargo test before push");
        assert_eq!(memories[0].status, MemoryStatus::Approved);
        assert!(
            response
                .assistant_message
                .content
                .contains("Remembered [repo]")
        );
        assert!(actions.iter().any(|action| {
            action.action_type == "create_memory"
                && action.details["memory_id"] == memories[0].id.to_string()
        }));

        Ok(())
    }

    #[tokio::test]
    async fn main_agent_can_manage_requested_skills_by_conversation() -> anyhow::Result<()> {
        let db = Db::connect("sqlite::memory:").await?;
        let agent = MainAgent::new(db.clone());
        let task = agent
            .create_task(CreateTask {
                title: "Deploy release".to_owned(),
                description: "Prepare release".to_owned(),
                task_type: TaskType::OneOff,
                priority: 0,
                requested_skills: vec!["shell".to_owned()],
                schedule: None,
                created_by: "test".to_owned(),
            })
            .await?;

        let response = agent
            .handle_user_message(MainAgentMessageInput {
                content: "add skill to task Deploy release: github, shell".to_owned(),
            })
            .await?;
        let updated = db.get_task(task.id).await?;

        assert_eq!(updated.requested_skills, vec!["shell", "github"]);
        assert_eq!(response.changed_tasks.len(), 1);
        assert!(
            response
                .assistant_message
                .content
                .contains("Updated requested skills")
        );

        agent
            .handle_user_message(MainAgentMessageInput {
                content: "remove skill shell from task Deploy release".to_owned(),
            })
            .await?;
        let updated = db.get_task(task.id).await?;

        assert_eq!(updated.requested_skills, vec!["github"]);

        Ok(())
    }

    #[tokio::test]
    async fn main_agent_can_list_memories_by_conversation() -> anyhow::Result<()> {
        let db = Db::connect("sqlite::memory:").await?;
        let agent = MainAgent::new(db.clone());
        let global_pending = db
            .create_memory(
                CreateMemory {
                    scope: "project".to_owned(),
                    content: "Prefer cargo test before pushing.".to_owned(),
                    source_task_id: None,
                    status: MemoryStatus::Pending,
                    confidence: 0.8,
                },
                "worker",
            )
            .await?;
        db.create_memory(
            CreateMemory {
                scope: "project".to_owned(),
                content: "Use browser screenshots for UI changes.".to_owned(),
                source_task_id: None,
                status: MemoryStatus::Approved,
                confidence: 0.9,
            },
            "worker",
        )
        .await?;

        let response = agent
            .handle_user_message(MainAgentMessageInput {
                content: "show memory candidates".to_owned(),
            })
            .await?;
        let actions = db.list_global_actions().await?;

        assert!(
            response
                .assistant_message
                .content
                .contains("Prefer cargo test")
        );
        assert!(
            !response
                .assistant_message
                .content
                .contains("browser screenshots")
        );
        assert!(actions.iter().any(|action| {
            action.action_type == "list_memories"
                && action.details["filter"] == "pending"
                && action.details["count"] == 1
        }));

        let response = agent
            .handle_user_message(MainAgentMessageInput {
                content: "list approved memories".to_owned(),
            })
            .await?;

        assert!(
            response
                .assistant_message
                .content
                .contains("browser screenshots")
        );

        let task = agent
            .create_task(CreateTask {
                title: "Deploy release".to_owned(),
                description: "Deploy the release".to_owned(),
                task_type: TaskType::OneOff,
                priority: 0,
                requested_skills: Vec::new(),
                schedule: None,
                created_by: "test".to_owned(),
            })
            .await?;
        let task_pending = db
            .create_memory(
                CreateMemory {
                    scope: "project".to_owned(),
                    content: "Deployment should verify staging first.".to_owned(),
                    source_task_id: Some(task.id),
                    status: MemoryStatus::Pending,
                    confidence: 0.73,
                },
                "worker",
            )
            .await?;

        let response = agent
            .handle_user_message(MainAgentMessageInput {
                content: "show memory candidates for task Deploy release".to_owned(),
            })
            .await?;
        let actions = db.list_task_actions(task.id).await?;

        assert!(
            response
                .assistant_message
                .content
                .contains("Task 'Deploy release' memory candidates")
        );
        assert!(
            response
                .assistant_message
                .content
                .contains("Deployment should verify staging first.")
        );
        assert!(
            actions
                .iter()
                .any(|action| action.action_type == "list_task_memories")
        );

        let response = agent
            .handle_user_message(MainAgentMessageInput {
                content: "approve all memory candidates for task Deploy release".to_owned(),
            })
            .await?;
        let task_pending = db.get_memory(task_pending.id).await?;
        let global_pending = db.get_memory(global_pending.id).await?;
        let actions = db.list_task_actions(task.id).await?;

        assert_eq!(task_pending.status, MemoryStatus::Approved);
        assert_eq!(global_pending.status, MemoryStatus::Pending);
        assert!(
            response
                .assistant_message
                .content
                .contains("Approved 1 pending memory candidate(s) from 'Deploy release'")
        );
        assert!(actions.iter().any(|action| {
            action.action_type == "bulk_set_task_memory_status"
                && action.details["status"] == "approved"
                && action.details["count"] == 1
        }));

        Ok(())
    }

    #[tokio::test]
    async fn main_agent_can_convert_task_type_by_conversation() -> anyhow::Result<()> {
        let db = Db::connect("sqlite::memory:").await?;
        let agent = MainAgent::new(db.clone());
        let task = agent
            .create_task(CreateTask {
                title: "Check GitHub issues".to_owned(),
                description: "Look for open issues".to_owned(),
                task_type: TaskType::OneOff,
                priority: 0,
                requested_skills: Vec::new(),
                schedule: None,
                created_by: "test".to_owned(),
            })
            .await?;

        let response = agent
            .handle_user_message(MainAgentMessageInput {
                content: "convert task Check GitHub issues to recurring every 45 seconds"
                    .to_owned(),
            })
            .await?;
        let updated = db.get_task(task.id).await?;

        assert_eq!(updated.task_type, TaskType::Recurring);
        assert_eq!(
            updated
                .schedule
                .as_ref()
                .and_then(|value| value.get("interval_seconds"))
                .and_then(serde_json::Value::as_i64),
            Some(45)
        );
        assert_eq!(response.changed_tasks.len(), 1);

        Ok(())
    }

    #[tokio::test]
    async fn main_agent_can_update_recurring_task_schedule_by_conversation() -> anyhow::Result<()> {
        let db = Db::connect("sqlite::memory:").await?;
        let agent = MainAgent::new(db.clone());
        let task = agent
            .create_task(CreateTask {
                title: "Check GitHub issues".to_owned(),
                description: "Look for open issues".to_owned(),
                task_type: TaskType::Recurring,
                priority: 0,
                requested_skills: Vec::new(),
                schedule: Some(serde_json::json!({ "interval_seconds": 60 })),
                created_by: "test".to_owned(),
            })
            .await?;

        let response = agent
            .handle_user_message(MainAgentMessageInput {
                content: "set task Check GitHub issues interval to 5 minutes".to_owned(),
            })
            .await?;
        let updated = db.get_task(task.id).await?;
        let actions = db.list_task_actions(task.id).await?;

        assert_eq!(
            updated.schedule,
            Some(serde_json::json!({ "interval_seconds": 300 }))
        );
        assert_eq!(response.changed_tasks.len(), 1);
        assert!(
            response
                .assistant_message
                .content
                .contains("recurring interval to 300 seconds")
        );
        assert!(
            actions
                .iter()
                .any(|action| action.action_type == "update_task_schedule")
        );

        Ok(())
    }

    #[tokio::test]
    async fn main_agent_can_manage_task_dependencies_by_conversation() -> anyhow::Result<()> {
        let db = Db::connect("sqlite::memory:").await?;
        let agent = MainAgent::new(db.clone());
        let dependency = agent
            .create_task(CreateTask {
                title: "Build package".to_owned(),
                description: "Build the release package".to_owned(),
                task_type: TaskType::OneOff,
                priority: 0,
                requested_skills: Vec::new(),
                schedule: None,
                created_by: "test".to_owned(),
            })
            .await?;
        let dependent = agent
            .create_task(CreateTask {
                title: "Deploy release".to_owned(),
                description: "Deploy the release".to_owned(),
                task_type: TaskType::OneOff,
                priority: 10,
                requested_skills: Vec::new(),
                schedule: None,
                created_by: "test".to_owned(),
            })
            .await?;

        let response = agent
            .handle_user_message(MainAgentMessageInput {
                content: "make task Deploy release depend on Build package".to_owned(),
            })
            .await?;
        let dependencies = db.list_task_dependencies(dependent.id).await?;

        assert_eq!(dependencies.len(), 1);
        assert_eq!(dependencies[0].depends_on_task_id, dependency.id);
        assert_eq!(response.changed_tasks.len(), 1);
        assert!(
            response
                .assistant_message
                .content
                .contains("Added dependency")
        );

        agent
            .add_task_resource_lock(dependent.id, "repo:oh-my-harness/Persistent-Agent")
            .await?;
        let response = agent
            .handle_user_message(MainAgentMessageInput {
                content: "show constraints for task Deploy release".to_owned(),
            })
            .await?;
        let actions = db.list_task_actions(dependent.id).await?;

        assert!(
            response
                .assistant_message
                .content
                .contains("Task 'Deploy release' constraints")
        );
        assert!(response.assistant_message.content.contains("Build package"));
        assert!(
            response
                .assistant_message
                .content
                .contains("repo:oh-my-harness/Persistent-Agent")
        );
        assert!(
            actions
                .iter()
                .any(|action| action.action_type == "list_task_constraints")
        );

        let response = agent
            .handle_user_message(MainAgentMessageInput {
                content: "remove dependency task Deploy release on Build package".to_owned(),
            })
            .await?;
        let dependencies = db.list_task_dependencies(dependent.id).await?;

        assert!(dependencies.is_empty());
        assert_eq!(response.changed_tasks.len(), 1);
        assert!(
            response
                .assistant_message
                .content
                .contains("Removed dependency")
        );

        Ok(())
    }

    #[tokio::test]
    async fn main_agent_can_add_task_note_by_conversation() -> anyhow::Result<()> {
        let db = Db::connect("sqlite::memory:").await?;
        let agent = MainAgent::new(db.clone());
        let task = agent
            .create_task(CreateTask {
                title: "Deploy release".to_owned(),
                description: "Deploy the release".to_owned(),
                task_type: TaskType::OneOff,
                priority: 0,
                requested_skills: Vec::new(),
                schedule: None,
                created_by: "test".to_owned(),
            })
            .await?;

        let response = agent
            .handle_user_message(MainAgentMessageInput {
                content: "add note to task Deploy release: wait for staging approval".to_owned(),
            })
            .await?;
        let notes = db.list_task_notes(task.id).await?;

        assert_eq!(notes.len(), 1);
        assert_eq!(notes[0].content, "wait for staging approval");
        assert_eq!(response.changed_tasks.len(), 1);
        assert!(response.assistant_message.content.contains("Added note"));

        let response = agent
            .handle_user_message(MainAgentMessageInput {
                content: "show notes for task Deploy release".to_owned(),
            })
            .await?;
        let actions = db.list_task_actions(task.id).await?;

        assert!(
            response
                .assistant_message
                .content
                .contains("Task 'Deploy release' notes")
        );
        assert!(
            response
                .assistant_message
                .content
                .contains("wait for staging approval")
        );
        assert!(
            actions
                .iter()
                .any(|action| action.action_type == "list_task_notes")
        );

        Ok(())
    }

    #[tokio::test]
    async fn main_agent_can_manage_resource_locks_by_conversation() -> anyhow::Result<()> {
        let db = Db::connect("sqlite::memory:").await?;
        let agent = MainAgent::new(db.clone());
        let task = agent
            .create_task(CreateTask {
                title: "Deploy release".to_owned(),
                description: "Deploy the release".to_owned(),
                task_type: TaskType::OneOff,
                priority: 0,
                requested_skills: Vec::new(),
                schedule: None,
                created_by: "test".to_owned(),
            })
            .await?;

        let response = agent
            .handle_user_message(MainAgentMessageInput {
                content: "add resource lock to task Deploy release: repo:persistent-agent"
                    .to_owned(),
            })
            .await?;
        let locks = db.list_task_resource_locks(task.id).await?;

        assert_eq!(locks.len(), 1);
        assert_eq!(locks[0].resource_key, "repo:persistent-agent");
        assert_eq!(response.changed_tasks.len(), 1);
        assert!(
            response
                .assistant_message
                .content
                .contains("Added resource lock")
        );

        let response = agent
            .handle_user_message(MainAgentMessageInput {
                content: "remove resource lock from task Deploy release: repo:persistent-agent"
                    .to_owned(),
            })
            .await?;
        let locks = db.list_task_resource_locks(task.id).await?;

        assert!(locks.is_empty());
        assert_eq!(response.changed_tasks.len(), 1);
        assert!(
            response
                .assistant_message
                .content
                .contains("Removed resource lock")
        );

        Ok(())
    }

    #[tokio::test]
    async fn main_agent_can_list_tasks_by_conversation() -> anyhow::Result<()> {
        let db = Db::connect("sqlite::memory:").await?;
        let agent = MainAgent::new(db.clone());
        agent
            .create_task(CreateTask {
                title: "Check repository issues".to_owned(),
                description: "Look for open issues".to_owned(),
                task_type: TaskType::Recurring,
                priority: 3,
                requested_skills: Vec::new(),
                schedule: Some(serde_json::json!({ "interval_seconds": 300 })),
                created_by: "test".to_owned(),
            })
            .await?;

        let response = agent
            .handle_user_message(MainAgentMessageInput {
                content: "list tasks".to_owned(),
            })
            .await?;
        let global_actions = db.list_global_actions().await?;

        assert!(
            response
                .assistant_message
                .content
                .contains("Task pool has 1")
        );
        assert!(
            response
                .assistant_message
                .content
                .contains("Check repository issues")
        );
        assert!(response.assistant_message.content.contains("recurring"));
        assert!(
            global_actions
                .iter()
                .any(|action| action.action_type == "list_tasks")
        );

        Ok(())
    }

    #[tokio::test]
    async fn main_agent_can_list_tasks_by_status_by_conversation() -> anyhow::Result<()> {
        let db = Db::connect("sqlite::memory:").await?;
        let agent = MainAgent::new(db.clone());
        let failed = agent
            .create_task(CreateTask {
                title: "Repair failing build".to_owned(),
                description: "Investigate the broken build".to_owned(),
                task_type: TaskType::OneOff,
                priority: 0,
                requested_skills: Vec::new(),
                schedule: None,
                created_by: "test".to_owned(),
            })
            .await?;
        agent
            .create_task(CreateTask {
                title: "Write release notes".to_owned(),
                description: "Draft the next release notes".to_owned(),
                task_type: TaskType::OneOff,
                priority: 1,
                requested_skills: Vec::new(),
                schedule: None,
                created_by: "test".to_owned(),
            })
            .await?;
        db.fail_task(failed.id, "build failed", "test").await?;

        let response = agent
            .handle_user_message(MainAgentMessageInput {
                content: "list failed tasks".to_owned(),
            })
            .await?;
        let global_actions = db.list_global_actions().await?;

        assert!(
            response
                .assistant_message
                .content
                .contains("failed tasks has 1 task")
        );
        assert!(
            response
                .assistant_message
                .content
                .contains("Repair failing build")
        );
        assert!(
            !response
                .assistant_message
                .content
                .contains("Write release notes")
        );
        assert!(global_actions.iter().any(|action| {
            action.action_type == "list_tasks_by_status"
                && action
                    .details
                    .get("status")
                    .and_then(serde_json::Value::as_str)
                    == Some("failed")
        }));

        Ok(())
    }

    #[tokio::test]
    async fn main_agent_can_list_waiting_for_user_tasks_by_conversation() -> anyhow::Result<()> {
        let db = Db::connect("sqlite::memory:").await?;
        let agent = MainAgent::new(db.clone());
        let task = agent
            .create_task(CreateTask {
                title: "Deploy release".to_owned(),
                description: "Deploy after approval".to_owned(),
                task_type: TaskType::OneOff,
                priority: 0,
                requested_skills: Vec::new(),
                schedule: None,
                created_by: "test".to_owned(),
            })
            .await?;
        agent
            .request_user_clarification(task.id, "Which environment should receive the release?")
            .await?;

        let response = agent
            .handle_user_message(MainAgentMessageInput {
                content: "show tasks waiting for my input".to_owned(),
            })
            .await?;
        let global_actions = db.list_global_actions().await?;

        assert!(
            response
                .assistant_message
                .content
                .contains("Deploy release")
        );
        assert!(
            response
                .assistant_message
                .content
                .contains("Which environment should receive the release?")
        );
        assert!(
            response
                .assistant_message
                .content
                .contains("reply to task <title>")
        );
        assert!(
            global_actions
                .iter()
                .any(|action| action.action_type == "list_waiting_for_user_tasks")
        );

        Ok(())
    }

    #[tokio::test]
    async fn main_agent_can_list_waiting_for_schedule_tasks_by_conversation() -> anyhow::Result<()>
    {
        let db = Db::connect("sqlite::memory:").await?;
        let agent = MainAgent::new(db.clone());
        let task = agent
            .create_task(CreateTask {
                title: "Check GitHub issues".to_owned(),
                description: "Check repository issues repeatedly".to_owned(),
                task_type: TaskType::Recurring,
                priority: 0,
                requested_skills: Vec::new(),
                schedule: Some(serde_json::json!({ "interval_seconds": 300 })),
                created_by: "test".to_owned(),
            })
            .await?;
        agent.complete_task(task.id, "checked once").await?;

        let response = agent
            .handle_user_message(MainAgentMessageInput {
                content: "show tasks waiting for schedule".to_owned(),
            })
            .await?;
        let global_actions = db.list_global_actions().await?;

        assert!(
            response
                .assistant_message
                .content
                .contains("Check GitHub issues")
        );
        assert!(response.assistant_message.content.contains("next_run_at"));
        assert!(
            response
                .assistant_message
                .content
                .contains("run scheduler scan")
        );
        assert!(
            global_actions
                .iter()
                .any(|action| action.action_type == "list_waiting_for_schedule_tasks")
        );

        Ok(())
    }

    #[tokio::test]
    async fn main_agent_can_show_scheduler_state_by_conversation() -> anyhow::Result<()> {
        let db = Db::connect("sqlite::memory:").await?;
        let agent = MainAgent::new(db.clone());
        let queued = agent
            .create_task(CreateTask {
                title: "Check repository issues".to_owned(),
                description: "Look for open issues".to_owned(),
                task_type: TaskType::OneOff,
                priority: 3,
                requested_skills: Vec::new(),
                schedule: None,
                created_by: "test".to_owned(),
            })
            .await?;
        let blocked = agent
            .create_task(CreateTask {
                title: "Deploy release".to_owned(),
                description: "Deploy after environment is known".to_owned(),
                task_type: TaskType::OneOff,
                priority: 0,
                requested_skills: Vec::new(),
                schedule: None,
                created_by: "test".to_owned(),
            })
            .await?;
        db.set_task_status(
            blocked.id,
            TaskStatus::WaitingForUser,
            "test",
            Some("Which environment?"),
        )
        .await?;

        let response = agent
            .handle_user_message(MainAgentMessageInput {
                content: "show scheduler status".to_owned(),
            })
            .await?;
        let global_actions = db.list_global_actions().await?;

        assert!(
            response
                .assistant_message
                .content
                .contains("Execution state")
        );
        assert!(response.assistant_message.content.contains("Next runnable"));
        assert!(response.assistant_message.content.contains(&queued.title));
        assert!(response.assistant_message.content.contains(&blocked.title));
        assert!(
            response
                .assistant_message
                .content
                .contains("Which environment?")
        );
        assert!(
            global_actions
                .iter()
                .any(|action| action.action_type == "inspect_scheduler_state")
        );

        Ok(())
    }

    #[tokio::test]
    async fn main_agent_can_list_global_actions_by_conversation() -> anyhow::Result<()> {
        let db = Db::connect("sqlite::memory:").await?;
        let agent = MainAgent::new(db.clone());

        agent.inspect_workspace_status().await?;
        let response = agent
            .handle_user_message(MainAgentMessageInput {
                content: "show main agent audit".to_owned(),
            })
            .await?;
        let global_actions = db.list_global_actions().await?;

        assert!(
            response
                .assistant_message
                .content
                .contains("Recent global main-agent actions")
        );
        assert!(
            response
                .assistant_message
                .content
                .contains("inspect_workspace_status")
        );
        assert!(global_actions.iter().any(|action| {
            action.action_type == "list_global_actions" && action.details["limit"] == 10
        }));

        Ok(())
    }

    #[tokio::test]
    async fn main_agent_can_list_task_artifacts_by_conversation() -> anyhow::Result<()> {
        let db = Db::connect("sqlite::memory:").await?;
        let agent = MainAgent::new(db.clone());
        let task = agent
            .create_task(CreateTask {
                title: "Deploy release".to_owned(),
                description: "Deploy the release".to_owned(),
                task_type: TaskType::OneOff,
                priority: 0,
                requested_skills: Vec::new(),
                schedule: None,
                created_by: "test".to_owned(),
            })
            .await?;
        let attempt = db
            .create_attempt(task.id, TaskStatus::Completed, Some("created report"))
            .await?;
        db.record_task_artifact(
            task.id,
            Some(attempt.id),
            "release-report.md",
            "file",
            "file://release-report.md",
            Some("Release summary"),
        )
        .await?;

        let response = agent
            .handle_user_message(MainAgentMessageInput {
                content: "show artifacts for task Deploy release".to_owned(),
            })
            .await?;
        let actions = db.list_task_actions(task.id).await?;

        assert!(
            response
                .assistant_message
                .content
                .contains("release-report.md")
        );
        assert!(
            response
                .assistant_message
                .content
                .contains("Release summary")
        );
        assert!(
            actions
                .iter()
                .any(|action| action.action_type == "list_task_artifacts")
        );

        Ok(())
    }

    #[tokio::test]
    async fn main_agent_can_list_task_history_by_conversation() -> anyhow::Result<()> {
        let db = Db::connect("sqlite::memory:").await?;
        let agent = MainAgent::new(db.clone());
        let task = agent
            .create_task(CreateTask {
                title: "Deploy release".to_owned(),
                description: "Deploy the release".to_owned(),
                task_type: TaskType::OneOff,
                priority: 0,
                requested_skills: Vec::new(),
                schedule: None,
                created_by: "test".to_owned(),
            })
            .await?;
        let attempt = db
            .create_attempt(
                task.id,
                TaskStatus::Completed,
                Some("created release report"),
            )
            .await?;
        db.record_attempt_event(
            attempt.id,
            task.id,
            "worker_completed",
            "Worker completed the release task.",
            serde_json::json!({ "summary": "created release report" }),
        )
        .await?;

        let response = agent
            .handle_user_message(MainAgentMessageInput {
                content: "show history for task Deploy release".to_owned(),
            })
            .await?;
        let actions = db.list_task_actions(task.id).await?;

        assert!(
            response
                .assistant_message
                .content
                .contains("Task 'Deploy release' history")
        );
        assert!(
            response
                .assistant_message
                .content
                .contains("worker_completed")
        );
        assert!(
            response
                .assistant_message
                .content
                .contains("created release report")
        );
        assert!(
            actions
                .iter()
                .any(|action| action.action_type == "list_task_history")
        );

        Ok(())
    }

    #[tokio::test]
    async fn main_agent_can_show_task_latest_result_by_conversation() -> anyhow::Result<()> {
        let db = Db::connect("sqlite::memory:").await?;
        let agent = MainAgent::new(db.clone());
        let task = agent
            .create_task(CreateTask {
                title: "Deploy release".to_owned(),
                description: "Deploy the release".to_owned(),
                task_type: TaskType::OneOff,
                priority: 0,
                requested_skills: Vec::new(),
                schedule: None,
                created_by: "test".to_owned(),
            })
            .await?;
        db.create_attempt(
            task.id,
            TaskStatus::Completed,
            Some("worker produced release report"),
        )
        .await?;
        agent
            .complete_task(task.id, "deployed to production")
            .await?;

        let response = agent
            .handle_user_message(MainAgentMessageInput {
                content: "show result for task Deploy release".to_owned(),
            })
            .await?;
        let actions = db.list_task_actions(task.id).await?;

        assert!(
            response
                .assistant_message
                .content
                .contains("Task 'Deploy release' latest result")
        );
        assert!(
            response
                .assistant_message
                .content
                .contains("deployed to production")
        );
        assert!(
            response
                .assistant_message
                .content
                .contains("worker produced release report")
        );
        assert!(
            actions
                .iter()
                .any(|action| action.action_type == "show_task_latest_result")
        );

        Ok(())
    }

    #[tokio::test]
    async fn main_agent_can_list_task_follow_ups_by_conversation() -> anyhow::Result<()> {
        let db = Db::connect("sqlite::memory:").await?;
        let agent = MainAgent::new(db.clone());
        let task = agent
            .create_task(CreateTask {
                title: "Deploy release".to_owned(),
                description: "Deploy the release".to_owned(),
                task_type: TaskType::OneOff,
                priority: 0,
                requested_skills: Vec::new(),
                schedule: None,
                created_by: "test".to_owned(),
            })
            .await?;
        let follow_up = db
            .create_task(
                CreateTask {
                    title: "Verify production deploy".to_owned(),
                    description: "Check production after deployment".to_owned(),
                    task_type: TaskType::OneOff,
                    priority: 0,
                    requested_skills: Vec::new(),
                    schedule: None,
                    created_by: "worker".to_owned(),
                },
                "worker",
            )
            .await?;
        db.record_action(
            Some(task.id),
            "worker",
            "create_follow_up_task",
            serde_json::json!({
                "follow_up_task_id": follow_up.id,
                "title": follow_up.title,
            }),
        )
        .await?;

        let response = agent
            .handle_user_message(MainAgentMessageInput {
                content: "show follow-up tasks for task Deploy release".to_owned(),
            })
            .await?;
        let actions = db.list_task_actions(task.id).await?;

        assert!(
            response
                .assistant_message
                .content
                .contains("created 1 follow-up task")
        );
        assert!(
            response
                .assistant_message
                .content
                .contains("Verify production deploy")
        );
        assert!(
            actions
                .iter()
                .any(|action| action.action_type == "list_task_follow_ups")
        );

        Ok(())
    }

    #[tokio::test]
    async fn main_agent_can_list_task_conversation_by_conversation() -> anyhow::Result<()> {
        let db = Db::connect("sqlite::memory:").await?;
        let agent = MainAgent::new(db.clone());
        let task = agent
            .create_task(CreateTask {
                title: "Deploy release".to_owned(),
                description: "Deploy the release".to_owned(),
                task_type: TaskType::OneOff,
                priority: 0,
                requested_skills: Vec::new(),
                schedule: None,
                created_by: "test".to_owned(),
            })
            .await?;
        agent
            .request_user_clarification(task.id, "Which environment?")
            .await?;
        agent.reply_to_task(task.id, "Use production.").await?;

        let response = agent
            .handle_user_message(MainAgentMessageInput {
                content: "show conversation for task Deploy release".to_owned(),
            })
            .await?;
        let actions = db.list_task_actions(task.id).await?;

        assert!(
            response
                .assistant_message
                .content
                .contains("Task 'Deploy release' conversation")
        );
        assert!(
            response
                .assistant_message
                .content
                .contains("Which environment?")
        );
        assert!(
            response
                .assistant_message
                .content
                .contains("Use production.")
        );
        assert!(
            actions
                .iter()
                .any(|action| action.action_type == "list_task_conversation")
        );

        Ok(())
    }

    #[tokio::test]
    async fn main_agent_can_request_scheduler_scan_by_conversation() -> anyhow::Result<()> {
        let db = Db::connect("sqlite::memory:").await?;
        let agent = MainAgent::new(db);

        let response = agent
            .handle_user_message(MainAgentMessageInput {
                content: "run scheduler tick".to_owned(),
            })
            .await?;

        assert!(response.scheduler_tick_requested);
        assert!(
            response
                .assistant_message
                .content
                .contains("Scheduler scan requested")
        );

        Ok(())
    }

    #[tokio::test]
    async fn main_agent_can_inspect_workspace_by_conversation() -> anyhow::Result<()> {
        let db = Db::connect("sqlite::memory:").await?;
        let agent = MainAgent::new(db.clone());

        let response = agent
            .handle_user_message(MainAgentMessageInput {
                content: "check project status".to_owned(),
            })
            .await?;
        let global_actions = db.list_global_actions().await?;

        assert!(response.assistant_message.content.contains("Workspace:"));
        assert!(response.assistant_message.content.contains("Git status:"));
        assert_eq!(response.changed_tasks.len(), 0);
        assert!(
            global_actions
                .iter()
                .any(|action| action.action_type == "inspect_workspace_status")
        );

        Ok(())
    }

    #[tokio::test]
    async fn main_agent_can_read_workspace_file_by_conversation() -> anyhow::Result<()> {
        let db = Db::connect("sqlite::memory:").await?;
        let agent = MainAgent::new(db.clone());

        let response = agent
            .handle_user_message(MainAgentMessageInput {
                content: "read file Cargo.toml".to_owned(),
            })
            .await?;
        let global_actions = db.list_global_actions().await?;

        assert!(
            response
                .assistant_message
                .content
                .contains("File: Cargo.toml")
        );
        assert!(
            response
                .assistant_message
                .content
                .contains("persistent-agent")
        );
        assert_eq!(response.changed_tasks.len(), 0);
        assert!(global_actions.iter().any(|action| {
            action.action_type == "inspect_workspace_file" && action.details["path"] == "Cargo.toml"
        }));

        Ok(())
    }

    #[tokio::test]
    async fn main_agent_can_list_workspace_directory_by_conversation() -> anyhow::Result<()> {
        let db = Db::connect("sqlite::memory:").await?;
        let agent = MainAgent::new(db.clone());

        let response = agent
            .handle_user_message(MainAgentMessageInput {
                content: "list workspace files".to_owned(),
            })
            .await?;
        let global_actions = db.list_global_actions().await?;

        assert!(response.assistant_message.content.contains("Directory: ."));
        assert!(response.assistant_message.content.contains("Cargo.toml"));
        assert_eq!(response.changed_tasks.len(), 0);
        assert!(global_actions.iter().any(|action| {
            action.action_type == "inspect_workspace_directory" && action.details["path"] == "."
        }));

        Ok(())
    }

    #[tokio::test]
    async fn main_agent_reports_workspace_file_read_errors_in_conversation() -> anyhow::Result<()> {
        let db = Db::connect("sqlite::memory:").await?;
        let agent = MainAgent::new(db.clone());

        let response = agent
            .handle_user_message(MainAgentMessageInput {
                content: "read file C:\\Windows\\win.ini".to_owned(),
            })
            .await?;
        let global_actions = db.list_global_actions().await?;

        assert!(
            response
                .assistant_message
                .content
                .contains("I could not inspect workspace file")
        );
        assert!(
            response
                .assistant_message
                .content
                .contains("must be relative")
        );
        assert_eq!(response.changed_tasks.len(), 0);
        assert!(global_actions.iter().any(|action| {
            action.action_type == "inspect_workspace_file_failed"
                && action.details["path"] == "C:\\Windows\\win.ini"
                && action.details["error"] == "workspace file path must be relative"
        }));

        Ok(())
    }

    #[tokio::test]
    async fn main_agent_can_explain_task_pool_state() -> anyhow::Result<()> {
        let db = Db::connect("sqlite::memory:").await?;
        let agent = MainAgent::new(db.clone());
        agent
            .create_task(CreateTask {
                title: "Queued work".to_owned(),
                description: "Ready to run".to_owned(),
                task_type: TaskType::OneOff,
                priority: 2,
                requested_skills: Vec::new(),
                schedule: None,
                created_by: "test".to_owned(),
            })
            .await?;

        let response = agent
            .handle_user_message(MainAgentMessageInput {
                content: "explain task pool state".to_owned(),
            })
            .await?;
        let global_actions = db.list_global_actions().await?;

        assert!(
            response
                .assistant_message
                .content
                .contains("Execution state")
        );
        assert!(response.assistant_message.content.contains("Queued work"));
        assert!(
            global_actions
                .iter()
                .any(|action| action.action_type == "explain_task_pool_state")
        );

        Ok(())
    }

    #[tokio::test]
    async fn main_agent_can_recommend_next_action_by_conversation() -> anyhow::Result<()> {
        let db = Db::connect("sqlite::memory:").await?;
        let agent = MainAgent::new(db.clone());
        agent
            .create_task(CreateTask {
                title: "Low priority cleanup".to_owned(),
                description: "Can wait".to_owned(),
                task_type: TaskType::OneOff,
                priority: 0,
                requested_skills: Vec::new(),
                schedule: None,
                created_by: "test".to_owned(),
            })
            .await?;
        let next = agent
            .create_task(CreateTask {
                title: "Fix release blocker".to_owned(),
                description: "Ready to run".to_owned(),
                task_type: TaskType::OneOff,
                priority: 5,
                requested_skills: Vec::new(),
                schedule: None,
                created_by: "test".to_owned(),
            })
            .await?;

        let response = agent
            .handle_user_message(MainAgentMessageInput {
                content: "what should I do next?".to_owned(),
            })
            .await?;
        let global_actions = db.list_global_actions().await?;

        assert_eq!(response.changed_tasks.len(), 0);
        assert!(!response.scheduler_tick_requested);
        assert!(
            response
                .assistant_message
                .content
                .contains("Recommended next action")
        );
        assert!(response.assistant_message.content.contains(&next.title));
        assert!(response.assistant_message.content.contains("run next task"));
        assert!(
            global_actions
                .iter()
                .any(|action| action.action_type == "recommend_next_action")
        );

        Ok(())
    }

    #[tokio::test]
    async fn main_agent_can_explain_why_task_is_not_running() -> anyhow::Result<()> {
        let db = Db::connect("sqlite::memory:").await?;
        let agent = MainAgent::new(db.clone());
        db.create_skill(
            CreateSkill {
                name: "deploy".to_owned(),
                description: "Deployment work".to_owned(),
                trigger_rules: vec!["deploy".to_owned()],
                tool_subset: vec!["shell".to_owned()],
                resource_path: None,
            },
            "test",
        )
        .await?;
        let dependency = agent
            .create_task(CreateTask {
                title: "Build package".to_owned(),
                description: "Build first".to_owned(),
                task_type: TaskType::OneOff,
                priority: 0,
                requested_skills: Vec::new(),
                schedule: None,
                created_by: "test".to_owned(),
            })
            .await?;
        let dependent = agent
            .create_task(CreateTask {
                title: "Deploy release".to_owned(),
                description: "Deploy after build".to_owned(),
                task_type: TaskType::OneOff,
                priority: 10,
                requested_skills: vec!["github".to_owned()],
                schedule: None,
                created_by: "test".to_owned(),
            })
            .await?;
        agent
            .add_task_dependency(dependent.id, dependency.id)
            .await?;

        let response = agent
            .handle_user_message(MainAgentMessageInput {
                content: "why is task Deploy release not running?".to_owned(),
            })
            .await?;
        let actions = db.list_task_actions(dependent.id).await?;

        assert!(
            response
                .assistant_message
                .content
                .contains("unfinished depend")
        );
        assert!(response.assistant_message.content.contains("Build package"));
        assert!(
            response
                .assistant_message
                .content
                .contains("Requested skills: github")
        );
        assert!(
            response
                .assistant_message
                .content
                .contains("Matched skills: deploy")
        );
        assert!(
            response
                .assistant_message
                .content
                .contains("Active skills for the next worker run: github, deploy")
        );
        assert!(actions.iter().any(|action| {
            action.action_type == "explain_task_state"
                && action.details["active_skills"] == serde_json::json!(["github", "deploy"])
        }));

        Ok(())
    }

    #[tokio::test]
    async fn main_agent_explains_resource_lock_conflicts() -> anyhow::Result<()> {
        let db = Db::connect("sqlite::memory:").await?;
        let agent = MainAgent::new(db.clone());
        let running = agent
            .create_task(CreateTask {
                title: "Fix repository issue".to_owned(),
                description: "Work in the repository".to_owned(),
                task_type: TaskType::OneOff,
                priority: 0,
                requested_skills: Vec::new(),
                schedule: None,
                created_by: "test".to_owned(),
            })
            .await?;
        let blocked = agent
            .create_task(CreateTask {
                title: "Update repository docs".to_owned(),
                description: "Also needs the repository".to_owned(),
                task_type: TaskType::OneOff,
                priority: 10,
                requested_skills: Vec::new(),
                schedule: None,
                created_by: "test".to_owned(),
            })
            .await?;
        agent
            .add_task_resource_lock(running.id, "repo:oh-my-harness/Persistent-Agent")
            .await?;
        agent
            .add_task_resource_lock(blocked.id, "repo:oh-my-harness/Persistent-Agent")
            .await?;
        db.set_task_status(running.id, TaskStatus::Running, "test", None)
            .await?;

        let response = agent
            .handle_user_message(MainAgentMessageInput {
                content: "why is task Update repository docs not running?".to_owned(),
            })
            .await?;
        let actions = db.list_task_actions(blocked.id).await?;

        assert!(
            response
                .assistant_message
                .content
                .contains("resource lock conflict")
        );
        assert!(
            response
                .assistant_message
                .content
                .contains("repo:oh-my-harness/Persistent-Agent")
        );
        assert!(
            response
                .assistant_message
                .content
                .contains("Fix repository issue")
        );
        assert!(actions.iter().any(|action| {
            action.action_type == "explain_task_state"
                && action.details["resource_lock_conflict_count"] == 1
        }));

        Ok(())
    }

    #[tokio::test]
    async fn main_agent_can_request_user_clarification() -> anyhow::Result<()> {
        let db = Db::connect("sqlite::memory:").await?;
        let agent = MainAgent::new(db.clone());
        let task = agent
            .create_task(CreateTask {
                title: "Deploy release".to_owned(),
                description: "Deploy the release".to_owned(),
                task_type: TaskType::OneOff,
                priority: 0,
                requested_skills: Vec::new(),
                schedule: None,
                created_by: "test".to_owned(),
            })
            .await?;

        let response = agent
            .handle_user_message(MainAgentMessageInput {
                content: "ask clarification for task Deploy release: Which environment?".to_owned(),
            })
            .await?;
        let updated = db.get_task(task.id).await?;
        let messages = db.list_task_conversation_messages(task.id, 20).await?;
        let actions = db.list_task_actions(task.id).await?;

        assert_eq!(updated.status, TaskStatus::WaitingForUser);
        assert_eq!(
            updated.blocked_reason.as_deref(),
            Some("Which environment?")
        );
        assert!(
            messages
                .iter()
                .any(|message| message.role == "assistant"
                    && message.content == "Which environment?")
        );
        assert!(
            actions
                .iter()
                .any(|action| action.action_type == "request_user_clarification")
        );
        assert_eq!(response.changed_tasks.len(), 1);

        Ok(())
    }

    #[tokio::test]
    async fn main_agent_can_reply_to_blocked_task_by_conversation() -> anyhow::Result<()> {
        let db = Db::connect("sqlite::memory:").await?;
        let agent = MainAgent::new(db.clone());
        let task = agent
            .create_task(CreateTask {
                title: "Deploy release".to_owned(),
                description: "Deploy the release".to_owned(),
                task_type: TaskType::OneOff,
                priority: 0,
                requested_skills: Vec::new(),
                schedule: None,
                created_by: "test".to_owned(),
            })
            .await?;
        agent
            .request_user_clarification(task.id, "Which environment?")
            .await?;

        let response = agent
            .handle_user_message(MainAgentMessageInput {
                content: "reply to task Deploy release: use production".to_owned(),
            })
            .await?;
        let updated = db.get_task(task.id).await?;
        let messages = db.list_task_conversation_messages(task.id, 20).await?;
        let actions = db.list_task_actions(task.id).await?;

        assert_eq!(updated.status, TaskStatus::Queued);
        assert!(updated.blocked_reason.is_none());
        assert_eq!(response.changed_tasks.len(), 1);
        assert!(
            response
                .assistant_message
                .content
                .contains("moved it back to the queue")
        );
        assert!(
            messages
                .iter()
                .any(|message| message.role == "user" && message.content == "use production")
        );
        assert!(
            actions
                .iter()
                .any(|action| action.action_type == "reply_to_task")
        );

        Ok(())
    }

    #[tokio::test]
    async fn main_agent_can_reply_to_the_only_blocked_task() -> anyhow::Result<()> {
        let db = Db::connect("sqlite::memory:").await?;
        let agent = MainAgent::new(db.clone());
        let task = agent
            .create_task(CreateTask {
                title: "Deploy release".to_owned(),
                description: "Deploy the release".to_owned(),
                task_type: TaskType::OneOff,
                priority: 0,
                requested_skills: Vec::new(),
                schedule: None,
                created_by: "test".to_owned(),
            })
            .await?;
        agent
            .request_user_clarification(task.id, "Which environment?")
            .await?;

        let response = agent
            .handle_user_message(MainAgentMessageInput {
                content: "reply to blocked task: use staging first".to_owned(),
            })
            .await?;
        let updated = db.get_task(task.id).await?;
        let messages = db.list_task_conversation_messages(task.id, 20).await?;

        assert_eq!(updated.status, TaskStatus::Queued);
        assert_eq!(response.changed_tasks.len(), 1);
        assert!(response.scheduler_tick_requested);
        assert!(
            messages
                .iter()
                .any(|message| message.role == "user" && message.content == "use staging first")
        );

        Ok(())
    }

    #[tokio::test]
    async fn main_agent_reply_to_active_task_does_not_request_scheduler_scan() -> anyhow::Result<()>
    {
        let db = Db::connect("sqlite::memory:").await?;
        let agent = MainAgent::new(db.clone());
        agent
            .create_task(CreateTask {
                title: "Deploy release".to_owned(),
                description: "Deploy the release".to_owned(),
                task_type: TaskType::OneOff,
                priority: 0,
                requested_skills: Vec::new(),
                schedule: None,
                created_by: "test".to_owned(),
            })
            .await?;

        let response = agent
            .handle_user_message(MainAgentMessageInput {
                content: "reply to task Deploy release: keep the release notes concise".to_owned(),
            })
            .await?;

        assert_eq!(response.changed_tasks.len(), 1);
        assert!(!response.scheduler_tick_requested);
        assert!(
            response
                .assistant_message
                .content
                .contains("Sent your reply")
        );

        Ok(())
    }

    #[tokio::test]
    async fn main_agent_can_split_goal_into_tasks() -> anyhow::Result<()> {
        let db = Db::connect("sqlite::memory:").await?;
        let agent = MainAgent::new(db.clone());

        let response = agent
            .handle_user_message(MainAgentMessageInput {
                content: "split goal: investigate issue; write fix; run tests".to_owned(),
            })
            .await?;
        let tasks = db.list_tasks().await?;

        assert_eq!(response.changed_tasks.len(), 3);
        assert_eq!(tasks.len(), 3);
        assert!(tasks.iter().any(|task| task.title == "investigate issue"));
        assert!(tasks.iter().all(|task| task.task_type == TaskType::OneOff));
        assert!(
            response
                .assistant_message
                .content
                .contains("Created 3 split task")
        );

        Ok(())
    }
}
