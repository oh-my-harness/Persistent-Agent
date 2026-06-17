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
    ConversationId, ConversationMessage, CreateSkill, CreateTask, Memory, MemoryId, MemoryStatus,
    Skill, Task, TaskAction, TaskArtifact, TaskId, TaskNote, TaskResourceLock, TaskStatus,
    TaskType, UpdateSkill, UpdateTask,
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

        self.db
            .record_action(
                Some(id),
                "main_agent",
                "explain_task_state",
                serde_json::json!({ "status": task.status }),
            )
            .await?;

        Ok(format_task_explanation(&task, &dependency_states))
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
        let scheduler_tick_requested = matches!(intent, MainAgentIntent::RunSchedulerTick);
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
            MainAgentIntent::ListGlobalActions => self.list_main_agent_actions().await?,
            MainAgentIntent::ExplainTaskPool => self.explain_task_pool_state().await?,
            MainAgentIntent::ExplainTask { selector } => match self.find_task(&selector).await? {
                Ok(task) => self.explain_task_state(task.id).await?,
                Err(reply) => reply,
            },
            MainAgentIntent::ListTaskArtifacts { selector } => {
                match self.find_task(&selector).await? {
                    Ok(task) => self.list_task_artifacts(task.id).await?,
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
            MainAgentIntent::ListMemories { filter } => self.list_memories_for_review(filter).await?,
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
                "Scheduler scan requested. I will ask the scheduler to check the task pool now."
                    .to_owned()
            }
            MainAgentIntent::Help => {
                "I can create tasks, split goals into tasks, list tasks, create/list/delete skills, show task artifacts, show memory candidates, show main-agent audit actions, explain task state, inspect workspace status, preview workspace files, request user clarification, pause/resume/cancel tasks, set priority, reorder the queue, add notes, add/remove requested skills, add/remove task dependencies, add/remove resource locks, approve/reject memory candidates, run a scheduler scan, convert tasks between one-off and recurring, or summarize the task pool. Example: split goal: investigate issue; write fix; run tests.".to_owned()
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

        let matches = self
            .db
            .list_tasks()
            .await?
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
    pub recent_messages: Vec<ConversationMessage>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
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
    ReprioritizeTask {
        selector: String,
        priority: i64,
    },
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
            Self::ReprioritizeTask { selector, priority } => {
                MainAgentIntent::ReprioritizeTask { selector, priority }
            }
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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

#[derive(Debug, Clone, PartialEq, Eq)]
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
    ListTasks,
    ListGlobalActions,
    ExplainTaskPool,
    ExplainTask {
        selector: String,
    },
    ListTaskArtifacts {
        selector: String,
    },
    InspectWorkspace,
    InspectWorkspaceFile {
        path: String,
    },
    ApproveMemory {
        selector: String,
    },
    RejectMemory {
        selector: String,
    },
    ListMemories {
        filter: MemoryListFilter,
    },
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

    if let Some(intent) = parse_explain_intent(trimmed, &normalized) {
        return intent;
    }

    if let Some(intent) = parse_split_intent(trimmed, &normalized) {
        return intent;
    }

    if let Some(path) = extract_workspace_file_inspection_path(trimmed, &normalized) {
        return MainAgentIntent::InspectWorkspaceFile { path };
    }

    if is_workspace_inspection_request(&normalized) {
        return MainAgentIntent::InspectWorkspace;
    }

    if is_scheduler_scan_request(&normalized) {
        return MainAgentIntent::RunSchedulerTick;
    }

    if let Some(intent) = parse_skill_definition_intent(trimmed, &normalized) {
        return intent;
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
        "plan_reprioritize_task",
        "plan_list_tasks",
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
    registry.register(Arc::new(PlanReprioritizeTaskTool::new(state.clone())));
    registry.register(Arc::new(PlanSimpleIntentTool::new(
        state.clone(),
        "plan_list_tasks",
        "Plan to list the current task pool.",
        MainAgentPlan::ListTasks,
    )));
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
            let task_type = match args
                .get("task_type")
                .and_then(|value| value.as_str())
                .unwrap_or("one_off")
            {
                "recurring" => TaskType::Recurring,
                _ => TaskType::OneOff,
            };
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
            };
            self.state.lock().await.plan = Some(plan);
            Ok(planner_tool_result("planned task state change"))
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
        "User message:\n{}\n\nTask pool summary:\n{}\n\nRecent main conversation:\n{}\n\nSupported planning tools:\n- plan_create_task: create one one-off or recurring task.\n- plan_split_tasks: split a goal into multiple one-off tasks.\n- plan_pause_task: pause one existing task by id or title fragment.\n- plan_resume_task: resume one existing task by id or title fragment.\n- plan_cancel_task: cancel one existing task by id or title fragment.\n- plan_reprioritize_task: set one existing task's priority.\n- plan_list_tasks: list tasks.\n- plan_summarize_task_pool: summarize task pool state.\n- plan_scheduler_scan: run one scheduler scan.\n\nCall one tool only when the user intent is clear.",
        context.user_message,
        format_advisor_summary(&context.task_pool_summary),
        format_advisor_recent_messages(&context.recent_messages),
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

fn clean_workspace_path(value: &str) -> String {
    value
        .trim()
        .trim_start_matches([':', '\u{ff1a}'])
        .trim()
        .trim_matches(['`', '"', '\''])
        .trim()
        .to_owned()
}

fn format_task_list(tasks: &[Task]) -> String {
    if tasks.is_empty() {
        return "Task pool is empty.".to_owned();
    }

    let mut lines = vec![format!("Task pool has {} task(s):", tasks.len())];
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

fn format_task_explanation(task: &Task, dependencies: &[Task]) -> String {
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
            if unsatisfied.is_empty() {
                lines.push("It is queued and eligible for the scheduler once it reaches the front of priority and queue order.".to_owned());
            } else {
                lines.push(format!(
                    "It is queued but blocked by {} unfinished dependenc{}:",
                    unsatisfied.len(),
                    if unsatisfied.len() == 1 { "y" } else { "ies" }
                ));
                for dependency in unsatisfied {
                    lines.push(format!("- '{}' is {}", dependency.title, dependency.status));
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
        let selector = extract_memory_selector(content);
        return (!selector.is_empty()).then_some(MainAgentIntent::ApproveMemory { selector });
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
        let selector = extract_memory_selector(content);
        return (!selector.is_empty()).then_some(MainAgentIntent::RejectMemory { selector });
    }

    None
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
        "memory candidate",
        "long-term memory",
        "approve",
        "accept",
        "reject",
        "discard",
        "memory",
        ZH_LONG_TERM_MEMORY,
        ZH_MEMORY,
        ZH_APPROVE,
        ZH_ACCEPT,
        ZH_REJECT,
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
        "task",
        ZH_CONVERT_TASK,
        ZH_TASK,
        ZH_PAUSE,
        ZH_RESUME,
        ZH_CANCEL,
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
    extract_number_after_last_any(normalized, &["every", "interval", ZH_INTERVAL, "\u{6bcf} "])
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

fn extract_number_after_last_any(normalized: &str, markers: &[&str]) -> Option<i64> {
    markers
        .iter()
        .filter_map(|marker| normalized.rfind(marker).map(|index| (index, *marker)))
        .max_by_key(|(index, _)| *index)
        .and_then(|(index, marker)| {
            let after = &normalized[index + marker.len()..];
            let digits: String = after
                .chars()
                .skip_while(|ch| !ch.is_ascii_digit() && *ch != '-')
                .take_while(|ch| ch.is_ascii_digit() || *ch == '-')
                .collect();
            digits.parse().ok()
        })
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
    }

    #[test]
    fn parses_task_dependency_intents() {
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
    fn parses_memory_review_intents() {
        assert_eq!(
            parse_intent("show memory candidates"),
            MainAgentIntent::ListMemories {
                filter: MemoryListFilter::Pending,
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
            parse_intent("reject memory candidate noisy temporary note"),
            MainAgentIntent::RejectMemory {
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
        assert_eq!(contexts.lock().expect("planner contexts lock").len(), 1);

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
        db.create_memory(
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
    async fn main_agent_can_explain_why_task_is_not_running() -> anyhow::Result<()> {
        let db = Db::connect("sqlite::memory:").await?;
        let agent = MainAgent::new(db.clone());
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
                requested_skills: Vec::new(),
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
            actions
                .iter()
                .any(|action| action.action_type == "explain_task_state")
        );

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
