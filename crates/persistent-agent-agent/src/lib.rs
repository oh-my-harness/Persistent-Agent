use persistent_agent_db::Db;
use persistent_agent_domain::{
    ConversationId, ConversationMessage, CreateTask, Task, TaskId, TaskNote, TaskStatus, TaskType,
    UpdateTask,
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

#[derive(Clone)]
pub struct MainAgent {
    db: Db,
}

impl MainAgent {
    pub fn new(db: Db) -> Self {
        Self { db }
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

        let intent = parse_intent(&input.content);
        let mut changed_tasks = Vec::new();
        let reply = match intent {
            MainAgentIntent::CreateTask {
                title,
                description,
                task_type,
                priority,
                interval_seconds,
            } => {
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
                        requested_skills: Vec::new(),
                        schedule,
                        created_by: "user".to_owned(),
                    })
                    .await?;
                let reply = format!(
                    "Created task '{}'. Status: {}, priority: {}.",
                    task.title, task.status, task.priority
                );
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
            MainAgentIntent::ExplainTaskPool => self.explain_task_pool_state().await?,
            MainAgentIntent::ExplainTask { selector } => match self.find_task(&selector).await? {
                Ok(task) => self.explain_task_state(task.id).await?,
                Err(reply) => reply,
            },
            MainAgentIntent::Help => {
                "I can create tasks, list tasks, explain task state, pause/resume/cancel tasks, set priority, reorder the queue, add notes, add/remove task dependencies, convert tasks between one-off and recurring, or summarize the task pool. Example: why is task Deploy release not running?".to_owned()
            }
        };

        let assistant_message = self
            .db
            .add_conversation_message(conversation.id, None, "assistant", &reply)
            .await?;

        Ok(MainAgentMessageResponse {
            conversation_id: conversation.id,
            user_message,
            assistant_message,
            changed_tasks,
        })
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
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum MainAgentIntent {
    CreateTask {
        title: String,
        description: String,
        task_type: TaskType,
        priority: i64,
        interval_seconds: Option<i64>,
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
    ListTasks,
    ExplainTaskPool,
    ExplainTask {
        selector: String,
    },
    Summarize,
    Help,
}

fn parse_intent(content: &str) -> MainAgentIntent {
    let trimmed = content.trim();
    let normalized = trimmed.to_lowercase();

    if let Some(intent) = parse_explain_intent(trimmed, &normalized) {
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

    if is_list_tasks_request(&normalized) {
        return MainAgentIntent::ListTasks;
    }

    if let Some(intent) = parse_dependency_intent(trimmed, &normalized) {
        return intent;
    }

    if let Some(intent) = parse_note_intent(trimmed, &normalized) {
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
        let title = extract_title(trimmed);

        return MainAgentIntent::CreateTask {
            title,
            description: trimmed.to_owned(),
            task_type,
            priority,
            interval_seconds,
        };
    }

    MainAgentIntent::Help
}

fn contains_any(haystack: &str, needles: &[&str]) -> bool {
    needles.iter().any(|needle| haystack.contains(needle))
}

fn is_create_request(normalized: &str) -> bool {
    contains_any(
        normalized,
        &[ZH_CREATE, ZH_NEW, ZH_ADD, ZH_ADD_ONE, "create", "add task"],
    )
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
        "add note to task",
        "add note",
        "note to task",
        "note task",
        "note",
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
}
