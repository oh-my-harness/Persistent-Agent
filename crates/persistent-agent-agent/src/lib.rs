use persistent_agent_db::Db;
use persistent_agent_domain::{
    ConversationId, ConversationMessage, CreateTask, Task, TaskId, TaskStatus, TaskType, UpdateTask,
};
use serde::{Deserialize, Serialize};

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
            } => {
                let task = self
                    .create_task(CreateTask {
                        title,
                        description,
                        task_type,
                        priority,
                        requested_skills: Vec::new(),
                        schedule: None,
                        created_by: "user".to_owned(),
                    })
                    .await?;
                let reply = format!(
                    "已创建任务：{}。当前状态是 {}，优先级 {}。",
                    task.title, task.status, task.priority
                );
                changed_tasks.push(task);
                reply
            }
            MainAgentIntent::PauseTask { selector } => match self.find_task(&selector).await? {
                Ok(task) => {
                    let task = self.pause_task(task.id).await?;
                    let reply = format!("已暂停任务：{}。", task.title);
                    changed_tasks.push(task);
                    reply
                }
                Err(reply) => reply,
            },
            MainAgentIntent::ResumeTask { selector } => match self.find_task(&selector).await? {
                Ok(task) => {
                    let task = self.resume_task(task.id).await?;
                    let reply = format!("已恢复任务：{}。", task.title);
                    changed_tasks.push(task);
                    reply
                }
                Err(reply) => reply,
            },
            MainAgentIntent::CancelTask { selector } => match self.find_task(&selector).await? {
                Ok(task) => {
                    let task = self.cancel_task(task.id).await?;
                    let reply = format!("已取消任务：{}。", task.title);
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
                            format!("已将任务 {} 的优先级调整为 {}。", task.title, task.priority);
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
                        "已将任务 {} 移动到队列位置 {}。",
                        task.title, task.queue_position
                    );
                    changed_tasks.push(task);
                    reply
                }
                Err(reply) => reply,
            },
            MainAgentIntent::Summarize => {
                let summary = self.summarize_task_pool().await?;
                format!(
                    "当前共有 {} 个任务：{} 个排队中，{} 个运行中，{} 个等待用户，{} 个已完成，{} 个已暂停。",
                    summary.total,
                    summary.queued,
                    summary.running,
                    summary.waiting_for_user,
                    summary.completed,
                    summary.paused
                )
            }
            MainAgentIntent::Help => {
                "我可以通过对话帮你创建任务、暂停/恢复/取消任务、调整优先级、移动队列位置，或总结任务池。比如：“创建循环任务：每天检查 GitHub issue 优先级 3”。".to_owned()
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
                "请告诉我要调整哪个任务，可以使用标题片段或任务 id。".to_owned()
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
            [] => Ok(Err(format!("没有找到匹配“{selector}”的任务。"))),
            _ => Ok(Err(format!(
                "找到了 {} 个匹配“{selector}”的任务，请使用更具体的标题或 id。",
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
    Summarize,
    Help,
}

fn parse_intent(content: &str) -> MainAgentIntent {
    let trimmed = content.trim();
    let normalized = trimmed.to_lowercase();

    if contains_any(
        &normalized,
        &["总结", "概览", "任务池", "summary", "summarize"],
    ) {
        return MainAgentIntent::Summarize;
    }

    if let Some(priority) = extract_priority(&normalized) {
        if !is_create_request(&normalized)
            && contains_any(
                &normalized,
                &["reprioritize", "set priority", "change priority", "优先级"],
            )
        {
            return MainAgentIntent::ReprioritizeTask {
                selector: extract_task_selector(trimmed, &["priority", "to", "优先级", "为", "到"]),
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
                    "队列",
                    "排序",
                    "移动",
                ],
            )
        {
            return MainAgentIntent::ReorderTask {
                selector: extract_task_selector(
                    trimmed,
                    &[
                        "queue", "position", "to", "队列", "排序", "位置", "为", "到",
                    ],
                ),
                queue_position,
            };
        }
    }

    if contains_any(&normalized, &["pause task", "暂停任务"]) || normalized.starts_with("pause ")
    {
        return MainAgentIntent::PauseTask {
            selector: extract_task_selector(trimmed, &[]),
        };
    }

    if contains_any(&normalized, &["resume task", "unpause task", "恢复任务"])
        || normalized.starts_with("resume ")
    {
        return MainAgentIntent::ResumeTask {
            selector: extract_task_selector(trimmed, &[]),
        };
    }

    if contains_any(&normalized, &["cancel task", "取消任务"]) || normalized.starts_with("cancel ")
    {
        return MainAgentIntent::CancelTask {
            selector: extract_task_selector(trimmed, &[]),
        };
    }

    if is_create_request(&normalized) {
        let task_type = if contains_any(&normalized, &["循环", "定期", "recurring", "repeat"]) {
            TaskType::Recurring
        } else {
            TaskType::OneOff
        };
        let priority = extract_priority(&normalized).unwrap_or(0);
        let title = extract_title(trimmed);

        return MainAgentIntent::CreateTask {
            title,
            description: trimmed.to_owned(),
            task_type,
            priority,
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
        &["创建", "新建", "添加", "加一个", "create", "add task"],
    )
}

fn extract_title(content: &str) -> String {
    for separator in ["：", ":", "，", ",", "\n"] {
        if let Some((_, tail)) = content.split_once(separator) {
            let title = tail.trim();
            if !title.is_empty() {
                return clamp_title(title);
            }
        }
    }

    let title = content
        .replace("创建循环任务", "")
        .replace("创建一次性任务", "")
        .replace("创建任务", "")
        .replace("新建任务", "")
        .replace("添加任务", "")
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
        "任务",
        "暂停",
        "恢复",
        "取消",
        "调整",
        "移动",
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
        .trim_matches([':', '：', ',', '，', '.', '。', '"', '\''])
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
    extract_number_after_any(normalized, &["priority", "优先级"])
}

fn extract_queue_position(normalized: &str) -> Option<i64> {
    extract_number_after_any(
        normalized,
        &[
            "queue position",
            "queue",
            "position",
            "队列",
            "排序",
            "位置",
        ],
    )
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
            }
        );
    }

    #[test]
    fn parses_recurring_chinese_create_task() {
        let content = "创建循环任务：每天检查仓库 issue 优先级 3";
        let intent = parse_intent(content);

        assert_eq!(
            intent,
            MainAgentIntent::CreateTask {
                title: "每天检查仓库 issue 优先级 3".to_owned(),
                description: content.to_owned(),
                task_type: TaskType::Recurring,
                priority: 3,
            }
        );
    }

    #[test]
    fn parses_task_pool_summary() {
        assert_eq!(parse_intent("总结任务池"), MainAgentIntent::Summarize);
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
            parse_intent("暂停任务 检查 GitHub issues"),
            MainAgentIntent::PauseTask {
                selector: "检查 GitHub issues".to_owned()
            }
        );
        assert_eq!(
            parse_intent("恢复任务 检查 GitHub issues"),
            MainAgentIntent::ResumeTask {
                selector: "检查 GitHub issues".to_owned()
            }
        );
        assert_eq!(
            parse_intent("取消任务 检查 GitHub issues"),
            MainAgentIntent::CancelTask {
                selector: "检查 GitHub issues".to_owned()
            }
        );
        assert_eq!(
            parse_intent("调整任务 检查 GitHub issues 优先级为 8"),
            MainAgentIntent::ReprioritizeTask {
                selector: "检查 GitHub issues".to_owned(),
                priority: 8,
            }
        );
        assert_eq!(
            parse_intent("移动任务 检查 GitHub issues 到队列位置 4"),
            MainAgentIntent::ReorderTask {
                selector: "检查 GitHub issues".to_owned(),
                queue_position: 4,
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
        assert!(response.assistant_message.content.contains("已暂停任务"));

        Ok(())
    }
}
