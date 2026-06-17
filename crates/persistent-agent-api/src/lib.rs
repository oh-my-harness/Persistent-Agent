use std::{collections::HashSet, convert::Infallible, time::Duration};

use axum::{
    Json, Router,
    extract::{Path, State},
    http::StatusCode,
    response::{
        IntoResponse, Sse,
        sse::{Event, KeepAlive},
    },
    routing::{delete, get, patch, post},
};
use persistent_agent_agent::{MainAgent, MainAgentMessageInput, TaskPoolSummary};
use persistent_agent_db::Db;
use persistent_agent_domain::{
    ConversationMessage, CreateSkill, CreateTask, Memory, MemoryId, MemoryStatus, Skill, SkillId,
    Task, TaskAction, TaskArtifact, TaskAttempt, TaskAttemptEvent, TaskDependency, TaskId,
    TaskNote, TaskResourceLock, UpdateMemory, UpdateSkill, UpdateTask,
};
use persistent_agent_scheduler::{Scheduler, SchedulerPolicy, SchedulerTick, WorkerBackend};
use serde::{Deserialize, Serialize};
use tokio::sync::broadcast;
use tokio::time::{interval, sleep};
use tokio_stream::{Stream, StreamExt, wrappers::BroadcastStream};
use tower_http::{cors::CorsLayer, trace::TraceLayer};
use uuid::Uuid;

#[derive(Clone)]
pub struct AppState {
    pub db: Db,
    pub main_agent: MainAgent,
    pub scheduler: Scheduler<WorkerBackend>,
    pub events: EventBus,
}

impl AppState {
    pub fn new(db: Db, worker: WorkerBackend) -> Self {
        Self::new_with_scheduler_policy(db, worker, SchedulerPolicy::serial())
    }

    pub fn new_with_scheduler_policy(
        db: Db,
        worker: WorkerBackend,
        scheduler_policy: SchedulerPolicy,
    ) -> Self {
        let main_agent = MainAgent::new(db.clone());
        let scheduler = Scheduler::with_policy(db.clone(), worker, scheduler_policy);
        Self {
            db,
            main_agent,
            scheduler,
            events: EventBus::new(),
        }
    }
}

#[derive(Clone)]
pub struct EventBus {
    tx: broadcast::Sender<AppEvent>,
}

impl EventBus {
    pub fn new() -> Self {
        let (tx, _) = broadcast::channel(256);
        Self { tx }
    }

    pub fn send(&self, event: AppEvent) {
        let _ = self.tx.send(event);
    }

    pub fn subscribe(&self) -> broadcast::Receiver<AppEvent> {
        self.tx.subscribe()
    }
}

impl Default for EventBus {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "type")]
pub enum AppEvent {
    TaskChanged { task: Task },
    MainAgentReply { message: ConversationMessage },
    MainAgentAction { action: TaskAction },
    SchedulerTick { tick: SchedulerTick },
    Heartbeat,
}

pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/healthz", get(healthz))
        .route("/api/tasks", get(list_tasks).post(create_task))
        .route("/api/tasks/{id}", get(get_task).patch(update_task))
        .route("/api/tasks/{id}/reprioritize", post(reprioritize_task))
        .route("/api/tasks/{id}/reorder", post(reorder_task))
        .route(
            "/api/tasks/{id}/dependencies",
            get(task_dependencies).post(add_task_dependency),
        )
        .route(
            "/api/tasks/{id}/dependencies/{depends_on_id}",
            delete(remove_task_dependency),
        )
        .route("/api/tasks/{id}/notes", get(task_notes).post(add_task_note))
        .route(
            "/api/tasks/{id}/resource-locks",
            get(task_resource_locks)
                .post(add_task_resource_lock)
                .delete(remove_task_resource_lock),
        )
        .route(
            "/api/tasks/{id}/messages",
            get(task_messages).post(send_task_message),
        )
        .route("/api/tasks/{id}/history", get(task_history))
        .route("/api/tasks/{id}/pause", post(pause_task))
        .route("/api/tasks/{id}/resume", post(resume_task))
        .route("/api/tasks/{id}/cancel", post(cancel_task))
        .route("/api/main-agent/task-pool-summary", get(task_pool_summary))
        .route(
            "/api/main-agent/messages",
            get(main_agent_messages).post(send_main_agent_message),
        )
        .route("/api/main-agent/actions", get(main_agent_actions))
        .route("/api/memories", get(list_memories))
        .route(
            "/api/memories/{id}",
            patch(update_memory).delete(delete_memory),
        )
        .route("/api/memories/{id}/approve", post(approve_memory))
        .route("/api/memories/{id}/reject", post(reject_memory))
        .route("/api/skills", get(list_skills).post(create_skill))
        .route("/api/skills/{id}", patch(update_skill).delete(delete_skill))
        .route("/api/scheduler/state", get(scheduler_state))
        .route("/api/scheduler/tick", post(run_scheduler_tick))
        .route("/api/events", get(events))
        .layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}

async fn healthz() -> Json<serde_json::Value> {
    Json(serde_json::json!({ "ok": true }))
}

async fn list_tasks(State(state): State<AppState>) -> Result<Json<Vec<Task>>, ApiError> {
    Ok(Json(state.db.list_tasks().await?))
}

async fn get_task(
    State(state): State<AppState>,
    Path(id): Path<Uuid>,
) -> Result<Json<Task>, ApiError> {
    Ok(Json(state.db.get_task(id).await?))
}

async fn create_task(
    State(state): State<AppState>,
    Json(input): Json<CreateTask>,
) -> Result<Json<Task>, ApiError> {
    let task = state.main_agent.create_task(input).await?;
    state
        .events
        .send(AppEvent::TaskChanged { task: task.clone() });
    Ok(Json(task))
}

async fn update_task(
    State(state): State<AppState>,
    Path(id): Path<Uuid>,
    Json(input): Json<UpdateTask>,
) -> Result<Json<Task>, ApiError> {
    let task = state.main_agent.update_task(id, input).await?;
    state
        .events
        .send(AppEvent::TaskChanged { task: task.clone() });
    Ok(Json(task))
}

#[derive(Debug, Deserialize)]
struct ReprioritizeRequest {
    priority: i64,
}

async fn reprioritize_task(
    State(state): State<AppState>,
    Path(id): Path<TaskId>,
    Json(input): Json<ReprioritizeRequest>,
) -> Result<Json<Task>, ApiError> {
    let task = state
        .main_agent
        .reprioritize_task(id, input.priority)
        .await?;
    state
        .events
        .send(AppEvent::TaskChanged { task: task.clone() });
    Ok(Json(task))
}

#[derive(Debug, Deserialize)]
struct ReorderRequest {
    queue_position: i64,
}

async fn reorder_task(
    State(state): State<AppState>,
    Path(id): Path<TaskId>,
    Json(input): Json<ReorderRequest>,
) -> Result<Json<Task>, ApiError> {
    let task = state
        .main_agent
        .reorder_task(id, input.queue_position)
        .await?;
    state
        .events
        .send(AppEvent::TaskChanged { task: task.clone() });
    Ok(Json(task))
}

async fn task_messages(
    State(state): State<AppState>,
    Path(id): Path<TaskId>,
) -> Result<Json<Vec<ConversationMessage>>, ApiError> {
    Ok(Json(
        state.db.list_task_conversation_messages(id, 100).await?,
    ))
}

#[derive(Debug, Deserialize)]
struct TaskMessageRequest {
    content: String,
}

#[derive(Debug, Serialize)]
struct TaskMessageResponse {
    user_message: ConversationMessage,
    assistant_message: Option<ConversationMessage>,
    task: Task,
}

async fn send_task_message(
    State(state): State<AppState>,
    Path(id): Path<TaskId>,
    Json(input): Json<TaskMessageRequest>,
) -> Result<Json<TaskMessageResponse>, ApiError> {
    let content = input.content.trim();
    if content.is_empty() {
        return Err(ApiError::bad_request("message content cannot be empty"));
    }

    let task = state.db.get_task(id).await?;
    let Some(conversation_id) = task.conversation_id else {
        return Err(ApiError::bad_request("task has no conversation"));
    };

    let user_message = state
        .db
        .add_conversation_message(conversation_id, Some(id), "user", content)
        .await?;

    let mut assistant_message = None;
    let task = if task.status == persistent_agent_domain::TaskStatus::WaitingForUser {
        let resumed = state.main_agent.resume_task(id).await?;
        assistant_message = Some(
            state
                .db
                .add_conversation_message(
                    conversation_id,
                    Some(id),
                    "assistant",
                    "Thanks, I have the extra context and moved this task back to the queue.",
                )
                .await?,
        );
        resumed
    } else {
        task
    };

    state
        .events
        .send(AppEvent::TaskChanged { task: task.clone() });

    Ok(Json(TaskMessageResponse {
        user_message,
        assistant_message,
        task,
    }))
}

#[derive(Debug, Serialize)]
struct TaskHistoryResponse {
    attempts: Vec<TaskAttempt>,
    attempt_events: Vec<TaskAttemptEvent>,
    artifacts: Vec<TaskArtifact>,
    memory_candidates: Vec<Memory>,
    dependencies: Vec<TaskDependency>,
    resource_locks: Vec<TaskResourceLock>,
    notes: Vec<TaskNote>,
    actions: Vec<TaskAction>,
}

async fn task_history(
    State(state): State<AppState>,
    Path(id): Path<TaskId>,
) -> Result<Json<TaskHistoryResponse>, ApiError> {
    state.db.get_task(id).await?;
    Ok(Json(TaskHistoryResponse {
        attempts: state.db.list_task_attempts(id).await?,
        attempt_events: state.db.list_task_attempt_events(id).await?,
        artifacts: state.db.list_task_artifacts(id).await?,
        memory_candidates: state.db.list_task_memories(id).await?,
        dependencies: state.db.list_task_dependencies(id).await?,
        resource_locks: state.db.list_task_resource_locks(id).await?,
        notes: state.db.list_task_notes(id).await?,
        actions: state.db.list_task_actions(id).await?,
    }))
}

async fn task_dependencies(
    State(state): State<AppState>,
    Path(id): Path<TaskId>,
) -> Result<Json<Vec<TaskDependency>>, ApiError> {
    state.db.get_task(id).await?;
    Ok(Json(state.db.list_task_dependencies(id).await?))
}

#[derive(Debug, Deserialize)]
struct TaskDependencyRequest {
    depends_on_task_id: TaskId,
}

async fn add_task_dependency(
    State(state): State<AppState>,
    Path(id): Path<TaskId>,
    Json(input): Json<TaskDependencyRequest>,
) -> Result<Json<TaskDependency>, ApiError> {
    let task = state
        .main_agent
        .add_task_dependency(id, input.depends_on_task_id)
        .await?;
    let dependency = state
        .db
        .get_task_dependency(id, input.depends_on_task_id)
        .await?;
    state.events.send(AppEvent::TaskChanged { task });
    Ok(Json(dependency))
}

async fn remove_task_dependency(
    State(state): State<AppState>,
    Path((id, depends_on_id)): Path<(TaskId, TaskId)>,
) -> Result<Json<TaskDependency>, ApiError> {
    let dependency = state.db.get_task_dependency(id, depends_on_id).await?;
    let task = state
        .main_agent
        .remove_task_dependency(id, depends_on_id)
        .await?;
    state.events.send(AppEvent::TaskChanged { task });
    Ok(Json(dependency))
}

async fn task_notes(
    State(state): State<AppState>,
    Path(id): Path<TaskId>,
) -> Result<Json<Vec<TaskNote>>, ApiError> {
    state.db.get_task(id).await?;
    Ok(Json(state.db.list_task_notes(id).await?))
}

#[derive(Debug, Deserialize)]
struct TaskNoteRequest {
    content: String,
}

async fn add_task_note(
    State(state): State<AppState>,
    Path(id): Path<TaskId>,
    Json(input): Json<TaskNoteRequest>,
) -> Result<Json<TaskNote>, ApiError> {
    let note = state.main_agent.add_task_note(id, &input.content).await?;
    let task = state.db.get_task(id).await?;
    state.events.send(AppEvent::TaskChanged { task });
    Ok(Json(note))
}

async fn task_resource_locks(
    State(state): State<AppState>,
    Path(id): Path<TaskId>,
) -> Result<Json<Vec<TaskResourceLock>>, ApiError> {
    state.db.get_task(id).await?;
    Ok(Json(state.db.list_task_resource_locks(id).await?))
}

#[derive(Debug, Deserialize)]
struct TaskResourceLockRequest {
    resource_key: String,
}

async fn add_task_resource_lock(
    State(state): State<AppState>,
    Path(id): Path<TaskId>,
    Json(input): Json<TaskResourceLockRequest>,
) -> Result<Json<TaskResourceLock>, ApiError> {
    let resource_lock = state
        .main_agent
        .add_task_resource_lock(id, &input.resource_key)
        .await?;
    let task = state.db.get_task(id).await?;
    state.events.send(AppEvent::TaskChanged { task });
    Ok(Json(resource_lock))
}

async fn remove_task_resource_lock(
    State(state): State<AppState>,
    Path(id): Path<TaskId>,
    Json(input): Json<TaskResourceLockRequest>,
) -> Result<Json<TaskResourceLock>, ApiError> {
    let resource_lock = state
        .main_agent
        .remove_task_resource_lock(id, &input.resource_key)
        .await?;
    let task = state.db.get_task(id).await?;
    state.events.send(AppEvent::TaskChanged { task });
    Ok(Json(resource_lock))
}

async fn pause_task(
    State(state): State<AppState>,
    Path(id): Path<TaskId>,
) -> Result<Json<Task>, ApiError> {
    let task = state.main_agent.pause_task(id).await?;
    state
        .events
        .send(AppEvent::TaskChanged { task: task.clone() });
    Ok(Json(task))
}

async fn resume_task(
    State(state): State<AppState>,
    Path(id): Path<TaskId>,
) -> Result<Json<Task>, ApiError> {
    let task = state.main_agent.resume_task(id).await?;
    state
        .events
        .send(AppEvent::TaskChanged { task: task.clone() });
    Ok(Json(task))
}

async fn cancel_task(
    State(state): State<AppState>,
    Path(id): Path<TaskId>,
) -> Result<Json<Task>, ApiError> {
    let task = state.main_agent.cancel_task(id).await?;
    state
        .events
        .send(AppEvent::TaskChanged { task: task.clone() });
    Ok(Json(task))
}

async fn task_pool_summary(
    State(state): State<AppState>,
) -> Result<Json<TaskPoolSummary>, ApiError> {
    Ok(Json(state.main_agent.summarize_task_pool().await?))
}

async fn main_agent_messages(
    State(state): State<AppState>,
) -> Result<Json<Vec<ConversationMessage>>, ApiError> {
    Ok(Json(
        state.main_agent.main_conversation_messages(100).await?,
    ))
}

async fn main_agent_actions(
    State(state): State<AppState>,
) -> Result<Json<Vec<TaskAction>>, ApiError> {
    Ok(Json(state.db.list_global_actions().await?))
}

async fn send_main_agent_message(
    State(state): State<AppState>,
    Json(input): Json<MainAgentMessageInput>,
) -> Result<Json<persistent_agent_agent::MainAgentMessageResponse>, ApiError> {
    let previous_global_action_ids = state
        .db
        .list_global_actions()
        .await?
        .into_iter()
        .map(|action| action.id)
        .collect::<HashSet<_>>();
    let response = state.main_agent.handle_user_message(input).await?;
    for task in &response.changed_tasks {
        state
            .events
            .send(AppEvent::TaskChanged { task: task.clone() });
    }
    state.events.send(AppEvent::MainAgentReply {
        message: response.assistant_message.clone(),
    });
    for action in state.db.list_global_actions().await? {
        if !previous_global_action_ids.contains(&action.id) {
            state.events.send(AppEvent::MainAgentAction { action });
        }
    }
    if response.scheduler_tick_requested {
        let tick = state.scheduler.tick().await?;
        state
            .events
            .send(AppEvent::SchedulerTick { tick: tick.clone() });
    }
    Ok(Json(response))
}

async fn run_scheduler_tick(
    State(state): State<AppState>,
) -> Result<Json<SchedulerTick>, ApiError> {
    let tick = state.scheduler.tick().await?;
    state
        .events
        .send(AppEvent::SchedulerTick { tick: tick.clone() });
    Ok(Json(tick))
}

#[derive(Debug, Serialize)]
struct SchedulerStateResponse {
    running_tasks: Vec<Task>,
    next_queued_task: Option<Task>,
    queued_count: usize,
    waiting_for_user_tasks: Vec<Task>,
    waiting_for_user_count: usize,
    waiting_for_schedule_count: usize,
    policy: SchedulerPolicy,
}

async fn scheduler_state(
    State(state): State<AppState>,
) -> Result<Json<SchedulerStateResponse>, ApiError> {
    let tasks = state.db.list_tasks().await?;
    let mut queued_tasks = tasks
        .iter()
        .filter(|task| task.status == persistent_agent_domain::TaskStatus::Queued)
        .cloned()
        .collect::<Vec<_>>();
    queued_tasks.sort_by(|left, right| {
        right
            .priority
            .cmp(&left.priority)
            .then_with(|| left.queue_position.cmp(&right.queue_position))
            .then_with(|| left.created_at.cmp(&right.created_at))
    });

    let mut waiting_for_user_tasks = tasks
        .iter()
        .filter(|task| task.status == persistent_agent_domain::TaskStatus::WaitingForUser)
        .cloned()
        .collect::<Vec<_>>();
    waiting_for_user_tasks.sort_by(|left, right| {
        left.updated_at
            .cmp(&right.updated_at)
            .then_with(|| left.queue_position.cmp(&right.queue_position))
    });

    Ok(Json(SchedulerStateResponse {
        running_tasks: tasks
            .iter()
            .filter(|task| task.status == persistent_agent_domain::TaskStatus::Running)
            .cloned()
            .collect(),
        next_queued_task: queued_tasks.first().cloned(),
        queued_count: queued_tasks.len(),
        waiting_for_user_count: waiting_for_user_tasks.len(),
        waiting_for_user_tasks,
        waiting_for_schedule_count: tasks
            .iter()
            .filter(|task| task.status == persistent_agent_domain::TaskStatus::WaitingForSchedule)
            .count(),
        policy: state.scheduler.policy(),
    }))
}

async fn list_memories(State(state): State<AppState>) -> Result<Json<Vec<Memory>>, ApiError> {
    Ok(Json(state.db.list_memories().await?))
}

async fn update_memory(
    State(state): State<AppState>,
    Path(id): Path<MemoryId>,
    Json(input): Json<UpdateMemory>,
) -> Result<Json<Memory>, ApiError> {
    Ok(Json(state.db.update_memory(id, input, "main_agent").await?))
}

async fn delete_memory(
    State(state): State<AppState>,
    Path(id): Path<MemoryId>,
) -> Result<Json<Memory>, ApiError> {
    Ok(Json(state.db.delete_memory(id, "main_agent").await?))
}

async fn approve_memory(
    State(state): State<AppState>,
    Path(id): Path<MemoryId>,
) -> Result<Json<Memory>, ApiError> {
    Ok(Json(
        state
            .db
            .set_memory_status(id, MemoryStatus::Approved, "main_agent")
            .await?,
    ))
}

async fn reject_memory(
    State(state): State<AppState>,
    Path(id): Path<MemoryId>,
) -> Result<Json<Memory>, ApiError> {
    Ok(Json(
        state
            .db
            .set_memory_status(id, MemoryStatus::Rejected, "main_agent")
            .await?,
    ))
}

async fn list_skills(State(state): State<AppState>) -> Result<Json<Vec<Skill>>, ApiError> {
    Ok(Json(state.db.list_skills().await?))
}

async fn create_skill(
    State(state): State<AppState>,
    Json(input): Json<CreateSkill>,
) -> Result<Json<Skill>, ApiError> {
    Ok(Json(state.db.create_skill(input, "main_agent").await?))
}

async fn update_skill(
    State(state): State<AppState>,
    Path(id): Path<SkillId>,
    Json(input): Json<UpdateSkill>,
) -> Result<Json<Skill>, ApiError> {
    Ok(Json(state.db.update_skill(id, input, "main_agent").await?))
}

async fn delete_skill(
    State(state): State<AppState>,
    Path(id): Path<SkillId>,
) -> Result<Json<Skill>, ApiError> {
    Ok(Json(state.db.delete_skill(id, "main_agent").await?))
}

async fn events(
    State(state): State<AppState>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let stream = BroadcastStream::new(state.events.subscribe()).filter_map(|event| match event {
        Ok(event) => {
            let data = serde_json::to_string(&event).unwrap_or_else(|_| "{}".to_owned());
            Some(Ok(Event::default().event("app").data(data)))
        }
        Err(_) => None,
    });

    Sse::new(stream).keep_alive(KeepAlive::new().interval(Duration::from_secs(15)))
}

pub async fn spawn_heartbeat(events: EventBus) {
    let mut interval = interval(Duration::from_secs(15));
    loop {
        interval.tick().await;
        events.send(AppEvent::Heartbeat);
    }
}

pub async fn spawn_scheduler_loop(state: AppState, interval_duration: Duration) {
    if interval_duration.is_zero() {
        tracing::info!("scheduler loop disabled");
        return;
    }

    tracing::info!(
        interval_seconds = interval_duration.as_secs(),
        "scheduler loop enabled"
    );

    sleep(interval_duration).await;
    let mut interval = interval(interval_duration);
    loop {
        interval.tick().await;
        match state.scheduler.tick().await {
            Ok(tick) => state.events.send(AppEvent::SchedulerTick { tick }),
            Err(error) => tracing::error!(?error, "scheduler loop tick failed"),
        }
    }
}

#[derive(Debug)]
pub struct ApiError {
    status: StatusCode,
    error: anyhow::Error,
}

impl ApiError {
    fn bad_request(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            error: anyhow::anyhow!(message.into()),
        }
    }
}

impl From<anyhow::Error> for ApiError {
    fn from(error: anyhow::Error) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            error,
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> axum::response::Response {
        tracing::error!(error = ?self.error, "api error");
        (
            self.status,
            Json(serde_json::json!({ "error": self.error.to_string() })),
        )
            .into_response()
    }
}
