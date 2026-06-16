use std::{convert::Infallible, time::Duration};

use axum::{
    Json, Router,
    extract::{Path, State},
    http::StatusCode,
    response::{
        IntoResponse, Sse,
        sse::{Event, KeepAlive},
    },
    routing::{get, post},
};
use persistent_agent_agent::{MainAgent, MainAgentMessageInput, TaskPoolSummary};
use persistent_agent_db::Db;
use persistent_agent_domain::{ConversationMessage, CreateTask, Task, TaskId, UpdateTask};
use persistent_agent_scheduler::{Scheduler, SchedulerTick, StubWorker};
use serde::{Deserialize, Serialize};
use tokio::sync::broadcast;
use tokio::time::interval;
use tokio_stream::{Stream, StreamExt, wrappers::BroadcastStream};
use tower_http::{cors::CorsLayer, trace::TraceLayer};
use uuid::Uuid;

#[derive(Clone)]
pub struct AppState {
    pub db: Db,
    pub main_agent: MainAgent,
    pub scheduler: Scheduler<StubWorker>,
    pub events: EventBus,
}

impl AppState {
    pub fn new(db: Db) -> Self {
        let main_agent = MainAgent::new(db.clone());
        let scheduler = Scheduler::new(db.clone(), StubWorker);
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
        .route("/api/tasks/{id}/pause", post(pause_task))
        .route("/api/tasks/{id}/resume", post(resume_task))
        .route("/api/tasks/{id}/cancel", post(cancel_task))
        .route("/api/main-agent/task-pool-summary", get(task_pool_summary))
        .route(
            "/api/main-agent/messages",
            get(main_agent_messages).post(send_main_agent_message),
        )
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

async fn send_main_agent_message(
    State(state): State<AppState>,
    Json(input): Json<MainAgentMessageInput>,
) -> Result<Json<persistent_agent_agent::MainAgentMessageResponse>, ApiError> {
    let response = state.main_agent.handle_user_message(input).await?;
    for task in &response.changed_tasks {
        state
            .events
            .send(AppEvent::TaskChanged { task: task.clone() });
    }
    state.events.send(AppEvent::MainAgentReply {
        message: response.assistant_message.clone(),
    });
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

#[derive(Debug)]
pub struct ApiError(anyhow::Error);

impl From<anyhow::Error> for ApiError {
    fn from(error: anyhow::Error) -> Self {
        Self(error)
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> axum::response::Response {
        tracing::error!(error = ?self.0, "api error");
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": self.0.to_string() })),
        )
            .into_response()
    }
}
