use std::{net::SocketAddr, time::Duration};

use persistent_agent_api::{AppState, router, spawn_heartbeat, spawn_scheduler_loop};
use persistent_agent_db::Db;
use persistent_agent_scheduler::{LlmWorker, LlmWorkerConfig, StubWorker, WorkerBackend};
use tracing_subscriber::{EnvFilter, fmt};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    fmt()
        .with_env_filter(EnvFilter::from_default_env().add_directive("info".parse()?))
        .init();

    let database_url = std::env::var("DATABASE_URL")
        .unwrap_or_else(|_| "sqlite://persistent-agent.db?mode=rwc".to_owned());
    let bind_addr = std::env::var("BIND_ADDR").unwrap_or_else(|_| "127.0.0.1:8080".to_owned());

    let db = Db::connect(&database_url).await?;
    let worker = build_worker();
    let state = AppState::new(db, worker);
    let scheduler_interval = scheduler_interval();

    tokio::spawn(spawn_heartbeat(state.events.clone()));
    tokio::spawn(spawn_scheduler_loop(state.clone(), scheduler_interval));

    let app = router(state);
    let addr: SocketAddr = bind_addr.parse()?;
    let listener = tokio::net::TcpListener::bind(addr).await?;
    tracing::info!(%addr, "persistent agent server listening");

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    Ok(())
}

fn build_worker() -> WorkerBackend {
    match std::env::var("DEEPSEEK_API_KEY") {
        Ok(api_key) if !api_key.trim().is_empty() => {
            let model =
                std::env::var("DEEPSEEK_MODEL").unwrap_or_else(|_| "deepseek-chat".to_owned());
            tracing::info!(%model, "using DeepSeek LLM worker");
            WorkerBackend::Llm(LlmWorker::new(LlmWorkerConfig::deepseek(api_key, model)))
        }
        _ => {
            tracing::info!("DEEPSEEK_API_KEY not set; using stub worker");
            WorkerBackend::Stub(StubWorker)
        }
    }
}

fn scheduler_interval() -> Duration {
    std::env::var("SCHEDULER_INTERVAL_SECONDS")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .map(Duration::from_secs)
        .unwrap_or_else(|| Duration::from_secs(30))
}

async fn shutdown_signal() {
    let _ = tokio::signal::ctrl_c().await;
}
