use std::{net::SocketAddr, sync::Arc, time::Duration};

use persistent_agent_agent::{MainAgent, MainAgentLlmConfig, OhMyHarnessMainAgentAdvisor};
use persistent_agent_api::{AppState, router, spawn_heartbeat, spawn_scheduler_loop};
use persistent_agent_db::Db;
use persistent_agent_scheduler::{
    LlmWorkerConfig, OhMyHarnessWorker, SchedulerPolicy, StubWorker, WorkerBackend,
};
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
    let llm_config = deepseek_config_from_env();
    let worker = build_worker(llm_config.clone());
    let main_agent = build_main_agent(db.clone(), llm_config);
    let scheduler_policy = scheduler_policy();
    let state = AppState::new_with_scheduler_policy_and_main_agent(
        db,
        worker,
        scheduler_policy,
        main_agent,
    );
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

fn build_worker(llm_config: Option<MainAgentLlmConfig>) -> WorkerBackend {
    match llm_config {
        Some(config) => {
            tracing::info!(model = %config.model, "using DeepSeek worker through oh-my-harness AgentHarness");
            WorkerBackend::Harness(OhMyHarnessWorker::new(LlmWorkerConfig::deepseek(
                config.api_key,
                config.model,
            )))
        }
        _ => {
            tracing::info!("DEEPSEEK_API_KEY not set; using stub worker");
            WorkerBackend::Stub(StubWorker)
        }
    }
}

fn build_main_agent(db: Db, llm_config: Option<MainAgentLlmConfig>) -> MainAgent {
    match llm_config {
        Some(config) => {
            tracing::info!(model = %config.model, "using DeepSeek main agent planner/advisor through oh-my-harness AgentHarness");
            let harness_main_agent = Arc::new(OhMyHarnessMainAgentAdvisor::new(config));
            MainAgent::new(db)
                .with_advisor(harness_main_agent.clone())
                .with_planner(harness_main_agent)
        }
        None => {
            tracing::info!("DEEPSEEK_API_KEY not set; using deterministic main agent only");
            MainAgent::new(db)
        }
    }
}

fn deepseek_config_from_env() -> Option<MainAgentLlmConfig> {
    let api_key = std::env::var("DEEPSEEK_API_KEY").ok()?;
    if api_key.trim().is_empty() {
        return None;
    }
    let model = std::env::var("DEEPSEEK_MODEL").unwrap_or_else(|_| "deepseek-chat".to_owned());
    Some(MainAgentLlmConfig::deepseek(api_key, model))
}

fn scheduler_interval() -> Duration {
    std::env::var("SCHEDULER_INTERVAL_SECONDS")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .map(Duration::from_secs)
        .unwrap_or_else(|| Duration::from_secs(30))
}

fn scheduler_policy() -> SchedulerPolicy {
    scheduler_policy_from_env(|name| std::env::var(name).ok())
}

fn scheduler_policy_from_env(get_env: impl Fn(&str) -> Option<String>) -> SchedulerPolicy {
    let worker_capacity = env_usize(&get_env, "SCHEDULER_WORKER_CAPACITY", 1);
    let lease_seconds = env_i64(&get_env, "SCHEDULER_LEASE_SECONDS", 300);
    let max_attempts = env_i64(&get_env, "SCHEDULER_MAX_ATTEMPTS", 1);
    let memory_auto_approve_confidence =
        env_f64(&get_env, "MEMORY_AUTO_APPROVE_CONFIDENCE").filter(|value| value.is_finite());

    SchedulerPolicy::new(worker_capacity, lease_seconds)
        .with_max_attempts(max_attempts)
        .with_memory_auto_approve_confidence(memory_auto_approve_confidence)
}

fn env_usize(get_env: impl Fn(&str) -> Option<String>, name: &str, default: usize) -> usize {
    get_env(name)
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(default)
}

fn env_i64(get_env: impl Fn(&str) -> Option<String>, name: &str, default: i64) -> i64 {
    get_env(name)
        .and_then(|value| value.parse::<i64>().ok())
        .unwrap_or(default)
}

fn env_f64(get_env: impl Fn(&str) -> Option<String>, name: &str) -> Option<f64> {
    get_env(name).and_then(|value| value.parse::<f64>().ok())
}

async fn shutdown_signal() {
    let _ = tokio::signal::ctrl_c().await;
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn scheduler_policy_uses_defaults_when_env_is_missing() {
        let policy = scheduler_policy_from_env(|_| None);

        assert_eq!(policy, SchedulerPolicy::serial());
    }

    #[test]
    fn scheduler_policy_reads_capacity_lease_and_retry_limits() {
        let values = HashMap::from([
            ("SCHEDULER_WORKER_CAPACITY", "4"),
            ("SCHEDULER_LEASE_SECONDS", "45"),
            ("SCHEDULER_MAX_ATTEMPTS", "3"),
            ("MEMORY_AUTO_APPROVE_CONFIDENCE", "0.75"),
        ]);
        let policy =
            scheduler_policy_from_env(|name| values.get(name).map(|value| value.to_string()));

        assert_eq!(
            policy,
            SchedulerPolicy {
                worker_capacity: 4,
                lease_seconds: 45,
                max_attempts: 3,
                memory_auto_approve_confidence: Some(0.75),
            }
        );
    }

    #[test]
    fn scheduler_policy_ignores_invalid_values() {
        let values = HashMap::from([
            ("SCHEDULER_WORKER_CAPACITY", "many"),
            ("SCHEDULER_LEASE_SECONDS", "soon"),
            ("SCHEDULER_MAX_ATTEMPTS", "often"),
            ("MEMORY_AUTO_APPROVE_CONFIDENCE", "confident"),
        ]);
        let policy =
            scheduler_policy_from_env(|name| values.get(name).map(|value| value.to_string()));

        assert_eq!(policy, SchedulerPolicy::serial());
    }
}
