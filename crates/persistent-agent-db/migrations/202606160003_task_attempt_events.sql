CREATE TABLE IF NOT EXISTS task_attempt_events (
  id TEXT PRIMARY KEY NOT NULL,
  attempt_id TEXT NOT NULL REFERENCES task_attempts(id) ON DELETE CASCADE,
  task_id TEXT NOT NULL REFERENCES tasks(id) ON DELETE CASCADE,
  event_type TEXT NOT NULL,
  message TEXT NOT NULL,
  details TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_task_attempt_events_task_id
  ON task_attempt_events(task_id, created_at ASC);

CREATE INDEX IF NOT EXISTS idx_task_attempt_events_attempt_id
  ON task_attempt_events(attempt_id, created_at ASC);
