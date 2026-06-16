CREATE TABLE IF NOT EXISTS task_artifacts (
  id TEXT PRIMARY KEY NOT NULL,
  task_id TEXT NOT NULL REFERENCES tasks(id) ON DELETE CASCADE,
  attempt_id TEXT REFERENCES task_attempts(id) ON DELETE SET NULL,
  name TEXT NOT NULL,
  artifact_type TEXT NOT NULL,
  uri TEXT NOT NULL,
  summary TEXT,
  created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_task_artifacts_task_id
  ON task_artifacts(task_id, created_at ASC);

CREATE INDEX IF NOT EXISTS idx_task_artifacts_attempt_id
  ON task_artifacts(attempt_id, created_at ASC);
