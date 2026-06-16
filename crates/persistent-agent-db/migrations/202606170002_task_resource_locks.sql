CREATE TABLE task_resource_locks (
    task_id TEXT NOT NULL,
    resource_key TEXT NOT NULL,
    lock_mode TEXT NOT NULL DEFAULT 'exclusive',
    created_at TEXT NOT NULL,
    PRIMARY KEY (task_id, resource_key),
    FOREIGN KEY(task_id) REFERENCES tasks(id) ON DELETE CASCADE
);

CREATE INDEX idx_task_resource_locks_task_id ON task_resource_locks(task_id);
CREATE INDEX idx_task_resource_locks_resource_key ON task_resource_locks(resource_key);
