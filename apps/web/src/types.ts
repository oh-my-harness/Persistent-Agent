export type TaskType = "one_off" | "recurring";

export type TaskStatus =
  | "draft"
  | "queued"
  | "running"
  | "waiting_for_user"
  | "waiting_for_schedule"
  | "completed"
  | "failed"
  | "cancelled"
  | "paused";

export interface Task {
  id: string;
  title: string;
  description: string;
  task_type: TaskType;
  status: TaskStatus;
  priority: number;
  queue_position: number;
  created_by: string;
  conversation_id?: string | null;
  requested_skills: string[];
  matched_skills: string[];
  schedule?: unknown;
  attempt_count: number;
  last_run_at?: string | null;
  next_run_at?: string | null;
  blocked_reason?: string | null;
  result_summary?: string | null;
  created_at: string;
  updated_at: string;
}

export interface TaskPoolSummary {
  total: number;
  draft: number;
  queued: number;
  running: number;
  waiting_for_user: number;
  waiting_for_schedule: number;
  completed: number;
  failed: number;
  cancelled: number;
  paused: number;
}

export interface SchedulerTick {
  recovered_tasks: Task[];
  requeued_tasks: Task[];
  claimed_task?: Task | null;
  outcome:
    | { type: "idle" }
    | { type: "completed"; summary: string; follow_up_tasks: Task[] }
    | { type: "blocked"; reason: string }
    | { type: "failed"; error: string }
    | { type: "retry_scheduled"; error: string; next_attempt: number; max_attempts: number }
    | { type: "superseded"; status: TaskStatus; reason: string };
}

export interface SchedulerState {
  running_tasks: Task[];
  next_queued_task?: Task | null;
  queued_count: number;
  waiting_for_user_count: number;
  waiting_for_schedule_count: number;
}

export type AppEvent =
  | { type: "task_changed"; task: Task }
  | { type: "main_agent_reply"; message: ConversationMessage }
  | { type: "scheduler_tick"; tick: SchedulerTick }
  | { type: "heartbeat" };

export interface ConversationMessage {
  id: string;
  conversation_id: string;
  task_id?: string | null;
  role: "user" | "assistant" | string;
  content: string;
  created_at: string;
}

export interface MainAgentMessageResponse {
  conversation_id: string;
  user_message: ConversationMessage;
  assistant_message: ConversationMessage;
  changed_tasks: Task[];
}

export interface TaskMessageResponse {
  user_message: ConversationMessage;
  assistant_message?: ConversationMessage | null;
  task: Task;
}

export interface TaskAttempt {
  id: string;
  task_id: string;
  status: TaskStatus;
  summary?: string | null;
  started_at: string;
  finished_at?: string | null;
}

export interface TaskAttemptEvent {
  id: string;
  attempt_id: string;
  task_id: string;
  event_type: string;
  message: string;
  details: unknown;
  created_at: string;
}

export interface TaskArtifact {
  id: string;
  task_id: string;
  attempt_id?: string | null;
  name: string;
  artifact_type: string;
  uri: string;
  summary?: string | null;
  created_at: string;
}

export interface TaskDependency {
  task_id: string;
  depends_on_task_id: string;
  created_at: string;
}

export interface TaskResourceLock {
  task_id: string;
  resource_key: string;
  lock_mode: string;
  created_at: string;
}

export interface TaskNote {
  id: string;
  task_id: string;
  actor: string;
  content: string;
  created_at: string;
}

export interface TaskAction {
  id: string;
  task_id?: string | null;
  actor: string;
  action_type: string;
  details: unknown;
  created_at: string;
}

export interface TaskHistory {
  attempts: TaskAttempt[];
  attempt_events: TaskAttemptEvent[];
  artifacts: TaskArtifact[];
  dependencies: TaskDependency[];
  resource_locks: TaskResourceLock[];
  notes: TaskNote[];
  actions: TaskAction[];
}

export type MemoryStatus = "pending" | "approved" | "rejected";

export interface Memory {
  id: string;
  scope: string;
  content: string;
  source_task_id?: string | null;
  status: MemoryStatus;
  confidence: number;
  created_at: string;
}

export interface Skill {
  id: string;
  name: string;
  description: string;
  trigger_rules: string[];
  tool_subset: string[];
  resource_path?: string | null;
  created_at: string;
  updated_at: string;
}
