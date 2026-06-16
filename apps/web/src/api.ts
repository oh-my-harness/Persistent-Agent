import type {
  ConversationMessage,
  MainAgentMessageResponse,
  Memory,
  SchedulerTick,
  Skill,
  Task,
  TaskHistory,
  TaskMessageResponse,
  TaskPoolSummary,
  TaskType,
} from "./types";

const apiBase = "";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${apiBase}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...init?.headers,
    },
  });

  if (!response.ok) {
    const body = await response.text();
    throw new Error(body || `Request failed: ${response.status}`);
  }

  return response.json() as Promise<T>;
}

export function listTasks(): Promise<Task[]> {
  return request<Task[]>("/api/tasks");
}

export function getTaskPoolSummary(): Promise<TaskPoolSummary> {
  return request<TaskPoolSummary>("/api/main-agent/task-pool-summary");
}

export interface CreateTaskInput {
  title: string;
  description: string;
  task_type: TaskType;
  priority: number;
  requested_skills: string[];
  schedule?: unknown;
  created_by: string;
}

export function createTask(input: CreateTaskInput): Promise<Task> {
  return request<Task>("/api/tasks", {
    method: "POST",
    body: JSON.stringify(input),
  });
}

export function pauseTask(id: string): Promise<Task> {
  return request<Task>(`/api/tasks/${id}/pause`, { method: "POST" });
}

export function resumeTask(id: string): Promise<Task> {
  return request<Task>(`/api/tasks/${id}/resume`, { method: "POST" });
}

export function cancelTask(id: string): Promise<Task> {
  return request<Task>(`/api/tasks/${id}/cancel`, { method: "POST" });
}

export function reprioritizeTask(id: string, priority: number): Promise<Task> {
  return request<Task>(`/api/tasks/${id}/reprioritize`, {
    method: "POST",
    body: JSON.stringify({ priority }),
  });
}

export function reorderTask(id: string, queuePosition: number): Promise<Task> {
  return request<Task>(`/api/tasks/${id}/reorder`, {
    method: "POST",
    body: JSON.stringify({ queue_position: queuePosition }),
  });
}

export function runSchedulerTick(): Promise<SchedulerTick> {
  return request<SchedulerTick>("/api/scheduler/tick", { method: "POST" });
}

export function listMainAgentMessages(): Promise<ConversationMessage[]> {
  return request<ConversationMessage[]>("/api/main-agent/messages");
}

export function sendMainAgentMessage(content: string): Promise<MainAgentMessageResponse> {
  return request<MainAgentMessageResponse>("/api/main-agent/messages", {
    method: "POST",
    body: JSON.stringify({ content }),
  });
}

export function listTaskMessages(id: string): Promise<ConversationMessage[]> {
  return request<ConversationMessage[]>(`/api/tasks/${id}/messages`);
}

export function sendTaskMessage(id: string, content: string): Promise<TaskMessageResponse> {
  return request<TaskMessageResponse>(`/api/tasks/${id}/messages`, {
    method: "POST",
    body: JSON.stringify({ content }),
  });
}

export function getTaskHistory(id: string): Promise<TaskHistory> {
  return request<TaskHistory>(`/api/tasks/${id}/history`);
}

export function listMemories(): Promise<Memory[]> {
  return request<Memory[]>("/api/memories");
}

export function approveMemory(id: string): Promise<Memory> {
  return request<Memory>(`/api/memories/${id}/approve`, { method: "POST" });
}

export function rejectMemory(id: string): Promise<Memory> {
  return request<Memory>(`/api/memories/${id}/reject`, { method: "POST" });
}

export interface UpdateMemoryInput {
  scope?: string;
  content?: string;
  confidence?: number;
}

export function updateMemory(id: string, input: UpdateMemoryInput): Promise<Memory> {
  return request<Memory>(`/api/memories/${id}`, {
    method: "PATCH",
    body: JSON.stringify(input),
  });
}

export function deleteMemory(id: string): Promise<Memory> {
  return request<Memory>(`/api/memories/${id}`, { method: "DELETE" });
}

export interface CreateSkillInput {
  name: string;
  description: string;
  trigger_rules: string[];
  tool_subset: string[];
  resource_path?: string | null;
}

export function listSkills(): Promise<Skill[]> {
  return request<Skill[]>("/api/skills");
}

export function createSkill(input: CreateSkillInput): Promise<Skill> {
  return request<Skill>("/api/skills", {
    method: "POST",
    body: JSON.stringify(input),
  });
}
