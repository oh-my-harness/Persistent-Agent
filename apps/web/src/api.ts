import type {
  ConversationMessage,
  MainAgentMessageResponse,
  Memory,
  SchedulerTick,
  Skill,
  Task,
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

export function listMemories(): Promise<Memory[]> {
  return request<Memory[]>("/api/memories");
}

export function approveMemory(id: string): Promise<Memory> {
  return request<Memory>(`/api/memories/${id}/approve`, { method: "POST" });
}

export function rejectMemory(id: string): Promise<Memory> {
  return request<Memory>(`/api/memories/${id}/reject`, { method: "POST" });
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
