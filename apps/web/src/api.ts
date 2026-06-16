import type { SchedulerTick, Task, TaskPoolSummary, TaskType } from "./types";

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
