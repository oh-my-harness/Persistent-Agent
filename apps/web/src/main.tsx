import React, { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import { QueryClient, QueryClientProvider, useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Bot, CirclePause, ListTodo, Play, Plus, RotateCw, SquareX, Zap } from "lucide-react";
import {
  cancelTask,
  createTask,
  getTaskPoolSummary,
  listTasks,
  pauseTask,
  resumeTask,
  runSchedulerTick,
} from "./api";
import type { Task, TaskType } from "./types";
import "./styles.css";

const queryClient = new QueryClient();

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <Shell />
    </QueryClientProvider>
  );
}

function Shell() {
  const [lastEvent, setLastEvent] = useState("Waiting for server events");
  const queryClient = useQueryClient();

  const tasks = useQuery({ queryKey: ["tasks"], queryFn: listTasks });
  const summary = useQuery({ queryKey: ["summary"], queryFn: getTaskPoolSummary });

  useEffect(() => {
    const source = new EventSource("/api/events");
    source.addEventListener("app", (event) => {
      setLastEvent(event.data);
      void queryClient.invalidateQueries({ queryKey: ["tasks"] });
      void queryClient.invalidateQueries({ queryKey: ["summary"] });
    });
    source.onerror = () => setLastEvent("Event stream disconnected");
    return () => source.close();
  }, [queryClient]);

  const visibleTasks = tasks.data ?? [];

  return (
    <main className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <Bot size={24} />
          <div>
            <strong>Persistent Agent</strong>
            <span>Conversational task operations</span>
          </div>
        </div>
        <nav>
          <a className="active"><ListTodo size={18} /> Task pool</a>
          <a><Zap size={18} /> Execution</a>
          <a><Bot size={18} /> Main agent</a>
        </nav>
      </aside>

      <section className="workspace">
        <header className="topbar">
          <div>
            <h1>Task Pool</h1>
            <p>Main agent can manage this queue through auditable task tools.</p>
          </div>
          <SchedulerButton />
        </header>

        <section className="summary-strip">
          <Metric label="Total" value={summary.data?.total ?? 0} />
          <Metric label="Queued" value={summary.data?.queued ?? 0} />
          <Metric label="Running" value={summary.data?.running ?? 0} />
          <Metric label="Needs user" value={summary.data?.waiting_for_user ?? 0} />
          <Metric label="Completed" value={summary.data?.completed ?? 0} />
        </section>

        <section className="content-grid">
          <TaskComposer />
          <section className="panel task-list-panel">
            <div className="panel-heading">
              <h2>Queue</h2>
              <span>{tasks.isLoading ? "Loading" : `${visibleTasks.length} tasks`}</span>
            </div>
            <div className="task-list">
              {visibleTasks.map((task) => (
                <TaskRow key={task.id} task={task} />
              ))}
              {visibleTasks.length === 0 && <p className="empty">No tasks yet.</p>}
            </div>
          </section>
          <section className="panel event-panel">
            <div className="panel-heading">
              <h2>Event Stream</h2>
              <span>SSE</span>
            </div>
            <pre>{lastEvent}</pre>
          </section>
        </section>
      </section>
    </main>
  );
}

function Metric({ label, value }: { label: string; value: number }) {
  return (
    <div className="metric">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function TaskComposer() {
  const queryClient = useQueryClient();
  const [title, setTitle] = useState("");
  const [description, setDescription] = useState("");
  const [taskType, setTaskType] = useState<TaskType>("one_off");
  const [priority, setPriority] = useState(0);

  const mutation = useMutation({
    mutationFn: createTask,
    onSuccess: async () => {
      setTitle("");
      setDescription("");
      setPriority(0);
      await queryClient.invalidateQueries({ queryKey: ["tasks"] });
      await queryClient.invalidateQueries({ queryKey: ["summary"] });
    },
  });

  return (
    <section className="panel composer">
      <div className="panel-heading">
        <h2>Main Agent Task Tool</h2>
        <span>create_task</span>
      </div>
      <form
        onSubmit={(event) => {
          event.preventDefault();
          if (!title.trim()) return;
          mutation.mutate({
            title,
            description,
            task_type: taskType,
            priority,
            requested_skills: [],
            created_by: "user",
          });
        }}
      >
        <label>
          Title
          <input value={title} onChange={(event) => setTitle(event.target.value)} placeholder="Check repository issues" />
        </label>
        <label>
          Description
          <textarea
            value={description}
            onChange={(event) => setDescription(event.target.value)}
            placeholder="Describe the work the agent should keep track of."
          />
        </label>
        <div className="form-row">
          <label>
            Type
            <select value={taskType} onChange={(event) => setTaskType(event.target.value as TaskType)}>
              <option value="one_off">One-off</option>
              <option value="recurring">Recurring</option>
            </select>
          </label>
          <label>
            Priority
            <input type="number" value={priority} onChange={(event) => setPriority(Number(event.target.value))} />
          </label>
        </div>
        <button className="primary" type="submit" disabled={mutation.isPending}>
          <Plus size={16} /> Create task
        </button>
      </form>
    </section>
  );
}

function TaskRow({ task }: { task: Task }) {
  const queryClient = useQueryClient();
  const refresh = async () => {
    await queryClient.invalidateQueries({ queryKey: ["tasks"] });
    await queryClient.invalidateQueries({ queryKey: ["summary"] });
  };
  const pause = useMutation({ mutationFn: pauseTask, onSuccess: refresh });
  const resume = useMutation({ mutationFn: resumeTask, onSuccess: refresh });
  const cancel = useMutation({ mutationFn: cancelTask, onSuccess: refresh });

  const statusClass = useMemo(() => `status ${task.status.replaceAll("_", "-")}`, [task.status]);

  return (
    <article className="task-row">
      <div>
        <div className="task-title-line">
          <h3>{task.title}</h3>
          <span className={statusClass}>{task.status.replaceAll("_", " ")}</span>
        </div>
        <p>{task.description || "No description"}</p>
        <div className="task-meta">
          <span>{task.task_type.replace("_", " ")}</span>
          <span>priority {task.priority}</span>
          <span>attempts {task.attempt_count}</span>
        </div>
      </div>
      <div className="task-actions">
        {task.status === "paused" ? (
          <button title="Resume task" onClick={() => resume.mutate(task.id)}><Play size={16} /></button>
        ) : (
          <button title="Pause task" onClick={() => pause.mutate(task.id)}><CirclePause size={16} /></button>
        )}
        <button title="Cancel task" onClick={() => cancel.mutate(task.id)}><SquareX size={16} /></button>
      </div>
    </article>
  );
}

function SchedulerButton() {
  const queryClient = useQueryClient();
  const mutation = useMutation({
    mutationFn: runSchedulerTick,
    onSuccess: async () => {
      await queryClient.invalidateQueries({ queryKey: ["tasks"] });
      await queryClient.invalidateQueries({ queryKey: ["summary"] });
    },
  });

  return (
    <button className="primary" onClick={() => mutation.mutate()} disabled={mutation.isPending}>
      <RotateCw size={16} /> Run scheduler tick
    </button>
  );
}

createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
