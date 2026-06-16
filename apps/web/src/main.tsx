import React, { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import { QueryClient, QueryClientProvider, useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Bot, Check, CirclePause, History, ListTodo, Pencil, Play, Plus, RotateCw, Send, SquareX, X, Zap } from "lucide-react";
import {
  addTaskDependency,
  approveMemory,
  addTaskResourceLock,
  cancelTask,
  createSkill,
  createTask,
  deleteMemory,
  deleteSkill,
  getTaskHistory,
  getTaskPoolSummary,
  getSchedulerState,
  listMainAgentMessages,
  listMemories,
  listSkills,
  listTaskMessages,
  listTasks,
  pauseTask,
  rejectMemory,
  removeTaskDependency,
  removeTaskResourceLock,
  reorderTask,
  reprioritizeTask,
  resumeTask,
  runSchedulerTick,
  sendMainAgentMessage,
  sendTaskMessage,
  updateMemory,
  updateSkill,
  updateTask,
} from "./api";
import type { AppEvent, ConversationMessage, Memory, SchedulerState, SchedulerTick, Task, TaskType } from "./types";
import type { Skill } from "./types";
import "./styles.css";

const queryClient = new QueryClient();

const TASK_STATUS_FILTERS: Array<{ label: string; value: "all" | Task["status"] }> = [
  { label: "All", value: "all" },
  { label: "Queued", value: "queued" },
  { label: "Running", value: "running" },
  { label: "Needs user", value: "waiting_for_user" },
  { label: "Scheduled", value: "waiting_for_schedule" },
  { label: "Completed", value: "completed" },
  { label: "Failed", value: "failed" },
  { label: "Paused", value: "paused" },
  { label: "Cancelled", value: "cancelled" },
];

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <Shell />
    </QueryClientProvider>
  );
}

function Shell() {
  const [eventLog, setEventLog] = useState<TimelineEvent[]>([]);
  const [taskStatusFilter, setTaskStatusFilter] = useState<"all" | Task["status"]>("all");
  const queryClient = useQueryClient();

  const tasks = useQuery({ queryKey: ["tasks"], queryFn: listTasks });
  const summary = useQuery({ queryKey: ["summary"], queryFn: getTaskPoolSummary });
  const schedulerState = useQuery({
    queryKey: ["scheduler-state"],
    queryFn: getSchedulerState,
    refetchInterval: 10_000,
  });

  useEffect(() => {
    const source = new EventSource("/api/events");
    source.addEventListener("app", (event) => {
      const parsed = parseAppEvent(event.data);
      setEventLog((current) => [toTimelineEvent(parsed, event.data), ...current].slice(0, 20));
      void queryClient.invalidateQueries({ queryKey: ["tasks"] });
      void queryClient.invalidateQueries({ queryKey: ["summary"] });
      void queryClient.invalidateQueries({ queryKey: ["scheduler-state"] });
      void queryClient.invalidateQueries({ queryKey: ["main-agent-messages"] });
      void queryClient.invalidateQueries({ queryKey: ["memories"] });
      void queryClient.invalidateQueries({ queryKey: ["skills"] });
    });
    source.onerror = () =>
      setEventLog((current) => [
        {
          id: `${Date.now()}-disconnected`,
          title: "Event stream disconnected",
          detail: "The UI will reconnect automatically when the server is available.",
          tone: "warning" as const,
          timestamp: new Date().toISOString(),
        },
        ...current,
      ].slice(0, 20));
    return () => source.close();
  }, [queryClient]);

  const allTasks = tasks.data ?? [];
  const visibleTasks = useMemo(
    () =>
      taskStatusFilter === "all"
        ? allTasks
        : allTasks.filter((task) => task.status === taskStatusFilter),
    [allTasks, taskStatusFilter],
  );

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
          <MainAgentChat />
          <SkillManager />
          <MemoryReview />
          <section className="panel task-list-panel">
            <div className="panel-heading">
              <h2>Queue</h2>
              <span>{tasks.isLoading ? "Loading" : `${visibleTasks.length}/${allTasks.length} tasks`}</span>
            </div>
            <div className="status-filter" aria-label="Task status filter">
              {TASK_STATUS_FILTERS.map((filter) => (
                <button
                  key={filter.value}
                  className={taskStatusFilter === filter.value ? "active" : ""}
                  type="button"
                  onClick={() => setTaskStatusFilter(filter.value)}
                >
                  {filter.label}
                </button>
              ))}
            </div>
            <div className="task-list">
              {visibleTasks.map((task) => (
                <TaskRow key={task.id} task={task} />
              ))}
              {visibleTasks.length === 0 && (
                <p className="empty">
                  {taskStatusFilter === "all" ? "No tasks yet." : "No tasks match this status."}
                </p>
              )}
            </div>
          </section>
          <section className="panel event-panel">
            <div className="panel-heading">
              <h2>Execution Monitor</h2>
              <span>SSE</span>
            </div>
            <ExecutionStatePanel state={schedulerState.data} loading={schedulerState.isLoading} />
            <EventTimeline events={eventLog} />
          </section>
        </section>
      </section>
    </main>
  );
}

function SkillManager() {
  const queryClient = useQueryClient();
  const skills = useQuery({ queryKey: ["skills"], queryFn: listSkills });
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [rules, setRules] = useState("");
  const refresh = async () => {
    await queryClient.invalidateQueries({ queryKey: ["skills"] });
    await queryClient.invalidateQueries({ queryKey: ["tasks"] });
  };
  const mutation = useMutation({
    mutationFn: createSkill,
    onSuccess: async () => {
      setName("");
      setDescription("");
      setRules("");
      await refresh();
    },
  });
  const update = useMutation({
    mutationFn: ({
      id,
      description,
      name,
      resourcePath,
      rules,
      tools,
    }: {
      id: string;
      description: string;
      name: string;
      resourcePath: string;
      rules: string;
      tools: string;
    }) =>
      updateSkill(id, {
        name: name.trim(),
        description,
        trigger_rules: splitCsv(rules),
        tool_subset: splitCsv(tools),
        resource_path: resourcePath.trim() || null,
      }),
    onSuccess: refresh,
  });
  const remove = useMutation({ mutationFn: deleteSkill, onSuccess: refresh });

  const visibleSkills = skills.data ?? [];

  return (
    <section className="panel skill-panel">
      <div className="panel-heading">
        <h2>Skills</h2>
        <span>{visibleSkills.length} defined</span>
      </div>
      <form
        className="skill-form"
        onSubmit={(event) => {
          event.preventDefault();
          if (!name.trim()) return;
          mutation.mutate({
            name: name.trim(),
            description,
            trigger_rules: splitCsv(rules),
            tool_subset: [],
            resource_path: null,
          });
        }}
      >
        <input value={name} onChange={(event) => setName(event.target.value)} placeholder="Skill name" />
        <input
          value={rules}
          onChange={(event) => setRules(event.target.value)}
          placeholder="Triggers, comma separated"
        />
        <textarea
          value={description}
          onChange={(event) => setDescription(event.target.value)}
          placeholder="What this skill helps with"
        />
        <button className="primary" type="submit" disabled={mutation.isPending}>
          <Plus size={16} /> Add skill
        </button>
      </form>
      <div className="skill-list">
        {visibleSkills.map((skill) => (
          <SkillRow
            key={skill.id}
            skill={skill}
            onDelete={() => remove.mutate(skill.id)}
            onUpdate={(name, description, rules, tools, resourcePath) =>
              update.mutate({ id: skill.id, name, description, rules, tools, resourcePath })
            }
          />
        ))}
        {visibleSkills.length === 0 && <p className="empty">Add a skill to enable automatic task matching.</p>}
      </div>
    </section>
  );
}

function SkillRow({
  skill,
  onDelete,
  onUpdate,
}: {
  skill: Skill;
  onDelete: () => void;
  onUpdate: (name: string, description: string, rules: string, tools: string, resourcePath: string) => void;
}) {
  const [editing, setEditing] = useState(false);
  const [name, setName] = useState(skill.name);
  const [description, setDescription] = useState(skill.description);
  const [rules, setRules] = useState(skill.trigger_rules.join(", "));
  const [tools, setTools] = useState(skill.tool_subset.join(", "));
  const [resourcePath, setResourcePath] = useState(skill.resource_path ?? "");

  useEffect(() => {
    setName(skill.name);
    setDescription(skill.description);
    setRules(skill.trigger_rules.join(", "));
    setTools(skill.tool_subset.join(", "));
    setResourcePath(skill.resource_path ?? "");
  }, [skill.description, skill.name, skill.resource_path, skill.tool_subset, skill.trigger_rules]);

  return (
    <article className="skill-row">
      <div>
        <strong>{skill.name}</strong>
        <p>{skill.description || "No description"}</p>
        <span>{skill.trigger_rules.join(", ") || "no triggers"}</span>
        {skill.tool_subset.length > 0 && <span>tools {skill.tool_subset.join(", ")}</span>}
        {skill.resource_path && <span>resource {skill.resource_path}</span>}
        {editing && (
          <div className="skill-edit">
            <label>
              Name
              <input value={name} onChange={(event) => setName(event.target.value)} />
            </label>
            <label>
              Triggers
              <input value={rules} onChange={(event) => setRules(event.target.value)} />
            </label>
            <label>
              Tool subset
              <input value={tools} onChange={(event) => setTools(event.target.value)} />
            </label>
            <label>
              Resource path
              <input value={resourcePath} onChange={(event) => setResourcePath(event.target.value)} />
            </label>
            <label>
              Description
              <textarea value={description} onChange={(event) => setDescription(event.target.value)} />
            </label>
          </div>
        )}
      </div>
      <div className="skill-actions">
        {editing ? (
          <>
            <button
              onClick={() => {
                if (name.trim()) {
                  onUpdate(name, description, rules, tools, resourcePath);
                  setEditing(false);
                }
              }}
            >
              Save
            </button>
            <button
              onClick={() => {
                setName(skill.name);
                setDescription(skill.description);
                setRules(skill.trigger_rules.join(", "));
                setTools(skill.tool_subset.join(", "));
                setResourcePath(skill.resource_path ?? "");
                setEditing(false);
              }}
            >
              Close
            </button>
          </>
        ) : (
          <button onClick={() => setEditing(true)}>Edit</button>
        )}
        <button onClick={onDelete}>Delete</button>
      </div>
    </article>
  );
}

function splitCsv(value: string): string[] {
  return value
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);
}

function MemoryReview() {
  const queryClient = useQueryClient();
  const memories = useQuery({ queryKey: ["memories"], queryFn: listMemories });
  const refresh = async () => {
    await queryClient.invalidateQueries({ queryKey: ["memories"] });
  };
  const approve = useMutation({ mutationFn: approveMemory, onSuccess: refresh });
  const reject = useMutation({ mutationFn: rejectMemory, onSuccess: refresh });
  const update = useMutation({
    mutationFn: ({ id, scope, content, confidence }: { id: string; scope: string; content: string; confidence: number }) =>
      updateMemory(id, { scope, content, confidence }),
    onSuccess: refresh,
  });
  const remove = useMutation({ mutationFn: deleteMemory, onSuccess: refresh });
  const visibleMemories = memories.data ?? [];

  return (
    <section className="panel memory-panel">
      <div className="panel-heading">
        <h2>Memory Review</h2>
        <span>{visibleMemories.filter((memory) => memory.status === "pending").length} pending</span>
      </div>
      <div className="memory-list">
        {visibleMemories.map((memory) => (
          <MemoryRow
            key={memory.id}
            memory={memory}
            onApprove={() => approve.mutate(memory.id)}
            onReject={() => reject.mutate(memory.id)}
            onDelete={() => remove.mutate(memory.id)}
            onUpdate={(scope, content, confidence) =>
              update.mutate({ id: memory.id, scope, content, confidence })
            }
          />
        ))}
        {visibleMemories.length === 0 && <p className="empty">Worker memory candidates will appear here.</p>}
      </div>
    </section>
  );
}

function MemoryRow({
  memory,
  onApprove,
  onDelete,
  onReject,
  onUpdate,
}: {
  memory: Memory;
  onApprove: () => void;
  onDelete: () => void;
  onReject: () => void;
  onUpdate: (scope: string, content: string, confidence: number) => void;
}) {
  const [editing, setEditing] = useState(false);
  const [scope, setScope] = useState(memory.scope);
  const [content, setContent] = useState(memory.content);
  const [confidence, setConfidence] = useState(memory.confidence);

  useEffect(() => {
    setScope(memory.scope);
    setContent(memory.content);
    setConfidence(memory.confidence);
  }, [memory.scope, memory.content, memory.confidence]);

  return (
    <article className="memory-row">
      <div>
        <div className="memory-title-line">
          <strong>{memory.scope}</strong>
          <span className={`status ${memory.status}`}>{memory.status}</span>
        </div>
        <p>{memory.content}</p>
        <span className="memory-confidence">confidence {memory.confidence.toFixed(2)}</span>
        {editing && (
          <div className="memory-edit">
            <label>
              Scope
              <input value={scope} onChange={(event) => setScope(event.target.value)} />
            </label>
            <label>
              Content
              <textarea value={content} onChange={(event) => setContent(event.target.value)} />
            </label>
            <label>
              Confidence
              <input
                max="1"
                min="0"
                step="0.05"
                type="number"
                value={confidence}
                onChange={(event) => setConfidence(Number(event.target.value))}
              />
            </label>
          </div>
        )}
      </div>
      <div className="memory-actions">
        {memory.status === "pending" && (
          <>
          <button onClick={onApprove}>Approve</button>
          <button onClick={onReject}>Reject</button>
          </>
        )}
        {editing ? (
          <>
            <button onClick={() => onUpdate(scope, content, confidence)}>Save</button>
            <button onClick={() => setEditing(false)}>Close</button>
          </>
        ) : (
          <button onClick={() => setEditing(true)}>Edit</button>
        )}
        <button onClick={onDelete}>Delete</button>
      </div>
    </article>
  );
}

function MainAgentChat() {
  const queryClient = useQueryClient();
  const [content, setContent] = useState("");
  const messages = useQuery({ queryKey: ["main-agent-messages"], queryFn: listMainAgentMessages });
  const mutation = useMutation({
    mutationFn: sendMainAgentMessage,
    onSuccess: async () => {
      setContent("");
      await queryClient.invalidateQueries({ queryKey: ["main-agent-messages"] });
      await queryClient.invalidateQueries({ queryKey: ["tasks"] });
      await queryClient.invalidateQueries({ queryKey: ["summary"] });
    },
  });

  const visibleMessages = messages.data ?? [];

  return (
    <section className="panel chat-panel">
      <div className="panel-heading">
        <h2>Main Agent Conversation</h2>
        <span>task tools</span>
      </div>
      <div className="message-list">
        {visibleMessages.map((message) => (
          <MessageBubble key={message.id} message={message} />
        ))}
        {visibleMessages.length === 0 && (
          <p className="empty">Try: create task: Check whether the Persistent-Agent README needs updates</p>
        )}
      </div>
      <form
        className="chat-form"
        onSubmit={(event) => {
          event.preventDefault();
          if (!content.trim()) return;
          mutation.mutate(content);
        }}
      >
        <input
          value={content}
          onChange={(event) => setContent(event.target.value)}
          placeholder="Ask the main agent to create, pause, resume, reprioritize, or summarize tasks"
        />
        <button className="primary icon-button" type="submit" disabled={mutation.isPending}>
          <Send size={16} />
        </button>
      </form>
    </section>
  );
}

function MessageBubble({ message }: { message: ConversationMessage }) {
  return (
    <div className={`message ${message.role === "user" ? "user-message" : "assistant-message"}`}>
      <span>{message.role === "user" ? "You" : "Main agent"}</span>
      <p>{message.content}</p>
    </div>
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

type TimelineEvent = {
  id: string;
  title: string;
  detail: string;
  tone: "info" | "success" | "warning" | "danger";
  timestamp: string;
};

function EventTimeline({ events }: { events: TimelineEvent[] }) {
  return (
    <div className="event-timeline">
      {events.map((event) => (
        <article className={`event-item ${event.tone}`} key={event.id}>
          <div>
            <strong>{event.title}</strong>
            <time>{new Date(event.timestamp).toLocaleTimeString()}</time>
          </div>
          <p>{event.detail}</p>
        </article>
      ))}
      {events.length === 0 && <p className="empty">Waiting for scheduler, task, and main-agent events.</p>}
    </div>
  );
}

function parseAppEvent(raw: string): AppEvent | null {
  try {
    return JSON.parse(raw) as AppEvent;
  } catch {
    return null;
  }
}

function toTimelineEvent(event: AppEvent | null, raw: string): TimelineEvent {
  const timestamp = new Date().toISOString();
  const fallback = {
    id: `${Date.now()}-${Math.random()}`,
    title: "Unknown event",
    detail: raw,
    tone: "warning" as const,
    timestamp,
  };

  if (!event) {
    return fallback;
  }

  switch (event.type) {
    case "task_changed":
      return {
        id: `${timestamp}-task-${event.task.id}`,
        title: `Task ${event.task.status.replaceAll("_", " ")}`,
        detail: event.task.title,
        tone: taskEventTone(event.task.status),
        timestamp,
      };
    case "main_agent_reply":
      return {
        id: `${timestamp}-main-agent-${event.message.id}`,
        title: "Main agent replied",
        detail: event.message.content,
        tone: "info",
        timestamp,
      };
    case "scheduler_tick":
      return schedulerTimelineEvent(event.tick, timestamp);
    case "heartbeat":
      return {
        id: `${timestamp}-heartbeat`,
        title: "Heartbeat",
        detail: "Server event stream is alive.",
        tone: "info",
        timestamp,
      };
    default:
      return fallback;
  }
}

function schedulerTimelineEvent(tick: SchedulerTick, timestamp: string): TimelineEvent {
  const recovered = tick.recovered_tasks.length;
  const requeued = tick.requeued_tasks.length;
  const followUps = tick.outcome.type === "completed" ? tick.outcome.follow_up_tasks.length : 0;
  const taskTitle = tick.claimed_task?.title ?? "No runnable task";
  const suffixParts = [
    recovered > 0 ? `${recovered} expired running task(s) recovered` : "",
    requeued > 0 ? `${requeued} recurring task(s) requeued` : "",
    followUps > 0 ? `${followUps} follow-up task(s) created` : "",
  ].filter(Boolean);
  const suffix = suffixParts.length > 0 ? ` ${suffixParts.join("; ")}.` : "";

  switch (tick.outcome.type) {
    case "completed":
      return {
        id: `${timestamp}-scheduler-completed-${tick.claimed_task?.id ?? "none"}`,
        title: "Scheduler completed a task",
        detail: `${taskTitle}: ${tick.outcome.summary}${suffix}`,
        tone: "success",
        timestamp,
      };
    case "blocked":
      return {
        id: `${timestamp}-scheduler-blocked-${tick.claimed_task?.id ?? "none"}`,
        title: "Scheduler needs user input",
        detail: `${taskTitle}: ${tick.outcome.reason}${suffix}`,
        tone: "warning",
        timestamp,
      };
    case "failed":
      return {
        id: `${timestamp}-scheduler-failed-${tick.claimed_task?.id ?? "none"}`,
        title: "Scheduler failed a task",
        detail: `${taskTitle}: ${tick.outcome.error}${suffix}`,
        tone: "danger",
        timestamp,
      };
    case "retry_scheduled":
      return {
        id: `${timestamp}-scheduler-retry-${tick.claimed_task?.id ?? "none"}`,
        title: "Scheduler scheduled a retry",
        detail: `${taskTitle}: ${tick.outcome.error}. Attempt ${tick.outcome.next_attempt} of ${tick.outcome.max_attempts} will run later.${suffix}`,
        tone: "warning",
        timestamp,
      };
    case "superseded":
      return {
        id: `${timestamp}-scheduler-superseded-${tick.claimed_task?.id ?? "none"}`,
        title: "Scheduler preserved task state",
        detail: `${taskTitle}: ${tick.outcome.reason}${suffix}`,
        tone: tick.outcome.status === "cancelled" ? "danger" : "warning",
        timestamp,
      };
    case "idle":
      return {
        id: `${timestamp}-scheduler-idle`,
        title: "Scheduler idle",
        detail: `${taskTitle}.${suffix}`,
        tone: "info",
        timestamp,
      };
  }
}

function ExecutionStatePanel({ loading, state }: { loading: boolean; state?: SchedulerState }) {
  const running = state?.running_tasks ?? [];
  const primaryRunning = running[0];
  const nextQueued = state?.next_queued_task;

  return (
    <div className="execution-state">
      <div className="execution-state-item">
        <span>Current run</span>
        <strong>{loading ? "Loading" : primaryRunning?.title ?? "Idle"}</strong>
        {running.length > 1 && <small>{running.length} running tasks</small>}
      </div>
      <div className="execution-state-item">
        <span>Next queued</span>
        <strong>{nextQueued?.title ?? "None"}</strong>
        <small>{state ? `${state.queued_count} queued` : "Waiting for state"}</small>
      </div>
      <div className="execution-state-item">
        <span>Needs user</span>
        <strong>{state?.waiting_for_user_count ?? 0}</strong>
      </div>
      <div className="execution-state-item">
        <span>Scheduled</span>
        <strong>{state?.waiting_for_schedule_count ?? 0}</strong>
      </div>
    </div>
  );
}

function taskEventTone(status: Task["status"]): TimelineEvent["tone"] {
  if (status === "completed") return "success";
  if (status === "failed" || status === "cancelled") return "danger";
  if (status === "waiting_for_user" || status === "paused") return "warning";
  return "info";
}

function TaskComposer() {
  const queryClient = useQueryClient();
  const skills = useQuery({ queryKey: ["skills"], queryFn: listSkills });
  const [title, setTitle] = useState("");
  const [description, setDescription] = useState("");
  const [taskType, setTaskType] = useState<TaskType>("one_off");
  const [priority, setPriority] = useState(0);
  const [intervalSeconds, setIntervalSeconds] = useState(300);
  const [requestedSkills, setRequestedSkills] = useState<string[]>([]);

  const mutation = useMutation({
    mutationFn: createTask,
    onSuccess: async () => {
      setTitle("");
      setDescription("");
      setPriority(0);
      setIntervalSeconds(300);
      setRequestedSkills([]);
      await queryClient.invalidateQueries({ queryKey: ["tasks"] });
      await queryClient.invalidateQueries({ queryKey: ["summary"] });
    },
  });
  const visibleSkills = skills.data ?? [];

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
            requested_skills: requestedSkills,
            schedule: taskType === "recurring" ? { interval_seconds: intervalSeconds } : undefined,
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
        {taskType === "recurring" && (
          <label>
            Interval seconds
            <input
              type="number"
              min="0"
              value={intervalSeconds}
              onChange={(event) => setIntervalSeconds(Number(event.target.value))}
            />
          </label>
        )}
        {visibleSkills.length > 0 && (
          <fieldset className="skill-picker">
            <legend>Requested skills</legend>
            <div>
              {visibleSkills.map((skill) => (
                <label className="skill-choice" key={skill.id}>
                  <input
                    checked={requestedSkills.includes(skill.name)}
                    type="checkbox"
                    onChange={(event) => {
                      setRequestedSkills((current) =>
                        event.target.checked
                          ? [...current, skill.name]
                          : current.filter((name) => name !== skill.name),
                      );
                    }}
                  />
                  <span>{skill.name}</span>
                </label>
              ))}
            </div>
          </fieldset>
        )}
        <button className="primary" type="submit" disabled={mutation.isPending}>
          <Plus size={16} /> Create task
        </button>
      </form>
    </section>
  );
}

function TaskRow({ task }: { task: Task }) {
  const queryClient = useQueryClient();
  const [showDetails, setShowDetails] = useState(false);
  const [priorityDraft, setPriorityDraft] = useState(task.priority);
  const [queueDraft, setQueueDraft] = useState(task.queue_position);
  const refresh = async () => {
    await queryClient.invalidateQueries({ queryKey: ["tasks"] });
    await queryClient.invalidateQueries({ queryKey: ["summary"] });
    await queryClient.invalidateQueries({ queryKey: ["task-history", task.id] });
  };
  useEffect(() => {
    setPriorityDraft(task.priority);
    setQueueDraft(task.queue_position);
  }, [task.priority, task.queue_position]);
  const pause = useMutation({ mutationFn: pauseTask, onSuccess: refresh });
  const resume = useMutation({ mutationFn: resumeTask, onSuccess: refresh });
  const cancel = useMutation({ mutationFn: cancelTask, onSuccess: refresh });
  const reprioritize = useMutation({
    mutationFn: () => reprioritizeTask(task.id, priorityDraft),
    onSuccess: refresh,
  });
  const reorder = useMutation({
    mutationFn: () => reorderTask(task.id, queueDraft),
    onSuccess: refresh,
  });

  const statusClass = useMemo(() => `status ${task.status.replaceAll("_", "-")}`, [task.status]);

  return (
    <article className="task-row">
      <div>
        <div className="task-title-line">
          <h3>{task.title}</h3>
          <span className={statusClass}>{task.status.replaceAll("_", " ")}</span>
        </div>
        <p>{task.description || "No description"}</p>
        {task.blocked_reason && <p className="blocked-reason">{task.blocked_reason}</p>}
        <div className="task-meta">
          <span>{task.task_type.replace("_", " ")}</span>
          <span>priority {task.priority}</span>
          <span>queue {task.queue_position}</span>
          <span>attempts {task.attempt_count}</span>
          {task.requested_skills.length > 0 && <span>requested {task.requested_skills.join(", ")}</span>}
          {task.matched_skills.length > 0 && <span>matched {task.matched_skills.join(", ")}</span>}
          {task.next_run_at && <span>next {new Date(task.next_run_at).toLocaleString()}</span>}
        </div>
        <div className="queue-controls">
          <label>
            Priority
            <input type="number" value={priorityDraft} onChange={(event) => setPriorityDraft(Number(event.target.value))} />
          </label>
          <button title="Apply priority" onClick={() => reprioritize.mutate()} disabled={reprioritize.isPending || priorityDraft === task.priority}>
            <Check size={15} />
          </button>
          <label>
            Queue
            <input type="number" value={queueDraft} onChange={(event) => setQueueDraft(Number(event.target.value))} />
          </label>
          <button title="Apply queue position" onClick={() => reorder.mutate()} disabled={reorder.isPending || queueDraft === task.queue_position}>
            <Check size={15} />
          </button>
        </div>
      </div>
      <div className="task-actions">
        <button title="Task details" onClick={() => setShowDetails((value) => !value)}><History size={16} /></button>
        {task.status === "paused" ? (
          <button title="Resume task" onClick={() => resume.mutate(task.id)}><Play size={16} /></button>
        ) : (
          <button title="Pause task" onClick={() => pause.mutate(task.id)}><CirclePause size={16} /></button>
        )}
        <button title="Cancel task" onClick={() => cancel.mutate(task.id)}><SquareX size={16} /></button>
      </div>
      {showDetails && <TaskDetailPanel task={task} />}
    </article>
  );
}

function TaskDetailPanel({ task }: { task: Task }) {
  return (
    <div className="task-detail">
      <TaskEditForm task={task} />
      <section className="task-result">
        <h4>Latest result</h4>
        {task.result_summary ? <p>{task.result_summary}</p> : <p className="empty">No result summary yet.</p>}
        {task.blocked_reason && (
          <>
            <h4>Needs user</h4>
            <p className="blocked-reason">{task.blocked_reason}</p>
          </>
        )}
      </section>
      <TaskConversation task={task} />
      <TaskHistoryPanel taskId={task.id} />
    </div>
  );
}

function TaskEditForm({ task }: { task: Task }) {
  const queryClient = useQueryClient();
  const skills = useQuery({ queryKey: ["skills"], queryFn: listSkills });
  const [editing, setEditing] = useState(false);
  const [title, setTitle] = useState(task.title);
  const [description, setDescription] = useState(task.description);
  const [requestedSkills, setRequestedSkills] = useState<string[]>(task.requested_skills);

  useEffect(() => {
    setTitle(task.title);
    setDescription(task.description);
    setRequestedSkills(task.requested_skills);
  }, [task.description, task.requested_skills, task.title]);

  const mutation = useMutation({
    mutationFn: () => updateTask(task.id, { title: title.trim(), description, requested_skills: requestedSkills }),
    onSuccess: async () => {
      setEditing(false);
      await queryClient.invalidateQueries({ queryKey: ["tasks"] });
      await queryClient.invalidateQueries({ queryKey: ["summary"] });
      await queryClient.invalidateQueries({ queryKey: ["task-history", task.id] });
    },
  });
  const skillOptions = Array.from(new Set([...(skills.data ?? []).map((skill) => skill.name), ...task.requested_skills])).sort();

  return (
    <section className="task-edit">
      <div className="task-edit-heading">
        <h4>Task brief</h4>
        {editing ? (
          <div>
            <button
              title="Save task brief"
              onClick={() => {
                if (title.trim()) mutation.mutate();
              }}
              disabled={mutation.isPending || !title.trim()}
            >
              <Check size={15} />
            </button>
            <button
              title="Cancel editing"
              onClick={() => {
                setTitle(task.title);
                setDescription(task.description);
                setRequestedSkills(task.requested_skills);
                setEditing(false);
              }}
            >
              <X size={15} />
            </button>
          </div>
        ) : (
          <button title="Edit task brief" onClick={() => setEditing(true)}>
            <Pencil size={15} />
          </button>
        )}
      </div>
      {editing ? (
        <div className="task-edit-fields">
          <label>
            Title
            <input value={title} onChange={(event) => setTitle(event.target.value)} />
          </label>
          <label>
            Description
            <textarea value={description} onChange={(event) => setDescription(event.target.value)} />
          </label>
          {skillOptions.length > 0 && (
            <fieldset className="skill-picker">
              <legend>Requested skills</legend>
              <div>
                {skillOptions.map((skillName) => (
                  <label className="skill-choice" key={skillName}>
                    <input
                      checked={requestedSkills.includes(skillName)}
                      type="checkbox"
                      onChange={(event) => {
                        setRequestedSkills((current) =>
                          event.target.checked
                            ? [...current, skillName]
                            : current.filter((name) => name !== skillName),
                        );
                      }}
                    />
                    <span>{skillName}</span>
                  </label>
                ))}
              </div>
            </fieldset>
          )}
        </div>
      ) : (
        <div className="task-brief">
          <strong>{task.title}</strong>
          <p>{task.description || "No description"}</p>
          {task.requested_skills.length > 0 && <span>requested {task.requested_skills.join(", ")}</span>}
          {task.matched_skills.length > 0 && <span>matched {task.matched_skills.join(", ")}</span>}
        </div>
      )}
    </section>
  );
}

function TaskHistoryPanel({ taskId }: { taskId: string }) {
  const queryClient = useQueryClient();
  const [dependencyTaskId, setDependencyTaskId] = useState("");
  const [resourceKey, setResourceKey] = useState("");
  const history = useQuery({
    queryKey: ["task-history", taskId],
    queryFn: () => getTaskHistory(taskId),
  });
  const attempts = history.data?.attempts ?? [];
  const attemptEvents = history.data?.attempt_events ?? [];
  const artifacts = history.data?.artifacts ?? [];
  const memoryCandidates = history.data?.memory_candidates ?? [];
  const dependencies = history.data?.dependencies ?? [];
  const resourceLocks = history.data?.resource_locks ?? [];
  const notes = history.data?.notes ?? [];
  const actions = history.data?.actions ?? [];
  const addLock = useMutation({
    mutationFn: (value: string) => addTaskResourceLock(taskId, value),
    onSuccess: async () => {
      setResourceKey("");
      await queryClient.invalidateQueries({ queryKey: ["task-history", taskId] });
    },
  });
  const addDependency = useMutation({
    mutationFn: (value: string) => addTaskDependency(taskId, value),
    onSuccess: async () => {
      setDependencyTaskId("");
      await queryClient.invalidateQueries({ queryKey: ["task-history", taskId] });
      await queryClient.invalidateQueries({ queryKey: ["tasks"] });
    },
  });
  const removeDependency = useMutation({
    mutationFn: (value: string) => removeTaskDependency(taskId, value),
    onSuccess: async () => {
      await queryClient.invalidateQueries({ queryKey: ["task-history", taskId] });
      await queryClient.invalidateQueries({ queryKey: ["tasks"] });
    },
  });
  const removeLock = useMutation({
    mutationFn: (value: string) => removeTaskResourceLock(taskId, value),
    onSuccess: async () => {
      await queryClient.invalidateQueries({ queryKey: ["task-history", taskId] });
    },
  });

  return (
    <div className="task-history">
      <div className="history-column">
        <h4>Attempts</h4>
        {attempts.map((attempt) => (
          <div className="history-item" key={attempt.id}>
            <div>
              <span className={`status ${attempt.status.replaceAll("_", "-")}`}>{attempt.status.replaceAll("_", " ")}</span>
              <time>{new Date(attempt.started_at).toLocaleString()}</time>
            </div>
            <p>{attempt.summary || "No summary"}</p>
          </div>
        ))}
        {!history.isLoading && attempts.length === 0 && <p className="empty">No attempts yet.</p>}
      </div>
      <div className="history-column">
        <h4>Worker Events</h4>
        {attemptEvents.map((event) => (
          <div className="history-item" key={event.id}>
            <div>
              <strong>{event.event_type}</strong>
              <time>{new Date(event.created_at).toLocaleString()}</time>
            </div>
            <p>{event.message}</p>
            <code>{JSON.stringify(event.details)}</code>
          </div>
        ))}
        {!history.isLoading && attemptEvents.length === 0 && <p className="empty">No worker events yet.</p>}
      </div>
      <div className="history-column">
        <h4>Artifacts</h4>
        {artifacts.map((artifact) => (
          <div className="history-item" key={artifact.id}>
            <div>
              <strong>{artifact.name}</strong>
              <span>{artifact.artifact_type}</span>
            </div>
            {artifact.summary && <p>{artifact.summary}</p>}
            <code>{artifact.uri}</code>
          </div>
        ))}
        {!history.isLoading && artifacts.length === 0 && <p className="empty">No artifacts yet.</p>}
      </div>
      <div className="history-column">
        <h4>Memory Candidates</h4>
        {memoryCandidates.map((memory) => (
          <div className="history-item" key={memory.id}>
            <div>
              <span className={`status ${memory.status}`}>{memory.status}</span>
              <span>{memory.confidence.toFixed(2)}</span>
            </div>
            <p>{memory.content}</p>
          </div>
        ))}
        {!history.isLoading && memoryCandidates.length === 0 && <p className="empty">No memory candidates yet.</p>}
      </div>
      <div className="history-column">
        <h4>Dependencies</h4>
        <form
          className="resource-lock-form"
          onSubmit={(event) => {
            event.preventDefault();
            const value = dependencyTaskId.trim();
            if (value) {
              addDependency.mutate(value);
            }
          }}
        >
          <input
            placeholder="depends_on_task_id"
            value={dependencyTaskId}
            onChange={(event) => setDependencyTaskId(event.target.value)}
          />
          <button disabled={addDependency.isPending || !dependencyTaskId.trim()} title="Add dependency">
            <Plus size={14} />
          </button>
        </form>
        {dependencies.map((dependency) => (
          <div className="history-item resource-lock-item" key={`${dependency.task_id}-${dependency.depends_on_task_id}`}>
            <div>
              <strong>{dependency.depends_on_task_id}</strong>
              <button
                disabled={removeDependency.isPending}
                title="Remove dependency"
                onClick={() => removeDependency.mutate(dependency.depends_on_task_id)}
              >
                <X size={14} />
              </button>
            </div>
            <p>depends on</p>
            <time>{new Date(dependency.created_at).toLocaleString()}</time>
          </div>
        ))}
        {!history.isLoading && dependencies.length === 0 && <p className="empty">No dependencies yet.</p>}
      </div>
      <div className="history-column">
        <h4>Resource Locks</h4>
        <form
          className="resource-lock-form"
          onSubmit={(event) => {
            event.preventDefault();
            const value = resourceKey.trim();
            if (value) {
              addLock.mutate(value);
            }
          }}
        >
          <input
            placeholder="repo:owner/name"
            value={resourceKey}
            onChange={(event) => setResourceKey(event.target.value)}
          />
          <button disabled={addLock.isPending || !resourceKey.trim()} title="Add resource lock">
            <Plus size={14} />
          </button>
        </form>
        {resourceLocks.map((lock) => (
          <div className="history-item resource-lock-item" key={`${lock.task_id}-${lock.resource_key}`}>
            <div>
              <strong>{lock.resource_key}</strong>
              <button
                disabled={removeLock.isPending}
                title="Remove resource lock"
                onClick={() => removeLock.mutate(lock.resource_key)}
              >
                <X size={14} />
              </button>
            </div>
            <p>{lock.lock_mode}</p>
            <time>{new Date(lock.created_at).toLocaleString()}</time>
          </div>
        ))}
        {!history.isLoading && resourceLocks.length === 0 && <p className="empty">No resource locks yet.</p>}
      </div>
      <div className="history-column">
        <h4>Notes</h4>
        {notes.map((note) => (
          <div className="history-item" key={note.id}>
            <div>
              <strong>{note.actor}</strong>
              <time>{new Date(note.created_at).toLocaleString()}</time>
            </div>
            <p>{note.content}</p>
          </div>
        ))}
        {!history.isLoading && notes.length === 0 && <p className="empty">No notes yet.</p>}
      </div>
      <div className="history-column">
        <h4>Actions</h4>
        {actions.map((action) => (
          <div className="history-item" key={action.id}>
            <div>
              <strong>{action.action_type}</strong>
              <time>{new Date(action.created_at).toLocaleString()}</time>
            </div>
            <p>{action.actor}</p>
            <code>{JSON.stringify(action.details)}</code>
          </div>
        ))}
        {!history.isLoading && actions.length === 0 && <p className="empty">No actions yet.</p>}
      </div>
    </div>
  );
}

function TaskConversation({ task }: { task: Task }) {
  const queryClient = useQueryClient();
  const [content, setContent] = useState("");
  const messages = useQuery({
    queryKey: ["task-messages", task.id],
    queryFn: () => listTaskMessages(task.id),
  });
  const mutation = useMutation({
    mutationFn: (value: string) => sendTaskMessage(task.id, value),
    onSuccess: async () => {
      setContent("");
      await queryClient.invalidateQueries({ queryKey: ["task-messages", task.id] });
      await queryClient.invalidateQueries({ queryKey: ["tasks"] });
      await queryClient.invalidateQueries({ queryKey: ["summary"] });
    },
  });
  const visibleMessages = messages.data ?? [];

  return (
    <div className="task-conversation">
      <div className="task-conversation-messages">
        {visibleMessages.map((message) => (
          <MessageBubble key={message.id} message={message} />
        ))}
        {visibleMessages.length === 0 && <p className="empty">No task conversation yet.</p>}
      </div>
      <form
        className="chat-form"
        onSubmit={(event) => {
          event.preventDefault();
          if (!content.trim()) return;
          mutation.mutate(content);
        }}
      >
        <input
          value={content}
          onChange={(event) => setContent(event.target.value)}
          placeholder="Reply with the missing context"
        />
        <button className="primary icon-button" type="submit" disabled={mutation.isPending}>
          <Send size={16} />
        </button>
      </form>
    </div>
  );
}

function SchedulerButton() {
  const queryClient = useQueryClient();
  const mutation = useMutation({
    mutationFn: runSchedulerTick,
    onSuccess: async () => {
      await queryClient.invalidateQueries({ queryKey: ["tasks"] });
      await queryClient.invalidateQueries({ queryKey: ["summary"] });
      await queryClient.invalidateQueries({ queryKey: ["scheduler-state"] });
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
