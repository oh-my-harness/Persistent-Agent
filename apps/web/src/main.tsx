import React, { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import { QueryClient, QueryClientProvider, useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Bot, CirclePause, ListTodo, Play, Plus, RotateCw, Send, SquareX, Zap } from "lucide-react";
import {
  approveMemory,
  cancelTask,
  createSkill,
  createTask,
  getTaskPoolSummary,
  listMainAgentMessages,
  listMemories,
  listSkills,
  listTaskMessages,
  listTasks,
  pauseTask,
  rejectMemory,
  resumeTask,
  runSchedulerTick,
  sendMainAgentMessage,
  sendTaskMessage,
} from "./api";
import type { ConversationMessage, Memory, Task, TaskType } from "./types";
import type { Skill } from "./types";
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
      void queryClient.invalidateQueries({ queryKey: ["main-agent-messages"] });
      void queryClient.invalidateQueries({ queryKey: ["memories"] });
      void queryClient.invalidateQueries({ queryKey: ["skills"] });
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
          <MainAgentChat />
          <SkillManager />
          <MemoryReview />
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

function SkillManager() {
  const queryClient = useQueryClient();
  const skills = useQuery({ queryKey: ["skills"], queryFn: listSkills });
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [rules, setRules] = useState("");
  const mutation = useMutation({
    mutationFn: createSkill,
    onSuccess: async () => {
      setName("");
      setDescription("");
      setRules("");
      await queryClient.invalidateQueries({ queryKey: ["skills"] });
    },
  });

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
          <SkillRow key={skill.id} skill={skill} />
        ))}
        {visibleSkills.length === 0 && <p className="empty">Add a skill to enable automatic task matching.</p>}
      </div>
    </section>
  );
}

function SkillRow({ skill }: { skill: Skill }) {
  return (
    <article className="skill-row">
      <strong>{skill.name}</strong>
      <p>{skill.description || "No description"}</p>
      <span>{skill.trigger_rules.join(", ") || "no triggers"}</span>
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
  onReject,
}: {
  memory: Memory;
  onApprove: () => void;
  onReject: () => void;
}) {
  return (
    <article className="memory-row">
      <div>
        <div className="memory-title-line">
          <strong>{memory.scope}</strong>
          <span className={`status ${memory.status}`}>{memory.status}</span>
        </div>
        <p>{memory.content}</p>
      </div>
      {memory.status === "pending" && (
        <div className="memory-actions">
          <button onClick={onApprove}>Approve</button>
          <button onClick={onReject}>Reject</button>
        </div>
      )}
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
          <p className="empty">Try: 创建任务：检查 Persistent-Agent 的 README 是否需要更新</p>
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
          placeholder="Ask the main agent to create or summarize tasks"
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

function TaskComposer() {
  const queryClient = useQueryClient();
  const [title, setTitle] = useState("");
  const [description, setDescription] = useState("");
  const [taskType, setTaskType] = useState<TaskType>("one_off");
  const [priority, setPriority] = useState(0);
  const [intervalSeconds, setIntervalSeconds] = useState(300);

  const mutation = useMutation({
    mutationFn: createTask,
    onSuccess: async () => {
      setTitle("");
      setDescription("");
      setPriority(0);
      setIntervalSeconds(300);
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
        {task.blocked_reason && <p className="blocked-reason">{task.blocked_reason}</p>}
        <div className="task-meta">
          <span>{task.task_type.replace("_", " ")}</span>
          <span>priority {task.priority}</span>
          <span>attempts {task.attempt_count}</span>
          {task.matched_skills.length > 0 && <span>matched {task.matched_skills.join(", ")}</span>}
          {task.next_run_at && <span>next {new Date(task.next_run_at).toLocaleString()}</span>}
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
      {task.status === "waiting_for_user" && <TaskConversation task={task} />}
    </article>
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
