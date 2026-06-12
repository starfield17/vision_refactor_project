import React, { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  Activity,
  Boxes,
  CircleDot,
  Database,
  FileText,
  ListRestart,
  Play,
  RadioTower,
  RefreshCw,
  Server,
  Settings,
  Square,
} from "lucide-react";
import "./styles.css";

type NodeRecord = {
  node_id: string;
  role: string;
  status: string;
  endpoint: string;
  version: string;
  last_seen_utc: string;
  payload?: Record<string, unknown>;
};

type JobRecord = {
  job_id: string;
  kind: string;
  target_role: string;
  target_node_id: string;
  upstream_job_id: string;
  status: string;
  created_at_utc: string;
  updated_at_utc: string;
  error: string;
  payload?: Record<string, unknown>;
};

type WorkerStatus = {
  ok: boolean;
  node: NodeRecord;
  status?: Record<string, unknown>;
  error?: string;
};

const apiBase = (import.meta.env.VITE_CONTROL_PLANE_API_URL || "http://127.0.0.1:7800").replace(
  /\/$/,
  "",
);
const apiToken = import.meta.env.VITE_CONTROL_PLANE_API_TOKEN || "";

function apiHeaders(extra: Record<string, string> = {}): Record<string, string> {
  return apiToken ? { ...extra, Authorization: `Bearer ${apiToken}` } : extra;
}

async function apiGet<T>(path: string): Promise<T> {
  const response = await fetch(`${apiBase}${path}`, { headers: apiHeaders() });
  if (!response.ok) {
    throw new Error(`${response.status} ${response.statusText}`);
  }
  return (await response.json()) as T;
}

async function apiPost<T>(path: string, body: Record<string, unknown>): Promise<T> {
  const response = await fetch(`${apiBase}${path}`, {
    method: "POST",
    headers: apiHeaders({ "Content-Type": "application/json" }),
    body: JSON.stringify(body),
  });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `${response.status} ${response.statusText}`);
  }
  return (await response.json()) as T;
}

function statusTone(status: string): string {
  if (["online", "running", "succeeded"].includes(status)) return "good";
  if (["queued", "upstream_unreachable"].includes(status)) return "warn";
  if (["failed", "cancelled", "interrupted"].includes(status)) return "bad";
  return "idle";
}

function Stat({ label, value, icon }: { label: string; value: string | number; icon: React.ReactNode }) {
  return (
    <div className="metric">
      <div className="metricIcon">{icon}</div>
      <div>
        <div className="metricLabel">{label}</div>
        <div className="metricValue">{value}</div>
      </div>
    </div>
  );
}

function App() {
  const [tab, setTab] = useState("overview");
  const [nodes, setNodes] = useState<NodeRecord[]>([]);
  const [jobs, setJobs] = useState<JobRecord[]>([]);
  const [workers, setWorkers] = useState<WorkerStatus[]>([]);
  const [models, setModels] = useState<Record<string, unknown>[]>([]);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [jobKind, setJobKind] = useState("train");
  const [jobPayload, setJobPayload] = useState("{\n  \"dry_run\": true\n}");
  const [logs, setLogs] = useState("");

  const counts = useMemo(() => {
    const online = nodes.filter((node) => node.status === "online").length;
    const active = jobs.filter((job) => ["queued", "running", "upstream_unreachable"].includes(job.status)).length;
    const failed = jobs.filter((job) => job.status === "failed").length;
    return { online, active, failed };
  }, [jobs, nodes]);

  async function refresh() {
    setBusy(true);
    setError("");
    try {
      const [nodeData, jobData, workerData, modelData] = await Promise.all([
        apiGet<{ ok: boolean; nodes: NodeRecord[] }>("/api/v1/nodes"),
        apiGet<{ ok: boolean; jobs: JobRecord[] }>("/api/v1/jobs?refresh=true"),
        apiGet<{ ok: boolean; workers: WorkerStatus[] }>("/api/v1/workers/status"),
        apiGet<{ ok: boolean; models: Record<string, unknown>[] }>("/api/v1/models"),
      ]);
      setNodes(nodeData.nodes || []);
      setJobs(jobData.jobs || []);
      setWorkers(workerData.workers || []);
      setModels(modelData.models || []);
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : String(exc));
    } finally {
      setBusy(false);
    }
  }

  async function submitJob() {
    setBusy(true);
    setError("");
    try {
      const payload = JSON.parse(jobPayload) as Record<string, unknown>;
      await apiPost("/api/v1/jobs", { kind: jobKind, payload });
      await refresh();
      setTab("jobs");
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : String(exc));
    } finally {
      setBusy(false);
    }
  }

  async function cancelJob(jobId: string) {
    setBusy(true);
    setError("");
    try {
      await apiPost(`/api/v1/jobs/${jobId}/cancel`, {});
      await refresh();
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : String(exc));
    } finally {
      setBusy(false);
    }
  }

  async function loadLogs(jobId: string) {
    setError("");
    try {
      const data = await apiGet<{ ok: boolean; text: string }>(`/api/v1/jobs/${jobId}/logs?tail=240`);
      setLogs(data.text || "");
      setTab("logs");
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : String(exc));
    }
  }

  useEffect(() => {
    void refresh();
    const timer = window.setInterval(() => void refresh(), 10000);
    return () => window.clearInterval(timer);
  }, []);

  return (
    <main className="shell">
      <aside className="sidebar">
        <div className="brand">
          <RadioTower size={22} />
          <span>Vision Control</span>
        </div>
        {[
          ["overview", Activity, "Overview"],
          ["nodes", Server, "Nodes"],
          ["jobs", Boxes, "Jobs"],
          ["models", Database, "Models"],
          ["logs", FileText, "Logs"],
          ["settings", Settings, "Settings"],
        ].map(([id, Icon, label]) => {
          const TabIcon = Icon as typeof Activity;
          return (
            <button key={id as string} className={tab === id ? "nav active" : "nav"} onClick={() => setTab(id as string)}>
              <TabIcon size={18} />
              <span>{label as string}</span>
            </button>
          );
        })}
      </aside>

      <section className="content">
        <header className="topbar">
          <div>
            <h1>Distributed Vision Operations</h1>
            <p>{apiBase}</p>
          </div>
          <button className="primary" onClick={() => void refresh()} disabled={busy} title="Refresh">
            <RefreshCw size={18} />
            <span>{busy ? "Refreshing" : "Refresh"}</span>
          </button>
        </header>

        {error ? <div className="alert">{error}</div> : null}

        {tab === "overview" ? (
          <div className="stack">
            <section className="metrics">
              <Stat label="Online Nodes" value={counts.online} icon={<Server size={18} />} />
              <Stat label="Active Jobs" value={counts.active} icon={<Activity size={18} />} />
              <Stat label="Failed Jobs" value={counts.failed} icon={<CircleDot size={18} />} />
              <Stat label="Model Manifests" value={models.length} icon={<Database size={18} />} />
            </section>

            <section className="panel">
              <div className="panelHead">
                <h2>Submit Job</h2>
                <button className="primary" onClick={() => void submitJob()} disabled={busy} title="Submit job">
                  <Play size={17} />
                  <span>Submit</span>
                </button>
              </div>
              <div className="formGrid">
                <label>
                  <span>Kind</span>
                  <select value={jobKind} onChange={(event) => setJobKind(event.target.value)}>
                    <option value="train">train</option>
                    <option value="autolabel">autolabel</option>
                    <option value="edge_run">edge_run</option>
                  </select>
                </label>
                <label className="payload">
                  <span>Payload JSON</span>
                  <textarea value={jobPayload} onChange={(event) => setJobPayload(event.target.value)} />
                </label>
              </div>
            </section>

            <NodeTable nodes={nodes} workers={workers} />
          </div>
        ) : null}

        {tab === "nodes" ? <NodeTable nodes={nodes} workers={workers} /> : null}
        {tab === "jobs" ? <JobTable jobs={jobs} onCancel={cancelJob} onLogs={loadLogs} /> : null}
        {tab === "models" ? <ModelTable models={models} /> : null}
        {tab === "logs" ? <pre className="logs">{logs || "No log selected."}</pre> : null}
        {tab === "settings" ? (
          <section className="panel">
            <h2>Settings</h2>
            <div className="settingsGrid">
              <div>
                <span>API URL</span>
                <strong>{apiBase}</strong>
              </div>
              <div>
                <span>API Token</span>
                <strong>{apiToken ? "configured" : "not configured"}</strong>
              </div>
              <div>
                <span>Refresh Interval</span>
                <strong>10s</strong>
              </div>
            </div>
          </section>
        ) : null}
      </section>
    </main>
  );
}

function NodeTable({ nodes, workers }: { nodes: NodeRecord[]; workers: WorkerStatus[] }) {
  const workerByNode = new Map(workers.map((worker) => [worker.node.node_id, worker]));
  return (
    <section className="panel">
      <div className="panelHead">
        <h2>Nodes</h2>
      </div>
      <div className="tableWrap">
        <table>
          <thead>
            <tr>
              <th>Status</th>
              <th>Node</th>
              <th>Role</th>
              <th>Endpoint</th>
              <th>Worker</th>
              <th>Last Seen</th>
            </tr>
          </thead>
          <tbody>
            {nodes.map((node) => {
              const worker = workerByNode.get(node.node_id);
              return (
                <tr key={node.node_id}>
                  <td><span className={`pill ${statusTone(node.status)}`}>{node.status}</span></td>
                  <td className="mono">{node.node_id}</td>
                  <td>{node.role}</td>
                  <td className="mono clip">{node.endpoint}</td>
                  <td>{worker?.ok ? "reachable" : worker?.error || "pending"}</td>
                  <td className="mono">{node.last_seen_utc}</td>
                </tr>
              );
            })}
            {!nodes.length ? <tr><td colSpan={6} className="empty">No registered nodes.</td></tr> : null}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function JobTable({
  jobs,
  onCancel,
  onLogs,
}: {
  jobs: JobRecord[];
  onCancel: (jobId: string) => Promise<void>;
  onLogs: (jobId: string) => Promise<void>;
}) {
  return (
    <section className="panel">
      <div className="panelHead">
        <h2>Jobs</h2>
      </div>
      <div className="tableWrap">
        <table>
          <thead>
            <tr>
              <th>Status</th>
              <th>Kind</th>
              <th>Control Job</th>
              <th>Node</th>
              <th>Upstream</th>
              <th>Updated</th>
              <th>Actions</th>
            </tr>
          </thead>
          <tbody>
            {jobs.map((job) => (
              <tr key={job.job_id}>
                <td><span className={`pill ${statusTone(job.status)}`}>{job.status}</span></td>
                <td>{job.kind}</td>
                <td className="mono clip">{job.job_id}</td>
                <td className="mono">{job.target_node_id || job.target_role}</td>
                <td className="mono clip">{job.upstream_job_id || "-"}</td>
                <td className="mono">{job.updated_at_utc}</td>
                <td className="actions">
                  <button onClick={() => void onLogs(job.job_id)} title="Open logs"><FileText size={16} /></button>
                  <button onClick={() => void onCancel(job.job_id)} title="Cancel job"><Square size={16} /></button>
                </td>
              </tr>
            ))}
            {!jobs.length ? <tr><td colSpan={7} className="empty">No jobs.</td></tr> : null}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function ModelTable({ models }: { models: Record<string, unknown>[] }) {
  return (
    <section className="panel">
      <div className="panelHead">
        <h2>Models</h2>
        <ListRestart size={18} />
      </div>
      <div className="tableWrap">
        <table>
          <thead>
            <tr>
              <th>Model</th>
              <th>Backend</th>
              <th>Path</th>
              <th>Manifest</th>
            </tr>
          </thead>
          <tbody>
            {models.map((model, index) => (
              <tr key={String(model.manifest_path || index)}>
                <td>{String(model.model_id || model.name || "-")}</td>
                <td>{String(model.backend || "-")}</td>
                <td className="mono clip">{String(model.model_path || model.path || "-")}</td>
                <td className="mono clip">{String(model.manifest_path || "-")}</td>
              </tr>
            ))}
            {!models.length ? <tr><td colSpan={4} className="empty">No model manifests.</td></tr> : null}
          </tbody>
        </table>
      </div>
    </section>
  );
}

createRoot(document.getElementById("root")!).render(<App />);
