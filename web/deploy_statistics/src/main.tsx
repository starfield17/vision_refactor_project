import React, { useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import { Play, RefreshCw, Square } from "lucide-react";
import "./styles.css";

type Dashboard = {
  overview: Record<string, number | string>;
  filtered_summary: Record<string, number>;
  source_rows: Array<Record<string, number | string>>;
  class_rows: Array<Record<string, number | string>>;
  events: Array<Record<string, unknown>>;
};

type Job = {
  job_id: string;
  kind: string;
  status: string;
  updated_at_utc: string;
  run_id: string;
};

const apiBase = import.meta.env.VITE_DEPLOY_STATISTICS_API_URL || "http://127.0.0.1:7797";
const apiToken = import.meta.env.VITE_DEPLOY_STATISTICS_API_TOKEN || "";

async function api(path: string, init: RequestInit = {}) {
  const headers = new Headers(init.headers);
  headers.set("Accept", "application/json");
  if (apiToken) headers.set("Authorization", `Bearer ${apiToken}`);
  if (init.body) headers.set("Content-Type", "application/json");
  const res = await fetch(`${apiBase}${path}`, { ...init, headers });
  const data = await res.json();
  if (!res.ok || data.ok === false) {
    throw new Error(data.detail || data.error || `HTTP ${res.status}`);
  }
  return data;
}

function App() {
  const [dashboard, setDashboard] = useState<Dashboard | null>(null);
  const [jobs, setJobs] = useState<Job[]>([]);
  const [overrides, setOverrides] = useState("deploy.edge.mode=local");
  const [message, setMessage] = useState("");

  async function refresh() {
    const [dash, jobData] = await Promise.all([
      api("/api/v1/statistics/dashboard?limit=200"),
      api("/api/v1/jobs?limit=30"),
    ]);
    setDashboard(dash.dashboard);
    setJobs(jobData.jobs || []);
  }

  async function submitEdge() {
    setMessage("");
    try {
      const cleanOverrides = overrides
        .split("\n")
        .map((line) => line.trim())
        .filter((line) => line && line.includes("="));
      const data = await api("/api/v1/deploy/edge/jobs", {
        method: "POST",
        body: JSON.stringify({ overrides: cleanOverrides }),
      });
      setMessage(`submitted ${data.job.job_id}`);
      await refresh();
    } catch (err) {
      setMessage(err instanceof Error ? err.message : String(err));
    }
  }

  async function remote(command: "start" | "stop") {
    try {
      const data = await api(`/api/v1/deploy/remote/${command}`, { method: "POST", body: "{}" });
      setMessage(`remote ${data.status}`);
    } catch (err) {
      setMessage(err instanceof Error ? err.message : String(err));
    }
  }

  useEffect(() => {
    refresh().catch((err) => setMessage(err.message));
    const id = window.setInterval(() => refresh().catch(() => undefined), 4000);
    return () => window.clearInterval(id);
  }, []);

  const overview = dashboard?.overview || {};
  const summary = dashboard?.filtered_summary || {};

  return (
    <main>
      <header>
        <div>
          <h1>Deploy & Statistics</h1>
          <p>{apiBase}</p>
        </div>
        <button className="iconButton" onClick={refresh} title="Refresh">
          <RefreshCw size={18} />
        </button>
      </header>

      <section className="metrics">
        <div><span>Total Events</span><strong>{overview.events_total ?? 0}</strong></div>
        <div><span>Total Detections</span><strong>{overview.detections_total ?? 0}</strong></div>
        <div><span>Avg Latency</span><strong>{Number(overview.avg_latency_ms ?? 0).toFixed(1)} ms</strong></div>
        <div><span>P95 Latency</span><strong>{Number(summary.latency_p95_ms ?? 0).toFixed(1)} ms</strong></div>
      </section>

      <section className="toolbar">
        <button className="primary" onClick={submitEdge}><Play size={16} /> Edge Job</button>
        <button onClick={() => remote("start")}><Play size={16} /> Remote</button>
        <button className="danger" onClick={() => remote("stop")}><Square size={16} /> Remote</button>
      </section>

      <section className="grid">
        <div className="panel">
          <h2>Deploy Overrides</h2>
          <textarea value={overrides} onChange={(event) => setOverrides(event.target.value)} />
          {message && <p className="message">{message}</p>}
        </div>
        <div className="panel">
          <h2>Sources</h2>
          <div className="tableWrap">
            <table>
              <thead><tr><th>Source</th><th>Events</th><th>Detections</th><th>P95</th></tr></thead>
              <tbody>
                {(dashboard?.source_rows || []).map((row) => (
                  <tr key={String(row.source_id)}>
                    <td>{row.source_id}</td>
                    <td>{row.events}</td>
                    <td>{row.detections}</td>
                    <td>{Number(row.latency_p95_ms).toFixed(1)} ms</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
        <div className="panel jobsPanel">
          <h2>Jobs</h2>
          <div className="tableWrap">
            <table className="jobsTable">
              <thead><tr><th>Kind</th><th>Status</th><th>Run</th><th>Updated</th></tr></thead>
              <tbody>
                {jobs.map((job) => (
                  <tr key={job.job_id}>
                    <td>{job.kind}</td>
                    <td><span className={`status ${job.status}`}>{job.status}</span></td>
                    <td className="runCell" title={job.run_id || job.job_id}>{job.run_id || job.job_id.slice(0, 8)}</td>
                    <td className="dateCell">{job.updated_at_utc.slice(0, 19).replace("T", " ")}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
        <div className="panel">
          <h2>Classes</h2>
          <div className="tableWrap">
            <table>
              <thead><tr><th>Class</th><th>Count</th><th>Share</th></tr></thead>
              <tbody>
                {(dashboard?.class_rows || []).map((row) => (
                  <tr key={String(row.class_name)}>
                    <td>{row.class_name}</td>
                    <td>{row.count}</td>
                    <td>{Number(row.share_pct).toFixed(1)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </section>
    </main>
  );
}

createRoot(document.getElementById("root")!).render(<App />);
