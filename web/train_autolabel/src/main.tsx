import React, { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import { Play, RefreshCw, Square } from "lucide-react";
import "./styles.css";

type Job = {
  job_id: string;
  kind: string;
  status: string;
  created_at_utc: string;
  updated_at_utc: string;
  run_id: string;
  run_dir: string;
  error: string;
  result: Record<string, unknown>;
};

const apiBase = import.meta.env.VITE_TRAIN_AUTOLABEL_API_URL || "http://127.0.0.1:7793";
const apiToken = import.meta.env.VITE_TRAIN_AUTOLABEL_API_TOKEN || "";

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
  const [jobs, setJobs] = useState<Job[]>([]);
  const [selectedKind, setSelectedKind] = useState("train");
  const [overrides, setOverrides] = useState("train.dry_run=true");
  const [message, setMessage] = useState("");
  const [busy, setBusy] = useState(false);

  const activeJob = useMemo(
    () => jobs.find((job) => job.status === "queued" || job.status === "running"),
    [jobs],
  );

  async function refresh() {
    const data = await api("/api/v1/jobs?limit=50");
    setJobs(data.jobs || []);
  }

  async function submitJob() {
    setBusy(true);
    setMessage("");
    try {
      const cleanOverrides = overrides
        .split("\n")
        .map((line) => line.trim())
        .filter((line) => line && line.includes("="));
      const path = selectedKind === "train" ? "/api/v1/train/jobs" : "/api/v1/autolabel/jobs";
      const data = await api(path, {
        method: "POST",
        body: JSON.stringify({ overrides: cleanOverrides }),
      });
      setMessage(`submitted ${data.job.job_id}`);
      await refresh();
    } catch (err) {
      setMessage(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  async function cancel(jobId: string) {
    setBusy(true);
    try {
      await api(`/api/v1/jobs/${jobId}/cancel`, { method: "POST", body: "{}" });
      await refresh();
    } catch (err) {
      setMessage(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  useEffect(() => {
    refresh().catch((err) => setMessage(err.message));
    const id = window.setInterval(() => refresh().catch(() => undefined), 3000);
    return () => window.clearInterval(id);
  }, []);

  return (
    <main>
      <header>
        <div>
          <h1>Train & Autolabel</h1>
          <p>{apiBase}</p>
        </div>
        <button className="iconButton" onClick={refresh} title="Refresh">
          <RefreshCw size={18} />
        </button>
      </header>

      <section className="toolbar">
        <div className="segmented">
          <button className={selectedKind === "train" ? "active" : ""} onClick={() => setSelectedKind("train")}>
            Train
          </button>
          <button
            className={selectedKind === "autolabel" ? "active" : ""}
            onClick={() => setSelectedKind("autolabel")}
          >
            Autolabel
          </button>
        </div>
        <button className="primary" onClick={submitJob} disabled={busy || Boolean(activeJob)}>
          <Play size={16} />
          Submit
        </button>
        {activeJob && (
          <button className="danger" onClick={() => cancel(activeJob.job_id)} disabled={busy}>
            <Square size={16} />
            Cancel
          </button>
        )}
      </section>

      <section className="grid">
        <div className="panel">
          <h2>Overrides</h2>
          <textarea value={overrides} onChange={(event) => setOverrides(event.target.value)} />
          {message && <p className="message">{message}</p>}
        </div>
        <div className="panel">
          <h2>Jobs</h2>
          <table>
            <thead>
              <tr>
                <th>Kind</th>
                <th>Status</th>
                <th>Run</th>
                <th>Updated</th>
              </tr>
            </thead>
            <tbody>
              {jobs.map((job) => (
                <tr key={job.job_id}>
                  <td>{job.kind}</td>
                  <td><span className={`status ${job.status}`}>{job.status}</span></td>
                  <td>{job.run_id || job.job_id.slice(0, 8)}</td>
                  <td>{job.updated_at_utc.slice(0, 19).replace("T", " ")}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </main>
  );
}

createRoot(document.getElementById("root")!).render(<App />);
