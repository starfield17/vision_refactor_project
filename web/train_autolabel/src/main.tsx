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

type JobKind = "train" | "autolabel";

type OverrideTemplate = {
  id: string;
  label: string;
  kind: JobKind;
  description: string;
  body: string;
};

const overrideTemplates: OverrideTemplate[] = [
  {
    id: "train_dry_run",
    label: "Train smoke test",
    kind: "train",
    description: "Validate the training config without a long run.",
    body: "train.dry_run=true\ntrain.epochs=1",
  },
  {
    id: "autolabel_model",
    label: "AutoLabel / ONNX",
    kind: "autolabel",
    description: "Use a trained YOLO or Faster-RCNN ONNX model.",
    body: [
      "autolabel.mode=model",
      "autolabel.model.backend=yolo",
      "autolabel.model.onnx_model=./work-dir/models/exp001/model-int8.onnx",
      "autolabel.visualize=true",
    ].join("\n"),
  },
  {
    id: "autolabel_locate_anything",
    label: "AutoLabel / LocateAnything",
    kind: "autolabel",
    description: "Open-vocabulary grounding for bootstrapping labels from class_map.names.",
    body: [
      "autolabel.mode=locate_anything",
      "autolabel.visualize=true",
      "locate_anything.model=nvidia/LocateAnything-3B",
      "locate_anything.device=cuda",
      "locate_anything.generation_mode=hybrid",
      "locate_anything.max_images=20",
    ].join("\n"),
  },
  {
    id: "autolabel_llm",
    label: "AutoLabel / Vision LLM",
    kind: "autolabel",
    description: "Use an OpenAI-compatible vision API with structured prompts.",
    body: [
      "autolabel.mode=llm",
      "autolabel.visualize=true",
      "autolabel.llm.max_images=5",
    ].join("\n"),
  },
];

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
  const [selectedKind, setSelectedKind] = useState<JobKind>("train");
  const [overrides, setOverrides] = useState(overrideTemplates[0].body);
  const [message, setMessage] = useState("");
  const [busy, setBusy] = useState(false);

  const activeJob = useMemo(
    () => jobs.find((job) => job.status === "queued" || job.status === "running"),
    [jobs],
  );

  const visibleTemplates = useMemo(
    () => overrideTemplates.filter((template) => template.kind === selectedKind),
    [selectedKind],
  );

  function chooseKind(kind: JobKind) {
    setSelectedKind(kind);
    const firstTemplate = overrideTemplates.find((template) => template.kind === kind);
    if (firstTemplate) setOverrides(firstTemplate.body);
  }

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
          <button className={selectedKind === "train" ? "active" : ""} onClick={() => chooseKind("train")}>
            Train
          </button>
          <button
            className={selectedKind === "autolabel" ? "active" : ""}
            onClick={() => chooseKind("autolabel")}
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
          <div className="presetGrid">
            {visibleTemplates.map((template) => (
              <button key={template.id} className="presetButton" onClick={() => setOverrides(template.body)}>
                <strong>{template.label}</strong>
                <span>{template.description}</span>
              </button>
            ))}
          </div>
          {selectedKind === "autolabel" && (
            <div className="hintBox">
              <strong>LocateAnything flow</strong>
              <span>
                Pick the LocateAnything preset to query every class in <code>class_map.names</code>, write labels into
                <code> data.labeled_dir</code>, and save raw grounding output under the run output directory.
              </span>
            </div>
          )}
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
