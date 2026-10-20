import { useCallback, useEffect, useState } from "react";
import {
  Activity,
  AlertTriangle,
  ArrowDownRight,
  ArrowUpRight,
  Bell,
  BookOpen,
  Boxes,
  CheckCircle2,
  ChevronRight,
  CircleDot,
  Clock3,
  Cpu,
  Gauge,
  LayoutDashboard,
  Package,
  Plane,
  ShieldCheck,
  Sparkles,
  Wifi,
  WifiOff,
  Wrench,
  Waves,
  Zap,
} from "lucide-react";

import "./App.css";
const API_BASE = (import.meta.env.VITE_API_BASE || "http://127.0.0.1:8000").replace(/\/+$/, "");
const navItems = [
  { id: "overview", label: "Fleet Overview", icon: LayoutDashboard },
  { id: "diagnostics", label: "AI Diagnostics", icon: Activity },
  { id: "assurance", label: "System Assurance", icon: ShieldCheck },
  { id: "maintenance", label: "Maintenance", icon: Wrench },
  { id: "inventory", label: "Spare Inventory", icon: Boxes },
  { id: "copilot", label: "Mechanic Co-Pilot", icon: BookOpen },
];

/* Generic local API list hook */
function useApiList(endpoint) {
  const [items, setItems] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    let active = true;

    async function loadData() {
      try {
        const response = await fetch(`${API_BASE}${endpoint}`);

        if (!response.ok) {
          throw new Error(`API returned HTTP ${response.status}`);
        }

        const payload = await response.json();

        if (!Array.isArray(payload.items)) {
          throw new Error("Unexpected API response format");
        }

        if (active) {
          setItems(payload.items);
          setError("");
        }
      } catch (err) {
        if (active) {
          setError(err.message || "Unable to load records");
        }
      } finally {
        if (active) {
          setLoading(false);
        }
      }
    }

    loadData();

    const timer = window.setInterval(loadData, 15000);

    return () => {
      active = false;
      window.clearInterval(timer);
    };
  }, [endpoint]);

  return { items, loading, error };
}

/* Fetch NASA C-MAPSS RUL predictions from the trained local model */
function useRulPredictions() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    let active = true;

    async function loadPredictions() {
      try {
        const response = await fetch(
          `${API_BASE}/api/v1/ai/rul-predictions`
        );

        if (!response.ok) {
          throw new Error(
            `RUL API returned HTTP ${response.status}`
          );
        }

        const payload = await response.json();

        if (!Array.isArray(payload.predictions)) {
          throw new Error("Invalid RUL API response");
        }

        if (active) {
          setData(payload);
          setError("");
        }
      } catch (err) {
        if (active) {
          setError(err.message || "Could not load RUL predictions");
        }
      } finally {
        if (active) {
          setLoading(false);
        }
      }
    }

    loadPredictions();

    return () => {
      active = false;
    };
  }, []);

  return { data, loading, error };
}

/* Fetch the saved evaluation and baseline-comparison reports */
function useModelMetrics() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    let active = true;

    async function loadMetrics() {
      try {
        const response = await fetch(
          `${API_BASE}/api/v1/ai/model-metrics`
        );

        if (!response.ok) {
          throw new Error(
            `Metrics API returned HTTP ${response.status}`
          );
        }

        const payload = await response.json();

        if (active) {
          setData(payload);
          setError("");
        }
      } catch (err) {
        if (active) {
          setError(err.message || "Could not load model metrics");
        }
      } finally {
        if (active) {
          setLoading(false);
        }
      }
    }

    loadMetrics();

    return () => {
      active = false;
    };
  }, []);

  return { data, loading, error };
}

function formatMetric(value) {
  if (value === null || value === undefined) {
    return "—";
  }

  const number = Number(value);

  return Number.isFinite(number) ? number.toFixed(2) : "—";
}

function MetricCard({
  icon: Icon,
  label,
  value,
  detail,
  change,
  tone = "cyan",
  down = false,
}) {
  return (
    <article className="metric-card">
      <div className={`metric-icon ${tone}`}>
        <Icon size={19} />
      </div>

      <div className="metric-label">{label}</div>
      <div className="metric-value">{value}</div>

      <div className="metric-bottom">
        <span className={down ? "metric-change negative" : "metric-change"}>
          {down ? (
            <ArrowDownRight size={14} />
          ) : (
            <ArrowUpRight size={14} />
          )}
          {change}
        </span>

        <span className="muted small">{detail}</span>
      </div>
    </article>
  );
}

function SectionHeading({ eyebrow, title, description }) {
  return (
    <div className="section-heading">
      <div>
        <div className="eyebrow">{eyebrow}</div>
        <h2>{title}</h2>
        {description && <p>{description}</p>}
      </div>
    </div>
  );
}

function StatusBadge({ tone = "neutral", children }) {
  return (
    <span className={`status-badge ${tone}`}>
      {children}
    </span>
  );
}

function TableState({ loading, error, empty, children }) {
  if (loading) {
    return (
      <div className="table-state">
        Loading service records…
      </div>
    );
  }

  if (error) {
    return (
      <div className="table-state error">
        Could not load data: {error}
      </div>
    );
  }

  if (empty) {
    return (
      <div className="table-state">
        No records found.
      </div>
    );
  }

  return children;
}

/* Database-backed asset table */
function AssetTable({ assets }) {
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            <th>ASSET ID</th>
            <th>SUBSYSTEM</th>
            <th>HEALTH INDEX</th>
            <th>STATUS</th>
            <th>EST. OPERATING WINDOW</th>
          </tr>
        </thead>

        <tbody>
          {assets.map((asset) => (
            <tr key={asset.id}>
              <td>
                <span className="asset-id">
                  <Plane size={15} />
                  {asset.id}
                </span>
              </td>

              <td>{asset.subsystem}</td>

              <td>
                <div className="health-cell">
                  <span>{asset.health}%</span>

                  <div className="health-track">
                    <div
                      className={`health-fill ${asset.tone}`}
                      style={{ width: `${asset.health}%` }}
                    />
                  </div>
                </div>
              </td>

              <td>
                <StatusBadge tone={asset.tone}>
                  {asset.status}
                </StatusBadge>
              </td>

              <td className="muted">
                {asset.operating_window}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function Overview({
  simulatedFault,
  onToggleFault,
  backendOnline,
  databaseOnline,
  assetsData,
  assetsLoading,
  assetsError,
}) {
  return (
    <>
      <div className="demo-banner">
        <div className="demo-banner-icon">
          <Zap size={17} />
        </div>

        <div>
          <strong>SIMULATION ENVIRONMENT</strong>
          <span>
            Asset records are served by the application API. Fleet summary
            figures and the trend chart are illustrative demo values,
            not live aircraft telemetry.
          </span>
        </div>

        <span className="demo-tag">DEMO DATA</span>
      </div>

      <div className="metrics-grid">
        <MetricCard
          icon={Gauge}
          label="Fleet availability"
          value={simulatedFault ? "85.8%" : "86.4%"}
          change={simulatedFault ? "−0.6 pp" : "+2.4 pp"}
          detail="simulated baseline"
          tone="cyan"
          down={simulatedFault}
        />

        <MetricCard
          icon={Plane}
          label="Ready assets"
          value={simulatedFault ? "18 / 22" : "19 / 22"}
          change={simulatedFault ? "−1 asset" : "Illustrative"}
          detail="simulated fleet"
          tone="green"
          down={simulatedFault}
        />

        <MetricCard
          icon={Activity}
          label="Items for review"
          value={simulatedFault ? "05" : "04"}
          change={simulatedFault ? "+1 demo event" : "Demo summary"}
          detail="not certified alerts"
          tone="amber"
          down={simulatedFault}
        />

        <MetricCard
          icon={Clock3}
          label="Planned maintenance"
          value="07"
          change="Next 72 cycles"
          detail="illustrative forecast"
          tone="purple"
        />
      </div>

      <div className="main-grid">
        <section className="panel health-panel">
          <div className="panel-heading">
            <div>
              <div className="eyebrow">FLEET MONITOR</div>
              <h3>Asset health overview</h3>
            </div>

            <span className="live-label">
              <span className="live-dot" />
              SERVICE API DATA
            </span>
          </div>

          <TableState
            loading={assetsLoading}
            error={assetsError}
            empty={assetsData.length === 0}
          >
            <AssetTable assets={assetsData} />
          </TableState>

          <div className="panel-footer">
            <span>
              {assetsLoading
                ? "Loading assets..."
                : `${assetsData.length} database records shown`}
            </span>
            <span>Synthetic demonstration records</span>
          </div>
        </section>

        <section className="panel attention-panel">
          <div className="panel-heading">
            <div>
              <div className="eyebrow">ATTENTION REQUIRED</div>
              <h3>Maintenance signals</h3>
            </div>

            <span className="alert-count">
              {simulatedFault ? "03" : "02"}
            </span>
          </div>

          {simulatedFault && (
            <div className="signal-item critical">
              <div className="signal-icon">
                <AlertTriangle size={17} />
              </div>

              <div className="signal-content">
                <strong>Simulated sensor anomaly</strong>
                <span>AG-021 · Injected demo event</span>
                <small>Human review required</small>
              </div>
            </div>
          )}

          <div className="signal-item warning">
            <div className="signal-icon">
              <Waves size={17} />
            </div>

            <div className="signal-content">
              <strong>Hydraulic health review</strong>
              <span>AG-021 · Synthetic reading</span>
              <small>Pending validation</small>
            </div>
          </div>

          <div className="signal-item info">
            <div className="signal-icon">
              <Cpu size={17} />
            </div>

            <div className="signal-content">
              <strong>Sensor integrity check</strong>
              <span>AG-032 · Simulated status</span>
              <small>Inspection suggested</small>
            </div>
          </div>

          <div className="simulation-box">
            <div className="simulation-box-top">
              <Sparkles size={17} />
              <span>FAULT LAB</span>
              <span className="simulation-label">MANUAL TEST</span>
            </div>

            <p>
              Inject a synthetic anomaly to test the interface.
              This button does not run an ML model.
            </p>

            <button
              className={
                simulatedFault
                  ? "secondary-button full-width"
                  : "primary-button full-width"
              }
              onClick={onToggleFault}
            >
              {simulatedFault
                ? "Clear simulated fault"
                : "Simulate sensor anomaly"}
            </button>
          </div>
        </section>
      </div>

      <div className="bottom-grid">
        <section className="panel trend-panel">
          <div className="panel-heading">
            <div>
              <div className="eyebrow">ILLUSTRATIVE TREND</div>
              <h3>Health index trend</h3>
            </div>

            <StatusBadge tone="blue">Sample series</StatusBadge>
          </div>

          <div className="chart-legend">
            <span>
              <i className="legend-dot cyan-dot" />
              Sample health index
            </span>
            <span className="muted">Not live telemetry</span>
          </div>

          <div className="chart-box">
            <div className="chart-y-labels">
              <span>100</span>
              <span>80</span>
              <span>60</span>
              <span>40</span>
            </div>

            <svg
              viewBox="0 0 620 160"
              preserveAspectRatio="none"
              role="img"
              aria-label="Illustrative sample health trend"
            >
              <defs>
                <linearGradient
                  id="areaFill"
                  x1="0"
                  y1="0"
                  x2="0"
                  y2="1"
                >
                  <stop
                    offset="0%"
                    stopColor="#38d9ce"
                    stopOpacity=".24"
                  />
                  <stop
                    offset="100%"
                    stopColor="#38d9ce"
                    stopOpacity="0"
                  />
                </linearGradient>
              </defs>

              {[20, 58, 96, 134].map((y) => (
                <line
                  key={y}
                  x1="0"
                  y1={y}
                  x2="620"
                  y2={y}
                  className="grid-line"
                />
              ))}

              <path
                d="M0 42 L65 38 L130 48 L195 43 L260 62 L325 57 L390 76 L455 70 L520 88 L570 96 L620 91 L620 155 L0 155 Z"
                fill="url(#areaFill)"
              />

              <path
                d="M0 42 L65 38 L130 48 L195 43 L260 62 L325 57 L390 76 L455 70 L520 88 L570 96 L620 91"
                fill="none"
                stroke="#38d9ce"
                strokeWidth="2.5"
                vectorEffect="non-scaling-stroke"
              />

              <circle cx="620" cy="91" r="4" fill="#38d9ce" />
            </svg>
          </div>

          <div className="chart-x-labels">
            <span>T−10</span>
            <span>T−8</span>
            <span>T−6</span>
            <span>T−4</span>
            <span>T−2</span>
            <span>Now</span>
          </div>
        </section>

        <section className="panel backend-panel">
          <div className="panel-heading">
            <div>
              <div className="eyebrow">SERVICE CONNECTIVITY</div>
              <h3>System status</h3>
            </div>

            {backendOnline ? (
              <CheckCircle2 className="success-icon" size={21} />
            ) : (
              <WifiOff className="warning-icon" size={21} />
            )}
          </div>

          <div className="system-row">
            <span>
              <CircleDot size={15} />
              Backend API
            </span>

            <StatusBadge tone={backendOnline ? "green" : "amber"}>
              {backendOnline ? "Connected" : "Unavailable"}
            </StatusBadge>
          </div>

          <div className="system-row">
            <span>
              <Package size={15} />
              SQLite database
            </span>

            <StatusBadge tone={databaseOnline ? "green" : "amber"}>
              {databaseOnline ? "Connected" : "Unavailable"}
            </StatusBadge>
          </div>

          <div className="system-row">
            <span>
              <Cpu size={15} />
              Predictive AI engine
            </span>

            <StatusBadge tone="neutral">
              Trained baseline
            </StatusBadge>
          </div>

          <div className="system-note">
            <ShieldCheck size={17} />
            <span>
              Backend and database connectivity are monitored.
              Full network-disconnected operation has not yet been verified.
            </span>
          </div>
        </section>
      </div>
    </>
  );
}

function EvidenceAuditPanel() {
  const [latestRun, setLatestRun] = useState(null);
  const [auditItems, setAuditItems] = useState([]);
  const [assets, setAssets] = useState([]);
  const [mappings, setMappings] = useState([]);
  const [engineUnit, setEngineUnit] = useState("1");
  const [assetId, setAssetId] = useState("");
  const [mappingType, setMappingType] = useState("DEMO_ONLY");
  const [mappingNote, setMappingNote] = useState(
    "Demonstration mapping only; no real-world equipment equivalence asserted."
  );
  const [mappingReference, setMappingReference] = useState("");
  const [sensorScreen, setSensorScreen] = useState(null);
  const [loading, setLoading] = useState(false);
  const [mappingSaving, setMappingSaving] = useState(false);
  const [error, setError] = useState("");
  const [mappingMessage, setMappingMessage] = useState("");

  const readResponse = useCallback(async (response, fallback) => {
    let payload = {};
    try {
      payload = await response.json();
    } catch {
      // The fallback message is used if the server response is not JSON.
    }
    if (!response.ok) {
      const message = typeof payload.detail === "string" ? payload.detail : fallback;
      throw new Error(message || fallback);
    }
    return payload;
  }, []);

  const refreshAuditLog = async () => {
    const response = await fetch(`${API_BASE}/api/v1/ai/audit-log?limit=10`);
    if (response.ok) {
      const payload = await response.json();
      setAuditItems(payload.items ?? []);
    }
  };

  const refreshMappings = async () => {
    const response = await fetch(`${API_BASE}/api/v1/ai/engine-mappings`);
    if (response.ok) {
      const payload = await response.json();
      setMappings(payload.items ?? []);
    }
  };

  const loadPanel = useCallback(async () => {
    try {
      const [latestResponse, logResponse, assetResponse, mapResponse, sensorResponse] =
        await Promise.all([
          fetch(`${API_BASE}/api/v1/ai/evidence-gate/latest`),
          fetch(`${API_BASE}/api/v1/ai/audit-log?limit=10`),
          fetch(`${API_BASE}/api/v1/assets`),
          fetch(`${API_BASE}/api/v1/ai/engine-mappings`),
          fetch(`${API_BASE}/api/v1/ai/sensor-screen`),
        ]);

      if (latestResponse.ok) {
        setLatestRun(await latestResponse.json());
      } else if (latestResponse.status === 404) {
        setLatestRun(null);
      } else {
        await readResponse(latestResponse, "Could not load the latest evidence-gate run.");
      }

      if (logResponse.ok) {
        const payload = await logResponse.json();
        setAuditItems(payload.items ?? []);
      }
      if (assetResponse.ok) {
        const payload = await assetResponse.json();
        const nextAssets = payload.items ?? [];
        setAssets(nextAssets);
        setAssetId((previous) => previous || nextAssets[0]?.id || "");
      }
      if (mapResponse.ok) {
        const payload = await mapResponse.json();
        setMappings(payload.items ?? []);
      }
      if (sensorResponse.ok) {
        setSensorScreen(await sensorResponse.json());
      }
    } catch (err) {
      setError(err.message || "Could not load evidence-gate data.");
    } finally {
      setLoading(false);
    }
  }, [readResponse]);

  useEffect(() => {
    let cancelled = false;
    queueMicrotask(() => {
      if (!cancelled) {
        void loadPanel();
      }
    });
    return () => {
      cancelled = true;
    };
  }, [loadPanel]);

  const runEvidenceGate = async () => {
    setLoading(true);
    setError("");
    setMappingMessage("");
    try {
      const response = await fetch(
        `${API_BASE}/api/v1/ai/evidence-gate/run`,
        { method: "POST" }
      );
      const payload = await readResponse(
        response,
        "Evidence Gate run failed. Check the backend terminal."
      );
      setLatestRun(payload);
      await Promise.all([refreshAuditLog(), refreshMappings()]);
      const sensorResponse = await fetch(`${API_BASE}/api/v1/ai/sensor-screen`);
      if (sensorResponse.ok) setSensorScreen(await sensorResponse.json());
    } catch (err) {
      setError(err.message || "Evidence Gate run failed.");
    } finally {
      setLoading(false);
    }
  };

  const saveMapping = async (event) => {
    event.preventDefault();
    setMappingSaving(true);
    setError("");
    setMappingMessage("");
    try {
      const response = await fetch(`${API_BASE}/api/v1/ai/engine-mappings`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          engine_unit: Number(engineUnit),
          asset_id: assetId,
          mapping_type: mappingType,
          mapping_note: mappingNote,
          mapping_reference: mappingReference.trim() || null,
        }),
      });
      await readResponse(response, "Could not save engine-to-asset mapping.");
      setMappingMessage(
        mappingType === "DEMO_ONLY"
          ? `Saved demo-only mapping for FD001-${String(engineUnit).padStart(3, "0")}. This does not count as verified evidence.`
          : "Mapping reference saved as user-supplied documentation; it has not been independently verified."
      );
      await refreshMappings();
    } catch (err) {
      setError(err.message || "Could not save mapping.");
    } finally {
      setMappingSaving(false);
    }
  };

  const decisions = (latestRun?.decisions ?? [])
    .slice()
    .sort((a, b) => {
      const priority = (item) =>
        item.decision === "REVIEW_REQUIRED" ? 0 :
        item.evidence_completeness !== "COMPLETE" ? 1 : 2;
      return priority(a) - priority(b) ||
        (a.predicted_rul_cycles ?? 0) - (b.predicted_rul_cycles ?? 0);
    })
    .slice(0, 10);

  return (
    <section className="panel page-panel">
      <div className="panel-heading">
        <div>
          <div className="eyebrow">EVIDENCE GATE & AUDIT</div>
          <h3>Evidence review and audit trail</h3>
        </div>
        <StatusBadge tone={latestRun ? "blue" : "neutral"}>
          {latestRun ? "Audit saved" : "Not run"}
        </StatusBadge>
      </div>

      <p className="body-copy">
        The evidence gate combines RUL, ensemble disagreement, a training-distribution
        sensor screen, and any explicitly recorded engine-to-asset mapping. The
        sensor screen is exploratory; user-supplied mappings are not independently verified.
      </p>

      <div className="diagnostic-state">
        <div className="diagnostic-symbol warning"><ShieldCheck size={24} /></div>
        <div className="diagnostic-copy">
          <strong>{loading ? "Processing evidence gate..." : "Evidence gate is active"}</strong>
          <span>
            {latestRun
              ? `Last run: ${latestRun.run_id}`
              : "No saved run loaded. Start a run to create an audit snapshot."}
          </span>
        </div>
        <button className="primary-button" onClick={runEvidenceGate} disabled={loading}>
          {loading ? "Please wait..." : "Run Evidence Gate"}
        </button>
      </div>

      {error && <div className="info-callout"><AlertTriangle size={18} /><span>{error}</span></div>}
      {mappingMessage && <div className="info-callout"><ShieldCheck size={18} /><span>{mappingMessage}</span></div>}

      <div className="panel-heading" style={{ marginTop: 24 }}>
        <div>
          <div className="eyebrow">SENSOR SCREENING</div>
          <h3>Training-distribution outlier screen</h3>
        </div>
        <StatusBadge tone={sensorScreen?.screen_positive_count ? "amber" : "blue"}>
          {sensorScreen ? `${sensorScreen.screen_positive_count} screen-positive` : "Loading screen"}
        </StatusBadge>
      </div>
      <p className="body-copy">
        Latest test-cycle readings are compared against the training set's 0.5th–99.5th
        percentile per-sensor ranges. This is a simple statistical screen, not a calibrated
        anomaly detector. A flagged value requires investigation; an unflagged value does not prove health.
      </p>

      <div className="panel-heading" style={{ marginTop: 24 }}>
        <div>
          <div className="eyebrow">EXPLICIT DATA LINKING</div>
          <h3>Engine-to-asset mapping</h3>
        </div>
        <StatusBadge tone="amber">Manual declaration</StatusBadge>
      </div>
      <p className="body-copy">
        NASA FD001 engine IDs and synthetic AG-* demo assets are separate datasets. Only save a
        DOCUMENTED mapping when you have a real crosswalk reference. Use DEMO_ONLY for interface testing.
      </p>

      <form onSubmit={saveMapping} style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: 12, alignItems: "end" }}>
        <label style={{ display: "grid", gap: 6 }}>
          <span className="muted">NASA engine unit</span>
          <select value={engineUnit} onChange={(event) => setEngineUnit(event.target.value)} required>
            {Array.from({ length: 100 }, (_, index) => index + 1).map((unit) => (
              <option key={unit} value={unit}>FD001-{String(unit).padStart(3, "0")}</option>
            ))}
          </select>
        </label>
        <label style={{ display: "grid", gap: 6 }}>
          <span className="muted">Asset reference</span>
          <select value={assetId} onChange={(event) => setAssetId(event.target.value)} required>
            {assets.map((asset) => (
              <option key={asset.id} value={asset.id}>{asset.id} — {asset.subsystem}</option>
            ))}
          </select>
        </label>
        <label style={{ display: "grid", gap: 6 }}>
          <span className="muted">Mapping type</span>
          <select value={mappingType} onChange={(event) => setMappingType(event.target.value)} required>
            <option value="DEMO_ONLY">Demo only (safe default)</option>
            <option value="DOCUMENTED">Documented (reference required)</option>
          </select>
        </label>
        <label style={{ display: "grid", gap: 6 }}>
          <span className="muted">Mapping note</span>
          <input value={mappingNote} onChange={(event) => setMappingNote(event.target.value)} minLength={5} maxLength={1000} required />
        </label>
        <label style={{ display: "grid", gap: 6 }}>
          <span className="muted">Reference (required for DOCUMENTED)</span>
          <input value={mappingReference} onChange={(event) => setMappingReference(event.target.value)} maxLength={1000} placeholder="Document ID / path / source" />
        </label>
        <button className="primary-button" type="submit" disabled={mappingSaving || assets.length === 0}>
          {mappingSaving ? "Saving..." : "Save Mapping"}
        </button>
      </form>

      {mappings.length > 0 ? (
        <div className="table-wrap" style={{ marginTop: 16 }}>
          <table>
            <thead><tr><th>ENGINE</th><th>ASSET REFERENCE</th><th>TYPE</th><th>BASIS</th><th>UPDATED</th></tr></thead>
            <tbody>
              {mappings.map((item) => (
                <tr key={item.engine_unit}>
                  <td className="asset-id">FD001-{String(item.engine_unit).padStart(3, "0")}</td>
                  <td>{item.asset_id} <span className="muted">{item.subsystem}</span></td>
                  <td><StatusBadge tone={item.mapping_type === "DOCUMENTED" ? "blue" : "amber"}>{item.mapping_type}</StatusBadge></td>
                  <td>{item.mapping_note}{item.mapping_reference ? ` · ${item.mapping_reference}` : ""}</td>
                  <td>{item.updated_at}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <div className="table-state">No engine-to-asset mappings saved. No mapping is inferred automatically.</div>
      )}

      {latestRun && (
        <>
          <div className="metrics-grid three" style={{ marginTop: 24 }}>
            <MetricCard icon={AlertTriangle} label="Review required" value={String(latestRun.review_required_count ?? 0)} detail="At least one review rule triggered" tone="purple" change="Human review" />
            <MetricCard icon={ShieldCheck} label="Evidence incomplete" value={String(latestRun.evidence_incomplete_count ?? 0)} detail="Unverified/missing evidence remains" tone="amber" change="No clearance" />
            <MetricCard icon={Activity} label="Sensor screen positive" value={String(latestRun.sensor_screen_positive_count ?? 0)} detail="At least one sensor outside its training quantiles" tone="cyan" change="Investigate flags" />
          </div>

          <div className="panel-footer">
            <span>Run ID: {latestRun.run_id}</span>
            <span>Evaluated: {latestRun.evaluated_at ?? "Timestamp unavailable"}</span>
          </div>

          <div className="table-wrap">
            <table>
              <thead><tr><th>ENGINE</th><th>PREDICTED RUL</th><th>DISAGREEMENT</th><th>SENSOR SCREEN</th><th>GATE DECISION</th><th>EVIDENCE</th><th>REASONS</th></tr></thead>
              <tbody>
                {decisions.map((item) => (
                  <tr key={item.engine_unit ?? item.engine_id}>
                    <td className="asset-id">{item.engine_id ?? `FD001-${String(item.engine_unit).padStart(3, "0")}`}</td>
                    <td>{formatMetric(item.predicted_rul_cycles)} <span className="muted">cycles</span></td>
                    <td>{formatMetric(item.model_disagreement_cycles)} <span className="muted">cycles</span></td>
                    <td>
                      <StatusBadge tone={item.sensor_anomaly_screen_status === "SCREEN_POSITIVE" ? "amber" : "neutral"}>{item.sensor_anomaly_screen_status ?? "UNKNOWN"}</StatusBadge>
                      <div className="muted">{item.anomalous_sensor_count ?? 0} flagged</div>
                    </td>
                    <td><StatusBadge tone={item.decision === "REVIEW_REQUIRED" ? "amber" : "blue"}>{item.decision ?? "UNKNOWN"}</StatusBadge></td>
                    <td><StatusBadge tone={item.evidence_completeness === "COMPLETE" ? "green" : "neutral"}>{item.evidence_completeness ?? "UNKNOWN"}</StatusBadge></td>
                    <td>{(item.review_reasons ?? []).length ? item.review_reasons.join(", ") : "No threshold-triggered rule; evidence still requires review"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="muted">Showing {decisions.length} prioritized decisions from {latestRun.count ?? latestRun.decisions?.length ?? 0} engines.</p>
        </>
      )}

      <div className="panel-heading" style={{ marginTop: 24 }}>
        <div><div className="eyebrow">PERSISTED EVENTS</div><h3>Recent audit records</h3></div>
        <StatusBadge tone="neutral">Latest 10 records</StatusBadge>
      </div>
      {auditItems.length > 0 ? (
        <div className="table-wrap">
          <table>
            <thead><tr><th>EVENT ID</th><th>RUN ID</th><th>ENGINE</th><th>DECISION</th><th>EVIDENCE</th><th>CREATED AT</th></tr></thead>
            <tbody>
              {auditItems.map((item) => (
                <tr key={item.id}>
                  <td>{item.id}</td>
                  <td className="asset-id">{String(item.run_id).slice(0, 8)}…</td>
                  <td>FD001-{String(item.engine_unit).padStart(3, "0")}</td>
                  <td><StatusBadge tone={item.decision === "REVIEW_REQUIRED" ? "amber" : "blue"}>{item.decision}</StatusBadge></td>
                  <td>{item.evidence_completeness}</td>
                  <td>{item.created_at}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <div className="table-state">{loading ? "Loading audit history..." : "No audit records loaded yet. Run the Evidence Gate or check the backend connection."}</div>
      )}

      <div className="info-callout">
        <ShieldCheck size={18} />
        <span>
          Sensor readings are screened against training-set quantiles and need engineering validation. Mapping references are user-supplied and are not verified by this application. Spare-part compatibility is still not cross-mapped. This benchmark workflow does not approve aircraft for operation.
        </span>
      </div>
    </section>
  );
}

function Diagnostics({
  simulatedFault,
  onToggleFault,
  rulData,
  rulLoading,
  rulError,
  metricsData,
  metricsLoading,
  metricsError,
}) {
  const allPredictions = rulData?.predictions ?? [];

  // Display the 10 engines with the lowest predicted RUL.
  const predictions = allPredictions
    .slice()
    .sort(
      (a, b) =>
        a.predicted_rul_cycles - b.predicted_rul_cycles
    )
    .slice(0, 10);

  // Summary counts use all engines, not only the displayed top 10.
  const ensembleSize =
    rulData?.ensemble_size ??
    allPredictions[0]?.ensemble_size ??
    0;

  const reviewCount =
    rulData?.review_count ??
    allPredictions.filter(
      (item) => item.evidence_status === "REVIEW"
    ).length;

  const provisionalCount =
    rulData?.provisional_count ??
    allPredictions.filter(
      (item) => item.evidence_status === "PROVISIONAL"
    ).length;

  const formatReviewReasons = (reasons = []) => {
    const labels = {
      LOW_PREDICTED_RUL: "Low predicted RUL",
      HIGH_MODEL_DISAGREEMENT: "High model disagreement",
    };

    return reasons
      .map((reason) => labels[reason] ?? reason)
      .join(", ");
  };

  const testMetrics =
    metricsData?.training_evaluation?.official_test;

  const comparisonResults =
    metricsData?.baseline_comparison?.results;

  const improvements =
    metricsData?.baseline_comparison?.improvement_vs_baselines;

  const meanBaselineGain =
    improvements?.["Training-mean baseline"]?.mae_reduction_percent;

  return (
    <>
      <SectionHeading
        eyebrow="MODEL ASSURANCE"
        title="AI Diagnostics"
        description="RUL inference, model evaluation and benchmark comparison."
      />

      <div className="metrics-grid three">
        <MetricCard
          icon={Activity}
          label="Official test MAE"
          value={
            metricsLoading
              ? "Loading"
              : metricsError
                ? "Error"
                : formatMetric(testMetrics?.mae)
          }
          detail="lower is better"
          tone="cyan"
          change="Cycles"
        />

        <MetricCard
          icon={Cpu}
          label="Official test RMSE"
          value={
            metricsLoading
              ? "Loading"
              : metricsError
                ? "Error"
                : formatMetric(testMetrics?.rmse)
          }
          detail="test prediction error"
          tone="green"
          change="Cycles"
        />

        <MetricCard
          icon={Gauge}
          label="MAE reduction vs mean"
          value={
            metricsLoading
              ? "Loading"
              : metricsError
                ? "Error"
                : meanBaselineGain == null
                  ? "—"
                  : `${formatMetric(meanBaselineGain)}%`
          }
          detail="same FD001 test set"
          tone="purple"
          change="Baseline comparison"
        />
      </div>

      {(metricsError || rulError) && (
        <div className="info-callout">
          <AlertTriangle size={18} />
          <span>
            {metricsError && `Metrics API: ${metricsError}. `}
            {rulError && `RUL API: ${rulError}. `}
            Check that the FastAPI backend is running.
          </span>
        </div>
      )}

      <section className="panel page-panel">
        <div className="panel-heading">
          <div>
            <div className="eyebrow">FAULT LAB</div>
            <h3>Sensor anomaly simulation</h3>
          </div>

          <StatusBadge tone="blue">
            Synthetic UI event
          </StatusBadge>
        </div>

        <p className="body-copy">
          This demonstration toggles a user-interface event. It does
          not inject sensor data into the trained model or diagnose
          a real aircraft.
        </p>

        <div className="diagnostic-state">
          <div
            className={`diagnostic-symbol ${
              simulatedFault ? "warning" : "healthy"
            }`}
          >
            {simulatedFault ? (
              <AlertTriangle size={26} />
            ) : (
              <CheckCircle2 size={26} />
            )}
          </div>

          <div className="diagnostic-copy">
            <strong>
              {simulatedFault
                ? "Synthetic anomaly injected"
                : "No injected anomaly"}
            </strong>

            <span>
              {simulatedFault
                ? "Demo event active. Human review is required in this simulation."
                : "No fault has been injected into the demo state."}
            </span>
          </div>

          <button
            className="primary-button"
            onClick={onToggleFault}
          >
            {simulatedFault ? "Clear event" : "Inject anomaly"}
          </button>
        </div>
      </section>

      <section className="panel page-panel">
        <div className="panel-heading">
          <div>
            <div className="eyebrow">RUL MODEL INFERENCE</div>
            <h3>Remaining Useful Life estimates</h3>
          </div>

          <StatusBadge tone="blue">NASA FD001</StatusBadge>
        </div>

        <p className="body-copy">
          Predictions come from a five-model bootstrap ensemble trained on NASA C-MAPSS FD001. Model disagreement indicates the
          spread across member predictions; it is not a calibrated
          confidence interval. RUL values are measured in benchmark
          cycles, not hours.
        </p>

                <div className="info-callout">
          <ShieldCheck size={18} />
          <span>
            Model disagreement describes spread across ensemble member predictions; it is not a calibrated confidence interval. RUL estimates use simulated NASA C-MAPSS FD001 benchmark data and are not validated for aircraft maintenance or operational clearance.
          </span>
        </div>

<div className="metrics-grid three">
          <MetricCard
            icon={Cpu}
            label="Ensemble members"
            value={
              rulLoading
                ? "Loading"
                : rulError
                  ? "Error"
                  : String(ensembleSize)
            }
            detail="Bootstrap models"
            tone="cyan"
            change="RUL ensemble"
          />

          <MetricCard
            icon={AlertTriangle}
            label="Engines for review"
            value={
              rulLoading
                ? "Loading"
                : rulError
                  ? "Error"
                  : String(reviewCount)
            }
            detail="Configured review rules"
            tone="purple"
            change="REVIEW"
          />

          <MetricCard
            icon={CheckCircle2}
            label="Provisional results"
            value={
              rulLoading
                ? "Loading"
                : rulError
                  ? "Error"
                  : String(provisionalCount)
            }
            detail="No configured rule triggered"
            tone="green"
            change="Not safety clearance"
          />
        </div>

        <TableState
          loading={rulLoading}
          error={rulError}
          empty={predictions.length === 0}
        >
          <>
            <div className="panel-footer">
              <span>
                Showing {predictions.length} of {rulData?.count ?? 0} test engines
              </span>
              <span>Unit: cycles</span>
            </div>

                        <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>ENGINE ID</th>
                    <th>LAST OBSERVED CYCLE</th>
                    <th>ENSEMBLE RUL</th>
                    <th>MODEL DISAGREEMENT</th>
                    <th>EVIDENCE STATUS</th>
                    <th>REVIEW REASONS</th>
                  </tr>
                </thead>

                <tbody>
                  {predictions.map((item) => (
                    <tr key={item.unit}>
                      <td className="asset-id">
                        FD001-{String(item.unit).padStart(3, "0")}
                      </td>

                      <td>{item.last_observed_cycle}</td>

                      <td>
                        <strong>
                          {formatMetric(item.predicted_rul_cycles)}
                        </strong>{" "}
                        <span className="muted">cycles</span>

                        <div className="muted">
                          Baseline: {formatMetric(item.baseline_rul_cycles)} cycles
                        </div>

                        <div className="muted">
                          Member p10–p90: {formatMetric(item.member_prediction_p10_cycles)}
                          {" – "}
                          {formatMetric(item.member_prediction_p90_cycles)}
                        </div>
                      </td>

                      <td>
                        <strong>
                          {formatMetric(item.model_disagreement_cycles)}
                        </strong>{" "}
                        <span className="muted">cycles</span>
                      </td>

                      <td>
                        <StatusBadge
                          tone={
                            item.evidence_status === "REVIEW"
                              ? "amber"
                              : "blue"
                          }
                        >
                          {item.evidence_status ?? "UNKNOWN"}
                        </StatusBadge>
                      </td>

                      <td>
                        {item.review_reasons?.length > 0 ? (
                          <span>
                            {formatReviewReasons(item.review_reasons)}
                          </span>
                        ) : (
                          <span className="muted">
                            No configured review rule triggered
                          </span>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </>
        </TableState>

        <div className="info-callout">
          <ShieldCheck size={18} />
          <span>
            Benchmark predictions only. These outputs are not validated
            for real aircraft maintenance or operational clearance.
          </span>
        </div>
      </section>

      <EvidenceAuditPanel />

      <section className="panel page-panel">
        <div className="panel-heading">
          <div>
            <div className="eyebrow">MODEL EVALUATION</div>
            <h3>Baseline comparison</h3>
          </div>

          <StatusBadge tone="blue">NASA FD001</StatusBadge>
        </div>

        <p className="body-copy">
          Methods are evaluated against the same official test-engine
          RUL labels. Lower MAE and RMSE indicate lower prediction errors.
        </p>

        <TableState
          loading={metricsLoading}
          error={metricsError}
          empty={
            !comparisonResults ||
            Object.keys(comparisonResults).length === 0
          }
        >
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>MODEL</th>
                  <th>MAE (CYCLES)</th>
                  <th>RMSE (CYCLES)</th>
                </tr>
              </thead>

              <tbody>
                {Object.entries(comparisonResults ?? {}).map(
                  ([name, scores]) => (
                    <tr key={name}>
                      <td>{name}</td>
                      <td>
                        <strong>{formatMetric(scores.mae)}</strong>
                      </td>
                      <td>{formatMetric(scores.rmse)}</td>
                    </tr>
                  )
                )}
              </tbody>
            </table>
          </div>
        </TableState>

        <div className="info-callout">
          <ShieldCheck size={18} />
          <span>
            Results apply to the NASA simulated benchmark. Beating these
            simple baselines does not establish state-of-the-art
            performance or validate the model for real aircraft.
          </span>
        </div>
      </section>
    </>
  );
}

function Maintenance({ orders, loading, error }) {
  return (
    <>
      <SectionHeading
        eyebrow="MRO WORKSPACE"
        title="Maintenance planning"
        description="Work orders retrieved from the maintenance service."
      />

      <section className="panel page-panel">
        <div className="panel-heading">
          <div>
            <div className="eyebrow">WORK ORDER QUEUE</div>
            <h3>Planned activities</h3>
          </div>

          <StatusBadge tone="blue">Service records</StatusBadge>
        </div>

        <TableState
          loading={loading}
          error={error}
          empty={orders.length === 0}
        >
          <div className="work-list">
            {orders.map((order) => (
              <div className="work-item" key={order.id}>
                <div className="work-icon">
                  <Wrench size={19} />
                </div>

                <div className="work-main">
                  <strong>{order.task}</strong>
                  <span>
                    {order.id} · Asset {order.asset_id}
                  </span>
                </div>

                <StatusBadge
                  tone={
                    order.priority === "High"
                      ? "amber"
                      : order.priority === "Medium"
                        ? "blue"
                        : "green"
                  }
                >
                  {order.priority}
                </StatusBadge>

                <span className="work-status">
                  {order.status}
                </span>

                <ChevronRight size={17} className="muted" />
              </div>
            ))}
          </div>
        </TableState>

        <div className="info-callout">
          <ShieldCheck size={18} />
          <span>
            Work orders currently contain synthetic demonstration records.
            Automated maintenance optimization is not connected yet.
          </span>
        </div>
      </section>
    </>
  );
}

function Inventory({ items, loading, error }) {
  return (
    <>
      <SectionHeading
        eyebrow="SUPPLY READINESS"
        title="Spare inventory"
        description="Inventory records retrieved from the service API."
      />

      <section className="panel page-panel">
        <div className="panel-heading">
          <div>
            <div className="eyebrow">PARTS REGISTER</div>
            <h3>Component availability</h3>
          </div>

          <Package size={21} className="muted" />
        </div>

        <TableState
          loading={loading}
          error={error}
          empty={items.length === 0}
        >
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>PART</th>
                  <th>STOCK CODE</th>
                  <th>ON HAND</th>
                  <th>REQUIRED</th>
                  <th>STATUS</th>
                </tr>
              </thead>

              <tbody>
                {items.map((item) => (
                  <tr key={item.code}>
                    <td>{item.name}</td>
                    <td className="asset-id">{item.code}</td>
                    <td>{item.stock}</td>
                    <td>{item.required}</td>

                    <td>
                      <StatusBadge
                        tone={
                          item.status === "Available"
                            ? "green"
                            : item.status === "Limited"
                              ? "amber"
                              : "red"
                        }
                      >
                        {item.status}
                      </StatusBadge>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </TableState>

        <div className="info-callout">
          <Boxes size={18} />
          <span>
            Inventory currently contains synthetic demonstration records.
            No actual procurement decisions are being made.
          </span>
        </div>
      </section>
    </>
  );
}

/* Step 70: read-only system assurance snapshot across local services. */
function firstDefined(...values) {
  return values.find((value) => value !== undefined && value !== null);
}

function displayValue(value, fallback = "Not reported") {
  if (value === undefined || value === null || value === "") return fallback;
  if (typeof value === "boolean") return value ? "Yes" : "No";
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}

function humanizeStatus(value) {
  if (typeof value !== "string" || !value) return "Not reported";
  return value.replaceAll("_", " ");
}

function clearanceLabel(value) {
  if (value === false) return "NOT GRANTED";
  if (value === true) return "TRUE - VERIFY POLICY";
  return "NOT REPORTED";
}

function AssuranceCard({ icon: Icon, title, status, tone = "neutral", children }) {
  return (
    <section className="panel assurance-card">
      <div className="panel-heading">
        <div className="assurance-card-title">
          <span className="assurance-icon"><Icon size={18} /></span>
          <div>
            <div className="eyebrow">SYSTEM MODULE</div>
            <h3>{title}</h3>
          </div>
        </div>
        <StatusBadge tone={tone}>{status}</StatusBadge>
      </div>
      <div className="assurance-card-body">{children}</div>
    </section>
  );
}

function AssuranceRow({ label, value }) {
  return (
    <div className="assurance-row">
      <span>{label}</span>
      <strong>{displayValue(value)}</strong>
    </div>
  );
}

function SystemAssurance() {
  const [snapshot, setSnapshot] = useState({});
  const [errors, setErrors] = useState({});
  const [loading, setLoading] = useState(true);
  const [refreshKey, setRefreshKey] = useState(0);

  useEffect(() => {
    let active = true;
    const endpoints = [
      ["physics", "/api/v1/ai/physics-check/rules-status"],
      ["rag", "/api/v1/rag/status"],
      ["parts", "/api/v1/parts/assessment"],
      ["schedule", "/api/v1/scheduling/latest"],
      ["digitalTwin", "/api/v1/digital-twin/latest"],
      ["audit", "/api/v1/parts/decision-audit?limit=10"],
    ];

    async function loadSnapshot() {
      setLoading(true);
      const results = await Promise.all(endpoints.map(async ([key, path]) => {
        try {
          const response = await fetch(`${API_BASE}${path}`);
          let payload = {};
          try { payload = await response.json(); } catch { /* handled below */ }
          if (!response.ok) {
            const detail = typeof payload.detail === "string" ? payload.detail : `HTTP ${response.status}`;
            return [key, null, detail];
          }
          return [key, payload, ""];
        } catch (error) {
          return [key, null, error?.message || "Local API unavailable"];
        }
      }));

      if (active) {
        const nextSnapshot = {};
        const nextErrors = {};
        results.forEach(([key, payload, error]) => {
          nextSnapshot[key] = payload;
          if (error) nextErrors[key] = error;
        });
        setSnapshot(nextSnapshot);
        setErrors(nextErrors);
        setLoading(false);
      }
    }

    loadSnapshot();
    return () => { active = false; };
  }, [refreshKey]);

  const physics = snapshot.physics ?? {};
  const rules = physics.engineering_rules ?? {};
  const rag = snapshot.rag ?? {};
  const parts = snapshot.parts ?? {};
  const schedule = snapshot.schedule ?? {};
  const digitalTwin = snapshot.digitalTwin ?? {};
  const audit = snapshot.audit ?? {};
  const scheduleGate = firstDefined(schedule.parts_clearance_gate, schedule.clearance_gate, schedule.parts_gate, {});
  const twinScenarios = firstDefined(digitalTwin.scenarios, digitalTwin.comparison?.scenarios, []);
  const twinGate = firstDefined(
    digitalTwin.parts_clearance_gate,
    digitalTwin.clearance_gate,
    twinScenarios.find?.((scenario) => scenario?.parts_clearance_gate)?.parts_clearance_gate,
    {}
  );
  const auditItems = firstDefined(audit.items, audit.events, []);
  const ruleStatus = firstDefined(rules.status, physics.rule_status, "Not reported");
  const partStatus = firstDefined(parts.assessment_status, parts.status, "Not reported");
  const scheduleStatus = firstDefined(schedule.schedule_status, schedule.status, "No saved schedule");
  const twinRunId = firstDefined(digitalTwin.run_id, digitalTwin.comparison_id, digitalTwin.latest_run_id, "No saved comparison");
  const ragStatus = firstDefined(rag.status, rag.index_status, rag.service_status, "API reachable; detailed status fields not exposed");

  return (
    <>
      <SectionHeading
        eyebrow="CROSS-MODULE ASSURANCE"
        title="System Assurance"
        description="Read-only status snapshot of AI, evidence, knowledge, parts and planning services."
      />

      <div className="demo-banner">
        <div className="demo-banner-icon"><ShieldCheck size={17} /></div>
        <div>
          <strong>ASSURANCE STATUS — NOT OPERATIONAL AUTHORIZATION</strong>
          <span>NASA C-MAPSS is simulated benchmark data. Parts mappings may be demo-only, physics rules are not approved, and scheduler outputs remain proposals.</span>
        </div>
        <button className="secondary-button assurance-refresh" onClick={() => setRefreshKey((value) => value + 1)} disabled={loading}>
          <Activity size={15} /> {loading ? "Refreshing..." : "Refresh status"}
        </button>
      </div>

      <div className="assurance-grid">
        <AssuranceCard
          icon={Gauge}
          title="Physics Rule Governance"
          status={errors.physics ? "UNAVAILABLE" : humanizeStatus(ruleStatus)}
          tone={errors.physics ? "amber" : (rules.rule_count > 0 ? "amber" : "neutral")}
        >
          {errors.physics ? <p className="body-copy">{errors.physics}</p> : <>
            <AssuranceRow label="Configured rules" value={firstDefined(rules.rule_count, rules.rules?.length, 0)} />
            <AssuranceRow label="Rule errors" value={firstDefined(rules.errors?.length, 0)} />
            <AssuranceRow label="Dataset" value={firstDefined(physics.dataset, "Not reported")} />
            <p className="assurance-note">Statistical sensor screening is separate from approved engineering physical bounds.</p>
          </>}
        </AssuranceCard>

        <AssuranceCard
          icon={BookOpen}
          title="Technical Knowledge Base"
          status={errors.rag ? "UNAVAILABLE" : "API REACHABLE"}
          tone={errors.rag ? "amber" : "blue"}
        >
          {errors.rag ? <p className="body-copy">{errors.rag}</p> : <>
            <AssuranceRow label="Service status" value={ragStatus} />
            <AssuranceRow label="Indexed documents" value={firstDefined(rag.document_count, rag.indexed_document_count, rag.documents_indexed, "Not exposed by status endpoint")} />
            <AssuranceRow label="Indexed chunks" value={firstDefined(rag.chunk_count, rag.indexed_chunk_count, rag.chunks_indexed, "Not exposed by status endpoint")} />
            <p className="assurance-note">Document approval fields are stored metadata and do not independently authenticate source documents.</p>
          </>}
        </AssuranceCard>

        <AssuranceCard
          icon={Boxes}
          title="Parts Assessment"
          status={errors.parts ? "UNAVAILABLE" : humanizeStatus(partStatus)}
          tone={errors.parts ? "amber" : (String(partStatus).includes("PARTIAL") || String(partStatus).includes("REVIEW") ? "amber" : "neutral")}
        >
          {errors.parts ? <p className="body-copy">{errors.parts}</p> : <>
            <AssuranceRow label="Open work orders" value={firstDefined(parts.open_work_order_count, parts.summary?.open_work_order_count, "Not reported")} />
            <AssuranceRow label="Mapped work orders" value={firstDefined(parts.open_work_orders_with_mappings, parts.summary?.open_work_orders_with_mappings, "Not reported")} />
            <AssuranceRow label="Unmapped work orders" value={firstDefined(parts.open_work_orders_without_mappings, parts.summary?.open_work_orders_without_mappings, "Not reported")} />
            <AssuranceRow label="Scheduling clearance" value={clearanceLabel(parts.scheduling_clearance)} />
            <AssuranceRow label="Stock reserved" value={parts.stock_reservations_created === false ? "NO" : displayValue(parts.stock_reservations_created)} />
          </>}
        </AssuranceCard>

        <AssuranceCard
          icon={Wrench}
          title="Scheduler"
          status={errors.schedule ? "UNAVAILABLE" : humanizeStatus(scheduleStatus)}
          tone={errors.schedule ? "amber" : "blue"}
        >
          {errors.schedule ? <p className="body-copy">{errors.schedule}</p> : <>
            <AssuranceRow label="Solver status" value={firstDefined(schedule.solver_status, "Not reported")} />
            <AssuranceRow label="Parts gate" value={firstDefined(scheduleGate.gate_status, "Not reported")} />
            <AssuranceRow label="Operational clearance" value={clearanceLabel(schedule.operational_clearance)} />
            <AssuranceRow label="Run ID" value={firstDefined(schedule.run_id, "Not reported")} />
          </>}
        </AssuranceCard>

        <AssuranceCard
          icon={Activity}
          title="Digital Twin"
          status={errors.digitalTwin ? "UNAVAILABLE" : "LATEST SNAPSHOT"}
          tone={errors.digitalTwin ? "amber" : "blue"}
        >
          {errors.digitalTwin ? <p className="body-copy">{errors.digitalTwin}</p> : <>
            <AssuranceRow label="Comparison ID" value={twinRunId} />
            <AssuranceRow label="Scenarios" value={Array.isArray(twinScenarios) ? twinScenarios.length : "Not reported"} />
            <AssuranceRow label="Parts gate" value={firstDefined(twinGate.gate_status, "Not reported")} />
            <AssuranceRow label="Operational clearance" value={clearanceLabel(digitalTwin.operational_clearance)} />
          </>}
        </AssuranceCard>

        <AssuranceCard
          icon={ShieldCheck}
          title="Parts Decision Audit"
          status={errors.audit ? "UNAVAILABLE" : "READ-ONLY"}
          tone={errors.audit ? "amber" : "blue"}
        >
          {errors.audit ? <p className="body-copy">{errors.audit}</p> : <>
            <AssuranceRow label="Reported record count" value={firstDefined(audit.count, audit.item_count, auditItems.length)} />
            <AssuranceRow label="Records returned" value={Array.isArray(auditItems) ? auditItems.length : 0} />
            <div className="assurance-audit-list">
              {(Array.isArray(auditItems) ? auditItems.slice(0, 4) : []).map((item, index) => (
                <div className="assurance-audit-item" key={item.id ?? item.audit_id ?? `${item.run_id ?? "record"}-${index}`}>
                  <strong>{displayValue(firstDefined(item.action, item.event_type, item.decision), "Audit event")}</strong>
                  <span>{displayValue(firstDefined(item.work_order_id, item.run_id, item.reference), "Reference not reported")}</span>
                </div>
              ))}
              {(!Array.isArray(auditItems) || auditItems.length === 0) && <p className="assurance-note">No audit rows returned by this endpoint.</p>}
            </div>
          </>}
        </AssuranceCard>
      </div>

      <div className="info-callout">
        <ShieldCheck size={18} />
        <span>This page reads status endpoints only. It does not run the Evidence Gate, start scheduling, create Digital Twin comparisons, modify mappings or reserve inventory. A status response is not engineering approval or maintenance authorization.</span>
      </div>
    </>
  );
}

/* STEP 71 LOCAL RAG COPILOT COMPONENT */
function Copilot() {
  const [query, setQuery] = useState("");
  const [topK, setTopK] = useState("5");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);

  const submitQuery = async (event) => {
    event.preventDefault();
    const cleanQuery = query.trim();
    if (!cleanQuery || loading) return;

    setLoading(true);
    setError("");
    setResult(null);
    try {
      const response = await fetch(`${API_BASE}/api/v1/rag/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: cleanQuery, top_k: Number(topK) }),
      });
      let payload = {};
      try {
        payload = await response.json();
      } catch {
        // Keep a readable fallback for non-JSON server responses.
      }
      if (!response.ok) {
        const detail = typeof payload.detail === "string"
          ? payload.detail
          : `RAG query failed with HTTP ${response.status}`;
        throw new Error(detail);
      }
      setResult(payload);
    } catch (requestError) {
      setError(requestError?.message || "Could not query local documents. Check the backend connection.");
    } finally {
      setLoading(false);
    }
  };

  const decision = result?.decision;
  const decisionLabel = decision === "APPROVED_SOURCE_MATCH_FOUND"
    ? "APPROVED METADATA MATCH — VERIFY APPLICABILITY"
    : decision === "UNVERIFIED_SOURCE_MATCH"
      ? "UNVERIFIED SOURCE MATCH"
      : decision === "INSUFFICIENT_EVIDENCE"
        ? "INSUFFICIENT EVIDENCE — ABSTAINED"
        : decision ? decision.replaceAll("_", " ") : "NO QUERY YET";
  const decisionTone = decision === "APPROVED_SOURCE_MATCH_FOUND"
    ? "blue"
    : decision === "UNVERIFIED_SOURCE_MATCH" || !decision
      ? "amber"
      : "red";
  const sources = Array.isArray(result?.sources) ? result.sources : [];

  return (
    <>
      <SectionHeading
        eyebrow="TECHNICAL KNOWLEDGE RETRIEVAL"
        title="Mechanic Co-Pilot"
        description="Search indexed technical documents and inspect citations, provenance and evidence sufficiency."
      />

      <section className="panel page-panel rag-copilot-panel">
        <div className="panel-heading">
          <div className="rag-panel-title">
            <div className="copilot-mark"><BookOpen size={25} /></div>
            <div>
              <div className="eyebrow">EXTRACTIVE RETRIEVAL</div>
              <h3>Search technical references</h3>
            </div>
          </div>
          <StatusBadge tone="blue">RAG RETRIEVAL</StatusBadge>
        </div>

        <p className="body-copy">
          Answers are retrieved from indexed passages by the document retrieval service. This interface does not generate repair procedures or infer missing instructions.
        </p>

        <form className="rag-query-form" onSubmit={submitQuery}>
          <label className="rag-query-label" htmlFor="rag-query-input">Your question</label>
          <textarea
            id="rag-query-input"
            className="rag-query-input"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            maxLength={1000}
            minLength={1}
            required
            rows={4}
            placeholder="Ask a question that should be answered by an indexed source passage…"
          />
          <div className="rag-query-controls">
            <label className="rag-topk-control">
              <span>Passages to retrieve</span>
              <select value={topK} onChange={(event) => setTopK(event.target.value)}>
                <option value="1">1</option>
                <option value="3">3</option>
                <option value="5">5</option>
                <option value="8">8</option>
                <option value="10">10</option>
              </select>
            </label>
            <button className="primary-button rag-submit-button" type="submit" disabled={loading || !query.trim()}>
              {loading ? "Searching sources…" : "Search documents"}
            </button>
          </div>
        </form>

        <div className="info-callout rag-safety-callout">
          <ShieldCheck size={18} />
          <span>
            Source approval is based on stored metadata and is not independently authenticated. Even a document marked APPROVED must be checked for authenticity, currency and applicability before any real maintenance use. Each query is recorded in the RAG audit log.
          </span>
        </div>

        {error && (
          <div className="info-callout rag-error-callout" role="alert">
            <AlertTriangle size={18} /><span>{error}</span>
          </div>
        )}

        {result && (
          <section className="rag-result" aria-live="polite">
            <div className="panel-heading rag-result-heading">
              <div>
                <div className="eyebrow">RETRIEVAL DECISION</div>
                <h3>Query result</h3>
              </div>
              <StatusBadge tone={decisionTone}>{decisionLabel}</StatusBadge>
            </div>
            <div className="rag-result-summary">
              <div><span>Matching passages</span><strong>{result.result_count ?? sources.length}</strong></div>
              <div><span>Audit ID</span><strong className="rag-audit-id">{result.audit_id ?? "Not returned"}</strong></div>
            </div>
            <div className={`rag-answer ${decision === "INSUFFICIENT_EVIDENCE" ? "rag-answer-abstain" : ""}`}>
              <strong>{decision === "INSUFFICIENT_EVIDENCE" ? "No supported answer found" : "Retrieval response"}</strong>
              <p>{result.answer || result.reason || "No answer text was returned."}</p>
              {result.reason && <p className="rag-reason"><strong>Decision reason:</strong> {result.reason}</p>}
            </div>

            {sources.length > 0 && (
              <div className="rag-sources">
                <div className="panel-heading rag-sources-heading">
                  <div><div className="eyebrow">CITED PASSAGES</div><h3>Retrieved evidence</h3></div>
                  <span className="muted small">Sorted by retrieval score</span>
                </div>
                {sources.map((source, index) => {
                  const provenance = source.provenance ?? {};
                  const approvalStatus = provenance.approval_status || "UNVERIFIED";
                  const isEligible = source.authoritative_eligible === true;
                  return (
                    <article className="rag-source-card" key={source.chunk_id || `${source.source || "source"}-${index}`}>
                      <div className="rag-source-card-heading">
                        <div>
                          <strong>{provenance.title || source.source || `Source ${index + 1}`}</strong>
                          <div className="rag-citation">{source.citation || source.source || "Citation not returned"}</div>
                        </div>
                        <StatusBadge tone={isEligible ? "blue" : "amber"}>
                          {isEligible ? "APPROVED METADATA — VERIFY" : approvalStatus.replaceAll("_", " ")}
                        </StatusBadge>
                      </div>
                      <div className="rag-source-meta">
                        <span>Score: {Number.isFinite(Number(source.score)) ? Number(source.score).toFixed(3) : "not reported"}</span>
                        {provenance.source_organization && <span>Organization: {provenance.source_organization}</span>}
                        {provenance.document_type && <span>Type: {provenance.document_type}</span>}
                        {provenance.revision && <span>Revision: {provenance.revision}</span>}
                        {provenance.effective_date && <span>Effective: {provenance.effective_date}</span>}
                        {provenance.approval_reference && <span>Approval reference: {provenance.approval_reference}</span>}
                        {provenance.reviewed_by && <span>Reviewer metadata: {provenance.reviewed_by}</span>}
                      </div>
                      <blockquote className="rag-excerpt">{source.excerpt || "No excerpt supplied."}</blockquote>
                      {source.authoritative_eligible === true && (
                        <p className="rag-source-warning">This document is marked approved in stored metadata. The application does not independently authenticate the issuer or confirm aircraft applicability.</p>
                      )}
                    </article>
                  );
                })}
              </div>
            )}
            <p className="rag-final-safety-note">{result.safety_note || "Retrieved text is evidence for human review, not maintenance authorization."}</p>
          </section>
        )}
      </section>
    </>
  );
}

function App() {
  const [page, setPage] = useState("overview");
  const [simulatedFault, setSimulatedFault] = useState(false);
  const [backendOnline, setBackendOnline] = useState(false);
  const [databaseOnline, setDatabaseOnline] = useState(false);
  const [notificationOpen, setNotificationOpen] = useState(false);

  const assets = useApiList("/api/v1/assets");
  const workOrders = useApiList("/api/v1/work-orders");
  const spares = useApiList("/api/v1/spares");
  const rul = useRulPredictions();
  const metrics = useModelMetrics();

  useEffect(() => {
    let active = true;

    async function checkBackend() {
      try {
        const response = await fetch(`${API_BASE}/health`);

        if (!response.ok) {
          throw new Error("Health check failed");
        }

        const payload = await response.json();

        if (active) {
          setBackendOnline(payload.status === "healthy");
          setDatabaseOnline(payload.database === "connected");
        }
      } catch {
        if (active) {
          setBackendOnline(false);
          setDatabaseOnline(false);
        }
      }
    }

    checkBackend();

    const timer = window.setInterval(checkBackend, 10000);

    return () => {
      active = false;
      window.clearInterval(timer);
    };
  }, []);

  const currentPage =
    navItems.find((item) => item.id === page) || navItems[0];

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">
            <ShieldCheck size={25} />
          </div>

          <div className="brand-copy">
            <strong>AERO-GUARD</strong>
            <span>NEXUS INTELLIGENCE</span>
          </div>
        </div>

        <div className="workspace-label">WORKSPACE</div>

        <nav className="nav-list">
          {navItems.map((item) => {
            const Icon = item.icon;

            return (
              <button
                key={item.id}
                className={`nav-item ${
                  page === item.id ? "active" : ""
                }`}
                onClick={() => setPage(item.id)}
              >
                <Icon size={18} />
                <span>{item.label}</span>

                {page === item.id && (
                  <span className="nav-active-mark" />
                )}
              </button>
            );
          })}
        </nav>

        <div className="sidebar-spacer" />

        <div className="edge-card">
          <div className="edge-card-heading">
            <span className="edge-pulse" />
            SYSTEM CONNECTIVITY
          </div>

          <strong>
            {backendOnline ? "API connected" : "API unavailable"}
          </strong>

          <span>
            {databaseOnline
              ? "Database connected"
              : "Database unavailable"}
          </span>

          <div className="edge-divider" />

          <div className="edge-foot">
            <Wifi size={14} />
            Service connectivity monitor
          </div>
        </div>

        <div className="sidebar-footer">
          <div className="avatar">AG</div>

          <div className="profile-copy">
            <strong>Operations Console</strong>
            <span>Maintenance intelligence</span>
          </div>

          <CircleDot size={15} className="muted" />
        </div>
      </aside>

      <main className="main-area">
        <header className="topbar">
          <div className="breadcrumbs">
            <span>PLATFORM</span>
            <ChevronRight size={13} />
            <strong>{currentPage.label.toUpperCase()}</strong>
          </div>

          <div className="topbar-actions">
            <div
              className={`connection-pill ${
                backendOnline ? "connected" : "disconnected"
              }`}
            >
              {backendOnline ? (
                <Wifi size={15} />
              ) : (
                <WifiOff size={15} />
              )}

              {backendOnline
                ? "BACKEND CONNECTED"
                : "BACKEND OFFLINE"}
            </div>

            <button
              className="icon-button"
              title="Notifications"
              onClick={() => setNotificationOpen((value) => !value)}
            >
              <Bell size={18} />
              {simulatedFault && (
                <span className="notification-dot" />
              )}
            </button>

            {notificationOpen && (
              <div className="notification-popover">
                <strong>Notifications</strong>
                <p>
                  {simulatedFault
                    ? "A simulated anomaly is active in the Fault Lab."
                    : "No new demo events. Use Fault Lab to test an anomaly."}
                </p>
              </div>
            )}

            <div className="topbar-avatar">AG</div>
          </div>
        </header>

        <div className="content-area">
          <div className="page-title-row">
            <div>
              <div className="eyebrow">
                AIR POWER / PREDICTIVE MAINTENANCE
              </div>

              <h1>
                {page === "overview"
                  ? "Fleet readiness overview"
                  : currentPage.label}
              </h1>

              <p>
                Evidence-led maintenance intelligence for higher
                fleet availability.
              </p>
            </div>

            <div className="page-title-meta">
              <div className="date-pill">
                <CircleDot size={14} />
                SYSTEM STATUS
              </div>

            </div>
          </div>

          {page === "overview" && (
            <Overview
              simulatedFault={simulatedFault}
              onToggleFault={() =>
                setSimulatedFault((value) => !value)
              }
              backendOnline={backendOnline}
              databaseOnline={databaseOnline}
              assetsData={assets.items}
              assetsLoading={assets.loading}
              assetsError={assets.error}
            />
          )}

          {page === "diagnostics" && (
            <Diagnostics
              simulatedFault={simulatedFault}
              onToggleFault={() =>
                setSimulatedFault((value) => !value)
              }
              rulData={rul.data}
              rulLoading={rul.loading}
              rulError={rul.error}
              metricsData={metrics.data}
              metricsLoading={metrics.loading}
              metricsError={metrics.error}
            />
          )}

          {page === "maintenance" && (
            <Maintenance
              orders={workOrders.items}
              loading={workOrders.loading}
              error={workOrders.error}
            />
          )}

          {page === "inventory" && (
            <Inventory
              items={spares.items}
              loading={spares.loading}
              error={spares.error}
            />
          )}

          {page === "assurance" && <SystemAssurance />}
          {page === "copilot" && <Copilot />}

          <footer className="app-footer">
            <span>
              AERO-GUARD NEXUS
              <span className="footer-separator">/</span>
              Maintenance Intelligence
            </span>

            <span>
              <ShieldCheck size={14} />
              Evidence-led intelligence · benchmark and synthetic records are identified
            </span>
          </footer>
        </div>
      </main>
    </div>
  );
}

export default App;