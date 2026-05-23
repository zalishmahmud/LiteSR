import { useState, useEffect, useRef, useCallback } from "react";

const WS_URL = `ws://${window.location.hostname}:8765`;

const EVENT_META = {
  "sampler:epoch":              { color: "#f0a500", icon: "⟳", label: "Epoch",              layer: 0 },
  "buffer:created":             { color: "#00ffc8", icon: "◈", label: "Buffer Created",    layer: 0 },
  "buffer:best_score_updated":  { color: "#ffdd57", icon: "★", label: "Best Score ↑",       layer: 0 },
  "buffer:islands_reset":       { color: "#ff6b6b", icon: "↺", label: "Islands Reset",      layer: 0 },
  "island:created":             { color: "#74b9ff", icon: "⬡", label: "Island Created",     layer: 1 },
  "island:program_registered":  { color: "#a29bfe", icon: "+", label: "Program Registered", layer: 1 },
  "cluster:created":            { color: "#55efc4", icon: "◎", label: "Cluster Created",    layer: 2 },
  "cluster:program_added":      { color: "#81ecec", icon: "●", label: "Program Added",      layer: 2 },
  "cluster:sampled":            { color: "#fd79a8", icon: "⟳", label: "Cluster Sampled",    layer: 2 },
};

const LAYER_COLORS = ["#00ffc8", "#74b9ff", "#fd79a8"];
const LAYER_LABELS = ["Buffer", "Island", "Cluster"];

// Smart score display: uses scientific notation when abs value < 0.0001
function fmtScore(v) {
  if (v == null) return "—";
  const n = +v;
  if (!isFinite(n)) return "—";
  if (n === 0) return "0";
  if (Math.abs(n) < 0.0001 && n !== 0) return n.toExponential(3);
  if (Math.abs(n) > 99999) return n.toExponential(3);
  return n.toFixed(4);
}

function getLayer(source) {
  if (!source) return 0;
  if (source.startsWith("ExperienceBuffer.island")) return 1;
  if (source.startsWith("ExperienceBuffer")) return 0;
  if (source.includes("cluster")) return 2;
  if (source.startsWith("island")) return 1;
  return 0;
}

// ── Full snapshot builder (stores per-cluster program history) ────────────────
function buildSnapshot(events) {
  const islands = {};
  const bestScores = {};

  for (const ev of events) {
    if (ev.type === "buffer:created") {
      const n = ev.data?.numIslands ?? 0;
      for (let i = 0; i < n; i++) {
        islands[i] = islands[i] || { id: i, numPrograms: 0, clusters: {} };
        if (bestScores[i] === undefined) bestScores[i] = null;
      }
    }
    if (ev.type === "island:created") {
      const id = ev.data?.islandId ?? "?";
      islands[id] = islands[id] || { id, numPrograms: 0, clusters: {} };
    }
    if (ev.type === "island:program_registered") {
      const id  = ev.data?.islandId ?? "?";
      if (!islands[id]) islands[id] = { id, numPrograms: 0, clusters: {} };
      islands[id].numPrograms = ev.data?.totalPrograms ?? islands[id].numPrograms + 1;
      const sig = ev.data?.signature ?? "unknown";
      if (!islands[id].clusters[sig]) {
        islands[id].clusters[sig] = {
          signature: sig,
          score: ev.data?.score ?? 0,
          programCount: 0,
          programs: [],          // full history of programs added to this cluster
          sampledCount: 0,
          lastSampledAt: null,
        };
      }
      const cl = islands[id].clusters[sig];
      cl.programCount++;
      if (ev.data?.score !== undefined) cl.score = ev.data.score;
      if (ev.data?.program) {
        cl.programs.push({ code: ev.data.program, addedAt: ev.relativeMs, eventId: ev.id });
      }
    }
    if (ev.type === "cluster:sampled") {
      // find matching island+cluster by signature if possible
      // (we emit islandId in sampled events from tracker)
      const islandId = ev.data?.islandId;
      const sig = ev.data?.signature;
      if (islandId !== undefined && sig && islands[islandId]?.clusters[sig]) {
        islands[islandId].clusters[sig].sampledCount++;
        islands[islandId].clusters[sig].lastSampledAt = ev.relativeMs;
      }
    }
    if (ev.type === "buffer:best_score_updated") {
      const id = ev.data?.islandId;
      if (id !== undefined) bestScores[id] = ev.data?.newScore ?? null;
    }
    if (ev.type === "buffer:islands_reset") {
      const scores = ev.data?.scoresBeforeReset ?? [];
      scores.forEach((s, i) => { if (s !== null) bestScores[i] = s; });
    }
  }

  return {
    islands: Object.values(islands)
      .sort((a, b) => Number(a.id) - Number(b.id))
      .map((isl) => ({
        ...isl,
        clusters: Object.values(isl.clusters).sort((a, b) => b.score - a.score),
        numClusters: Object.keys(isl.clusters).length,
      })),
    bestScores,
  };
}

// ── DataTree ──────────────────────────────────────────────────────────────────
function DataTree({ data, color, depth = 0 }) {
  const [open, setOpen] = useState(depth < 2);
  if (data === null || data === undefined) return <span style={{ color: "#4a7090" }}>null</span>;
  if (typeof data === "boolean") return <span style={{ color: "#ff9f43" }}>{String(data)}</span>;
  if (typeof data === "number") return <span style={{ color: "#00ffc8" }}>{data}</span>;
  if (typeof data === "string") return <span style={{ color: "#a8e6cf" }}>"{data.length > 100 ? data.slice(0, 100) + "…" : data}"</span>;
  if (Array.isArray(data)) {
    if (data.length === 0) return <span style={{ color: "#4a7090" }}>[]</span>;
    return (
      <span>
        <span onClick={() => setOpen(o => !o)} style={{ color: "#4a7090", cursor: "pointer" }}>
          [{open ? "▾" : "▸"} {data.length}]
        </span>
        {open && data.map((item, i) => (
          <div key={i} style={{ paddingLeft: 14 }}>
            <span style={{ color: "#2a4060" }}>{i}: </span>
            <DataTree data={item} color={color} depth={depth + 1} />
          </div>
        ))}
      </span>
    );
  }
  if (typeof data === "object") {
    return (
      <span>
        {Object.keys(data).map((k) => (
          <div key={k} style={{ paddingLeft: depth > 0 ? 14 : 0 }}>
            <span style={{ color: color || "#74b9ff", opacity: 0.8 }}>{k}</span>
            <span style={{ color: "#4a7090" }}>: </span>
            <DataTree data={data[k]} color={color} depth={depth + 1} />
          </div>
        ))}
      </span>
    );
  }
  return <span style={{ color: "#c8d8e8" }}>{String(data)}</span>;
}

// ── MiniBarChart ──────────────────────────────────────────────────────────────
function MiniBarChart({ events }) {
  const bins = 24;
  const counts = Array(bins).fill(0);
  if (events.length > 1) {
    const maxT = events[events.length - 1]?.relativeMs || 1;
    events.forEach((e) => {
      const bin = Math.min(bins - 1, Math.floor((e.relativeMs / (maxT + 1)) * bins));
      counts[bin]++;
    });
  }
  const maxC = Math.max(...counts, 1);
  return (
    <div style={{ display: "flex", alignItems: "flex-end", gap: "2px", height: "52px" }}>
      {counts.map((c, i) => (
        <div key={i} style={{
          flex: 1,
          background: `rgba(0,255,200,${0.1 + 0.75 * (c / maxC)})`,
          borderRadius: "1px",
          height: `${Math.max(3, (c / maxC) * 100)}%`,
          transition: "height 0.3s",
        }} />
      ))}
    </div>
  );
}

// ── StatusBadge ───────────────────────────────────────────────────────────────
function StatusBadge({ status }) {
  const cfg = {
    connecting: { color: "#ffdd57", label: "CONNECTING" },
    open:       { color: "#00ffc8", label: "LIVE" },
    closed:     { color: "#ff6b6b", label: "DISCONNECTED" },
    error:      { color: "#ff6b6b", label: "ERROR" },
  }[status] || { color: "#888", label: status.toUpperCase() };
  return (
    <div style={{
      display: "flex", alignItems: "center", gap: 6,
      background: cfg.color + "18", border: `1px solid ${cfg.color}44`,
      borderRadius: 20, padding: "4px 12px",
    }}>
      <span style={{
        width: 7, height: 7, borderRadius: "50%", background: cfg.color, display: "inline-block",
        animation: status === "open" ? "pulse 1.2s infinite" : "none",
      }} />
      <span style={{ fontSize: 10, color: cfg.color, fontWeight: 700, letterSpacing: ".15em" }}>
        {cfg.label}
      </span>
    </div>
  );
}

// ── CodeBlock ─────────────────────────────────────────────────────────────────
// Safe tokeniser: escapes HTML first, then applies colours token-by-token
// so it never matches inside its own span tags.
function tokenizePython(raw) {
  const KEYWORDS = new Set([
    "def","return","import","from","class","if","elif","else","for","in",
    "while","and","or","not","True","False","None","lambda","with","as",
    "try","except","finally","raise","yield","pass","break","continue",
    "global","nonlocal","assert","del","is",
  ]);

  const lines = raw.split("\n");
  return lines.map((line) => {
    const parts = [];
    let i = 0;

    const esc = (s) => s.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");

    while (i < line.length) {
      // comment — rest of line
      if (line[i] === "#") {
        parts.push(`<span style="color:#3a6080">${esc(line.slice(i))}</span>`);
        i = line.length;
        continue;
      }
      // triple-quoted string (rare in one line but handle it)
      if (line.slice(i,i+3) === '"""' || line.slice(i,i+3) === "'''") {
        const q = line.slice(i,i+3);
        let j = i + 3;
        while (j < line.length && line.slice(j,j+3) !== q) j++;
        j = Math.min(j+3, line.length);
        parts.push(`<span style="color:#a8e6cf">${esc(line.slice(i,j))}</span>`);
        i = j; continue;
      }
      // single/double quoted string
      if (line[i] === '"' || line[i] === "'") {
        const q = line[i]; let j = i+1;
        while (j < line.length && line[j] !== q) { if (line[j] === "\\") j++; j++; }
        j = Math.min(j+1, line.length);
        parts.push(`<span style="color:#a8e6cf">${esc(line.slice(i,j))}</span>`);
        i = j; continue;
      }
      // identifier or keyword
      if (/[a-zA-Z_]/.test(line[i])) {
        let j = i;
        while (j < line.length && /[\w]/.test(line[j])) j++;
        const word = line.slice(i,j);
        if (KEYWORDS.has(word)) {
          parts.push(`<span style="color:#74b9ff">${esc(word)}</span>`);
        } else if (/^[A-Z]/.test(word)) {
          parts.push(`<span style="color:#ffdd57">${esc(word)}</span>`);
        } else {
          parts.push(esc(word));
        }
        i = j; continue;
      }
      // number
      if (/[0-9]/.test(line[i]) || (line[i] === "." && /[0-9]/.test(line[i+1]||""))) {
        let j = i;
        while (j < line.length && /[0-9._eEjJ+\-]/.test(line[j])) j++;
        parts.push(`<span style="color:#00ffc8">${esc(line.slice(i,j))}</span>`);
        i = j; continue;
      }
      // operators / punctuation
      if ("+-*/%=<>!&|^~@,.:;()[]{}".includes(line[i])) {
        parts.push(`<span style="color:#fd79a8">${esc(line[i])}</span>`);
        i++; continue;
      }
      // anything else (whitespace etc.)
      parts.push(esc(line[i]));
      i++;
    }
    return parts.join("");
  }).join("\n");
}

function CodeBlock({ code }) {
  const [copied, setCopied] = useState(false);
  const copy = () => {
    navigator.clipboard.writeText(code).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    });
  };
  const highlighted = tokenizePython(code);
  return (
    <div style={{ position: "relative" }}>
      <button onClick={copy} style={{
        position: "absolute", top: 8, right: 8,
        background: copied ? "#00ffc822" : "#0a1828",
        border: `1px solid ${copied ? "#00ffc8" : "#1a3050"}`,
        borderRadius: 4, color: copied ? "#00ffc8" : "#3a6080",
        padding: "3px 9px", fontSize: 10, cursor: "pointer", zIndex: 1,
      }}>{copied ? "✓ copied" : "copy"}</button>
      <pre style={{
        margin: 0, padding: "12px 14px", paddingRight: 60,
        background: "#040c14", borderRadius: 6,
        border: "1px solid #0e2030",
        fontSize: 11, lineHeight: 1.7, overflowX: "auto",
        color: "#c8d8e8", fontFamily: "'JetBrains Mono', monospace",
        whiteSpace: "pre",
      }} dangerouslySetInnerHTML={{ __html: highlighted }} />
    </div>
  );
}

// ── ClusterDetailPanel ────────────────────────────────────────────────────────
function ClusterDetailPanel({ cluster, islandId, onClose }) {
  const [selectedProgIdx, setSelectedProgIdx] = useState(cluster.programs.length - 1);
  const prog = cluster.programs[selectedProgIdx];
  const maxScore = cluster.score;

  return (
    <div className="fade-in" style={{
      position: "absolute", inset: 0, zIndex: 10,
      background: "#080c14",
      display: "flex", flexDirection: "column",
    }}>
      {/* Header */}
      <div style={{
        padding: "12px 18px", borderBottom: "1px solid #1a2a3a",
        display: "flex", alignItems: "center", gap: 10,
        background: "linear-gradient(90deg,#080c14,#0c1828)",
      }}>
        <button onClick={onClose} style={{
          background: "#0a1828", border: "1px solid #1a3050", borderRadius: 5,
          color: "#74b9ff", padding: "5px 10px", fontSize: 11, cursor: "pointer",
        }}>← Back</button>
        <div>
          <div style={{ fontSize: 9, color: "#3a5070", letterSpacing: ".2em" }}>
            ISLAND #{islandId} · CLUSTER
          </div>
          <div style={{ fontSize: 12, color: "#55efc4", fontWeight: 600 }}>
            [{cluster.signature.length > 50
              ? cluster.signature.slice(0, 50) + "…"
              : cluster.signature}]
          </div>
        </div>
        <div style={{ flex: 1 }} />
        {[
          { l: "SCORE",    v: fmtScore(cluster.score),      c: "#ffdd57" },
          { l: "PROGRAMS", v: cluster.programCount,          c: "#a29bfe" },
          { l: "SAMPLED",  v: cluster.sampledCount,          c: "#fd79a8" },
        ].map(s => (
          <div key={s.l} style={{ textAlign: "right", marginLeft: 16 }}>
            <div style={{ fontSize: 8, color: "#3a5070", letterSpacing: ".2em" }}>{s.l}</div>
            <div style={{ fontSize: 18, fontWeight: 700, color: s.c }}>{s.v}</div>
          </div>
        ))}
        <button onClick={onClose} style={{ background: "none", border: "none", color: "#3a5070", cursor: "pointer", fontSize: 18, marginLeft: 8 }}>×</button>
      </div>

      <div style={{ display: "flex", flex: 1, overflow: "hidden" }}>
        {/* Left: program list */}
        <div style={{ width: 220, borderRight: "1px solid #1a2a3a", display: "flex", flexDirection: "column" }}>
          <div style={{ padding: "10px 14px", borderBottom: "1px solid #1a2a3a", fontSize: 9, color: "#3a5070", letterSpacing: ".2em" }}>
            PROGRAM HISTORY ({cluster.programs.length})
          </div>
          <div style={{ flex: 1, overflowY: "auto" }}>
            {cluster.programs.length === 0 && (
              <div style={{ padding: 16, color: "#1e3040", fontSize: 11 }}>No programs captured yet.</div>
            )}
            {cluster.programs.map((p, i) => {
              const isSel = selectedProgIdx === i;
              const isLatest = i === cluster.programs.length - 1;
              return (
                <div
                  key={i}
                  className="ev-row"
                  onClick={() => setSelectedProgIdx(i)}
                  style={{
                    padding: "9px 14px",
                    borderLeft: `2px solid ${isSel ? "#55efc4" : "transparent"}`,
                    borderBottom: "1px solid #0d1520",
                    background: isSel ? "#0a1e2a" : "transparent",
                    cursor: "pointer",
                  }}
                >
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 3 }}>
                    <span style={{ fontSize: 11, color: isSel ? "#55efc4" : "#5a8098", fontWeight: isSel ? 600 : 400 }}>
                      v{i}
                    </span>
                    {isLatest && (
                      <span style={{
                        fontSize: 8, color: "#00ffc8", background: "#00ffc818",
                        border: "1px solid #00ffc844", borderRadius: 3, padding: "1px 5px",
                      }}>LATEST</span>
                    )}
                  </div>
                  <div style={{ fontSize: 9, color: "#2a4860" }}>+{p.addedAt}ms</div>
                  <div style={{ fontSize: 10, color: "#2a5070", marginTop: 3, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                    {p.code.slice(0, 40)}{p.code.length > 40 ? "…" : ""}
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Right: program code + score chart */}
        <div style={{ flex: 1, display: "flex", flexDirection: "column", overflowY: "auto", padding: "16px 20px", gap: 16 }}>
          {prog ? (
            <>
              <div>
                <div style={{ fontSize: 9, color: "#3a5070", letterSpacing: ".2em", marginBottom: 8 }}>
                  PROGRAM v{selectedProgIdx} · added at +{prog.addedAt}ms
                </div>
                <CodeBlock code={prog.code} />
              </div>

              {/* Score bar for this cluster vs max */}
              <div style={{ background: "#0c1520", borderRadius: 8, padding: 14, border: "1px solid #1a2a3a" }}>
                <div style={{ fontSize: 9, color: "#3a5070", letterSpacing: ".2em", marginBottom: 10 }}>
                  CLUSTER SCORE
                </div>
                <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
                  <div style={{ flex: 1, height: 8, background: "#1a2a3a", borderRadius: 4 }}>
                    <div style={{
                      height: "100%", borderRadius: 4,
                      width: `${Math.max(2, Math.min(100, (cluster.score / (Math.abs(cluster.score) + 1)) * 100 + 50))}%`,
                      background: "linear-gradient(90deg,#55efc4,#00ffc8)",
                      transition: "width .4s",
                    }} />
                  </div>
                  <span style={{ fontSize: 14, fontWeight: 700, color: "#ffdd57", minWidth: 60, textAlign: "right" }}>
                    {fmtScore(cluster.score)}
                  </span>
                </div>

                {/* Signature breakdown */}
                <div style={{ marginTop: 12 }}>
                  <div style={{ fontSize: 9, color: "#3a5070", letterSpacing: ".2em", marginBottom: 6 }}>
                    SIGNATURE (test scores)
                  </div>
                  <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
                    {cluster.signature.split(",").map((v, i) => (
                      <div key={i} style={{
                        background: "#0a1828", borderRadius: 4, padding: "4px 10px",
                        fontSize: 11, color: "#74b9ff",
                        border: "1px solid #1a3050",
                      }}>
                        t{i}: <span style={{ color: "#00ffc8" }}>{parseFloat(v).toFixed(4)}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              {/* Program evolution timeline */}
              {cluster.programs.length > 1 && (
                <div style={{ background: "#0c1520", borderRadius: 8, padding: 14, border: "1px solid #1a2a3a" }}>
                  <div style={{ fontSize: 9, color: "#3a5070", letterSpacing: ".2em", marginBottom: 10 }}>
                    PROGRAM EVOLUTION TIMELINE
                  </div>
                  <div style={{ display: "flex", alignItems: "center", gap: 0, position: "relative" }}>
                    <div style={{
                      position: "absolute", top: "50%", left: 0, right: 0,
                      height: 1, background: "#1a2a3a", transform: "translateY(-50%)",
                    }} />
                    {cluster.programs.map((p, i) => {
                      const isSel = selectedProgIdx === i;
                      const isLatest = i === cluster.programs.length - 1;
                      return (
                        <div key={i} onClick={() => setSelectedProgIdx(i)} style={{
                          display: "flex", flexDirection: "column", alignItems: "center",
                          flex: 1, cursor: "pointer", position: "relative", zIndex: 1,
                        }}>
                          <div style={{
                            width: isSel ? 14 : 8,
                            height: isSel ? 14 : 8,
                            borderRadius: "50%",
                            background: isSel ? "#55efc4" : isLatest ? "#00ffc8" : "#1a3a50",
                            border: `2px solid ${isSel ? "#55efc4" : isLatest ? "#00ffc8" : "#1a3a50"}`,
                            transition: "all .2s",
                            boxShadow: isSel ? "0 0 8px #55efc4" : "none",
                          }} />
                          <div style={{ fontSize: 8, color: isSel ? "#55efc4" : "#2a4060", marginTop: 4 }}>v{i}</div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}
            </>
          ) : (
            <div style={{ color: "#1e3040", fontSize: 12 }}>Select a program version from the left.</div>
          )}
        </div>
      </div>
    </div>
  );
}

// ── IslandDetailPanel ─────────────────────────────────────────────────────────
function IslandDetailPanel({ island, bestScore, onClose, onSelectCluster }) {
  const sorted = [...island.clusters].sort((a, b) => b.score - a.score);
  const maxScore = sorted.length ? sorted[0].score : 1;

  return (
    <div className="fade-in" style={{
      position: "absolute", inset: 0, zIndex: 5,
      background: "#080c14",
      display: "flex", flexDirection: "column",
    }}>
      {/* Header */}
      <div style={{
        padding: "12px 18px", borderBottom: "1px solid #1a2a3a",
        display: "flex", alignItems: "center", gap: 12,
        background: "linear-gradient(90deg,#080c14,#0c1828)",
      }}>
        <button onClick={onClose} style={{
          background: "#0a1828", border: "1px solid #1a3050", borderRadius: 5,
          color: "#74b9ff", padding: "5px 10px", fontSize: 11, cursor: "pointer",
        }}>← Islands</button>
        <div>
          <div style={{ fontSize: 9, color: "#3a5070", letterSpacing: ".2em" }}>ISLAND</div>
          <div style={{ fontFamily: "'Syne',sans-serif", fontSize: 26, fontWeight: 800, color: "#74b9ff", lineHeight: 1 }}>
            #{island.id}
          </div>
        </div>
        <div style={{ flex: 1 }} />
        {[
          { l: "BEST SCORE", v: fmtScore(bestScore), c: "#ffdd57" },
          { l: "PROGRAMS",   v: island.numPrograms,                                 c: "#a29bfe" },
          { l: "CLUSTERS",   v: island.numClusters,                                 c: "#55efc4" },
        ].map(s => (
          <div key={s.l} style={{ textAlign: "right", marginLeft: 16 }}>
            <div style={{ fontSize: 8, color: "#3a5070", letterSpacing: ".2em" }}>{s.l}</div>
            <div style={{ fontSize: 18, fontWeight: 700, color: s.c }}>{s.v}</div>
          </div>
        ))}
        <button onClick={onClose} style={{ background: "none", border: "none", color: "#3a5070", cursor: "pointer", fontSize: 18, marginLeft: 8 }}>×</button>
      </div>

      <div style={{ flex: 1, overflowY: "auto", padding: "16px 20px" }}>
        <div style={{ fontSize: 9, color: "#3a5070", letterSpacing: ".2em", marginBottom: 12 }}>
          CLUSTERS — click to drill in
        </div>

        {sorted.length === 0 && (
          <div style={{ color: "#1e3040", fontSize: 12 }}>No clusters yet.</div>
        )}

        <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
          {sorted.map((cl, rank) => {
            const pct = maxScore !== 0 ? Math.max(4, (cl.score / Math.abs(maxScore)) * 80 + 10) : 10;
            const isTop = rank === 0;
            return (
              <div
                key={cl.signature}
                className="ev-row"
                onClick={() => onSelectCluster(cl)}
                style={{
                  background: "#0c1520",
                  border: `1px solid ${isTop ? "#ffdd5740" : "#1a2a3a"}`,
                  borderRadius: 8, padding: "12px 16px",
                  cursor: "pointer", transition: "all .15s",
                }}
              >
                <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 8 }}>
                  {/* Rank badge */}
                  <div style={{
                    width: 24, height: 24, borderRadius: 6,
                    background: isTop ? "#ffdd5722" : "#0a1828",
                    border: `1px solid ${isTop ? "#ffdd57" : "#1a2a3a"}`,
                    display: "flex", alignItems: "center", justifyContent: "center",
                    fontSize: 10, fontWeight: 700,
                    color: isTop ? "#ffdd57" : "#3a5070",
                  }}>#{rank + 1}</div>
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <div style={{ fontSize: 10, color: "#3a6080", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                      [{cl.signature.length > 55 ? cl.signature.slice(0, 55) + "…" : cl.signature}]
                    </div>
                  </div>
                  <div style={{ fontSize: 14, fontWeight: 700, color: isTop ? "#ffdd57" : "#55efc4" }}>
                    {fmtScore(cl.score)}
                  </div>
                  <div style={{ fontSize: 11, color: "#4a7090" }}>→</div>
                </div>

                {/* Score bar */}
                <div style={{ height: 4, background: "#1a2a3a", borderRadius: 2, marginBottom: 8 }}>
                  <div style={{
                    height: "100%", borderRadius: 2,
                    width: `${pct}%`,
                    background: isTop
                      ? "linear-gradient(90deg,#ffdd57,#ffa502)"
                      : "linear-gradient(90deg,#55efc4,#00ffc8)",
                    transition: "width .4s",
                  }} />
                </div>

                <div style={{ display: "flex", gap: 8 }}>
                  <span style={{ fontSize: 10, color: "#3a5870", background: "#0a1828", borderRadius: 3, padding: "2px 8px" }}>
                    {cl.programCount} programs
                  </span>
                  <span style={{ fontSize: 10, color: "#3a5870", background: "#0a1828", borderRadius: 3, padding: "2px 8px" }}>
                    sampled {cl.sampledCount}×
                  </span>
                  {cl.lastSampledAt && (
                    <span style={{ fontSize: 10, color: "#2a4060", background: "#0a1828", borderRadius: 3, padding: "2px 8px" }}>
                      last at +{cl.lastSampledAt}ms
                    </span>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}

// ── IslandGrid ────────────────────────────────────────────────────────────────
function IslandGrid({ snapshot, onSelectIsland }) {
  return (
    <div style={{ padding: "14px 18px", flex: 1, overflowY: "auto" }}>
      <div style={{ fontSize: "9px", color: "#1e3040", letterSpacing: ".25em", marginBottom: 12 }}>
        ◈ LIVE SNAPSHOT — click an island to explore
      </div>
      {snapshot.islands.length === 0 ? (
        <div style={{ color: "#1e3040", fontSize: 12 }}>
          Islands will appear once your script connects and starts registering programs.
        </div>
      ) : (
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill,minmax(240px,1fr))", gap: 10 }}>
          {snapshot.islands.map((isl) => {
            const best = snapshot.bestScores[isl.id];
            const bestStr = fmtScore(best);
            return (
              <div
                key={isl.id}
                className="ev-row"
                onClick={() => onSelectIsland(isl)}
                style={{
                  background: "#0c1520",
                  border: "1px solid #1a2a3a",
                  borderRadius: 8, padding: 14,
                  cursor: "pointer",
                }}
              >
                <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 10 }}>
                  <div>
                    <div style={{ fontSize: "9px", color: "#3a5070", letterSpacing: ".2em" }}>ISLAND</div>
                    <div style={{ fontFamily: "'Syne',sans-serif", fontSize: 22, fontWeight: 800, color: "#74b9ff" }}>
                      #{isl.id}
                    </div>
                  </div>
                  <div style={{ textAlign: "right" }}>
                    <div style={{ fontSize: "9px", color: "#3a5070" }}>BEST SCORE</div>
                    <div style={{ fontSize: 18, fontWeight: 700, color: "#ffdd57" }}>{bestStr}</div>
                  </div>
                </div>
                <div style={{ display: "flex", gap: 8, marginBottom: 10 }}>
                  <span style={{ background: "#0a1828", borderRadius: 4, padding: "3px 9px", fontSize: 10, color: "#a29bfe" }}>
                    {isl.numPrograms} progs
                  </span>
                  <span style={{ background: "#0a1828", borderRadius: 4, padding: "3px 9px", fontSize: 10, color: "#55efc4" }}>
                    {isl.numClusters} clusters
                  </span>
                </div>
                <div style={{ display: "flex", flexDirection: "column", gap: 5 }}>
                  {isl.clusters.length === 0 && (
                    <div style={{ color: "#1e3040", fontSize: 10 }}>no clusters yet</div>
                  )}
                  {isl.clusters.slice(0, 5).map((cl) => {
                    const maxS = Math.max(...isl.clusters.map(c => c.score), 0.01);
                    const pct = Math.max(4, (cl.score / maxS) * 100);
                    return (
                      <div key={cl.signature} style={{ fontSize: 10 }}>
                        <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 2, color: "#2a4860" }}>
                          <span style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", maxWidth: 140 }}>
                            [{cl.signature.slice(0, 18)}{cl.signature.length > 18 ? "…" : ""}]
                          </span>
                          <span style={{ color: "#55efc4" }}>{cl.programCount}p · {fmtScore(cl.score)}</span>
                        </div>
                        <div style={{ height: 3, background: "#1a2a3a", borderRadius: 2 }}>
                          <div style={{ height: "100%", width: `${pct}%`, background: "linear-gradient(90deg,#55efc4,#00ffc8)", borderRadius: 2, transition: "width .4s" }} />
                        </div>
                      </div>
                    );
                  })}
                  {isl.clusters.length > 5 && (
                    <div style={{ color: "#2a4060", fontSize: 10 }}>+{isl.clusters.length - 5} more — click to see all</div>
                  )}
                </div>
                <div style={{ marginTop: 10, fontSize: 10, color: "#1e3040", textAlign: "right" }}>
                  tap to explore →
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

// ── Main App ──────────────────────────────────────────────────────────────────
export default function App() {
  const [events, setEvents]               = useState([]);
  const [wsStatus, setWsStatus]           = useState("closed");
  const [selectedEvent, setSelectedEvent] = useState(null);
  const [filter, setFilter]               = useState("all");
  const [autoScroll, setAutoScroll]       = useState(true);
  const [wsUrl, setWsUrl]                 = useState(WS_URL);
  const [editingUrl, setEditingUrl]       = useState(false);

  // drill-down state
  const [selectedIsland,  setSelectedIsland]  = useState(null);
  const [selectedCluster, setSelectedCluster] = useState(null);

  const wsRef          = useRef(null);
  const logRef         = useRef(null);
  const reconnectTimer = useRef(null);

  const snapshot   = buildSnapshot(events);
  const typeCounts = {};
  events.forEach((e) => { typeCounts[e.type] = (typeCounts[e.type] || 0) + 1; });

  // Epoch: pick the latest sampler:epoch event
  const currentEpoch = (() => {
    for (let i = events.length - 1; i >= 0; i--) {
      if (events[i].type === "sampler:epoch") return events[i].data?.epoch ?? 0;
    }
    return 0;
  })();

  // Global best score: max across all islands
  const globalBestScore = (() => {
    const scores = Object.values(snapshot.bestScores).filter(s => s !== null && isFinite(+s));
    return scores.length ? Math.max(...scores) : null;
  })();

  // keep selectedIsland in sync with live snapshot
  const liveIsland = selectedIsland
    ? snapshot.islands.find(i => String(i.id) === String(selectedIsland.id)) || selectedIsland
    : null;
  const liveCluster = selectedCluster && liveIsland
    ? liveIsland.clusters.find(c => c.signature === selectedCluster.signature) || selectedCluster
    : selectedCluster;

  // events are newest-first so no auto-scroll needed

  const connect = useCallback((url) => {
    if (wsRef.current) { wsRef.current.onclose = null; wsRef.current.close(); }
    setWsStatus("connecting");
    const ws = new WebSocket(url);
    wsRef.current = ws;
    ws.onopen    = () => setWsStatus("open");
    ws.onmessage = (msg) => {
      try {
        const data = JSON.parse(msg.data);
        if (data.type === "_backlog") setEvents(data.events);
        else setEvents((prev) => [...prev, data]);
      } catch (_) {}
    };
    ws.onclose = () => {
      setWsStatus("closed");
      reconnectTimer.current = setTimeout(() => connect(url), 3000);
    };
    ws.onerror = () => setWsStatus("error");
  }, []);

  useEffect(() => {
    connect(wsUrl);
    return () => {
      clearTimeout(reconnectTimer.current);
      if (wsRef.current) wsRef.current.onclose = null;
      wsRef.current?.close();
    };
  }, []);

  const filteredEvents = (filter === "all"
    ? events
    : events.filter((e) => e.type?.startsWith(filter))
  ).slice().reverse();  // newest first

  return (
    <div style={{ background: "#080c14", height: "100vh", fontFamily: "'JetBrains Mono','Fira Code',monospace", color: "#c8d8e8", display: "flex", flexDirection: "column", overflow: "hidden" }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;600;700&family=Syne:wght@700;800&display=swap');
        * { box-sizing: border-box; }
        ::-webkit-scrollbar { width: 5px; }
        ::-webkit-scrollbar-track { background: #0d1520; }
        ::-webkit-scrollbar-thumb { background: #1e3050; border-radius: 3px; }
        @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.35} }
        @keyframes fadeSlide { from{opacity:0;transform:translateY(5px)} to{opacity:1;transform:translateY(0)} }
        @keyframes epochPop  { 0%{transform:scale(1.35);color:#fff} 100%{transform:scale(1);color:#f0a500} }
        @keyframes scorePop  { 0%{transform:scale(1.3);color:#fff}  100%{transform:scale(1);color:#ffdd57} }
        .ev-row { transition: background .12s, border-color .12s !important; cursor: pointer; }
        .ev-row:hover { background: #0d1e30 !important; border-color: #2a4060 !important; }
        .ev-row.sel { background: #0f2035 !important; }
        .pill { transition: all .15s; cursor: pointer; }
        .pill:hover { filter: brightness(1.25); }
        .btn { transition: all .15s; cursor: pointer; }
        .btn:hover { filter: brightness(1.3); transform: translateY(-1px); }
        .btn:active { transform: translateY(0); }
        .fade-in { animation: fadeSlide .18s ease; }
      `}</style>

      {/* ── HEADER ── */}
      <div style={{ borderBottom: "1px solid #1a2a3a", padding: "12px 20px", display: "flex", alignItems: "center", gap: 16, background: "linear-gradient(90deg,#080c14,#0c1624)", flexWrap: "wrap" }}>
        <div>
          <div style={{ fontSize: 9, color: "#3a5070", letterSpacing: ".3em", marginBottom: 2 }}>LiteSR · EVOLUTIONARY ALGORITHM</div>
          <div style={{ fontFamily: "'Syne',sans-serif", fontSize: 17, fontWeight: 800, color: "#e8f4ff", letterSpacing: ".04em" }}>
            Experience Buffer <span style={{ color: "#00ffc8" }}>Live Tracker</span>
          </div>
        </div>

        <StatusBadge status={wsStatus} />

        {/* Breadcrumb for drill-down */}
        {(selectedIsland || selectedCluster) && (
          <div style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 11, color: "#3a5070" }}>
            <span style={{ cursor: "pointer", color: "#4a8098" }} onClick={() => { setSelectedIsland(null); setSelectedCluster(null); }}>Islands</span>
            {selectedIsland && <>
              <span>›</span>
              <span style={{ cursor: "pointer", color: selectedCluster ? "#4a8098" : "#74b9ff" }}
                onClick={() => setSelectedCluster(null)}>
                Island #{selectedIsland.id}
              </span>
            </>}
            {selectedCluster && <>
              <span>›</span>
              <span style={{ color: "#55efc4" }}>
                [{selectedCluster.signature.slice(0, 20)}{selectedCluster.signature.length > 20 ? "…" : ""}]
              </span>
            </>}
          </div>
        )}

        {editingUrl ? (
          <input autoFocus defaultValue={wsUrl}
            onBlur={e => { setWsUrl(e.target.value); setEditingUrl(false); }}
            onKeyDown={e => { if (e.key === "Enter") { setWsUrl(e.target.value); setEditingUrl(false); connect(e.target.value); } }}
            style={{ background: "#0d1828", border: "1px solid #2a4050", borderRadius: 4, color: "#74b9ff", padding: "4px 8px", fontSize: 11, width: 200 }}
          />
        ) : (
          <span onClick={() => setEditingUrl(true)} style={{ fontSize: 10, color: "#2a5070", cursor: "pointer", borderBottom: "1px dashed #2a5070" }}>{wsUrl}</span>
        )}

        <div style={{ flex: 1 }} />

        {/* Epoch counter — prominent pill */}
        <div style={{
          display: "flex", flexDirection: "column", alignItems: "center",
          background: "#f0a50014", border: "1px solid #f0a50044",
          borderRadius: 8, padding: "6px 16px", marginLeft: 8,
        }}>
          <div style={{ fontSize: 8, color: "#f0a500aa", letterSpacing: ".2em", marginBottom: 2 }}>EPOCH</div>
          <div style={{
            fontFamily: "'Syne',sans-serif", fontSize: 26, fontWeight: 800,
            color: "#f0a500", lineHeight: 1,
            animation: currentEpoch > 0 ? "epochPop .3s ease" : "none",
            key: currentEpoch,
          }}>{currentEpoch}</div>
        </div>

        {/* Global best score pill */}
        <div style={{
          display: "flex", flexDirection: "column", alignItems: "center",
          background: "#ffdd5714", border: "1px solid #ffdd5744",
          borderRadius: 8, padding: "6px 16px", marginLeft: 4,
        }}>
          <div style={{ fontSize: 8, color: "#ffdd57aa", letterSpacing: ".2em", marginBottom: 2 }}>BEST SCORE</div>
          <div style={{
            fontFamily: "'Syne',sans-serif", fontSize: 26, fontWeight: 800,
            color: "#ffdd57", lineHeight: 1,
            animation: globalBestScore !== null ? "scorePop .3s ease" : "none",
          }}>{fmtScore(globalBestScore)}</div>
        </div>

        {[
          { l: "SAMPLES",  v: typeCounts["sampler:epoch"] ? (typeCounts["sampler:epoch"] * 4) : 0, c: "#f0a500" },
          { l: "ISLANDS",  v: snapshot.islands.length,                                              c: "#74b9ff" },
          { l: "CLUSTERS", v: snapshot.islands.reduce((a, i) => a + i.numClusters, 0),              c: "#55efc4" },
          { l: "PROGRAMS", v: snapshot.islands.reduce((a, i) => a + i.numPrograms, 0),              c: "#ffdd57" },
        ].map((s) => (
          <div key={s.l} style={{ textAlign: "right", marginLeft: 8 }}>
            <div style={{ fontSize: 8, color: "#3a5070", letterSpacing: ".2em" }}>{s.l}</div>
            <div style={{ fontSize: 18, fontWeight: 700, color: s.c }}>{s.v}</div>
          </div>
        ))}

        <div style={{ display: "flex", gap: 6, marginLeft: 10 }}>
          <button className="btn" onClick={() => connect(wsUrl)} style={{ background: "#0a1828", border: "1px solid #1e3a50", borderRadius: 5, color: "#74b9ff", padding: "7px 13px", fontSize: 11 }}>RECONNECT</button>
          <button className="btn" onClick={() => { setEvents([]); setSelectedEvent(null); setSelectedIsland(null); setSelectedCluster(null); }} style={{ background: "#140a0a", border: "1px solid #3a1a1a", borderRadius: 5, color: "#ff6b6b", padding: "7px 13px", fontSize: 11 }}>CLEAR</button>
        </div>
      </div>

      {/* ── BODY ── */}
      <div style={{ display: "flex", flex: 1, overflow: "hidden", minHeight: 0 }}>

        {/* LEFT — event log (fixed panel, independent scroll) */}
        <div style={{ width: 340, minWidth: 260, borderRight: "1px solid #1a2a3a", display: "flex", flexDirection: "column", height: "100%", overflow: "hidden" }}>
          <div style={{ padding: "8px 12px", borderBottom: "1px solid #1a2a3a", display: "flex", gap: 5, flexWrap: "wrap", alignItems: "center", flexShrink: 0 }}>
            {[["all","#c8d8e8"],["sampler","#f0a500"],["buffer","#00ffc8"],["island","#74b9ff"],["cluster","#fd79a8"]].map(([f,c]) => (
              <button key={f} className="pill" onClick={() => setFilter(f)} style={{
                background: filter===f ? c+"18" : "transparent",
                border: `1px solid ${filter===f ? c : "#1a2a3a"}`,
                borderRadius: 4, color: filter===f ? c : "#3a5070",
                padding: "3px 10px", fontSize: 10, letterSpacing: ".1em", textTransform: "uppercase",
              }}>{f}</button>
            ))}
            <div style={{ flex: 1 }} />
            <span style={{ fontSize: 9, color: "#2a4060", letterSpacing: ".1em" }}>NEWEST FIRST</span>
          </div>
          <div ref={logRef} style={{ flex: 1, overflowY: "auto", minHeight: 0 }}>
            {filteredEvents.length === 0 && (
              <div style={{ padding: "40px 20px", textAlign: "center", color: "#1e3040", fontSize: 12 }}>
                {wsStatus === "open" ? "Waiting for events…" : "Not connected. Run your script."}
              </div>
            )}
            {filteredEvents.map((ev) => {
              const meta = EVENT_META[ev.type] || { color: "#888", icon: "?", label: ev.type };
              const layer = getLayer(ev.source);
              const sel = selectedEvent?.id === ev.id;
              return (
                <div key={ev.id} className={`ev-row ${sel ? "sel" : ""}`}
                  onClick={() => setSelectedEvent(sel ? null : ev)}
                  style={{ padding: `6px 12px 6px ${12+layer*10}px`, borderLeft: `2px solid ${sel?"#00ffc8":"transparent"}`, borderBottom: "1px solid #0d1520", display: "flex", alignItems: "center", gap: 8, background: sel?"#0f2035":"transparent" }}>
                  <span style={{ color: meta.color, fontSize: 13, minWidth: 14 }}>{meta.icon}</span>
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
                      <span style={{ fontSize: 11, color: meta.color, fontWeight: 600 }}>{meta.label}</span>
                      <span style={{ fontSize: 9, color: "#1e3040" }}>+{ev.relativeMs}ms</span>
                    </div>
                    <div style={{ fontSize: 10, color: "#2a4860", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{ev.source}</div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* CENTER — drill-down area */}
        <div style={{ flex: 1, display: "flex", flexDirection: "column", overflow: "hidden", position: "relative" }}>

          {/* Event detail strip */}
          <div style={{ borderBottom: "1px solid #1a2a3a", background: "#080f18", minHeight: selectedEvent ? 200 : 42, padding: "10px 18px", transition: "min-height .2s", flexShrink: 0 }}>
            {!selectedEvent ? (
              <div style={{ color: "#1e3040", fontSize: 11 }}>← Click any event to inspect payload</div>
            ) : (() => {
              const meta = EVENT_META[selectedEvent.type] || { color: "#888", icon: "?", label: selectedEvent.type };
              return (
                <div className="fade-in">
                  <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 8 }}>
                    <span style={{ color: meta.color, fontSize: 16 }}>{meta.icon}</span>
                    <div>
                      <div style={{ color: meta.color, fontWeight: 700, fontSize: 12 }}>{meta.label}</div>
                      <div style={{ color: "#3a5870", fontSize: 10 }}>{selectedEvent.source} · #{selectedEvent.id} · +{selectedEvent.relativeMs}ms</div>
                    </div>
                    <div style={{ flex: 1 }} />
                    <button onClick={() => setSelectedEvent(null)} style={{ background: "none", border: "none", color: "#3a5070", cursor: "pointer", fontSize: 18 }}>×</button>
                  </div>
                  <div style={{ background: "#050d18", borderRadius: 6, padding: "8px 14px", border: `1px solid ${meta.color}22`, fontSize: 11, lineHeight: 1.7, overflowX: "auto", maxHeight: 130, overflowY: "auto" }}>
                    <DataTree data={selectedEvent.data} color={meta.color} depth={0} />
                  </div>
                </div>
              );
            })()}
          </div>

          {/* Drill-down layers stacked absolutely */}
          <div style={{ flex: 1, overflow: "hidden", position: "relative" }}>
            {/* Layer 0: island grid (always rendered beneath) */}
            <div style={{ position: "absolute", inset: 0, overflowY: "auto", display: "flex", flexDirection: "column" }}>
              <IslandGrid snapshot={snapshot} onSelectIsland={(isl) => { setSelectedIsland(isl); setSelectedCluster(null); }} />
            </div>

            {/* Layer 1: island detail */}
            {liveIsland && !selectedCluster && (
              <IslandDetailPanel
                island={liveIsland}
                bestScore={snapshot.bestScores[liveIsland.id]}
                onClose={() => setSelectedIsland(null)}
                onSelectCluster={(cl) => setSelectedCluster(cl)}
              />
            )}

            {/* Layer 2: cluster detail */}
            {liveIsland && liveCluster && (
              <ClusterDetailPanel
                cluster={liveCluster}
                islandId={liveIsland.id}
                onClose={() => setSelectedCluster(null)}
              />
            )}
          </div>
        </div>

        {/* RIGHT — legend */}
        <div style={{ width: 180, minWidth: 140, borderLeft: "1px solid #1a2a3a", padding: "14px 12px", display: "flex", flexDirection: "column", gap: 16, overflowY: "auto" }}>
          <div>
            <div style={{ fontSize: 9, color: "#1e3040", letterSpacing: ".25em", marginBottom: 8 }}>HIERARCHY</div>
            {LAYER_LABELS.map((l, i) => (
              <div key={l} style={{ display: "flex", alignItems: "center", gap: 7, marginBottom: 7 }}>
                <div style={{ width: 9, height: 9, borderRadius: 2, background: LAYER_COLORS[i] }} />
                <span style={{ fontSize: 11, color: "#6a90b0" }}>{l}</span>
              </div>
            ))}
          </div>
          <div>
            <div style={{ fontSize: 9, color: "#1e3040", letterSpacing: ".25em", marginBottom: 8 }}>EVENT TYPES</div>
            {Object.entries(EVENT_META).map(([type, meta]) => (
              <div key={type} style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 6, opacity: filter !== "all" && !type.startsWith(filter) ? 0.2 : 1 }}>
                <span style={{ color: meta.color, fontSize: 12, minWidth: 14 }}>{meta.icon}</span>
                <div style={{ flex: 1, minWidth: 0 }}>
                  <span style={{ fontSize: 10, color: "#5a8098" }}>{meta.label}</span>
                  {typeCounts[type] ? <span style={{ float: "right", fontSize: 10, color: meta.color, fontWeight: 600 }}>{typeCounts[type]}</span> : null}
                </div>
              </div>
            ))}
          </div>
          <div style={{ flex: 1 }}>
            <div style={{ fontSize: 9, color: "#1e3040", letterSpacing: ".25em", marginBottom: 8 }}>RATE OVER TIME</div>
            <MiniBarChart events={events} />
          </div>
          <div style={{ background: "#0a1018", borderRadius: 6, padding: 10, border: "1px solid #1a2a3a", fontSize: 10, lineHeight: 1.7, color: "#2a4860" }}>
            <div style={{ color: "#3a6080", marginBottom: 4, fontWeight: 600 }}>QUICK START</div>
            <div style={{ color: "#00ffc8" }}>import llmsr_tracker</div>
            <div style={{ color: "#a29bfe" }}>llmsr_tracker.patch()</div>
          </div>
        </div>
      </div>
    </div>
  );
}