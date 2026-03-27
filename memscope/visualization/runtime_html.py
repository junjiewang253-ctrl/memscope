from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List


def _format_bytes(num_bytes: float | int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024
    return f"{num_bytes} B"


def _safe_json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False)


def _build_summary(report: Dict[str, Any]) -> Dict[str, Any]:
    metadata = report.get("metadata", {})
    comparisons = report.get("comparisons", {})
    peak = report.get("peak", {})

    return {
        "rank": metadata.get("rank", "unknown"),
        "world_size": metadata.get("world_size", "unknown"),
        "device": metadata.get("device", "unknown"),
        "device_name": metadata.get("device_name", "unknown"),
        "dtype": metadata.get("dtype", "unknown"),
        "tp": metadata.get("tensor_model_parallel_size", "unknown"),
        "pp": metadata.get("pipeline_model_parallel_size", "unknown"),
        "dp": metadata.get("data_parallel_size", "unknown"),
        "optimizer": metadata.get("optimizer", "unknown"),
        "runtime_peak_memory_bytes": peak.get("memory_bytes", 0),
        "runtime_peak_reserved_bytes": peak.get("reserved_bytes", 0),
        "runtime_peak_phase": peak.get("phase", ""),
        "runtime_peak_module": peak.get("module", ""),
        "runtime_peak_step": peak.get("step", 0),
        "static_peak_memory_bytes": comparisons.get("static_peak_memory_bytes", 0),
        "peak_diff_bytes": comparisons.get("peak_diff_bytes", 0),
        "peak_diff_ratio": comparisons.get("peak_diff_ratio", 0),
    }


def _build_step_boundaries(runtime_trace: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    step_events = []
    for ev in runtime_trace:
        if ev.get("event_type") != "step":
            continue
        step_events.append({
            "step": ev.get("step", 0),
            "phase": ev.get("phase", ""),
            "module": ev.get("module", ""),
            "allocated_after": ev.get("mem_allocated_after", 0),
            "reserved_after": ev.get("mem_reserved_after", 0),
            "max_allocated": ev.get("max_mem_allocated", 0),
            "max_reserved": ev.get("max_mem_reserved", 0),
            "notes": ev.get("notes", ""),
        })
    return step_events


def _build_module_aggregate(runtime_trace: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    agg = defaultdict(lambda: {
        "module": "",
        "count": 0,
        "max_allocated_after": 0,
        "max_delta_allocated": 0,
        "phases": set(),
        "event_types": set(),
        "sample_notes": set(),
    })

    for ev in runtime_trace:
        module = ev.get("module", "")
        if not module:
            continue

        item = agg[module]
        item["module"] = module
        item["count"] += 1
        item["max_allocated_after"] = max(
            item["max_allocated_after"],
            ev.get("mem_allocated_after", 0),
        )
        item["max_delta_allocated"] = max(
            item["max_delta_allocated"],
            ev.get("delta_allocated", 0),
        )
        item["phases"].add(ev.get("phase", ""))
        item["event_types"].add(ev.get("event_type", ""))
        note = ev.get("notes", "")
        if note:
            item["sample_notes"].add(note)

    rows = []
    for _, item in agg.items():
        rows.append({
            "module": item["module"],
            "count": item["count"],
            "max_allocated_after": item["max_allocated_after"],
            "max_delta_allocated": item["max_delta_allocated"],
            "phases": ", ".join(sorted(x for x in item["phases"] if x)),
            "event_types": ", ".join(sorted(x for x in item["event_types"] if x)),
            "sample_notes": " | ".join(list(sorted(item["sample_notes"]))[:3]),
        })

    rows.sort(key=lambda x: x["max_allocated_after"], reverse=True)
    return rows


def _build_phase_aggregate(runtime_trace: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    agg = defaultdict(lambda: {
        "phase": "",
        "count": 0,
        "max_allocated_after": 0,
        "max_delta_allocated": 0,
    })

    for ev in runtime_trace:
        phase = ev.get("phase", "")
        if not phase:
            continue
        item = agg[phase]
        item["phase"] = phase
        item["count"] += 1
        item["max_allocated_after"] = max(
            item["max_allocated_after"], ev.get("mem_allocated_after", 0)
        )
        item["max_delta_allocated"] = max(
            item["max_delta_allocated"], ev.get("delta_allocated", 0)
        )

    rows = list(agg.values())
    rows.sort(key=lambda x: x["max_allocated_after"], reverse=True)
    return rows


def render_runtime_report_html(report: Dict[str, Any], title: str = "MemScope Runtime Report") -> str:
    runtime_trace = report.get("runtime_trace", [])
    top_events = report.get("top_events", [])

    summary = _build_summary(report)
    step_boundaries = _build_step_boundaries(runtime_trace)
    module_rows = _build_module_aggregate(runtime_trace)
    phase_rows = _build_phase_aggregate(runtime_trace)

    payload = {
        "summary": summary,
        "step_boundaries": step_boundaries,
        "top_events": top_events,
        "module_rows": module_rows,
        "phase_rows": phase_rows,
        "raw_report": report,
    }

    payload_json = _safe_json_dumps(payload)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>{title}</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    :root {{
      --bg: #0b1020;
      --panel: #121933;
      --panel-2: #1a2345;
      --text: #e7ecff;
      --muted: #98a2c7;
      --accent: #6ea8fe;
      --good: #5ad8a6;
      --warn: #ffcc66;
      --bad: #ff7b72;
      --border: #2c3768;
    }}

    * {{
      box-sizing: border-box;
    }}

    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.5;
    }}

    .container {{
      width: 100%;
      max-width: 1400px;
      margin: 0 auto;
      padding: 20px;
    }}

    h1, h2, h3 {{
      margin-top: 0;
      color: var(--text);
    }}

    p {{
      color: var(--muted);
    }}

    .grid {{
      display: grid;
      gap: 16px;
    }}

    .grid-4 {{
      grid-template-columns: repeat(4, minmax(0, 1fr));
    }}

    .grid-2 {{
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }}

    .card {{
      background: linear-gradient(180deg, var(--panel), var(--panel-2));
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 16px;
      box-shadow: 0 10px 30px rgba(0,0,0,0.18);
      overflow: hidden;
    }}

    .metric-label {{
      font-size: 13px;
      color: var(--muted);
      margin-bottom: 8px;
    }}

    .metric-value {{
      font-size: 26px;
      font-weight: 700;
      color: var(--text);
      word-break: break-word;
    }}

    .subtle {{
      color: var(--muted);
      font-size: 13px;
    }}

    .table-wrap {{
      overflow-x: auto;
      border-radius: 12px;
      border: 1px solid var(--border);
    }}

    table {{
      width: 100%;
      border-collapse: collapse;
      min-width: 900px;
      background: rgba(255,255,255,0.01);
    }}

    th, td {{
      text-align: left;
      padding: 10px 12px;
      border-bottom: 1px solid rgba(255,255,255,0.06);
      vertical-align: top;
      font-size: 14px;
    }}

    th {{
      color: var(--muted);
      background: rgba(255,255,255,0.03);
      position: sticky;
      top: 0;
      z-index: 1;
    }}

    .section {{
      margin-top: 20px;
    }}

    .toolbar {{
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      margin-bottom: 12px;
    }}

    input[type="text"] {{
      width: 100%;
      max-width: 360px;
      padding: 10px 12px;
      background: #0f1630;
      color: var(--text);
      border: 1px solid var(--border);
      border-radius: 10px;
      outline: none;
    }}

    .badge {{
      display: inline-block;
      padding: 4px 8px;
      border-radius: 999px;
      font-size: 12px;
      background: rgba(110,168,254,0.15);
      color: var(--accent);
      border: 1px solid rgba(110,168,254,0.35);
    }}

    .row {{
      display: flex;
      gap: 16px;
      flex-wrap: wrap;
    }}

    .row > .card {{
      flex: 1 1 420px;
    }}

    .code {{
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 13px;
      color: #d6e2ff;
      word-break: break-word;
    }}

    @media (max-width: 1100px) {{
      .grid-4 {{
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }}
      .grid-2 {{
        grid-template-columns: repeat(1, minmax(0, 1fr));
      }}
    }}

    @media (max-width: 700px) {{
      .grid-4 {{
        grid-template-columns: repeat(1, minmax(0, 1fr));
      }}
      .container {{
        padding: 12px;
      }}
      .metric-value {{
        font-size: 22px;
      }}
    }}
  </style>
</head>
<body>
  <div class="container">
    <h1>MemScope Runtime Report</h1>
    <p>Runtime memory trace visualization for one rank. This page is generated from <span class="code">runtime_report.json</span>.</p>

    <div class="grid grid-4">
      <div class="card">
        <div class="metric-label">Runtime peak allocated</div>
        <div class="metric-value" id="peakAllocated"></div>
        <div class="subtle" id="peakMeta"></div>
      </div>
      <div class="card">
        <div class="metric-label">Runtime peak reserved</div>
        <div class="metric-value" id="peakReserved"></div>
        <div class="subtle" id="deviceMeta"></div>
      </div>
      <div class="card">
        <div class="metric-label">Static peak estimate</div>
        <div class="metric-value" id="staticPeak"></div>
        <div class="subtle" id="parallelMeta"></div>
      </div>
      <div class="card">
        <div class="metric-label">Static vs runtime diff</div>
        <div class="metric-value" id="peakDiff"></div>
        <div class="subtle" id="peakDiffRatio"></div>
      </div>
    </div>

    <div class="section row">
      <div class="card">
        <h2>Step-level memory timeline</h2>
        <div id="stepTimeline" style="height: 420px;"></div>
      </div>
      <div class="card">
        <h2>Phase summary</h2>
        <div id="phaseBar" style="height: 420px;"></div>
      </div>
    </div>

    <div class="section card">
      <h2>Top events</h2>
      <div class="toolbar">
        <input type="text" id="topEventsSearch" placeholder="Filter top events by module / notes / phase..." />
      </div>
      <div class="table-wrap">
        <table id="topEventsTable">
          <thead>
            <tr>
              <th>Step</th>
              <th>Event type</th>
              <th>Module</th>
              <th>Phase</th>
              <th>Allocated after</th>
              <th>Reserved after</th>
              <th>Delta allocated</th>
              <th>Notes</th>
            </tr>
          </thead>
          <tbody></tbody>
        </table>
      </div>
    </div>

    <div class="section card">
      <h2>Module aggregate</h2>
      <div class="toolbar">
        <input type="text" id="moduleSearch" placeholder="Filter modules..." />
      </div>
      <div class="table-wrap">
        <table id="moduleTable">
          <thead>
            <tr>
              <th>Module</th>
              <th>Count</th>
              <th>Max allocated after</th>
              <th>Max delta allocated</th>
              <th>Phases</th>
              <th>Event types</th>
              <th>Sample notes</th>
            </tr>
          </thead>
          <tbody></tbody>
        </table>
      </div>
    </div>

    <div class="section card">
      <h2>Raw metadata</h2>
      <pre id="metadataBox" class="code"></pre>
    </div>
  </div>

  <script>
    const payload = {payload_json};

    function formatBytes(numBytes) {{
      const units = ["B", "KiB", "MiB", "GiB", "TiB"];
      let value = Number(numBytes || 0);
      let i = 0;
      while (value >= 1024 && i < units.length - 1) {{
        value /= 1024;
        i += 1;
      }}
      return `${{value.toFixed(2)}} ${{units[i]}}`;
    }}

    function formatRatio(x) {{
      const v = Number(x || 0) * 100.0;
      return `${{v.toFixed(2)}}%`;
    }}

    function setSummary() {{
      const s = payload.summary;
      document.getElementById("peakAllocated").textContent = formatBytes(s.runtime_peak_memory_bytes);
      document.getElementById("peakReserved").textContent = formatBytes(s.runtime_peak_reserved_bytes);
      document.getElementById("staticPeak").textContent = formatBytes(s.static_peak_memory_bytes);
      document.getElementById("peakDiff").textContent = formatBytes(s.peak_diff_bytes);

      document.getElementById("peakMeta").textContent =
        `phase=${{s.runtime_peak_phase}}, step=${{s.runtime_peak_step}}, module=${{s.runtime_peak_module}}`;

      document.getElementById("deviceMeta").textContent =
        `${{s.device_name}} · ${{s.device}} · dtype=${{s.dtype}}`;

      document.getElementById("parallelMeta").textContent =
        `rank=${{s.rank}} / world=${{s.world_size}} · TP=${{s.tp}} PP=${{s.pp}} DP=${{s.dp}} · optimizer=${{s.optimizer}}`;

      document.getElementById("peakDiffRatio").textContent =
        `peak_diff_ratio = ${{formatRatio(s.peak_diff_ratio)}}`;
    }}

    function renderStepTimeline() {{
      const rows = payload.step_boundaries;
      const x = rows.map(r => `step ${{r.step}} · ${{r.phase}}`);
      const allocated = rows.map(r => r.allocated_after);
      const reserved = rows.map(r => r.reserved_after);
      const hover = rows.map(r =>
        `step=${{r.step}}<br>phase=${{r.phase}}<br>allocated=${{formatBytes(r.allocated_after)}}<br>reserved=${{formatBytes(r.reserved_after)}}<br>notes=${{r.notes || ""}}`
      );

      const trace1 = {{
        x,
        y: allocated,
        type: "scatter",
        mode: "lines+markers",
        name: "allocated_after",
        line: {{ color: "#6ea8fe", width: 3 }},
        marker: {{ size: 7 }},
        text: hover,
        hovertemplate: "%{{text}}<extra></extra>",
      }};

      const trace2 = {{
        x,
        y: reserved,
        type: "scatter",
        mode: "lines+markers",
        name: "reserved_after",
        line: {{ color: "#ffcc66", width: 3 }},
        marker: {{ size: 7 }},
        text: hover,
        hovertemplate: "%{{text}}<extra></extra>",
      }};

      Plotly.newPlot("stepTimeline", [trace1, trace2], {{
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(0,0,0,0)",
        font: {{ color: "#e7ecff" }},
        xaxis: {{ tickangle: -35, gridcolor: "rgba(255,255,255,0.08)" }},
        yaxis: {{
          title: "Bytes",
          gridcolor: "rgba(255,255,255,0.08)"
        }},
        legend: {{ orientation: "h" }},
        margin: {{ l: 60, r: 20, t: 20, b: 120 }},
      }}, {{ responsive: true }});
    }}

    function renderPhaseBar() {{
      const rows = payload.phase_rows;
      const x = rows.map(r => r.phase);
      const y = rows.map(r => r.max_allocated_after);
      const text = rows.map(r =>
        `phase=${{r.phase}}<br>count=${{r.count}}<br>max_allocated=${{formatBytes(r.max_allocated_after)}}<br>max_delta=${{formatBytes(r.max_delta_allocated)}}`
      );

      Plotly.newPlot("phaseBar", [{{
        x,
        y,
        type: "bar",
        marker: {{
          color: ["#6ea8fe", "#5ad8a6", "#ffcc66", "#ff7b72", "#9d7dff", "#7ee0ff"]
        }},
        text,
        hovertemplate: "%{{text}}<extra></extra>",
      }}], {{
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(0,0,0,0)",
        font: {{ color: "#e7ecff" }},
        xaxis: {{ gridcolor: "rgba(255,255,255,0.08)" }},
        yaxis: {{ title: "Max allocated after (bytes)", gridcolor: "rgba(255,255,255,0.08)" }},
        margin: {{ l: 60, r: 20, t: 20, b: 60 }},
      }}, {{ responsive: true }});
    }}

    function populateTable(tableId, rows, columns) {{
      const tbody = document.querySelector(`#${{tableId}} tbody`);
      tbody.innerHTML = "";
      for (const row of rows) {{
        const tr = document.createElement("tr");
        for (const col of columns) {{
          const td = document.createElement("td");
          const value = row[col.key];
          td.textContent = col.format ? col.format(value, row) : (value ?? "");
          tr.appendChild(td);
        }}
        tbody.appendChild(tr);
      }}
    }}

    function renderTopEvents(filterText = "") {{
      const q = filterText.trim().toLowerCase();
      let rows = payload.top_events || [];
      if (q) {{
        rows = rows.filter(r =>
          (r.module || "").toLowerCase().includes(q) ||
          (r.phase || "").toLowerCase().includes(q) ||
          (r.event_type || "").toLowerCase().includes(q) ||
          (r.notes || "").toLowerCase().includes(q)
        );
      }}

      populateTable("topEventsTable", rows, [
        {{ key: "step" }},
        {{ key: "event_type" }},
        {{ key: "module" }},
        {{ key: "phase" }},
        {{ key: "mem_allocated_after", format: v => formatBytes(v) }},
        {{ key: "mem_reserved_after", format: v => formatBytes(v) }},
        {{ key: "delta_allocated", format: v => formatBytes(v) }},
        {{ key: "notes" }},
      ]);
    }}

    function renderModuleRows(filterText = "") {{
      const q = filterText.trim().toLowerCase();
      let rows = payload.module_rows || [];
      if (q) {{
        rows = rows.filter(r =>
          (r.module || "").toLowerCase().includes(q) ||
          (r.phases || "").toLowerCase().includes(q) ||
          (r.event_types || "").toLowerCase().includes(q) ||
          (r.sample_notes || "").toLowerCase().includes(q)
        );
      }}

      populateTable("moduleTable", rows, [
        {{ key: "module" }},
        {{ key: "count" }},
        {{ key: "max_allocated_after", format: v => formatBytes(v) }},
        {{ key: "max_delta_allocated", format: v => formatBytes(v) }},
        {{ key: "phases" }},
        {{ key: "event_types" }},
        {{ key: "sample_notes" }},
      ]);
    }}

    function renderMetadata() {{
      const metadata = payload.raw_report?.metadata || {{}};
      const comparisons = payload.raw_report?.comparisons || {{}};
      document.getElementById("metadataBox").textContent =
        JSON.stringify({{ metadata, comparisons }}, null, 2);
    }}

    function bindSearch() {{
      document.getElementById("topEventsSearch").addEventListener("input", (e) => {{
        renderTopEvents(e.target.value);
      }});
      document.getElementById("moduleSearch").addEventListener("input", (e) => {{
        renderModuleRows(e.target.value);
      }});
    }}

    setSummary();
    renderStepTimeline();
    renderPhaseBar();
    renderTopEvents();
    renderModuleRows();
    renderMetadata();
    bindSearch();
  </script>
</body>
</html>
"""


def write_runtime_report_html(report_or_path: Dict[str, Any] | str | Path, outpath: str | Path) -> Path:
    if isinstance(report_or_path, (str, Path)):
        report_path = Path(report_or_path)
        with report_path.open("r", encoding="utf-8") as f:
            report = json.load(f)
    else:
        report = report_or_path

    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    html = render_runtime_report_html(report)
    outpath.write_text(html, encoding="utf-8")
    return outpath