"""
dashboard.py — Live monitoring dashboard for the improve.py training loop.

Usage:
    uv run python dashboard.py            # opens http://localhost:8765
    uv run python dashboard.py --port 8888
    uv run python dashboard.py --no-browser
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import threading
import webbrowser
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer

LEADERBOARD_PATH = "leaderboard.json"
SUMMARIES_PATH   = "summaries.json"
LOG_PATH         = "improve.log"
LOG_TAIL_LINES   = 100

# ── Training process state ────────────────────────────────────────────────────
_proc: subprocess.Popen | None = None
_proc_lock = threading.Lock()


def _is_running() -> bool:
    with _proc_lock:
        return _proc is not None and _proc.poll() is None


def _start_training(extra_args: list[str] | None = None) -> dict:
    global _proc
    with _proc_lock:
        if _proc is not None and _proc.poll() is None:
            return {"ok": False, "error": "Already running"}
        cmd = ["uv", "run", "python", "improve.py"] + (extra_args or [])
        try:
            _proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return {"ok": True, "pid": _proc.pid}
        except Exception as e:
            return {"ok": False, "error": str(e)}


def _stop_training() -> dict:
    global _proc
    with _proc_lock:
        if _proc is None or _proc.poll() is not None:
            return {"ok": False, "error": "Not running"}
        try:
            _proc.terminate()
            return {"ok": True}
        except Exception as e:
            return {"ok": False, "error": str(e)}


# ── Data helpers ──────────────────────────────────────────────────────────────

def _load_json(path: str) -> list:
    if os.path.exists(path):
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            pass
    return []


def _log_tail(n: int = LOG_TAIL_LINES) -> list[str]:
    if not os.path.exists(LOG_PATH):
        return []
    try:
        with open(LOG_PATH, "rb") as f:
            f.seek(0, 2)
            buf = min(f.tell(), 65536)
            f.seek(-buf, 2)
            raw = f.read().decode("utf-8", errors="replace")
        return raw.splitlines()[-n:]
    except Exception:
        return []


def build_api_data() -> dict:
    board     = _load_json(LEADERBOARD_PATH)
    summaries = _load_json(SUMMARIES_PATH)

    # ── bests in chronological order ─────────────────────────────────────────
    bests = sorted(
        [e for e in board if e.get("commit_hash")],
        key=lambda e: e.get("trial", 0),
    )

    improvement_series = []
    running_best = None
    for e in bests:
        score    = e.get("score", 0)
        val_net  = e.get("val_net_per_round",   e.get("test_net_per_round", 0))
        val_hit  = e.get("val_round_hit_rate",  e.get("test_round_hit_rate", 0)) * 100
        test_net = e.get("test_net_per_round",  0)
        test_hit = e.get("test_round_hit_rate", 0) * 100
        delta = round(score - running_best, 4) if running_best is not None else None
        running_best = score
        improvement_series.append({
            "trial":    e.get("trial", 0),
            "strategy": e.get("strategy_name", ""),
            "keywords": e.get("keywords", []),
            "score":    round(score, 4),
            "val_net":  round(val_net,  2),
            "val_hit":  round(val_hit,  1),
            "test_net": round(test_net, 2),
            "test_hit": round(test_hit, 1),
            "commit":   (e.get("commit_hash") or "")[:7],
            "timestamp": e.get("timestamp", ""),
            "delta_score": delta,
        })

    # ── recent trials (newest first) ─────────────────────────────────────────
    recent_rows = []
    for e in sorted(board, key=lambda e: e.get("trial", 0), reverse=True)[:60]:
        val_net  = e.get("val_net_per_round",  e.get("test_net_per_round", 0))
        val_hit  = e.get("val_round_hit_rate", e.get("test_round_hit_rate", 0)) * 100
        test_net = e.get("test_net_per_round", 0)
        test_hit = e.get("test_round_hit_rate", 0) * 100
        recent_rows.append({
            "trial":    e.get("trial", 0),
            "strategy": e.get("strategy_name", ""),
            "keywords": ", ".join(e.get("keywords", [])),
            "score":    round(e.get("score", 0), 4),
            "val_net":  round(val_net,  2),
            "val_hit":  round(val_hit,  1),
            "test_net": round(test_net, 2),
            "test_hit": round(test_hit, 1),
            "is_best":  bool(e.get("commit_hash")),
            "commit":   (e.get("commit_hash") or "")[:7],
            "timestamp": e.get("timestamp", ""),
        })

    # ── stats ─────────────────────────────────────────────────────────────────
    best_entry = max(board, key=lambda e: e.get("score", -999), default=None)
    current_best_score = best_entry["score"] if best_entry else None

    training_elapsed_s = None
    log_lines = _log_tail()
    for line in log_lines:
        if line.startswith("[") and len(line) > 20:
            try:
                ts = datetime.strptime(line[1:20], "%Y-%m-%d %H:%M:%S")
                training_elapsed_s = int((datetime.now() - ts).total_seconds())
                break
            except Exception:
                pass

    stats = {
        "total_trials":        len(board),
        "total_bests":         len(bests),
        "current_best_score":  round(current_best_score, 4) if current_best_score is not None else None,
        "best_strategy":       best_entry.get("strategy_name", "") if best_entry else "",
        "best_val_net":        round(best_entry.get("val_net_per_round",  best_entry.get("test_net_per_round",  0)), 2) if best_entry else 0,
        "best_val_hit":        round(best_entry.get("val_round_hit_rate", best_entry.get("test_round_hit_rate", 0)) * 100, 1) if best_entry else 0,
        "best_test_net":       round(best_entry.get("test_net_per_round", 0), 2) if best_entry else 0,
        "best_test_hit":       round(best_entry.get("test_round_hit_rate", 0) * 100, 1) if best_entry else 0,
        "training_elapsed_s":  training_elapsed_s,
        "is_running":          _is_running(),
        "pid":                 _proc.pid if _proc and _proc.poll() is None else None,
    }

    return {
        "stats":              stats,
        "improvement_series": improvement_series,
        "recent_trials":      recent_rows,
        "summaries":          summaries,
        "log_tail":           log_lines,
        "server_time":        datetime.now().strftime("%H:%M:%S"),
    }


# ── HTML ──────────────────────────────────────────────────────────────────────

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>LotoFoot RL — Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
:root {
  --bg:#0d1117; --card:#161b22; --card2:#1c2128; --border:#30363d;
  --green:#3fb950; --orange:#f0883e; --blue:#58a6ff; --purple:#bc8cff;
  --red:#f85149; --yellow:#d29922; --cyan:#39d353;
  --muted:#8b949e; --text:#e6edf3; --text2:#c9d1d9;
}
*{box-sizing:border-box;margin:0;padding:0;}
body{background:var(--bg);color:var(--text);font-family:'Segoe UI',system-ui,sans-serif;font-size:13px;}
a{color:var(--blue);}

/* ── header ── */
header{background:var(--card);border-bottom:1px solid var(--border);
  padding:12px 20px;display:flex;align-items:center;justify-content:space-between;gap:12px;flex-wrap:wrap;}
header h1{font-size:17px;font-weight:600;color:var(--blue);}
.header-right{display:flex;align-items:center;gap:12px;flex-wrap:wrap;}

/* ── status + buttons ── */
.status-dot{width:9px;height:9px;border-radius:50%;display:inline-block;margin-right:5px;}
.status-dot.running{background:var(--green);animation:pulse 1.5s infinite;}
.status-dot.stopped{background:var(--muted);}
@keyframes pulse{0%,100%{opacity:1;}50%{opacity:.25;}}
.status-label{font-size:12px;}

.btn{padding:5px 14px;border:none;border-radius:6px;cursor:pointer;
  font-size:12px;font-weight:600;transition:.15s;}
.btn-start{background:#238636;color:#fff;}
.btn-start:hover{background:#2ea043;}
.btn-stop{background:#b62324;color:#fff;}
.btn-stop:hover{background:#da3633;}
.btn:disabled{opacity:.4;cursor:not-allowed;}
.server-time{color:var(--muted);font-size:11px;}

/* ── layout ── */
.grid{display:grid;gap:14px;padding:14px 20px;}
.grid-4{grid-template-columns:repeat(4,1fr);}
.grid-2{grid-template-columns:1fr 1fr;}
.grid-3{grid-template-columns:repeat(3,1fr);}
@media(max-width:1100px){.grid-4{grid-template-columns:repeat(2,1fr);}}
@media(max-width:700px){.grid-4,.grid-2,.grid-3{grid-template-columns:1fr;}}

.card{background:var(--card);border:1px solid var(--border);border-radius:8px;padding:14px;}
.card-title{font-size:10.5px;text-transform:uppercase;letter-spacing:.08em;
  color:var(--muted);margin-bottom:10px;}

/* ── stat cards ── */
.stat-value{font-size:26px;font-weight:700;line-height:1;}
.stat-sub{font-size:11.5px;color:var(--muted);margin-top:4px;}
.green{color:var(--green);} .orange{color:var(--orange);}
.blue{color:var(--blue);}   .red{color:var(--red);}
.yellow{color:var(--yellow);} .purple{color:var(--purple);}

/* ── charts ── */
.chart-wrap{position:relative;height:200px;}

/* ── tables ── */
.tbl-wrap{overflow-x:auto;max-height:280px;overflow-y:auto;}
table{width:100%;border-collapse:collapse;font-size:12px;}
th{color:var(--muted);text-transform:uppercase;font-size:10px;letter-spacing:.05em;
  padding:5px 7px;border-bottom:1px solid var(--border);text-align:left;white-space:nowrap;
  position:sticky;top:0;background:var(--card);z-index:1;}
td{padding:5px 7px;border-bottom:1px solid #21262d;white-space:nowrap;}
tr:last-child td{border-bottom:none;}
tr.is-best td{background:rgba(63,185,80,.06);}
tr:hover td{background:var(--card2);}
.badge{display:inline-block;font-size:10px;padding:1px 5px;border-radius:10px;font-weight:600;}
.badge-best{background:rgba(63,185,80,.2);color:var(--green);}
.mono{font-family:'Cascadia Code','Fira Code',monospace;}

/* ── log ── */
.log-box{background:#010409;border:1px solid var(--border);border-radius:6px;
  font-family:monospace;font-size:11.5px;padding:10px 12px;
  height:260px;overflow-y:auto;white-space:pre-wrap;word-break:break-all;color:#8b949e;}
.log-best{color:var(--green);font-weight:bold;}
.log-trial{color:var(--blue);}
.log-err{color:var(--red);}
.log-hyp{color:var(--purple);}
.log-stat{color:var(--text2);}

/* ── summaries timeline ── */
.summary-timeline{display:flex;flex-direction:column;gap:12px;max-height:520px;overflow-y:auto;}
.summary-card{background:var(--card2);border:1px solid var(--border);border-radius:8px;padding:12px 14px;
  position:relative;}
.summary-card::before{content:'';position:absolute;left:-1px;top:0;bottom:0;width:3px;
  border-radius:8px 0 0 8px;background:var(--green);}
.summary-num{font-size:11px;color:var(--muted);margin-bottom:4px;}
.summary-title{font-weight:600;font-size:13px;margin-bottom:6px;}
.summary-meta{display:flex;gap:14px;flex-wrap:wrap;margin-bottom:8px;font-size:11.5px;}
.summary-meta span{color:var(--muted);}
.summary-meta strong{color:var(--text);}
.summary-section{font-size:11.5px;margin-bottom:6px;}
.summary-section h4{color:var(--muted);text-transform:uppercase;font-size:10px;
  letter-spacing:.07em;margin-bottom:3px;}
.summary-section ul{padding-left:16px;}
.summary-section li{margin-bottom:2px;color:var(--text2);}
.summary-hyp li{color:var(--purple);}
.summary-chip{display:inline-block;background:#21262d;border:1px solid var(--border);
  border-radius:4px;padding:1px 6px;font-size:10.5px;margin:1px;font-family:monospace;}
</style>
</head>
<body>

<header>
  <div style="display:flex;align-items:center;gap:10px;">
    <h1>LotoFoot RL — Dashboard</h1>
  </div>
  <div class="header-right">
    <div style="display:flex;align-items:center;gap:6px;">
      <span class="status-dot stopped" id="statusDot"></span>
      <span class="status-label" id="statusLabel">Stopped</span>
    </div>
    <button class="btn btn-start" id="btnStart" onclick="startTraining()">▶ Start training</button>
    <button class="btn btn-stop"  id="btnStop"  onclick="stopTraining()" disabled>■ Stop</button>
    <span class="server-time" id="serverTime">—</span>
  </div>
</header>

<!-- Stat cards -->
<div class="grid grid-4">
  <div class="card">
    <div class="card-title">Total Trials</div>
    <div class="stat-value blue"   id="sTrials">—</div>
    <div class="stat-sub"          id="sBests">—</div>
  </div>
  <div class="card">
    <div class="card-title">Best Score (val)</div>
    <div class="stat-value green"  id="sScore">—</div>
    <div class="stat-sub"          id="sStrategy">—</div>
  </div>
  <div class="card">
    <div class="card-title">Best Val vs Test Net/Round</div>
    <div class="stat-value orange" id="sValNet">—</div>
    <div class="stat-sub"          id="sTestNet">—</div>
  </div>
  <div class="card">
    <div class="card-title">Training Time</div>
    <div class="stat-value yellow" id="sTime">—</div>
    <div class="stat-sub"          id="sRate">—</div>
  </div>
</div>

<!-- Charts row 1: net P&L -->
<div class="grid grid-2">
  <div class="card">
    <div class="card-title">Net P&L / Round at each new best ($) — Val vs Test</div>
    <div class="chart-wrap"><canvas id="chartNet"></canvas></div>
  </div>
  <div class="card">
    <div class="card-title">Round Hit Rate at each new best (%) — Val vs Test</div>
    <div class="chart-wrap"><canvas id="chartHit"></canvas></div>
  </div>
</div>

<!-- Charts row 2: score over all trials + distribution -->
<div class="grid grid-2">
  <div class="card">
    <div class="card-title">Score over all trials</div>
    <div class="chart-wrap"><canvas id="chartAll"></canvas></div>
  </div>
  <div class="card">
    <div class="card-title">Score distribution (all trials)</div>
    <div class="chart-wrap"><canvas id="chartDist"></canvas></div>
  </div>
</div>

<!-- Improvement history + summaries -->
<div class="grid grid-2">
  <div class="card">
    <div class="card-title">Improvement history (all bests)</div>
    <div class="tbl-wrap">
    <table id="tblBests">
      <thead><tr>
        <th>#</th><th>Trial</th><th>Strategy</th><th>Score</th>
        <th>Val Net</th><th>Val Hit</th><th>Test Net</th><th>Test Hit</th>
        <th>Δ</th><th>Commit</th>
      </tr></thead>
      <tbody></tbody>
    </table>
    </div>
  </div>
  <div class="card">
    <div class="card-title">Agent summaries — what worked &amp; what to try next</div>
    <div class="summary-timeline" id="summaryTimeline">
      <div style="color:var(--muted);font-size:12px;">No summaries yet. A summary is written each time a new best is found.</div>
    </div>
  </div>
</div>

<!-- Recent trials -->
<div class="grid">
  <div class="card">
    <div class="card-title">Recent trials (last 60)</div>
    <div class="tbl-wrap">
    <table id="tblRecent">
      <thead><tr>
        <th>Trial</th><th>Strategy</th><th>Keywords</th>
        <th>Score</th><th>Val Net</th><th>Val Hit</th><th>Test Net</th><th>Test Hit</th>
        <th>Flag</th><th>Time</th>
      </tr></thead>
      <tbody></tbody>
    </table>
    </div>
  </div>
</div>

<!-- Live log -->
<div class="grid">
  <div class="card">
    <div class="card-title">Live log (improve.log)</div>
    <div class="log-box" id="logBox">Waiting for data…</div>
  </div>
</div>

<div style="height:20px;"></div>

<script>
// ── Chart factory ────────────────────────────────────────────────────────────
const BASE = {
  animation:false, responsive:true, maintainAspectRatio:false,
  plugins:{legend:{labels:{color:'#8b949e',font:{size:11},boxWidth:12}}},
  scales:{
    x:{grid:{color:'#21262d'},ticks:{color:'#8b949e',font:{size:10}}},
    y:{grid:{color:'#21262d'},ticks:{color:'#8b949e',font:{size:10}}},
  },
};

function mkLine(id, ds, extraOpts={}) {
  const ctx = document.getElementById(id).getContext('2d');
  return new Chart(ctx, {
    type:'line', data:{labels:[], datasets:ds},
    options: deepMerge(BASE, extraOpts),
  });
}
function mkBar(id, color) {
  const ctx = document.getElementById(id).getContext('2d');
  return new Chart(ctx, {
    type:'bar',
    data:{labels:[],datasets:[{data:[],backgroundColor:color+'55',borderColor:color,borderWidth:1}]},
    options:{...BASE, plugins:{...BASE.plugins,legend:{display:false}}},
  });
}
function deepMerge(a, b) {
  const r = {...a};
  for (const k in b) r[k] = (b[k] && typeof b[k]==='object' && !Array.isArray(b[k]))
    ? deepMerge(a[k]||{}, b[k]) : b[k];
  return r;
}

function ds(label, color, stepped=true) {
  return {label, data:[], borderColor:color, backgroundColor:color+'18',
    borderWidth:2, pointRadius:4, stepped: stepped ? 'before' : false,
    fill:true, tension:0};
}

const chartNet  = mkLine('chartNet',
  [ds('Val net/round','#3fb950'), ds('Test net/round','#58a6ff')],
  {scales:{y:{ticks:{callback:v=>'$'+v.toFixed(0)}}}});

const chartHit  = mkLine('chartHit',
  [ds('Val hit%','#f0883e'), ds('Test hit%','#bc8cff')],
  {scales:{y:{min:0,max:100,ticks:{callback:v=>v+'%'}}}});

const chartAll  = mkLine('chartAll',
  [ds('Score (all trials)','#58a6ff', false)],
  {plugins:{legend:{display:false}},
   elements:{point:{radius:3,hoverRadius:5}}});

const chartDist = mkBar('chartDist','#58a6ff');

// ── Helpers ──────────────────────────────────────────────────────────────────
const esc = s => String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');

function fmtTime(s) {
  if (!s && s!==0) return '—';
  const h=Math.floor(s/3600), m=Math.floor((s%3600)/60), ss=s%60;
  return h>0 ? `${h}h ${m}m` : m>0 ? `${m}m ${ss}s` : `${ss}s`;
}
function fmtNet(v) {
  const cls = v>=0?'green':'red';
  return `<span class="${cls}">${v>=0?'+$':'-$'}${Math.abs(v).toFixed(2)}</span>`;
}
function fmtScore(v) {
  const cls = v>=0?'green':'red';
  return `<span class="${cls}">${v>=0?'+':''}${v.toFixed(4)}</span>`;
}
function fmtHit(v) { return v.toFixed(1)+'%'; }

function colorLog(line) {
  const e = esc(line);
  if (line.includes('★ NEW BEST'))  return `<span class="log-best">${e}</span>`;
  if (line.includes('→'))           return `<span class="log-hyp">${e}</span>`;
  if (line.includes('── Trial'))    return `<span class="log-trial">${e}</span>`;
  if (line.includes('ERROR'))       return `<span class="log-err">${e}</span>`;
  if (line.match(/score=|val_net=|val_hit=/)) return `<span class="log-stat">${e}</span>`;
  return e;
}

function buildDist(trials) {
  if (!trials.length) return {labels:[],counts:[]};
  const scores = trials.map(t=>t.score);
  const mn = Math.floor(Math.min(...scores)/10)*10;
  const mx = Math.ceil(Math.max(...scores)/10)*10;
  const step = Math.max(10, Math.round((mx-mn)/10/10)*10);
  const buckets = {};
  for (let b=mn; b<=mx; b+=step) buckets[b]=0;
  for (const s of scores) { const b=Math.floor(s/step)*step; buckets[b]=(buckets[b]||0)+1; }
  const keys = Object.keys(buckets).map(Number).sort((a,b)=>a-b);
  return {labels:keys.map(k=>`${k}`), counts:keys.map(k=>buckets[k])};
}

// ── Start / Stop ─────────────────────────────────────────────────────────────
async function startTraining() {
  document.getElementById('btnStart').disabled = true;
  try {
    const r = await fetch('/api/start', {method:'POST'});
    const d = await r.json();
    if (!d.ok) alert('Could not start: ' + d.error);
  } catch(e) { alert('Error: '+e); }
}
async function stopTraining() {
  document.getElementById('btnStop').disabled = true;
  try {
    const r = await fetch('/api/stop', {method:'POST'});
    const d = await r.json();
    if (!d.ok) alert('Could not stop: ' + d.error);
  } catch(e) { alert('Error: '+e); }
}

function updateRunControls(isRunning) {
  const dot   = document.getElementById('statusDot');
  const label = document.getElementById('statusLabel');
  const btnS  = document.getElementById('btnStart');
  const btnT  = document.getElementById('btnStop');
  dot.className   = 'status-dot ' + (isRunning ? 'running' : 'stopped');
  label.textContent = isRunning ? 'Training running…' : 'Stopped';
  btnS.disabled   = isRunning;
  btnT.disabled   = !isRunning;
}

// ── Summaries ────────────────────────────────────────────────────────────────
function renderSummaries(summaries) {
  const el = document.getElementById('summaryTimeline');
  if (!summaries || !summaries.length) return;
  el.innerHTML = '';
  // Newest first
  [...summaries].reverse().forEach(s => {
    const isFirst = !s.score_delta;
    const deltaHtml = s.score_delta != null
      ? `<span class="${s.score_delta>=0?'green':'red'}">${s.score_delta>=0?'+':''}${s.score_delta.toFixed(4)}</span>`
      : '';
    const chips = (s.keywords||[]).map(k=>`<span class="summary-chip">${esc(k)}</span>`).join('');
    const insights = (s.what_worked||[]).map(i=>`<li>${esc(i)}</li>`).join('');
    const hyps = (s.hypotheses_for_next_agent||[]).map(h=>`<li>${esc(h)}</li>`).join('');
    const notTried = (s.predefined_not_yet_tried||[]).map(n=>`<span class="summary-chip">${esc(n)}</span>`).join('');

    el.innerHTML += `
<div class="summary-card">
  <div class="summary-num">Improvement #${s.improvement_number} &nbsp;·&nbsp; ${esc(s.timestamp||'')}
    ${s.commit ? `&nbsp;·&nbsp; <span class="mono" style="color:var(--muted)">${esc(s.commit)}</span>` : ''}
  </div>
  <div class="summary-title">
    <span class="badge badge-best">${esc(s.strategy)}</span>
    &nbsp; score ${fmtScore(s.score)}
    ${deltaHtml ? `&nbsp; Δ ${deltaHtml}` : ''}
  </div>
  <div class="summary-meta">
    <span>Val net <strong>${fmtNet(s.val_net)}</strong></span>
    <span>Val hit <strong>${s.val_hit_pct}%</strong></span>
    <span>Test net <strong>${fmtNet(s.test_net)}</strong></span>
    <span>Test hit <strong>${s.test_hit_pct}%</strong></span>
  </div>
  <div style="margin-bottom:8px;">${chips}</div>
  ${s.key_change ? `<div class="summary-section"><h4>Key change</h4><div style="color:var(--orange);font-size:11.5px;">${esc(s.key_change)}</div></div>` : ''}
  ${insights ? `<div class="summary-section"><h4>What worked</h4><ul>${insights}</ul></div>` : ''}
  ${hyps ? `<div class="summary-section summary-hyp"><h4>🔬 Next agent should try</h4><ul>${hyps}</ul></div>` : ''}
  ${notTried ? `<div class="summary-section"><h4>Predefined strategies not yet tried</h4>${notTried}</div>` : ''}
</div>`;
  });
}

// ── Main render ──────────────────────────────────────────────────────────────
let prevLogLen = 0;

function render(d) {
  const {stats, improvement_series, recent_trials, summaries, log_tail} = d;

  document.getElementById('serverTime').textContent = 'Updated ' + d.server_time;
  updateRunControls(stats.is_running);

  // Stat cards
  document.getElementById('sTrials').textContent = stats.total_trials ?? '—';
  document.getElementById('sBests').textContent  =
    `${stats.total_bests ?? 0} best${stats.total_bests!==1?'s':''} found`;
  document.getElementById('sScore').textContent  = stats.current_best_score != null
    ? (stats.current_best_score>=0?'+':'')+stats.current_best_score.toFixed(4) : '—';
  document.getElementById('sStrategy').textContent = stats.best_strategy || '—';
  document.getElementById('sValNet').textContent  =
    `Val ${stats.best_val_net>=0?'+$':'-$'}${Math.abs(stats.best_val_net).toFixed(2)}/round (${stats.best_val_hit}% hit)`;
  document.getElementById('sTestNet').textContent =
    `Test ${stats.best_test_net>=0?'+$':'-$'}${Math.abs(stats.best_test_net).toFixed(2)}/round (${stats.best_test_hit}% hit)`;
  document.getElementById('sTime').textContent   = fmtTime(stats.training_elapsed_s);
  document.getElementById('sRate').textContent   =
    (stats.training_elapsed_s && stats.total_trials)
    ? `${(stats.total_trials/(stats.training_elapsed_s/3600)).toFixed(1)} trials/hr`
    : '—';

  // Charts: net & hit rate (val + test)
  const labels = improvement_series.map((e,i) => `#${i+1} ${e.strategy}`);
  chartNet.data.labels = labels;
  chartNet.data.datasets[0].data = improvement_series.map(e=>e.val_net);
  chartNet.data.datasets[1].data = improvement_series.map(e=>e.test_net);
  chartNet.update();

  chartHit.data.labels = labels;
  chartHit.data.datasets[0].data = improvement_series.map(e=>e.val_hit);
  chartHit.data.datasets[1].data = improvement_series.map(e=>e.test_hit);
  chartHit.update();

  // Score over all trials
  const allSorted = [...recent_trials].reverse();
  chartAll.data.labels   = allSorted.map(e=>'#'+e.trial);
  chartAll.data.datasets[0].data = allSorted.map(e=>e.score);
  chartAll.update();

  // Distribution
  const dist = buildDist(recent_trials);
  chartDist.data.labels = dist.labels;
  chartDist.data.datasets[0].data = dist.counts;
  chartDist.update();

  // Improvement history table
  const bestTbody = document.querySelector('#tblBests tbody');
  bestTbody.innerHTML = '';
  [...improvement_series].reverse().forEach((e, idx) => {
    const rank = improvement_series.length - idx;
    const delta = e.delta_score != null
      ? `<span class="${e.delta_score>=0?'green':'red'}">${e.delta_score>=0?'+':''}${e.delta_score.toFixed(4)}</span>`
      : '—';
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td class="mono">${rank}</td>
      <td class="mono">${e.trial}</td>
      <td><span class="badge badge-best">${esc(e.strategy)}</span></td>
      <td>${fmtScore(e.score)}</td>
      <td>${fmtNet(e.val_net)}</td><td>${fmtHit(e.val_hit)}</td>
      <td>${fmtNet(e.test_net)}</td><td>${fmtHit(e.test_hit)}</td>
      <td>${delta}</td>
      <td class="mono" style="color:var(--muted)">${e.commit||'—'}</td>`;
    bestTbody.appendChild(tr);
  });

  // Recent trials table
  const recentTbody = document.querySelector('#tblRecent tbody');
  recentTbody.innerHTML = '';
  recent_trials.forEach(e => {
    const tr = document.createElement('tr');
    if (e.is_best) tr.classList.add('is-best');
    tr.innerHTML = `
      <td class="mono">${e.trial}</td>
      <td>${esc(e.strategy)}</td>
      <td style="color:var(--muted);max-width:220px;overflow:hidden;text-overflow:ellipsis"
          title="${esc(e.keywords)}">${esc(e.keywords)}</td>
      <td>${fmtScore(e.score)}</td>
      <td>${fmtNet(e.val_net)}</td><td>${fmtHit(e.val_hit)}</td>
      <td>${fmtNet(e.test_net)}</td><td>${fmtHit(e.test_hit)}</td>
      <td>${e.is_best?'<span class="badge badge-best">★</span>':''}</td>
      <td style="color:var(--muted)">${e.timestamp||'—'}</td>`;
    recentTbody.appendChild(tr);
  });

  // Summaries
  renderSummaries(summaries);

  // Log
  const logBox = document.getElementById('logBox');
  const atBottom = logBox.scrollHeight - logBox.scrollTop - logBox.clientHeight < 60;
  logBox.innerHTML = log_tail.length
    ? log_tail.map(colorLog).join('\n')
    : 'No log found. Start improve.py to begin training.';
  if (atBottom || log_tail.length !== prevLogLen) logBox.scrollTop = logBox.scrollHeight;
  prevLogLen = log_tail.length;
}

// ── Polling ──────────────────────────────────────────────────────────────────
async function fetchAndRender() {
  try {
    const r = await fetch('/api/data');
    if (r.ok) render(await r.json());
    else document.getElementById('serverTime').textContent = 'Error fetching data';
  } catch(e) {
    document.getElementById('serverTime').textContent = 'Connection error — retrying…';
  }
}
fetchAndRender();
setInterval(fetchAndRender, 3000);
</script>
</body>
</html>
"""


# ── HTTP handler ──────────────────────────────────────────────────────────────

class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # silence access log

    def _json(self, code: int, data: dict):
        body = json.dumps(data, default=str).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            body = HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/api/data":
            try:
                self._json(200, build_api_data())
            except Exception as e:
                self._json(500, {"error": str(e)})
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == "/api/start":
            self._json(200, _start_training())
        elif self.path == "/api/stop":
            self._json(200, _stop_training())
        else:
            self.send_response(404)
            self.end_headers()


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Live training dashboard")
    parser.add_argument("--port",       type=int,  default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    server = HTTPServer(("", args.port), Handler)
    url = f"http://localhost:{args.port}"
    print(f"Dashboard  →  {url}")
    print("Press Ctrl-C to stop.")

    if not args.no_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nDashboard stopped.")
        # Also stop training if it was started from here
        _stop_training()


if __name__ == "__main__":
    main()
