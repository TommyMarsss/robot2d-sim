"""生成单文件静态 HTML 交互式回放报告。

数据与逻辑全部内嵌（JSON + 原生 JS + Canvas），不依赖任何第三方库、
不启动任何服务，直接用浏览器打开即可。

功能：
  * 拖动时间轴 / 逐步前进后退 / 自动播放；
  * 任意时刻的估计位姿（箭头）与 2σ 协方差椭圆；
  * 可切换显示：估计轨迹、真值轨迹、误差曲线、NIS 曲线；
  * 发散 / 绑架 / 恢复事件在时间轴与地图上明确标出。
"""

from __future__ import annotations

import json
import math

from .ekf import FilterEvent
from .logio import HistoryEntry

_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>EKF 定位回放报告</title>
<style>
  body { font-family: -apple-system, "PingFang SC", sans-serif; margin: 16px;
         background: #fafafa; color: #222; }
  h1 { font-size: 18px; }
  #controls { margin: 8px 0; display: flex; gap: 12px; align-items: center;
              flex-wrap: wrap; }
  #controls button { padding: 4px 12px; }
  #slider { flex: 1; min-width: 200px; }
  #info { font-family: ui-monospace, monospace; font-size: 12px;
          white-space: pre-wrap; background: #fff; border: 1px solid #ddd;
          padding: 8px; margin: 8px 0; }
  canvas { background: #fff; border: 1px solid #ccc; }
  #plots { display: flex; gap: 12px; flex-wrap: wrap; margin-top: 8px; }
  .event-list { font-size: 12px; }
  .kidnap { color: #c00; font-weight: bold; }
  .diverged { color: #d60; font-weight: bold; }
  .recovered { color: #080; }
  label { font-size: 13px; user-select: none; }
</style>
</head>
<body>
<h1>EKF 定位回放报告</h1>
<div id="controls">
  <button id="btnPlay">▶ 播放</button>
  <button id="btnPrev">◀ 上一步</button>
  <button id="btnNext">下一步 ▶</button>
  <input id="slider" type="range" min="0" max="0" value="0" step="1">
  <span id="tlabel"></span>
</div>
<div id="controls">
  <label><input type="checkbox" id="showEst" checked> 估计轨迹</label>
  <label><input type="checkbox" id="showTruth" checked> 真值轨迹</label>
  <label><input type="checkbox" id="showEllipse" checked> 协方差椭圆 (2σ)</label>
  <label><input type="checkbox" id="showErr" checked> 误差 / NIS 曲线</label>
</div>
<canvas id="map" width="640" height="560"></canvas>
<div id="info"></div>
<div id="plots">
  <canvas id="errPlot" width="620" height="180"></canvas>
  <canvas id="nisPlot" width="620" height="180"></canvas>
</div>
<div id="events" class="event-list"></div>
<script>
"use strict";
const DATA = __DATA__;
const H = DATA.history, EV = DATA.events, LM = DATA.landmarks;
const N = H.length;
let idx = 0, playing = false, timer = null;

const mapC = document.getElementById("map");
const mctx = mapC.getContext("2d");
const errC = document.getElementById("errPlot"), ectx = errC.getContext("2d");
const nisC = document.getElementById("nisPlot"), nctx = nisC.getContext("2d");

// ---- 坐标变换：世界坐标 -> 画布 ----
let scale = 40, ox = 0, oy = 0;
function computeView() {
  let xs = [], ys = [];
  for (const h of H) { xs.push(h.x); ys.push(h.y); }
  for (const id in LM) { xs.push(LM[id][0]); ys.push(LM[id][1]); }
  for (const tr of DATA.truth) { xs.push(tr[1]); ys.push(tr[2]); }
  const x0 = Math.min(...xs), x1 = Math.max(...xs);
  const y0 = Math.min(...ys), y1 = Math.max(...ys);
  const pad = 1.0;
  scale = Math.min(mapC.width / (x1 - x0 + 2 * pad),
                   mapC.height / (y1 - y0 + 2 * pad));
  ox = mapC.width / 2 - scale * (x0 + x1) / 2;
  oy = mapC.height / 2 - scale * (y0 + y1) / 2;
}
function W2C(x, y) { return [ox + scale * x, oy - scale * y]; }

// ---- 2x2 协方差 -> 椭圆参数（闭式特征值）----
function ellipse(p, ns) {
  const a = p[0], b = p[1], d = p[3];
  const tr = a + d, det = a * d - b * b;
  const disc = Math.sqrt(Math.max(0, tr * tr / 4 - det));
  const l1 = tr / 2 + disc, l2 = tr / 2 - disc;
  const ang = 0.5 * Math.atan2(2 * b, a - d);
  return [ns * Math.sqrt(Math.max(l1, 0)), ns * Math.sqrt(Math.max(l2, 0)), ang];
}

function drawMap() {
  mctx.clearRect(0, 0, mapC.width, mapC.height);
  const showEst = document.getElementById("showEst").checked;
  const showTruth = document.getElementById("showTruth").checked;
  const showEllipse = document.getElementById("showEllipse").checked;

  // 网格
  mctx.strokeStyle = "#eee"; mctx.lineWidth = 1;
  const step = scale; // 1m
  for (let gx = ox % step; gx < mapC.width; gx += step) {
    mctx.beginPath(); mctx.moveTo(gx, 0); mctx.lineTo(gx, mapC.height); mctx.stroke();
  }
  for (let gy = oy % step; gy < mapC.height; gy += step) {
    mctx.beginPath(); mctx.moveTo(0, gy); mctx.lineTo(mapC.width, gy); mctx.stroke();
  }

  // 路标
  mctx.fillStyle = "#333";
  mctx.font = "11px sans-serif";
  for (const id in LM) {
    const [cx, cy] = W2C(LM[id][0], LM[id][1]);
    mctx.beginPath(); mctx.moveTo(cx - 5, cy); mctx.lineTo(cx + 5, cy);
    mctx.moveTo(cx, cy - 5); mctx.lineTo(cx, cy + 5);
    mctx.strokeStyle = "#333"; mctx.stroke();
    mctx.fillText("L" + id, cx + 6, cy - 4);
  }

  // 真值轨迹
  if (showTruth && DATA.truth.length > 1) {
    mctx.strokeStyle = "#2a2"; mctx.lineWidth = 1.5; mctx.beginPath();
    DATA.truth.forEach((tr, i) => {
      const [cx, cy] = W2C(tr[1], tr[2]);
      i ? mctx.lineTo(cx, cy) : mctx.moveTo(cx, cy);
    });
    mctx.stroke();
  }

  // 估计轨迹（到当前时刻）
  if (showEst) {
    mctx.strokeStyle = "#06c"; mctx.lineWidth = 1.5; mctx.beginPath();
    for (let i = 0; i <= idx; i++) {
      const [cx, cy] = W2C(H[i].x, H[i].y);
      i ? mctx.lineTo(cx, cy) : mctx.moveTo(cx, cy);
    }
    mctx.stroke();
  }

  // 事件标记（地图上）
  for (const e of EV) {
    const h = H[e.index];
    if (!h) continue;
    const [cx, cy] = W2C(h.x, h.y);
    mctx.beginPath();
    mctx.fillStyle = e.kind === "kidnap" ? "#c00" :
                     e.kind === "diverged" ? "#d60" : "#0a0";
    mctx.arc(cx, cy, 5, 0, 2 * Math.PI); mctx.fill();
  }

  // 当前协方差椭圆
  const h = H[idx];
  const [cx, cy] = W2C(h.x, h.y);
  if (showEllipse) {
    const [a, b, ang] = ellipse(h.P, 2.0);
    mctx.beginPath();
    mctx.ellipse(cx, cy, Math.max(a * scale, 1), Math.max(b * scale, 1),
                 -ang, 0, 2 * Math.PI);
    mctx.fillStyle = "rgba(0,102,204,0.15)"; mctx.fill();
    mctx.strokeStyle = "#06c"; mctx.lineWidth = 1; mctx.stroke();
  }

  // 当前位姿箭头
  const L = 0.4 * scale;
  mctx.beginPath();
  mctx.moveTo(cx, cy);
  mctx.lineTo(cx + L * Math.cos(h.theta), cy - L * Math.sin(h.theta));
  mctx.strokeStyle = "#003"; mctx.lineWidth = 2.5; mctx.stroke();
  mctx.beginPath(); mctx.arc(cx, cy, 3.5, 0, 2 * Math.PI);
  mctx.fillStyle = "#003"; mctx.fill();

  // 当前真值点
  if (showTruth && h.truth) {
    const [tx, ty] = W2C(h.truth[0], h.truth[1]);
    mctx.beginPath(); mctx.arc(tx, ty, 3.5, 0, 2 * Math.PI);
    mctx.fillStyle = "#2a2"; mctx.fill();
  }
}

function drawPlot(ctx, canvas, series, opts) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const t0 = H[0].t, t1 = H[N - 1].t;
  const X = t => (t - t0) / Math.max(t1 - t0, 1e-9) * (canvas.width - 40) + 30;
  let vmax = 0;
  for (const s of series) for (const v of s.values)
    if (v !== null && isFinite(v)) vmax = Math.max(vmax, v);
  if (opts.minMax) vmax = Math.max(vmax, opts.minMax);
  vmax *= 1.1;
  const Y = v => canvas.height - 20 - (v / vmax) * (canvas.height - 35);

  // 事件竖线
  for (const e of EV) {
    ctx.strokeStyle = e.kind === "kidnap" ? "#c00" :
                      e.kind === "diverged" ? "#d60" : "#0a0";
    ctx.setLineDash([4, 3]); ctx.beginPath();
    ctx.moveTo(X(e.t), 0); ctx.lineTo(X(e.t), canvas.height); ctx.stroke();
    ctx.setLineDash([]);
  }
  // 门限线
  if (opts.gate !== undefined) {
    ctx.strokeStyle = "#999"; ctx.setLineDash([6, 4]); ctx.beginPath();
    ctx.moveTo(0, Y(opts.gate)); ctx.lineTo(canvas.width, Y(opts.gate));
    ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle = "#999"; ctx.font = "10px sans-serif";
    ctx.fillText("门限 " + opts.gate, 4, Y(opts.gate) - 3);
  }
  // 曲线
  for (const s of series) {
    ctx.strokeStyle = s.color; ctx.lineWidth = 1.2; ctx.beginPath();
    let started = false;
    for (let i = 0; i < N; i++) {
      const v = s.values[i];
      if (v === null || !isFinite(v)) { started = false; continue; }
      const px = X(H[i].t), py = Y(Math.min(v, vmax));
      started ? ctx.lineTo(px, py) : ctx.moveTo(px, py);
      started = true;
    }
    ctx.stroke();
  }
  // 当前时刻游标
  ctx.strokeStyle = "#003"; ctx.lineWidth = 1; ctx.beginPath();
  ctx.moveTo(X(H[idx].t), 0); ctx.lineTo(X(H[idx].t), canvas.height); ctx.stroke();
  ctx.fillStyle = "#333"; ctx.font = "11px sans-serif";
  ctx.fillText(opts.title, 8, 12);
}

function draw() {
  drawMap();
  const h = H[idx];
  document.getElementById("tlabel").textContent =
    `t=${h.t.toFixed(2)}s  (${idx + 1}/${N})`;
  const sig = Math.sqrt(Math.max(h.P[0], 0)).toFixed(3);
  const sigY = Math.sqrt(Math.max(h.P[3], 0)).toFixed(3);
  const sigT = Math.sqrt(Math.max(h.P[8], 0)).toFixed(4);
  let txt = `估计位姿: x=${h.x.toFixed(3)}  y=${h.y.toFixed(3)}  θ=${h.theta.toFixed(3)} rad\n` +
            `标准差:   σx=${sig}  σy=${sigY}  σθ=${sigT}\n` +
            `NIS: ${h.nis === null ? "-" : h.nis.toFixed(2)}   状态: ` +
            (h.diverged ? "⚠ 发散/绑架" : "正常");
  if (h.truth) {
    const e = Math.hypot(h.x - h.truth[0], h.y - h.truth[1]);
    txt += `\n真值:     x=${h.truth[0].toFixed(3)}  y=${h.truth[1].toFixed(3)}  ` +
           `位置误差=${e.toFixed(3)} m`;
  }
  document.getElementById("info").textContent = txt;

  if (document.getElementById("showErr").checked) {
    document.getElementById("plots").style.display = "flex";
    drawPlot(ectx, errC, [
      {color: "#c33", values: DATA.posErr},
      {color: "#36c", values: DATA.headErr},
    ], {title: "位置误差(m, 红) / 航向误差(rad, 蓝)"});
    drawPlot(nctx, nisC, [{color: "#555", values: DATA.nis}],
             {title: "NIS（新息平方）", gate: DATA.nisGate, minMax: DATA.nisGate});
  } else {
    document.getElementById("plots").style.display = "none";
  }
}

// ---- 控件 ----
const slider = document.getElementById("slider");
slider.max = N - 1;
slider.addEventListener("input", () => { idx = +slider.value; draw(); });
function setIdx(i) { idx = Math.max(0, Math.min(N - 1, i)); slider.value = idx; draw(); }
document.getElementById("btnPrev").onclick = () => setIdx(idx - 1);
document.getElementById("btnNext").onclick = () => setIdx(idx + 1);
document.getElementById("btnPlay").onclick = function () {
  playing = !playing;
  this.textContent = playing ? "⏸ 暂停" : "▶ 播放";
  if (playing) timer = setInterval(() => {
    if (idx >= N - 1) { document.getElementById("btnPlay").click(); return; }
    setIdx(idx + 1);
  }, 60);
  else clearInterval(timer);
};
document.addEventListener("keydown", e => {
  if (e.key === "ArrowLeft") setIdx(idx - 1);
  if (e.key === "ArrowRight") setIdx(idx + 1);
});
for (const id of ["showEst", "showTruth", "showEllipse", "showErr"])
  document.getElementById(id).addEventListener("change", draw);

// 事件列表
const evDiv = document.getElementById("events");
evDiv.innerHTML = "<b>诊断事件：</b> " + (EV.length ? EV.map(e =>
  `<span class="${e.kind}">[t=${e.t.toFixed(2)}] ${e.kind}: ${e.detail}</span>`
).join("； ") : "无");

computeView();
draw();
</script>
</body>
</html>
"""


def build_report_data(history: list[HistoryEntry],
                      events: list[FilterEvent],
                      landmarks: dict[str, tuple[float, float]],
                      nis_gate: float) -> dict:
    """把滤波历史整理成报告用的 JSON 数据结构。"""
    hist_json = []
    pos_err: list[float | None] = []
    head_err: list[float | None] = []
    nis_series: list[float | None] = []
    truth_track: list[list[float]] = []
    last_truth: tuple[float, float, float] | None = None

    for h in history:
        if h.truth is not None:
            last_truth = h.truth
            truth_track.append([h.t, h.truth[0], h.truth[1], h.truth[2]])
        hist_json.append({
            "t": round(h.t, 4), "x": round(h.x, 4), "y": round(h.y, 4),
            "theta": round(h.theta, 5),
            "P": [h.P[0][0], h.P[0][1], h.P[1][0], h.P[1][1],
                  0, 0, 0, 0, h.P[2][2]],
            "nis": None if h.nis is None else round(h.nis, 3),
            "diverged": h.diverged,
            "truth": h.truth,
        })
        if last_truth is not None:
            pos_err.append(round(math.hypot(h.x - last_truth[0],
                                            h.y - last_truth[1]), 4))
            dth = h.theta - last_truth[2]
            dth = math.atan2(math.sin(dth), math.cos(dth))
            head_err.append(round(abs(dth), 5))
        else:
            pos_err.append(None)
            head_err.append(None)
        nis_series.append(None if h.nis is None else round(h.nis, 3))

    # 事件 -> 最近的历史索引（供地图标记）
    events_json = []
    times = [h.t for h in history]
    for e in events:
        best = min(range(len(times)), key=lambda i: abs(times[i] - e.t)) if times else 0
        events_json.append({"t": round(e.t, 3), "kind": e.kind,
                            "detail": e.detail, "index": best})

    return {
        "history": hist_json,
        "events": events_json,
        "landmarks": {k: list(v) for k, v in landmarks.items()},
        "truth": [[round(v, 4) for v in tr] for tr in truth_track],
        "posErr": pos_err,
        "headErr": head_err,
        "nis": nis_series,
        "nisGate": nis_gate,
    }


def render_html(data: dict) -> str:
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    return _HTML_TEMPLATE.replace("__DATA__", payload)


def write_report(path: str, history: list[HistoryEntry],
                 events: list[FilterEvent],
                 landmarks: dict[str, tuple[float, float]],
                 nis_gate: float) -> None:
    data = build_report_data(history, events, landmarks, nis_gate)
    with open(path, "w", encoding="utf-8") as f:
        f.write(render_html(data))
