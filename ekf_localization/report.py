"""把一次 EKF 运行结果渲染成**单一静态 HTML 文件**。

- 数据与全部逻辑内嵌（无网络、无第三方库、无需启动服务）；
- 核心滤波代码不 import 本模块，因此算法测试可脱离报告独立运行。

在浏览器中提供：时间轴拖动 / 逐步进退 / 自动播放、地图回放
（位姿、协方差椭圆、路标、估计与真值轨迹、发散与绑架事件标注）、
位置误差 / 航向误差 / NIS 三条时间序列与事件区间。
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional

from .ekf import EKFResult
from .logformat import LogBundle
from .monitor import MonitorStatus

# StepRecord 的紧凑行编码（JS 端按下标读取）
F_T, F_KIND, F_X, F_Y, F_H = 0, 1, 2, 3, 4
F_P00, F_P01, F_P11, F_P22 = 5, 6, 7, 8
F_NIS, F_EXY, F_ETH, F_ST, F_FL, F_LM = 9, 10, 11, 12, 13, 14
F_GX, F_GY, F_GH = 15, 16, 17

STATUS_CODE = {
    MonitorStatus.HEALTHY.value: 0,
    MonitorStatus.WARN.value: 1,
    MonitorStatus.DIVERGED.value: 2,
    MonitorStatus.KIDNAPPED.value: 3,
}
EVENT_CODE = {
    MonitorStatus.WARN.value: 1,
    MonitorStatus.DIVERGED.value: 2,
    MonitorStatus.KIDNAPPED.value: 3,
    MonitorStatus.RECOVERED.value: 4,
}

FLAG_REJECTED = 1
FLAG_RESET = 2


def _finite(v: Optional[float]) -> Optional[float]:
    if v is None:
        return None
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    return v


def build_payload(result: EKFResult, bundle: Optional[LogBundle] = None) -> Dict:
    lm_ids = sorted(result.landmarks.keys())
    lm_index = {k: i for i, k in enumerate(lm_ids)}
    landmarks = [[k, result.landmarks[k][0], result.landmarks[k][1]] for k in lm_ids]

    rows: List[List] = []
    for s in result.steps:
        kind = {"init": 0, "odom": 1, "obs": 2}[s.kind]
        c = s.cov
        flag = 0
        if s.rejected:
            flag |= FLAG_REJECTED
        if s.reset:
            flag |= FLAG_RESET
        row = [
            round(s.t, 6), kind,
            s.state[0], s.state[1], s.state[2],
            c[0], c[1], c[4], c[8],
            _finite(s.nis), _finite(s.err_xy), _finite(s.err_theta_deg),
            STATUS_CODE.get(s.status, 0), flag,
            lm_index.get(s.landmark, -1) if s.landmark else -1,
            s.truth[0] if s.truth else None,
            s.truth[1] if s.truth else None,
            s.truth[2] if s.truth else None,
        ]
        rows.append(row)

    events = [
        [round(e.t, 6), EVENT_CODE.get(e.status.value, 0), e.detail]
        for e in result.events
    ]

    t0 = result.steps[0].t if result.steps else 0.0
    t1 = result.steps[-1].t if result.steps else 0.0
    return {
        "steps": rows,
        "landmarks": landmarks,
        "events": events,
        "t0": t0,
        "t1": t1,
        "source": bundle.source if bundle else "",
    }


def render_html(result: EKFResult, bundle: Optional[LogBundle] = None) -> str:
    payload = json.dumps(build_payload(result, bundle), ensure_ascii=False, separators=(",", ":"))
    # 静态 HTML 内嵌 JSON：防止闭合标签注入
    payload = payload.replace("</", "<\\/")
    return _TEMPLATE.replace("__PAYLOAD__", payload)


def write_html(
    result: EKFResult, path: str | Path, bundle: Optional[LogBundle] = None
) -> Path:
    path = Path(path)
    path.write_text(render_html(result, bundle), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# 前端模板（原生 JS + Canvas，无第三方库；浅色/深色双主题）
# ---------------------------------------------------------------------------

_TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>EKF 2D 定位回放</title>
<style>
:root {
  color-scheme: light;
  --page:#f9f9f7; --surface:#fcfcfb; --ink:#0b0b0b; --ink-2:#52514e;
  --muted:#898781; --grid:#e1e0d9; --axis:#c3c2b7; --border:rgba(11,11,11,.10);
  --est:#2a78d6; --truth:#eb6834; --lm:#1baf7a;
  --good:#0ca30c; --warn:#b97d00; --serious:#ec835a; --critical:#d03b3b;
  --ellipse-fill:rgba(42,120,214,.10);
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --page:#0d0d0d; --surface:#1a1a19; --ink:#ffffff; --ink-2:#c3c2b7;
    --muted:#898781; --grid:#2c2c2a; --axis:#383835; --border:rgba(255,255,255,.10);
    --est:#3987e5; --truth:#d95926; --lm:#199e70;
    --good:#35b535; --warn:#fab219; --serious:#ec835a; --critical:#e66767;
    --ellipse-fill:rgba(57,135,229,.14);
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --page:#0d0d0d; --surface:#1a1a19; --ink:#ffffff; --ink-2:#c3c2b7;
  --muted:#898781; --grid:#2c2c2a; --axis:#383835; --border:rgba(255,255,255,.10);
  --est:#3987e5; --truth:#d95926; --lm:#199e70;
  --good:#35b535; --warn:#fab219; --serious:#ec835a; --critical:#e66767;
  --ellipse-fill:rgba(57,135,229,.14);
}
* { box-sizing: border-box; }
body {
  margin:0; background:var(--page); color:var(--ink);
  font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif;
}
header { padding:16px 20px 8px; }
h1 { font-size:18px; margin:0 0 2px; font-weight:650; }
.sub { color:var(--ink-2); font-size:12.5px; }
.wrap { padding:8px 20px 24px; }
.panel {
  background:var(--surface); border:1px solid var(--border);
  border-radius:10px; padding:12px 14px; margin-top:12px;
}
.controls { display:flex; flex-wrap:wrap; gap:10px 14px; align-items:center; }
.controls .group { display:flex; align-items:center; gap:6px; }
button {
  font:inherit; color:var(--ink); background:var(--surface);
  border:1px solid var(--axis); border-radius:7px; padding:5px 11px; cursor:pointer;
}
button:hover { border-color:var(--muted); }
button:disabled { opacity:.4; cursor:default; }
input[type=range] { flex:1 1 320px; accent-color:var(--est); min-width:180px; }
label.toggle { color:var(--ink-2); display:inline-flex; gap:5px; align-items:center; user-select:none; }
label.toggle input { accent-color:var(--est); }
select { font:inherit; background:var(--surface); color:var(--ink); border:1px solid var(--axis); border-radius:6px; padding:3px 6px; }
.spacer { flex:1; }
.badge {
  display:inline-flex; align-items:center; gap:6px; font-weight:600;
  border-radius:999px; padding:3px 12px; font-size:12.5px; border:1px solid var(--border);
}
.badge .dot { width:9px; height:9px; border-radius:50%; background:var(--good); }
.badge.st-warn .dot{background:var(--warn);} .badge.st-warn{color:var(--warn);}
.badge.st-div .dot{background:var(--serious);} .badge.st-div{color:var(--serious);}
.badge.st-kid .dot{background:var(--critical);} .badge.st-kid{color:var(--critical);}
.readout { display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:6px 18px; margin-top:10px; }
.readout .item { min-width:0; }
.readout .k { color:var(--muted); font-size:11.5px; }
.readout .v { font-variant-numeric:tabular-nums; font-weight:600; }
.legend { display:flex; gap:16px; flex-wrap:wrap; color:var(--ink-2); font-size:12.5px; margin-top:8px; }
.legend .key { display:inline-flex; gap:6px; align-items:center; }
.legend .sw { width:18px; height:0; border-top:2px solid; }
.legend .swd { width:10px; height:10px; border-radius:2px; border:2px solid var(--surface); }
.canvas-box { position:relative; width:100%; }
canvas { display:block; width:100%; }
.tip {
  position:absolute; pointer-events:none; display:none; z-index:5;
  background:var(--surface); border:1px solid var(--axis); border-radius:7px;
  padding:6px 9px; font-size:12px; color:var(--ink);
  box-shadow:0 3px 14px rgba(0,0,0,.18); font-variant-numeric:tabular-nums; white-space:nowrap;
}
.event-log { max-height:120px; overflow:auto; font-size:12.5px; margin-top:8px; border-top:1px solid var(--grid); }
.event-log .row { display:flex; gap:10px; padding:4px 2px; border-bottom:1px solid var(--grid); cursor:pointer; }
.event-log .row:hover { background:var(--page); }
.event-log .et { color:var(--muted); font-variant-numeric:tabular-nums; min-width:64px; }
.pill { font-weight:600; min-width:52px; }
.pill.p1{color:var(--warn);} .pill.p2{color:var(--serious);} .pill.p3{color:var(--critical);} .pill.p4{color:var(--good);}
.note { color:var(--muted); font-size:12px; }
</style>
</head>
<body>
<header>
  <h1>EKF 平面位姿定位回放 <span class="note" id="src"></span></h1>
  <div class="sub">状态 (x, y, θ) · 里程计增量预测 + 距离-方位观测更新 · 单文件离线报告（数据与逻辑内嵌）</div>
</header>
<div class="wrap">

  <section class="panel">
    <div class="controls">
      <div class="group">
        <button id="btn-first" title="回到开头">⏮</button>
        <button id="btn-prev" title="上一步 (←)">◀</button>
        <button id="btn-play" title="播放/暂停 (空格)">▶</button>
        <button id="btn-next" title="下一步 (→)">▶</button>
        <button id="btn-last" title="跳到末尾">⏭</button>
      </div>
      <div class="group">
        <button id="btn-prev-ev" title="上一个事件">⚠◀</button>
        <button id="btn-next-ev" title="下一个事件">▶⚠</button>
      </div>
      <input type="range" id="slider" min="0" step="1" value="0">
      <span class="v" id="time-label" style="font-variant-numeric:tabular-nums;min-width:96px;text-align:right"></span>
      <select id="speed" title="播放速度">
        <option value="0.25">0.25×</option><option value="1">1×</option>
        <option value="2" selected>2×</option><option value="5">5×</option>
        <option value="15">15×</option>
      </select>
      <button id="btn-theme">🌓 主题</button>
    </div>
    <div class="controls" style="margin-top:8px">
      <label class="toggle"><input type="checkbox" id="tg-est" checked>估计轨迹</label>
      <label class="toggle"><input type="checkbox" id="tg-truth" checked>真值轨迹</label>
      <label class="toggle"><input type="checkbox" id="tg-ellipse" checked>协方差椭圆</label>
      <label class="toggle"><input type="checkbox" id="tg-lm" checked>路标</label>
      <label class="toggle"><input type="checkbox" id="tg-events" checked>事件标注</label>
      <div class="spacer"></div>
      <span class="badge" id="badge"><span class="dot"></span><span id="badge-text">健康</span></span>
    </div>
    <div class="readout">
      <div class="item"><div class="k">位姿 x (m)</div><div class="v" id="r-x">–</div></div>
      <div class="item"><div class="k">位姿 y (m)</div><div class="v" id="r-y">–</div></div>
      <div class="item"><div class="k">航向 θ (°)</div><div class="v" id="r-th">–</div></div>
      <div class="item"><div class="k">σx / σy (m)</div><div class="v" id="r-sxy">–</div></div>
      <div class="item"><div class="k">σθ (°)</div><div class="v" id="r-sth">–</div></div>
      <div class="item"><div class="k">位置误差 (m)</div><div class="v" id="r-exy">–</div></div>
      <div class="item"><div class="k">航向误差 (°)</div><div class="v" id="r-eth">–</div></div>
      <div class="item"><div class="k">本步 NIS</div><div class="v" id="r-nis">–</div></div>
    </div>
  </section>

  <section class="panel">
    <div class="canvas-box"><canvas id="map"></canvas><div class="tip" id="tip-map"></div></div>
    <div class="legend">
      <span class="key"><span class="sw" style="border-color:var(--est)"></span>估计轨迹</span>
      <span class="key"><span class="sw" style="border-color:var(--truth)"></span>真值轨迹</span>
      <span class="key"><span class="swd" style="background:var(--ellipse-fill);border-color:var(--est)"></span>95% 协方差椭圆（x-y）</span>
      <span class="key"><span class="swd" style="background:var(--lm)"></span>已知路标</span>
      <span class="key">▲ <span style="color:var(--serious)">发散</span> / <span style="color:var(--critical)">绑架</span> / <span style="color:var(--good)">恢复</span></span>
    </div>
  </section>

  <section class="panel">
    <div class="canvas-box"><canvas id="charts"></canvas><div class="tip" id="tip-charts"></div></div>
    <div class="legend">
      <span class="key"><span class="sw" style="border-color:var(--est)"></span>位置误差 (m)</span>
      <span class="key"><span class="sw" style="border-color:var(--truth)"></span>航向误差 (°)</span>
      <span class="note">底栏：NIS（虚线为 χ²(2) 95%/99% 门限与绑架门限 27.6）；点击/拖动任意曲线可跳转时刻。</span>
    </div>
    <div class="event-log" id="event-log"></div>
  </section>

  <p class="note">快捷键：空格 播放/暂停 · ←/→ 单步 · Shift+←/→ 跳转事件 · Home/End 开头/末尾。角度统一规范在 (−180°, 180°]，方位残差按最短角差计算。</p>
</div>

<script>
"use strict";
const DATA = __PAYLOAD__;

// ---- 行字段索引（与 Python 端 build_payload 对应）----
const T=0,KIND=1,X=2,Y=3,H=4,P00=5,P01=6,P11=7,P22=8,NIS=9,EXY=10,ETH=11,ST=12,FL=13,LM=14,GX=15,GY=16,GH=17;
const CHI2_95=5.991, CHI2_99=9.210, KID_NIS=27.631;
const STATUS_TXT=["健康","警戒","已发散","疑似绑架"];
const EVENT_TXT={1:"警戒",2:"发散",3:"绑架",4:"恢复"};
const steps=DATA.steps, landmarks=DATA.landmarks, events=DATA.events;

const css=name=>getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const $=id=>document.getElementById(id);
const fmt=(v,d=3)=>v==null? "–" : Number(v).toFixed(d);
const normA=a=>{a=(a+Math.PI)%(2*Math.PI)-Math.PI; if(a<=-Math.PI)a+=2*Math.PI; return a;};

// ---- 全局状态 ----
let idx=0, playing=false, speed=2, rafId=null;
const show={est:true,truth:true,ellipse:true,lm:true,events:true};

// ---- 画布通用 ----
function setupCanvas(cv){
  const dpr=window.devicePixelRatio||1;
  const w=cv.clientWidth, h=cv.clientHeight;
  cv.width=Math.round(w*dpr); cv.height=Math.round(h*dpr);
  const ctx=cv.getContext("2d"); ctx.setTransform(dpr,0,0,dpr,0,0);
  return {ctx,w,h};
}
function cssPx(name,fallback){const v=css(name); return v||fallback;}

// ============================ 地图 ============================
const mapCv=$("map");
function worldBounds(){
  let x0=1e9,y0=1e9,x1=-1e9,y1=-1e9;
  const see=(x,y)=>{ if(!isFinite(x))return; x0=Math.min(x0,x);x1=Math.max(x1,x);y0=Math.min(y0,y);y1=Math.max(y1,y);};
  for(const s of steps){see(s[X],s[Y]); if(s[GX]!=null){see(s[GX],s[GY]);}}
  for(const [,lx,ly] of landmarks) see(lx,ly);
  if(!isFinite(x0)){x0=-1;y0=-1;x1=1;y1=1;}
  return {x0,y0,x1,y1};
}
const BOUNDS=worldBounds();
function computeView(w,h){
  const pad=46;
  let bw=BOUNDS.x1-BOUNDS.x0, bh=BOUNDS.y1-BOUNDS.y0;
  bw=Math.max(bw,1e-6); bh=Math.max(bh,1e-6);
  const k=Math.min((w-2*pad)/bw,(h-2*pad)/bh);
  const cx=(BOUNDS.x0+BOUNDS.x1)/2, cy=(BOUNDS.y0+BOUNDS.y1)/2;
  // 屏幕坐标：y 翻转
  return {
    k, ox:w/2-cx*k, oy:h/2+cy*k,
    sx:x=>w/2+(x-cx)*k, sy:y=>h/2-(y-cy)*k,
  };
}
function niceStep(d){const p=Math.pow(10,Math.floor(Math.log10(d))); const r=d/p;
  return (r<1.5?1:r<3?2:r<7?5:10)*p;}
function drawGrid(ctx,w,h,v){
  const bx0=BOUNDS.x0-1, by0=BOUNDS.y0-1, bx1=BOUNDS.x1+1, by1=BOUNDS.y1+1;
  const worldPerPx=1/v.k;
  const step=niceStep(80*worldPerPx);
  ctx.strokeStyle=css("--grid"); ctx.fillStyle=css("--muted");
  ctx.lineWidth=1; ctx.font="10.5px system-ui"; ctx.textAlign="center"; ctx.textBaseline="top";
  for(let gx=Math.ceil(bx0/step)*step; gx<=bx1; gx+=step){
    const sx=v.sx(gx); ctx.beginPath(); ctx.moveTo(sx,0); ctx.lineTo(sx,h); ctx.stroke();
    ctx.fillText(gx.toFixed(step<1?1:0), sx, h-14);
  }
  ctx.textAlign="left"; ctx.textBaseline="middle";
  for(let gy=Math.ceil(by0/step)*step; gy<=by1; gy+=step){
    const sy=v.sy(gy); ctx.beginPath(); ctx.moveTo(0,sy); ctx.lineTo(w,sy); ctx.stroke();
    ctx.fillText(gy.toFixed(step<1?1:0), 4, sy);
  }
}
function drawTrail(ctx,v,xf,yf,color,upto){
  ctx.strokeStyle=color; ctx.lineWidth=2; ctx.lineJoin="round"; ctx.lineCap="round";
  let started=false;
  ctx.beginPath();
  for(let i=0;i<=upto;i++){const s=steps[i];
    if(s[xf]==null||s[yf]==null){started=false;continue;}
    const sx=v.sx(s[xf]), sy=v.sy(s[yf]);
    if(!started){ctx.moveTo(sx,sy);started=true;} else ctx.lineTo(sx,sy);
  }
  ctx.stroke();
}
function eig2(a,b,d){ // 2x2 symmetric [[a,b],[b,d]]
  const tr=a+d, disc=Math.hypot(a-d,2*b), l1=(tr+disc)/2, l2=(tr-disc)/2;
  let vx,vy;
  if(Math.abs(b)>1e-300){vx=b;vy=l1-a;} else if(Math.abs(a-l1)<Math.abs(d-l1)){vx=1;vy=0;} else {vx=0;vy=1;}
  return [l1,l2,Math.atan2(vy,vx)];
}
function drawEllipse(ctx,v,s){
  const [l1,l2,ang]=eig2(s[P00],s[P01],s[P11]);
  if(l1<=0||l2<=0)return;
  const a=Math.sqrt(l1)*2.4476*v.k, b=Math.sqrt(l2)*2.4476*v.k; // 95% (chi2_2=5.991)
  const cx=v.sx(s[X]), cy=v.sy(s[Y]);
  ctx.save(); ctx.translate(cx,cy); ctx.rotate(-ang);
  ctx.beginPath(); ctx.ellipse(0,0,Math.max(a,0.5),Math.max(b,0.5),0,0,7);
  ctx.fillStyle=css("--ellipse-fill"); ctx.strokeStyle=css("--est"); ctx.lineWidth=2;
  ctx.fill(); ctx.stroke(); ctx.restore();
}
function drawRobot(ctx,v,s){
  const cx=v.sx(s[X]), cy=v.sy(s[Y]), th=s[H], r=9;
  // 航向三角形
  ctx.save(); ctx.translate(cx,cy); ctx.rotate(-th);
  ctx.beginPath(); ctx.moveTo(r+4,0); ctx.lineTo(-r*0.7,r*0.75); ctx.lineTo(-r*0.7,-r*0.75); ctx.closePath();
  ctx.fillStyle=css("--est"); ctx.fill();
  ctx.lineWidth=2; ctx.strokeStyle=css("--surface"); ctx.stroke();
  ctx.restore();
}
function eventSpans(){ // 返回 [[startT,endT or null, code]] 基于事件配对
  const out=[]; let open=null;
  for(const [t,code] of events){
    if(code===2||code===3){ if(open) out.push(open); open=[t,null,code]; }
    else if(code===4){ if(open){open[1]=t; out.push(open); open=null;} }
  }
  if(open){open[1]=DATA.t1; out.push(open);}
  return out;
}
const SPANS=eventSpans();
function stepIndexAt(t){ // 最后一个 t<=目标 的步
  let lo=0,hi=steps.length-1;
  while(lo<hi){const m=(lo+hi+1)>>1; steps[m][T]<=t?lo=m:hi=m-1;} return lo;
}
function drawEvents(ctx,v,upto){
  // 相同时刻附近的事件标签按类型上下错开，避免“绑架/恢复”叠字
  for(const [t,code,detail] of events){
    if(t>steps[upto][T]+1e-9) continue;
    const si=stepIndexAt(t), s=steps[si];
    const sx=v.sx(s[X]), sy=v.sy(s[Y]);
    const color=code===3?css("--critical"):code===2?css("--serious"):code===1?css("--warn"):css("--good");
    const recovered=code===4;
    const dy=recovered?16:-16;
    ctx.save(); ctx.translate(sx,sy+dy);
    ctx.fillStyle=color; ctx.strokeStyle=css("--surface"); ctx.lineWidth=2;
    ctx.beginPath();
    if(recovered){ ctx.moveTo(0,6); ctx.lineTo(6,-5); ctx.lineTo(-6,-5); }
    else { ctx.moveTo(0,-6); ctx.lineTo(6,5); ctx.lineTo(-6,5); }
    ctx.closePath(); ctx.fill(); ctx.stroke();
    if(code!==1){
      ctx.font="600 10.5px system-ui"; ctx.textAlign="center";
      ctx.textBaseline=recovered?"top":"bottom";
      ctx.fillStyle=color;
      ctx.fillText(recovered?"恢复":(code===3?"绑架":"发散"), 0, recovered?8:-8);
    }
    ctx.restore();
  }
}
function drawLandmarks(ctx,v){
  ctx.font="11px system-ui"; ctx.textAlign="left"; ctx.textBaseline="middle";
  for(const [id,lx,ly] of landmarks){
    const sx=v.sx(lx), sy=v.sy(ly);
    ctx.fillStyle=css("--lm");
    ctx.fillRect(sx-5,sy-5,10,10);
    ctx.lineWidth=2; ctx.strokeStyle=css("--surface"); ctx.strokeRect(sx-5,sy-5,10,10);
    ctx.fillStyle=css("--muted"); ctx.fillText(id, sx+8, sy);
  }
}
function renderMap(){
  mapCv.style.height="440px";
  const {ctx,w,h}=setupCanvas(mapCv);
  ctx.clearRect(0,0,w,h);
  const v=computeView(w,h);
  drawGrid(ctx,w,h,v);
  // 异常区间在轨迹上的淡色带：用地面投影难以表现，改为事件三角标注
  const s=steps[idx];
  if(show.truth) drawTrail(ctx,v,GX,GY,css("--truth"),idx);
  if(show.est) drawTrail(ctx,v,X,Y,css("--est"),idx);
  if(show.lm) drawLandmarks(ctx,v);
  if(show.events) drawEvents(ctx,v,idx);
  if(show.ellipse) drawEllipse(ctx,v,s);
  drawRobot(ctx,v,s);
}

// ============================ 误差 / NIS 图（三个面板，共用时间轴）=================
const chCv=$("charts");
const CH={h:[110,90,110], gap:26, top:14};
function chartGeom(w,totalH){
  const panels=[]; let y=CH.top;
  for(const ph of CH.h){panels.push({x:54,y,w:w-64,h:ph-8}); y+=ph+CH.gap;}
  return {panels,totalH:y+16};
}
function panelScale(p,minV,maxV){
  if(maxV-minV<1e-9){maxV=minV+1;}
  return {ty:v=>p.y+p.h-(v-minV)/(maxV-minV)*p.h, minV,maxV};
}
function drawPanelBg(ctx,p,minV,maxV,unit,ylabelColor){
  ctx.strokeStyle=css("--grid"); ctx.fillStyle=css("--muted");
  ctx.lineWidth=1; ctx.font="10.5px system-ui"; ctx.textAlign="right"; ctx.textBaseline="middle";
  const st=niceStep((maxV-minV)/4);
  const lo=Math.ceil(minV/st)*st;
  for(let gv=lo; gv<=maxV+1e-9; gv+=st){
    const ty=p.y+p.h-(gv-minV)/(maxV-minV)*p.h;
    ctx.beginPath(); ctx.moveTo(p.x,ty); ctx.lineTo(p.x+p.w,ty); ctx.stroke();
    ctx.fillText(gv.toFixed(Math.abs(st)<1?1:0), p.x-7, ty);
  }
  // 轴线
  ctx.strokeStyle=css("--axis");
  ctx.beginPath(); ctx.moveTo(p.x,p.y); ctx.lineTo(p.x,p.y+p.h); ctx.lineTo(p.x+p.w,p.y+p.h); ctx.stroke();
  ctx.save(); ctx.translate(13,p.y+p.h/2); ctx.rotate(-Math.PI/2);
  ctx.textAlign="center"; ctx.fillStyle=ylabelColor||css("--ink-2");
  ctx.fillText(unit,0,0); ctx.restore();
}
function drawSpans(ctx,p,ty){
  for(const [t0,t1,code] of SPANS){
    const color=code===3?css("--critical"):css("--serious");
    const a=Math.max(0,t0-DATA.t0), b=Math.max(0,(t1??DATA.t1)-DATA.t0);
    const span=Math.max(DATA.t1-DATA.t0,1e-9);
    const x0=p.x+a/span*p.w, x1=p.x+b/span*p.w;
    ctx.save(); ctx.globalAlpha=.10; ctx.fillStyle=color;
    ctx.fillRect(x0,p.y,Math.max(x1-x0,1),p.h); ctx.restore();
    ctx.save(); ctx.globalAlpha=.55; ctx.strokeStyle=color; ctx.setLineDash([4,3]); ctx.lineWidth=1;
    ctx.beginPath(); ctx.moveTo(x0,p.y); ctx.lineTo(x0,p.y+p.h); ctx.stroke(); ctx.restore();
  }
}
function drawSeries(ctx,p,field,color,minV,maxV,connectNulls){
  const sc=panelScale(p,minV,maxV), span=Math.max(DATA.t1-DATA.t0,1e-9);
  ctx.strokeStyle=color; ctx.lineWidth=2; ctx.lineJoin="round"; ctx.lineCap="round";
  let started=false; ctx.beginPath();
  for(const s of steps){
    const val=s[field];
    if(val==null || !isFinite(val)){ if(!connectNulls)started=false; continue;}
    const cv=Math.max(minV,Math.min(maxV,val));
    const x=p.x+(s[T]-DATA.t0)/span*p.w, y=sc.ty(cv);
    if(!started){ctx.moveTo(x,y);started=true;} else ctx.lineTo(x,y);
  }
  ctx.stroke();
}
function drawNISPoints(ctx,p,minV,maxV){
  const sc=panelScale(p,minV,maxV), span=Math.max(DATA.t1-DATA.t0,1e-9);
  let n=0;
  for(const s of steps){ if(s[NIS]==null)continue;
    // 正常 NIS 已由连线表达；这里只突出超 99% 门限的点，避免近两千个点糊成一片
    if(s[NIS]<=CHI2_99) continue;
    const x=p.x+(s[T]-DATA.t0)/span*p.w, y=sc.ty(Math.max(minV,Math.min(maxV,s[NIS])));
    ctx.beginPath(); ctx.arc(x,y,s[NIS]>KID_NIS?3.8:2.8,0,7);
    ctx.fillStyle=s[NIS]>KID_NIS?css("--critical"):css("--warn"); ctx.fill();
    ctx.lineWidth=1.5; ctx.strokeStyle=css("--surface"); ctx.stroke();
    n++;
  }
}
function drawThreshold(ctx,p,val,minV,maxV,txt,dashColor,side){
  const sc=panelScale(p,minV,maxV);
  if(val<minV||val>maxV)return;
  const y=sc.ty(val);
  ctx.save(); ctx.setLineDash([5,4]); ctx.strokeStyle=dashColor||css("--muted"); ctx.lineWidth=1;
  ctx.beginPath(); ctx.moveTo(p.x,y); ctx.lineTo(p.x+p.w,y); ctx.stroke(); ctx.restore();
  ctx.fillStyle=dashColor||css("--muted"); ctx.font="9.5px system-ui";
  ctx.textBaseline="bottom";
  ctx.textAlign = side==="left" ? "left" : "right";
  ctx.fillText(txt, side==="left" ? p.x+3 : p.x+p.w-2, y-1);
}
function drawPlayhead(ctx,p){
  const span=Math.max(DATA.t1-DATA.t0,1e-9);
  const x=p.x+(steps[idx][T]-DATA.t0)/span*p.w;
  ctx.strokeStyle=css("--ink"); ctx.lineWidth=1.5;
  ctx.beginPath(); ctx.moveTo(x,p.y-4); ctx.lineTo(x,p.y+p.h); ctx.stroke();
}
function renderCharts(){
  const w=chCv.clientWidth;
  const g=chartGeom(w,0);
  const totalH=g.totalH;
  chCv.style.height=totalH+"px";
  const {ctx}=setupCanvas(chCv);
  const [p1,p2,p3]=g.panels;
  ctx.clearRect(0,0,w,totalH);

  // 值域
  const exyVals=steps.map(s=>s[EXY]).filter(v=>v!=null&&isFinite(v));
  const ethVals=steps.map(s=>s=>s[ETH]).filter(v=>v!=null&&isFinite(v)).map(Math.abs);
  const nisVals=steps.map(s=>s[NIS]).filter(v=>v!=null&&isFinite(v));
  const maxEXY=Math.max(0.2, ...exyVals)*1.08;
  const maxETH=Math.max(5, ...ethVals)*1.08;
  // NIS 面板线性、可视上限裁剪，保证门限可见
  const nisCap=Math.max(KID_NIS*1.15, percentile(nisVals,.98)*1.1);

  drawPanelBg(ctx,p1,0,maxEXY,"位置误差 (m)",css("--est"));
  drawPanelBg(ctx,p2,0,maxETH,"|航向误差| (°)",css("--truth"));
  drawPanelBg(ctx,p3,0,nisCap,"NIS",css("--ink-2"));
  for(const p of g.panels) drawSpans(ctx,p,null);
  drawSeries(ctx,p1,EXY,css("--est"),0,maxEXY,true);
  drawHeadingAbs(ctx,p2,0,maxETH);
  drawSeries(ctx,p3,NIS,css("--lm"),0,nisCap,true);
  drawNISPoints(ctx,p3,0,nisCap);
  drawThreshold(ctx,p3,CHI2_95,0,nisCap,"95% 5.99",css("--muted"),"left");
  drawThreshold(ctx,p3,CHI2_99,0,nisCap,"99% 9.21",css("--warn"),"right");
  drawThreshold(ctx,p3,KID_NIS,0,nisCap,"绑架 27.6",css("--critical"),"right");
  for(const p of g.panels) drawPlayhead(ctx,p);
  drawTimeAxis(ctx,p1,p2,p3);
}
function drawHeadingAbs(ctx,p,minV,maxV){
  const sc=panelScale(p,minV,maxV), span=Math.max(DATA.t1-DATA.t0,1e-9);
  ctx.strokeStyle=css("--truth"); ctx.lineWidth=2;
  let started=false; ctx.beginPath();
  for(const s of steps){ const val=s[ETH];
    if(val==null||!isFinite(val)){started=false;continue;}
    const x=p.x+(s[T]-DATA.t0)/span*p.w, y=sc.ty(Math.abs(val));
    if(!started){ctx.moveTo(x,y);started=true;} else ctx.lineTo(x,y);
  }
  ctx.stroke();
}
function drawTimeAxis(ctx,p1,p2,p3){
  const p=p3, span=Math.max(DATA.t1-DATA.t0,1e-9);
  const y=p.y+p.h+16, step=niceStep(span/10);
  ctx.strokeStyle=css("--axis"); ctx.fillStyle=css("--muted");
  ctx.lineWidth=1; ctx.font="10.5px system-ui"; ctx.textAlign="center"; ctx.textBaseline="top";
  ctx.beginPath(); ctx.moveTo(p.x,y-8); ctx.lineTo(p.x+p.w,y-8); ctx.stroke();
  const ticks=[];
  for(let t=Math.ceil(DATA.t0/step)*step; t<=DATA.t1+1e-9; t+=step)
    ticks.push([p.x+(t-DATA.t0)/span*p.w,t]);
  ticks.forEach(([x,t],i)=>{
    ctx.beginPath(); ctx.moveTo(x,y-8); ctx.lineTo(x,y-4); ctx.stroke();
    // 首尾标签向内对齐，其余居中，避免被画布边缘裁切
    ctx.textAlign = i===0?"left" : i===ticks.length-1?"right":"center";
    ctx.fillText(t.toFixed(1)+"s",x,y-2);
  });
  ctx.textAlign="center";
}
function percentile(arr,q){ if(!arr.length)return 0;
  const a=[...arr].sort((x,y)=>x-y); return a[Math.min(a.length-1,Math.floor(a.length*q))];}

// ---- 图上悬停/点击跳转 ----
function chartXToTime(clientX){
  const r=chCv.getBoundingClientRect();
  const g=chartGeom(r.width,0), p=g.panels[0];
  const frac=(clientX-r.left-p.x)/p.w;
  if(frac<0||frac>1)return null;
  return DATA.t0+frac*Math.max(DATA.t1-DATA.t0,1e-9);
}
function seekTime(t){ setIdx(stepIndexAt(t)); }
chCv.addEventListener("click",e=>{const t=chartXToTime(e.clientX); if(t!=null)seekTime(t);});
let dragChart=false;
chCv.addEventListener("mousedown",()=>dragChart=true);
window.addEventListener("mouseup",()=>dragChart=false);
chCv.addEventListener("mousemove",e=>{
  const tip=$("tip-charts");
  if(dragChart){const t=chartXToTime(e.clientX); if(t!=null)seekTime(t);}
  const t=chartXToTime(e.clientX);
  if(t==null){tip.style.display="none"; return;}
  const si=stepIndexAt(t), s=steps[si];
  tip.style.display="block";
  tip.style.left=(e.clientX-chCv.getBoundingClientRect().left+12)+"px";
  tip.style.top=(e.clientY-chCv.getBoundingClientRect().top+8)+"px";
  tip.innerHTML=`t=${s[T].toFixed(2)}s<br>位置误差 ${fmt(s[EXY])} m · 航向 ${fmt(s[ETH],1)}°<br>NIS ${s[NIS]==null?"–":fmt(s[NIS],2)}${s[FL]&2?' &nbsp;<b style="color:'+css("--critical")+'">重置</b>':""}`;
});
chCv.addEventListener("mouseleave",()=>$("tip-charts").style.display="none");
mapCv.addEventListener("mousemove",e=>{
  const r=mapCv.getBoundingClientRect();
  const v=computeView(r.width,r.height); const mx=e.clientX-r.left, my=e.clientY-r.top;
  const s=steps[idx];
  const rx=v.sx(s[X]), ry=v.sy(s[Y]);
  const tip=$("tip-map");
  if(Math.hypot(mx-rx,my-ry)<14){
    tip.style.display="block"; tip.style.left=(mx+12)+"px"; tip.style.top=(my+10)+"px";
    tip.innerHTML=`估计位姿<br>x=${fmt(s[X])} y=${fmt(s[Y])} θ=${fmt(s[H]*180/Math.PI,1)}°<br>σx=${fmt(Math.sqrt(Math.max(s[P00],0)))} σy=${fmt(Math.sqrt(Math.max(s[P11],0)))} σθ=${fmt(Math.sqrt(Math.max(s[P22],0))*180/Math.PI,1)}°`;
  } else {
    let hit=null;
    for(const [id,lx,ly] of landmarks){const sx=v.sx(lx),sy=v.sy(ly);
      if(Math.abs(mx-sx)<9&&Math.abs(my-sy)<9){hit=[id,lx,ly];break;}}
    if(hit){tip.style.display="block";tip.style.left=(mx+12)+"px";tip.style.top=(my+10)+"px";
      tip.innerHTML=`路标 ${hit[0]}<br>x=${fmt(hit[1])} y=${fmt(hit[2])}`;}
    else tip.style.display="none";
  }
});
mapCv.addEventListener("mouseleave",()=>$("tip-map").style.display="none");

// ============================ 回放控制 ============================
const slider=$("slider"); slider.max=steps.length-1;
function setBadge(code){
  const b=$("badge"); b.className="badge "+(["","st-warn","st-div","st-kid"][code]||"");
  $("badge-text").textContent=STATUS_TXT[code];
}
function setIdx(i){
  idx=Math.max(0,Math.min(steps.length-1,i));
  slider.value=idx;
  const s=steps[idx];
  $("time-label").textContent=s[T].toFixed(2)+" / "+DATA.t1.toFixed(2)+" s";
  $("r-x").textContent=fmt(s[X]); $("r-y").textContent=fmt(s[Y]);
  $("r-th").textContent=fmt(s[H]*180/Math.PI,1);
  $("r-sxy").textContent=fmt(Math.sqrt(Math.max(s[P00],0)))+" / "+fmt(Math.sqrt(Math.max(s[P11],0)));
  $("r-sth").textContent=fmt(Math.sqrt(Math.max(s[P22],0))*180/Math.PI,1);
  $("r-exy").textContent=fmt(s[EXY]); $("r-eth").textContent=fmt(s[ETH],1);
  $("r-nis").textContent=s[NIS]==null?"–":fmt(s[NIS],2)+(s[FL]&1?"  (野值降权)":"")+(s[FL]&2?"  ⚠重置":"");
  setBadge(s[ST]);
  renderMap(); renderCharts();
}
function togglePlay(){
  playing=!playing; $("btn-play").textContent=playing?"⏸":"▶";
  if(playing){ if(idx>=steps.length-1) setIdx(0); let last=performance.now();
    const loop=now=>{ if(!playing)return;
      const dt=(now-last)/1000; last=now;
      if(dt>0){ const sim=dt*speed; // 1 秒仿真对应 1 秒墙钟 × speed
        let j=idx; while(j<steps.length-1 && steps[j+1][T]-steps[idx][T]<sim) j++;
        if(j===idx) j=Math.min(idx+1,steps.length-1);
        setIdx(j);
        if(idx>=steps.length-1){togglePlay();return;}
      }
      rafId=requestAnimationFrame(loop);
    }; rafId=requestAnimationFrame(loop);
  }
}
$("btn-play").onclick=togglePlay;
$("btn-first").onclick=()=>setIdx(0);
$("btn-last").onclick=()=>setIdx(steps.length-1);
$("btn-prev").onclick=()=>setIdx(idx-1);
$("btn-next").onclick=()=>setIdx(idx+1);
slider.addEventListener("input",()=>setIdx(+slider.value));
$("speed").onchange=e=>speed=+e.target.value;
function jumpEvent(dir){
  const cur=steps[idx][T];
  let cand=events.map(e=>e[0]).filter(t=>dir>0?t>cur+1e-9:t<cur-1e-9).sort((a,b)=>dir>0?a-b:b-a);
  if(cand.length) seekTime(cand[0]);
}
$("btn-next-ev").onclick=()=>jumpEvent(1);
$("btn-prev-ev").onclick=()=>jumpEvent(-1);
for(const [id,key] of [["tg-est","est"],["tg-truth","truth"],["tg-ellipse","ellipse"],["tg-lm","lm"],["tg-events","events"]])
  $(id).onchange=e=>{show[key]=e.target.checked; renderMap();};
$("btn-theme").onclick=()=>{
  const root=document.documentElement;
  const darkNow = root.dataset.theme==="dark" ||
    (!root.dataset.theme && matchMedia("(prefers-color-scheme: dark)").matches);
  root.dataset.theme = darkNow ? "light" : "dark";
  renderMap(); renderCharts();
};
window.addEventListener("keydown",e=>{
  if(e.target.tagName==="INPUT"||e.target.tagName==="SELECT")return;
  if(e.code==="Space"){e.preventDefault();togglePlay();}
  else if(e.code==="ArrowLeft"){e.preventDefault(); e.shiftKey?jumpEvent(-1):setIdx(idx-1);}
  else if(e.code==="ArrowRight"){e.preventDefault(); e.shiftKey?jumpEvent(1):setIdx(idx+1);}
  else if(e.code==="Home")setIdx(0); else if(e.code==="End")setIdx(steps.length-1);
});

// 事件列表
const log=$("event-log");
if(!events.length){ log.innerHTML='<div class="row note">全程健康：未触发发散或绑架判定。</div>'; }
else for(const [t,code,detail] of events){
  const d=document.createElement("div"); d.className="row";
  d.innerHTML=`<span class="et">t=${t.toFixed(2)}s</span><span class="pill p${code}">${EVENT_TXT[code]||""}</span><span>${detail||""}</span>`;
  d.onclick=()=>seekTime(t); log.appendChild(d);
}
if(DATA.source) $("src").textContent="· "+DATA.source;

let rzT=null;
window.addEventListener("resize",()=>{clearTimeout(rzT);rzT=setTimeout(()=>{renderMap();renderCharts();},80);});
setIdx(0);
</script>
</body>
</html>
"""
