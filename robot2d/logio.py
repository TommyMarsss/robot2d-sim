"""传感器日志解析与滤波驱动。

日志格式：JSON Lines，每行一个事件，字段：

  {"type": "config", "landmarks": {"0": [x, y], ...}}   # 可选，须在使用前出现
  {"type": "init", "t": 0.0, "x": .., "y": .., "theta": ..}  # 可选初始位姿
  {"type": "odom", "t": 1.0, "d": 0.5, "dtheta": 0.02}       # 里程计增量
  {"type": "obs",  "t": 1.0, "landmark": "3", "range": 4.2, "bearing": -0.3}
  {"type": "truth", "t": 1.0, "x": .., "y": .., "theta": ..} # 可选真值（仅用于报告）

时间戳允许乱序与重复：解析后按 (t, 原始序号) 稳定排序；
完全重复的事件（类型与载荷均相同）只保留第一条。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Iterable, Iterator

from .ekf import EKFLocalizer, wrap_angle


@dataclass(frozen=True)
class LogEvent:
    seq: int
    t: float
    type: str               # "odom" | "obs" | "truth"
    data: tuple             # odom: (d, dtheta); obs: (landmark, range, bearing); truth: (x, y, theta)


@dataclass
class ParsedLog:
    landmarks: dict[str, tuple[float, float]] = field(default_factory=dict)
    init_pose: tuple[float, float, float] | None = None
    events: list[LogEvent] = field(default_factory=list)  # 已排序、去重


def parse_log(lines: Iterable[str]) -> ParsedLog:
    """解析 JSONL 日志，返回按时间排序、去重后的事件序列。"""
    parsed = ParsedLog()
    raw: list[LogEvent] = []
    for seq, line in enumerate(lines):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        rec = json.loads(line)
        typ = rec.get("type")
        if typ == "config":
            parsed.landmarks.update(
                {str(k): (float(v[0]), float(v[1])) for k, v in rec["landmarks"].items()})
        elif typ == "init":
            parsed.init_pose = (float(rec["x"]), float(rec["y"]), float(rec["theta"]))
        elif typ == "odom":
            raw.append(LogEvent(seq, float(rec["t"]), "odom",
                                (float(rec["d"]), float(rec["dtheta"]))))
        elif typ == "obs":
            raw.append(LogEvent(seq, float(rec["t"]), "obs",
                                (str(rec["landmark"]), float(rec["range"]),
                                 float(rec["bearing"]))))
        elif typ == "truth":
            raw.append(LogEvent(seq, float(rec["t"]), "truth",
                                (float(rec["x"]), float(rec["y"]), float(rec["theta"]))))
        else:
            raise ValueError(f"未知事件类型: {typ!r} (第 {seq + 1} 行)")

    # 排序：先按时间戳；同刻事件按类型固定优先级（先运动后观测再真值），
    # 再按载荷内容排序，保证乱序日志重排后的处理顺序与原始顺序完全无关。
    type_order = {"odom": 0, "obs": 1, "truth": 2}
    raw.sort(key=lambda e: (e.t, type_order[e.type], e.data))

    # 去重：类型与载荷完全相同的重复事件只保留第一条
    seen: set[tuple] = set()
    for ev in raw:
        key = (ev.t, ev.type, ev.data)
        if key in seen:
            continue
        seen.add(key)
        parsed.events.append(ev)
    return parsed


@dataclass
class HistoryEntry:
    t: float
    x: float
    y: float
    theta: float
    P: list[list[float]]
    nis: float | None
    diverged: bool
    truth: tuple[float, float, float] | None


def run_filter(parsed: ParsedLog,
               ekf: EKFLocalizer | None = None) -> tuple[EKFLocalizer, list[HistoryEntry]]:
    """对解析后的日志运行 EKF，返回滤波器与逐步历史（供报告与测试使用）。"""
    if ekf is None:
        init = parsed.init_pose or (0.0, 0.0, 0.0)
        ekf = EKFLocalizer(parsed.landmarks, x0=init)

    history: list[HistoryEntry] = []
    truth: tuple[float, float, float] | None = None

    def snapshot(t: float, nis: float | None) -> None:
        history.append(HistoryEntry(
            t=t, x=ekf.x[0], y=ekf.x[1], theta=ekf.x[2],
            P=[row[:] for row in ekf.P], nis=nis,
            diverged=ekf.monitor.diverged, truth=truth))

    for ev in parsed.events:
        if ev.type == "odom":
            ekf.predict(ev.data[0], ev.data[1])
            snapshot(ev.t, None)
        elif ev.type == "obs":
            nis = ekf.update(ev.data[0], ev.data[1], ev.data[2], t=ev.t)
            snapshot(ev.t, nis)
        elif ev.type == "truth":
            truth = (ev.data[0], ev.data[1], wrap_angle(ev.data[2]))
            # 真值只记录，不影响滤波；附到最近一条历史或单独记录
            if history:
                history[-1].truth = truth
    return ekf, history
