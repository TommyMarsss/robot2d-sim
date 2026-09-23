"""传感器日志格式。

日志采用 **JSON Lines**（每行一个 JSON 对象，UTF-8），字段含义：

首行（可选，建议有）``meta``
    ``{"type": "meta", "landmarks": {"lm1": [x, y], ...},
       "initial_state": [x, y, theta], "initial_cov": [[...3x3...]],
       "alphas": [a1, a2, a3, a4], "meas_var": [sigma_r^2, sigma_phi^2]}``

数据行
    - 里程计增量：``{"type": "odom", "t": 0.10, "d_trans": 0.05, "d_rot": 0.01}``
    - 路标观测：``{"type": "obs", "t": 0.15, "id": "lm1",
      "range": 2.31, "bearing": -0.42}``
    - 真值（可选，仅用于回放对比/测试）：
      ``{"type": "truth", "t": 0.10, "x": 0.1, "y": 0.0, "theta": 0.02}``

约定：

- 长度单位米，角度单位**弧度**，方位 ``bearing`` 为相对机器人航向的夹角；
- 时间戳 ``t`` 为秒，**允许乱序与重复**；重复时间戳按文件中的原始先后顺序处理；
- 观测的 ``id`` 必须能在 ``meta.landmarks`` 中找到（数据关联已知）。
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple, Union

from .angles import normalize_angle

# 默认噪声参数（可被 meta 覆盖）
DEFAULT_ALPHAS = (0.08, 0.04, 0.06, 0.05)
DEFAULT_MEAS_VAR = (0.02, 0.005)  # 距离方差 m^2，方位方差 rad^2
DEFAULT_INITIAL_COV = (0.25, 0.25, 0.09)  # sigma_x^2, sigma_y^2, sigma_th^2


@dataclass(frozen=True)
class Landmark:
    id: str
    x: float
    y: float

    def xy(self) -> Tuple[float, float]:
        return self.x, self.y


@dataclass(frozen=True)
class OdometryIncrement:
    t: float
    d_trans: float
    d_rot: float
    seq: int = -1  # 文件中的原始序号，用于稳定排序与重复时间戳


@dataclass(frozen=True)
class Observation:
    t: float
    id: str
    rng: float
    bearing: float
    seq: int = -1


@dataclass(frozen=True)
class TruthPose:
    t: float
    x: float
    y: float
    theta: float
    seq: int = -1


@dataclass
class LogBundle:
    landmarks: Dict[str, Landmark]
    odometry: List[OdometryIncrement]
    observations: List[Observation]
    truth: List[TruthPose] = field(default_factory=list)
    initial_state: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    initial_cov_diag: Tuple[float, float, float] = DEFAULT_INITIAL_COV
    alphas: Tuple[float, float, float, float] = DEFAULT_ALPHAS
    meas_var: Tuple[float, float] = DEFAULT_MEAS_VAR
    source: str = ""

    def time_span(self) -> Tuple[float, float]:
        ts = [e.t for e in self.odometry + self.observations]
        return (min(ts), max(ts)) if ts else (0.0, 0.0)


def _stable_sort_odom(events: List) -> List:
    # 同类型同刻按文件原始顺序（稳定排序）。
    return sorted(events, key=lambda e: (e.t, e.seq))


def _stable_sort_obs(events: List) -> List:
    # 观测按 (时间, 路标 id, 文件序)：同一时刻的多路标观测顺序与文件排列
    # 无关，保证乱序日志产生逐位一致的处理结果。
    return sorted(events, key=lambda e: (e.t, e.id, e.seq))


def parse_log(
    source: Union[str, Path, Iterable[str], Sequence[dict]],
) -> LogBundle:
    """解析 JSON Lines 日志。

    接受文件路径、行迭代器，或已解析好的 dict 列表。
    时间戳乱序/重复时做稳定排序，重复时间戳保持原始先后。
    """
    if isinstance(source, (str, Path)) and _looks_like_path(source):
        lines: Iterable[str] = Path(source).read_text(encoding="utf-8").splitlines()
        source_name = str(source)
    elif isinstance(source, (str, Path)):
        lines = str(source).splitlines()
        source_name = "<string>"
    else:
        lines = source  # type: ignore[assignment]
        source_name = "<objects>"

    landmarks: Dict[str, Landmark] = {}
    odom: List[OdometryIncrement] = []
    obs: List[Observation] = []
    truth: List[TruthPose] = []
    initial_state: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    initial_cov_diag = DEFAULT_INITIAL_COV
    alphas = DEFAULT_ALPHAS
    meas_var = DEFAULT_MEAS_VAR
    seq = 0

    for lineno, item in enumerate(_iter_objects(lines), start=1):
        etype = item.get("type")
        if etype == "meta":
            for lm_id, xy in item.get("landmarks", {}).items():
                landmarks[str(lm_id)] = Landmark(str(lm_id), float(xy[0]), float(xy[1]))
            if "initial_state" in item:
                x0, y0, t0 = item["initial_state"]
                initial_state = (float(x0), float(y0), normalize_angle(float(t0)))
            if "initial_cov" in item:
                c = item["initial_cov"]
                initial_cov_diag = (
                    float(c[0][0]),
                    float(c[1][1]),
                    float(c[2][2]),
                )
            elif "initial_cov_diag" in item:
                d = item["initial_cov_diag"]
                initial_cov_diag = (float(d[0]), float(d[1]), float(d[2]))
            if "alphas" in item:
                a = item["alphas"]
                alphas = tuple(float(v) for v in a)  # type: ignore[assignment]
            if "meas_var" in item:
                m = item["meas_var"]
                meas_var = (float(m[0]), float(m[1]))
        elif etype == "odom":
            odom.append(
                OdometryIncrement(
                    t=float(item["t"]),
                    d_trans=float(item["d_trans"]),
                    d_rot=normalize_angle(float(item["d_rot"])),
                    seq=seq,
                )
            )
            seq += 1
        elif etype == "obs":
            obs.append(
                Observation(
                    t=float(item["t"]),
                    id=str(item["id"]),
                    rng=float(item["range"]),
                    bearing=normalize_angle(float(item["bearing"])),
                    seq=seq,
                )
            )
            seq += 1
        elif etype == "truth":
            truth.append(
                TruthPose(
                    t=float(item["t"]),
                    x=float(item["x"]),
                    y=float(item["y"]),
                    theta=normalize_angle(float(item["theta"])),
                    seq=seq,
                )
            )
            seq += 1
        else:
            raise ValueError(f"第 {lineno} 行存在未知记录类型: {etype!r}")

    for o in obs:
        if o.id not in landmarks:
            raise ValueError(f"观测引用了未定义的路标 id={o.id!r}（t={o.t}）")
    if not math.isfinite(meas_var[0]) or meas_var[0] <= 0 or meas_var[1] <= 0:
        raise ValueError("meas_var 必须为正数")

    return LogBundle(
        landmarks=landmarks,
        odometry=_stable_sort_odom(odom),
        observations=_stable_sort_obs(obs),
        truth=sorted(truth, key=lambda e: (e.t, e.seq)),
        initial_state=initial_state,
        initial_cov_diag=tuple(max(v, 1e-12) for v in initial_cov_diag),  # type: ignore[arg-type]
        alphas=alphas,
        meas_var=meas_var,
        source=source_name,
    )


def _looks_like_path(s: Union[str, Path]) -> bool:
    if isinstance(s, Path):
        return True
    s = str(s)
    if "\n" in s:
        return False
    try:
        return Path(s).is_file()
    except OSError:
        return False


def _iter_objects(lines: Iterable) -> Iterable[dict]:
    for item in lines:
        if isinstance(item, dict):
            yield item
            continue
        line = item.strip()
        if not line or line.startswith("#"):
            continue
        obj = json.loads(line)
        if not isinstance(obj, dict):
            raise ValueError("每行必须是 JSON 对象")
        yield obj


def dump_jsonl(bundle: LogBundle) -> str:
    """把 :class:`LogBundle` 序列化回 JSON Lines（主要供测试使用）。"""
    meta = {
        "type": "meta",
        "landmarks": {k: [v.x, v.y] for k, v in bundle.landmarks.items()},
        "initial_state": list(bundle.initial_state),
        "initial_cov_diag": list(bundle.initial_cov_diag),
        "alphas": list(bundle.alphas),
        "meas_var": list(bundle.meas_var),
    }
    out = [json.dumps(meta, ensure_ascii=False)]
    events: List[Tuple[float, int, dict]] = []
    for o in bundle.odometry:
        events.append(
            (o.t, o.seq, {"type": "odom", "t": o.t, "d_trans": o.d_trans, "d_rot": o.d_rot})
        )
    for o in bundle.observations:
        events.append(
            (o.t, o.seq, {"type": "obs", "t": o.t, "id": o.id, "range": o.rng, "bearing": o.bearing})
        )
    for g in bundle.truth:
        events.append(
            (g.t, -1, {"type": "truth", "t": g.t, "x": g.x, "y": g.y, "theta": g.theta})
        )
    for _, _, e in sorted(events, key=lambda z: (z[0], z[1])):
        out.append(json.dumps(e, ensure_ascii=False))
    return "\n".join(out) + "\n"
