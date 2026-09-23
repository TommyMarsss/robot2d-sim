#!/usr/bin/env python3
"""生成固定的传感器日志（含真值），供 EKF 回放与测试使用。

场景：
- healthy   : 匀速转弯绕圈，环道内外均有路标；里程计/观测带高斯噪声，
              全程应基本健康，仅偶有单点野值；
- kidnapped : 行驶中途被“搬运”到别处并转头（里程计无感知），
              搬运后附近另有一组路标，滤波器应判绑架、重定位并恢复；
- diverged  : 路标只在起点附近与绕圈终点附近可见；中途有约 20s 观测空窗，
              里程计存在未建模的恒定航向偏置；重新看到路标时新息持续偏大
              -> 判发散 -> 协方差自适应膨胀后重新收敛恢复。

用法::

    python tools/generate_logs.py [--outdir data] [--seed 20260922]

只用标准库（random 的高斯采样）。
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ekf_localization.angles import normalize_angle

MEAS_VAR = [0.02, 0.005]
DT = 0.1
V = 0.5                 # m/s
OMEGA = 0.1             # rad/s -> 转弯半径 5 m
SIGMA_R = math.sqrt(MEAS_VAR[0])
SIGMA_PHI = math.sqrt(MEAS_VAR[1])

SCENARIO_ALPHAS = {
    # 发散场景：滤波器相信的过程噪声很小，未建模偏置会快速吃掉协方差裕度
    "healthy": [0.06, 0.03, 0.05, 0.04],
    "kidnapped": [0.06, 0.03, 0.05, 0.04],
    "diverged": [0.012, 0.006, 0.010, 0.008],
}
SCENARIO_BIAS = {"healthy": 0.0, "kidnapped": 0.0, "diverged": 0.0042}
SCENARIO_RANGE = {"healthy": 11.0, "kidnapped": 9.0, "diverged": 6.0}
# 健康/绑架场景匀速转弯绕圈；发散场景直行，穿过两个路标簇之间的观测空窗
SCENARIO_OMEGA = {"healthy": 0.1, "kidnapped": 0.1, "diverged": 0.0}
SCENARIO_STEPS = {"healthy": 700, "kidnapped": 700, "diverged": 520}
# 固定盐值（不能用内置 hash()，它默认每个进程加盐导致日志不可复现）
SCENARIO_SALT = {"healthy": 101, "kidnapped": 202, "diverged": 303}


def _landmarks(scenario: str) -> Dict[str, Tuple[float, float]]:
    if scenario == "healthy":
        return {
            f"lm{i+1}": (
                9.0 * math.cos(2 * math.pi * i / 6 + 0.3),
                9.0 * math.sin(2 * math.pi * i / 6 + 0.3),
            )
            for i in range(6)
        }
    start_cluster = {
        "lm1": (2.0, 2.0), "lm2": (-2.5, 1.5), "lm3": (0.5, -3.0),
        "lm4": (3.5, -0.5), "lm5": (-1.0, 4.0),
    }
    if scenario == "diverged":
        # 起点簇 + 远端簇（x≈18~23），中间形成约 10s 的观测空窗
        start_cluster.update(
            {"lm6": (18.0, 2.5), "lm7": (21.0, -2.0), "lm8": (23.5, 1.5)}
        )
        return start_cluster
    # kidnapped：搬运落点 (~7.8, 5.2) 附近再布一组路标
    start_cluster.update(
        {"lm6": (6.5, 5.5), "lm7": (10.5, 4.0), "lm8": (8.5, 1.0)}
    )
    return start_cluster


def _odom_noise(rng: random.Random, alphas: List[float], d_trans: float, d_rot: float):
    st = math.sqrt(max(alphas[0] * d_trans**2 + alphas[1] * d_rot**2, 1e-12))
    sr = math.sqrt(max(alphas[2] * d_trans**2 + alphas[3] * d_rot**2, 1e-14))
    return rng.gauss(0.0, st), rng.gauss(0.0, sr)


def simulate(
    scenario: str, rng: random.Random, n_steps: Optional[int] = None, obs_every: int = 2
) -> List[dict]:
    landmarks = _landmarks(scenario)
    alphas = SCENARIO_ALPHAS[scenario]
    obs_range = SCENARIO_RANGE[scenario]
    bias_rot = SCENARIO_BIAS[scenario]
    omega = SCENARIO_OMEGA[scenario]
    if n_steps is None:
        n_steps = SCENARIO_STEPS[scenario]

    x, y, th = 0.0, 0.0, 0.0
    events: List[dict] = [
        {
            "type": "meta",
            "landmarks": {k: [round(vx, 4), round(vy, 4)] for k, (vx, vy) in landmarks.items()},
            "initial_state": [0.0, 0.0, 0.0],
            "initial_cov_diag": [0.25, 0.25, 0.09],
            "alphas": alphas,
            "meas_var": MEAS_VAR,
        }
    ]

    def add_truth(t: float) -> None:
        events.append(
            {"type": "truth", "t": round(t, 4), "x": x, "y": y, "theta": normalize_angle(th)}
        )

    add_truth(0.0)
    kidnap_step = 180 if scenario == "kidnapped" else -1

    for k in range(1, n_steps + 1):
        t = k * DT

        if k == kidnap_step:
            # 瞬间平移 + 转头，里程计完全无感知。
            # 离散时间轴上该时刻只记录运动合并后的真值位姿（见循环末），
            # 避免同一 t 出现两个真值导致插值歧义。
            x += 3.5
            y -= 2.5
            th = normalize_angle(th + 1.2)

        d_trans_true = V * DT
        d_rot_true = omega * DT
        x += d_trans_true * math.cos(th)
        y += d_trans_true * math.sin(th)
        th = normalize_angle(th + d_rot_true)
        add_truth(t)

        nt, nr = _odom_noise(rng, alphas, d_trans_true, d_rot_true)
        events.append(
            {
                "type": "odom",
                "t": round(t, 4),
                "d_trans": round(d_trans_true + nt, 6),
                "d_rot": round(normalize_angle(d_rot_true + nr + bias_rot), 6),
            }
        )

        if k % obs_every == 0 or k == kidnap_step:
            for lm_id, (lx, ly) in landmarks.items():
                dx, dy = lx - x, ly - y
                r_true = math.hypot(dx, dy)
                if r_true > obs_range and k != kidnap_step:
                    continue
                brg_true = normalize_angle(math.atan2(dy, dx) - th)
                events.append(
                    {
                        "type": "obs",
                        "t": round(t, 4),
                        "id": lm_id,
                        "range": round(r_true + rng.gauss(0.0, SIGMA_R), 5),
                        "bearing": round(
                            normalize_angle(brg_true + rng.gauss(0.0, SIGMA_PHI)), 5
                        ),
                    }
                )

    return events


def write_log(path: Path, events: List[dict], shuffle: bool = False, seed: int = 0) -> None:
    meta = [e for e in events if e["type"] == "meta"]
    body = [e for e in events if e["type"] != "meta"]
    if shuffle:
        random.Random(seed + 777).shuffle(body)
    with path.open("w", encoding="utf-8") as f:
        for e in meta:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
        for e in body:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="生成 EKF 传感器日志")
    ap.add_argument("--outdir", default="data")
    ap.add_argument("--seed", type=int, default=20260922)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    for scenario in ("healthy", "kidnapped", "diverged"):
        rng = random.Random(args.seed + SCENARIO_SALT[scenario])
        events = simulate(scenario, rng)
        p = outdir / f"{scenario}.jsonl"
        write_log(p, events)
        print(f"写出 {p}  ({len(events)} 行)")

    # 乱序版本（绑架场景）：数据行整体打乱；解析器稳定排序后结果应一致
    events = simulate("kidnapped", random.Random(args.seed + SCENARIO_SALT["kidnapped"]))
    p = outdir / "kidnapped_shuffled.jsonl"
    write_log(p, events, shuffle=True, seed=args.seed)
    print(f"写出 {p}  (乱序/重复时间戳压力用例)")


if __name__ == "__main__":
    main()
