"""示例传感器日志生成器（仿真）。

模拟机器人在带若干已知路标的环境中沿圆弧行驶，输出：
  * 带噪声的里程计增量与距离-方位观测；
  * 真值轨迹（truth 事件，仅供报告对比）；
  * 可选的“绑架”事件（真值被瞬间平移，里程计无体现）；
  * 输出前将事件时间戳打乱并注入少量重复行，以模拟乱序日志。

用法：python -m robot2d.simlog -o sample_log.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import random

from .ekf import EKFConfig, wrap_angle

LANDMARKS = {
    "0": (5.0, 0.0),
    "1": (5.0, 5.0),
    "2": (0.0, 5.0),
    "3": (-3.0, 2.0),
    "4": (2.5, -3.0),
}


def generate(seed: int = 42, steps: int = 400, dt: float = 0.1,
             kidnap_at: int | None = 260, shuffle: bool = True) -> list[str]:
    rng = random.Random(seed)
    cfg = EKFConfig()

    lines: list[dict] = [{"type": "config", "landmarks": LANDMARKS},
                         {"type": "init", "t": 0.0, "x": 0.0, "y": 0.0, "theta": 0.0}]

    x, y, th = 0.0, 0.0, 0.0
    events: list[dict] = []
    for i in range(steps):
        t = (i + 1) * dt
        # 真值运动：绕路标群缓慢环行（保持在观测范围内）
        v = 0.5
        omega = 0.15 + 0.05 * math.sin(i * dt * 0.5)
        d_true, dth_true = v * dt, omega * dt

        # 绑架：真值瞬间平移，里程计观测不到
        if kidnap_at is not None and i == kidnap_at:
            x += 2.5
            y += 1.5

        th_mid = th + 0.5 * dth_true
        x += d_true * math.cos(th_mid)
        y += d_true * math.sin(th_mid)
        th = wrap_angle(th + dth_true)

        # 带噪声里程计
        d_noisy = d_true + rng.gauss(0.0, cfg.odom_alpha_d * abs(d_true)
                                     + cfg.odom_alpha_d_th * abs(dth_true)
                                     + cfg.odom_floor_d)
        dth_noisy = dth_true + rng.gauss(0.0, cfg.odom_alpha_th * abs(dth_true)
                                         + cfg.odom_floor_th)
        events.append({"type": "odom", "t": round(t, 6),
                       "d": d_noisy, "dtheta": dth_noisy})
        events.append({"type": "truth", "t": round(t, 6), "x": x, "y": y, "theta": th})

        # 每隔几步观测所有可见路标
        if i % 2 == 0:
            for lid, (lx, ly) in LANDMARKS.items():
                dx, dy = lx - x, ly - y
                r = math.hypot(dx, dy)
                if r > 9.0:
                    continue
                phi = wrap_angle(math.atan2(dy, dx) - th)
                if abs(phi) > 2.4:  # 视场角限制
                    continue
                events.append({"type": "obs", "t": round(t, 6), "landmark": lid,
                               "range": r + rng.gauss(0.0, cfg.meas_std_r),
                               "bearing": phi + rng.gauss(0.0, cfg.meas_std_phi)})

    # 注入重复事件（模拟日志重复上报）
    for _ in range(steps // 20):
        events.append(dict(rng.choice(events)))

    if shuffle:
        rng.shuffle(events)

    lines.extend(events)
    return [json.dumps(rec) for rec in lines]


def main() -> None:
    ap = argparse.ArgumentParser(description="生成示例传感器日志 (JSONL)")
    ap.add_argument("-o", "--output", required=True, help="输出文件路径")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--no-kidnap", action="store_true", help="不注入绑架事件")
    ap.add_argument("--ordered", action="store_true", help="不打乱时间戳")
    args = ap.parse_args()
    lines = generate(seed=args.seed, steps=args.steps,
                     kidnap_at=None if args.no_kidnap else 260,
                     shuffle=not args.ordered)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"已生成 {len(lines)} 行 -> {args.output}")


if __name__ == "__main__":
    main()
