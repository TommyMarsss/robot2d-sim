#!/usr/bin/env python3
"""命令行入口：运行 EKF 处理一份日志，打印健康摘要并生成静态 HTML 报告。

用法::

    python run_ekf.py data/kidnapped.jsonl -o report_kidnapped.html
    python run_ekf.py data/diverged.jsonl
    python run_ekf.py data/healthy.jsonl --quiet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ekf_localization import EKFRobotLocalizer, HealthMonitor, MonitorConfig, parse_log
from ekf_localization.report import write_html


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="2D EKF 定位：处理日志并生成 HTML 回放")
    ap.add_argument("log", help="JSON Lines 传感器日志路径")
    ap.add_argument("-o", "--output", default=None, help="HTML 输出路径（默认 report_<日志名>.html）")
    ap.add_argument("--no-report", action="store_true", help="只运行滤波，不生成 HTML")
    ap.add_argument("--quiet", action="store_true", help="不打印逐步摘要")
    args = ap.parse_args(argv)

    bundle = parse_log(args.log)
    localizer = EKFRobotLocalizer(
        alphas=bundle.alphas,
        meas_var=bundle.meas_var,
        monitor=HealthMonitor(MonitorConfig()),
    )
    result = localizer.run_log(bundle)

    if not args.quiet:
        n_obs = sum(1 for s in result.steps if s.kind == "obs")
        n_rej = sum(1 for s in result.steps if s.rejected)
        print(f"日志: {bundle.source}")
        print(
            f"事件 {len(result.steps)} 步（观测 {n_obs}，野值降权 {n_rej}），"
            f"时间范围 {bundle.time_span()[0]:.2f}–{bundle.time_span()[1]:.2f}s"
        )
        fx, fy, fth = result.final_state()
        print(f"末态估计: x={fx:.3f} m, y={fy:.3f} m, θ={fth:.3f} rad")
        errs = [s.err_xy for s in result.steps if s.err_xy is not None]
        if errs:
            print(
                f"位置误差: 均值 {sum(errs)/len(errs):.3f} m, "
                f"最大 {max(errs):.3f} m，末态 {errs[-1]:.3f} m"
            )
        print("-" * 72)
        print(result.monitor.summary())
        print("-" * 72)

    if not args.no_report:
        out = Path(args.output) if args.output else Path(f"report_{Path(args.log).stem}.html")
        write_html(result, out, bundle=bundle)
        print(f"报告已生成: {out.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
