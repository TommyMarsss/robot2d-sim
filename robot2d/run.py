"""命令行入口：处理传感器日志并生成 HTML 回放报告。

用法：
    python -m robot2d.run 日志.jsonl -o 报告.html
    python -m robot2d.run 日志.jsonl --landmarks '{"0": [5, 0]}'  # 日志无 config 行时
"""

from __future__ import annotations

import argparse
import json
import sys

from .ekf import EKFConfig, EKFLocalizer
from .logio import parse_log, run_filter
from .report import write_report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="EKF 机器人定位：处理日志并生成回放报告")
    ap.add_argument("log", help="传感器日志文件 (JSONL)")
    ap.add_argument("-o", "--output", default="ekf_report.html", help="输出 HTML 路径")
    ap.add_argument("--landmarks", default=None,
                    help='JSON 形式的路标表，如 \'{"0": [5, 0], "1": [5, 5]}\'；'
                         "日志含 config 行时可省略")
    ap.add_argument("--meas-std-r", type=float, default=EKFConfig.meas_std_r)
    ap.add_argument("--meas-std-phi", type=float, default=EKFConfig.meas_std_phi)
    args = ap.parse_args(argv)

    with open(args.log, encoding="utf-8") as f:
        parsed = parse_log(f)

    if args.landmarks:
        extra = {str(k): (float(v[0]), float(v[1]))
                 for k, v in json.loads(args.landmarks).items()}
        extra.update(parsed.landmarks)  # 日志内的 config 优先
        parsed.landmarks = extra
    if not parsed.landmarks:
        print("错误：未提供路标表（日志无 config 行且未指定 --landmarks）",
              file=sys.stderr)
        return 2

    cfg = EKFConfig(meas_std_r=args.meas_std_r, meas_std_phi=args.meas_std_phi)
    init = parsed.init_pose or (0.0, 0.0, 0.0)
    ekf = EKFLocalizer(parsed.landmarks, x0=init, config=cfg)
    ekf, history = run_filter(parsed, ekf)

    write_report(args.output, history, ekf.events, parsed.landmarks, cfg.nis_gate)

    print(f"处理 {len(parsed.events)} 个事件 -> {args.output}")
    if ekf.events:
        for e in ekf.events:
            print(f"  [t={e.t:.2f}] {e.kind}: {e.detail}")
    else:
        print("  未检测到发散/绑架事件")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
