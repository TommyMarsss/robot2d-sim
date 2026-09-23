"""EKF 定位核心测试。

运行：python -m unittest discover -s tests -v
"""

from __future__ import annotations

import math
import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from robot2d.ekf import (CHI2_2DF_99, EKFConfig, EKFLocalizer, wrap_angle)
from robot2d.linalg import is_positive_semidefinite
from robot2d.logio import parse_log, run_filter
from robot2d.simlog import LANDMARKS, generate

LANDMARKS_STR = {k: tuple(v) for k, v in LANDMARKS.items()}


def make_log_lines(**kwargs) -> list[str]:
    return generate(**kwargs)


class TestWrapAngle(unittest.TestCase):
    def test_basic(self):
        self.assertAlmostEqual(wrap_angle(0.0), 0.0)
        self.assertAlmostEqual(wrap_angle(math.pi), math.pi)
        self.assertAlmostEqual(wrap_angle(-math.pi), math.pi)
        self.assertAlmostEqual(wrap_angle(3 * math.pi), math.pi)
        self.assertAlmostEqual(wrap_angle(math.pi + 0.1), -math.pi + 0.1)
        self.assertAlmostEqual(wrap_angle(-math.pi - 0.1), math.pi - 0.1)

    def test_no_jump_across_boundary(self):
        # 跨越 ±pi 时输出连续（差值被环绕到小角度）
        a = wrap_angle(math.pi - 1e-7)
        b = wrap_angle(math.pi + 1e-7)
        self.assertLess(abs(wrap_angle(b - a)), 1e-5)

    def test_large_angles(self):
        # 大角度取模存在浮点边界效应，比较时用环绕差判断角度等价
        def angle_eq(a, b):
            return abs(wrap_angle(a - b)) < 1e-9
        self.assertTrue(angle_eq(wrap_angle(100.0 * math.pi), 0.0))
        self.assertTrue(angle_eq(wrap_angle(101.0 * math.pi), math.pi))
        self.assertTrue(angle_eq(wrap_angle(100.5 * math.pi), math.pi / 2))


class TestCovarianceStability(unittest.TestCase):
    """长时间运行后协方差必须保持对称、半正定、有限。"""

    def _check_P(self, ekf: EKFLocalizer):
        P = ekf.P
        for i in range(3):
            for j in range(3):
                self.assertTrue(math.isfinite(P[i][j]), "协方差出现非有限值")
            self.assertGreater(P[i][i], 0.0, "协方差对角线必须为正")
        self.assertTrue(is_positive_semidefinite(P),
                        f"协方差非半正定: {P}")

    def test_long_prediction_run(self):
        # 纯航位推算 5000 步（无观测），协方差持续增长但不可退化
        ekf = EKFLocalizer(LANDMARKS_STR)
        for i in range(5000):
            ekf.predict(0.05, 0.01 * math.sin(i * 0.01))
        self._check_P(ekf)

    def test_long_mixed_run(self):
        # 预测 + 观测混合 3000 步
        rng = random.Random(7)
        ekf = EKFLocalizer(LANDMARKS_STR)
        for i in range(3000):
            ekf.predict(0.05 + rng.gauss(0, 0.005), rng.gauss(0, 0.002))
            if i % 3 == 0:
                lx, ly = LANDMARKS_STR[str(i % len(LANDMARKS_STR))]
                dx, dy = lx - ekf.x[0], ly - ekf.x[1]
                r = math.hypot(dx, dy) + rng.gauss(0, 0.15)
                phi = wrap_angle(math.atan2(dy, dx) - ekf.x[2]
                                 + rng.gauss(0, 0.05))
                ekf.update(str(i % len(LANDMARKS_STR)), r, phi, t=i * 0.1)
            if i % 100 == 0:
                self._check_P(ekf)
        self._check_P(ekf)

    def test_many_updates_no_collapse(self):
        # 大量观测更新后协方差不得塌缩为奇异
        ekf = EKFLocalizer(LANDMARKS_STR)
        for i in range(2000):
            ekf.predict(0.01, 0.0)
            lx, ly = LANDMARKS_STR["0"]
            dx, dy = lx - ekf.x[0], ly - ekf.x[1]
            ekf.update("0", math.hypot(dx, dy),
                       wrap_angle(math.atan2(dy, dx) - ekf.x[2]), t=i * 0.1)
        self._check_P(ekf)


class TestAngleWrapInFilter(unittest.TestCase):
    def test_heading_wraps_continuously(self):
        # 机器人原地旋转多圈，航向估计不得出现跳变误差
        ekf = EKFLocalizer(LANDMARKS_STR, x0=(0.0, 0.0, math.pi - 0.01))
        prev = ekf.x[2]
        for _ in range(100):
            ekf.predict(0.0, 0.01)
            # 相邻两步的环绕差必须是小量
            self.assertLess(abs(wrap_angle(ekf.x[2] - prev)), 0.05)
            prev = ekf.x[2]
        # 累计转角 1.0 rad，越过 +pi 后应环绕到负值附近
        self.assertAlmostEqual(ekf.x[2], wrap_angle(math.pi - 0.01 + 1.0))

    def test_observation_across_boundary(self):
        # 路标方位恰好在 ±pi 边界：新息不得出现 ~2pi 的虚假大残差
        ekf = EKFLocalizer({"0": (0.0, 1.0)}, x0=(0.0, 0.0, math.pi / 2))
        # 真实方位角 = atan2(1,0) - theta ≈ 0；构造一个环绕到 -pi 附近的观测
        # 真方位 0.01 rad，传感器报 -2*pi+0.01 的等价角
        nis = ekf.update("0", 1.0, wrap_angle(0.01 - 2 * math.pi), t=0.0)
        self.assertIsNotNone(nis)
        self.assertLess(nis, CHI2_2DF_99, "环绕边界处的新息被错误放大")


class TestDivergenceDetection(unittest.TestCase):
    def _run(self, **gen_kwargs):
        lines = make_log_lines(**gen_kwargs)
        parsed = parse_log(lines)
        ekf = EKFLocalizer(parsed.landmarks,
                           x0=parsed.init_pose or (0, 0, 0))
        return run_filter(parsed, ekf)

    def test_no_false_alarm_on_clean_log(self):
        ekf, _ = self._run(seed=1, kidnap_at=None)
        self.assertEqual(ekf.events, [],
                         f"干净日志不应触发事件: {ekf.events}")

    def test_kidnap_detected_and_recovers(self):
        ekf, history = self._run(seed=42, kidnap_at=260)
        kinds = [e.kind for e in ekf.events]
        self.assertIn("kidnap", kinds, f"绑架未被检测到: {kinds}")
        # 绑架后应最终恢复
        self.assertIn("recovered", kinds, f"绑架后未恢复: {kinds}")
        # 恢复后最终估计应重新接近真值
        final = history[-1]
        self.assertIsNotNone(final.truth)
        err = math.hypot(final.x - final.truth[0], final.y - final.truth[1])
        self.assertLess(err, 1.0, f"恢复后位置误差过大: {err:.3f} m")

    def test_sustained_bad_observations_flag_divergence(self):
        # 观测被系统性破坏（距离恒偏大 3m）→ 持续异常新息 → 发散
        ekf = EKFLocalizer({"0": (5.0, 0.0)})
        events = []
        for i in range(60):
            ekf.predict(0.05, 0.0)
            lx, ly = 5.0, 0.0
            dx, dy = lx - ekf.x[0], ly - ekf.x[1]
            r_true = math.hypot(dx, dy)
            phi = wrap_angle(math.atan2(dy, dx) - ekf.x[2])
            ekf.update("0", r_true + 3.0, phi, t=i * 0.1)
            events.extend(ekf.events)
        kinds = [e.kind for e in ekf.events]
        self.assertTrue(any(k in ("diverged", "kidnap") for k in kinds),
                        "持续异常观测未触发发散判定")


class TestOutOfOrderTimestamps(unittest.TestCase):
    def test_shuffled_equals_ordered(self):
        ordered = parse_log(make_log_lines(seed=5, shuffle=False))
        shuffled = parse_log(make_log_lines(seed=5, shuffle=True))
        # 去重排序后事件序列必须一致
        self.assertEqual(
            [(e.t, e.type, e.data) for e in ordered.events],
            [(e.t, e.type, e.data) for e in shuffled.events])

    def test_filter_results_identical(self):
        def run(lines):
            parsed = parse_log(lines)
            ekf = EKFLocalizer(parsed.landmarks,
                               x0=parsed.init_pose or (0, 0, 0))
            return run_filter(parsed, ekf)[1]

        h1 = run(make_log_lines(seed=9, shuffle=False))
        h2 = run(make_log_lines(seed=9, shuffle=True))
        self.assertEqual(len(h1), len(h2))
        for a, b in zip(h1, h2):
            self.assertAlmostEqual(a.x, b.x, places=12)
            self.assertAlmostEqual(a.y, b.y, places=12)
            self.assertAlmostEqual(a.theta, b.theta, places=12)

    def test_duplicates_removed(self):
        lines = make_log_lines(seed=3, shuffle=False)
        # 人为复制一半行再解析
        dup = lines + lines[len(lines) // 2:]
        p1 = parse_log(lines)
        p2 = parse_log(dup)
        self.assertEqual(len(p1.events), len(p2.events))


class TestAccuracy(unittest.TestCase):
    def test_tracks_ground_truth(self):
        # 无绑架时，最终位置误差应在合理范围
        lines = make_log_lines(seed=11, kidnap_at=None)
        parsed = parse_log(lines)
        ekf = EKFLocalizer(parsed.landmarks, x0=parsed.init_pose or (0, 0, 0))
        _, history = run_filter(parsed, ekf)
        final = history[-1]
        err = math.hypot(final.x - final.truth[0], final.y - final.truth[1])
        self.assertLess(err, 0.5, f"最终位置误差过大: {err:.3f} m")


if __name__ == "__main__":
    unittest.main()
