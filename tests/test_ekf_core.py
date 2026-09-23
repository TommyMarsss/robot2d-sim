"""EKF 主体测试：协方差数值稳定性、角度环绕、滤波精度。"""

import math
import unittest

from ekf_localization import linalg as la
from ekf_localization.angles import normalize_angle
from ekf_localization.ekf import EKFRobotLocalizer

ALPHAS = (0.06, 0.03, 0.05, 0.04)
MEAS_VAR = (0.02, 0.005)


def make_localizer():
    return EKFRobotLocalizer(alphas=ALPHAS, meas_var=MEAS_VAR)


def synthetic_circle(n=600, dt=0.1, v=0.5, omega=0.1, landmarks=None):
    """无噪声的绕圈真值序列，返回 (odom 增量 list, 观测函数, 真值序列)。"""
    if landmarks is None:
        landmarks = {"lm": (8.0, 0.0)}
    x = y = th = 0.0
    poses = [(0.0, 0.0, 0.0)]
    for k in range(1, n + 1):
        ds, dr = v * dt, omega * dt
        x += ds * math.cos(th)
        y += ds * math.sin(th)
        th = normalize_angle(th + dr)
        poses.append((x, y, normalize_angle(th)))
    return poses, landmarks


class TestCovarianceStability(unittest.TestCase):
    def _check_pd(self, loc):
        eigvals, _ = la.jacobi_eigen(loc.cov)
        for ev in eigvals:
            self.assertGreater(ev, 0.0, f"协方差非正定，特征值={eigvals}")
            self.assertTrue(math.isfinite(ev))
        # 对称性
        for i in range(3):
            for j in range(3):
                self.assertAlmostEqual(loc.cov[i][j], loc.cov[j][i], places=12)

    def test_long_run_pure_odometry_stays_pd(self):
        loc = make_localizer()
        loc.initialize((0.0, 0.0, 0.0), (0.25, 0.25, 0.09))
        for k in range(1, 20001):
            loc.predict(0.05, 0.01, t=k * 0.1)
            if k % 1000 == 0:
                self._check_pd(loc)
        self._check_pd(loc)
        # 长距离纯航位推算后不确定度应增长但不爆炸
        self.assertLess(loc.cov[0][0], 1e5)

    def test_long_run_with_observations_stays_pd(self):
        loc = make_localizer()
        loc.initialize((0.0, 0.0, 0.0), (0.25, 0.25, 0.09))
        lm = (5.0, 4.0)
        for k in range(1, 5001):
            t = k * 0.1
            loc.predict(0.05, 0.01, t)
            if k % 2 == 0:
                # 用与当前估计自洽的理想观测，雅可比不退化
                rng = math.hypot(lm[0] - loc.state[0], lm[1] - loc.state[1])
                brg = normalize_angle(
                    math.atan2(lm[1] - loc.state[1], lm[0] - loc.state[0]) - loc.state[2]
                )
                loc.update(rng, brg, lm, t, "lm")
            if k % 500 == 0:
                self._check_pd(loc)
        self._check_pd(loc)

    def test_repeated_identical_observations_no_collapse(self):
        """大量重复观测不能把协方差压成奇异。"""
        loc = make_localizer()
        loc.initialize((0.0, 0.0, 0.0), (0.25, 0.25, 0.09))
        lm = (3.0, 0.0)
        for k in range(2000):
            rng, brg = 3.0, 0.0  # 完全一致的观测
            loc.update(rng, brg, lm, t=k * 0.1, landmark_id="lm")
        eigvals, _ = la.jacobi_eigen(loc.cov)
        # 角度方向永远不被距离-方位观测完全约束，必须有可观方差
        self.assertGreater(min(eigvals), 1e-10)
        self._check_pd_check(loc)

    def _check_pd_check(self, loc):
        eigvals, _ = la.jacobi_eigen(loc.cov)
        self.assertTrue(all(ev > 0 for ev in eigvals))

    def test_joseph_form_residual_r(self):
        # 更新后 P 应约等于 (I-KH)P_pred：观测充分时方差显著缩小
        loc = make_localizer()
        loc.initialize((0.5, 0.2, 0.1), (0.25, 0.25, 0.09))
        p_before = [row[:] for row in loc.cov]
        loc.update(math.hypot(3.0 - 0.5, 0.0 - 0.2),
                   normalize_angle(math.atan2(-0.2, 2.5) - 0.1),
                   (3.0, 0.0), t=0.1)
        # 迹变小（观测有信息）
        self.assertLess(
            loc.cov[0][0] + loc.cov[1][1], p_before[0][0] + p_before[1][1]
        )


class TestAngleWrapInFilter(unittest.TestCase):
    def test_heading_near_pi_update(self):
        loc = make_localizer()
        loc.initialize((0.0, 0.0, math.pi - 0.02), (0.01, 0.01, 0.01))
        lm = (-3.0, 0.0)  # 机器人几乎正后方
        # 真值航向就是 pi；真实 bearing = atan2(0,-3) - pi = pi - pi = 0
        rng_true, brg_true = 3.0, 0.0
        loc.update(rng_true, brg_true, lm, t=0.1)
        # 不应出现 ±2π 量级的跳变修正
        self.assertAlmostEqual(loc.state[2], math.pi, delta=0.05)

    def test_bearing_residual_wraps(self):
        loc = make_localizer()
        loc.initialize((0.0, 0.0, -math.pi + 0.02), (0.01, 0.01, 0.01))
        lm = (3.0, 0.0)
        # 预测 bearing = 0 - (-pi+0.02) = pi-0.02，观测给 -pi+0.02 表示几乎同方向
        # 残差应约 0.04 rad，而非约 2pi
        rec = loc.update(3.0, math.pi - 0.06, lm, t=0.1)
        self.assertIsNotNone(rec.innovation)
        self.assertLess(abs(rec.innovation[1]), 0.1)

    def test_full_rotation_track(self):
        """连续转弯让 theta 多次越过 ±pi，估计始终在主值区间且无跳变误差。"""
        loc = make_localizer()
        loc.initialize((0.0, 0.0, 0.0), (1e-6, 1e-6, 1e-6))
        th_true = 0.0
        lm = (0.0, 10.0)
        for k in range(1, 1000):
            dr = 0.02
            th_true = normalize_angle(th_true + dr)
            loc.predict(0.0, dr, t=k * 0.1)
            if k % 5 == 0:
                brg = normalize_angle(math.atan2(10.0, 0.0) - th_true)
                loc.update(10.0, brg, lm, t=k * 0.1, landmark_id="lm")
            # 状态角必须规范
            self.assertGreaterEqual(loc.state[2], -math.pi)
            self.assertLessEqual(loc.state[2], math.pi)
        # 累计 19.98 rad ≈ 3.18 圈后仍应跟踪真值
        err = abs(normalize_angle(loc.state[2] - th_true))
        self.assertLess(err, 0.05)


class TestFilterAccuracy(unittest.TestCase):
    def test_no_noise_converges_to_truth(self):
        """无噪声理想数据下 EKF 应收敛到真值（两个路标，状态完全可观）。"""
        poses, _ = synthetic_circle(n=300)
        lms = {"a": (8.0, 0.0), "b": (0.0, 8.0)}
        loc = make_localizer()
        loc.initialize((0.2, -0.2, 0.1), (0.25, 0.25, 0.09))
        for k in range(1, 301):
            t = k * 0.1
            loc.predict(0.05, 0.01, t)
            if k % 2 == 0:
                gx, gy, gth = poses[k]
                for lid, (lx, ly) in lms.items():
                    dx, dy = lx - gx, ly - gy
                    loc.update(
                        math.hypot(dx, dy),
                        normalize_angle(math.atan2(dy, dx) - gth),
                        (lx, ly), t, lid,
                    )
        gx, gy, _ = poses[300]
        self.assertLess(math.hypot(loc.state[0] - gx, loc.state[1] - gy), 0.15)


if __name__ == "__main__":
    unittest.main()
