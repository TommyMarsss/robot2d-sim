"""健康监控测试：发散触发、绑架触发、恢复与迟滞。"""

import math
import unittest

from ekf_localization.ekf import EKFRobotLocalizer
from ekf_localization.monitor import (
    HealthMonitor,
    MonitorConfig,
    MonitorStatus,
)


def fresh_monitor(**kw):
    return HealthMonitor(MonitorConfig(**kw))


class TestMonitorStateMachine(unittest.TestCase):
    def test_healthy_under_normal_nis(self):
        m = fresh_monitor()
        for k in range(30):
            st = m.record_observation(t=0.1 * k, nis=1.5, pose_std_xy=0.3)
        self.assertEqual(st, MonitorStatus.HEALTHY)
        self.assertEqual(m.events, [])

    def test_single_outlier_does_not_trigger(self):
        m = fresh_monitor()
        for k in range(10):
            m.record_observation(0.1 * k, 1.0, 0.3)
        st = m.record_observation(1.0, 50.0, 0.3)  # 单次极端
        self.assertNotEqual(st, MonitorStatus.KIDNAPPED)
        # 随后恢复正常，不留事件
        for k in range(8):
            m.record_observation(1.1 + 0.1 * k, 1.0, 0.3)
        self.assertEqual(m.status, MonitorStatus.HEALTHY)

    def test_kidnap_triggers_on_consecutive_extremes(self):
        m = fresh_monitor()
        m.record_observation(0.0, 1.0, 0.3)
        m.record_observation(0.2, 1.0, 0.3)
        m.record_observation(0.4, 40.0, 0.3)
        st = m.record_observation(0.6, 40.0, 0.3)  # 连续第二次极端
        self.assertEqual(st, MonitorStatus.KIDNAPPED)
        kinds = [e.status for e in m.events]
        self.assertIn(MonitorStatus.KIDNAPPED, kinds)

    def test_diverge_triggers_on_persistent_large_nis(self):
        m = fresh_monitor()
        st = MonitorStatus.HEALTHY
        for k in range(1, 12):
            st = m.record_observation(0.2 * k, nis=15.0, pose_std_xy=0.5)
        self.assertEqual(st, MonitorStatus.DIVERGED)

    def test_sticky_until_consecutive_good_then_recovers(self):
        m = fresh_monitor(recover_count=5)
        for k in range(1, 4):
            m.record_observation(0.2 * k, 40.0, 0.5)
        self.assertEqual(m.status, MonitorStatus.KIDNAPPED)
        # 零星一次正常不应解除
        m.record_observation(1.0, 1.0, 0.5)
        self.assertEqual(m.status, MonitorStatus.KIDNAPPED)
        # 再来一次大残差仍保持绑架
        m.record_observation(1.2, 30.0, 0.5)
        self.assertEqual(m.status, MonitorStatus.KIDNAPPED)
        # 连续 5 次正常 -> 恢复
        for k in range(4):
            m.record_observation(1.4 + 0.2 * k, 1.0, 0.5)
        self.assertEqual(m.status, MonitorStatus.KIDNAPPED)
        m.record_observation(2.2, 1.0, 0.5)
        self.assertEqual(m.status, MonitorStatus.HEALTHY)
        self.assertIn(MonitorStatus.RECOVERED, [e.status for e in m.events])

    def test_long_observation_gap_classified_diverge_not_kidnap(self):
        m = fresh_monitor()
        m.record_observation(0.0, 1.0, 0.3)
        # 20 秒观测黑窗后连续极端新息 -> 应判发散（航位推算漂移），非绑架
        m.record_observation(20.0, 40.0, 2.0)
        st = m.record_observation(20.2, 40.0, 2.0)
        self.assertEqual(st, MonitorStatus.DIVERGED)

    def test_pose_std_inflation_flags_diverge(self):
        m = fresh_monitor(max_pose_std=2.0)
        m.record_motion_only(0.1, 1.0)
        self.assertEqual(m.status, MonitorStatus.HEALTHY)
        m.record_motion_only(0.2, 2.5)
        self.assertEqual(m.status, MonitorStatus.DIVERGED)


class TestFilterLevelDetection(unittest.TestCase):
    """在滤波器层面构造绑架：直接给与预测严重不符的连续观测。"""

    def test_filter_reports_kidnap_and_recovers(self):
        loc = EKFRobotLocalizer()
        loc.initialize((0.0, 0.0, 0.0), (0.01, 0.01, 0.01))
        lm = (5.0, 0.0)
        # 正常观测一段时间
        for k in range(1, 11):
            t = 0.2 * k
            loc.predict(0.0, 0.0, t)
            loc.update(5.0, 0.0, lm, t, "lm")
        self.assertEqual(loc.monitor.status, MonitorStatus.HEALTHY)

        # “绑架”：机器人实际被搬到远处，但路标观测按新位置给出
        # 用另一个远处路标产生距离突变（仍引用同 id 模拟地标重名不现实，
        # 因此直接给距离 8、方位大变 -> 连续两帧极端新息）
        loc.predict(0.0, 0.0, 2.2)
        loc.update(8.5, 0.35, lm, 2.2, "lm")
        # 第二次极端观测（时间间隔很短，属连续观测中突变 -> 绑架）
        rec = loc.update(8.5, 0.35, lm, 2.4, "lm")
        self.assertEqual(rec.status, MonitorStatus.KIDNAPPED.value)
        self.assertTrue(rec.reset)

        # 之后持续给与新位置一致的观测，应重新收敛并恢复
        for k in range(15, 40):
            t = 0.2 * k
            loc.predict(0.0, 0.0, t)
            rec = loc.update(8.5, 0.35, lm, t, "lm")
        self.assertEqual(loc.monitor.status, MonitorStatus.HEALTHY)
        self.assertIn(MonitorStatus.RECOVERED, [e.status for e in loc.monitor.events])


if __name__ == "__main__":
    unittest.main()
