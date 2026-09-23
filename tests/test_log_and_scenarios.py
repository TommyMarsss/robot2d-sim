"""时间戳乱序/重复处理，以及端到端日志 -> EKF 结果的测试。"""

import importlib.util
import pathlib
import random
import unittest

from ekf_localization.ekf import run_filter
from ekf_localization.logformat import (
    Landmark,
    LogBundle,
    Observation,
    OdometryIncrement,
    dump_jsonl,
    parse_log,
)
from ekf_localization.monitor import MonitorStatus

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load_generator():
    spec = importlib.util.spec_from_file_location(
        "generate_logs", ROOT / "tools" / "generate_logs.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


gen = _load_generator()


class TestTimestampOrdering(unittest.TestCase):
    def _bundle_from_events(self, events):
        return parse_log(events)

    def test_shuffled_events_give_same_result(self):
        events = gen.simulate("kidnapped", random.Random(42))
        meta = [e for e in events if e["type"] == "meta"]
        body = [e for e in events if e["type"] != "meta"]

        ordered = parse_log(meta + body)
        shuffled = body[:]
        random.Random(99).shuffle(shuffled)
        unordered = parse_log(meta + shuffled)

        # 解析后按规范顺序：里程计按 (t,seq)，观测按 (t,id,seq)
        self.assertEqual(
            [e.t for e in ordered.odometry],
            [e.t for e in unordered.odometry],
        )
        self.assertEqual(
            [(e.t, e.id) for e in ordered.observations],
            [(e.t, e.id) for e in unordered.observations],
        )

        r1 = run_filter(ordered)
        r2 = run_filter(unordered)
        self.assertEqual(len(r1.steps), len(r2.steps))
        for a, b in zip(r1.steps, r2.steps):
            self.assertEqual(a.kind, b.kind)
            self.assertAlmostEqual(a.t, b.t, places=9)
            self.assertAlmostEqual(a.state[0], b.state[0], places=12)
            self.assertAlmostEqual(a.state[1], b.state[1], places=12)
            self.assertAlmostEqual(a.state[2], b.state[2], places=12)

    def test_duplicate_timestamps_have_deterministic_order(self):
        events = [
            {"type": "meta", "landmarks": {"lm1": [1.0, 0.0], "lm2": [1.0, 1.0]}},
            {"type": "obs", "t": 1.0, "id": "lm2", "range": 1.4, "bearing": 0.7},
            {"type": "odom", "t": 1.0, "d_trans": 0.1, "d_rot": 0.0},
            {"type": "obs", "t": 1.0, "id": "lm1", "range": 0.9, "bearing": 0.0},
        ]
        bundle = parse_log(events)
        # 同一时刻：里程计先于观测；多个观测按路标 id 规范化（lm1 先于 lm2），
        # 与文件内出现顺序无关。
        merged = []
        oi = obi = 0
        while oi < len(bundle.odometry) or obi < len(bundle.observations):
            if obi >= len(bundle.observations):
                merged.append(("odom", bundle.odometry[oi].seq)); oi += 1
            elif oi >= len(bundle.odometry):
                merged.append(("obs", bundle.observations[obi].id)); obi += 1
            elif bundle.observations[obi].t < bundle.odometry[oi].t:
                merged.append(("obs", bundle.observations[obi].id)); obi += 1
            else:
                merged.append(("odom", bundle.odometry[oi].seq)); oi += 1
        self.assertEqual(merged, [("odom", 1), ("obs", "lm1"), ("obs", "lm2")])

    def test_jsonl_roundtrip_and_file_load(self, tmp_path=None):
        import tempfile, pathlib as p

        bundle = LogBundle(
            landmarks={"a": Landmark("a", 1.0, 2.0)},
            odometry=[OdometryIncrement(0.1, 0.05, 0.01, 0)],
            observations=[Observation(0.2, "a", 1.3, -0.2, 1)],
        )
        text = dump_jsonl(bundle)
        with tempfile.TemporaryDirectory() as d:
            f = p.Path(d) / "log.jsonl"
            f.write_text(text, encoding="utf-8")
            loaded = parse_log(f)
        self.assertEqual(loaded.landmarks["a"].xy(), (1.0, 2.0))
        self.assertAlmostEqual(loaded.odometry[0].d_trans, 0.05, places=12)
        self.assertAlmostEqual(loaded.observations[0].bearing, -0.2, places=12)


class TestEndToEndScenarios(unittest.TestCase):
    def test_healthy_scenario(self):
        events = gen.simulate("healthy", random.Random(7))
        result = run_filter(parse_log(events))
        errs = [s.err_xy for s in result.steps if s.err_xy is not None]
        # 绝大多数时刻位置误差应小于 0.5m
        self.assertLess(sum(e > 0.5 for e in errs) / len(errs), 0.05)
        # 不应出现绑架误报
        self.assertNotIn(
            MonitorStatus.KIDNAPPED, [e.status for e in result.monitor.events]
        )

    def test_kidnapped_scenario_detected_and_recovers(self):
        events = gen.simulate("kidnapped", random.Random(7))
        result = run_filter(parse_log(events))
        statuses = [e.status for e in result.monitor.events]
        self.assertIn(MonitorStatus.KIDNAPPED, statuses)
        # 绑架事件应发生在 t≈18s 附近（而不是起点就报）
        kt = [e.t for e in result.monitor.events if e.status == MonitorStatus.KIDNAPPED]
        self.assertTrue(any(16.0 < t < 22.0 for t in kt))
        self.assertIn(MonitorStatus.RECOVERED, statuses)
        # 恢复后末态误差应重新变小
        tail = [s.err_xy for s in result.steps[-50:] if s.err_xy is not None]
        self.assertLess(sum(tail) / len(tail), 0.6)

    def test_diverged_scenario_detected(self):
        events = gen.simulate("diverged", random.Random(7))
        result = run_filter(parse_log(events))
        statuses = [e.status for e in result.monitor.events]
        self.assertIn(MonitorStatus.DIVERGED, statuses)
        # 发散判定发生在重捕获远端路标期间（25s 之后）
        dt_ev = [e.t for e in result.monitor.events if e.status == MonitorStatus.DIVERGED]
        self.assertTrue(any(t > 25.0 for t in dt_ev))


if __name__ == "__main__":
    unittest.main()
