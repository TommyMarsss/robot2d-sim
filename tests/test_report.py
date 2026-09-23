"""报告生成冒烟测试：单文件内嵌、无外部依赖、数据可解析。"""

import json
import pathlib
import random
import re
import tempfile
import unittest

from ekf_localization.ekf import run_filter
from ekf_localization.logformat import parse_log
from ekf_localization.report import build_payload, render_html, write_html

import importlib.util

ROOT = pathlib.Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("generate_logs", ROOT / "tools" / "generate_logs.py")
gen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gen)


class TestReport(unittest.TestCase):
    def setUp(self):
        events = gen.simulate("kidnapped", random.Random(123))
        self.bundle = parse_log(events)
        self.result = run_filter(self.bundle)

    def test_payload_shape(self):
        p = build_payload(self.result, self.bundle)
        self.assertEqual(len(p["steps"]), len(self.result.steps))
        self.assertGreaterEqual(len(p["landmarks"]), 3)
        # 每行 18 个字段
        row0 = p["steps"][0]
        self.assertEqual(len(row0), 18)
        # 至少包含绑架/恢复事件编码
        codes = {e[1] for e in p["events"]}
        self.assertIn(3, codes)  # kidnapped

    def test_html_is_self_contained(self):
        html = render_html(self.result, self.bundle)
        # 无外部资源引用
        self.assertNotRegex(html, r"<script[^>]+src=")
        self.assertNotRegex(html, r"<link[^>]+href=")
        self.assertNotIn("http://", html)
        self.assertNotIn("https://", html)
        # 数据确实内嵌
        marker = "const DATA = "
        i = html.index(marker) + len(marker)
        data, _end = json.JSONDecoder().raw_decode(html[i:])
        self.assertEqual(len(data["steps"]), len(self.result.steps))

    def test_write_html_file(self):
        with tempfile.TemporaryDirectory() as d:
            p = pathlib.Path(d) / "r.html"
            write_html(self.result, p, self.bundle)
            text = p.read_text(encoding="utf-8")
        self.assertIn("<!DOCTYPE html>", text)
        self.assertIn("EKF", text)


if __name__ == "__main__":
    unittest.main()
