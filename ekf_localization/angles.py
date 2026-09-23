"""角度工具：统一按弧度处理，规范到 (-pi, pi]。"""

from __future__ import annotations

import math

PI = math.pi
TWO_PI = 2.0 * math.pi


def normalize_angle(a: float) -> float:
    """把任意角度规范到主值区间 ``(-pi, pi]``。

    与常见的 ``[-pi, pi)`` 实现只差 pi 这一个点的归属；
    选 ``(-pi, pi]`` 使 +pi 保持为 +pi，对正负 180° 跨越更直观。
    """
    a = (a + PI) % TWO_PI - PI
    # 上面的写法对 -pi 得到 -pi（区间外），将其翻到 +pi。
    if a <= -PI:
        a += TWO_PI
    return a


def angle_diff(a: float, b: float) -> float:
    """有符号最短角差 ``a - b``，结果在 (-pi, pi]，正确处理环绕。"""
    return normalize_angle(a - b)


def deg_to_rad(d: float) -> float:
    return d * PI / 180.0


def rad_to_deg(r: float) -> float:
    return r * 180.0 / PI
