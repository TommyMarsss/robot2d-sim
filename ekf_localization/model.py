"""运动模型与观测模型。

状态向量 ``x = [x, y, theta]^T``，单位：米 / 米 / 弧度。

运动输入是带噪声的**里程计增量** ``u = [d_trans, d_rot]``（本体内前进位移
与航向增量），采用一阶近似的航位推算模型：

.. math::

    x'     = x + d_t \\cos\\theta
    y'     = y + d_t \\sin\\theta
    \\theta' = \\theta + d_r

观测为相对已知路标 ``(l_x, l_y)`` 的距离-方位：

.. math::

    r = \\sqrt{dx^2 + dy^2}, \\qquad
    \\phi = \\operatorname{atan2}(dy, dx) - \\theta
"""

from __future__ import annotations

import math
from typing import Tuple

from . import linalg as la
from .angles import normalize_angle

# 状态索引
PX, PY, PTH = 0, 1, 2


def full_state_transition(
    state: la.Vector, d_trans: float, d_rot: float
) -> la.Vector:
    """非线性状态转移 f(x, u)，角度规范回主值。"""
    th = state[PTH]
    return [
        state[PX] + d_trans * math.cos(th),
        state[PY] + d_trans * math.sin(th),
        normalize_angle(state[PTH] + d_rot),
    ]


def transition_jacobian(
    state: la.Vector, d_trans: float
) -> la.Matrix:
    """f 对状态的雅可比 F = df/dx（3×3）。"""
    th = state[PTH]
    f = la.eye(3)
    f[PX][PTH] = -d_trans * math.sin(th)
    f[PY][PTH] = d_trans * math.cos(th)
    return f


def motion_noise_cov(
    state: la.Vector,
    d_trans: float,
    d_rot: float,
    alpha1: float,
    alpha2: float,
    alpha3: float,
    alpha4: float,
) -> la.Matrix:
    """里程计增量噪声映射到状态空间的协方差 ``F L F^T``。

    噪声系数（经典 Probabilistic Robotics 约定）：

    - ``alpha1``: 位移引起位移误差（单位 m / m）的方差系数；
    - ``alpha2``: 转动引起位移误差（m / rad）；
    - ``alpha3``: 位移引起转动误差（rad / m）；
    - ``alpha4``: 转动引起转动误差（rad / rad）。
    """
    var_trans = alpha1 * d_trans * d_trans + alpha2 * d_rot * d_rot
    var_rot = alpha3 * d_trans * d_trans + alpha4 * d_rot * d_rot
    # 保证静止或极小时也有一个噪声底，避免“零增量零协方差”退化。
    var_trans = max(var_trans, 1e-10)
    var_rot = max(var_rot, 1e-12)
    th = state[PTH]
    # 控制噪声 M = diag(var_trans, var_rot)（2×2），经 V（3×2）映射到状态：
    # V = [[cos th, 0], [sin th, 0], [0, 1]]  ->  Q = V M V^T
    q00 = var_trans * math.cos(th) ** 2
    q01 = var_trans * math.sin(th) * math.cos(th)
    q11 = var_trans * math.sin(th) ** 2
    return la.mat(
        [
            [q00, q01, 0.0],
            [q01, q11, 0.0],
            [0.0, 0.0, var_rot],
        ]
    )


def bearing_observation(
    state: la.Vector, landmark: Tuple[float, float]
) -> Tuple[float, float]:
    """预测观测 ``(range, bearing)``，bearing 规范到 (-pi, pi]。"""
    dx = landmark[0] - state[PX]
    dy = landmark[1] - state[PY]
    rng = math.hypot(dx, dy)
    if rng < 1e-9:
        return rng, 0.0
    brg = normalize_angle(math.atan2(dy, dx) - state[PTH])
    return rng, brg


def observation_jacobian(
    state: la.Vector, landmark: Tuple[float, float]
) -> la.Matrix:
    """观测 h 对状态的雅可比 H（2×3）。

    行依次为 range、bearing；列依次为 x、y、theta。
    """
    dx = landmark[0] - state[PX]
    dy = landmark[1] - state[PY]
    d2 = dx * dx + dy * dy
    if d2 < 1e-18:  # 理论上不会恰好在路标上，数值兜底
        d2 = 1e-18
    d = math.sqrt(d2)
    # d r / d x = -dx/d, ...
    hr = [-dx / d, -dy / d, 0.0]
    # d bearing / d x = dy / d2 ; bearing = atan2(dy,dx) - theta
    hb = [dy / d2, -dx / d2, -1.0]
    return [hr, hb]
