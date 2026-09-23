"""纯 Python 3.11 标准库实现的小规模稠密矩阵运算。

矩阵用 list[list[float]]（行优先）表示，向量用 list[float]。
仅实现 EKF 定位所需的少量操作，维度小（<=3），不追求性能。
"""

from __future__ import annotations

import math


def zeros(rows: int, cols: int) -> list[list[float]]:
    return [[0.0] * cols for _ in range(rows)]


def eye(n: int) -> list[list[float]]:
    m = zeros(n, n)
    for i in range(n):
        m[i][i] = 1.0
    return m


def transpose(a: list[list[float]]) -> list[list[float]]:
    return [list(row) for row in zip(*a)]


def mat_mul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    n, k, m = len(a), len(b), len(b[0])
    bt = transpose(b)
    return [[sum(a[i][p] * bt[j][p] for p in range(k)) for j in range(m)] for i in range(n)]


def mat_add(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    return [[x + y for x, y in zip(ra, rb)] for ra, rb in zip(a, b)]


def mat_sub(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    return [[x - y for x, y in zip(ra, rb)] for ra, rb in zip(a, b)]


def mat_vec(a: list[list[float]], v: list[float]) -> list[float]:
    return [sum(x * y for x, y in zip(row, v)) for row in a]


def scale(a: list[list[float]], s: float) -> list[list[float]]:
    return [[x * s for x in row] for row in a]


def diag(v: list[float]) -> list[list[float]]:
    m = zeros(len(v), len(v))
    for i, x in enumerate(v):
        m[i][i] = x
    return m


def inv2(m: list[list[float]]) -> list[list[float]]:
    """2x2 矩阵求逆。"""
    a, b = m[0]
    c, d = m[1]
    det = a * d - b * c
    if abs(det) < 1e-15:
        raise ArithmeticError("2x2 matrix is singular")
    s = 1.0 / det
    return [[d * s, -b * s], [-c * s, a * s]]


def symmetrize(p: list[list[float]]) -> list[list[float]]:
    """返回 (P + P^T) / 2，消除浮点累积的不对称性。"""
    n = len(p)
    out = zeros(n, n)
    for i in range(n):
        for j in range(n):
            out[i][j] = 0.5 * (p[i][j] + p[j][i])
    return out


def is_positive_semidefinite(p: list[list[float]], tol: float = -1e-9) -> bool:
    """用 Sylvester 判据（所有顺序主子式 >= 0）检查对称矩阵是否半正定。

    仅适用于对称矩阵；调用前应先 symmetrize。
    """
    n = len(p)
    # 1 阶主子式
    for i in range(n):
        if p[i][i] < tol:
            return False
    # 全部 2 阶主子式
    for i in range(n):
        for j in range(i + 1, n):
            if p[i][i] * p[j][j] - p[i][j] * p[j][i] < tol:
                return False
    # 3 阶（整体行列式），通过高斯消元求
    if n >= 3 and _determinant(p) < tol:
        return False
    return True


def _determinant(m: list[list[float]]) -> float:
    n = len(m)
    a = [row[:] for row in m]
    det = 1.0
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(a[r][col]))
        if abs(a[piv][col]) < 1e-300:
            return 0.0
        if piv != col:
            a[col], a[piv] = a[piv], a[col]
            det = -det
        det *= a[col][col]
        for r in range(col + 1, n):
            f = a[r][col] / a[col][col]
            for c in range(col, n):
                a[r][c] -= f * a[col][c]
    return det


def ensure_psd(p: list[list[float]], floor: float = 1e-9) -> list[list[float]]:
    """强制对称半正定：对称化 + 对角线下限 + 必要时对角加载（jitter）。

    返回修正后的新矩阵；输入不被修改。
    """
    n = len(p)
    out = symmetrize(p)
    for i in range(n):
        if not math.isfinite(out[i][i]) or out[i][i] < floor:
            out[i][i] = floor
    # 若仍非半正定（负的特征值来自非对角项），逐级加大对角加载
    jitter = floor
    for _ in range(8):
        if is_positive_semidefinite(out):
            return out
        for i in range(n):
            out[i][i] += jitter
        jitter *= 10.0
    return out


def ellipse_2x2(p: list[list[float]], n_sigma: float = 2.0) -> tuple[float, float, float]:
    """由 2x2 协方差子块求置信椭圆：(长半轴, 短半轴, 长轴转角 rad)。

    用 2x2 对称矩阵特征值的闭式解。
    """
    a, b = p[0][0], p[0][1]
    c, d = p[1][0], p[1][1]
    tr = a + d
    det = a * d - b * c
    disc = math.sqrt(max(0.0, tr * tr / 4.0 - det))
    l1 = tr / 2.0 + disc
    l2 = tr / 2.0 - disc
    angle = 0.5 * math.atan2(2.0 * b, a - d)
    return (n_sigma * math.sqrt(max(l1, 0.0)),
            n_sigma * math.sqrt(max(l2, 0.0)),
            angle)
