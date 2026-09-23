"""纯 Python 小矩阵运算（3x3 为主，观测更新用到 2x2）。

仅实现本项目需要的操作，全部用嵌套 list 表示矩阵::

    A = [[a00, a01, ...], ...]   # A[i][j] = 第 i 行第 j 列

设计目标：
- 零第三方依赖（无 numpy）；
- 对称矩阵（协方差）做对称化、对称特征分解，避免长时间运行产生虚特征值；
- 提供 Cholesky 正定检查与 Joseph 形式更新，从数值上保证协方差半正定。
"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

Matrix = List[List[float]]
Vector = List[float]


# ---------------------------------------------------------------------------
# 基础操作
# ---------------------------------------------------------------------------

def zeros(rows: int, cols: int) -> Matrix:
    return [[0.0 for _ in range(cols)] for _ in range(rows)]


def eye(n: int) -> Matrix:
    m = zeros(n, n)
    for i in range(n):
        m[i][i] = 1.0
    return m


def mat(a: Sequence[Sequence[float]]) -> Matrix:
    """由嵌套序列构造矩阵（深拷贝浮点化）。"""
    return [[float(v) for v in row] for row in a]


def vec(x: Sequence[float]) -> Vector:
    return [float(v) for v in x]


def shape(a: Matrix) -> Tuple[int, int]:
    return len(a), len(a[0]) if a else 0


def transpose(a: Matrix) -> Matrix:
    r, c = shape(a)
    return [[a[i][j] for i in range(r)] for j in range(c)]


def add(a: Matrix, b: Matrix) -> Matrix:
    r, c = shape(a)
    return [[a[i][j] + b[i][j] for j in range(c)] for i in range(r)]


def sub(a: Matrix, b: Matrix) -> Matrix:
    r, c = shape(a)
    return [[a[i][j] - b[i][j] for j in range(c)] for i in range(r)]


def scale(a: Matrix, s: float) -> Matrix:
    r, c = shape(a)
    return [[a[i][j] * s for j in range(c)] for i in range(r)]


def mul(a: Matrix, b: Matrix) -> Matrix:
    """矩阵乘法 A (r×k) * B (k×c)。"""
    r, k = shape(a)
    k2, c = shape(b)
    if k != k2:
        raise ValueError(f"矩阵维度不匹配: ({r},{k}) x ({k2},{c})")
    out = zeros(r, c)
    for i in range(r):
        ai = a[i]
        oi = out[i]
        for t in range(k):
            ait = ai[t]
            if ait == 0.0:
                continue
            bt = b[t]
            for j in range(c):
                oi[j] += ait * bt[j]
    return out


def mv(a: Matrix, x: Vector) -> Vector:
    """矩阵 × 列向量。"""
    r, c = shape(a)
    return [sum(a[i][j] * x[j] for j in range(c)) for i in range(r)]


def vadd(a: Vector, b: Vector) -> Vector:
    return [x + y for x, y in zip(a, b)]


def vsub(a: Vector, b: Vector) -> Vector:
    return [x - y for x, y in zip(a, b)]


# ---------------------------------------------------------------------------
# 对称矩阵（协方差）专用
# ---------------------------------------------------------------------------

def symmetrize(a: Matrix) -> Matrix:
    """P <- (P + P^T) / 2，消除浮点不对称。"""
    r, c = shape(a)
    out = zeros(r, c)
    for i in range(r):
        for j in range(i, c):
            v = 0.5 * (a[i][j] + a[j][i])
            out[i][j] = v
            out[j][i] = v
    return out


def cholesky(a: Matrix) -> Matrix:
    """对称正定矩阵的 Cholesky 下三角分解 A = L L^T。

    非正定时抛出 :class:`NonPositiveDefiniteError`。
    """
    n = len(a)
    L = zeros(n, n)
    for i in range(n):
        for j in range(i + 1):
            s = sum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                d = a[i][i] - s
                if not (d > 0.0):
                    raise NonPositiveDefiniteError(
                        f"Cholesky 失败: 第 {i} 个对角元 {d!r} <= 0"
                    )
                L[i][j] = math.sqrt(d)
            else:
                L[i][j] = (a[i][j] - s) / L[j][j]
    return L


class NonPositiveDefiniteError(ValueError):
    """协方差矩阵失去正定性。"""


# ---------------------------------------------------------------------------
# 线性求解与求逆（小矩阵高斯消元，带部分选主元）
# ---------------------------------------------------------------------------

def solve(a: Matrix, b: Matrix) -> Matrix:
    """解 A X = B，A 为方阵，B 为 n×k。"""
    n = len(a)
    m = [row[:] + br[:] for row, br in zip(a, b)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[piv][col]) < 1e-14:
            raise NonPositiveDefiniteError("矩阵奇异，无法求解")
        if piv != col:
            m[col], m[piv] = m[piv], m[col]
        pivval = m[col][col]
        for r in range(n):
            if r == col:
                continue
            factor = m[r][col] / pivval
            if factor == 0.0:
                continue
            for j in range(col, n + len(b[0])):
                m[r][j] -= factor * m[col][j]
    k = len(b[0])
    return [[m[i][n + j] / m[i][i] for j in range(k)] for i in range(n)]


def inverse(a: Matrix) -> Matrix:
    return solve(a, eye(len(a)))


# ---------------------------------------------------------------------------
# 2x2 对称特征分解（协方差椭圆）——解析法
# ---------------------------------------------------------------------------

def eig2x2_symmetric(a: Matrix) -> Tuple[Tuple[float, float], Matrix]:
    """2×2 对称矩阵的特征值与归一化特征向量。

    :returns: ``((lambda_major, lambda_minor), V)``，
        ``V`` 的两列为对应特征向量（major 在前）。
    """
    (a00, a01), (_, a11) = a[0], a[1]
    tr = a00 + a11
    disc = math.hypot(a00 - a11, 2.0 * a01)
    lam1 = 0.5 * (tr + disc)
    lam2 = 0.5 * (tr - disc)

    def _eigvec(lam: float) -> Vector:
        # (a00-lam) x + a01 y = 0
        if abs(a01) > 1e-300:
            vx, vy = a01, lam - a00
        elif abs(a00 - lam) < abs(a11 - lam):
            vx, vy = 1.0, 0.0
        else:
            vx, vy = 0.0, 1.0
        nrm = math.hypot(vx, vy)
        return [vx / nrm, vy / nrm]

    v1 = _eigvec(lam1)
    v2 = [-v1[1], v1[0]]  # 与 v1 正交
    return (lam1, lam2), [
        [v1[0], v2[0]],
        [v1[1], v2[1]],
    ]


# ---------------------------------------------------------------------------
# 3x3 对称特征分解（角度周期边界需要时可用的 Jacobi 法，当前主流程用 eig2）
# ---------------------------------------------------------------------------

def jacobi_eigen(a: Matrix, max_sweeps: int = 64, tol: float = 1e-14) -> Tuple[Vector, Matrix]:
    """实对称矩阵的 Jacobi 特征分解，返回 (特征值, 列特征向量)。"""
    n = len(a)
    m = [row[:] for row in a]
    v = eye(n)
    for _ in range(max_sweeps):
        off = math.sqrt(
            sum(m[i][j] * m[i][j] for i in range(n) for j in range(n) if i != j)
        )
        if off < tol:
            break
        for p in range(n - 1):
            for q in range(p + 1, n):
                if abs(m[p][q]) < 1e-300:
                    continue
                tau = (m[q][q] - m[p][p]) / (2.0 * m[p][q])
                if tau >= 0.0:
                    t = 1.0 / (tau + math.sqrt(1.0 + tau * tau))
                else:
                    t = -1.0 / (-tau + math.sqrt(1.0 + tau * tau))
                c = 1.0 / math.sqrt(1.0 + t * t)
                s = t * c
                mpp, mqq, mpq = m[p][p], m[q][q], m[p][q]
                m[p][p] = mpp - t * mpq
                m[q][q] = mqq + t * mpq
                m[p][q] = m[q][p] = 0.0
                for k in range(n):
                    if k in (p, q):
                        continue
                    mkp, mkq = m[k][p], m[k][q]
                    m[k][p] = m[p][k] = c * mkp - s * mkq
                    m[k][q] = m[q][k] = s * mkp + c * mkq
                for k in range(n):
                    vkp, vkq = v[k][p], v[k][q]
                    v[k][p] = c * vkp - s * vkq
                    v[k][q] = s * vkp + c * vkq
    return [m[i][i] for i in range(n)], v
