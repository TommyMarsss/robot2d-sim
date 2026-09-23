"""线性代数与角度工具测试。"""

import math
import unittest

from ekf_localization import linalg as la
from ekf_localization.angles import angle_diff, normalize_angle


class TestLinalg(unittest.TestCase):
    def test_matmul_identity(self):
        a = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 10.0]]
        inv = la.inverse(a)
        prod = la.mul(a, inv)
        for i in range(3):
            for j in range(3):
                self.assertAlmostEqual(prod[i][j], 1.0 if i == j else 0.0, places=10)

    def test_solve_consistency(self):
        a = [[4.0, 1.0, 2.0], [1.0, 3.0, 0.5], [2.0, 0.5, 5.0]]
        b = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
        x = la.solve(a, b)
        chk = la.mul(a, x)
        for i in range(3):
            for j in range(2):
                self.assertAlmostEqual(chk[i][j], b[i][j], places=11)

    def test_cholesky_spd(self):
        a = [[4.0, 1.0, 0.5], [1.0, 3.0, 0.2], [0.5, 0.2, 2.0]]
        lower = la.cholesky(a)
        rebuilt = la.mul(lower, la.transpose(lower))
        for i in range(3):
            for j in range(3):
                self.assertAlmostEqual(rebuilt[i][j], a[i][j], places=12)

    def test_cholesky_rejects_non_spd(self):
        bad = [[1.0, 2.0], [2.0, 1.0]]  # 特征值 3 与 -1
        with self.assertRaises(la.NonPositiveDefiniteError):
            la.cholesky(bad)

    def test_eig2x2(self):
        a = [[3.0, 1.0], [1.0, 3.0]]
        (l1, l2), v = la.eig2x2_symmetric(a)
        self.assertAlmostEqual(l1, 4.0, places=12)
        self.assertAlmostEqual(l2, 2.0, 12)
        rebuilt = la.mul(la.mul(v, [[l1, 0.0], [0.0, l2]]), la.transpose(v))
        for i in range(2):
            for j in range(2):
                self.assertAlmostEqual(rebuilt[i][j], a[i][j], places=12)

    def test_jacobi_eigen_diagonal_dominance(self):
        # 带相关性的对称矩阵
        a = [[2.0, 0.5, -0.2], [0.5, 1.0, 0.3], [-0.2, 0.3, 1.5]]
        eigvals, v = la.jacobi_eigen(a)
        # 特征值均为正且和等于迹
        self.assertTrue(all(ev > 0 for ev in eigvals))
        self.assertAlmostEqual(sum(eigvals), 4.5, places=10)
        rebuilt = la.mul(la.mul(v, [[eigvals[i] if i == j else 0.0 for j in range(3)]
                                    for i in range(3)]), la.transpose(v))
        for i in range(3):
            for j in range(3):
                self.assertAlmostEqual(rebuilt[i][j], a[i][j], places=10)


class TestAngles(unittest.TestCase):
    def test_normalize_branch_cut(self):
        self.assertAlmostEqual(normalize_angle(3 * math.pi), math.pi, places=12)
        self.assertAlmostEqual(normalize_angle(-3 * math.pi), math.pi, places=12)
        self.assertAlmostEqual(normalize_angle(math.pi + 0.01), -math.pi + 0.01, places=12)

    def test_normalize_invariants(self):
        for a in [-10.0, -2 * math.pi, -0.001, 0.0, 1.0, 6.28, 100.0]:
            n = normalize_angle(a)
            self.assertGreaterEqual(n, -math.pi)
            self.assertLessEqual(n, math.pi)
            self.assertAlmostEqual(math.cos(n), math.cos(a), places=12)
            self.assertAlmostEqual(math.sin(n), math.sin(a), places=12)

    def test_angle_diff_wrap(self):
        # 179° 与 -179° 只差 2°，而不是 358°
        self.assertAlmostEqual(
            angle_diff(math.radians(179), math.radians(-179)),
            math.radians(-2),
            places=12,
        )
        self.assertAlmostEqual(
            angle_diff(math.radians(-179), math.radians(179)),
            math.radians(2),
            places=12,
        )
        self.assertAlmostEqual(angle_diff(0.5, 0.5), 0.0, places=14)


if __name__ == "__main__":
    unittest.main()
