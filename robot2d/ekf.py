"""扩展卡尔曼滤波（EKF）平面机器人定位核心。

状态向量 x = [x, y, theta]^T，theta 为航向角（弧度，归一化到 (-pi, pi]）。

运动模型（里程计增量，中点积分）：
    输入 u = (d, dtheta)：机器人坐标系下前进距离 d 与原地转角 dtheta
    x'     = x + d * cos(theta + dtheta/2)
    y'     = y + d * sin(theta + dtheta/2)
    theta' = wrap(theta + dtheta)

观测模型（已知路标的距离-方位）：
    z = (r, phi)，r 为到路标距离，phi 为相对方位角（机器人坐标系）
    h(x) = (sqrt(dx^2+dy^2), wrap(atan2(dy, dx) - theta))

协方差数值稳定性措施：
  * 更新使用 Joseph 形式，对舍入误差更鲁棒；
  * 每步对称化 P；
  * 对角线设置下限，必要时对角加载，保证 P 始终对称半正定（见 linalg.ensure_psd）。

发散 / 绑架检测：
  * 每次观测更新计算 NIS（归一化新息平方）= y^T S^{-1} y；
  * 2 自由度卡方 99% 门限为 9.21，持续超出门限判为发散；
  * 单次极大 NIS（远超门限）且此前跟踪良好，判为“绑架”并主动放大协方差
    以便滤波器重新收敛；恢复后报告 recovered 事件。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from . import linalg as la

TWO_PI = 2.0 * math.pi

# 2 自由度卡方分布分位数
CHI2_2DF_99 = 9.21
# 单次 NIS 超过该值视为“绑架”级别的突变
KIDNAP_NIS = 100.0


def wrap_angle(a: float) -> float:
    """把角度归一化到 (-pi, pi]，正确处理 ±180° 环绕。"""
    return math.pi - ((math.pi - a) % TWO_PI)


@dataclass
class EKFConfig:
    """噪声与检测参数。"""

    # 里程计噪声：标准差 = 系数 * |运动量| + 下限
    odom_alpha_d: float = 0.08      # 平移噪声随平移量
    odom_alpha_th: float = 0.15     # 旋转噪声随旋转量
    odom_alpha_d_th: float = 0.03   # 平移噪声随旋转量（转弯时打滑）
    odom_floor_d: float = 0.005     # 平移噪声下限 (m)
    odom_floor_th: float = 0.002    # 旋转噪声下限 (rad)

    # 观测噪声（距离-方位）
    meas_std_r: float = 0.15        # 距离标准差 (m)
    meas_std_phi: float = 0.05      # 方位标准差 (rad)

    # 协方差数值保护
    cov_floor: float = 1e-9

    # 发散 / 绑架检测
    nis_gate: float = CHI2_2DF_99   # 单次新息门限
    kidnap_nis: float = KIDNAP_NIS  # 绑架级别 NIS
    diverge_window: int = 10        # 滑动窗口长度（观测次数）
    diverge_ratio: float = 0.6      # 窗口内超限比例超过该值判为发散
    recover_count: int = 5          # 连续正常观测达到该次数判为恢复
    kidnap_inflate: float = 25.0    # 绑架后协方差放大倍数


@dataclass
class FilterEvent:
    """滤波过程中产生的诊断事件。"""

    t: float
    kind: str        # "diverged" | "kidnap" | "recovered"
    detail: str


@dataclass
class DivergenceMonitor:
    """基于 NIS 滑动窗口的发散 / 绑架检测器。"""

    cfg: EKFConfig
    window: list[bool] = field(default_factory=list)  # 最近若干次观测是否超门限
    ok_streak: int = 0
    diverged: bool = False

    def update(self, nis: float, t: float) -> FilterEvent | None:
        cfg = self.cfg
        over = nis > cfg.nis_gate
        self.window.append(over)
        if len(self.window) > cfg.diverge_window:
            self.window.pop(0)

        if not self.diverged:
            if nis > cfg.kidnap_nis:
                self.diverged = True
                self.ok_streak = 0
                return FilterEvent(t, "kidnap", f"NIS={nis:.1f} >> {cfg.nis_gate}")
            if (len(self.window) == cfg.diverge_window
                    and sum(self.window) >= cfg.diverge_ratio * cfg.diverge_window):
                self.diverged = True
                self.ok_streak = 0
                return FilterEvent(t, "diverged",
                                   f"{sum(self.window)}/{len(self.window)} 次新息超门限")
            return None

        # 已处于发散状态：等待恢复
        if over:
            self.ok_streak = 0
        else:
            self.ok_streak += 1
            if self.ok_streak >= cfg.recover_count:
                self.diverged = False
                self.ok_streak = 0
                return FilterEvent(t, "recovered", "新息连续恢复正常")
        return None


class EKFLocalizer:
    """平面位姿 EKF 定位器。"""

    def __init__(
        self,
        landmarks: dict[str, tuple[float, float]],
        x0: tuple[float, float, float] = (0.0, 0.0, 0.0),
        p0: tuple[float, float, float] = (0.1, 0.1, 0.05),
        config: EKFConfig | None = None,
    ) -> None:
        self.cfg = config or EKFConfig()
        self.landmarks = dict(landmarks)
        self.x = [float(x0[0]), float(x0[1]), wrap_angle(float(x0[2]))]
        self.P = la.diag([p0[0] ** 2, p0[1] ** 2, p0[2] ** 2])
        self.monitor = DivergenceMonitor(self.cfg)
        self.events: list[FilterEvent] = []
        self.last_nis: float | None = None

    # ------------------------------------------------------------------ predict

    def predict(self, d: float, dtheta: float) -> None:
        """里程计增量预测。d: 前进距离(m)，dtheta: 转角(rad)。"""
        cfg = self.cfg
        x, y, th = self.x
        th_mid = th + 0.5 * dtheta
        c, s = math.cos(th_mid), math.sin(th_mid)

        self.x = [x + d * c, y + d * s, wrap_angle(th + dtheta)]

        # 状态雅可比 F = dx'/dx
        F = [
            [1.0, 0.0, -d * s],
            [0.0, 1.0, d * c],
            [0.0, 0.0, 1.0],
        ]
        # 控制雅可比 V = dx'/du，u = (d, dtheta)
        V = [
            [c, -0.5 * d * s],
            [s, 0.5 * d * c],
            [0.0, 1.0],
        ]
        var_d = (cfg.odom_alpha_d * abs(d) + cfg.odom_alpha_d_th * abs(dtheta)
                 + cfg.odom_floor_d) ** 2
        var_th = (cfg.odom_alpha_th * abs(dtheta) + cfg.odom_floor_th) ** 2
        Qu = [[var_d, 0.0], [0.0, var_th]]

        P = la.mat_mul(la.mat_mul(F, self.P), la.transpose(F))
        P = la.mat_add(P, la.mat_mul(la.mat_mul(V, Qu), la.transpose(V)))
        self.P = la.ensure_psd(P, self.cfg.cov_floor)

    # ------------------------------------------------------------------ update

    def update(self, landmark_id: str, r: float, phi: float,
               t: float = 0.0) -> float | None:
        """路标距离-方位观测更新。返回本次 NIS；未知路标返回 None。"""
        if landmark_id not in self.landmarks:
            return None
        lx, ly = self.landmarks[landmark_id]
        x, y, th = self.x
        dx, dy = lx - x, ly - y
        q = dx * dx + dy * dy
        if q < 1e-12:
            return None  # 与路标重合，观测无信息
        sq = math.sqrt(q)

        # 新息（方位分量做角度环绕，避免 ±180° 处跳变）
        innov = [r - sq, wrap_angle(phi - (math.atan2(dy, dx) - th))]

        H = [
            [-dx / sq, -dy / sq, 0.0],
            [dy / q, -dx / q, -1.0],
        ]
        R = la.diag([self.cfg.meas_std_r ** 2, self.cfg.meas_std_phi ** 2])

        Ht = la.transpose(H)
        S = la.mat_add(la.mat_mul(la.mat_mul(H, self.P), Ht), R)
        S = la.symmetrize(S)
        try:
            S_inv = la.inv2(S)
        except ArithmeticError:
            # S 奇异：说明 P 退化，先修复 P 再跳过本次更新
            self.P = la.ensure_psd(self.P, self.cfg.cov_floor * 1e3)
            return None

        K = la.mat_mul(la.mat_mul(self.P, Ht), S_inv)
        dx_state = la.mat_vec(K, innov)
        self.x = [self.x[0] + dx_state[0],
                  self.x[1] + dx_state[1],
                  wrap_angle(self.x[2] + dx_state[2])]

        # Joseph 形式更新，数值上保持对称半正定
        I_KH = la.mat_sub(la.eye(3), la.mat_mul(K, H))
        P = la.mat_add(
            la.mat_mul(la.mat_mul(I_KH, self.P), la.transpose(I_KH)),
            la.mat_mul(la.mat_mul(K, R), la.transpose(K)),
        )
        self.P = la.ensure_psd(P, self.cfg.cov_floor)

        sinv_y = la.mat_vec(S_inv, innov)
        nis = sum(a * b for a, b in zip(sinv_y, innov))  # y^T S^-1 y
        self.last_nis = nis

        event = self.monitor.update(nis, t)
        if event is not None:
            self.events.append(event)
            if event.kind == "kidnap":
                self._inflate_covariance(self.cfg.kidnap_inflate)
        return nis

    # ------------------------------------------------------------------ misc

    def _inflate_covariance(self, factor: float) -> None:
        """绑架后放大协方差，使滤波器愿意相信后续观测、重新收敛。"""
        self.P = la.ensure_psd(la.scale(self.P, factor), self.cfg.cov_floor)

    @property
    def pose(self) -> tuple[float, float, float]:
        return (self.x[0], self.x[1], self.x[2])

    def covariance_ellipse(self, n_sigma: float = 2.0) -> tuple[float, float, float]:
        """位置分量的置信椭圆：(长半轴, 短半轴, 转角)。"""
        sub = [[self.P[0][0], self.P[0][1]], [self.P[1][0], self.P[1][1]]]
        return la.ellipse_2x2(sub, n_sigma)
