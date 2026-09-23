"""扩展卡尔曼滤波主体。

流程（时间轴上里程计与观测按 ``(t, seq)`` 稳定归并后依次处理）：

1. 里程计增量 -> 预测：
   ``x = f(x,u);  P = F P F^T + Q``
2. 路标观测 -> 更新：
   ``nu = z - h(x)``（方位分量做角度环绕）；
   ``S = H P H^T + R; K = P H^T S^{-1}``；
   ``x += K nu``；
   协方差用 **Joseph 形式** :math:`(I-KH)P(I-KH)^T + KRK^T`，
   再做对称化与特征值下限钳制，保证长期运行仍正定不奇异。
3. 每次更新把 NIS 喂给 :class:`HealthMonitor`；
   单点野值做鲁棒 R 膨胀；判定绑架时做不确定性重置（重定位），
   判定发散时做协方差自适应膨胀，二者之后都能凭正常观测自动恢复。

核心类不依赖报告模块，可单独实例化、单步驱动与测试。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from . import linalg as la
from .angles import angle_diff, normalize_angle
from .logformat import (
    LogBundle,
    TruthPose,
)
from .model import (
    PTH,
    PX,
    PY,
    bearing_observation,
    full_state_transition,
    motion_noise_cov,
    observation_jacobian,
    transition_jacobian,
)
from .monitor import (
    HealthEvent,
    HealthMonitor,
    MonitorConfig,
    MonitorStatus,
)

# 协方差防护参数
EIG_FLOOR = (1e-9, 1e-9, 1e-10)   # x, y, theta 方差绝对下限
EIG_CEIL = (1e5, 1e5, 1e3)        # 方差上限，防止重置后无限膨胀
ROBUST_GATE = 9.21                # chi2(2) 99%：超过即对该观测鲁棒降权
KIDNAP_RESET_P = (4.0, 4.0, 0.5)  # 绑架后重置的协方差对角 (m^2, m^2, rad^2)
DIVERGE_INFLATE = 4.0             # 发散判定时的协方差膨胀倍数


@dataclass
class StepRecord:
    """时间轴上一拍（一次预测或观测更新）之后的完整状态。"""

    t: float
    kind: str                 # "init" | "odom" | "obs"
    state: Tuple[float, float, float]
    cov: Tuple[float, ...]    # 3x3 按行展平
    nis: Optional[float] = None
    status: str = MonitorStatus.HEALTHY.value
    landmark: Optional[str] = None
    innovation: Optional[Tuple[float, float]] = None
    rejected: bool = False    # 被鲁棒门控极端降权（仍记录 NIS）
    reset: bool = False       # 本拍执行了绑架重置 / 发散膨胀
    truth: Optional[Tuple[float, float, float]] = None
    err_xy: Optional[float] = None
    err_theta_deg: Optional[float] = None

    def cov_matrix(self) -> la.Matrix:
        c = self.cov
        return [list(c[0:3]), list(c[3:6]), list(c[6:9])]


@dataclass
class EKFResult:
    steps: List[StepRecord]
    monitor: HealthMonitor
    landmarks: Dict[str, Tuple[float, float]]
    events: List[HealthEvent] = field(default_factory=list)

    def final_state(self) -> Tuple[float, float, float]:
        return self.steps[-1].state

    def max_pose_std_trace(self) -> float:
        return max(math.sqrt(s.cov[0] + s.cov[4]) for s in self.steps)


class EKFRobotLocalizer:
    """可单步驱动的 EKF 定位器。

    用法::

        loc = EKFRobotLocalizer(alphas=..., meas_var=...)
        loc.initialize(x0, P0)
        loc.predict(d_trans, d_rot, t)
        loc.update(rng, bearing, landmark_xy, t, landmark_id="lm1")
    """

    def __init__(
        self,
        alphas: Tuple[float, float, float, float] = (0.08, 0.04, 0.06, 0.05),
        meas_var: Tuple[float, float] = (0.02, 0.005),
        monitor: Optional[HealthMonitor] = None,
    ) -> None:
        self.alphas = tuple(alphas)
        self.meas_R = la.mat([[meas_var[0], 0.0], [0.0, meas_var[1]]])
        self.monitor = monitor or HealthMonitor(MonitorConfig())
        self.state: la.Vector = [0.0, 0.0, 0.0]
        self.cov: la.Matrix = la.eye(3)
        self._initialized = False

    # ------------------------------------------------------------------
    def initialize(
        self,
        state0: Tuple[float, float, float],
        cov_diag: Tuple[float, float, float],
    ) -> None:
        self.state = [float(state0[0]), float(state0[1]), normalize_angle(state0[2])]
        self.cov = la.mat(
            [
                [max(cov_diag[0], EIG_FLOOR[0]), 0.0, 0.0],
                [0.0, max(cov_diag[1], EIG_FLOOR[1]), 0.0],
                [0.0, 0.0, max(cov_diag[2], EIG_FLOOR[2])],
            ]
        )
        self._initialized = True

    @property
    def pose(self) -> Tuple[float, float, float]:
        return self.state[PX], self.state[PY], self.state[PTH]

    def pose_std_xy(self) -> float:
        return math.sqrt(max(self.cov[PX][PX], 0.0) + max(self.cov[PY][PY], 0.0))

    # ------------------------------------------------------------------
    def predict(self, d_trans: float, d_rot: float, t: float) -> StepRecord:
        if not self._initialized:
            raise RuntimeError("调用 predict 前需要先 initialize")
        f_mat = transition_jacobian(self.state, d_trans)
        q_mat = motion_noise_cov(self.state, d_trans, d_rot, *self.alphas)
        self.state = full_state_transition(self.state, d_trans, d_rot)
        self.cov = la.add(la.mul(la.mul(f_mat, self.cov), la.transpose(f_mat)), q_mat)
        self.cov = self._repair_cov(self.cov)

        self.monitor.record_motion_only(t, self.pose_std_xy())
        return StepRecord(
            t=t,
            kind="odom",
            state=tuple(self.pose),
            cov=tuple(self._flat_cov()),
            status=self.monitor.status.value,
        )

    def update(
        self,
        rng: float,
        bearing: float,
        landmark: Tuple[float, float],
        t: float,
        landmark_id: str = "",
    ) -> StepRecord:
        if not self._initialized:
            raise RuntimeError("调用 update 前需要先 initialize")

        z_pred = bearing_observation(self.state, landmark)
        nu = [rng - z_pred[0], angle_diff(bearing, z_pred[1])]
        h_mat = observation_jacobian(self.state, landmark)

        # --- 名义新息统计（用名义 R，供监控判读） ---
        r_eff = [row[:] for row in self.meas_R]
        s_nom = la.add(la.mul(la.mul(h_mat, self.cov), la.transpose(h_mat)), r_eff)
        nis = self._quadratic_inverse_2x2(s_nom, nu)

        rejected = False
        prev_status = self.monitor.status
        # --- 鲁棒门控：NIS 超 99% 门限 -> 按比例膨胀 R，避免单点野值拉偏 ---
        # 已处于发散/绑架（粘滞）期间不再降权观测——观测正是重新锁定的手段。
        sticky_bad = prev_status in (MonitorStatus.DIVERGED, MonitorStatus.KIDNAPPED)
        if nis > ROBUST_GATE and not sticky_bad:
            scale = min(nis / ROBUST_GATE, 100.0)
            r_eff = la.scale(r_eff, scale)
            rejected = scale > 25.0

        reset = False
        self._apply_kalman_update(h_mat, nu, r_eff)

        # --- 健康判定（用名义 NIS） ---
        status = self.monitor.record_observation(t, nis, self.pose_std_xy())

        # --- 绑架：重置不确定性并用本观测重新更新一次，加速重定位 ---
        if status == MonitorStatus.KIDNAPPED and prev_status != MonitorStatus.KIDNAPPED:
            self._reset_covariance(KIDNAP_RESET_P)
            self._apply_kalman_update(h_mat, nu, self.meas_R)
            reset = True
        # --- 发散：协方差自适应膨胀一拍（不直接相信当前估计的精度） ---
        elif status == MonitorStatus.DIVERGED and prev_status not in (
            MonitorStatus.DIVERGED,
            MonitorStatus.KIDNAPPED,
        ):
            self._reset_covariance(
                tuple(min(c * DIVERGE_INFLATE, EIG_CEIL[i]) for i, c in enumerate(self._diag()))
            )
            reset = True

        return StepRecord(
            t=t,
            kind="obs",
            state=tuple(self.pose),
            cov=tuple(self._flat_cov()),
            nis=nis,
            status=status.value,
            landmark=landmark_id,
            innovation=(nu[0], nu[1]),
            rejected=rejected,
            reset=reset,
        )

    # ------------------------------------------------------------------
    def _apply_kalman_update(
        self, h_mat: la.Matrix, nu: la.Vector, r_mat: la.Matrix
    ) -> None:
        s_mat = la.add(la.mul(la.mul(h_mat, self.cov), la.transpose(h_mat)), r_mat)
        s_inv = la.inverse(s_mat)
        k_mat = la.mul(la.mul(self.cov, la.transpose(h_mat)), s_inv)
        gain = la.mv(k_mat, nu)
        self.state = [
            self.state[0] + gain[0],
            self.state[1] + gain[1],
            normalize_angle(self.state[2] + gain[2]),
        ]
        # Joseph 形式：P = (I - KH) P (I - KH)^T + K R K^T
        i_kh = la.sub(la.eye(3), la.mul(k_mat, h_mat))
        term1 = la.mul(la.mul(i_kh, self.cov), la.transpose(i_kh))
        term2 = la.mul(la.mul(k_mat, r_mat), la.transpose(k_mat))
        self.cov = self._repair_cov(la.add(term1, term2))

    @staticmethod
    def _quadratic_inverse_2x2(s_mat: la.Matrix, nu: la.Vector) -> float:
        a, b = s_mat[0][0], s_mat[0][1]
        c, d = s_mat[1][0], s_mat[1][1]
        det = a * d - b * c
        if det <= 1e-300:
            return float("inf")
        sinv = [[d / det, -b / det], [-c / det, a / det]]
        return (
            nu[0] * (sinv[0][0] * nu[0] + sinv[0][1] * nu[1])
            + nu[1] * (sinv[1][0] * nu[0] + sinv[1][1] * nu[1])
        )

    def _diag(self) -> Tuple[float, float, float]:
        return self.cov[0][0], self.cov[1][1], self.cov[2][2]

    def _reset_covariance(self, diag: Tuple[float, float, float]) -> None:
        self.cov = la.mat(
            [
                [diag[0], 0.0, 0.0],
                [0.0, diag[1], 0.0],
                [0.0, 0.0, diag[2]],
            ]
        )

    @staticmethod
    def _repair_cov(p_mat: la.Matrix) -> la.Matrix:
        """协方差数值防护。

        对称化 -> Jacobi 特征分解 -> 特征值钳制到正区间 -> 重建；
        再按状态分量补对角方差下限（加秩一 PSD 项，不破坏半正定性）。
        注意 Jacobi 返回的特征值顺序任意，因此只能用统一的全局门限。
        """
        p_mat = la.symmetrize(p_mat)
        eigvals, eigvecs = la.jacobi_eigen(p_mat)
        global_floor, global_ceil = 1e-10, 1e5
        clamped = [min(max(v, global_floor), global_ceil) for v in eigvals]
        if any(abs(c - v) > 1e-18 for c, v in zip(clamped, eigvals)):
            d = la.zeros(3, 3)
            for i in range(3):
                d[i][i] = clamped[i]
            p_mat = la.symmetrize(la.mul(la.mul(eigvecs, d), la.transpose(eigvecs)))
        # 分量级方差下限：P_ii += gap 等价于加半正定秩一项 gap e_i e_i^T
        for i, floor_i in enumerate(EIG_FLOOR):
            gap = floor_i - p_mat[i][i]
            if gap > 0.0:
                p_mat[i][i] += gap
        return la.symmetrize(p_mat)

    def _flat_cov(self) -> List[float]:
        return [self.cov[i][j] for i in range(3) for j in range(3)]

    # ------------------------------------------------------------------
    def run_log(self, bundle: LogBundle) -> EKFResult:
        """处理整份日志（乱序/重复时间戳在解析时已稳定排序）。"""
        self.initialize(bundle.initial_state, bundle.initial_cov_diag)
        self.monitor = HealthMonitor(self.monitor.config)

        steps: List[StepRecord] = [
            StepRecord(
                t=bundle.time_span()[0] if (bundle.odometry or bundle.observations) else 0.0,
                kind="init",
                state=tuple(self.pose),
                cov=tuple(self._flat_cov()),
            )
        ]

        odom_i = obs_i = 0
        while odom_i < len(bundle.odometry) or obs_i < len(bundle.observations):
            take_obs = self._take_obs_first(bundle, odom_i, obs_i)
            if take_obs:
                o = bundle.observations[obs_i]
                obs_i += 1
                lm = bundle.landmarks[o.id]
                rec = self.update(o.rng, o.bearing, lm.xy(), o.t, o.id)
            else:
                u = bundle.odometry[odom_i]
                odom_i += 1
                rec = self.predict(u.d_trans, u.d_rot, u.t)
            steps.append(rec)

        _annotate_truth(steps, bundle.truth)
        return EKFResult(
            steps=steps,
            monitor=self.monitor,
            landmarks={k: v.xy() for k, v in bundle.landmarks.items()},
            events=list(self.monitor.events),
        )

    @staticmethod
    def _take_obs_first(
        bundle: LogBundle, odom_i: int, obs_i: int
    ) -> bool:
        """归并两路有序事件。

        同一时刻严格先处理里程计预测、再融合观测；观测在解析阶段已按
        ``(t, id, seq)`` 排序，因此乱序日志会得到完全一致的处理序列。
        （同刻的多个里程计增量按文件先后处理——其顺序在物理上不可交换。）
        """
        if obs_i >= len(bundle.observations):
            return False
        if odom_i >= len(bundle.odometry):
            return True
        return bundle.observations[obs_i].t < bundle.odometry[odom_i].t


def run_filter(bundle: LogBundle, monitor_config: Optional[MonitorConfig] = None) -> EKFResult:
    """便捷函数：用日志自带噪声参数运行 EKF。"""
    loc = EKFRobotLocalizer(
        alphas=bundle.alphas,
        meas_var=bundle.meas_var,
        monitor=HealthMonitor(monitor_config or MonitorConfig()),
    )
    return loc.run_log(bundle)


# ---------------------------------------------------------------------------
# 真值对齐与误差
# ---------------------------------------------------------------------------

def _interpolate_truth(truth: List[TruthPose], t: float) -> Optional[Tuple[float, float, float]]:
    if not truth:
        return None
    if t <= truth[0].t:
        g = truth[0]
        return g.x, g.y, g.theta
    if t >= truth[-1].t:
        g = truth[-1]
        return g.x, g.y, g.theta
    lo, hi = 0, len(truth) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if truth[mid].t <= t:
            lo = mid
        else:
            hi = mid
    a, b = truth[lo], truth[hi]
    frac = (t - a.t) / (b.t - a.t) if b.t > a.t else 0.0
    th = normalize_angle(a.theta + angle_diff(b.theta, a.theta) * frac)
    return (
        a.x + (b.x - a.x) * frac,
        a.y + (b.y - a.y) * frac,
        th,
    )


def _annotate_truth(steps: List[StepRecord], truth: List[TruthPose]) -> None:
    for s in steps:
        g = _interpolate_truth(truth, s.t)
        if g is None:
            continue
        s.truth = g
        s.err_xy = math.hypot(s.state[0] - g[0], s.state[1] - g[1])
        s.err_theta_deg = math.degrees(angle_diff(s.state[2], g[2]))
