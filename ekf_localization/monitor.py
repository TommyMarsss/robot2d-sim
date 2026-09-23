"""滤波器健康监控：基于归一化新息平方（NIS）检测发散与绑架。

判据（多条件、有迟滞，避免单次野值误报）：

- **NIS**：``nu^T S^{-1} nu``。观测模型 2 维时，理论上服从 chi2(2)，
  95% 分位为 5.99，99% 分位为 9.21。
- 连续 ``warn_count`` 次 NIS 超过 ``nis_warn``（默认取 99% 分位），
  或滑动窗口均值超过 ``nis_mean_warn``（持续偏大，不是单次跳变），
  判为 :attr:`MonitorStatus.DIVERGED`（估计可能已发散）。
- NIS 极大（超过 ``kidnap_nis``，默认 chi2(2) 的 1-1e-6 分位 ~27.6）
  且立即连续触发，判为 :attr:`MonitorStatus.KIDNAPPED`
  （机器人被意外移动 / 绑架）。
- 恢复：观测重新连续 ``recover_count`` 次落在正常区间
  （NIS < ``nis_recover``，默认取 90% 分位 4.61），状态回到健康。

无观测期间不更新统计，只保留上一次状态；运动学上的“纯航位推算误差
持续增长”由另一个判据兜底（见 :class:`MonitorConfig` 的
``max_pose_std``）：若长时间无观测修正、位置标准差膨胀到阈值以上，
同样报告发散风险。
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Deque, List, Optional


class MonitorStatus(str, Enum):
    HEALTHY = "healthy"
    WARN = "warn"
    DIVERGED = "diverged"
    KIDNAPPED = "kidnapped"
    RECOVERED = "recovered"


# chi2(2) 分位（观测维数为 2：距离 + 方位）
CHI2_2 = {
    0.90: 4.605,
    0.95: 5.991,
    0.99: 9.210,
    0.999: 13.816,
    1e-6: 27.631,
}


@dataclass
class MonitorConfig:
    nis_warn: float = CHI2_2[0.99]            # 单点警戒
    nis_mean_warn: float = CHI2_2[0.95]       # 窗口均值警戒（持续偏大）
    kidnap_nis: float = CHI2_2[1e-6]          # 单点极端 -> 疑似绑架
    nis_recover: float = CHI2_2[0.90]         # 恢复门槛
    warn_count: int = 4                       # 连续超限 -> 发散
    kidnap_count: int = 2                     # 连续极端 -> 绑架
    recover_count: int = 5                    # 连续正常 -> 恢复
    window: int = 12                          # NIS 滑动窗口
    max_pose_std: float = 5.0                 # 位置标准差膨胀上限（米）
    kidnap_max_gap: float = 5.0               # 距上次观测超过该秒数时极端新息判为发散而非绑架


@dataclass
class HealthEvent:
    t: float
    status: MonitorStatus
    nis: Optional[float]
    detail: str


@dataclass
class HealthMonitor:
    config: MonitorConfig = field(default_factory=MonitorConfig)

    status: MonitorStatus = MonitorStatus.HEALTHY
    nis_history: List[float] = field(default_factory=list)
    events: List[HealthEvent] = field(default_factory=list)
    first_flag_t: Optional[float] = None  # 最近一次进入异常的时刻（供标注）

    _window: Deque[float] = field(default_factory=deque)
    _bad_streak: int = 0
    _extreme_streak: int = 0
    _good_streak: int = 0
    _last_obs_t: Optional[float] = None
    _stale_grace: int = 0  # 观测黑窗后的若干帧内禁止判绑架

    # ------------------------------------------------------------------
    def record_observation(
        self, t: float, nis: float, pose_std_xy: float
    ) -> MonitorStatus:
        """喂入一次观测更新的结果，返回更新后的状态。

        异常状态（DIVERGED / KIDNAPPED）具有**粘滞性**：一旦判定，不会被
        单次正常观测立即取消，而需要连续 ``recover_count`` 次正常新息才宣告
        RECOVERED（记录一个恢复事件）并回到 HEALTHY。

        绑架与发散的区分：连续观测中（距上次观测 <= ``kidnap_max_gap`` 秒）
        突然出现极端新息判绑架；长时间观测黑窗后重捕获的极端新息更可能是
        航位推算漂移导致的发散，不归为绑架。
        """
        cfg = self.config
        self.nis_history.append(nis)

        self._window.append(nis)
        if len(self._window) > cfg.window:
            self._window.popleft()

        gap = None if self._last_obs_t is None else t - self._last_obs_t
        self._last_obs_t = t
        recent_gap = gap is not None and gap <= cfg.kidnap_max_gap
        if gap is not None and gap > cfg.kidnap_max_gap:
            # 观测黑窗重捕获：给若干帧宽限，期间只判发散，不判绑架
            self._stale_grace = max(self._stale_grace, cfg.kidnap_count)
        allow_kidnap = recent_gap and self._stale_grace == 0
        if self._stale_grace > 0:
            self._stale_grace -= 1

        extreme = nis > cfg.kidnap_nis
        bad = nis > cfg.nis_warn
        window_bad = (
            len(self._window) >= max(cfg.window // 2, 2)
            and self._window_mean() > cfg.nis_mean_warn
        )
        pose_bad = pose_std_xy > cfg.max_pose_std

        self._extreme_streak = self._extreme_streak + 1 if extreme else 0
        self._bad_streak = self._bad_streak + 1 if (bad or window_bad or pose_bad) else 0
        self._good_streak = self._good_streak + 1 if nis <= cfg.nis_recover else 0

        prev = self.status

        # 恢复判定优先：当前点正常且连续正常次数达标，即使滑动窗口里还
        # 残留历史大值（窗口均值滞后）也立即恢复。
        if prev in (MonitorStatus.DIVERGED, MonitorStatus.KIDNAPPED):
            if not (bad or extreme) and self._good_streak >= cfg.recover_count:
                self._emit(t, MonitorStatus.RECOVERED, nis, pose_std_xy)
                self._bad_streak = self._extreme_streak = self._good_streak = 0
                self.status = MonitorStatus.HEALTHY
                self.first_flag_t = None
                return self.status
            new_status = prev
        elif prev == MonitorStatus.WARN and not (bad or window_bad or pose_bad):
            # 单点正常即静默解除警戒
            self._bad_streak = self._extreme_streak = self._good_streak = 0
            self.status = MonitorStatus.HEALTHY
            return self.status
        else:
            new_status = prev

        if (
            self._extreme_streak >= cfg.kidnap_count
            and allow_kidnap
            and prev != MonitorStatus.DIVERGED
        ):
            new_status = MonitorStatus.KIDNAPPED
        elif (
            (self._bad_streak >= cfg.warn_count
             or (self._extreme_streak >= cfg.kidnap_count and not allow_kidnap))
            and (bad or extreme or pose_bad)
            and prev != MonitorStatus.KIDNAPPED
        ):
            # 持续异常的“当下”仍异常才判发散；不能只凭窗口里的历史大值触发，
            # 否则会在滤波器已经快速重收敛之后误报。
            new_status = MonitorStatus.DIVERGED
        elif bad or window_bad or pose_bad:
            if prev not in (MonitorStatus.DIVERGED, MonitorStatus.KIDNAPPED):
                new_status = MonitorStatus.WARN

        # WARN 只是瞬时观察态（统计上健康数据也会偶发超 99% 门限），
        # 不写入事件列表；只有发散/绑架/恢复构成需要报告的判定。
        if new_status != prev and new_status != MonitorStatus.WARN:
            self._emit(t, new_status, nis, pose_std_xy)
            if new_status in (MonitorStatus.DIVERGED, MonitorStatus.KIDNAPPED):
                self.first_flag_t = t
            self.status = new_status
        elif new_status == MonitorStatus.WARN and prev == MonitorStatus.HEALTHY:
            self.status = MonitorStatus.WARN
        return self.status

    def _emit(
        self,
        t: float,
        status: MonitorStatus,
        nis: float,
        pose_std: float,
        detail: Optional[str] = None,
    ) -> None:
        self.events.append(
            HealthEvent(
                t, status, nis, detail or self._describe(status, nis, pose_std)
            )
        )

    def record_motion_only(self, t: float, pose_std_xy: float) -> MonitorStatus:
        """无观测更新的一拍：协方差膨胀超过上限时判发散风险。"""
        if pose_std_xy <= self.config.max_pose_std:
            return self.status
        if self.status in (MonitorStatus.DIVERGED, MonitorStatus.KIDNAPPED):
            return self.status
        self.status = MonitorStatus.DIVERGED
        self.first_flag_t = t
        self._emit(
            t,
            MonitorStatus.DIVERGED,
            float("nan"),
            pose_std_xy,
            detail=(
                f"长时间无有效观测，位置标准差 {pose_std_xy:.2f}m "
                f"超过上限 {self.config.max_pose_std:.2f}m，定位可能已发散"
            ),
        )
        return self.status

    # ------------------------------------------------------------------
    def _window_mean(self) -> float:
        if not self._window:
            return 0.0
        return sum(self._window) / len(self._window)

    def _describe(self, status: MonitorStatus, nis: float, pose_std: float) -> str:
        if status == MonitorStatus.KIDNAPPED:
            return (
                f"连续极端新息 NIS={nis:.1f} (>{self.config.kidnap_nis:.1f})，"
                "疑似机器人被意外移动（绑架）"
            )
        if status == MonitorStatus.DIVERGED:
            return (
                f"新息持续偏大 NIS={nis:.1f}，窗口均值={self._window_mean():.1f}，"
                f"位置标准差={pose_std:.2f}m，定位可能已发散"
            )
        if status == MonitorStatus.WARN:
            return f"新息异常 NIS={nis:.1f}，保持观察"
        if status == MonitorStatus.RECOVERED:
            return "连续正常观测，滤波已重新收敛（恢复）"
        return "状态正常"

    def summary(self) -> str:
        if not self.events:
            return "全程健康，未检测到发散或绑架。"
        lines = []
        for e in self.events:
            lines.append(f"t={e.t:7.2f}s  {e.status.value:9s}  {e.detail}")
        return "\n".join(lines)
