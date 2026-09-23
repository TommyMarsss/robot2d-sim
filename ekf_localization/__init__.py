"""2D 平面位姿 (x, y, theta) 扩展卡尔曼滤波定位包。

模块划分
--------
- :mod:`linalg`  : 纯 Python 的小矩阵运算（仅依赖标准库）。
- :mod:`angles`  : 角度归一化 / 环绕处理。
- :mod:`model`   : 里程计运动模型与距离-方位观测模型（雅可比）。
- :mod:`ekf`      : EKF 主体、协方差数值防护、乱序日志处理。
- :mod:`monitor`  : 新息监控、发散 / 绑架检测与恢复。
- :mod:`logformat`: 传感器日志的解析与序列化（JSON 行）。

报告生成（HTML）位于独立模块 :mod:`report`，核心滤波不依赖它。
"""

from .angles import normalize_angle
from .ekf import EKFResult, EKFRobotLocalizer, StepRecord
from .logformat import Landmark, LogBundle, Observation, OdometryIncrement, parse_log
from .model import bearing_observation, full_state_transition
from .monitor import HealthMonitor, MonitorConfig, MonitorStatus

__all__ = [
    "Landmark",
    "LogBundle",
    "Observation",
    "OdometryIncrement",
    "EKFResult",
    "EKFRobotLocalizer",
    "StepRecord",
    "HealthMonitor",
    "MonitorConfig",
    "MonitorStatus",
    "bearing_observation",
    "full_state_transition",
    "normalize_angle",
    "parse_log",
]
