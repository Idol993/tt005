"""
模块化相位调制权重衰减器 (Phase-Modulated Weight Decay)

一个为多模块神经网络训练设计的智能权重衰减系统，具有以下特性：
- 每个模块独立维护可学习的衰减系数
- 衰减系数依赖于梯度范数、参数范数和训练相位
- 共振抑制机制防止相邻模块同频衰减协同共振
- 复数域阻抗映射确定实际衰减幅度
- 总衰减能量守恒约束
"""

from .phase_encoder import CyclicPhaseEncoder
from .module_coefficient import ModuleDecayCoefficient
from .resonance_suppressor import ResonanceSuppressor
from .complex_impedance import ComplexImpedanceMapper
from .energy_conservation import EnergyConservationRegulator
from .decay import PhaseModulatedWeightDecay

__all__ = [
    "CyclicPhaseEncoder",
    "ModuleDecayCoefficient",
    "ResonanceSuppressor",
    "ComplexImpedanceMapper",
    "EnergyConservationRegulator",
    "PhaseModulatedWeightDecay",
    "__version__",
]

__version__ = "0.1.0"
