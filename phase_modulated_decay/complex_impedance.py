import torch
import torch.nn as nn
from typing import Optional, Tuple
import math


class ComplexImpedanceMapper(nn.Module):
    """
    复数域阻抗映射器

    将每个模块的衰减系数映射到复数域，通过计算复阻抗来决定实际施加的衰减幅度。

    核心思想：
    - 将衰减系数视为复平面上的阻抗 Z = R + jX
        - 实部 R（电阻）：耗散性衰减，对应传统L2正则化
        - 虚部 X（电抗）：无功/储能性调节，产生相位调制效应
    - 衰减幅度由阻抗的模 |Z| 决定
    - 衰减的相位特性由辐角 arg(Z) 决定
    - 通过复域计算可以更精细地控制衰减的动态特性

    设计要点：
    - 实部始终非负（保证能量耗散方向正确）
    - 虚部可正可负（允许感性/容性调节）
    - 使用softplus保证实部正定性
    - 最终衰减幅度 = |Z| * cos(arg(Z)) = R（仅实部贡献有效衰减）
    - 但虚部通过与相位偏移的交互影响衰减的动态变化
    """

    def __init__(
        self,
        num_modules: int,
        phase_coupling_strength: float = 0.3,
        reactance_range: float = 2.0,
        impedance_softplus_beta: float = 1.0,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.num_modules = num_modules
        self.phase_coupling_strength = phase_coupling_strength
        self.reactance_range = reactance_range
        self.impedance_softplus_beta = impedance_softplus_beta

        factory_kwargs = {"device": device, "dtype": dtype}

        self.reactance_bias = nn.Parameter(
            torch.zeros(num_modules, **factory_kwargs)
        )

        self.phase_to_reactance = nn.Parameter(
            torch.randn(num_modules, **factory_kwargs) * 0.1
        )

        self.coupling_weights = nn.Parameter(
            torch.randn(num_modules, num_modules, **factory_kwargs) * 0.01
        )

    def forward(
        self,
        decay_magnitudes: torch.Tensor,
        phase_shifts: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        将衰减系数映射到复阻抗域，计算实际衰减幅度

        Args:
            decay_magnitudes: 各模块衰减幅度（实部基础值），形状 [num_modules]
            phase_shifts: 各模块相位偏移，形状 [num_modules]

        Returns:
            actual_decays: 实际施加的衰减幅度（有效衰减 = R），形状 [num_modules]
            complex_impedances: 复阻抗 Z = R + jX，形状 [num_modules, 2] (R, X)
            impedance_info: 额外信息字典包含 |Z|, arg(Z) 等
        """
        resistance = nn.functional.softplus(
            decay_magnitudes, beta=self.impedance_softplus_beta
        )

        phase_contribution = self.phase_to_reactance * phase_shifts
        coupling_contribution = self.coupling_weights.sum(dim=0) * 0.1

        raw_reactance = (
            self.reactance_bias
            + phase_contribution
            + self.phase_coupling_strength * coupling_contribution
        )
        reactance = torch.tanh(raw_reactance / self.reactance_range) * self.reactance_range

        impedance_magnitude = torch.sqrt(resistance ** 2 + reactance ** 2)
        impedance_angle = torch.atan2(reactance, resistance)

        effective_decay = resistance

        complex_impedances = torch.stack([resistance, reactance], dim=-1)

        return effective_decay, complex_impedances, impedance_magnitude, impedance_angle

    def get_impedance_summary(
        self,
        decay_magnitudes: torch.Tensor,
        phase_shifts: torch.Tensor,
    ) -> dict:
        """
        获取详细的阻抗分析摘要，用于调试和可视化
        """
        effective_decay, complex_impedances, mag, angle = self.forward(
            decay_magnitudes, phase_shifts
        )

        return {
            "resistance": complex_impedances[:, 0].detach().cpu().tolist(),
            "reactance": complex_impedances[:, 1].detach().cpu().tolist(),
            "impedance_magnitude": mag.detach().cpu().tolist(),
            "impedance_angle_deg": (angle * 180.0 / math.pi).detach().cpu().tolist(),
            "effective_decay": effective_decay.detach().cpu().tolist(),
            "quality_factor": (
                (complex_impedances[:, 1].abs() / (complex_impedances[:, 0] + 1e-8))
                .detach()
                .cpu()
                .tolist()
            ),
        }

    def extra_repr(self) -> str:
        return (
            f"num_modules={self.num_modules}, "
            f"phase_coupling_strength={self.phase_coupling_strength:.3f}, "
            f"reactance_range={self.reactance_range:.3f}"
        )
