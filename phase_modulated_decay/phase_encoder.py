import torch
import torch.nn as nn
import math
from typing import Tuple, Optional


class CyclicPhaseEncoder(nn.Module):
    """
    循环相位编码器

    将训练进度 (0~1) 编码为高维循环相位表示，用于捕获训练的整体相位信息。
    使用多频率正弦/余弦基函数构建连续且具有周期性的相位编码。

    核心思想：
    - 训练可以看作多个嵌套的循环过程（快速参数适应、中速特征学习、慢速结构形成）
    - 使用不同频率的基函数同时编码多个时间尺度的相位
    - 输出相位编码可用于调制各模块的衰减强度
    """

    def __init__(
        self,
        num_frequencies: int = 8,
        base_frequency: float = 2.0 * math.pi,
        frequency_multiplier: float = 2.0,
        include_linear: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.num_frequencies = num_frequencies
        self.base_frequency = base_frequency
        self.frequency_multiplier = frequency_multiplier
        self.include_linear = include_linear

        factory_kwargs = {"device": device, "dtype": dtype}

        frequencies = torch.tensor(
            [
                base_frequency * (frequency_multiplier ** i)
                for i in range(num_frequencies)
            ],
            **factory_kwargs
        )
        self.register_buffer("frequencies", frequencies)

        phases = torch.randn(num_frequencies, **factory_kwargs) * 0.1
        self.phases = nn.Parameter(phases)

        self.output_dim = num_frequencies * 2 + (1 if include_linear else 0)

    def forward(self, training_progress: torch.Tensor) -> torch.Tensor:
        """
        编码训练进度为循环相位表示

        Args:
            training_progress: 训练进度张量，形状 [batch] 或标量，值域 [0, 1]

        Returns:
            相位编码张量，形状 [..., output_dim]
        """
        if training_progress.dim() == 0:
            training_progress = training_progress.unsqueeze(0)

        training_progress = training_progress.clamp(0.0, 1.0)

        angle = torch.einsum("...,f->...f", training_progress, self.frequencies)
        angle = angle + self.phases

        sin_components = torch.sin(angle)
        cos_components = torch.cos(angle)

        components = [sin_components, cos_components]

        if self.include_linear:
            components.append(training_progress.unsqueeze(-1))

        encoding = torch.cat(components, dim=-1)
        return encoding

    def get_phase_spectrum(
        self, training_progress: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取各频率分量的振幅谱和相位谱，用于可视化分析

        Returns:
            (amplitudes, phases) 各形状 [..., num_frequencies]
        """
        if training_progress.dim() == 0:
            training_progress = training_progress.unsqueeze(0)

        angle = torch.einsum("...,f->...f", training_progress, self.frequencies)
        angle = angle + self.phases

        amplitudes = torch.ones_like(angle)
        phases_out = angle % (2 * math.pi)

        return amplitudes, phases_out

    def extra_repr(self) -> str:
        return (
            f"num_frequencies={self.num_frequencies}, "
            f"base_frequency={self.base_frequency:.3f}, "
            f"frequency_multiplier={self.frequency_multiplier}, "
            f"output_dim={self.output_dim}"
        )
