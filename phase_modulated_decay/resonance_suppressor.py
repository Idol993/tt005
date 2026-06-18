import torch
import torch.nn as nn
from typing import Optional, List, Tuple
import math


class ResonanceSuppressor(nn.Module):
    """
    共振抑制机制

    检测并抑制相邻模块之间衰减系数振荡的同频共振现象。
    当两个模块的振荡频率接近时，自动调整其中一者的相位偏移，
    避免因同频衰减导致的参数更新协同共振（同时过度压缩或膨胀）。

    核心机制：
    1. 频率耦合检测：通过频率相似度矩阵识别可能共振的模块对
    2. 相位差分析：检测危险的同相或反相关系
    3. 自适应相位偏移：对检测到的共振对施加相位调制
    4. 衰减幅度修正：在共振时微调衰减强度以破坏协同效应
    """

    def __init__(
        self,
        num_modules: int,
        frequency_threshold: float = 0.15,
        phase_threshold: float = math.pi / 4,
        max_phase_shift: float = math.pi / 2,
        suppression_strength: float = 0.5,
        adaptivity_rate: float = 0.1,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.num_modules = num_modules
        self.frequency_threshold = frequency_threshold
        self.phase_threshold = phase_threshold
        self.max_phase_shift = max_phase_shift
        self.suppression_strength = suppression_strength
        self.adaptivity_rate = adaptivity_rate

        factory_kwargs = {"device": device, "dtype": dtype}

        adjacency = self._build_adjacency_matrix(num_modules)
        self.register_buffer(
            "adjacency",
            torch.tensor(adjacency, **factory_kwargs),
        )

        self.register_buffer(
            "accumulated_phase_shifts",
            torch.zeros(num_modules, **factory_kwargs),
        )

        self.register_buffer(
            "resonance_counters",
            torch.zeros(num_modules, num_modules, **factory_kwargs),
        )

        phase_shift_params = torch.zeros(num_modules, **factory_kwargs)
        self.correction_phases = nn.Parameter(phase_shift_params)

        amplitude_modulation = torch.ones(num_modules, **factory_kwargs)
        self.amplitude_modulation = nn.Parameter(amplitude_modulation)

    def _build_adjacency_matrix(self, num_modules: int) -> List[List[float]]:
        """
        构建模块邻接矩阵

        默认使用链式邻接：模块i与i-1和i+1相邻。
        用户可以通过修改此矩阵自定义模块拓扑。
        """
        adjacency = [[0.0] * num_modules for _ in range(num_modules)]
        for i in range(num_modules):
            for j in range(num_modules):
                if abs(i - j) == 1:
                    adjacency[i][j] = 1.0
        return adjacency

    def set_adjacency(self, adjacency: torch.Tensor) -> None:
        """
        自定义模块邻接关系

        Args:
            adjacency: 形状 [num_modules, num_modules] 的邻接矩阵
        """
        assert adjacency.shape == (self.num_modules, self.num_modules)
        self.adjacency.copy_(adjacency.to(self.adjacency.device))

    def forward(
        self,
        decay_coefficients: torch.Tensor,
        base_phase_shifts: torch.Tensor,
        oscillation_frequencies: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        执行共振抑制，输出修正后的衰减系数和相位偏移

        Args:
            decay_coefficients: 原始衰减系数，形状 [num_modules]
            base_phase_shifts: 原始相位偏移，形状 [num_modules]
            oscillation_frequencies: 各模块振荡频率估计，形状 [num_modules]

        Returns:
            corrected_decays: 修正后的衰减系数，形状 [num_modules]
            corrected_phases: 修正后的相位偏移，形状 [num_modules]
            resonance_map: 共振强度图，形状 [num_modules, num_modules]
        """
        freq_diff = torch.abs(
            oscillation_frequencies.unsqueeze(0) - oscillation_frequencies.unsqueeze(1)
        )

        freq_similarity = torch.exp(-(freq_diff ** 2) / (2 * self.frequency_threshold ** 2))

        phase_diff = torch.abs(
            base_phase_shifts.unsqueeze(0) - base_phase_shifts.unsqueeze(1)
        )
        phase_diff = torch.minimum(phase_diff, 2 * math.pi - phase_diff)

        in_phase_risk = torch.exp(-(phase_diff ** 2) / (2 * self.phase_threshold ** 2))
        anti_phase_risk = torch.exp(
            -((phase_diff - math.pi) ** 2) / (2 * self.phase_threshold ** 2)
        )
        phase_risk = 0.5 * (in_phase_risk + anti_phase_risk)

        resonance_map = freq_similarity * phase_risk * self.adjacency
        resonance_map = resonance_map - torch.diag(torch.diag(resonance_map))

        self.resonance_counters.mul_(1.0 - self.adaptivity_rate)
        self.resonance_counters.add_(self.adaptivity_rate * resonance_map.detach())

        per_module_resonance = self.resonance_counters.sum(dim=1)

        phase_corrections = torch.zeros_like(base_phase_shifts)
        for i in range(self.num_modules):
            for j in range(self.num_modules):
                if i == j or self.adjacency[i, j] < 0.5:
                    continue
                if self.resonance_counters[i, j] > 0.3:
                    if per_module_resonance[i] >= per_module_resonance[j]:
                        sign = 1.0 if torch.rand(1).item() > 0.5 else -1.0
                        target_shift = sign * self.max_phase_shift * min(
                            self.resonance_counters[i, j], 1.0
                        )
                        current_correction = self.correction_phases[i]
                        new_correction = (
                            1.0 - self.adaptivity_rate
                        ) * current_correction + self.adaptivity_rate * target_shift
                        phase_corrections[i] += new_correction - current_correction

        self.accumulated_phase_shifts.add_(phase_corrections.detach())
        self.accumulated_phase_shifts = torch.clamp(
            self.accumulated_phase_shifts,
            -2 * math.pi,
            2 * math.pi,
        )

        corrected_phases = base_phase_shifts + self.correction_phases + phase_corrections

        resonance_suppression = 1.0 - self.suppression_strength * per_module_resonance.clamp(
            0.0, 1.0
        )
        amplitude_factor = self.amplitude_modulation.sigmoid() * 0.5 + 0.5
        corrected_decays = decay_coefficients * resonance_suppression * amplitude_factor

        return corrected_decays, corrected_phases, resonance_map

    def get_resonance_summary(self) -> dict:
        """获取当前共振状态摘要"""
        per_module = self.resonance_counters.sum(dim=1)
        active_pairs = (self.resonance_counters > 0.3).nonzero(as_tuple=False)
        return {
            "total_resonance": self.resonance_counters.sum().item(),
            "per_module_resonance": per_module.detach().cpu().tolist(),
            "active_resonance_pairs": active_pairs.detach().cpu().tolist(),
            "accumulated_phases": self.accumulated_phase_shifts.detach().cpu().tolist(),
        }

    def reset_state(self) -> None:
        """重置内部状态"""
        self.accumulated_phase_shifts.zero_()
        self.resonance_counters.zero_()

    def extra_repr(self) -> str:
        return (
            f"num_modules={self.num_modules}, "
            f"frequency_threshold={self.frequency_threshold:.3f}, "
            f"phase_threshold={self.phase_threshold:.3f}, "
            f"suppression_strength={self.suppression_strength:.3f}"
        )
