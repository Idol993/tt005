import torch
import torch.nn as nn
from typing import Optional, Dict, List


class ModuleDecayCoefficient(nn.Module):
    """
    模块级衰减系数计算器

    为每个模块独立维护可学习的衰减系数，该系数由三个因素决定：
    1. 模块当前梯度范数 - 反映当前优化动态
    2. 模块参数范数 - 反映当前参数规模
    3. 训练整体相位编码 - 反映训练全局状态

    设计要点：
    - 使用小型MLP将上述三个输入映射到衰减系数
    - 每个模块拥有独立的MLP参数，允许学习不同的衰减策略
    - 输出经过sigmoid/tanh映射，确保数值稳定性
    - 维护衰减系数的历史，用于检测振荡频率
    """

    def __init__(
        self,
        num_modules: int,
        phase_encoding_dim: int,
        hidden_dim: int = 32,
        history_length: int = 50,
        min_decay: float = 1e-5,
        max_decay: float = 1.0,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.num_modules = num_modules
        self.phase_encoding_dim = phase_encoding_dim
        self.history_length = history_length
        self.min_decay = min_decay
        self.max_decay = max_decay

        factory_kwargs = {"device": device, "dtype": dtype}

        input_dim = 2 + phase_encoding_dim

        self.module_nets = nn.ModuleList()
        for _ in range(num_modules):
            net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim, **factory_kwargs),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim, **factory_kwargs),
                nn.SiLU(),
                nn.Linear(hidden_dim, 2, **factory_kwargs),
            )
            self.module_nets.append(net)

        self.register_buffer(
            "decay_history",
            torch.zeros(history_length, num_modules, **factory_kwargs),
        )
        self.register_buffer(
            "history_ptr",
            torch.tensor(0, dtype=torch.long, device=device if device else torch.device("cpu")),
        )

    def forward(
        self,
        grad_norms: torch.Tensor,
        param_norms: torch.Tensor,
        phase_encoding: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算各模块的衰减系数

        Args:
            grad_norms: 各模块梯度范数，形状 [num_modules]
            param_norms: 各模块参数范数，形状 [num_modules]
            phase_encoding: 训练相位编码，形状 [phase_encoding_dim] 或 [batch, phase_encoding_dim]

        Returns:
            decay_coefficients: 各模块衰减系数，形状 [num_modules]
            decay_phase_shifts: 各模块相位偏移（用于共振抑制），形状 [num_modules]
        """
        assert grad_norms.shape[-1] == self.num_modules
        assert param_norms.shape[-1] == self.num_modules

        if phase_encoding.dim() == 1:
            phase_encoding = phase_encoding.unsqueeze(0)

        batch_size = phase_encoding.shape[0]
        decay_coeffs = []
        phase_shifts = []

        for i in range(self.num_modules):
            module_input = torch.cat(
                [
                    grad_norms[..., i : i + 1].expand(batch_size, -1),
                    param_norms[..., i : i + 1].expand(batch_size, -1),
                    phase_encoding,
                ],
                dim=-1,
            )

            net_out = self.module_nets[i](module_input)

            raw_coeff = torch.sigmoid(net_out[..., 0])
            raw_phase = torch.tanh(net_out[..., 1]) * torch.pi

            coeff = self.min_decay + (self.max_decay - self.min_decay) * raw_coeff

            decay_coeffs.append(coeff)
            phase_shifts.append(raw_phase)

        decay_coeffs = torch.stack(decay_coeffs, dim=-1)
        phase_shifts = torch.stack(phase_shifts, dim=-1)

        if batch_size == 1:
            decay_coeffs = decay_coeffs.squeeze(0)
            phase_shifts = phase_shifts.squeeze(0)
            self._update_history(decay_coeffs.detach())

        return decay_coeffs, phase_shifts

    def _update_history(self, decay_coeffs: torch.Tensor) -> None:
        """更新衰减系数历史记录"""
        self.decay_history[self.history_ptr] = decay_coeffs
        self.history_ptr = (self.history_ptr + 1) % self.history_length

    def get_oscillation_frequencies(self) -> torch.Tensor:
        """
        估计各模块衰减系数的振荡频率

        使用自相关分析估计每个模块衰减系数的主导振荡频率。

        Returns:
            frequencies: 各模块振荡频率估计，形状 [num_modules]，值域 [0, 1]（归一化频率）
        """
        history = self.decay_history
        centered = history - history.mean(dim=0, keepdim=True)

        freqs = []
        for i in range(self.num_modules):
            series = centered[:, i]
            if series.abs().sum() < 1e-8:
                freqs.append(torch.tensor(0.0, device=series.device))
                continue

            max_lag = min(self.history_length // 2, 20)
            autocorr = []
            for lag in range(1, max_lag + 1):
                c = (series[:-lag] * series[lag:]).mean()
                autocorr.append(c)

            autocorr = torch.stack(autocorr)

            if autocorr.abs().sum() < 1e-8:
                freqs.append(torch.tensor(0.0, device=series.device))
                continue

            peaks = []
            for j in range(1, len(autocorr) - 1):
                if (
                    autocorr[j] > autocorr[j - 1]
                    and autocorr[j] > autocorr[j + 1]
                    and autocorr[j] > 0
                ):
                    peaks.append(j + 1)

            if len(peaks) == 0:
                freqs.append(torch.tensor(0.0, device=series.device))
            else:
                period = peaks[0]
                freq = 1.0 / max(period, 1)
                freqs.append(torch.tensor(freq, device=series.device))

        return torch.stack(freqs)

    def reset_history(self) -> None:
        """重置历史记录"""
        self.decay_history.zero_()
        self.history_ptr.zero_()

    def extra_repr(self) -> str:
        return (
            f"num_modules={self.num_modules}, "
            f"phase_encoding_dim={self.phase_encoding_dim}, "
            f"history_length={self.history_length}, "
            f"decay_range=[{self.min_decay}, {self.max_decay}]"
        )
