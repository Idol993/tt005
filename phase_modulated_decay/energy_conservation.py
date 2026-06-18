import torch
import torch.nn as nn
from typing import Optional, Tuple


class EnergyConservationRegulator(nn.Module):
    """
    能量守恒调节器

    确保整个系统的总衰减能量守恒。即所有模块施加的加权衰减总量保持在目标范围内，
    防止某些模块衰减增强导致整体正则化过强或过弱。

    核心机制：
    1. 计算当前总衰减能量 E = Σ w_i * λ_i * ||θ_i||²
       其中 w_i 是模块权重，λ_i 是衰减系数，θ_i 是模块参数
    2. 将 E 与目标能量 E_target 比较
    3. 通过全局缩放因子和模块间重新分配两种方式调节
    4. 使用滑动平均平滑调节过程，避免剧烈波动

    设计要点：
    - 支持软约束（轻微偏离允许）和硬约束（强制满足）
    - 全局缩放保证总量守恒，但可能破坏模块间相对关系
    - 模块间重新分配（保持总和不变）可以精细调节但计算复杂
    - 默认采用混合策略：先全局粗调，再模块间细调
    """

    def __init__(
        self,
        num_modules: int,
        target_total_energy: Optional[float] = None,
        energy_tolerance: float = 0.1,
        ema_decay: float = 0.95,
        max_scale_factor: float = 2.0,
        redistribution_rate: float = 0.3,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.num_modules = num_modules
        self.target_total_energy = target_total_energy
        self.energy_tolerance = energy_tolerance
        self.ema_decay = ema_decay
        self.max_scale_factor = max_scale_factor
        self.redistribution_rate = redistribution_rate

        factory_kwargs = {"device": device, "dtype": dtype}

        module_weights = torch.ones(num_modules, **factory_kwargs) / num_modules
        self.module_weights = nn.Parameter(module_weights)

        self.register_buffer(
            "ema_total_energy",
            torch.tensor(0.0, **factory_kwargs),
        )
        self.register_buffer(
            "ema_scale_factor",
            torch.tensor(1.0, **factory_kwargs),
        )
        self.register_buffer(
            "energy_history",
            torch.zeros(100, **factory_kwargs),
        )
        self.register_buffer(
            "history_ptr",
            torch.tensor(0, dtype=torch.long, device=device if device else torch.device("cpu")),
        )

    def forward(
        self,
        decay_coefficients: torch.Tensor,
        param_norms: torch.Tensor,
        grad_norms: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        调节各模块衰减系数，保证总衰减能量守恒

        Args:
            decay_coefficients: 当前各模块衰减系数，形状 [num_modules]
            param_norms: 各模块参数范数 ||θ_i||，形状 [num_modules]
            grad_norms: 各模块梯度范数（可选，用于智能重新分配），形状 [num_modules]

        Returns:
            regulated_decays: 调节后的衰减系数，形状 [num_modules]
            info: 包含能量统计信息的字典
        """
        weights = torch.softmax(self.module_weights, dim=0)

        per_module_energy = weights * decay_coefficients * (param_norms ** 2)
        total_energy = per_module_energy.sum()

        self.ema_total_energy.mul_(self.ema_decay).add_(
            (1.0 - self.ema_decay) * total_energy.detach()
        )

        self.energy_history[self.history_ptr] = total_energy.detach()
        self.history_ptr = (self.history_ptr + 1) % 100

        if self.target_total_energy is None:
            if self.ema_total_energy.item() < 1e-8:
                target = total_energy.detach().clamp(min=1e-6)
            else:
                target = self.ema_total_energy.detach()
        else:
            target = torch.tensor(
                self.target_total_energy,
                device=decay_coefficients.device,
                dtype=decay_coefficients.dtype,
            )

        deviation_ratio = (total_energy - target).abs() / (target + 1e-8)

        if deviation_ratio > self.energy_tolerance:
            raw_scale = target / (total_energy + 1e-8)
            scale_factor = torch.clamp(
                raw_scale,
                1.0 / self.max_scale_factor,
                self.max_scale_factor,
            )

            self.ema_scale_factor.mul_(self.ema_decay).add_(
                (1.0 - self.ema_decay) * scale_factor.detach()
            )

            scaled_decays = decay_coefficients * self.ema_scale_factor
        else:
            scaled_decays = decay_coefficients

        if grad_norms is not None and self.redistribution_rate > 0:
            importance = grad_norms / (grad_norms.sum() + 1e-8)
            uniform = torch.ones_like(importance) / self.num_modules

            target_distribution = (
                1.0 - self.redistribution_rate
            ) * weights + self.redistribution_rate * importance
            target_distribution = target_distribution / (target_distribution.sum() + 1e-8)

            current_total = (scaled_decays * weights).sum() + 1e-8
            redistributed_decays = scaled_decays * (target_distribution / (weights + 1e-8))

            new_total = (redistributed_decays * weights).sum() + 1e-8
            redistributed_decays = redistributed_decays * (current_total / new_total)

            regulated_decays = (
                1.0 - self.redistribution_rate
            ) * scaled_decays + self.redistribution_rate * redistributed_decays
        else:
            regulated_decays = scaled_decays

        final_energy = (weights * regulated_decays * (param_norms ** 2)).sum()

        info = {
            "initial_energy": total_energy.item(),
            "target_energy": target.item(),
            "final_energy": final_energy.item(),
            "deviation_ratio": deviation_ratio.item(),
            "scale_factor": self.ema_scale_factor.item(),
            "per_module_energy": per_module_energy.detach().cpu().tolist(),
            "module_weights": weights.detach().cpu().tolist(),
        }

        return regulated_decays, info

    def set_target_energy(self, target: float) -> None:
        """设置目标总衰减能量"""
        self.target_total_energy = target

    def get_energy_statistics(self) -> dict:
        """获取能量统计信息"""
        history = self.energy_history
        valid_history = history[history.nonzero(as_tuple=True)]
        if valid_history.numel() == 0:
            return {}
        return {
            "ema_energy": self.ema_total_energy.item(),
            "mean_energy": valid_history.mean().item(),
            "std_energy": valid_history.std().item() if valid_history.numel() > 1 else 0.0,
            "min_energy": valid_history.min().item(),
            "max_energy": valid_history.max().item(),
            "current_scale": self.ema_scale_factor.item(),
        }

    def reset_state(self) -> None:
        """重置内部状态"""
        self.ema_total_energy.zero_()
        self.ema_scale_factor.fill_(1.0)
        self.energy_history.zero_()
        self.history_ptr.zero_()

    def extra_repr(self) -> str:
        return (
            f"num_modules={self.num_modules}, "
            f"target_energy={self.target_total_energy}, "
            f"energy_tolerance={self.energy_tolerance:.3f}, "
            f"max_scale={self.max_scale_factor:.3f}"
        )
