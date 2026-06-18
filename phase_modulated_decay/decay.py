import torch
import torch.nn as nn
from typing import Optional, List, Dict, Tuple, Union
from collections import OrderedDict

from .phase_encoder import CyclicPhaseEncoder
from .module_coefficient import ModuleDecayCoefficient
from .resonance_suppressor import ResonanceSuppressor
from .complex_impedance import ComplexImpedanceMapper
from .energy_conservation import EnergyConservationRegulator


class PhaseModulatedWeightDecay(nn.Module):
    """
    模块化相位调制权重衰减器 (Phase-Modulated Weight Decay)

    为多模块神经网络训练设计的智能权重衰减系统。

    核心特性：
    1. 模块级独立衰减：每个模块拥有独立的可学习衰减系数
    2. 多因素驱动：衰减取决于梯度范数、参数范数和训练整体相位
    3. 共振抑制：检测并抑制相邻模块之间的同频衰减共振
    4. 复数域阻抗映射：通过复阻抗计算精细控制衰减动态
    5. 能量守恒：全局总衰减能量保持在合理范围

    使用方式：
        1. 将模型划分为若干模块（named_modules）
        2. 创建 PhaseModulatedWeightDecay 实例
        3. 在每次 optimizer.step() 之后调用 apply_decay()
        4. 或使用 as_closure() 包装优化器的 step 函数

    示例：
        >>> model = MyModel()
        >>> modules = OrderedDict([
        ...     ("layer1", model.layer1),
        ...     ("layer2", model.layer2),
        ...     ("layer3", model.layer3),
        ... ])
        >>> pmwd = PhaseModulatedWeightDecay(modules)
        >>> optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        >>>
        >>> for epoch in range(num_epochs):
        ...     for step, (x, y) in enumerate(dataloader):
        ...         optimizer.zero_grad()
        ...         loss = criterion(model(x), y)
        ...         loss.backward()
        ...         optimizer.step()
        ...
        ...         progress = (epoch * len(dataloader) + step) / (num_epochs * len(dataloader))
        ...         pmwd.apply_decay(progress)
    """

    def __init__(
        self,
        modules: Union[OrderedDict, List[Tuple[str, nn.Module]], nn.Module],
        base_decay: float = 1e-4,
        num_frequencies: int = 8,
        phase_hidden_dim: int = 32,
        history_length: int = 50,
        frequency_threshold: float = 0.15,
        phase_coupling_strength: float = 0.3,
        target_total_energy: Optional[float] = None,
        learnable_decay: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.base_decay = base_decay
        self.learnable_decay = learnable_decay

        if isinstance(modules, nn.Module):
            module_list = []
            for name, module in modules.named_children():
                if any(p.requires_grad for p in module.parameters()):
                    module_list.append((name, module))
            if module_list:
                self._module_names, self._target_modules = zip(*module_list)
                self._module_names = list(self._module_names)
                self._target_modules = list(self._target_modules)
            else:
                self._module_names = []
                self._target_modules = []
        elif isinstance(modules, OrderedDict):
            self._module_names = list(modules.keys())
            self._target_modules = list(modules.values())
        elif isinstance(modules, list):
            self._module_names = [m[0] for m in modules]
            self._target_modules = [m[1] for m in modules]
        else:
            raise ValueError("modules must be OrderedDict, list of tuples, or nn.Module")

        self.num_modules = len(self._module_names)

        if self.num_modules == 0:
            raise ValueError("No trainable modules found")

        factory_kwargs = {"device": device, "dtype": dtype}

        self.phase_encoder = CyclicPhaseEncoder(
            num_frequencies=num_frequencies,
            **factory_kwargs,
        )

        phase_dim = self.phase_encoder.output_dim

        self.module_decay = ModuleDecayCoefficient(
            num_modules=self.num_modules,
            phase_encoding_dim=phase_dim,
            hidden_dim=phase_hidden_dim,
            history_length=history_length,
            **factory_kwargs,
        )

        self.resonance_suppressor = ResonanceSuppressor(
            num_modules=self.num_modules,
            frequency_threshold=frequency_threshold,
            **factory_kwargs,
        )

        self.impedance_mapper = ComplexImpedanceMapper(
            num_modules=self.num_modules,
            phase_coupling_strength=phase_coupling_strength,
            **factory_kwargs,
        )

        self.energy_regulator = EnergyConservationRegulator(
            num_modules=self.num_modules,
            target_total_energy=target_total_energy,
            **factory_kwargs,
        )

        self.register_buffer(
            "_current_decays",
            torch.ones(self.num_modules, **factory_kwargs) * base_decay,
        )

    def _compute_norms(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算各模块的参数范数和梯度范数"""
        param_norms = []
        grad_norms = []

        for module in self._target_modules:
            params = [p for p in module.parameters() if p.requires_grad]
            if not params:
                param_norms.append(torch.tensor(0.0, device=self._current_decays.device))
                grad_norms.append(torch.tensor(0.0, device=self._current_decays.device))
                continue

            param_norm = torch.norm(torch.stack([torch.norm(p.detach()) for p in params]))
            param_norms.append(param_norm)

            grads = [p.grad.detach() for p in params if p.grad is not None]
            if grads:
                grad_norm = torch.norm(torch.stack([torch.norm(g) for g in grads]))
            else:
                grad_norm = torch.tensor(0.0, device=self._current_decays.device)
            grad_norms.append(grad_norm)

        return torch.stack(param_norms), torch.stack(grad_norms)

    def compute_decay_coefficients(
        self,
        training_progress: float,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        计算各模块的最终衰减系数

        Args:
            training_progress: 当前训练进度，值域 [0, 1]

        Returns:
            final_decays: 各模块最终衰减系数，形状 [num_modules]
            info: 包含中间计算结果的调试信息字典
        """
        progress_tensor = torch.tensor(
            training_progress,
            device=self._current_decays.device,
            dtype=self._current_decays.dtype,
        )

        phase_encoding = self.phase_encoder(progress_tensor)

        param_norms, grad_norms = self._compute_norms()

        base_decays, base_phases = self.module_decay(
            grad_norms, param_norms, phase_encoding
        )

        base_decays = base_decays * self.base_decay

        oscillation_freqs = self.module_decay.get_oscillation_frequencies()

        suppressed_decays, suppressed_phases, resonance_map = (
            self.resonance_suppressor(base_decays, base_phases, oscillation_freqs)
        )

        effective_decays, complex_impedances, imp_mag, imp_angle = (
            self.impedance_mapper(suppressed_decays, suppressed_phases)
        )

        max_allowed_decay = self.base_decay * 10.0
        min_allowed_decay = self.base_decay * 0.01

        if not self.learnable_decay:
            final_decays = torch.ones_like(effective_decays) * self.base_decay
            weights = torch.softmax(self.energy_regulator.module_weights, dim=0)
            per_module_energy = weights * final_decays * (param_norms ** 2)
            total_energy = per_module_energy.sum()
            energy_info = {
                "initial_energy": total_energy.item(),
                "target_energy": total_energy.item(),
                "final_energy": total_energy.item(),
                "deviation_ratio": 0.0,
                "scale_factor": 1.0,
                "per_module_energy": per_module_energy.detach().cpu().tolist(),
                "module_weights": weights.detach().cpu().tolist(),
            }
        else:
            clamped_decays = torch.clamp(effective_decays, min_allowed_decay, max_allowed_decay)
            final_decays, energy_info = self.energy_regulator(
                clamped_decays, param_norms, grad_norms
            )

        self._current_decays = final_decays.detach().clone()

        recomputed_energy = (
            torch.softmax(self.energy_regulator.module_weights, dim=0)
            * final_decays
            * (param_norms ** 2)
        ).sum()
        energy_info["final_energy"] = recomputed_energy.item()

        info = {
            "phase_encoding": phase_encoding.detach().cpu(),
            "param_norms": param_norms.detach().cpu(),
            "grad_norms": grad_norms.detach().cpu(),
            "base_decays": base_decays.detach().cpu(),
            "base_phases": base_phases.detach().cpu(),
            "oscillation_frequencies": oscillation_freqs.detach().cpu(),
            "suppressed_decays": suppressed_decays.detach().cpu(),
            "suppressed_phases": suppressed_phases.detach().cpu(),
            "resonance_map": resonance_map.detach().cpu(),
            "complex_impedances": complex_impedances.detach().cpu(),
            "impedance_magnitudes": imp_mag.detach().cpu(),
            "impedance_angles": imp_angle.detach().cpu(),
            "energy_info": energy_info,
            "final_decays": final_decays.detach().cpu(),
        }

        return final_decays, info

    @torch.no_grad()
    def apply_decay(
        self,
        training_progress: float,
        return_info: bool = False,
    ) -> Optional[Dict]:
        """
        对所有模块参数应用权重衰减

        Args:
            training_progress: 当前训练进度 [0, 1]
            return_info: 是否返回调试信息

        Returns:
            如果 return_info=True，返回调试信息字典；否则返回 None
        """
        final_decays, info = self.compute_decay_coefficients(training_progress)

        for i, module in enumerate(self._target_modules):
            decay = final_decays[i].item()
            for param in module.parameters():
                if param.requires_grad:
                    param.mul_(1.0 - decay)

        return info if return_info else None

    def decay_parameters(
        self,
        named_params: List[Tuple[str, torch.Tensor]],
        training_progress: float,
    ) -> None:
        """
        对指定参数列表应用衰减（与优化器集成时使用）

        Args:
            named_params: (name, param) 对列表
            training_progress: 当前训练进度 [0, 1]
        """
        final_decays, _ = self.compute_decay_coefficients(training_progress)

        module_param_map = {name: i for i, name in enumerate(self._module_names)}

        with torch.no_grad():
            for name, param in named_params:
                module_name = name.split(".")[0]
                if module_name in module_param_map:
                    decay = final_decays[module_param_map[module_name]].item()
                    param.mul_(1.0 - decay)

    def get_module_decays(self) -> Dict[str, float]:
        """获取各模块当前的衰减系数"""
        return {
            name: self._current_decays[i].item()
            for i, name in enumerate(self._module_names)
        }

    def get_diagnostics(self) -> Dict:
        """获取完整的诊断信息"""
        return {
            "module_names": list(self._module_names),
            "current_decays": self.get_module_decays(),
            "resonance": self.resonance_suppressor.get_resonance_summary(),
            "energy": self.energy_regulator.get_energy_statistics(),
        }

    def as_closure(
        self,
        optimizer: torch.optim.Optimizer,
        training_progress_fn,
    ):
        """
        创建一个闭包函数，用于包装优化器的 step

        Args:
            optimizer: PyTorch 优化器
            training_progress_fn: 可调用对象，返回当前训练进度 [0, 1]

        Returns:
            包装后的 step 函数
        """
        original_step = optimizer.step

        def step(closure=None):
            result = original_step(closure)
            progress = training_progress_fn()
            self.apply_decay(progress)
            return result

        return step

    def extra_repr(self) -> str:
        return (
            f"num_modules={self.num_modules}, "
            f"base_decay={self.base_decay}, "
            f"learnable={self.learnable_decay}, "
            f"modules={list(self._module_names)}"
        )
