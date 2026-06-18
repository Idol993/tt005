"""
单元测试：验证模块化相位调制权重衰减器各组件功能
"""

import torch
import torch.nn as nn
from collections import OrderedDict
import sys
import os
import math

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from phase_modulated_decay import (
    CyclicPhaseEncoder,
    ModuleDecayCoefficient,
    ResonanceSuppressor,
    ComplexImpedanceMapper,
    EnergyConservationRegulator,
    PhaseModulatedWeightDecay,
)


def test_phase_encoder():
    print("Test 1: CyclicPhaseEncoder")
    encoder = CyclicPhaseEncoder(num_frequencies=8, base_frequency=2 * math.pi)

    progress = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
    encoding = encoder(progress)

    assert encoding.shape == (5, encoder.output_dim), f"Shape mismatch: {encoding.shape}"
    assert not torch.isnan(encoding).any(), "NaN in encoding"
    assert not torch.isinf(encoding).any(), "Inf in encoding"

    amp, phase = encoder.get_phase_spectrum(progress)
    assert amp.shape == (5, 8)
    assert phase.shape == (5, 8)

    print("  PASSED")


def test_module_coefficient():
    print("Test 2: ModuleDecayCoefficient")
    num_modules = 4
    phase_dim = 17

    coeff_calc = ModuleDecayCoefficient(
        num_modules=num_modules,
        phase_encoding_dim=phase_dim,
        history_length=30,
    )

    grad_norms = torch.rand(num_modules) * 0.1
    param_norms = torch.rand(num_modules) * 5.0
    phase_encoding = torch.rand(phase_dim)

    for _ in range(35):
        decays, phases = coeff_calc(grad_norms, param_norms, phase_encoding)

    assert decays.shape == (num_modules,)
    assert phases.shape == (num_modules,)
    assert (decays >= coeff_calc.min_decay).all()
    assert (decays <= coeff_calc.max_decay).all()
    assert not torch.isnan(decays).any()

    freqs = coeff_calc.get_oscillation_frequencies()
    assert freqs.shape == (num_modules,)
    assert (freqs >= 0).all() and (freqs <= 1).all()

    print("  PASSED")


def test_resonance_suppressor():
    print("Test 3: ResonanceSuppressor")
    num_modules = 4

    suppressor = ResonanceSuppressor(num_modules=num_modules)

    decays = torch.tensor([0.001, 0.0012, 0.0009, 0.0011])
    phases = torch.tensor([0.1, 0.15, 0.12, 0.08])
    freqs = torch.tensor([0.3, 0.32, 0.1, 0.5])

    corrected_decays, corrected_phases, res_map = suppressor(decays, phases, freqs)

    assert corrected_decays.shape == (num_modules,)
    assert corrected_phases.shape == (num_modules,)
    assert res_map.shape == (num_modules, num_modules)
    assert torch.diag(res_map).abs().sum() < 1e-8

    summary = suppressor.get_resonance_summary()
    assert "total_resonance" in summary
    assert "per_module_resonance" in summary

    custom_adj = torch.tensor([
        [0, 1, 1, 0],
        [1, 0, 1, 1],
        [1, 1, 0, 1],
        [0, 1, 1, 0],
    ], dtype=torch.float32)
    suppressor.set_adjacency(custom_adj)
    assert (suppressor.adjacency == custom_adj).all()

    print("  PASSED")


def test_complex_impedance():
    print("Test 4: ComplexImpedanceMapper")
    num_modules = 4

    mapper = ComplexImpedanceMapper(num_modules=num_modules)

    decays = torch.rand(num_modules) * 0.01
    phases = torch.randn(num_modules)

    effective, z, mag, angle = mapper(decays, phases)

    assert effective.shape == (num_modules,)
    assert z.shape == (num_modules, 2)
    assert mag.shape == (num_modules,)
    assert angle.shape == (num_modules,)

    resistance = z[:, 0]
    reactance = z[:, 1]
    reconstructed_mag = torch.sqrt(resistance ** 2 + reactance ** 2)
    assert torch.allclose(mag, reconstructed_mag, atol=1e-5)

    assert (resistance >= 0).all(), "Resistance should be non-negative (softplus)"

    summary = mapper.get_impedance_summary(decays, phases)
    assert "resistance" in summary
    assert "reactance" in summary
    assert "effective_decay" in summary

    print("  PASSED")


def test_energy_conservation():
    print("Test 5: EnergyConservationRegulator")
    num_modules = 4

    regulator = EnergyConservationRegulator(
        num_modules=num_modules,
        target_total_energy=0.01,
        energy_tolerance=0.05,
    )

    decays = torch.tensor([0.001, 0.002, 0.0015, 0.0008])
    param_norms = torch.tensor([3.0, 4.0, 2.5, 5.0])
    grad_norms = torch.tensor([0.1, 0.05, 0.2, 0.08])

    for _ in range(5):
        regulated, info = regulator(decays, param_norms, grad_norms)

    assert regulated.shape == (num_modules,)
    assert "initial_energy" in info
    assert "target_energy" in info
    assert "final_energy" in info

    stats = regulator.get_energy_statistics()
    assert "ema_energy" in stats

    print("  PASSED")


def test_end_to_end():
    print("Test 6: End-to-end PhaseModulatedWeightDecay")

    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = nn.Linear(10, 20)
            self.layer2 = nn.Linear(20, 10)
            self.head = nn.Linear(10, 5)

        def forward(self, x):
            x = torch.relu(self.layer1(x))
            x = torch.relu(self.layer2(x))
            return self.head(x)

    model = SimpleModel()

    pmwd = PhaseModulatedWeightDecay(
        modules=model,
        base_decay=1e-2,
        num_frequencies=4,
    )

    assert pmwd.num_modules == 3, f"Expected 3 modules, got {pmwd.num_modules}"

    X = torch.randn(32, 10)
    y = torch.randint(0, 5, (32,))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    for step in range(20):
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        optimizer.step()

        progress = step / 20
        info = pmwd.apply_decay(progress, return_info=True)

        if step == 19:
            assert info is not None
            assert "final_decays" in info
            assert "base_decays" in info
            assert "resonance_map" in info

    decays = pmwd.get_module_decays()
    assert len(decays) == 3

    diagnostics = pmwd.get_diagnostics()
    assert "current_decays" in diagnostics
    assert "resonance" in diagnostics
    assert "energy" in diagnostics

    param_before = model.layer1.weight.clone().detach()
    pmwd.apply_decay(0.5)
    param_after = model.layer1.weight.detach()
    assert not torch.allclose(param_before, param_after), "Parameters should change after decay"

    print("  PASSED")


def test_decay_parameters_interface():
    print("Test 7: decay_parameters interface")

    model = nn.Sequential(
        OrderedDict([
            ("layer1", nn.Linear(10, 20)),
            ("layer2", nn.Linear(20, 10)),
        ])
    )

    pmwd = PhaseModulatedWeightDecay(modules=model, base_decay=1e-3)

    named_params = list(model.named_parameters())

    params_before = {n: p.clone().detach() for n, p in named_params}

    pmwd.decay_parameters(named_params, training_progress=0.3)

    for n, p in named_params:
        assert not torch.allclose(p.detach(), params_before[n]), f"Param {n} not decayed"

    print("  PASSED")


def test_small_base_decay_not_amplified():
    print("Test 8: Small base_decay is not anomalously amplified")

    base_decay = 1e-4

    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = nn.Linear(10, 20)
            self.layer2 = nn.Linear(20, 15)
            self.layer3 = nn.Linear(15, 5)
        def forward(self, x):
            x = torch.relu(self.layer1(x))
            x = torch.relu(self.layer2(x))
            return self.layer3(x)

    model = SimpleModel()
    pmwd = PhaseModulatedWeightDecay(
        modules=model,
        base_decay=base_decay,
        num_frequencies=4,
    )

    X = torch.randn(32, 10)
    y = torch.randint(0, 5, (32,))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    max_allowed = base_decay * 10.0
    min_allowed = base_decay * 0.01

    decay_records = {name: [] for name in pmwd._module_names}

    for step in range(100):
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        optimizer.step()

        progress = step / 100
        info = pmwd.apply_decay(progress, return_info=True)

        for name, decay in pmwd.get_module_decays().items():
            decay_records[name].append(decay)

        if step % 20 == 0:
            for name, decay in pmwd.get_module_decays().items():
                assert decay >= min_allowed * 0.99, f"{name} decay {decay:.6e} too small, min={min_allowed:.6e}"
                assert decay <= max_allowed * 1.01, f"{name} decay {decay:.6e} too large, max={max_allowed:.6e}"

    final_decays = pmwd.get_module_decays()
    print(f"  Final decays:")
    for name, decay in final_decays.items():
        print(f"    {name}: {decay:.6e} (base={base_decay:.6e})")
        assert decay >= min_allowed * 0.99, f"{name} decay {decay:.6e} too small"
        assert decay <= max_allowed * 1.01, f"{name} decay {decay:.6e} too large"

    decay_values = list(final_decays.values())
    decay_range = max(decay_values) / (min(decay_values) + 1e-12)
    print(f"  Decay ratio (max/min): {decay_range:.2f}x")

    mean_decay = sum(decay_values) / len(decay_values)
    for v in decay_values:
        rel_diff = abs(v - mean_decay) / (mean_decay + 1e-12)
        print(f"    Relative diff from mean: {rel_diff:.2%}")

    print("  PASSED")


def test_energy_consistency_final_decay():
    print("Test 9: Energy consistency with target_total_energy")

    target_energy = 0.001

    model = nn.Sequential(
        OrderedDict([
            ("layer1", nn.Linear(10, 20)),
            ("layer2", nn.Linear(20, 15)),
            ("layer3", nn.Linear(15, 5)),
        ])
    )

    pmwd = PhaseModulatedWeightDecay(
        modules=model,
        base_decay=1e-4,
        target_total_energy=target_energy,
    )

    X = torch.randn(64, 10)
    y = torch.randint(0, 5, (64,))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    for step in range(50):
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        optimizer.step()

        progress = step / 50
        info = pmwd.apply_decay(progress, return_info=True)

        if step >= 40:
            reported_energy = info["energy_info"]["final_energy"]

            weights = torch.softmax(pmwd.energy_regulator.module_weights, dim=0)
            param_norms = info["param_norms"]
            final_decays = info["final_decays"]

            recomputed_energy = (weights.cpu() * final_decays * (param_norms ** 2)).sum().item()

            rel_error = abs(reported_energy - recomputed_energy) / (abs(reported_energy) + 1e-12)
            print(f"  Step {step}: reported={reported_energy:.6e}, recomputed={recomputed_energy:.6e}, rel_error={rel_error:.2%}")

            assert rel_error < 0.001, (
                f"Energy mismatch at step {step}: "
                f"reported={reported_energy:.6e}, recomputed={recomputed_energy:.6e}"
            )

            target_error = abs(reported_energy - target_energy) / (target_energy + 1e-12)
            print(f"    Target error: {target_error:.2%}")

    print("  PASSED")


def test_resonance_phase_continuous_adjustment():
    print("Test 10: Resonance phase shifts accumulate continuously")

    num_modules = 3

    suppressor = ResonanceSuppressor(
        num_modules=num_modules,
        frequency_threshold=0.2,
        max_phase_shift=1.0,
        adaptivity_rate=0.2,
    )

    decays = torch.ones(num_modules) * 1e-4
    base_phases = torch.zeros(num_modules)

    oscillation_freqs = torch.tensor([0.25, 0.26, 0.5])

    phase_history = []
    resonance_history = []
    risk_history = []

    for step in range(50):
        corrected_decays, corrected_phases, res_map = suppressor(decays, base_phases, oscillation_freqs)

        corr_phases_np = suppressor.correction_phases.detach().clone()
        phase_history.append(corr_phases_np)
        resonance_history.append(res_map[0, 1].item())

        phase_diff = abs(corr_phases_np[0] - corr_phases_np[1])
        phase_diff = min(phase_diff, 2 * math.pi - phase_diff)
        freq_diff = abs(oscillation_freqs[0] - oscillation_freqs[1])
        freq_similarity = math.exp(-(freq_diff ** 2) / (2 * 0.2 ** 2))
        risk = freq_similarity * math.exp(-(phase_diff ** 2) / (2 * (math.pi/4) ** 2))
        risk_history.append(risk)

        if step % 10 == 0:
            print(f"  Step {step}:")
            print(f"    correction_phases: {corr_phases_np.tolist()}")
            print(f"    resonance[0,1]: {res_map[0, 1].item():.4f}")
            print(f"    risk: {risk:.4f}")

    phase_history = torch.stack(phase_history)
    phase_changes_0 = torch.abs(phase_history[1:, 0] - phase_history[:-1, 0])
    phase_changes_1 = torch.abs(phase_history[1:, 1] - phase_history[:-1, 1])

    total_change_0 = phase_changes_0.sum().item()
    total_change_1 = phase_changes_1.sum().item()
    print(f"  Total phase change - module 0: {total_change_0:.4f}, module 1: {total_change_1:.4f}")

    assert total_change_0 > 0.05 or total_change_1 > 0.05, (
        f"Phase shifts should accumulate. Got {total_change_0:.4f} and {total_change_1:.4f}"
    )

    initial_risk = risk_history[0]
    final_risk = risk_history[-1]
    print(f"  Risk change: {initial_risk:.4f} -> {final_risk:.4f}")

    assert final_risk < initial_risk * 0.9 or final_risk < 0.3, (
        f"Resonance risk should decrease. Initial={initial_risk:.4f}, Final={final_risk:.4f}"
    )

    print("  PASSED")


if __name__ == "__main__":
    print("Running unit tests...\n")

    try:
        test_phase_encoder()
        test_module_coefficient()
        test_resonance_suppressor()
        test_complex_impedance()
        test_energy_conservation()
        test_end_to_end()
        test_decay_parameters_interface()
        test_small_base_decay_not_amplified()
        test_energy_consistency_final_decay()
        test_resonance_phase_continuous_adjustment()

        print("\n" + "=" * 50)
        print("All tests PASSED!")
    except Exception as e:
        print(f"\nTest FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
