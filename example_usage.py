"""
示例：使用模块化相位调制权重衰减器训练一个简单神经网络

该脚本演示了如何：
1. 构建一个多模块神经网络
2. 创建 PhaseModulatedWeightDecay 实例
3. 在训练循环中集成衰减器
4. 监控衰减系数、共振状态和能量守恒
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from phase_modulated_decay import PhaseModulatedWeightDecay


class MultiModuleNet(nn.Module):
    """简单的多模块分类网络"""

    def __init__(self, input_dim=784, hidden_dim=256, num_classes=10):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(hidden_dim // 2, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.classifier(x)


def generate_synthetic_data(num_samples=1000, input_dim=784, num_classes=10):
    """生成合成数据集用于演示"""
    X = torch.randn(num_samples, input_dim)
    y = torch.randint(0, num_classes, (num_samples,))
    return X, y


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = MultiModuleNet().to(device)

    modules = OrderedDict(
        [
            ("layer1", model.layer1),
            ("layer2", model.layer2),
            ("layer3", model.layer3),
            ("classifier", model.classifier),
        ]
    )

    pmwd = PhaseModulatedWeightDecay(
        modules=modules,
        base_decay=1e-4,
        num_frequencies=8,
        phase_hidden_dim=32,
        history_length=50,
        device=device,
    )

    print(f"Created PMWD with {pmwd.num_modules} modules:")
    for name in pmwd._module_names:
        print(f"  - {name}")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    X, y = generate_synthetic_data()
    X, y = X.to(device), y.to(device)

    num_epochs = 50
    batch_size = 64
    num_batches = len(X) // batch_size
    total_steps = num_epochs * num_batches

    print(f"\nStarting training: {num_epochs} epochs, {total_steps} total steps")
    print("=" * 70)

    global_step = 0

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0

        perm = torch.randperm(len(X))
        X_shuffled = X[perm]
        y_shuffled = y[perm]

        for batch_idx in range(num_batches):
            start = batch_idx * batch_size
            end = start + batch_size

            x_batch = X_shuffled[start:end]
            y_batch = y_shuffled[start:end]

            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            training_progress = global_step / total_steps
            return_info = (global_step % 100 == 0)
            info = pmwd.apply_decay(training_progress, return_info=return_info)

            _, predicted = outputs.max(1)
            epoch_total += y_batch.size(0)
            epoch_correct += predicted.eq(y_batch).sum().item()
            epoch_loss += loss.item()

            if return_info and info is not None:
                print(f"\nStep {global_step} (progress: {training_progress:.3f})")
                print(f"  Loss: {loss.item():.4f}")
                print(f"  Module decays:")
                for name, decay in pmwd.get_module_decays().items():
                    print(f"    {name}: {decay:.6f}")
                print(f"  Energy: {info['energy_info']['final_energy']:.6f} "
                      f"(target: {info['energy_info']['target_energy']:.6f})")

            global_step += 1

        epoch_loss /= num_batches
        epoch_acc = 100.0 * epoch_correct / epoch_total

        if (epoch + 1) % 5 == 0:
            print(
                f"Epoch [{epoch + 1}/{num_epochs}] "
                f"Loss: {epoch_loss:.4f} "
                f"Acc: {epoch_acc:.2f}% "
                f"Progress: {training_progress:.3f}"
            )

    print("\n" + "=" * 70)
    print("Training complete!")
    print("\nFinal diagnostics:")
    diagnostics = pmwd.get_diagnostics()
    print(f"  Module decays:")
    for name, decay in diagnostics["current_decays"].items():
        print(f"    {name}: {decay:.6f}")
    print(f"  Energy statistics: {diagnostics['energy']}")
    print(f"  Resonance summary: {diagnostics['resonance']['total_resonance']:.4f}")


if __name__ == "__main__":
    main()
