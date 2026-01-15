"""
Deep Signal Propagation Stress Test
====================================
HC (Hyper-Connections) vs mHC (Manifold-Constrained Hyper-Connections)

This experiment tests signal propagation stability in deep networks.
- Baseline: Standard ResNet (y = x + F(x))
- HC: Unconstrained Hyper-Connections (no Sinkhorn-Knopp constraint)
- mHC: Manifold-Constrained HC (uses mHC.cu library)
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'mHC.cu', 'src', 'python'))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import warnings
warnings.filterwarnings('ignore')

NUM_GPUS = torch.cuda.device_count()
print(f"Available GPUs: {NUM_GPUS}")
for i in range(NUM_GPUS):
    print(f"  cuda:{i} - {torch.cuda.get_device_name(i)}")

try:
    from mhc import MHCLayer, sinkhorn_knopp
    MHC_AVAILABLE = True
    print("mHC CUDA library loaded successfully!")
except ImportError as e:
    MHC_AVAILABLE = False
    print(f"Warning: mHC CUDA library not available: {e}")
    print("Using pure PyTorch fallback for mHC")


# =============================================================================
# 1. Configuration
# =============================================================================

@dataclass
class ExperimentConfig:
    """Experiment configuration"""
    input_dim: int = 128
    hidden_dim: int = 128
    expansion_rate: int = 4
    batch_size: int = 64
    num_samples: int = 5000
    learning_rate: float = 1e-4
    num_epochs: int = 100
    sinkhorn_iters: int = 20
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42


# =============================================================================
# 2. Synthetic Dataset
# =============================================================================

class SyntheticDataset(torch.utils.data.Dataset):
    """
    Synthetic dataset: x -> tanh(Wx + b)
    A nonlinear transformation for deep networks to learn.
    """
    def __init__(self, num_samples: int, input_dim: int, seed: int = 42):
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.x = torch.randn(num_samples, input_dim)
        W = torch.randn(input_dim, input_dim) * 0.1
        b = torch.randn(input_dim) * 0.01
        self.y = torch.tanh(self.x @ W + b)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


# =============================================================================
# 3. Pure PyTorch Sinkhorn-Knopp (Fallback)
# =============================================================================

def sinkhorn_knopp_pytorch(matrix: torch.Tensor, num_iterations: int = 20, eps: float = 1e-5) -> torch.Tensor:
    """
    PyTorch implementation of Sinkhorn-Knopp algorithm.
    Transforms a matrix into a doubly stochastic matrix.
    """
    M = torch.exp(matrix)
    for _ in range(num_iterations):
        M = M / (M.sum(dim=-1, keepdim=True) + eps)
        M = M / (M.sum(dim=-2, keepdim=True) + eps)
    return M


# =============================================================================
# 4. Model Architectures
# =============================================================================

class ResidualBlock(nn.Module):
    """Basic Residual Block: computes F(x)"""
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BaselineResNet(nn.Module):
    """
    Baseline: Standard ResNet
    y = x + F(x)
    """
    def __init__(self, config: ExperimentConfig, num_layers: int):
        super().__init__()
        self.input_proj = nn.Linear(config.input_dim, config.hidden_dim)
        self.output_proj = nn.Linear(config.hidden_dim, config.input_dim)
        self.blocks = nn.ModuleList([
            ResidualBlock(config.hidden_dim) for _ in range(num_layers)
        ])
        self.num_layers = num_layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        for block in self.blocks:
            h = h + block(h)
        return self.output_proj(h)


class HCModel(nn.Module):
    """
    HC (Hyper-Connections) - Unconstrained

    Without constraints on H^res, signals can explode in deeper networks.
    This demonstrates why mHC is necessary.
    """
    def __init__(self, config: ExperimentConfig, num_layers: int):
        super().__init__()
        self.n = config.expansion_rate
        self.hidden_dim = config.hidden_dim

        self.input_proj = nn.Linear(config.input_dim, config.hidden_dim * self.n)
        self.output_proj = nn.Linear(config.hidden_dim * self.n, config.input_dim)

        self.blocks = nn.ModuleList([
            ResidualBlock(config.hidden_dim) for _ in range(num_layers)
        ])

        # H^res: (n x n) matrix, learned without constraints
        self.H_res = nn.ParameterList([
            nn.Parameter(torch.eye(self.n) + 0.01 * torch.randn(self.n, self.n))
            for _ in range(num_layers)
        ])

        # H^pre: selects which streams contribute to F(x) input (n,)
        self.H_pre = nn.ParameterList([
            nn.Parameter(torch.ones(self.n) / self.n)
            for _ in range(num_layers)
        ])

        # H^post: distributes F(x) output to streams (n,)
        self.H_post = nn.ParameterList([
            nn.Parameter(torch.ones(self.n) / self.n)
            for _ in range(num_layers)
        ])

        self.num_layers = num_layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]

        h = self.input_proj(x)
        h = h.reshape(B, self.n, self.hidden_dim)

        for i, block in enumerate(self.blocks):
            H_res = self.H_res[i]
            H_pre = torch.sigmoid(self.H_pre[i])
            H_post = 2.0 * torch.sigmoid(self.H_post[i])

            block_input = torch.einsum('k,bkc->bc', H_pre, h)
            block_output = block(block_input)
            h_mixed = torch.einsum('ij,bjc->bic', H_res, h)
            h = h_mixed + torch.einsum('k,bc->bkc', H_post, block_output)

        h = h.reshape(B, self.n * self.hidden_dim)
        return self.output_proj(h)

    def get_cumulative_H_res(self) -> torch.Tensor:
        """Compute cumulative H^res matrix"""
        result = torch.eye(self.n, device=self.H_res[0].device)
        for H in self.H_res:
            result = result @ H
        return result


class mHCModelPyTorch(nn.Module):
    """
    mHC (Manifold-Constrained HC) - PyTorch Fallback

    Constrains H^res to be doubly stochastic via Sinkhorn-Knopp,
    preserving signal magnitude through layers.
    """
    def __init__(self, config: ExperimentConfig, num_layers: int):
        super().__init__()
        self.n = config.expansion_rate
        self.hidden_dim = config.hidden_dim
        self.sinkhorn_iters = config.sinkhorn_iters

        self.input_proj = nn.Linear(config.input_dim, config.hidden_dim * self.n)
        self.output_proj = nn.Linear(config.hidden_dim * self.n, config.input_dim)

        self.blocks = nn.ModuleList([
            ResidualBlock(config.hidden_dim) for _ in range(num_layers)
        ])

        # Logits for H^res (before Sinkhorn-Knopp)
        self.H_res_logits = nn.ParameterList([
            nn.Parameter(torch.zeros(self.n, self.n))
            for _ in range(num_layers)
        ])

        self.H_pre = nn.ParameterList([
            nn.Parameter(torch.ones(self.n) / self.n)
            for _ in range(num_layers)
        ])

        self.H_post = nn.ParameterList([
            nn.Parameter(torch.ones(self.n) / self.n)
            for _ in range(num_layers)
        ])

        self.num_layers = num_layers

    def get_constrained_H_res(self, idx: int) -> torch.Tensor:
        """Return H^res constrained by Sinkhorn-Knopp"""
        return sinkhorn_knopp_pytorch(self.H_res_logits[idx], self.sinkhorn_iters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]

        h = self.input_proj(x)
        h = h.reshape(B, self.n, self.hidden_dim)

        for i, block in enumerate(self.blocks):
            H_res = self.get_constrained_H_res(i)
            H_pre = torch.sigmoid(self.H_pre[i])
            H_post = 2.0 * torch.sigmoid(self.H_post[i])

            block_input = torch.einsum('k,bkc->bc', H_pre, h)
            block_output = block(block_input)
            h_mixed = torch.einsum('ij,bjc->bic', H_res, h)
            h = h_mixed + torch.einsum('k,bc->bkc', H_post, block_output)

        h = h.reshape(B, self.n * self.hidden_dim)
        return self.output_proj(h)

    def get_cumulative_H_res(self) -> torch.Tensor:
        """Compute cumulative H^res matrix"""
        result = torch.eye(self.n, device=self.H_res_logits[0].device)
        for i in range(len(self.H_res_logits)):
            H_res = self.get_constrained_H_res(i)
            result = result @ H_res.detach()
        return result


class mHCModelCUDA(nn.Module):
    """
    mHC (Manifold-Constrained HC) - Using mHC.cu CUDA library

    Uses CUDA-optimized Sinkhorn-Knopp and fused operations.
    """
    def __init__(self, config: ExperimentConfig, num_layers: int):
        super().__init__()
        self.n = config.expansion_rate
        self.hidden_dim = config.hidden_dim

        self.input_proj = nn.Linear(config.input_dim, config.hidden_dim * self.n)
        self.output_proj = nn.Linear(config.hidden_dim * self.n, config.input_dim)

        self.mhc_layers = nn.ModuleList([
            MHCLayer(
                hidden_dim=config.hidden_dim,
                expansion_rate=config.expansion_rate,
                sinkhorn_iters=config.sinkhorn_iters,
                use_dynamic_h=False,
            )
            for _ in range(num_layers)
        ])

        self.blocks = nn.ModuleList([
            ResidualBlock(config.hidden_dim) for _ in range(num_layers)
        ])

        self.num_layers = num_layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]

        h = self.input_proj(x)
        h = h.reshape(B, self.n, self.hidden_dim)

        for i, (mhc_layer, block) in enumerate(zip(self.mhc_layers, self.blocks)):
            h_pre = torch.sigmoid(mhc_layer.H_pre) if hasattr(mhc_layer, 'H_pre') else torch.ones(self.n, device=h.device) / self.n
            block_input = h.mean(dim=1)
            block_output = block(block_input)
            h = mhc_layer(h)

        h = h.reshape(B, self.n * self.hidden_dim)
        return self.output_proj(h)


if MHC_AVAILABLE:
    mHCModel = mHCModelCUDA
else:
    mHCModel = mHCModelPyTorch


# =============================================================================
# 5. Metrics Tracker
# =============================================================================

class MetricsTracker:
    """Tracks training metrics"""
    def __init__(self):
        self.losses: List[float] = []
        self.gradient_norms: List[float] = []
        self.amax_gains: List[float] = []

    def update(self, loss: float, grad_norm: float, amax_gain: float):
        self.losses.append(loss)
        self.gradient_norms.append(grad_norm)
        self.amax_gains.append(amax_gain)


def compute_gradient_norm(model: nn.Module) -> float:
    """Compute total gradient norm of the model"""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    return np.sqrt(total_norm)


def compute_amax_gain(model: nn.Module) -> float:
    """
    Amax Gain: max row/column sum of cumulative H^res matrix.

    - Ideal case (mHC): ~1.0 (doubly stochastic)
    - Unstable case (HC): >> 1.0 (signal explosion)
    """
    if hasattr(model, 'get_cumulative_H_res'):
        with torch.no_grad():
            cumulative_H = model.get_cumulative_H_res()
            row_sums = cumulative_H.sum(dim=1)
            col_sums = cumulative_H.sum(dim=0)
            amax = max(row_sums.max().item(), col_sums.max().item())
            return amax
    return 1.0


# =============================================================================
# 6. Training Loop
# =============================================================================

def train_model(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    config: ExperimentConfig,
    model_name: str
) -> MetricsTracker:
    """Train a single model"""
    model = model.to(config.device)
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
    criterion = nn.MSELoss()
    tracker = MetricsTracker()

    print(f"\n{'='*60}")
    print(f"Training: {model_name}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"{'='*60}")

    for epoch in range(config.num_epochs):
        model.train()
        epoch_loss = 0.0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(config.device)
            batch_y = batch_y.to(config.device)

            optimizer.zero_grad()

            try:
                output = model(batch_x)
                loss = criterion(output, batch_y)
            except RuntimeError as e:
                print(f"  [Epoch {epoch}] Runtime error: {e}")
                tracker.update(float('nan'), float('nan'), float('nan'))
                return tracker

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"  [Epoch {epoch}] NaN/Inf detected! Training unstable.")
                tracker.update(float('nan'), float('nan'), float('nan'))
                return tracker

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        grad_norm = compute_gradient_norm(model)
        amax_gain = compute_amax_gain(model)

        tracker.update(avg_loss, grad_norm, amax_gain)

        if epoch % 20 == 0 or epoch == config.num_epochs - 1:
            print(f"  Epoch {epoch:3d} | Loss: {avg_loss:.6f} | "
                  f"Grad Norm: {grad_norm:.4f} | Amax Gain: {amax_gain:.4f}")

    return tracker


# =============================================================================
# 7. Experiment Runner
# =============================================================================

def run_single_depth_experiment(
    depth: int,
    config: ExperimentConfig,
    device: str,
    dataset: SyntheticDataset
) -> Dict[str, MetricsTracker]:
    """Run all models at a single depth on a specific GPU"""
    depth_config = ExperimentConfig(
        input_dim=config.input_dim,
        hidden_dim=config.hidden_dim,
        expansion_rate=config.expansion_rate,
        batch_size=config.batch_size,
        num_samples=config.num_samples,
        learning_rate=config.learning_rate,
        num_epochs=config.num_epochs,
        sinkhorn_iters=config.sinkhorn_iters,
        device=device,
        seed=config.seed
    )

    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=config.batch_size, shuffle=True
    )

    results = {}

    print(f"\n[{device}] {'='*60}")
    print(f"[{device}] DEPTH = {depth} layers")
    print(f"[{device}] {'='*60}")

    # 1. Baseline ResNet
    torch.manual_seed(config.seed)
    baseline = BaselineResNet(depth_config, depth)
    results["Baseline"] = train_model(
        baseline, train_loader, depth_config, f"Baseline (D={depth}) [{device}]"
    )

    # 2. HC (Unconstrained)
    torch.manual_seed(config.seed)
    hc = HCModel(depth_config, depth)
    results["HC"] = train_model(
        hc, train_loader, depth_config, f"HC (D={depth}) [{device}]"
    )

    # 3. mHC (Constrained)
    torch.manual_seed(config.seed)
    mhc = mHCModelPyTorch(depth_config, depth)
    results["mHC"] = train_model(
        mhc, train_loader, depth_config, f"mHC (D={depth}) [{device}]"
    )

    return results


def run_depth_scaling_experiment(
    depths: List[int],
    config: ExperimentConfig
) -> Dict[str, Dict[int, MetricsTracker]]:
    """Depth scaling experiment with Multi-GPU parallelization"""
    print("\n" + "="*80)
    print("DEPTH SCALING EXPERIMENT (Multi-GPU Parallel)")
    print("="*80)
    print(f"Testing depths: {depths}")
    print(f"Expansion rate (n): {config.expansion_rate}")
    print(f"Available GPUs: {NUM_GPUS}")

    dataset = SyntheticDataset(config.num_samples, config.input_dim, config.seed)

    results = {
        "Baseline": {},
        "HC": {},
        "mHC": {}
    }

    if NUM_GPUS >= 2:
        print(f"Using {NUM_GPUS} GPUs for parallel execution")

        with ThreadPoolExecutor(max_workers=NUM_GPUS) as executor:
            futures = {}
            for i, depth in enumerate(depths):
                device = f"cuda:{i % NUM_GPUS}"
                future = executor.submit(
                    run_single_depth_experiment,
                    depth, config, device, dataset
                )
                futures[future] = depth

            for future in as_completed(futures):
                depth = futures[future]
                try:
                    depth_results = future.result()
                    for model_name in ["Baseline", "HC", "mHC"]:
                        results[model_name][depth] = depth_results[model_name]
                    print(f"\n✓ Depth {depth} completed!")
                except Exception as e:
                    print(f"\n✗ Depth {depth} failed: {e}")
                    for model_name in ["Baseline", "HC", "mHC"]:
                        tracker = MetricsTracker()
                        tracker.update(float('nan'), float('nan'), float('nan'))
                        results[model_name][depth] = tracker
    else:
        print("Single GPU mode - sequential execution")
        for depth in depths:
            depth_results = run_single_depth_experiment(
                depth, config, config.device, dataset
            )
            for model_name in ["Baseline", "HC", "mHC"]:
                results[model_name][depth] = depth_results[model_name]

    return results


# =============================================================================
# 8. Visualization
# =============================================================================

def plot_results(results: Dict[str, Dict[int, MetricsTracker]], save_path: str = None):
    """Visualize experiment results"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Deep Signal Propagation Stress Test: HC vs mHC", fontsize=14, fontweight='bold')

    depths = sorted(list(results["Baseline"].keys()))
    colors = {"Baseline": "blue", "HC": "red", "mHC": "green"}
    markers = {"Baseline": "o", "HC": "s", "mHC": "^"}

    # 1. Final Loss by Depth
    ax = axes[0, 0]
    for model_name in ["Baseline", "HC", "mHC"]:
        final_losses = []
        for d in depths:
            tracker = results[model_name][d]
            if tracker.losses and not np.isnan(tracker.losses[-1]):
                final_losses.append(tracker.losses[-1])
            else:
                final_losses.append(np.nan)
        ax.plot(depths, final_losses, f'{markers[model_name]}-',
                label=model_name, color=colors[model_name], linewidth=2, markersize=8)
    ax.set_xlabel("Depth (Number of Layers)")
    ax.set_ylabel("Final Loss")
    ax.set_title("Final Loss vs Depth")
    ax.legend()
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    # 2. Amax Gain by Depth (Key metric)
    ax = axes[0, 1]
    for model_name in ["HC", "mHC"]:
        amax_gains = []
        for d in depths:
            tracker = results[model_name][d]
            if tracker.amax_gains and not np.isnan(tracker.amax_gains[-1]):
                amax_gains.append(tracker.amax_gains[-1])
            else:
                amax_gains.append(np.nan)
        ax.plot(depths, amax_gains, f'{markers[model_name]}-',
                label=model_name, color=colors[model_name], linewidth=2, markersize=8)
    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=2, label='Ideal (=1.0)')
    ax.set_xlabel("Depth (Number of Layers)")
    ax.set_ylabel("Amax Gain (log scale)")
    ax.set_title("Amax Gain vs Depth (Signal Stability)")
    ax.legend()
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    # 3. Training Curves for Deepest Model
    max_depth = max(depths)
    ax = axes[1, 0]
    for model_name in ["Baseline", "HC", "mHC"]:
        tracker = results[model_name][max_depth]
        if tracker.losses:
            valid_losses = [l if not np.isnan(l) else None for l in tracker.losses]
            ax.plot(valid_losses, label=model_name, color=colors[model_name], linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"Training Curve (Depth={max_depth})")
    ax.legend()
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    # 4. Gradient Norm for Deepest Model
    ax = axes[1, 1]
    for model_name in ["Baseline", "HC", "mHC"]:
        tracker = results[model_name][max_depth]
        if tracker.gradient_norms:
            valid_norms = [n if not np.isnan(n) else None for n in tracker.gradient_norms]
            ax.plot(valid_norms, label=model_name, color=colors[model_name], linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Gradient Norm")
    ax.set_title(f"Gradient Norm (Depth={max_depth})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to: {save_path}")

    plt.show()


def print_summary(results: Dict[str, Dict[int, MetricsTracker]]):
    """Print experiment summary"""
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)

    depths = sorted(list(results["Baseline"].keys()))

    print(f"\n{'Model':<12} | ", end="")
    for d in depths:
        print(f"D={d:<6} | ", end="")
    print()
    print("-"*70)

    for metric_name, metric_getter in [
        ("Final Loss", lambda t: t.losses[-1] if t.losses else float('nan')),
        ("Grad Norm", lambda t: t.gradient_norms[-1] if t.gradient_norms else float('nan')),
        ("Amax Gain", lambda t: t.amax_gains[-1] if t.amax_gains else float('nan'))
    ]:
        print(f"\n{metric_name}:")
        for model_name in ["Baseline", "HC", "mHC"]:
            print(f"  {model_name:<10} | ", end="")
            for d in depths:
                value = metric_getter(results[model_name][d])
                if np.isnan(value):
                    print(f"{'NaN':<8} | ", end="")
                elif value > 1000:
                    print(f"{value:<8.1e} | ", end="")
                else:
                    print(f"{value:<8.4f} | ", end="")
            print()


# =============================================================================
# 9. Main Execution
# =============================================================================

def main():
    """Main experiment execution"""
    print("="*80)
    print("Deep Signal Propagation Stress Test")
    print("HC (Hyper-Connections) vs mHC (Manifold-Constrained HC)")
    print("="*80)

    config = ExperimentConfig()

    print(f"\nConfiguration:")
    print(f"  - Input/Hidden Dim: {config.input_dim}")
    print(f"  - Expansion Rate (n): {config.expansion_rate}")
    print(f"  - Sinkhorn Iterations: {config.sinkhorn_iters}")
    print(f"  - Learning Rate: {config.learning_rate}")
    print(f"  - Epochs: {config.num_epochs}")
    print(f"  - Device: {config.device}")
    print(f"  - mHC CUDA Available: {MHC_AVAILABLE}")

    depths = [10, 30, 50, 100]

    results = run_depth_scaling_experiment(depths, config)
    print_summary(results)

    try:
        plot_results(results, "depth_scaling_results.png")
    except Exception as e:
        print(f"Could not generate plot: {e}")

    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)
    print("""
    Expected conclusions:

    1. Amax Gain (Key metric):
       - HC: Amax >> 1.0 as depth increases (signal explosion)
       - mHC: Amax ~ 1.0 regardless of depth (doubly stochastic constraint)

    2. Training stability:
       - HC: Loss divergence or NaN possible in deep networks
       - mHC: Maintains stable training curves

    3. Key difference:
       - HC cumulative H^res: Product of H^res deviates from identity
       - mHC cumulative H^res: Product of doubly stochastic matrices
                               remains doubly stochastic
                               -> Row/column sums always equal 1
    """)


if __name__ == "__main__":
    main()
