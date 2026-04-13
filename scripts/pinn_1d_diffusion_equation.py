"""
Physics-Informed Neural Network (PINN) for the 1D Diffusion Equation
================================================================

PDE:  ∂u/∂t = α · ∂²u/∂x²   in  x ∈ [0, 1], t ∈ [0, 1]

boundary conditions (BC):
    u(0, t) = 0
    u(1, t) = 0

Initial conditions (Dirichlet):
    u(x, 0) = sin(π·x)
Initial conditions (von von_neumann):
    u(x, 0) = cos(π·x)

Exact solution (Dirichlet bc):
    u(x, t) = exp(-α·π²·t) · sin(π·x)
Exact solution (von von_neumann bc):
    u(x, t) = exp(-α·π²·t) · cos(π·x)

This script demonstrates the basic principles of PINNs:
  - NN as universal approximation for functions
  - Physics loss using automatic differentiation (autograd)
  - Boundary / Initial condition loss
  - Training loop using Adam + L-BFGS
"""

import torch
import argparse
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ── Reproducibility ────────────────────────────────────────────────────────
torch.manual_seed(42)
np.random.seed(42)

DEVICE = torch.device("mps" if torch.mps.is_available() else "cpu")
print(f"Gerät: {DEVICE}")

# ── Default Hyperparameter ────────────────────────────────────────────────────────────
ALPHA = 0.01  # Diffusion constant
N_COLLOC = 5_000  # collocation points (physics Loss)
N_BC = 200  # points per bc
N_IC = 500  # points for initial condition
HIDDEN_SIZE = 64  # neurons per layer
N_LAYERS = 4  # number of hidden layers
LR_ADAM = 1e-3 # Adam learning rate
EPOCHS_ADAM = 5_000 # epochs for Adam
EPOCHS_LBFGS = 500 # epochs for LBFGS
T_FINAL = 1 # final time

# ── 1. Neural Network ───────────────────────────────────────────────────────────────


class PINN(nn.Module):
    """
    Input:  (x, t)  →  2 Features
    Output:  u(x, t) →  1 Scalar

    Tanh activation: smooth and infinitly often differentiable –
    """

    def __init__(self, hidden: int = HIDDEN_SIZE, n_layers: int = N_LAYERS):
        super().__init__()
        layers = [nn.Linear(2, hidden), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*layers)

        # Xavier-Initialization for stable training
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        xt = torch.cat([x, t], dim=1)  # Shape: (N, 2)
        return self.net(xt)  # Shape: (N, 1)


# ── 2. Physics Loss (PDE residue) ───────────────────────────────────────


def physics_loss(model: PINN, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Computes ||∂u/∂t - α·∂²u/∂x²||²  at collocation points.

    autograd differentiates – no finite difference error
    """
    x = x.requires_grad_(True)
    t = t.requires_grad_(True)

    u = model(x, t)

    # First derivatives
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[
        0
    ]
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[
        0
    ]
    # Second deriviative in x
    u_xx = torch.autograd.grad(
        u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True
    )[0]

    residuum = u_t - ALPHA * u_xx
    return torch.mean(residuum**2)


# ── 3. Boundary & Initial Condition Loss ─────────────────────────────────────


def bc_loss(
    model: PINN, t_bc: torch.Tensor, x0: torch.Tensor, x1: torch.Tensor, bc: str
) -> torch.Tensor:
    if bc == "dirichlet":
        """u(0,t) = 0  und  u(1,t) = 0"""
        u_left = model(x0, t_bc)
        u_right = model(x1, t_bc)
        return torch.mean(u_left**2) + torch.mean(u_right**2)
    elif bc == "von_neumann":
        """∂u/∂x(0,t) = 0  und  ∂u/∂x(1,t) = 0"""
        x0 = x0.requires_grad_(True)
        x1 = x1.requires_grad_(True)

        u_left = model(x0, t_bc)
        u_right = model(x1, t_bc)

        u_x_left = torch.autograd.grad(
            u_left, x0, grad_outputs=torch.ones_like(u_left), create_graph=True
        )[0]
        u_x_right = torch.autograd.grad(
            u_right, x1, grad_outputs=torch.ones_like(u_right), create_graph=True
        )[0]

        return torch.mean(u_x_left**2) + torch.mean(u_x_right**2)
    else:
        print("Unknown boundary conditions!")


def ic_loss(model: PINN, x_ic: torch.Tensor, t0: torch.Tensor, bc: str) -> torch.Tensor:
    if bc == "dirichlet":
        """u(x, 0) = sin(π·x)"""
        u_pred = model(x_ic, t0)
        u_true = torch.sin(torch.pi * x_ic)
        return torch.mean((u_pred - u_true) ** 2)
    elif bc == "von_neumann":
        """u(x, 0) = cos(π·x)"""
        u_pred = model(x_ic, t0)
        u_true = torch.cos(torch.pi * x_ic)
        return torch.mean((u_pred - u_true) ** 2)


# ── 4. Generate training data ────────────────────────────────────────────────


def make_training_data():
    # Collocation points (random on the inner)
    x_col = torch.rand(N_COLLOC, 1, device=DEVICE)
    t_col = T_FINAL * torch.rand(N_COLLOC, 1, device=DEVICE)

    # Boundary conditions
    t_bc = T_FINAL * torch.rand(N_BC, 1, device=DEVICE)
    x0 = torch.zeros(N_BC, 1, device=DEVICE)
    x1 = torch.ones(N_BC, 1, device=DEVICE)

    # Initial conditions
    x_ic = torch.rand(N_IC, 1, device=DEVICE)
    t0 = torch.zeros(N_IC, 1, device=DEVICE)

    return x_col, t_col, t_bc, x0, x1, x_ic, t0


# ── 5. Default hyperparameters and parse input ───────────────────────────────────────────────────────────────


def parse_args():
    parser = argparse.ArgumentParser(description="PINN – 1D Diffusion equation")
    parser.add_argument(
        "--alpha", type=float, default=1e-2, help="Diffusion constant"
    )
    parser.add_argument(
        "--n_colloc", type=int, default=5_000, help="Collocation points"
    )
    parser.add_argument("--n_bc", type=int, default=200, help="Boundary points")
    parser.add_argument("--n_ic", type=int, default=500, help="Initial points")
    parser.add_argument(
        "--hidden_size", type=int, default=64, help="Neurons per layer"
    )
    parser.add_argument(
        "--n_layers", type=int, default=4, help="Number of hidden layers"
    )
    parser.add_argument("--lr_adams", type=float, default=1e-3, help="Adam learning rate")
    parser.add_argument("--epochs_adam", type=int, default=5_000, help="Adam epchs")
    parser.add_argument("--epochs_lbfgs", type=int, default=500, help="LBFGS epochs")
    parser.add_argument("--t_final", type=int, default=1, help="Final time point")
    parser.add_argument(
        "--boundary_condition", type=str, default="dirichlet", help="Boundary condition"
    )
    return parser.parse_args()


# ── 6. Training ───────────────────────────────────────────────────────────────


def train():
    model = PINN().to(DEVICE)
    x_col, t_col, t_bc, x0, x1, x_ic, t0 = make_training_data()

    history = {"total": [], "physics": [], "bc": [], "ic": []}

    # ── Phase 1: Adam ──────────────────────────────────────────────────────────
    print("\n── Phase 1: Adam ─────────────────────────────────────")
    optimizer = torch.optim.Adam(model.parameters(), lr=LR_ADAM)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=200,  # wait 200 epochs without improvment
        factor=0.5,  # halve the learning rate
    )

    for epoch in range(1, EPOCHS_ADAM + 1):
        optimizer.zero_grad()

        loss_phys = physics_loss(model, x_col, t_col)
        loss_bc = bc_loss(model, t_bc, x0, x1, BC)
        loss_ic = ic_loss(model, x_ic, t0, BC)
        if BC == "dirichlet":
            loss = loss_phys + loss_bc + loss_ic
        elif BC == "von_neumann":
            loss = loss_phys + 10 * loss_bc + loss_ic

        loss.backward()
        optimizer.step()
        # scheduler.step(loss.detach())

        if epoch % 500 == 0:
            history["total"].append(loss.item())
            history["physics"].append(loss_phys.item())
            history["bc"].append(loss_bc.item())
            history["ic"].append(loss_ic.item())
            print(
                f"  Epoch {epoch:5d} | Total: {loss.item():.2e} "
                f"| PDE: {loss_phys.item():.2e} "
                f"| BC: {loss_bc.item():.2e} "
                f"| IC: {loss_ic.item():.2e}"
            )

    # ── Phase 2: L-BFGS (fine-tuning) ──────────────────────────────────────
    print("\n── Phase 2: L-BFGS ───────────────────────────────────")
    optimizer_lbfgs = torch.optim.LBFGS(
        model.parameters(),
        lr=0.1,
        max_iter=20,
        history_size=50,
        line_search_fn="strong_wolfe",
    )

    step = [0]

    def closure():
        optimizer_lbfgs.zero_grad()
        lp = physics_loss(model, x_col, t_col)
        lb = bc_loss(model, t_bc, x0, x1, BC)
        li = ic_loss(model, x_ic, t0, BC)
        if BC == "dirichlet":
            loss = lp + lb + li
        elif BC == "von_neumann":
            loss = lp + 10 * lb + li
        loss.backward()
        step[0] += 1
        if step[0] % 100 == 0:
            print(f"L-BFGS step {step[0]:4d} | Total: {loss.item():.2e}")
        return loss

    for _ in range(EPOCHS_LBFGS // 20):
        optimizer_lbfgs.step(closure)

    return model, history


# ── 6. Visualization ─────────────────────────────────────────────────────────


def plot_results(model: PINN, history: dict):
    model.eval()

    # Grid for prediction
    x_vals = np.linspace(0, 1, 200)
    t_vals = np.linspace(0, T_FINAL, 200)
    X, T = np.meshgrid(x_vals, t_vals)

    x_flat = torch.tensor(X.flatten(), dtype=torch.float32, device=DEVICE).unsqueeze(1)
    t_flat = torch.tensor(T.flatten(), dtype=torch.float32, device=DEVICE).unsqueeze(1)

    with torch.no_grad():
        u_pred = model(x_flat, t_flat).cpu().numpy().reshape(200, 200)

    # Analytic solutions
    if BC == "dirichlet":
        u_exact = np.exp(-ALPHA * np.pi**2 * T) * np.sin(np.pi * X)
    elif BC == "von_neumann":
        u_exact = np.exp(-ALPHA * np.pi**2 * T) * np.cos(np.pi * X)
    else:
        print("Unknown boundary condition!")
    error = np.abs(u_pred - u_exact)

    # ── Plot ───────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(15, 10))
    fig.suptitle("PINN – 1D Diffusion Equation", fontsize=15, fontweight="bold")
    gs = GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    # 1. PINN-prediction
    ax1 = fig.add_subplot(gs[0, 0])
    c1 = ax1.contourf(X, T, u_pred, levels=50, cmap="hot")
    fig.colorbar(c1, ax=ax1)
    ax1.set_title("PINN Prediction u(x,t)")
    ax1.set_xlabel("x")
    ax1.set_ylabel("t")

    # 2. Analytic solution
    ax2 = fig.add_subplot(gs[0, 1])
    c2 = ax2.contourf(X, T, u_exact, levels=50, cmap="hot")
    fig.colorbar(c2, ax=ax2)
    ax2.set_title("Analytic Solution")
    ax2.set_xlabel("x")
    ax2.set_ylabel("t")

    # 3. Absolute error
    ax3 = fig.add_subplot(gs[0, 2])
    c3 = ax3.contourf(X, T, error, levels=50, cmap="viridis")
    fig.colorbar(c3, ax=ax3)
    ax3.set_title("Absolute Error |PINN − Exact|")
    ax3.set_xlabel("x")
    ax3.set_ylabel("t")

    # 4. Profiles at selected times
    ax4 = fig.add_subplot(gs[1, :2])
    for t_snap in [0.0, 0.1, 0.3, 0.5, 1.0]:
        idx = np.argmin(np.abs(t_vals - t_snap))
        ax4.plot(x_vals, u_pred[idx], label=f"PINN t={t_snap:.1f}")
        ax4.plot(x_vals, u_exact[idx], "--", alpha=0.5)
    ax4.set_title("Temperature profiles at selected time steps (— PINN, -- Exact)")
    ax4.set_xlabel("x")
    ax4.set_ylabel("u(x,t)")
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

    # 5. Loss-history
    ax5 = fig.add_subplot(gs[1, 2])
    epochs_plot = range(500, 500 * len(history["total"]) + 1, 500)
    ax5.semilogy(epochs_plot, history["total"], label="Total")
    ax5.semilogy(epochs_plot, history["physics"], label="PDE")
    ax5.semilogy(epochs_plot, history["bc"], label="BC")
    ax5.semilogy(epochs_plot, history["ic"], label="IC")
    ax5.set_title("Loss-History (Adam-Loss)")
    ax5.set_xlabel("Epoch")
    ax5.set_ylabel("MSE Loss (log)")
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    plt.savefig(f"../figures/pinn_heat_results_{BC}.png", dpi=150, bbox_inches="tight")
    print(f"\nPlot saved: pinn_heat_results_{BC}.png")
    plt.show()

    # Relative L2-error
    l2 = np.sqrt(np.mean(error**2)) / np.sqrt(np.mean(u_exact**2))
    print(f"\nRelative L2-Error: {l2:.4e}")


# ── 7. Main ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()

    ALPHA = args.alpha
    N_COLLOC = args.n_colloc
    N_BC = args.n_bc
    N_IC = args.n_ic
    HIDDEN_SIZE = args.hidden_size
    N_LAYERS = args.n_layers
    LR_ADAM = args.lr_adams
    EPOCHS_ADAM = args.epochs_adam
    EPOCHS_LBFGS = args.epochs_lbfgs
    T_FINAL = args.t_final
    BC = args.boundary_condition

    print("=" * 55)
    print("  PINN – 1D Diffusion Equation")
    print(f"  α = {ALPHA},  Collocation points = {N_COLLOC}")
    print("=" * 55)

    model, history = train()
    plot_results(model, history)

    # Save model
    torch.save(model.state_dict(), f"../models/pinn_heat_model_{BC}.pt")
    print(f"Model saved: pinn_heat_model_{BC}.pt")
