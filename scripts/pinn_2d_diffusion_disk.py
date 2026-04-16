"""
Physics-Informed Neural Network (PINN) – 2D Diffusion on a Disk
================================================================

PDE:  ∂u/∂t = α · (∂²u/∂x² + ∂²u/∂y²)   on  Ω = {(x,y) : x² + y² ≤ 1}

Boundary condition (Neumann – impermeable walls):
    ∂u/∂n = ∂u/∂x · x + ∂u/∂y · y = 0   on  ∂Ω (unit circle)

Initial condition:
    u(x, y, 0) = exp(-(x² + y²) / (2·σ²))   (Gaussian bell at the centre)

Validation:
    Due to the radial symmetry of the initial condition the solution remains
    radially symmetric: u(x,y,t) = u(r,t) with r = √(x²+y²).
    We compare against an FDM reference solution in polar coordinates.

New concepts compared to the 1D case:
    - Sampling on a disk (uniform in area)
    - Position-dependent normal vector on the circular boundary
    - 2D Laplace operator: u_xx + u_yy
    - Numerical reference solution instead of analytical solution
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import argparse

# ── Reproducibility ───────────────────────────────────────────────────────────
torch.manual_seed(42)
np.random.seed(42)

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
print(f"Device: {DEVICE}")


# ── Argument parsing ──────────────────────────────────────────────────────────


def parse_args():
    parser = argparse.ArgumentParser(description="PINN – 2D Diffusion on a Disk")
    parser.add_argument(
        "--alpha", type=float, default=0.05, help="Diffusion coefficient"
    )
    parser.add_argument("--sigma", type=float, default=0.2, help="Width of the Gaussian IC")
    parser.add_argument("--t_final", type=float, default=1.0, help="Final time")
    parser.add_argument(
        "--n_colloc", type=int, default=8_000, help="Collocation points (interior)"
    )
    parser.add_argument("--n_bc", type=int, default=500, help="Boundary points")
    parser.add_argument("--n_ic", type=int, default=1_000, help="Initial condition points")
    parser.add_argument(
        "--hidden_size", type=int, default=64, help="Neurons per layer"
    )
    parser.add_argument("--n_layers", type=int, default=5, help="Hidden layers")
    parser.add_argument("--epochs_adam", type=int, default=8_000, help="Adam epochs")
    parser.add_argument("--epochs_lbfgs", type=int, default=500, help="L-BFGS steps")
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate")
    parser.add_argument("--w_bc", type=float, default=10.0, help="BC loss weight")
    parser.add_argument("--w_ic", type=float, default=5.0, help="IC loss weight")
    return parser.parse_args()


# ── 1. Network ────────────────────────────────────────────────────────────────


class PINN(nn.Module):
    """
    Input:  (x, y, t) → 3 features
    Output: u(x, y, t) → 1 scalar

    One layer more than in the 1D case because the solution depends on
    3 variables.
    """

    def __init__(self, hidden: int, n_layers: int):
        super().__init__()
        layers = [nn.Linear(3, hidden), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*layers)

        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, y, t):
        xyt = torch.cat([x, y, t], dim=1)  # Shape: (N, 3)
        return self.net(xyt)               # Shape: (N, 1)


# ── 2. Sampling ───────────────────────────────────────────────────────────────


def sample_disk(n: int, t_final: float):
    """
    Uniform sampling inside the unit disk.

    Naive sampling: r = torch.rand → too many points near the centre,
    because the annular area grows with r. Correction: r = sqrt(torch.rand).
    """
    r = torch.sqrt(torch.rand(n, 1))
    theta = 2 * torch.pi * torch.rand(n, 1)
    x = r * torch.cos(theta)
    y = r * torch.sin(theta)
    t = t_final * torch.rand(n, 1)
    return x.to(DEVICE), y.to(DEVICE), t.to(DEVICE)


def sample_boundary(n: int, t_final: float):
    """
    Uniform sampling on the unit circle (boundary).
    The outward normal vector here is simply n = (x, y).
    """
    theta = 2 * torch.pi * torch.rand(n, 1)
    x = torch.cos(theta)
    y = torch.sin(theta)
    t = t_final * torch.rand(n, 1)
    return x.to(DEVICE), y.to(DEVICE), t.to(DEVICE)


def sample_ic(n: int, sigma: float):
    """
    Initial condition: t=0, Gaussian bell at the centre.
    """
    r = torch.sqrt(torch.rand(n, 1))
    theta = 2 * torch.pi * torch.rand(n, 1)
    x = r * torch.cos(theta)
    y = r * torch.sin(theta)
    t = torch.zeros(n, 1)
    u_true = torch.exp(-(x**2 + y**2) / (2 * sigma**2))
    return x.to(DEVICE), y.to(DEVICE), t.to(DEVICE), u_true.to(DEVICE)


# ── 3. Loss terms ─────────────────────────────────────────────────────────────


def physics_loss(model, x, y, t):
    """
    Residual: ∂u/∂t - α·(∂²u/∂x² + ∂²u/∂y²) = 0

    2D Laplacian = u_xx + u_yy, both computed via autograd.
    """
    x = x.requires_grad_(True)
    y = y.requires_grad_(True)
    t = t.requires_grad_(True)

    u = model(x, y, t)

    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(
        u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True
    )[0]
    u_yy = torch.autograd.grad(
        u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True
    )[0]

    residual = u_t - ALPHA * (u_xx + u_yy)
    return torch.mean(residual**2)


def bc_loss(model, x, y, t):
    """
    Neumann boundary condition: ∂u/∂n = ∂u/∂x·x + ∂u/∂y·y = 0

    On the unit circle the outward normal vector is n = (x, y).
    The flux through the boundary is the dot product of the gradient with n.
    """
    x = x.requires_grad_(True)
    y = y.requires_grad_(True)

    u = model(x, y, t)
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]

    # Normal derivative = gradient · normal vector
    dudn = u_x * x + u_y * y
    return torch.mean(dudn**2)


def ic_loss(model, x, y, t, u_true):
    """
    u(x, y, 0) = exp(-(x²+y²) / (2σ²))
    """
    u_pred = model(x, y, t)
    return torch.mean((u_pred - u_true) ** 2)


# ── 4. Numerical reference solution (FDM in polar coordinates) ────────────────


def fdm_reference(
    alpha: float, sigma: float, t_final: float, nr: int = 200, nt: int = None
):
    """
    Finite-difference solution of the radially symmetric diffusion equation:

        ∂u/∂t = α · (∂²u/∂r² + 1/r · ∂u/∂r)

    Boundary condition: ∂u/∂r = 0 at r=1  (Neumann)
    Symmetry condition: ∂u/∂r = 0 at r=0

    This reference is valid only because the Gaussian IC is radially symmetric.
    """
    r = np.linspace(0, 1, nr)
    dr = r[1] - r[0]

    # Stability condition: α·dt/dr² < 0.5
    # → dt < 0.5·dr²/α  → nt > t_final / (0.5·dr²/α)
    nt_min = int(
        np.ceil(t_final / (0.4 * dr**2 / alpha))
    )  # 0.4 instead of 0.5 as a safety margin
    if nt is None or nt < nt_min:
        nt = nt_min
        print(f"  FDM: nt automatically set to {nt} (stability condition)")
    dt = t_final / nt

    # Check stability condition
    assert alpha * dt / dr**2 < 0.5, "FDM unstable – increase nt"

    u = np.exp(-(r**2) / (2 * sigma**2))

    for _ in range(nt):
        u_new = u.copy()
        # Interior points
        for i in range(1, nr - 1):
            u_new[i] = u[i] + alpha * dt * (
                (u[i + 1] - 2 * u[i] + u[i - 1]) / dr**2
                + (u[i + 1] - u[i - 1]) / (2 * r[i] * dr)
            )
        # Symmetry at r=0: ∂u/∂r = 0  →  ghost-point method
        u_new[0] = u[0] + alpha * dt * 2 * (u[1] - u[0]) / dr**2
        # Neumann at r=1: ∂u/∂r = 0  →  u[nr] = u[nr-2]
        u_new[-1] = u_new[-2]
        u = u_new

    return r, u


# ── 5. Training ───────────────────────────────────────────────────────────────


def train(args):
    global ALPHA
    ALPHA = args.alpha

    model = PINN(args.hidden_size, args.n_layers).to(DEVICE)

    # Generate data once
    x_col, y_col, t_col = sample_disk(args.n_colloc, args.t_final)
    x_bc, y_bc, t_bc = sample_boundary(args.n_bc, args.t_final)
    x_ic, y_ic, t_ic, u_ic_true = sample_ic(args.n_ic, args.sigma)

    history = {"total": [], "physics": [], "bc": [], "ic": []}

    # ── Phase 1: Adam ──────────────────────────────────────────────────────────
    print("\n── Phase 1: Adam ─────────────────────────────────────")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=300, factor=0.5
    )

    for epoch in range(1, args.epochs_adam + 1):
        optimizer.zero_grad()

        loss_phys = physics_loss(model, x_col, y_col, t_col)
        loss_bc = bc_loss(model, x_bc, y_bc, t_bc)
        loss_ic = ic_loss(model, x_ic, y_ic, t_ic, u_ic_true)
        loss = loss_phys + args.w_bc * loss_bc + args.w_ic * loss_ic

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step(loss.detach())

        if epoch % 500 == 0:
            history["total"].append(loss.item())
            history["physics"].append(loss_phys.item())
            history["bc"].append(loss_bc.item())
            history["ic"].append(loss_ic.item())
            lr_now = optimizer.param_groups[0]["lr"]
            print(
                f"  Epoch {epoch:5d} | Total: {loss.item():.2e} "
                f"| PDE: {loss_phys.item():.2e} "
                f"| BC: {loss_bc.item():.2e} "
                f"| IC: {loss_ic.item():.2e} "
                f"| LR: {lr_now:.2e}"
            )

    # ── Phase 2: L-BFGS ───────────────────────────────────────────────────────
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
        lp = physics_loss(model, x_col, y_col, t_col)
        lb = bc_loss(model, x_bc, y_bc, t_bc)
        li = ic_loss(model, x_ic, y_ic, t_ic, u_ic_true)
        loss = lp + args.w_bc * lb + args.w_ic * li
        loss.backward()
        step[0] += 1
        if step[0] % 100 == 0:
            print(f"  L-BFGS step {step[0]:4d} | Total: {loss.item():.2e}")
        return loss

    for _ in range(args.epochs_lbfgs // 20):
        optimizer_lbfgs.step(closure)

    return model, history


# ── 6. Visualisation ──────────────────────────────────────────────────────────


def plot_results(model, history, args):
    model.eval()

    # Grid over the disk
    n_grid = 150
    lin = np.linspace(-1, 1, n_grid)
    X, Y = np.meshgrid(lin, lin)
    mask = X**2 + Y**2 <= 1.0  # only points inside the circle

    def predict_at_time(t_val):
        x_flat = (
            torch.tensor(X[mask].flatten(), dtype=torch.float32).unsqueeze(1).to(DEVICE)
        )
        y_flat = (
            torch.tensor(Y[mask].flatten(), dtype=torch.float32).unsqueeze(1).to(DEVICE)
        )
        t_flat = torch.full_like(x_flat, t_val)
        with torch.no_grad():
            u = model(x_flat, y_flat, t_flat).cpu().numpy().flatten()
        result = np.full(X.shape, np.nan)
        result[mask] = u
        return result

    t_snaps = [0.0, 0.1, 0.3, args.t_final]

    # FDM reference
    r_ref, u_ref = fdm_reference(args.alpha, args.sigma, args.t_final)

    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(
        "PINN – 2D Diffusion on a Disk (Neumann boundary)",
        fontsize=14,
        fontweight="bold",
    )
    gs = GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.35)

    # ── Row 1: 2D snapshots ───────────────────────────────────────────────────
    for i, t_val in enumerate(t_snaps):
        ax = fig.add_subplot(gs[0, i])
        u2d = predict_at_time(t_val)
        vmax = np.nanmax(predict_at_time(0.0))
        im = ax.imshow(
            u2d, origin="lower", extent=[-1, 1, -1, 1], cmap="hot", vmin=0, vmax=vmax
        )
        # Draw circular boundary
        theta_plot = np.linspace(0, 2 * np.pi, 300)
        ax.plot(np.cos(theta_plot), np.sin(theta_plot), "w--", lw=0.8)
        ax.set_title(f"t = {t_val:.2f}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(im, ax=ax, fraction=0.046)

    # ── Row 2 left: Radial profile PINN vs FDM ────────────────────────────────
    ax_rad = fig.add_subplot(gs[1, :2])
    r_vals = np.linspace(0, 1, 100)
    x_r = torch.tensor(r_vals, dtype=torch.float32).unsqueeze(1).to(DEVICE)
    y_r = torch.zeros_like(x_r)

    for t_val in [0.0, 0.1, 0.3, args.t_final]:
        t_r = torch.full_like(x_r, t_val)
        with torch.no_grad():
            u_pinn = model(x_r, y_r, t_r).cpu().numpy().flatten()
        ax_rad.plot(r_vals, u_pinn, label=f"PINN t={t_val:.2f}")

    # FDM for t_final only
    ax_rad.plot(r_ref, u_ref, "k--", lw=2, label=f"FDM t={args.t_final:.2f}")
    ax_rad.set_title("Radial profile u(r,0,t) – PINN vs FDM reference")
    ax_rad.set_xlabel("r = x  (along y=0)")
    ax_rad.set_ylabel("u")
    ax_rad.legend(fontsize=8)
    ax_rad.grid(True, alpha=0.3)

    # ── Row 2 right: Conservation of total mass ───────────────────────────────
    ax_mass = fig.add_subplot(gs[1, 2:])
    t_mass = np.linspace(0, args.t_final, 30)
    masses = []

    # Monte-Carlo integration of the mass over the disk
    n_mc = 3000
    r_mc = torch.sqrt(torch.rand(n_mc, 1)).to(DEVICE)
    th_mc = (2 * torch.pi * torch.rand(n_mc, 1)).to(DEVICE)
    x_mc = r_mc * torch.cos(th_mc)
    y_mc = r_mc * torch.sin(th_mc)

    for t_val in t_mass:
        t_mc = torch.full((n_mc, 1), t_val).to(DEVICE)
        with torch.no_grad():
            u_mc = model(x_mc, y_mc, t_mc).cpu().numpy().flatten()
        # Area of unit disk = π, Monte-Carlo estimator: π · mean(u)
        masses.append(np.pi * np.mean(u_mc))

    mass_0 = masses[0]
    ax_mass.plot(t_mass, np.array(masses) / mass_0, "b-o", markersize=3)
    ax_mass.axhline(1.0, color="k", linestyle="--", label="Ideal (conserved)")
    ax_mass.set_title("Mass conservation (Neumann → mass conserved)")
    ax_mass.set_xlabel("t")
    ax_mass.set_ylabel("Mass / Mass(t=0)")
    ax_mass.legend()
    ax_mass.grid(True, alpha=0.3)
    ax_mass.set_ylim(0.8, 1.2)

    # ── Row 3: Loss history ───────────────────────────────────────────────────
    ax_loss = fig.add_subplot(gs[2, :])
    epochs_plot = range(500, 500 * len(history["total"]) + 1, 500)
    ax_loss.semilogy(epochs_plot, history["total"], label="Total")
    ax_loss.semilogy(epochs_plot, history["physics"], label="PDE")
    ax_loss.semilogy(epochs_plot, history["bc"], label="BC (Neumann)")
    ax_loss.semilogy(epochs_plot, history["ic"], label="IC")
    ax_loss.set_title("Loss history (Adam phase)")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("MSE Loss (log)")
    ax_loss.legend()
    ax_loss.grid(True, alpha=0.3)

    plt.savefig("../figures/pinn_diffusion_2d_disk.png", dpi=150, bbox_inches="tight")
    print("\nPlot saved: pinn_diffusion_2d_disk.png")
    plt.show()


# ── 7. Main ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()

    print("=" * 60)
    print("  PINN – 2D Diffusion on a Disk")
    print(f"  α={args.alpha}, σ={args.sigma}, T={args.t_final}")
    print(f"  Collocation={args.n_colloc}, BC={args.n_bc}, IC={args.n_ic}")
    print(f"  Network: {args.n_layers} layers × {args.hidden_size} neurons")
    print("=" * 60)

    model, history = train(args)
    plot_results(model, history, args)

    torch.save(model.state_dict(), "../models/pinn_diffusion_2d_disk.pt")
    print("Model saved: pinn_diffusion_2d_disk.pt")
