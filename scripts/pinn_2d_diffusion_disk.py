"""
Physics-Informed Neural Network (PINN) – 2D Diffusion auf einer Kreisscheibe
=============================================================================

PDE:  ∂u/∂t = α · (∂²u/∂x² + ∂²u/∂y²)   auf  Ω = {(x,y) : x² + y² ≤ 1}

Randbedingung (Neumann – undurchlässige Wände):
    ∂u/∂n = ∂u/∂x · x + ∂u/∂y · y = 0   auf  ∂Ω (Einheitskreis)

Anfangsbedingung:
    u(x, y, 0) = exp(-(x² + y²) / (2·σ²))   (Gauß-Glocke im Zentrum)

Validierung:
    Durch die Radialsymmetrie der Anfangsbedingung bleibt die Lösung
    radialsymmetrisch: u(x,y,t) = u(r,t) mit r = √(x²+y²).
    Wir vergleichen mit einer FDM-Referenzlösung in Polarkoordinaten.

Neue Konzepte gegenüber dem 1D-Fall:
    - Sampling auf einer Kreisscheibe (gleichmäßig in der Fläche)
    - Ortsabhängiger Normalenvektor auf dem Kreisrand
    - 2D-Laplace-Operator: u_xx + u_yy
    - Numerische Referenzlösung statt analytischer Lösung
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import argparse

# ── Reproduzierbarkeit ────────────────────────────────────────────────────────
torch.manual_seed(42)
np.random.seed(42)

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
print(f"Gerät: {DEVICE}")


# ── Argparse ──────────────────────────────────────────────────────────────────


def parse_args():
    parser = argparse.ArgumentParser(description="PINN – 2D Diffusion auf Kreisscheibe")
    parser.add_argument(
        "--alpha", type=float, default=0.05, help="Diffusionskoeffizient"
    )
    parser.add_argument("--sigma", type=float, default=0.2, help="Breite der Gauß-IC")
    parser.add_argument("--t_final", type=float, default=1.0, help="Endzeitpunkt")
    parser.add_argument(
        "--n_colloc", type=int, default=8_000, help="Kollokationspunkte (Inneres)"
    )
    parser.add_argument("--n_bc", type=int, default=500, help="Randpunkte")
    parser.add_argument("--n_ic", type=int, default=1_000, help="Anfangspunkte")
    parser.add_argument(
        "--hidden_size", type=int, default=64, help="Neuronen pro Schicht"
    )
    parser.add_argument("--n_layers", type=int, default=5, help="Versteckte Schichten")
    parser.add_argument("--epochs_adam", type=int, default=8_000, help="Adam Epochen")
    parser.add_argument("--epochs_lbfgs", type=int, default=500, help="L-BFGS Schritte")
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam Lernrate")
    parser.add_argument("--w_bc", type=float, default=10.0, help="Gewicht BC-Loss")
    parser.add_argument("--w_ic", type=float, default=5.0, help="Gewicht IC-Loss")
    return parser.parse_args()


# ── 1. Netzwerk ───────────────────────────────────────────────────────────────


class PINN(nn.Module):
    """
    Eingabe:  (x, y, t) → 3 Features
    Ausgabe:  u(x, y, t) → 1 Skalar

    Ein Layer mehr als im 1D-Fall, weil die Lösung von 3 Variablen abhängt.
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
        return self.net(xyt)  # Shape: (N, 1)


# ── 2. Sampling ───────────────────────────────────────────────────────────────


def sample_disk(n: int, t_final: float):
    """
    Gleichmäßiges Sampling im Inneren der Einheitskreisscheibe.

    Naives Sampling: r = torch.rand → zu viele Punkte nahe dem Zentrum,
    weil die Ringfläche mit r wächst. Korrektur: r = sqrt(torch.rand).
    """
    r = torch.sqrt(torch.rand(n, 1))
    theta = 2 * torch.pi * torch.rand(n, 1)
    x = r * torch.cos(theta)
    y = r * torch.sin(theta)
    t = t_final * torch.rand(n, 1)
    return x.to(DEVICE), y.to(DEVICE), t.to(DEVICE)


def sample_boundary(n: int, t_final: float):
    """
    Gleichmäßiges Sampling auf dem Einheitskreis (Rand).
    Der Normalenvektor ist hier einfach n = (x, y).
    """
    theta = 2 * torch.pi * torch.rand(n, 1)
    x = torch.cos(theta)
    y = torch.sin(theta)
    t = t_final * torch.rand(n, 1)
    return x.to(DEVICE), y.to(DEVICE), t.to(DEVICE)


def sample_ic(n: int, sigma: float):
    """
    Anfangsbedingung: t=0, Gauß-Glocke im Zentrum.
    """
    r = torch.sqrt(torch.rand(n, 1))
    theta = 2 * torch.pi * torch.rand(n, 1)
    x = r * torch.cos(theta)
    y = r * torch.sin(theta)
    t = torch.zeros(n, 1)
    u_true = torch.exp(-(x**2 + y**2) / (2 * sigma**2))
    return x.to(DEVICE), y.to(DEVICE), t.to(DEVICE), u_true.to(DEVICE)


# ── 3. Loss-Terme ─────────────────────────────────────────────────────────────


def physics_loss(model, x, y, t):
    """
    Residuum: ∂u/∂t - α·(∂²u/∂x² + ∂²u/∂y²) = 0

    2D-Laplace = u_xx + u_yy, beide via autograd.
    """
    x = x.requires_grad_(True)
    y = y.requires_grad_(True)
    t = t.requires_grad_(True)

    u = model(x, y, t)

    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[
        0
    ]
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[
        0
    ]
    u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[
        0
    ]
    u_xx = torch.autograd.grad(
        u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True
    )[0]
    u_yy = torch.autograd.grad(
        u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True
    )[0]

    residuum = u_t - ALPHA * (u_xx + u_yy)
    return torch.mean(residuum**2)


def bc_loss(model, x, y, t):
    """
    Neumann-Randbedingung: ∂u/∂n = ∂u/∂x·x + ∂u/∂y·y = 0

    Auf dem Einheitskreis gilt: Normalenvektor n = (x, y).
    Der Fluss durch den Rand ist das Skalarprodukt des Gradienten mit n.
    """
    x = x.requires_grad_(True)
    y = y.requires_grad_(True)

    u = model(x, y, t)
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[
        0
    ]
    u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[
        0
    ]

    # Normalableitung = Gradient · Normalenvektor
    dudn = u_x * x + u_y * y
    return torch.mean(dudn**2)


def ic_loss(model, x, y, t, u_true):
    """
    u(x, y, 0) = exp(-(x²+y²) / (2σ²))
    """
    u_pred = model(x, y, t)
    return torch.mean((u_pred - u_true) ** 2)


# ── 4. Numerische Referenzlösung (FDM in Polarkoordinaten) ───────────────────


def fdm_reference(
    alpha: float, sigma: float, t_final: float, nr: int = 200, nt: int = None
):
    """
    Finite-Differenzen-Lösung der radialsymmetrischen Diffusionsgleichung:

        ∂u/∂t = α · (∂²u/∂r² + 1/r · ∂u/∂r)

    Randbedingung: ∂u/∂r = 0 bei r=1  (von Neumann)
    Symmetriebedingung: ∂u/∂r = 0 bei r=0

    Diese Referenz gilt nur weil die Gauß-IC radialsymmetrisch ist.
    """
    r = np.linspace(0, 1, nr)
    dr = r[1] - r[0]

    # Stabilitätsbedingung: α·dt/dr² < 0.5
    # → dt < 0.5·dr²/α  → nt > t_final / (0.5·dr²/α)
    nt_min = int(
        np.ceil(t_final / (0.4 * dr**2 / alpha))
    )  # 0.4 statt 0.5 als Sicherheitspuffer
    if nt is None or nt < nt_min:
        nt = nt_min
        print(f"  FDM: nt automatisch auf {nt} gesetzt (Stabilitätsbedingung)")
    dt = t_final / nt

    # Stabilitätsbedingung prüfen
    assert alpha * dt / dr**2 < 0.5, "FDM instabil – nt erhöhen"

    u = np.exp(-(r**2) / (2 * sigma**2))

    for _ in range(nt):
        u_new = u.copy()
        # Innere Punkte
        for i in range(1, nr - 1):
            u_new[i] = u[i] + alpha * dt * (
                (u[i + 1] - 2 * u[i] + u[i - 1]) / dr**2
                + (u[i + 1] - u[i - 1]) / (2 * r[i] * dr)
            )
        # Symmetrie bei r=0: ∂u/∂r = 0  →  Ghost-Point-Methode
        u_new[0] = u[0] + alpha * dt * 2 * (u[1] - u[0]) / dr**2
        # Neumann bei r=1: ∂u/∂r = 0  →  u[nr] = u[nr-2]
        u_new[-1] = u_new[-2]
        u = u_new

    return r, u


# ── 5. Training ───────────────────────────────────────────────────────────────


def train(args):
    global ALPHA
    ALPHA = args.alpha

    model = PINN(args.hidden_size, args.n_layers).to(DEVICE)

    # Daten einmal erzeugen
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
                f"  Epoche {epoch:5d} | Gesamt: {loss.item():.2e} "
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
            print(f"  L-BFGS Schritt {step[0]:4d} | Gesamt: {loss.item():.2e}")
        return loss

    for _ in range(args.epochs_lbfgs // 20):
        optimizer_lbfgs.step(closure)

    return model, history


# ── 6. Visualisierung ─────────────────────────────────────────────────────────


def plot_results(model, history, args):
    model.eval()

    # Gitter über die Kreisscheibe
    n_grid = 150
    lin = np.linspace(-1, 1, n_grid)
    X, Y = np.meshgrid(lin, lin)
    mask = X**2 + Y**2 <= 1.0  # nur Punkte innerhalb des Kreises

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

    # FDM-Referenz
    r_ref, u_ref = fdm_reference(args.alpha, args.sigma, args.t_final)

    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(
        "PINN – 2D Diffusion auf Kreisscheibe (Neumann-Rand)",
        fontsize=14,
        fontweight="bold",
    )
    gs = GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.35)

    # ── Zeile 1: 2D-Schnappschüsse ────────────────────────────────────────────
    for i, t_val in enumerate(t_snaps):
        ax = fig.add_subplot(gs[0, i])
        u2d = predict_at_time(t_val)
        vmax = np.nanmax(predict_at_time(0.0))
        im = ax.imshow(
            u2d, origin="lower", extent=[-1, 1, -1, 1], cmap="hot", vmin=0, vmax=vmax
        )
        # Kreisrand einzeichnen
        theta_plot = np.linspace(0, 2 * np.pi, 300)
        ax.plot(np.cos(theta_plot), np.sin(theta_plot), "w--", lw=0.8)
        ax.set_title(f"t = {t_val:.2f}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(im, ax=ax, fraction=0.046)

    # ── Zeile 2 links: Radialprofil PINN vs FDM ───────────────────────────────
    ax_rad = fig.add_subplot(gs[1, :2])
    r_vals = np.linspace(0, 1, 100)
    x_r = torch.tensor(r_vals, dtype=torch.float32).unsqueeze(1).to(DEVICE)
    y_r = torch.zeros_like(x_r)

    for t_val in [0.0, 0.1, 0.3, args.t_final]:
        t_r = torch.full_like(x_r, t_val)
        with torch.no_grad():
            u_pinn = model(x_r, y_r, t_r).cpu().numpy().flatten()
        ax_rad.plot(r_vals, u_pinn, label=f"PINN t={t_val:.2f}")

    # FDM nur für t_final
    ax_rad.plot(r_ref, u_ref, "k--", lw=2, label=f"FDM t={args.t_final:.2f}")
    ax_rad.set_title("Radialprofil u(r,0,t) – PINN vs FDM Referenz")
    ax_rad.set_xlabel("r = x  (entlang y=0)")
    ax_rad.set_ylabel("u")
    ax_rad.legend(fontsize=8)
    ax_rad.grid(True, alpha=0.3)

    # ── Zeile 2 rechts: Erhaltung der Gesamtmasse ─────────────────────────────
    ax_mass = fig.add_subplot(gs[1, 2:])
    t_mass = np.linspace(0, args.t_final, 30)
    masses = []

    # Monte-Carlo-Integration der Masse über die Kreisscheibe
    n_mc = 3000
    r_mc = torch.sqrt(torch.rand(n_mc, 1)).to(DEVICE)
    th_mc = (2 * torch.pi * torch.rand(n_mc, 1)).to(DEVICE)
    x_mc = r_mc * torch.cos(th_mc)
    y_mc = r_mc * torch.sin(th_mc)

    for t_val in t_mass:
        t_mc = torch.full((n_mc, 1), t_val).to(DEVICE)
        with torch.no_grad():
            u_mc = model(x_mc, y_mc, t_mc).cpu().numpy().flatten()
        # Fläche Einheitskreis = π, Monte-Carlo-Schätzer: π · mean(u)
        masses.append(np.pi * np.mean(u_mc))

    mass_0 = masses[0]
    ax_mass.plot(t_mass, np.array(masses) / mass_0, "b-o", markersize=3)
    ax_mass.axhline(1.0, color="k", linestyle="--", label="Ideal (konserviert)")
    ax_mass.set_title("Massenerhaltung (Neumann → Masse konserviert)")
    ax_mass.set_xlabel("t")
    ax_mass.set_ylabel("Masse / Masse(t=0)")
    ax_mass.legend()
    ax_mass.grid(True, alpha=0.3)
    ax_mass.set_ylim(0.8, 1.2)

    # ── Zeile 3: Loss-Verlauf ─────────────────────────────────────────────────
    ax_loss = fig.add_subplot(gs[2, :])
    epochs_plot = range(500, 500 * len(history["total"]) + 1, 500)
    ax_loss.semilogy(epochs_plot, history["total"], label="Gesamt")
    ax_loss.semilogy(epochs_plot, history["physics"], label="PDE")
    ax_loss.semilogy(epochs_plot, history["bc"], label="BC (Neumann)")
    ax_loss.semilogy(epochs_plot, history["ic"], label="IC")
    ax_loss.set_title("Loss-Verlauf (Adam-Phase)")
    ax_loss.set_xlabel("Epoche")
    ax_loss.set_ylabel("MSE Loss (log)")
    ax_loss.legend()
    ax_loss.grid(True, alpha=0.3)

    plt.savefig("../figures/pinn_diffusion_2d_disk.png", dpi=150, bbox_inches="tight")
    print("\nPlot gespeichert: pinn_diffusion_2d_disk.png")
    plt.show()


# ── 7. Hauptprogramm ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()

    print("=" * 60)
    print("  PINN – 2D Diffusion auf Kreisscheibe")
    print(f"  α={args.alpha}, σ={args.sigma}, T={args.t_final}")
    print(f"  Kollokation={args.n_colloc}, BC={args.n_bc}, IC={args.n_ic}")
    print(f"  Netz: {args.n_layers} Schichten × {args.hidden_size} Neuronen")
    print("=" * 60)

    model, history = train(args)
    plot_results(model, history, args)

    torch.save(model.state_dict(), "../models/pinn_diffusion_2d_disk.pt")
    print("Modell gespeichert: pinn_diffusion_2d_disk.pt")
