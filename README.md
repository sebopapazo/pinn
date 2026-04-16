# PINN: Physics-Informed Neural Networks

[![GitHub License](https://img.shields.io/github/license/sebopapazo/pinn)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)

Physics-Informed Neural Networks (PINNs) for solving partial differential equations (PDEs) by embedding physical laws directly into neural network training. Ideal for scientific computing in physics, including forward/inverse problems without extensive data [web:16].

This repo provides PyTorch implementations for high-dimensional PDEs, extending traditional numerical methods with ML efficiency.

## Features
- Solves PDEs like Burgers', Helmholtz, or Poisson equations.
- Supports boundary conditions and physics losses.
- Hyperparameter tuning with tools like XGBoost-inspired grids (aligned with your ML interests).
- Jupyter notebooks for quick experimentation.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/sebopapazo/pinn.git
   cd pinn
   ```
2. Create a virtual environment (recommended for MacOS/VSCode/Jupyter):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install torch numpy scipy matplotlib pydoe deepxde  # Add transformers if used
   pip install -r requirements.txt  # If you add this file
   ```

Test with:
```bash
python -c "import torch; print(torch.__version__)"
```

## Quick Start / Usage
Train a PINN on a sample PDE (e.g., 1D Burgers' equation):

```python
# Example from train.py or notebook
import torch
from pinn_model import PINN  # Your main model

model = PINN()
model.train(num_epochs=10000, lr=1e-3)
model.plot_solution()
```

Run a notebook:
```bash
jupyter notebook notebooks/burgers_pinn.ipynb
```

Expected output: Loss convergence plot and PDE solution visualization [web:17].

## Examples
- **Forward PDE solving**: Approximates solutions on arbitrary grids.
- **Inverse problems**: Infers parameters from sparse data.
- Add screenshots or GIFs here, e.g., ![Loss Curve](assets/loss.png)

For physics-informed setups, losses include data mismatch + PDE residual + boundary terms [web:16].

## Contributing
1. Fork the repo and create a feature branch (`git checkout -b feature/amazing-feature`).
2. Commit changes (`git commit -m 'Add amazing feature'`).
3. Push and open a PR.

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines. Issues/PRs welcome for new PDEs or optimizations!

## Related Work
Inspired by seminal PINN papers [web:12]. Check similar repos like [sebbas/poisson-pinn](https://github.com/sebbas/poisson-pinn) [web:11].

## License
MIT License - see [LICENSE](LICENSE) [web:1].