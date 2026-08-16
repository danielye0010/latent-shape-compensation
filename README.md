# Latent-Parameterized Geometric Compensation

A simulation-driven prototype for geometric compensation in manufacturing using a neural decoder as a compact shape parameterization.

The core idea is simple: instead of optimizing every 3D point independently, the compensation geometry is generated through a low-dimensional latent representation. The resulting pre-compensated shape is evaluated under a prescribed synthetic deformation model and optimized to reduce the mismatch between the manufactured shape and the target design.

## Method

Each experiment uses three components:

1. **Target geometry** — a synthetic 3D point cloud representing the desired final shape.
2. **Latent-parameterized compensation** — a neural decoder maps a latent vector to a candidate pre-compensated point cloud.
3. **Manufacturing distortion model** — the candidate geometry is passed through a prescribed synthetic deformation process, and the resulting built shape is compared with the target using Chamfer distance and/or pointwise error.

The decoder provides a structured geometric parameterization, while the latent variables provide a compact set of compensation coordinates.

### Optimization scope

The deformation functions in the current experiments are evaluated with a **stop-gradient update** (`detach()` in PyTorch). The optimization therefore treats the current deformation realization as an external response and updates the compensation geometry through the decoder/latent parameterization, rather than differentiating through the deformation law itself.

This makes the repository a useful prototype for studying latent geometric parameterization under synthetic manufacturing response without requiring a fully differentiable process model.

## Experiments

| Script | Experiment | Description |
|---|---|---|
| `baseline.py` | Baseline | One spherical target with one nonlinear synthetic deformation model. The decoder and latent representation are initialized jointly, followed by latent-focused compensation refinement. |
| `variation1.py` | Multiple deformation conditions | One target geometry evaluated across five different deformation parameter sets using a shared decoder and separate latent vectors. |
| `variation2.py` | Multiple geometry scales | Five spherical target sizes evaluated under one shared deformation model using a shared decoder and separate latent vectors. |

## Outputs

The scripts generate visualization files such as:

- `comparison.png` — target, raw deformation, compensation geometry, and final built geometry
- `error.png` — pointwise geometric error visualization
- `rotation.gif` — rotating visualization of the compensated result
- `sample_*_detailed.png` and `sample1_rotation.gif` — multi-deformation experiment outputs
- `sphere_*_comparison.png` — multi-size experiment outputs

Generated images and GIFs are treated as experiment outputs and are ignored by Git by default.

## Installation

```bash
pip install -r requirements.txt
```

The experiments require Python with PyTorch, NumPy, Matplotlib, and Pillow. CUDA is used automatically when available; otherwise the scripts run on CPU.

## Running

```bash
python baseline.py
python variation1.py
python variation2.py
```

The experiments are self-contained and generate their target point clouds and synthetic deformation conditions directly in code.

## Research intent

This repository explores whether a compact neural shape parameterization can organize geometric compensation more effectively than unconstrained pointwise updates. It is intended as a controlled computational prototype for latent-parameterized compensation, not as a calibrated physical manufacturing model.
