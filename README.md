# Latent-Parameterized Geometric Compensation

A neural geometric-compensation framework that searches a low-dimensional latent space instead of optimizing thousands of 3D coordinates independently.

The core idea is to represent a pre-compensated geometry through a neural decoder, apply a synthetic manufacturing-distortion model, and optimize the latent representation so the final built shape approaches the target design. This gives the compensation problem a compact, structured geometric parameterization rather than an unconstrained pointwise one.

## Highlights

- **Low-dimensional compensation search** through latent variables
- **Neural decoder geometry parameterization** for structured 3D shape updates
- **Simulation-driven optimization** against manufacturing distortion
- **Chamfer-distance geometric objectives** for target recovery
- Experiments across **multiple deformation conditions** and **multiple geometry scales**
- Automatic visualization of compensation quality and pointwise geometric error

## Method

Each experiment combines three components:

1. **Target geometry** — the desired final 3D shape.
2. **Latent-parameterized compensation** — a neural decoder maps latent coordinates to a candidate pre-compensated geometry.
3. **Manufacturing response** — a prescribed deformation process maps the compensated geometry to the simulated built shape.

Optimization updates the latent representation to reduce the geometric mismatch between the built shape and the target.

## Experiments

| Script | Experiment | Description |
|---|---|---|
| `baseline.py` | Baseline | One spherical target with one nonlinear synthetic deformation model. |
| `variation1.py` | Multiple deformation conditions | One target geometry evaluated across five deformation parameter sets using a shared decoder and separate latent vectors. |
| `variation2.py` | Multiple geometry scales | Five spherical target sizes evaluated under one shared deformation model using a shared decoder and separate latent vectors. |

## Outputs

The scripts generate visualization files such as:

- `comparison.png` — target, uncompensated deformation, compensation geometry, and final built geometry
- `error.png` — pointwise geometric-error visualization
- `rotation.gif` — rotating 3D visualization of the compensated result
- `sample_*_detailed.png` and `sample1_rotation.gif` — multi-deformation outputs
- `sphere_*_comparison.png` — multi-size outputs

## Installation

```bash
pip install -r requirements.txt
```

CUDA is used automatically when available; the experiments also run on CPU.

## Running

```bash
python baseline.py
python variation1.py
python variation2.py
```

The experiments are self-contained and generate their target point clouds and synthetic deformation conditions directly in code.

## Research direction

This project explores a broader idea for manufacturing compensation: **learn or construct a compact shape space first, then solve compensation inside that space**. The same framework can be extended from synthetic distortion functions to FEM-based, experimental, or learned manufacturing-response models.

## Implementation note

The current synthetic deformation functions are evaluated with a stop-gradient update (`detach()` in PyTorch). In this implementation, the optimizer updates compensation through the decoder/latent parameterization while treating each evaluated deformation realization as an external response. A fully differentiable process model could be plugged into the same framework when available.
