# Latent Shape Compensation

A self-supervised framework for correcting geometric distortion in manufacturing by optimizing a latent vector through a neural decoder — no labels or ground-truth data needed.

---

##  Method

This project learns to generate **pre-compensated 3D shapes** that will deform into the target design:

- **Input**: 1. A 3D point cloud of the target shape； 2. A deformation function
- **Output**: A compensated shape that can align with the target design model after distortion

Core components:
-  **Latent vector `z`**: Encodes the compensation strategy
-  **Decoder-only MLP**: Maps `z` to a 3D point cloud
-  **Loss**: Chamfer distance

## 🧪 Experiment Setups

| Folder        | Description                               |
|---------------|-------------------------------------------|
| `baseline/`   | Single object, single deformation         |
| `variation1/` | Multiple sizes, fixed deformation         |
| `variation2/` | Single object, multiple deformations      |

### 3. Output files
- `comparison.png` — 4-view comparison (original, deformed, compensated, built)
- `error.png` — Heatmap of error magnitude
- `rotation.gif` — Rotating 3D view of final built shape
