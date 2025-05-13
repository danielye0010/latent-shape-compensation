import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib import animation

# ----------------------------------------
# 1. Device & reproducibility
# ----------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(0)
np.random.seed(0)

# ----------------------------------------
# 2. Generate multiple spheres of different sizes
# ----------------------------------------
radii = [0.8, 1.0, 1.2, 1.5, 0.6]  # different sphere radii
N_samples = len(radii)

N_theta, N_phi = 30, 60  # resolution per sphere
theta = np.linspace(0, np.pi, N_theta)
phi   = np.linspace(0, 2 * np.pi, N_phi)
Θ, Φ  = np.meshgrid(theta, phi)
M = N_theta * N_phi

# assemble design point clouds and convert to tensors
design_list = []
design_ts = []
for r in radii:
    x = r * np.sin(Θ) * np.cos(Φ)
    y = r * np.sin(Θ) * np.sin(Φ)
    z = r * np.cos(Θ)
    pts = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)  # (M,3)
    design_list.append(pts)
    design_ts.append(torch.as_tensor(pts, dtype=torch.float32, device=device))
print(f"Prepared {N_samples} design spheres, each with {M} points")

# ------------------------------------------------
# 3. A single, shared deformation function
# ------------------------------------------------
def deform(p):
    x, y, z = p[:,0], p[:,1], p[:,2]
    dx = 0.10 * torch.sin(3.0 * torch.pi * y) + 0.05 * y
    dy = 0.10 * torch.cos(3.0 * torch.pi * x) - 0.05 * x
    dz = -0.08 * (x**2 + y**2)
    dz += 0.03 * torch.sin(30 * x) * torch.cos(30 * y)
    dz += 0.01 * torch.randn_like(z)
    return torch.stack([dx, dy, dz], dim=1).detach()

# ------------------------------------------------
# 4. Shared PointDecoder: latent → compensation points
# ------------------------------------------------
class PointDecoder(nn.Module):
    def __init__(self, z_dim=64, out_points=M):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, 256), nn.ReLU(),
            nn.Linear(256, 256),   nn.ReLU(),
            nn.Linear(256, out_points * 3)
        )
        self.out_points = out_points

    def forward(self, z):
        B = z.shape[0]
        pts = self.net(z)                        # (B, M*3)
        return pts.view(B, self.out_points, 3)   # (B, M, 3)

decoder = PointDecoder().to(device)

# ---------------------------------------
# 5. Chamfer distance for point clouds
# ---------------------------------------
def chamfer(a, b):
    d = torch.cdist(a, b)
    return d.min(1).values.mean() + d.min(0).values.mean()

# --------------------------------
# 6. Initialize latent vectors z
# --------------------------------
z = torch.zeros(N_samples, 64, device=device, requires_grad=True)

# ----------------------------------------
# 7. Joint training across all spheres
# ----------------------------------------
optimizer = torch.optim.Adam(list(decoder.parameters()) + [z], lr=1e-3)
epochs = 1000
print("\n==== Training: different-size spheres, same deformation ====")
for epoch in range(1, epochs + 1):
    optimizer.zero_grad()
    loss_total = 0.0

    comp_batch = decoder(z)  # (N_samples, M, 3)
    for i in range(N_samples):
        comp_i  = comp_batch[i]
        built_i = comp_i + deform(comp_i)
        loss_i  = chamfer(built_i, design_ts[i])
        loss_total += loss_i

    loss = loss_total / N_samples
    loss.backward()
    optimizer.step()

    if epoch % 200 == 0 or epoch == 1:
        print(f" Epoch {epoch:4d} | Avg Chamfer Loss = {loss.item():.4e}")

# ------------------------------------------------
# 8. Evaluate and prepare arrays for visualization
# ------------------------------------------------
with torch.no_grad():
    comp_batch = decoder(z)  # (N_samples, M, 3)
    raws = [ (design_ts[i] + deform(design_ts[i])).cpu().numpy()
             for i in range(N_samples) ]
    comps = [ comp_batch[i].cpu().numpy() for i in range(N_samples) ]
    builts = [ comps[i] + deform(torch.as_tensor(comps[i], device=device)).cpu().numpy()
               for i in range(N_samples) ]

# --------------------------------------
# 9. Visualization: comparison per sample
# --------------------------------------
max_r = max(radii)
lim = max_r * 1.2
ticks = [-max_r, 0.0, max_r]

for i in range(N_samples):
    D = design_list[i]
    R = raws[i]
    C = comps[i]
    B = builts[i]

    fig, axs = plt.subplots(1, 3, figsize=(15, 5), subplot_kw={'projection': '3d'})
    for ax, pts, title, cmap in zip(
        axs,
        [D, R, B],
        ['Design Sphere', 'Raw Deformed', 'Compensated Built'],
        ['Greys',        'viridis',      'inferno']
    ):
        sc = ax.scatter(pts[:,0], pts[:,1], pts[:,2],
                        c=pts[:,2], cmap=cmap, s=10, alpha=0.9, edgecolors='k', linewidths=0.1)
        ax.set_title(f"{title} (r={radii[i]})", fontsize=14)
        # fixed, shared axis limits so sphere sizes appear different
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(-lim, lim)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_zticks(ticks)
        ax.set_box_aspect([1,1,1])
        ax.view_init(elev=20, azim=35)
        ax.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"sphere_{i+1}_comparison.png", dpi=300)
    plt.close(fig)

print("\nSaved visualizations: sphere_i_comparison.png for i=1..{N_samples}")
