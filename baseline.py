import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib import animation

# ----------------------------------------
# 1. Device setup & reproducibility
# ----------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(0)
np.random.seed(0)

# ----------------------------------------
# 2. Generate spherical design point cloud
# ----------------------------------------
N_theta, N_phi = 30, 60
theta = np.linspace(0, np.pi, N_theta)
phi   = np.linspace(0, 2 * np.pi, N_phi)
θ, φ  = np.meshgrid(theta, phi)

r = 1.0  # sphere radius
x = r * np.sin(θ) * np.cos(φ)
y = r * np.sin(θ) * np.sin(φ)
z = r * np.cos(θ)

design = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)  # (M,3)
design_t = torch.as_tensor(design, dtype=torch.float32, device=device)
M = design.shape[0]

# ------------------------------------------------
# 3. Realistic non-linear deformation function
# ------------------------------------------------
def deform(p):
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    # directional distortions (simulate forces)
    dx = 0.10 * torch.sin(2 * torch.pi * y) + 0.05 * x * z
    dy = 0.10 * torch.cos(2 * torch.pi * x) + 0.05 * y * z
    dz = 0.12 * torch.sin(x**2 + y**2) \
       + 0.05 * torch.sin(6 * torch.pi * z) * torch.cos(3 * torch.pi * y)
    # manufacturing noise
    dz = dz + 0.02 * torch.randn_like(z)
    return torch.stack([dx, dy, dz], dim=1).detach()

# ------------------------------------------------
# 4. PointDecoder: latent z → (M,3) point cloud
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
        return self.net(z).view(self.out_points, 3)

decoder = PointDecoder().to(device)

# -----------------------------------------------
# 5. Chamfer distance for point cloud alignment
# -----------------------------------------------
def chamfer(a, b):
    d = torch.cdist(a, b)
    return d.min(1).values.mean() + d.min(0).values.mean()

λ = 0.0  # MSE weight (keep zero to focus on Chamfer)

# -----------------------------------------------
# 6. Initialize latent vector z
# -----------------------------------------------
z = torch.zeros(1, 64, device=device, requires_grad=True)

# -------------------------------------------------
# 7. Warm-up: joint train decoder & z with MSE loss
# -------------------------------------------------
warmup_steps = 800
warmup_lr    = 3e-3
opt_all = torch.optim.Adam(list(decoder.parameters()) + [z], lr=warmup_lr)
print("==== Warm-up Stage ====")
for step in range(warmup_steps):
    comp  = decoder(z)
    built = comp + deform(comp)
    loss  = torch.mean((built - design_t)**2)
    opt_all.zero_grad()
    loss.backward()
    opt_all.step()
    if step % 200 == 0:
        print(f"  Warm-up step {step:4d} | MSE = {loss.item():.4e}")

# ----------------------------------------
# 8. Freeze decoder parameters
# ----------------------------------------
for p in decoder.parameters():
    p.requires_grad_(False)

# -------------------------------------------------
# 9. Boost rounds: optimize z (and last layer)
# -------------------------------------------------
boost_rounds = 4
K_steps      = 600
boost_lr     = 1e-3
unfreeze_last = 2
decoder_params = list(decoder.parameters())

print("\n==== Boost Rounds ====")
for r in range(1, boost_rounds + 1):
    if r > boost_rounds - unfreeze_last:
        # unfreeze last linear weight & bias
        for p in decoder_params:
            p.requires_grad_(False)
        decoder_params[-2].requires_grad_(True)
        decoder_params[-1].requires_grad_(True)
        print(f"  Round {r}: unfreeze last layer")
        opt = torch.optim.Adam([p for p in decoder.parameters() if p.requires_grad] + [z], lr=boost_lr)
    else:
        print(f"  Round {r}: optimize latent z only")
        opt = torch.optim.Adam([z], lr=boost_lr)

    for _ in range(K_steps):
        comp  = decoder(z)
        built = comp + deform(comp)
        loss  = chamfer(built, design_t) + λ * torch.mean((built - design_t)**2)
        opt.zero_grad()
        loss.backward()
        opt.step()
    print(f"  Round {r:2d} | Loss = {loss.item():.4e}")

# ------------------------------------------------
# 10. Final inference & metrics
# ------------------------------------------------
with torch.no_grad():
    comp_final  = decoder(z)
    built_final = comp_final + deform(comp_final)

final_chamfer = chamfer(built_final, design_t).item()
final_mse     = torch.mean((built_final - design_t)**2).item()
print(f"\nFinal Chamfer: {final_chamfer:.4f}")
print(f"Final MSE    : {final_mse:.4f}")

# ------------------------------------------------
# 11. Prepare data for visualization
# ------------------------------------------------
design_np    = design                                # original
raw_np       = (design_t + deform(design_t)).cpu().numpy()   # raw deformation
comp_np      = comp_final.cpu().numpy()              # compensation only
built_np     = built_final.cpu().numpy()             # final built
error_np     = np.linalg.norm(built_np - design_np, axis=1)  # error magnitude

# ------------------------------------------------
# 12. Static comparison: 1×4 subplots
# ------------------------------------------------
fig, axes = plt.subplots(1, 4, figsize=(20, 5), subplot_kw={'projection': '3d'})
sc_configs = [
    (design_np, "Original Design",      'Greys'),
    (raw_np,    "Raw Deformation",      'viridis'),
    (comp_np,   "Decoder Compensation", 'plasma'),
    (built_np,  "Final Built Shape",    'inferno'),
]
for ax, (pts, title, cmap) in zip(axes, sc_configs):
    sc = ax.scatter(pts[:,0], pts[:,1], pts[:,2],
                    c=pts[:,2], cmap=cmap, s=12, alpha=0.9, edgecolors='k', linewidths=0.1)
    ax.set_title(title, pad=10, fontsize=14)
    ax.set_box_aspect([1,1,1])
    ax.view_init(elev=20, azim=35)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
plt.tight_layout()
plt.savefig("comparison.png", dpi=300)
plt.close(fig)

# ------------------------------------------------
# 13. Error magnitude heatmap
# ------------------------------------------------
fig_err = plt.figure(figsize=(6,6))
ax_err = fig_err.add_subplot(111, projection='3d')
sc_e = ax_err.scatter(built_np[:,0], built_np[:,1], built_np[:,2],
                      c=error_np, cmap='magma', s=12, alpha=0.9, edgecolors='k', linewidths=0.1)
ax_err.set_title("Error Magnitude", fontsize=16, pad=10)
ax_err.set_box_aspect([1,1,1])
ax_err.view_init(elev=20, azim=35)
ax_err.set_xticks([]); ax_err.set_yticks([]); ax_err.set_zticks([])
cbar = fig_err.colorbar(sc_e, shrink=0.5, pad=0.1)
cbar.set_label("Error", fontsize=12)
plt.tight_layout()
plt.savefig("error.png", dpi=300)
plt.close(fig_err)

# ------------------------------------------------
# 14. Animated rotation of final built shape
# ------------------------------------------------
fig_rot = plt.figure(figsize=(6,6))
ax_rot = fig_rot.add_subplot(111, projection='3d')
sc_rot = ax_rot.scatter(built_np[:,0], built_np[:,1], built_np[:,2],
                        c=built_np[:,2], cmap='inferno', s=12, alpha=0.9, edgecolors='k', linewidths=0.1)
ax_rot.set_box_aspect([1,1,1])
ax_rot.set_xticks([]); ax_rot.set_yticks([]); ax_rot.set_zticks([])

def update(angle):
    ax_rot.view_init(elev=20, azim=angle)
    return sc_rot,

ani = animation.FuncAnimation(fig_rot, update, frames=np.arange(0, 360, 5),
                              interval=50, blit=True)
ani.save("rotation.gif", writer='pillow', fps=20)
plt.close(fig_rot)

print("Saved files: comparison.png, error.png, rotation.gif")
