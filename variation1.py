import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib import animation

# ----------------------------------------
# 1. Device & reproducibility
# ----------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(0)
np.random.seed(0)

# ----------------------------------------
# 2. Dense sphere design point cloud
# ----------------------------------------
N_theta, N_phi = 50, 100
theta = np.linspace(0, np.pi, N_theta)
phi   = np.linspace(0, 2 * np.pi, N_phi)
θ, φ  = np.meshgrid(theta, phi)

r = 1.0
x = r * np.sin(θ) * np.cos(φ)
y = r * np.sin(θ) * np.sin(φ)
z = r * np.cos(θ)

design = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)  # (M,3)
design_t = torch.as_tensor(design, dtype=torch.float32, device=device)
M = design.shape[0]
print(f"Design point cloud has {M} points")

# ------------------------------------------------------
# 3. Define multiple deformation parameter sets (5 samples)
# ------------------------------------------------------
deform_params = [
    {'bend_amp':0.15,'bend_freq':3.0,'twist_amp':0.08,'sag':0.08,'ripple_amp':0.03,'ripple_freq':30.0,'noise':0.01},
    {'bend_amp':0.20,'bend_freq':2.5,'twist_amp':0.05,'sag':0.12,'ripple_amp':0.04,'ripple_freq':25.0,'noise':0.008},
    {'bend_amp':0.10,'bend_freq':4.0,'twist_amp':0.10,'sag':0.05,'ripple_amp':0.05,'ripple_freq':40.0,'noise':0.012},
    {'bend_amp':0.18,'bend_freq':3.5,'twist_amp':0.12,'sag':0.10,'ripple_amp':0.02,'ripple_freq':20.0,'noise':0.009},
    {'bend_amp':0.12,'bend_freq':2.0,'twist_amp':0.06,'sag':0.15,'ripple_amp':0.06,'ripple_freq':35.0,'noise':0.015},
]
N_samples = len(deform_params)

def make_deform_fn(p):
    """Return a deform(p) using parameter dict p."""
    def deform(pt):
        x, y, z = pt[:,0], pt[:,1], pt[:,2]
        # large-scale bending
        dx = p['bend_amp'] * torch.sin(p['bend_freq'] * torch.pi * y)
        dy = p['bend_amp'] * torch.cos(p['bend_freq'] * torch.pi * x)
        # twist around Z-axis
        dx = dx +  p['twist_amp'] * y
        dy = dy -  p['twist_amp'] * x
        # sagging under gravity
        dz = -p['sag'] * (x**2 + y**2)
        # fine ripple texture
        dz = dz + p['ripple_amp'] * torch.sin(p['ripple_freq'] * x) * torch.cos(p['ripple_freq'] * y)
        # random noise
        dz = dz + p['noise'] * torch.randn_like(z)
        return torch.stack([dx, dy, dz], dim=1).detach()
    return deform

deform_funcs = [make_deform_fn(p) for p in deform_params]

# ------------------------------------------------
# 4. Shared PointDecoder: latent z → (M,3)
# ------------------------------------------------
class PointDecoder(nn.Module):
    def __init__(self, z_dim=128, out_points=M):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, 512), nn.ReLU(),
            nn.Linear(512, 512),   nn.ReLU(),
            nn.Linear(512, 512),   nn.ReLU(),
            nn.Linear(512, out_points * 3)
        )
        self.out_points = out_points

    def forward(self, z):
        pts = self.net(z)             # (1, M*3)
        return pts.view(self.out_points, 3)  # (M,3)

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
z = torch.zeros(N_samples, 128, device=device, requires_grad=True)

# ----------------------------------------
# 7. Joint training: all samples
# ----------------------------------------
optimizer = torch.optim.Adam(list(decoder.parameters()) + [z], lr=5e-4)
epochs = 1500
print("\n==== Multi-sample Training ====")
for epoch in range(1, epochs + 1):
    optimizer.zero_grad()
    total_loss = 0.0
    for i, deform_fn in enumerate(deform_funcs):
        comp_i  = decoder(z[i].unsqueeze(0))       # (M,3)
        built_i = comp_i + deform_fn(comp_i)       # (M,3)
        loss_i  = chamfer(built_i, design_t)
        total_loss += loss_i
    avg_loss = total_loss / N_samples
    avg_loss.backward()
    optimizer.step()

    if epoch % 300 == 0 or epoch == 1:
        print(f" Epoch {epoch:4d} | Avg Chamfer = {avg_loss.item():.4e}")

# -------------------------------
# 8. Evaluate & prepare arrays
# -------------------------------
with torch.no_grad():
    comps  = [decoder(z[i].unsqueeze(0)) for i in range(N_samples)]
    raws   = [design_t + deform_funcs[i](design_t) for i in range(N_samples)]
    builts = [comps[i] + deform_funcs[i](comps[i]) for i in range(N_samples)]
    errors = [torch.norm(builts[i] - design_t, dim=1).cpu().numpy() for i in range(N_samples)]

# --------------------------------------
# 9. Visualization: per-sample grid
# --------------------------------------
for i in range(N_samples):
    D = design
    R = raws[i].cpu().numpy()
    C = comps[i].cpu().numpy()
    B = builts[i].cpu().numpy()
    E = errors[i]

    fig, axs = plt.subplots(1, 4, figsize=(20, 5), subplot_kw={'projection': '3d'})
    cfg = [
        (D, 'Design',     'Greys'),
        (R, f'Raw #{i+1}', 'viridis'),
        (B, f'Built #{i+1}','plasma'),
        (B, 'Error map',   'magma', E)
    ]
    for ax, (pts, title, cmap, *err) in zip(axs, cfg):
        c = err[0] if err else pts[:,2]
        sc = ax.scatter(pts[:,0], pts[:,1], pts[:,2],
                        c=c, cmap=cmap, s=6, alpha=0.9,
                        edgecolors='none')
        ax.set_title(title, fontsize=14)
        ax.set_box_aspect([1,1,1])
        ax.view_init(elev=20, azim=35)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        if title == 'Error map':
            fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.1).set_label('Error')
    plt.tight_layout()
    plt.savefig(f"sample_{i+1}_detailed.png", dpi=300)
    plt.close(fig)

# --------------------------------------
# 10. Animated rotation GIF for sample 1
# --------------------------------------
fig_gif = plt.figure(figsize=(6,6))
ax_gif  = fig_gif.add_subplot(111, projection='3d')
pts = builts[0].cpu().numpy()
sc = ax_gif.scatter(pts[:,0], pts[:,1], pts[:,2],
                    c=pts[:,2], cmap='inferno', s=6, alpha=0.9, edgecolors='none')
ax_gif.set_box_aspect([1,1,1])
ax_gif.set_xticks([]); ax_gif.set_yticks([]); ax_gif.set_zticks([])

def update(angle):
    ax_gif.view_init(elev=20, azim=angle)
    return sc,

ani = animation.FuncAnimation(fig_gif, update, frames=np.arange(0,360,4),
                              interval=50, blit=True)
ani.save("sample1_rotation.gif", writer='pillow', fps=20)
plt.close(fig_gif)

print("Saved detailed visualizations: sample_i_detailed.png for i=1..5 and sample1_rotation.gif")
