from chsel_experiments.env import poke_real_nonros
from pytorch_volumetric import voxel
import numpy as np
import matplotlib.colors
from matplotlib import pyplot as plt
from stucco.icp.costs import KnownSDFLookupCost, FreeSpaceLookupCost, OccupiedLookupCost
import torch

plt.switch_backend('Qt5Agg')

task = poke_real_nonros.Levels.DRILL
env = poke_real_nonros.PokeRealNoRosEnv(task, device="cpu", clean_cache=True)

s = env.target_sdf
query_range = np.copy(s.ranges)
# middle y slice
query_range[1] = [0, 0]
coords, pts = voxel.get_coordinates_and_points_in_grid(s.resolution, query_range, device=s.device)
# if filter_pts is not None:
#     pts = filter_pts(pts)
sdf_val, sdf_grad = s(pts)

# color code them
norm = matplotlib.colors.Normalize(vmin=sdf_val.min().cpu() - 0.2, vmax=sdf_val.max().cpu())
color_map = matplotlib.cm.ScalarMappable(norm=norm)
rgb = color_map.to_rgba(sdf_val.reshape(-1).cpu())
rgb = rgb[:, :-1]

plt.ion()
plt.show()


def fmt(x):
    s = f"{x:.1f}"
    if s.endswith("0"):
        s = f"{x:.0f}"
    if x == 0:
        return "surface"
    return rf"{s}" if plt.rcParams["text.usetex"] else f"{s}"


ax = plt.gca()
ax.set_xlabel('x')
ax.set_ylabel('z')
x = coords[0]
z = coords[2]
v = sdf_val.reshape(len(x), len(z)).transpose(0, 1)
cset1 = ax.contourf(x, z, v, norm=norm, cmap='Greys_r')
cset2 = ax.contour(x, z, v, colors='k', levels=[0], linestyles='dashed')
ax.clabel(cset2, cset2.levels, inline=True, fontsize=13, fmt=fmt)

# choose some known free points
c = 'r'
plot_grad_scale = 1
i = [915, 1574, 2005]
kpts = pts[i]
kpts.requires_grad = True
kpts.retain_grad = True
cc = FreeSpaceLookupCost.apply(s, kpts.unsqueeze(0))
cc.sum().backward()
xx = kpts[:, 0].detach()
zz = kpts[:, 2].detach()
ox = 0.007
oz = -0.01
ax.scatter(xx, zz, color=c)
dx = kpts.grad[:, 0] * plot_grad_scale
dz = kpts.grad[:, 2] * plot_grad_scale
ax.quiver(xx, zz, -dx, -dz, color=c, minlength=1, minshaft=1, scale=0.05)
ax.text(xx[1] + ox, zz[1] + oz, r"$-\nabla \hat{c}_f(\tilde{\mathbf{x}})$", color=c, fontsize=16)

# choose some known surface points (should they be not just to the surface but to other SDF values?)
# c = 'deepskyblue'
c = 'black'
plot_grad_scale = 1
i = [550, 1665, 2005]
kpts = pts[i]
kpts.requires_grad = True
kpts.retain_grad = True
cc = KnownSDFLookupCost.apply(s, kpts.unsqueeze(0), torch.zeros(len(i)).unsqueeze(0))
cc.sum().backward()
xx = kpts[:, 0].detach()
zz = kpts[:, 2].detach()
ox = 0.061
oz = 0.015
ax.scatter(xx.detach(), zz.detach(), color=c)
dx = kpts.grad[:, 0] * plot_grad_scale
dz = kpts.grad[:, 2] * plot_grad_scale
ax.quiver(xx.detach(), zz.detach(), -dx, -dz, color=c, minlength=1, minshaft=1, scale=0.25)
ax.text(xx[0] + ox, zz[0] + oz, r"$-\nabla \hat{c}_k(\tilde{\mathbf{x}}, 0)$", color=c, fontsize=16)

# occupied
c = 'indigo'
plot_grad_scale = 1
i = [558, 1695, 2205]
kpts = pts[i]
kpts.requires_grad = True
kpts.retain_grad = True
cc = OccupiedLookupCost.apply(s, kpts.unsqueeze(0))
cc.sum().backward()
xx = kpts[:, 0].detach()
zz = kpts[:, 2].detach()
ox = -0.009
oz = -0.032
ax.scatter(xx.detach(), zz.detach(), color=c)
dx = kpts.grad[:, 0] * plot_grad_scale
dz = kpts.grad[:, 2] * plot_grad_scale
ax.quiver(xx.detach(), zz.detach(), -dx, -dz, color=c, minlength=1, minshaft=1, scale=0.25)
ax.text(xx[0] + ox, zz[0] + oz, r"$-\nabla \hat{c}_o(\tilde{\mathbf{x}})$", color=c, fontsize=16)
