import os

from torch import optim

from stucco import cfg
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import math

from stucco.icp.volumetric import plot_restart_losses
from stucco.svgd import RBF, SVGD


def f(x, y):
    # return np.sin(np.sqrt(x ** 2 + y ** 2)) / np.abs(x * y)
    return (1 - torch.exp(-(x ** 2 + y ** 2)) * torch.abs(x + y)) + (x ** 2 + y ** 2) * 0.1


x = np.linspace(-3, 3, 30)
y = np.linspace(-3, 3, 30)
X, Y = np.meshgrid(x, y)

device = "cuda"
X = torch.tensor(X, device=device)
Y = torch.tensor(Y, device=device)


class CostProb:
    def __init__(self, cost, scale=1.):
        self.cost = cost
        self.scale = scale

    def log_prob(self, XY):
        X = XY[..., 0]
        Y = XY[..., 1]
        c = self.cost(X, Y)
        # p = N exp(-c * self.scale)
        # logp \propto -c * self.scale
        return -c * self.scale

    def prob(self, XY):
        return torch.exp(self.log_prob(XY))


P = CostProb(f, scale=3)

# decide what to plot; either the function value itself or the conversion to a probability distribution
Z = f(X, Y)
Z = P.prob(torch.stack((X, Y), dim=-1))

fig = plt.figure(figsize=(10, 8))
ax = plt.axes(projection='3d')
ax.plot_surface(X.cpu(), Y.cpu(), Z.cpu(), rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.view_init(elev=90, azim=90)
plt.tight_layout()
# plt.show()
# exit()

# generate random initial particles
B = 100
K = RBF()
particles = (torch.rand(B, 2, device=device) - 0.5) * 5
# particles = (torch.rand(B, 2, device=device) - 0.5) * 2.0 + 1
vis_particles = None

use_lbfgs = True
if use_lbfgs:
    optimizer = optim.LBFGS([particles], lr=1e-1)
else:
    optimizer = optim.Adam([particles], lr=1e-1)

svgd = SVGD(P, K, optimizer)
max_iterations = 100
losses = []

for i in range(max_iterations):
    with torch.no_grad():
        if vis_particles is not None:
            vis_particles.remove()
        z = P.prob(particles)
        vis_particles = ax.scatter(particles[:, 0].cpu(), particles[:, 1].cpu(), z.cpu() + 3e-2, color=(1, 0, 0))


    def closure():
        svgd.optim.zero_grad()
        p = -svgd.phi(particles)
        particles.grad = p
        logprob = svgd.log_prob.detach().clone()
        cost = -logprob / P.scale
        return cost.mean()


    svgd.optim.step(closure)
    cost = -svgd.log_prob.detach().clone() / P.scale

    # logprob = svgd.step(particles)
    # cost = -logprob / P.scale

    losses.append(cost)
    plt.savefig(os.path.join(cfg.DATA_DIR, 'img/svgd', f"{i}.png"))

plot_restart_losses(losses)
