import enum
import os

from torch import optim

from stucco import cfg
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import math
from arm_pytorch_utilities.rand import seed

from stucco.icp.volumetric import plot_restart_losses
from stucco.svgd import RBF, SVGD
import cma


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

seed(0)
# generate random initial particles
B = 100
K = RBF()
particles = (torch.rand(B, 2, device=device) - 0.5) * 5
# particles = (torch.rand(B, 2, device=device) - 0.5) * 0.1 + 1
vis_particles = None


def plot_particles(i):
    global vis_particles
    with torch.no_grad():
        if vis_particles is not None:
            vis_particles.remove()
        z = P.prob(particles)
        vis_particles = ax.scatter(particles[:, 0].cpu(), particles[:, 1].cpu(), z.cpu() + 3e-2, color=(1, 0, 0))
        ax.set_xlim([min(particles[:, 0].min().cpu(), -3), max(particles[:, 0].max().cpu(), 3)])
        ax.set_ylim([min(particles[:, 1].min().cpu(), -3), max(particles[:, 1].max().cpu(), 3)])
    img_path = os.path.join(cfg.DATA_DIR, 'img/svgd', f"{i}.png")
    plt.savefig(img_path)
    print(f"particles saved to {img_path}")


class OptimizationMethod(enum.Enum):
    SVGD = 0
    CMAES = 1


method = OptimizationMethod.CMAES
losses = []

if method is OptimizationMethod.SVGD:
    use_lbfgs = True
    if use_lbfgs:
        optimizer = optim.LBFGS([particles], lr=1e-1)
    else:
        optimizer = optim.Adam([particles], lr=1e-1)

    svgd = SVGD(P, K, optimizer)
    max_iterations = 100

    for i in range(max_iterations):
        plot_particles(i)


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

elif method is OptimizationMethod.CMAES:
    x0 = particles[0].cpu().numpy()
    sigma = 1.0
    options = {"popsize": B, "seed": np.random.randint(0, 10000), "tolfun": 1e-5, "tolfunhist": 1e-6}
    es = cma.CMAEvolutionStrategy(x0=x0, sigma0=sigma, inopts=options)
    i = 0
    while not es.stop():
        solutions = es.ask()
        # convert back to R, T, s
        particles = torch.tensor(solutions, device=device)
        plot_particles(i)

        cost = f(particles[..., 0], particles[..., 1])
        losses.append(cost)
        es.tell(solutions, cost.cpu().numpy())
        i += 1

    # convert ES back to R, T
    solutions = es.ask()

plot_restart_losses(losses)