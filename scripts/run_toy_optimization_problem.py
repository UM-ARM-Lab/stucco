import enum
import os

from torch import optim

from base_experiments import cfg
import matplotlib.pyplot as plt
import numpy as np
import torch
from arm_pytorch_utilities.rand import seed
from arm_pytorch_utilities import grad

from stucco.registration_util import plot_poke_losses
from stucco.svgd import RBF, SVGD
import cma
from ribs.archives import GridArchive
from ribs.emitters import EvolutionStrategyEmitter, GradientArborescenceEmitter
from ribs.schedulers import Scheduler


def f(x, y):
    # return np.sin(np.sqrt(x ** 2 + y ** 2)) / np.abs(x * y)
    return (1 - torch.exp(-(x ** 2 + y ** 2)) * torch.abs(x + y)) + (x ** 2 + y ** 2) * 0.1


def gradf(xy):
    def ff(x):
        return f(x[..., 0], x[..., 1]).view(-1, 1, 1)

    return grad.batch_jacobian(ff, xy)


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
    CMAME = 2
    CMA_MEGA = 3


method = OptimizationMethod.CMA_MEGA
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
        particles = torch.tensor(solutions, device=device)
        plot_particles(i)

        cost = f(particles[..., 0], particles[..., 1])
        losses.append(cost)
        es.tell(solutions, cost.cpu().numpy())
        i += 1

    # convert ES back to R, T
    solutions = es.ask()

elif method is OptimizationMethod.CMAME:
    # use x,y as the behavior space we want to optimize over (equal to the actual search space)
    bins = 50
    archive = GridArchive(solution_dim=2, dims=[bins, bins], ranges=[(-3, 3), (-3, 3)])
    initial_x = np.zeros(2)
    emitters = [
        EvolutionStrategyEmitter(archive, x0=initial_x, sigma0=1.0, batch_size=B) for _ in range(5)
    ]
    scheduler = Scheduler(archive, emitters)
    for i in range(100):
        solutions = scheduler.ask()
        # evaluate the models and record the objective and behavior
        # note that objective is -cost
        particles = torch.tensor(solutions, device=device)
        cost = f(particles[..., 0], particles[..., 1])
        losses.append(cost)

        # those particles are just random ones found during the search - what we want is a look at the best particles
        if i > 0:
            df = archive.as_pandas()
            o = df.batch_objectives()
            s = df.batch_solutions()
            if len(s) > B:
                order = np.argpartition(-o, B)
                s = s[order[:B]]
            particles = torch.tensor(s, device=device)
            plot_particles(i)

        # behavior is just the solution
        bcs = solutions
        scheduler.tell(-cost.cpu().numpy(), bcs)

    # from ribs.visualize import grid_archive_heatmap
    # plt.figure(figsize=(8,6))
    # grid_archive_heatmap(archive)

elif method is OptimizationMethod.CMA_MEGA:
    # use x,y as the behavior space we want to optimize over (equal to the actual search space)
    bins = 50
    archive = GridArchive(solution_dim=2, dims=[bins, bins], ranges=[(-3, 3), (-3, 3)])
    initial_x = np.zeros(2)
    emitters = [
        GradientArborescenceEmitter(archive, x0=initial_x, sigma0=1.0, lr=0.002, grad_opt="adam", selection_rule="mu",
                                    bounds=None, batch_size=B - 1)
    ]
    scheduler = Scheduler(archive, emitters)
    for i in range(100):
        # dqd part
        solutions = scheduler.ask_dqd()
        bcs = solutions
        # evaluate the models and record the objective and behavior
        # note that objective is -cost
        # get objective gradient and also the behavior gradient
        particles = torch.tensor(solutions, device=device)
        cost = f(particles[..., 0], particles[..., 1])
        objective = -cost.cpu().numpy()
        objective_grad = -gradf(particles)

        behavior_grad = np.tile(np.eye(2), (particles.shape[0], 1, 1))

        jacobian = np.concatenate((objective_grad.cpu().numpy(), behavior_grad), axis=1)
        scheduler.tell_dqd(objective, bcs, jacobian)

        # regular part
        solutions = scheduler.ask()
        particles = torch.tensor(solutions, device=device)
        cost = f(particles[..., 0], particles[..., 1])
        losses.append(cost)

        # those particles are just random ones found during the search - what we want is a look at the best particles
        if i > 0:
            df = archive.as_pandas()
            o = df.objective_batch()
            s = df.solution_batch()
            if len(s) > B:
                order = np.argpartition(-o, B)
                s = s[order[:B]]
            particles = torch.tensor(s, device=device)
            plot_particles(i)

        # behavior is just the solution
        bcs = solutions
        scheduler.tell(-cost.cpu().numpy(), bcs)

    # from ribs.visualize import grid_archive_heatmap
    # plt.figure(figsize=(8,6))
    # grid_archive_heatmap(archive)

plot_poke_losses(losses)
