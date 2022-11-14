import numpy as np
import torch
from torch import autograd as autograd


class RBF(torch.nn.Module):
    def __init__(self, sigma=None):
        super(RBF, self).__init__()

        self.sigma = sigma

    def forward(self, X, Y):
        XX = X.matmul(X.t())
        XY = X.matmul(Y.t())
        YY = Y.matmul(Y.t())

        dnorm2 = -2 * XY + XX.diag().unsqueeze(1) + YY.diag().unsqueeze(0)

        # Apply the median heuristic (PyTorch does not give true median)
        if self.sigma is None:
            np_dnorm2 = dnorm2.detach().cpu().numpy()
            h = np.median(np_dnorm2) / (2 * np.log(X.size(0) + 1))
            sigma = np.sqrt(h).item()
        else:
            sigma = self.sigma

        gamma = 1.0 / (1e-8 + 2 * sigma ** 2)
        K_XY = (-gamma * dnorm2).exp()

        return K_XY


class SVGD:
    def __init__(self, P, K, optimizer):
        self.P = P
        self.K = K
        self.optim = optimizer

    def phi(self, X):
        X = X.detach().requires_grad_(True)

        self.log_prob = self.P.log_prob(X)
        score_func = autograd.grad(self.log_prob.sum(), X)[0]

        K_XX = self.K(X, X.detach())
        grad_K = -autograd.grad(K_XX.sum(), X)[0]

        phi = (K_XX.detach().matmul(score_func) + grad_K) / X.size(0)

        return phi

    def step(self, X):
        self.optim.zero_grad()
        p = -self.phi(X)
        X.grad = p
        self.optim.step()
        return self.log_prob.detach().clone()
