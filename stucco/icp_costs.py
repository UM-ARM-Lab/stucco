import torch
from torch.nn import MSELoss
from pytorch3d.ops.knn import knn_gather, _KNN


class ICPPoseCost:
    def __call__(self, knn_res:_KNN, R, T, s):
        """Cost for given pose guesses of ICP

        :param R: B x 3 x 3
        :param T: B x 3
        :param s: B
        :return: scalar that is the mean of the costs
        """
        return 0


class ComposeCost(ICPPoseCost):
    def __init__(self, *args):
        self.costs = args

    def __call__(self, *args, **kwargs):
        return sum(cost(*args, **kwargs) for cost in self.costs)


class SurfaceNormalCost(ICPPoseCost):
    def __init__(self, Xnorm, Ynorm, scale=1.):
        self.Xnorm = Xnorm
        self.Ynorm = Ynorm
        self.loss = MSELoss()
        self.scale = scale

    def __call__(self, knn_res: _KNN, R, T, s):
        corresponding_ynorm = knn_gather(self.Ynorm, knn_res.idx).squeeze(-2)
        transformed_norms = s[:, None, None] * torch.bmm(self.Xnorm, R)
        return self.loss(corresponding_ynorm, transformed_norms) * self.scale
