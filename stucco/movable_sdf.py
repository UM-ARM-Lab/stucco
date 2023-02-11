import torch
from arm_pytorch_utilities import tensor_utils


class PlanarMovableSDF:
    """Measure distance from a point to the robot surface for the robot in some configuration"""

    def __init__(self, d_cache, min_x, min_y, max_x, max_y, cache_resolution, cache_y_len):
        self.d_cache = d_cache
        self.min_x, self.min_y, self.max_x, self.max_y = min_x, min_y, max_x, max_y
        self.cache_y_len = cache_y_len
        self.cache_resolution = cache_resolution
        # for debugging, view a = self.d_cache.reshape(len(x), len(y))

    def __call__(self, configs, pts):
        if not torch.is_tensor(self.d_cache):
            self.d_cache = tensor_utils.ensure_tensor(pts.device, pts.dtype, self.d_cache)

        M = configs.shape[0]
        N = pts.shape[0]
        # take advantage of the fact our system has no rotation to just translate points; otherwise would need full tsf
        query_pts = pts.repeat(M, 1, 1).transpose(0, 1)
        # needed transpose to allow broadcasting
        query_pts -= configs
        # flatten
        query_pts = query_pts.transpose(0, 1).view(M * N, -1)
        d = torch.ones(M * N, dtype=pts.dtype, device=pts.device)
        # eliminate points outside bounding box
        valid = (self.min_x <= query_pts[:, 0]) & (query_pts[:, 0] <= self.max_x) & \
                (self.min_y <= query_pts[:, 1]) & (query_pts[:, 1] <= self.max_y)
        # convert coordinates to indices
        query = query_pts[valid]
        x_id = torch.round((query[:, 0] - self.min_x) / self.cache_resolution).to(dtype=torch.long)
        y_id = torch.round((query[:, 1] - self.min_y) / self.cache_resolution).to(dtype=torch.long)
        idx = x_id * self.cache_y_len + y_id
        d[valid] = self.d_cache[idx]

        return d.view(M, N)


