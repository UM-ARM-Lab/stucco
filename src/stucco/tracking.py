from dataclasses import dataclass
import torch
import logging
import abc
from arm_pytorch_utilities import tensor_utils, serialization

logger = logging.getLogger(__name__)


@dataclass
class ContactParameters:
    length: float = 0.1
    penetration_length: float = 0.01
    hard_assignment_threshold: float = 0.4  # for soft assignment, probability threshold for belonging to same component
    intersection_tolerance: float = 0.002  # how much intersection into the robot's surface we ignore


# resample function from http://scipy-cookbook.readthedocs.io/items/ParticleFilter.html
def resample(weights):
    n = len(weights)
    indices = []
    C = [0.0] + [torch.sum(weights[: i + 1]) for i in range(n)]
    u0, j = torch.rand((1,), device=weights.device), 0
    for u in [(u0 + i) / n for i in range(n)]:
        while j < len(C) and u > C[j]:
            j += 1
        indices.append(j - 1)
    return indices


class ContactSet(serialization.Serializable):
    def __init__(self, params: ContactParameters, immovable_collision_checker=None, device='cpu', dtype=torch.float32,
                 visualizer=None, dim=2):
        self.device = device
        self.dtype = dtype
        self.p = params
        self.immovable_collision_checker = immovable_collision_checker
        self.visualizer = visualizer
        self.dim = dim

    @abc.abstractmethod
    def update(self, x, dx, p=None, info=None):
        """Update contact set with observed transition

        :param x: current end effector pose
        :param dx: latest change in end effector pose while in contact
        :param p: current position of the contact point, or none if not in contact
        :param info: dictionary with optional keys from InfoKeys that provide extra debugging information
        """

    @abc.abstractmethod
    def get_batch_data_for_dynamics(self, total_num):
        """Return initial contact data needed for dynamics for total_num batch size"""

    @abc.abstractmethod
    def dynamics(self, x, u, contact_data):
        """Perform batch dynamics on x with control u using contact data from get_batch_data_for_dynamics

        :return next x, without contact mask, updated contact data"""


class LinearTranslationalDynamics:
    @staticmethod
    def _extract_linear_translation(dx):
        # for base, x is just position so dx is linear translation
        return dx

    def __call__(self, p, x, dx):
        dpos = LinearTranslationalDynamics._extract_linear_translation(dx)
        new_p = p + dpos
        new_x = x + dx
        return new_p, new_x


class ContactSetSoft(ContactSet):
    """
    Track contact points and contact configurations without hard object assignments (partitions of points that move
    together). Implemented using a particle filter with each particle being the set of all contact points and configs.
    """

    def __init__(self, pxpen, *args, pxdyn=LinearTranslationalDynamics(), n_particles=100, n_eff_threshold=0.8,
                 replace_bad_points=True, **kwargs):
        super(ContactSetSoft, self).__init__(*args, **kwargs)
        # key functions
        self.pxpen = pxpen
        self.pxdyn = pxdyn

        self.adjacency = None
        self.connection_prob = None
        self.n_particles = n_particles
        self.n_eff_threshold = n_eff_threshold
        self._do_replace_bad_points = replace_bad_points

        self.weights = torch.ones(self.n_particles, device=self.device) / self.n_particles
        self.weights_normalization = 1

        # each observation has a contact point (on robot surface) and contact config (pose of end effector)
        self.pts = None
        self.configs = None
        self.acts = None

        # [n_particles, N, D] where N is the number of data points, and D is data dimensionality
        self.sampled_pts = None
        self.sampled_configs = None
        self.map_particle = None

    def _compute_full_adjacency(self, pts):
        # don't typically need to compute full adjacency
        dd = torch.cdist(pts, pts)
        return self._distance_to_probability(dd)

    def get_posterior_points(self):
        return self.map_particle

    def _partition_points(self, adjacent):
        # holds a partition of indices 0..N-1
        groups = []
        N = adjacent.shape[0]
        all_selected = torch.zeros(N, device=adjacent.device, dtype=torch.bool)
        i = 0
        while not all_selected.all():
            if all_selected[i]:
                i += 1
                continue
            this_group = adjacent[i].clone()
            # TODO should use some connected component finding algo; for simplicity for now just look at 1 depth
            for j in range(N):
                if this_group[j]:
                    this_group |= adjacent[j]
            groups.append(this_group)
            all_selected |= this_group
        return groups

    def get_hard_assignment(self, threshold=None):
        pts = self.get_posterior_points()
        if pts is None:
            return []
        connection_prob = self._compute_full_adjacency(pts)

        if threshold is None:
            threshold = torch.rand_like(connection_prob)

        adjacent = threshold < connection_prob
        return self._partition_points(adjacent)

    def get_partitioned_points(self, threshold):
        pts = self.get_posterior_points()
        partitions = self.get_hard_assignment(threshold)
        return [pts[partition] for partition in partitions]

    def _distance_to_probability(self, distance, sigma=None):
        # parameter where higher means a greater drop off in probability with distance
        if sigma is None:
            sigma = 1 / self.p.length
        return torch.exp(-sigma * distance ** 2)

    def predict_particles(self, query_pts, dx):
        """Apply action to all particles"""
        # dd = (self.pts[-1] - self.sampled_pts).norm(dim=-1)
        dd = torch.cdist(query_pts, self.sampled_pts)

        # each point can at most be affected by 1 dx per contact
        # we compare its sampled probability against the most likely motion
        # take the minimum distance to any of our query points for probability
        dd, closest_dx_index = dd.min(dim=-2)

        # convert to probability
        connection_prob = self._distance_to_probability(dd)

        # sample particles which make hard assignments
        # independently sample uniform [0, 1) and compare against prob - note that connections are symmetric
        # sampled_prob[i,j] is the ith particle's probability of connection between the latest point and the jth point
        sampled_prob = torch.rand(connection_prob.shape, device=self.pts.device)
        adjacent = sampled_prob < connection_prob

        # don't actually need to label connected components because we just need to propagate for the latest
        # apply dx to each particle's cluster that contains the latest x
        self.sampled_pts[adjacent], self.sampled_configs[adjacent] = self.pxdyn(self.sampled_pts[adjacent],
                                                                                self.sampled_configs[adjacent],
                                                                                dx[closest_dx_index][adjacent])
        return adjacent

    def update_particles(self, config=None):
        """Update the weight of each particle corresponding to their ability to explain the observation"""
        if self.pts is None:
            return

        # all contact points should be outside the robot
        tol = self.p.intersection_tolerance
        # if given an explicit config to check against
        if config is not None:
            # for efficiency, just consider the given configuration (should probably consider all points, but may be enough)
            query_points = self.sampled_pts.view(-1, self.sampled_pts.shape[-1])
            d = self.pxpen(config.view(1, -1), query_points).view(self.n_particles, -1)
            # negative distance indicates penetration
            d += tol
            d[d > 0] = 0
            # collect sum then offset by max to prevent obs_weights from going to 0
            obs_weights = d.sum(dim=1)
        else:
            # otherwise check all configs against all points for each particle
            obs_weights = torch.zeros(self.n_particles, dtype=self.pts.dtype, device=self.pts.device)
            for i in range(self.n_particles):
                d = self.pxpen(self.sampled_configs[i], self.sampled_pts[i])
                d += tol
                d = d[d < 0]
                obs_weights[i] = d.sum()

        # prevent every particle going to 0
        obs_weights -= obs_weights.max()
        # convert to probability
        penetration_sigma = 1 / self.p.penetration_length
        obs_weights = self._distance_to_probability(obs_weights, sigma=penetration_sigma)

        min_weight = 1e-15
        self.weights = self.weights * obs_weights
        self.weights[self.weights < min_weight] = min_weight

        # normalize weights to resampling probabilities
        self.weights_normalization = self.weights.sum()
        self.weights = self.weights / self.weights_normalization

        # from pfilter
        # Compute effective sample size and entropy of weighting vector.
        # These are useful statistics for adaptive particle filtering.
        self.n_eff = (1.0 / (self.weights ** 2).sum()) / self.n_particles
        self.weight_entropy = torch.sum(self.weights * torch.log(self.weights))
        logger.debug(f"PF total weights: {self.weights_normalization.item()} n_eff {self.n_eff.item()}")

        # preserve current sample set before any replenishment
        self.original_particles = self.sampled_pts.clone(), self.sampled_configs.clone()

        # store MAP estimate
        argmax_weight = torch.argmax(self.weights)
        self.map_particle = self.sampled_pts[argmax_weight]

        # resampling (systematic resampling) step
        if self.n_eff < self.n_eff_threshold:
            indices = resample(self.weights)
            self.sampled_pts = self.sampled_pts[indices, :]
            self.sampled_configs = self.sampled_configs[indices, :]

            # if even after resampling we have bad points, replace those points with good points
            resampled_weights = self.weights[indices]
            resampled_n_eff = (1.0 / (resampled_weights ** 2).sum()) / self.n_particles
            if resampled_n_eff < self.n_eff_threshold and self._do_replace_bad_points:
                self.replace_bad_points()

            self.weights = torch.ones(self.n_particles, device=self.device) / self.n_particles

    def replace_bad_points(self):
        # for points that remain low probability, replace them with a non penetrating one
        for i in range(self.n_particles):
            d = self.pxpen(self.sampled_configs[i], self.sampled_pts[i])
            d += self.p.intersection_tolerance
            d[d > 0] = 0
            # each column is the distance of that point to each config
            bad_pts = d.sum(dim=0) < 0
            if not torch.any(bad_pts):
                continue
            good_pts = ~bad_pts
            # can't do anything if we have no good points...
            if not torch.any(good_pts):
                continue
            # find the distance of bad points wrt good points
            d_to_good = torch.cdist(self.sampled_pts[i, bad_pts], self.sampled_pts[i, good_pts])
            # replace the bad point with a copy of the closest good point
            closest_idx = torch.argmin(d_to_good, dim=1)
            self.sampled_pts[i, bad_pts] = self.sampled_pts[i, good_pts][closest_idx]
            self.sampled_configs[i, bad_pts] = self.sampled_configs[i, good_pts][closest_idx]

    def update(self, x, dx, p=None, info=None):
        d = self.device
        dtype = self.dtype
        x = tensor_utils.ensure_tensor(d, dtype, x)

        if info is not None:
            u = info['u']
        else:
            u = torch.zeros_like(x)

        cur_config = x
        if p is None or dx is None:
            # step without contact, eliminate particles that conflict with this config in freespace
            self.update_particles(cur_config)
            return None, None

        dx = tensor_utils.ensure_tensor(d, dtype, dx)
        # handle multiple number of points
        # where contact point would be without this movement
        if len(p.shape) < 2:
            p = p.reshape(1, -1)
        N = len(p)
        cur_pt = p[:, :self.dim]
        # 1 dx per sensor
        prev_pt, prev_config = self.pxdyn(cur_pt.view(N, -1), cur_config.view(1, -1), -dx)
        u = u.repeat(N, 1)
        # both should be (N, -1)

        if self.pts is None:
            self.pts = prev_pt
            self.configs = prev_config
            self.acts = u
            self.sampled_pts = self.pts.repeat(self.n_particles, 1, 1)
            self.sampled_configs = self.configs.repeat(self.n_particles, 1, 1)
        else:
            self.pts = torch.cat((self.pts, prev_pt))
            self.configs = torch.cat((self.configs, prev_config))
            self.acts = torch.cat((self.acts, u))
            self.sampled_pts = torch.cat([self.sampled_pts, prev_pt.repeat(self.n_particles, 1, 1)], dim=1)
            self.sampled_configs = torch.cat([self.sampled_configs, prev_config.repeat(self.n_particles, 1, 1)], dim=1)

        # classic alternation of predict and update steps
        self.predict_particles(prev_pt, dx)
        # check all configs against all points
        self.update_particles(None)

        return True, True

    def dynamics(self, x, u, contact_data):
        raise NotImplementedError()

    def get_batch_data_for_dynamics(self, total_num):
        raise NotImplementedError()
