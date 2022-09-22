import abc

import matplotlib
import torch
from multidim_indexing import torch_view

import pytorch_kinematics


def matrix_to_pos_rot(m):
    pos = m[:3, 3]
    rot = pytorch_kinematics.matrix_to_quaternion(m[:3, :3])
    rot = pytorch_kinematics.transforms.wxyz_to_xyzw(rot)
    return pos, rot


def move_figure(f, x, y):
    """Move figure's upper left corner to pixel (x, y)"""
    backend = matplotlib.get_backend()
    if backend == 'TkAgg':
        f.canvas.manager.window.wm_geometry("+%d+%d" % (x, y))
    elif backend == 'WXAgg':
        f.canvas.manager.window.SetPosition((x, y))
    else:
        # This works for QT and GTK
        # You can also use window.setGeometry
        f.canvas.manager.window.move(x, y)


def get_divisible_range_by_resolution(resolution, range_per_dim):
    # ensure value range divides resolution evenly
    temp_range = []
    for low, high in range_per_dim:
        span = high - low
        span = round(span / resolution)
        temp_range.append((low, low + span * resolution))
    return temp_range


def get_coordinates_and_points_in_grid(resolution, range_per_dim, dtype=torch.float, device='cpu', get_points=True):
    # create points along the value ranges
    coords = [torch.arange(low, high + resolution, resolution, dtype=dtype, device=device) for low, high in
              range_per_dim]
    pts = torch.cartesian_prod(*coords) if get_points else None
    return coords, pts


class Voxels(abc.ABC):
    @abc.abstractmethod
    def get_known_pos_and_values(self):
        """Return the position (N x 3) and values (N) of known voxels"""


class VoxelGrid(Voxels):
    def __init__(self, resolution, range_per_dim, dtype=torch.float, device='cpu'):
        self.resolution = resolution
        self.invalid_val = 0

        self.range_per_dim = get_divisible_range_by_resolution(resolution, range_per_dim)
        self.coords, self.pts = get_coordinates_and_points_in_grid(resolution, range_per_dim, dtype=dtype,
                                                                   device=device)
        # underlying data
        self._data = torch.zeros([len(coord) for coord in self.coords], dtype=dtype, device=device)
        self.voxels = torch_view.TorchMultidimView(self._data, range_per_dim, invalid_value=self.invalid_val)

    def get_known_pos_and_values(self):
        known = self.voxels.raw_data != self.invalid_val
        indices = known.nonzero()
        # these points are in object frame
        pos = self.voxels.ensure_value_key(indices)
        val = self.voxels.raw_data[indices]
        return pos, val

    def get_voxel_center_points(self):
        return self.pts

    def __getitem__(self, pts):
        return self.voxels[pts]

    def __setitem__(self, pts, value):
        self.voxels[pts] = value


class VoxelSet(Voxels):
    def __init__(self, positions, values):
        self.positions = positions
        self.values = values

    def __setitem__(self, pts, value):
        self.positions = torch.cat((self.positions, pts.view(-1, self.positions.shape[-1])), dim=0)
        self.values = torch.cat((self.values, value))

    def get_known_pos_and_values(self):
        return self.positions, self.values


class ObjectFrameSDF(abc.ABC):
    @abc.abstractmethod
    def __call__(self, points_in_object_frame):
        """
        Evaluate the signed distance function at given points in the object frame
        :param points_in_object_frame: B x N x d d-dimensional points (2 or 3) of B batches; located in object frame
        :return: tuple of B x N signed distance from closest object surface in m and B x N x d SDF gradient pointing
            towards higher SDF values (away from surface when outside the object and towards the surface when inside)
        """

    def get_voxel_view(self, voxels: VoxelGrid = None) -> torch_view.TorchMultidimView:
        """
        Get a voxel view of a part of the SDF
        :param voxels: the voxel over which to evaluate the SDF; if left as none, take the default range which is
        implementation dependent
        :return:
        """
        if voxels is None:
            voxels = VoxelGrid(0.01, [[-1, 1], [-1, 1], [-0.6, 1]])

        pts = voxels.get_voxel_center_points()
        sdf_val, sdf_grad = self.__call__(pts.unsqueeze(0))
        cached_underlying_sdf = sdf_val.reshape([len(coord) for coord in voxels.coords])

        return torch_view.TorchMultidimView(cached_underlying_sdf, voxels.range_per_dim, invalid_value=self.__call__)

    def get_filtered_points(self, unary_filter, voxels: VoxelGrid = None) -> torch.tensor:
        """
        Get a N x d sequence of points extracted from a voxel grid such that their SDF values satisfy a given
        unary filter (on their SDF value)
        :param unary_filter: filter on the SDF value of each point, evaluting to true results in accepting that point
        :param voxels:
        :return:
        """
        model_voxels = self.get_voxel_view(voxels)
        interior = unary_filter(model_voxels.raw_data)
        indices = interior.nonzero()
        # these points are in object frame
        return model_voxels.ensure_value_key(indices)
