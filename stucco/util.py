import matplotlib
import torch
from multidim_indexing import torch_view


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


class VoxelGrid:
    def __init__(self, resolution, range_per_dim, dtype=torch.float, device='cpu'):
        self.resolution = resolution

        range_per_dim = get_divisible_range_by_resolution(resolution, range_per_dim)
        coords, _ = get_coordinates_and_points_in_grid(resolution, range_per_dim, dtype=dtype, device=device)
        # underlying data
        self._data = torch.zeros([len(coord) for coord in coords], dtype=dtype, device=device)
        self.voxels = torch_view.TorchMultidimView(self._data, range_per_dim, invalid_value=0)

    def __getitem__(self, pts):
        return self.voxels[pts]

    def __setitem__(self, pts, value):
        self.voxels[pts] = value
