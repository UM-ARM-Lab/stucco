import os
import typing

import numpy as np
import open3d as o3d
import torch
from torchmcubes import marching_cubes

from stucco import sdf


def export_pc(f, pc, augmented_data: typing.Any = ""):
    pc_serialized = [f"{pt[0]:.4f} {pt[1]:.4f} {pt[2]:.4f} {augmented_data}" for pt in pc]
    f.write("\n".join(pc_serialized))
    f.write("\n")


def export_pcs(f, pc_free, pc_occ):
    if len(pc_free):
        export_pc(f, pc_free, 0)
    if len(pc_occ):
        export_pc(f, pc_occ, 1)


def export_transform(f, T):
    T_serialized = [f"{t[0]:.4f} {t[1]:.4f} {t[2]:.4f} {t[3]:.4f}" for t in T]
    f.write("\n".join(T_serialized))
    f.write("\n")


def export_pc_register_against(point_cloud_file: str, target_sdf: sdf.ObjectFrameSDF, surface_thresh=0.005):
    os.makedirs(os.path.dirname(point_cloud_file), exist_ok=True)
    with open(point_cloud_file, 'w') as f:
        pc_surface = target_sdf.get_filtered_points(
            lambda voxel_sdf: (voxel_sdf < surface_thresh) & (voxel_sdf > -surface_thresh))
        if len(pc_surface) == 0:
            raise RuntimeError(f"Surface threshold of {surface_thresh} leads to no surface points")
        pc_free = target_sdf.get_filtered_points(lambda voxel_sdf: voxel_sdf >= surface_thresh)

        total_pts = len(pc_free) + len(pc_surface)
        f.write(f"{total_pts}\n")
        export_pcs(f, pc_free, pc_surface)


def export_free_surface(free_surface_file, free_voxels, pokes, vis=None):
    verts, faces = marching_cubes(free_voxels.get_voxel_values(), 1)

    verts = verts.cpu().numpy()
    faces = faces.cpu().numpy()

    # Use Open3D for visualization
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.compute_triangle_normals()
    mesh.compute_vertex_normals()
    sa = mesh.get_surface_area()
    points_to_sample_per_area = 0.5
    points_to_sample = round(points_to_sample_per_area * sa)
    pcd = mesh.sample_points_uniformly(number_of_points=points_to_sample)
    # convert back to our coordinates
    points = np.asarray(pcd.points)
    # x and z seems to be swapped for some reason
    points = np.stack((points[:, 2], points[:, 1], points[:, 0])).T
    points = free_voxels.voxels.ensure_value_key(
        torch.tensor(points, dtype=free_voxels.voxels.dtype, device=free_voxels.voxels.device), force=True)
    # by default will be pointing towards the inside of the object
    normals = np.asarray(pcd.normals)
    normals = np.stack((normals[:, 2], normals[:, 1], normals[:, 0])).T

    if vis is not None:
        # o3d.visualization.draw_geometries([mesh, pcd], window_name='Marching cubes (CUDA)')
        normal_scale = 0.02
        for i, pt in enumerate(points):
            vis.draw_point(f"free_surface.{i}", pt, color=(0, 1, 0), scale=0.5, length=0.005)
            vis.draw_2d_line(f"free_surface_normal.{i}", pt, normals[i], color=(1, 0, 0),
                             scale=normal_scale)

    os.makedirs(os.path.dirname(free_surface_file), exist_ok=True)
    with open(free_surface_file, 'a') as f:
        # write out the poke index and the size of the point cloud
        f.write(f"{pokes} {points_to_sample}\n")
        export_pc(f, points)
        export_pc(f, normals)
