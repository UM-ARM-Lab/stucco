import copy
import os
import subprocess
from timeit import default_timer as timer

import pybullet as p
import torch

from stucco import cfg
from stucco import icp
from stucco.icp import methods
from stucco.env.pybullet_env import make_sphere
from stucco.experiments.registration_nopytorch3d import saved_traj_dir_base, saved_traj_dir_for_method
from stucco.icp import costs as icp_costs, volumetric
from stucco.icp.medial_constraints import MedialBall
from stucco.sdf import ObjectFrameSDF

import logging

logger = logging.getLogger(__name__)


def do_registration(model_points_world_frame, model_points_register, best_tsf_guess, B,
                    volumetric_cost: icp_costs.VolumetricCost, reg_method: icp.ICPMethod, **kwargs):
    """Register a set of observed surface points in world frame to an object using some method

    :param model_points_world_frame:
    :param model_points_register:
    :param best_tsf_guess: initial estimate of object frame to world frame
    :param B: batch size, how many transforms to estimate
    :param volumetric_cost:
    :param reg_method:
    :return: B x 4 x 4 transform from world frame to object frame, B RMSE for each of the batches
    """
    # perform ICP and visualize the transformed points
    # compare not against current model points (which may be few), but against the maximum number of model points
    if reg_method == icp.ICPMethod.ICP:
        T, distances = methods.icp_pytorch3d(model_points_world_frame, model_points_register,
                                             given_init_pose=best_tsf_guess.inverse(), batch=B)
    elif reg_method == icp.ICPMethod.ICP_REVERSE:
        T, distances = methods.icp_pytorch3d(model_points_register, model_points_world_frame,
                                             given_init_pose=best_tsf_guess, batch=B)
        T = T.inverse()
    elif reg_method == icp.ICPMethod.ICP_SGD:
        T, distances = methods.icp_pytorch3d_sgd(model_points_world_frame, model_points_register,
                                                 given_init_pose=best_tsf_guess.inverse(), batch=B,
                                                 learn_translation=True,
                                                 use_matching_loss=True, **kwargs)
    elif reg_method == icp.ICPMethod.ICP_SGD_REVERSE:
        T, distances = methods.icp_pytorch3d_sgd(model_points_register, model_points_world_frame,
                                                 given_init_pose=best_tsf_guess, batch=B, learn_translation=True,
                                                 use_matching_loss=True, **kwargs)
        T = T.inverse()
    # use only volumetric loss
    elif reg_method == icp.ICPMethod.ICP_SGD_VOLUMETRIC_NO_ALIGNMENT:
        T, distances = methods.icp_pytorch3d_sgd(model_points_world_frame, model_points_register,
                                                 given_init_pose=best_tsf_guess.inverse(), batch=B,
                                                 pose_cost=volumetric_cost,
                                                 max_iterations=20, lr=0.01,
                                                 learn_translation=True,
                                                 use_matching_loss=False, **kwargs)
    elif reg_method in [icp.ICPMethod.VOLUMETRIC, icp.ICPMethod.VOLUMETRIC_NO_FREESPACE,
                        icp.ICPMethod.VOLUMETRIC_ICP_INIT, icp.ICPMethod.VOLUMETRIC_LIMITED_REINIT,
                        icp.ICPMethod.VOLUMETRIC_LIMITED_REINIT_FULL,
                        icp.ICPMethod.VOLUMETRIC_CMAES, icp.ICPMethod.VOLUMETRIC_CMAME,
                        icp.ICPMethod.VOLUMETRIC_CMAMEGA,
                        icp.ICPMethod.VOLUMETRIC_SVGD]:
        if reg_method == icp.ICPMethod.VOLUMETRIC_NO_FREESPACE:
            volumetric_cost = copy.copy(volumetric_cost)
            volumetric_cost.scale_known_freespace = 0
        if reg_method == icp.ICPMethod.VOLUMETRIC_ICP_INIT:
            # try always using the prior
            # best_tsf_guess = exploration.random_upright_transforms(B, model_points_register.dtype,
            #                                                        model_points_register.device)
            T, distances = methods.icp_pytorch3d(model_points_register, model_points_world_frame,
                                                 given_init_pose=best_tsf_guess, batch=B)
            best_tsf_guess = T

        optimization = volumetric.Optimization.SGD
        if reg_method == icp.ICPMethod.VOLUMETRIC_CMAES:
            optimization = volumetric.Optimization.CMAES
        elif reg_method == icp.ICPMethod.VOLUMETRIC_CMAME:
            optimization = volumetric.Optimization.CMAME
        elif reg_method == icp.ICPMethod.VOLUMETRIC_CMAMEGA:
            optimization = volumetric.Optimization.CMAMEGA
        elif reg_method == icp.ICPMethod.VOLUMETRIC_SVGD:
            optimization = volumetric.Optimization.SVGD
        # so given_init_pose expects world frame to object frame
        T, distances = methods.icp_volumetric(volumetric_cost, model_points_world_frame, optimization=optimization,
                                              given_init_pose=best_tsf_guess.inverse(), save_loss_plot=False,
                                              batch=B, **kwargs)
    else:
        raise RuntimeError(f"Unsupported ICP method {reg_method}")
    # T, distances = icp.icp_mpc(model_points_world_frame, model_points_register,
    #                            icp_costs.ICPPoseCostMatrixInputWrapper(volumetric_cost),
    #                            given_init_pose=best_tsf_guess, batch=B, draw_mesh=exp.draw_mesh)

    # T, distances = icp.icp_stein(model_points_world_frame, model_points_register, given_init_pose=T.inverse(),
    #                              batch=B)
    return T, distances


ball_ids = []


def do_medial_constraint_registration(model_points_world_frame, obj_sdf: ObjectFrameSDF, best_tsf_guess, B,
                                      level, seed: int, pokes: int, vis=None, obj_factory=None,
                                      plot_balls=False, experiment_name="poke"):
    global ball_ids
    ball_generate_exe = os.path.join(cfg.ROOT_DIR, "../medial_constraint/build/medial_constraint_icp")
    if not os.path.isfile(ball_generate_exe):
        raise RuntimeError(f"Expecting medial constraint ball generating executable to be at {ball_generate_exe}! "
                           f"make sure that repository is cloned and built")
    free_surface_path = f"{saved_traj_dir_base(level, experiment_name=experiment_name)}_{seed}_free_surface.txt"
    points_path = f"{saved_traj_dir_base(level, experiment_name=experiment_name)}_{seed}.txt"
    balls_path = f"{saved_traj_dir_for_method(icp.ICPMethod.MEDIAL_CONSTRAINT, experiment_name=experiment_name)}/{level.name}_{seed}_balls.txt"
    # if it hasn't been already generated (it generates for all pokes so only needs to be run once per traj)
    if not os.path.isfile(balls_path):
        os.makedirs(os.path.dirname(balls_path), exist_ok=True)
        subprocess.run([ball_generate_exe, free_surface_path, points_path, balls_path])
        if not os.path.isfile(balls_path):
            raise RuntimeError(f"Medial balls file not created after execution - errors in ball generation!")

    # to efficiently batch process them, we'll be using it as just a tensor with semantics attached to each dimension
    balls = None
    elapsed = 0
    with open(balls_path) as f:
        data = f.readlines()
        i = 0
        while i < len(data):
            header = data[i].split()
            this_poke = int(header[0])
            num_balls = int(header[1])
            if len(header) > 2:
                elapsed = float(header[2])
            if this_poke < pokes:
                # keep going forward
                i += num_balls + 1
                continue
            elif this_poke > pokes:
                # assuming the pokes are ordered, if we're past then there won't be anymore of this poke later
                break

            balls = torch.tensor([[float(v) for v in line.strip().split()] for line in data[i + 1:i + num_balls]])
            break
    if balls is None:
        raise RuntimeError(f"Could now find balls for poke {pokes} in {balls_path}")
    if vis is not None and plot_balls:
        for ball_id in ball_ids:
            p.removeBody(ball_id)
        ball_ids = []
        for ball in balls:
            # create ball and visualize in vis
            ball_ids.append(
                make_sphere(ball[MedialBall.R], ball[MedialBall.X: MedialBall.Z + 1], mass=0, visual_only=True,
                            rgba=(0.7, 0.1, 0.2, 0.3)))

    start = timer()
    balls = balls.to(device=model_points_world_frame.device, dtype=model_points_world_frame.dtype)
    T, distances = methods.icp_medial_constraints(obj_sdf, balls,
                                                  model_points_world_frame,
                                                  given_init_pose=best_tsf_guess,
                                                  batch=B,
                                                  # parameters when combined with ground truth initialization helps debug
                                                  # maxiter=100, sigma=0.0001,
                                                  verbose=False,
                                                  save_loss_plot=False,
                                                  vis=vis, obj_factory=obj_factory)
    T = T.inverse()
    end = timer()
    elapsed += end - start
    # approximate extra time for generating the mesh from the swept freespace
    elapsed += 1
    logger.info(f"medial constraint RMSE: {distances.mean().item()}")
    return T, distances, elapsed
