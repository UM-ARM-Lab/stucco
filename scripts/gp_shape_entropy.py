import argparse
import time

import matplotlib
import torch
import pybullet as p
import logging
import os
from datetime import datetime

import gpytorch
from matplotlib import cm
from matplotlib import pyplot as plt

from arm_pytorch_utilities import tensor_utils
from arm_pytorch_utilities.grad import jacobian

from stucco import cfg
from stucco import tracking
from stucco.env import arm
from stucco.env.arm import Levels
from stucco.env_getters.arm import RetrievalGetter
from stucco.exploration import GPModelWithDerivatives, ICPEVSampleModelPoints

from stucco.retrieval_controller import sample_model_points, KeyboardController

import pytorch_kinematics.transforms as tf

ch = logging.StreamHandler()
fh = logging.FileHandler(os.path.join(cfg.ROOT_DIR, "logs", "{}.log".format(datetime.now())))

logging.basicConfig(level=logging.INFO,
                    format='[%(levelname)s %(asctime)s %(pathname)s:%(lineno)d] %(message)s',
                    datefmt='%m-%d %H:%M:%S', handlers=[ch, fh])

logging.getLogger('matplotlib.font_manager').disabled = True

logger = logging.getLogger(__name__)


def test_icp(target_obj_id, vis):
    # z = env._observe_ee(return_z=True)[-1]
    # # test ICP using fixed set of points
    # o = p.getBasePositionAndOrientation(env.target_object_id)[0]
    # contact_points = np.stack([
    #     [o[0] - 0.045, o[1] - 0.05],
    #     [o[0] - 0.05, o[1] - 0.01],
    #     [o[0] - 0.045, o[1] + 0.02],
    #     [o[0] - 0.045, o[1] + 0.04],
    #     [o[0] - 0.01, o[1] + 0.05]
    # ])
    # actions = np.stack([
    #     [0.7, -0.7],
    #     [0.9, 0.2],
    #     [0.8, 0],
    #     [0.5, 0.6],
    #     [0, -0.8]
    # ])
    # contact_points = np.stack(contact_points)
    #
    # angle = 0.5
    # dx = -0.4
    # dy = 0.2
    # c, s = math.cos(angle), math.sin(angle)
    # rot = np.array([[c, -s],
    #                 [s, c]])
    # contact_points = np.dot(contact_points, rot.T)
    # contact_points[:, 0] += dx
    # contact_points[:, 1] += dy
    # actions = np.dot(actions, rot.T)
    #
    # state_c, action_c = state_action_color_pairs[0]
    # env.visualize_state_actions("fixed", contact_points, actions, state_c, action_c, 0.05)

    # for mustard bottle there's a hole in the model inside, we restrict it to avoid sampling points nearby
    model_points, model_normals, _ = sample_model_points(target_obj_id, num_points=100, force_z=None, mid_z=0.05,
                                                         seed=0, clean_cache=False, random_sample_sigma=0.2,
                                                         name="mustard_normal", vis=vis, restricted_points=(
            [(0.01897749298212774, -0.008559855822130511, 0.001455972652355926)]))

    device, dtype = model_points.device, model_points.dtype
    pose = p.getBasePositionAndOrientation(target_obj_id)
    link_to_current_tf = tf.Transform3d(pos=pose[0], rot=tf.xyzw_to_wxyz(
        tensor_utils.ensure_tensor(device, dtype, pose[1])), dtype=dtype, device=device)
    model_points = link_to_current_tf.transform_points(model_points)
    model_normals = link_to_current_tf.transform_normals(model_normals)

    for i, pt in enumerate(model_points):
        vis.draw_point(f"mpt.{i}", pt, color=(0, 0, 1), length=0.003)
        vis.draw_2d_line(f"mn.{i}", pt, -model_normals[i], color=(0, 0, 0), size=2., scale=0.03)

    # perform ICP and visualize the transformed points
    # history, transformed_contact_points = icp.icp(model_points[:, :2], contact_points,
    #                                               point_pairs_threshold=len(contact_points), verbose=True)

    # better to have few A than few B and then invert the transform
    # T, distances, i = icp.icp_2(contact_points, model_points[:, :2])
    # transformed_contact_points = np.dot(np.c_[contact_points, np.ones((contact_points.shape[0], 1))], T.T)
    # T, distances, i = icp.icp_2(model_points[:, :2], contact_points)
    # transformed_model_points = np.dot(np.c_[model_points[:, :2], np.ones((model_points.shape[0], 1))],
    #                                   np.linalg.inv(T).T)
    # for i, pt in enumerate(transformed_model_points):
    #     pt = [pt[0], pt[1], z]
    #     env.vis.draw_point(f"tmpt.{i}", pt, color=(0, 1, 0), length=0.003)


def franke(X, Y):
    term1 = .75 * torch.exp(-((9 * X - 2).pow(2) + (9 * Y - 2).pow(2)) / 4)
    term2 = .75 * torch.exp(-((9 * X + 1).pow(2)) / 49 - (9 * Y + 1) / 10)
    term3 = .5 * torch.exp(-((9 * X - 7).pow(2) + (9 * Y - 3).pow(2)) / 4)
    term4 = .2 * torch.exp(-(9 * X - 4).pow(2) - (9 * Y - 7).pow(2))

    f = term1 + term2 + term3 - term4
    dfx = -2 * (9 * X - 2) * 9 / 4 * term1 - 2 * (9 * X + 1) * 9 / 49 * term2 + \
          -2 * (9 * X - 7) * 9 / 4 * term3 + 2 * (9 * X - 4) * 9 * term4
    dfy = -2 * (9 * Y - 2) * 9 / 4 * term1 - 9 / 10 * term2 + \
          -2 * (9 * Y - 3) * 9 / 4 * term3 + 2 * (9 * Y - 7) * 9 * term4

    return f, dfx, dfy


from mesh_to_sdf import mesh_to_voxels
import trimesh
import skimage


def create_sdf(path):
    mesh = trimesh.load(path)
    voxels = mesh_to_voxels(mesh, 64, pad=True)

    vertices, faces, normals, _ = skimage.measure.marching_cubes(voxels, level=0)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals)
    mesh.show()
    return voxels


def direct_fit_2d():
    x = torch.tensor([
        [0.05, 0.01],
        [0.1, 0.02],
        [0.12, 0.025],
        [0.15, 0.05],
        [0.15, 0.08],
        [0.15, 0.10],
    ])
    f = torch.tensor([
        0, 0, 0, 0,
        0, 0
    ])
    df = torch.tensor([
        [0, 1],
        [-0.3, 0.8],
        [-0.1, 0.9],
        [-0.7, 0.2],
        [-0.8, 0.],
        [-0.8, 0.],
    ])
    train_x = x * 10
    train_y = torch.cat((f.view(-1, 1), df), dim=1)

    # official sample code
    # xv, yv = torch.meshgrid([torch.linspace(0, 1, 10), torch.linspace(0, 1, 10)])
    # train_x = torch.cat((
    #     xv.contiguous().view(xv.numel(), 1),
    #     yv.contiguous().view(yv.numel(), 1)),
    #     dim=1
    # )

    # f, dfx, dfy = franke(train_x[:, 0], train_x[:, 1])
    # train_y = torch.stack([f, dfx, dfy], -1).squeeze(1)

    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=3)  # Value + x-derivative + y-derivative
    model = GPModelWithDerivatives(train_x, train_y, likelihood)

    training_iter = 50

    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)  # Includes GaussianLikelihood parameters

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(training_iter):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        print("Iter %d/%d - Loss: %.3f   lengthscales: %.3f, %.3f   noise: %.3f" % (
            i + 1, training_iter, loss.item(),
            model.covar_module.base_kernel.lengthscale.squeeze()[0],
            model.covar_module.base_kernel.lengthscale.squeeze()[1],
            model.likelihood.noise.item()
        ))
        optimizer.step()
    # Set into eval mode
    model.eval()
    likelihood.eval()

    # Initialize plots
    # fig, ax = plt.subplots(2, 3, figsize=(14, 10))
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))

    # Test points
    n1, n2 = 50, 50
    xv, yv = torch.meshgrid([torch.linspace(-1, 3, n1), torch.linspace(-1, 3, n2)])
    # f, dfx, dfy = franke(xv, yv)

    # Make predictions
    with torch.no_grad(), gpytorch.settings.fast_computations(log_prob=False, covar_root_decomposition=False):
        test_x = torch.stack([xv.reshape(n1 * n2, 1), yv.reshape(n1 * n2, 1)], -1).squeeze(1)
        predictions = likelihood(model(test_x))
        mean = predictions.mean

    def gp_var(cx):
        pred = likelihood(model(cx))
        return pred.variance

    # query for next place to go based on gradient of variance
    gp_var_jacobian = jacobian(gp_var, x[-1])
    gp_var_grad = gp_var_jacobian[0]  # first dimension corresponds to SDF, other 2 are for the SDF gradient

    # project gradient onto tangent plane

    extent = (xv.min(), xv.max(), yv.min(), yv.max())
    # ax[0, 0].imshow(f, extent=extent, cmap=cm.jet)
    # ax[0, 0].set_title('True values')
    # ax[0, 1].imshow(dfx, extent=extent, cmap=cm.jet)
    # ax[0, 1].set_title('True x-derivatives')
    # ax[0, 2].imshow(dfy, extent=extent, cmap=cm.jet)
    # ax[0, 2].set_title('True y-derivatives')

    # ax = ax[1,0]
    imprint_norm = matplotlib.colors.Normalize(vmin=-5, vmax=5)
    fig.colorbar(matplotlib.cm.ScalarMappable(norm=imprint_norm), ax=ax)
    ax.imshow(mean[:, 0].detach().numpy().reshape(n1, n2).T, extent=extent, origin='lower', cmap=cm.jet)
    ax.set_title('Predicted values')

    # lower, upper = predictions.confidence_region()
    # std = (upper - lower) / 2
    # imprint_norm = matplotlib.colors.Normalize(vmin=0, vmax=torch.round(std.max()))
    # fig.colorbar(matplotlib.cm.ScalarMappable(norm=imprint_norm), ax=ax)
    # ax.imshow(std[:, 0].detach().numpy().reshape(n1, n2).T, extent=extent, origin='lower', cmap=cm.jet)
    # ax.set_title('Uncertainty (standard deviation)')

    ax.quiver(train_x[:, 0], train_x[:, 1], df[:, 0], df[:, 1], scale=10)

    # highlight the object boundary
    threshold = 0.01
    boundary = torch.abs(mean[:, 0]) < threshold
    boundary_x = test_x[boundary, :]
    ax.scatter(boundary_x[:, 0], boundary_x[:, 1], marker='.', c='k')

    # ax[1, 1].imshow(mean[:, 1].detach().numpy().reshape(n1, n2), extent=extent, cmap=cm.jet)
    # ax[1, 1].set_title('Predicted x-derivatives')
    # ax[1, 2].imshow(mean[:, 2].detach().numpy().reshape(n1, n2), extent=extent, cmap=cm.jet)
    # ax[1, 2].set_title('Predicted y-derivatives')
    plt.show()
    print('done')


def keyboard_control(env):
    print("waiting for arrow keys to be pressed to command a movement")
    contact_params = RetrievalGetter.contact_parameters(env)
    pt_to_config = arm.ArmPointToConfig(env)
    contact_set = tracking.ContactSetSoft(pt_to_config, contact_params)
    ctrl = KeyboardController(env.contact_detector, contact_set, nu=2)

    obs = env._obs()
    info = None
    while not ctrl.done():
        try:
            env.visualize_contact_set(contact_set)
            u = ctrl.command(obs, info)
            obs, _, done, info = env.step(u)
        except:
            pass
        time.sleep(0.05)
    print(ctrl.u_history)
    cleaned_u = [u for u in ctrl.u_history if u != (0, 0)]
    print(cleaned_u)


parser = argparse.ArgumentParser(description='Downstream task of blind object retrieval')
parser.add_argument('method',
                    choices=['ours', 'online-birch', 'online-dbscan', 'online-kmeans', 'gmphd'],
                    help='which method to run')
parser.add_argument('--seed', metavar='N', type=int, nargs='+',
                    default=[0],
                    help='random seed(s) to run')
parser.add_argument('--no_gui', action='store_true', help='force no GUI')
# run parameters
task_map = {"FB": Levels.FLAT_BOX, "BC": Levels.BEHIND_CAN, "IB": Levels.IN_BETWEEN, "SC": Levels.SIMPLE_CLUTTER,
            "TC": Levels.TOMATO_CAN}
parser.add_argument('--task', default="IB", choices=task_map.keys(), help='what task to run')

args = parser.parse_args()

if __name__ == "__main__":
    level = task_map[args.task]
    method_name = args.method

    # env = RetrievalGetter.env(level=level, mode=p.DIRECT if args.no_gui else p.GUI)

    # direct_fit_2d()
    # experiment = ICPErrorVarianceExploration()
    # experiment.run(run_name="icp_var_debug_3")
    experiment = ICPEVSampleModelPoints()
    experiment.run(run_name="icp_var_sample_points_overstep")
    # experiment = GPVarianceExploration()
    # experiment.run(run_name="gp_var")
