import torch
from stucco.tracking import ContactSetSoft, ContactParameters
from arm_pytorch_utilities.rand import seed
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


class CircularPointToConfig:
    """Measure distance from a point to the robot surface for the robot in some configuration"""

    def __init__(self, robot_radius):
        self.r = robot_radius

    def __call__(self, configs, pts):
        M = configs.shape[0]
        N = pts.shape[0]

        # take advantage of the fact our system has no rotation to just translate points; otherwise would need full tsf
        query_pts = pts.repeat(M, 1, 1).transpose(0, 1)
        # needed transpose to allow broadcasting
        query_pts -= configs
        # flatten
        query_pts = query_pts.transpose(0, 1).view(M * N, -1)

        d = query_pts.norm(dim=-1) - self.r

        return d.view(M, N)


PREV_PT_RGBA = (0.1, 0.8, 0.1, 0.5)
CUR_PT_RGBA = "#FFA500"


def plot_and_update(r, contact_set, x, dx, p):
    seed(1)

    fig, ax = plt.subplots()
    ax.axis('equal')
    ax.grid(True, linestyle='--')
    ax.set(xlim=(-1.5, 1.5), ylim=(-1.5, 1.5))
    s = 12

    # plot robot
    prev_x = x - dx
    prev_robot = Circle(prev_x[0], r, color=(0.8, 0.1, 0.1, 0.5), zorder=0)
    ax.add_patch(prev_robot)

    prev_pt = p - dx

    robot = Circle(x, r, color=(0.8, 0.1, 0.1, 0.8), zorder=1)
    ax.add_patch(robot)

    prev_sampled_pts = None
    if contact_set.sampled_pts is not None:
        prev_sampled_pts = contact_set.sampled_pts.clone()

    contact_set.update(x, dx, p)

    for i in range(contact_set.n_particles):
        ax.set_title(f'particle {i}')
        actors = []
        cx = contact_set.sampled_pts[i, :, 0].view(1, -1)
        cy = contact_set.sampled_pts[i, :, 1].view(1, -1)

        lx = prev_pt[:, 0].view(-1)
        ly = prev_pt[:, 1].view(-1)
        if prev_sampled_pts is not None:
            px = torch.cat((prev_sampled_pts[i, :, 0], lx))
            py = torch.cat((prev_sampled_pts[i, :, 1], ly))
        else:
            px = lx
            py = ly

        actors.append(ax.scatter(px, py, s=s, c=[PREV_PT_RGBA], zorder=2))
        actors.append(ax.quiver(px, py, cx - px, cy - py, scale=1, width=0.01, units='xy', zorder=3))
        actors.append(ax.scatter(cx, cy, s=s, c=[CUR_PT_RGBA for _ in cx], zorder=4))
        actors.append(ax.scatter(p[:, 0], p[:, 1], s=s, c=[(0, 0, 0)], zorder=5))

        plt.pause(0.05)
        plt.savefig(f'/home/zhsh/Pictures/fig.png')

        user_input = input('enter for next, q for break\n')

        if user_input == 'q':
            break

        for actor in actors:
            actor.remove()


def test_multiple_dynamics():
    r = 0.1
    pt_to_config = CircularPointToConfig(r)
    params = ContactParameters(length=0.5, penetration_length=0.1, hard_assignment_threshold=0.4,
                               intersection_tolerance=0.0)

    contact_set = ContactSetSoft(pt_to_config, params, n_particles=10)

    # simulation
    x = torch.tensor([0., 0.])

    plt.ion()
    plt.show()

    x = x + torch.tensor([0.3, 0.])
    dx = torch.tensor([[0.3, 0.], [0.15, 0.]])
    p = torch.tensor([[0.4, 0.23], [0.4, 0.4]])
    plot_and_update(r, contact_set, x, dx, p)

    x = x + torch.tensor([0.2, 0.])
    dx = torch.tensor([[0.2, 0.], [0.1, 0.]])
    p = torch.tensor([[0.6, -0.15], [0.6, -0.25]])
    plot_and_update(r, contact_set, x, dx, p)

    x = x + torch.tensor([0.3, 0.])
    dx = torch.tensor([[0.3, 0.], [0.15, 0.]])
    p = torch.tensor([[0.9, -0.35], [0.9, 0.55]])
    plot_and_update(r, contact_set, x, dx, p)

def test_single_dynamics():
    r = 0.5
    pt_to_config = CircularPointToConfig(r)
    params = ContactParameters(length=0.5, penetration_length=0.1, hard_assignment_threshold=0.4,
                               intersection_tolerance=0.0)

    contact_set = ContactSetSoft(pt_to_config, params, n_particles=10)

    # simulation
    x = torch.tensor([0., 0.])
    dx = torch.tensor([0.3, 0.])

    plt.ion()
    plt.show()

    x = x + dx
    plot_and_update(r, contact_set, x, dx.view(1, -1), torch.tensor([[0.4, 0.23]]))

    dx = torch.tensor([0.1, 0.])
    x = x + dx
    plot_and_update(r, contact_set, x, dx.view(1, -1), torch.tensor([[0.5, 0.15]]))

    dx = torch.tensor([0.1, 0.3])
    x = x + dx
    plot_and_update(r, contact_set, x, dx.view(1, -1), torch.tensor([[0.5, 0.]]))

    dx = torch.tensor([0.1, -0.3])
    x = x + dx
    plot_and_update(r, contact_set, x, dx.view(1, -1), torch.tensor([[0.6, -0.1]]))


if __name__ == "__main__":
    # test_single_dynamics()
    test_multiple_dynamics()
