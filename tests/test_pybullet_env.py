import numpy as np
import pybullet as p
from stucco.env.pybullet_env import closest_point_on_surface, make_sphere, ContactInfo


def test_closest_point_on_surface():
    clientID = p.connect(p.GUI)
    r = 1
    objId = make_sphere(r, [0., 0, 0])

    closest_surface_pt = closest_point_on_surface(objId, [0.1, 0, 0], return_full_contact_info=False)
    assert np.allclose(closest_surface_pt, [r, 0, 0])

    p.disconnect(clientID)


if __name__ == "__main__":
    test_closest_point_on_surface()
