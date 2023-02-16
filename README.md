# STUCCO
This is the official library code for the paper [Soft Tracking Using Contacts for Cluttered Objects (STUCCO) to Perform Blind Object Retrieval](https://ieeexplore.ieee.org/document/9696372).
If you use it, please cite

```
@article{zhong2022soft,
  title={Soft tracking using contacts for cluttered objects to perform blind object retrieval},
  author={Zhong, Sheng and Fazeli, Nima and Berenson, Dmitry},
  journal={IEEE Robotics and Automation Letters},
  volume={7},
  number={2},
  pages={3507--3514},
  year={2022},
  publisher={IEEE}
}
```

## Installation
```pip install stucco```


## Usage
This package is meant as a light-weight library for usage in your projects. 
See the [website](https://johnsonzhong.me/projects/stucco/) for videos and a high level introduction.
To reproduce the results from the paper, see [stucco_experiments](https://github.com/UM-ARM-Lab/stucco_experiments).

This library provides code for both 1) contact detection and isolation, and 2) contact tracking. However, they can be
used independently of each other; i.e. you can supply the contact point manually to update the tracker instead of getting
it from the detector.

This section describes how to use each component, and provide implementation tips. The `pxpen` function measuring
distance between contact points and robot surfaces in given configurations need to be efficient, and we provide a guide
on how to implement them. The other key function, `pxdyn`, just needs to be callable with signature

```
(B x N x 3 points, B x N x SE(3) poses, B x N x se(3) change in poses) -> (B x N x 3 new points, B x N x SE(3) new poses)
```

Where `B` represent arbitrary batch dimension(s), `N` represent a number of contact points per step, some of which may
be missing or 1 and should behave under standard broadcasting rules.

### Contact Detection and Isolation

Detection and isolation uses the momentum observer. At high frequency, we get residual feedback that estimates applied
external wrench (force and torque) at the end effector. In simulation, we can get applied forces directly.

To manage the process, we have a `ContactDetector` object, created like:

```python
from stucco.detection import ContactDetector
from stucco.detection import ResidualPlanarContactSensor
import numpy as np

# sample points on the robot surface and the associated surface normals (your function)
# these should be in link frame
surface_points, surface_normals = get_robot_points()

# for end-effector force-torque residual, torque magnitudes are a lot smaller
# in sim without noise, select a precision to balance out the magnitudes
residual_precision = np.diag([1, 1, 1, 50, 50, 50])
residual_threshold = 3

# the Planar version is concretely implemented; a similar one could be implemented to handle more general cases
contact_detector = ContactDetector(residual_precision)
sensor = ResidualPlanarContactSensor(surface_points, surface_normals, residual_threshold)
contact_detector.register_contact_sensor(sensor)
```

You then feed this object high frequency residual data along with end-effector poses

```python
# get reaction force and reaction torque at end-effector 
if contact_detector.observe_residual(np.r_[reaction_force, reaction_torque], pose):
    contact_detector.observe_dx(dx)
    # other book-keeping in case of making a contact
```

This object can later be queried like `contact_detector.in_contact()` and passed to update the tracking

### Contact Point Tracking

The tracking is performed through the `ContactSetSoft` object, created like:

```python
from stucco.tracking import ContactSetSoft, ContactParameters, LinearTranslationalDynamics
from stucco.movable_sdf import PlanarMovableSDF

# tune through maximizing median FMI and minimizing median contact error on a training set
contact_params = ContactParameters(length=0.02,
                                   penetration_length=0.002,
                                   hard_assignment_threshold=0.4,
                                   intersection_tolerance=0.002)

# need an efficient implementation of pxpen; point to robot surface distance at a certain config
# see section below for how to implement one
# here we pass in a cached discretized signed distance field and its description
pxpen = PlanarMovableSDF(d_cache, min_x, min_y, max_x, max_y, cache_resolution, cache_y_len)

# pxdyn is LinearTranslationalDynamics by default, here we are making it explicit
contact_set = ContactSetSoft(pxpen, contact_params, pxdyn=LinearTranslationalDynamics())
```

You then update it every control step with robot pose and contact point info

```python
# get latest contact point through the contact detector 
# (or can be supplied manually through other means)
# supplying None indicates we are not in contact
# also retrieve dx for each p
p, dx = contact_detector.get_last_contact_location()
# observed current x
contact_set.update(x, dx, p)
```

Segment the belief into hard assignments of objects for downstream usage:

```python
# MAP particle
pts = contact_set.get_posterior_points()
# contact parameters are stored in contact_set.p
# list of indices; each element of list corresponds to an object
groups = contact_set.get_hard_assignment(contact_set.p.hard_assignment_threshold)

for group in groups:
    object_pts = pts[group]
    # use points associated with the object downstream
```

### Implementing `pxpen` (point to robot surface distance)

Our recommendation for this function is to discretize and cache the signed distance function (SDF)
of the robot end effector in link frame. To support this, we provide the base class `PlanarPointToConfig` that supplies
all the other functionality when provided the SDF cache and accompanying information.

Here are some tips for how to create this discretized SDF:

```python
import os
import torch
import numpy as np
from stucco.movable_sdf import PlanarMovableSDF


# note that this is for a planar environment with fixed orientation; 
# however, it is very easy to extend to 3D and free rotations; 
# the extension to free rotations will require a parallel way to perform rigid body transforms 
# on multiple points, which can be provided by pytorch_kinematics.transforms
class SamplePointToConfig(PlanarMovableSDF):
    def __init__(self):
        # save cache to file for easy loading (use your own path)
        fullname = 'sample_point_to_config.pkl'
        if os.path.exists(fullname):
            super().__init__(*torch.load(fullname))
        else:
            # first time creating cache
            # we need some environment where we can get its bounding box and query an SDF
            # create robot in simulation (use your own function)
            robot_id, gripper_id, pos = create_sim_robot()
            # get axis-aligned bounding box values
            aabb_min, aabb_max = get_aabb()
            min_x, min_y = aabb_min[:2]
            max_x, max_y = aabb_max[:2]

            # select a cache resolution (doesn't have to be very small)
            cache_resolution = 0.001
            # create mesh grid
            x = np.arange(min_x, max_x + cache_resolution, cache_resolution)
            y = np.arange(min_y, max_y + cache_resolution, cache_resolution)
            cache_y_len = len(y)

            d = np.zeros((len(x), len(y)))
            for i, xi in enumerate(x):
                for j, yj in enumerate(y):
                    pt = [xi, yj, pos[2]]
                    # point query of SDF (use your own function)
                    d[i, j] = closest_point_on_surface(robot_id, pt)
            # flatten to allow parallel query of multiple indices
            d_cache = d.reshape(-1)
            # save things in (rotated) link frame
            min_x -= pos[0]
            max_x -= pos[0]
            min_y -= pos[1]
            max_y -= pos[1]
            data = [d_cache, min_x, min_y, max_x, max_y, cache_resolution, cache_y_len]
            torch.save(data, fullname)
            super().__init__(*data)
```