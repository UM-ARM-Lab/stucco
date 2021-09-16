# Soft Tracking Using Contacts for Cluttered Objects (STUCCO) to Perform Blind Object Retrieval
![real-min](https://user-images.githubusercontent.com/5508542/133346489-cc83db1c-28d3-4a69-8ae2-865b12b4cbd7.png)

## Requirements
- python 3.6+
- pytorch 1.5+

## Installation
1. install required libraries (clone then cd and `pip install -e .`)
[pytorch utilities](https://github.com/UM-ARM-Lab/arm_pytorch_utilities),
[pytorch kinematics](https://github.com/UM-ARM-Lab/pytorch_kinematics),
[YCB object models](https://github.com/eleramp/pybullet-object-models) (only for testing envs)
2. `pip install -e .`

## Usage
This library provides code for both 1) contact detection and isolation, and 2) contact tracking.
However, they can be used indepedently of each other; i.e. you can supply the contact point manually
to update the tracker instead of getting it from the detector.

This section describes how to use each component, and provide implementation tips.
The `pxpen` function measuring distance between contact points and robot surfaces in given configurations
need to be efficient, and we provide a guide on how to implement them.

### Contact Detection and Isolation
Detection and isolation is using the momentum observer. At high frequency, we get residual feedback
that estimates applied external wrench (force-torque) at the end effector. In simulation, we can
get this directly.

To manage the process, we have a `ContactDetector` object, created like:
```python
from stucco.detection import ContactDetector
import numpy as np

# for end-effector force-torque residual, torque magnitudes are a lot smaller
# in sim without noise, select a precision to balance out the magnitudes
residual_precision = np.diag([1, 1, 1, 50, 50, 50])
residual_threshold = 3

contact_detector = ContactDetector(residual_precision, residual_threshold)
# not a concrete class, look at ContactDetectorPlanar and ContactDetectorPlanarPybulletGripper for how to implement
# the detector for certain environment classes
```
You then feed this object high frequency residual data along with end-effector poses
```python
# get reaction force and reaction torque at end-effector 
if contact_detector.observe_residual(np.r_[reaction_force, reaction_torque], pose):
    # other book-keeping in case of making a contact
```
This object can later be queried like `contact_detector.in_contact()` and passed to update the tracking

### Contact Point Tracking
The tracking is performed through the `ContactSetSoft` object, created like:
```python
from stucco.tracking import ContactSetSoft, ContactParameters

# get contact parameters tuned through maximizing median FMI and minimizing median contact error on a training set
contact_params = ContactParameters(max_pos_move_per_action=0.03, # not currently important, may be useful for dynamics later
                                   length=0.02,
                                   penetration_length=0.002,
                                   hard_assignment_threshold=0.4,
                                   intersection_tolerance=0.002)

# we need an efficient implementation of pxpen; point to robot surface distance at a certain configuration
# see section below for how to implement one
from stucco.env import arm
pt_to_config = arm.ArmPointToConfig(env)

contact_set = ContactSetSoft(pt_to_config, contact_params)
```

You then update it every control step (such as inside a controller) with contact information and change in robot
```python
# get latest contact point through the contact detector (or can be supplied manually through other means)
# supplying None indicates we are not in contact
p = contact_detector.get_last_contact_location()
# observed x and dx 
contact_set.update(x, dx, p)
```

Segment the belief into hard assignments of objects for downstream usage:
```python
# MAP particle
pts = contact_set.get_posterior_points()
# contact parameters are stored in contact_set.p
# list of indices; each element of list corresponds to an object
groups = contact_set.get_hard_assignment(contact_set.p.hard_assignment_threshold)
```

### Implementing `pxpen` (point to robot surface distance)
Our recommendation for this function is to discretize and cache the signed distance function (SDF)
of the robot end effector in link frame. To support this, 
we provide the base class `PlanarPointToConfig` that supplies all the other functionality when provided
the SDF cache and accompanying information.

Here are some tips for how to create this discretized SDF:
```python
import os
import torch
import numpy as np
from stucco.env.env import PlanarPointToConfig

# note that this is for a planar environment with fixed orientation; however, it is very easy to extend to
# 3D and free rotations; the extension to free rotations will require a parallel way to perform rigid body transforms
# on multiple points, which can be provided by pytorch_kinematics.transforms
class SamplePointToConfig(PlanarPointToConfig):
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

## Reproduce Paper
1. collect training data
```shell
python collect_tracking_training_data --task SELECT1 --gui
python collect_tracking_training_data --task SELECT2 --gui
python collect_tracking_training_data --task SELECT3 --gui
python collect_tracking_training_data --task SELECT4 --gui
```
2. evaluate all tracking methods on this data
```shell
python evaluate_contact_tracking.py
```
3. plot tracking method performances on the training data
```shell
python plot_contact_tracking_res.py
```
4. run simulated BOR tasks (there is a visual bug after resetting environment 8 times, so we split up the runs for different seeds)
```shell
python retrieval_main.py ours --task FB --seed 0 1 2 3 4 5 6 7; python retrieval_main.py ours --task FB --seed 8 9 10 11 12 13 14 15; python retrieval_main.py ours --task FB --seed 16 17 18 19
python retrieval_main.py ours --task BC --seed 0 1 2 3 4 5 6 7; python retrieval_main.py ours --task BC --seed 8 9 10 11 12 13 14 15; python retrieval_main.py ours --task BC --seed 16 17 18 19
python retrieval_main.py ours --task IB --seed 0 1 2 3 4 5 6 7; python retrieval_main.py ours --task IB --seed 8 9 10 11 12 13 14 15; python retrieval_main.py ours --task IB --seed 16 17 18 19
```
repeat with baselines by replacing `ours` with `online-birch` and other baselines
