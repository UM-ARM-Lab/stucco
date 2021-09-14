# Soft Tracking Using Contacts for Cluttered Objects (STUCCO) to Perform Blind Object Retrieval
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

[comment]: <> (The method is split into contact detection and isolation, and tracking.)

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
from stucco.tracking import ContactSetSoft

# get contact parameters tuned through maximizing median FMI and minimizing median contact error on a training set
contact_params = RetrievalGetter.contact_parameters(env)
# we need an efficient implementation of pxpen; point to robot surface distance at a certain configuration
# in this case it's specialized to that environment
pt_to_config = arm.ArmPointToConfig(env)
contact_set = ContactSetSoft(pt_to_config, contact_params)
```

You then update it every control step (such as inside a controller) with contact information and change in robot
```python
# additional debugging/visualization information is stored in info, such as control and ground truth object poses
# observed x and dx 
contact_set.update(x, dx, contact_detector, info=info)
```

Segment the belief into hard assignments of objects for downstream usage:
```python
# MAP particle
pts = contact_set.get_posterior_points()
# contact parameters are stored in contact_set.p
# list of indices; each element of list corresponds to an object
groups = contact_set.get_hard_assignment(contact_set.p.hard_assignment_threshold)
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
