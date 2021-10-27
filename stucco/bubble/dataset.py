import numpy as np
import torch

from bubble_utils.bubble_datasets.bubble_dataset_base import BubbleDatasetBase
from mmint_camera_utils.camera_utils import project_depth_image


class BubbleYCBDataset(BubbleDatasetBase):

    def __init__(self, *args, wrench_frame=None, tf_frame='grasp_frame', left=True, right=True, **kwargs):
        self.wrench_frame = wrench_frame
        self.tf_frame = tf_frame
        self.left = left
        self.right = right
        super().__init__(*args, **kwargs)

    @property
    def name(self):
        return 'bubble_ycb_dataset'

    def _get_sample(self, fc):
        """
        This function returns the loaded sample for the filecode fc.
        The return is a dict of the sample values
        """
        # fc: index of the line in the datalegend (self.dl) of the sample
        dl_line = self.dl.iloc[fc]
        scene_name = dl_line['Scene']
        undef_fc = dl_line['UndeformedFC']
        state_fc = dl_line['StateFC']
        # Load initial state:
        imprints = []
        pc = []
        # TODO load camera matrices (different for different camera names)
        K = np.array([216.42568969726562, 0.0, 118.77024841308594, 0.0, 216.42568969726562, 87.25411987304688, 0.0, 0.0,
                      1.0]).reshape(3, 3)
        if self.right:
            camera_name = 'right'
            imprints.append(self._get_depth_imprint(undef_fc=undef_fc, def_fc=state_fc, scene_name=scene_name,
                                                    camera_name=camera_name))
            depth_img = self._load_depth_img(state_fc, scene_name, camera_name)
            # these are in optical/camera frame
            # TODO convert them from camera frame to world frame
            pc.append(self._process_bubble_img(project_depth_image(depth_img, K)))
        if self.left:
            camera_name = 'left'
            imprints.append(self._get_depth_imprint(undef_fc=undef_fc, def_fc=state_fc, scene_name=scene_name,
                                                    camera_name=camera_name))
            depth_img = self._load_depth_img(state_fc, scene_name, camera_name)
            pc.append(self._process_bubble_img(project_depth_image(depth_img, K)))


        init_wrench = self._get_wrench(fc=state_fc, scene_name=scene_name, frame_id=self.wrench_frame)

        init_tf = self._get_tfs(state_fc, scene_name=scene_name, frame_id=self.tf_frame)
        if init_tf is not None:
            init_pos = init_tf[..., :3]
            init_quat = init_tf[..., 3:]
        else:
            init_pos = np.zeros(3)
            init_quat = np.zeros(4)

        # imprint and point cloud have 1-to-1 correspendence, so can filter based on that
        sample = {
            'imprint': torch.tensor(imprints, dtype=torch.float),
            'pc': torch.tensor(pc, dtype=torch.float),
            'wrench': init_wrench,
            'pos': init_pos,
            'quat': init_quat,
            'undef_fc': undef_fc,
            'fc': state_fc,
            'scene_name': scene_name,
        }

        return sample
