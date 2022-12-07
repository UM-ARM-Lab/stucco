import enum
from stucco.icp.stein import icp_stein
from stucco.icp.methods import icp, icp_2, icp_3, icp_pytorch3d, icp_pytorch3d_sgd, icp_volumetric, icp_mpc, \
    icp_medial_constraints


class ICPMethod(enum.Enum):
    ICP = 0
    ICP_SGD = 1

    VOLUMETRIC = 2
    VOLUMETRIC_ICP_INIT = 7
    VOLUMETRIC_NO_FREESPACE = 3
    VOLUMETRIC_LIMITED_REINIT = 9
    VOLUMETRIC_CMAES = 10
    VOLUMETRIC_CMAME = 14
    VOLUMETRIC_CMAMEGA = 15
    VOLUMETRIC_SVGD = 11
    VOLUMETRIC_LIMITED_REINIT_FULL = 13

    ICP_SGD_VOLUMETRIC_NO_ALIGNMENT = 4
    ICP_REVERSE = 5
    ICP_SGD_REVERSE = 6

    # freespace baselines
    MEDIAL_CONSTRAINT = 8
    CVO = 12

    NONE = 16


class InitMethod(enum.Enum):
    ORIGIN = 0
    CONTACT_CENTROID = 1
    RANDOM = 2
