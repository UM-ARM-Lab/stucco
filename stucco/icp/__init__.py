import enum
from stucco.icp.stein import icp_stein
from stucco.icp.methods import icp, icp_2, icp_3, icp_pytorch3d, icp_pytorch3d_sgd, icp_volumetric, icp_mpc


class ICPMethod(enum.Enum):
    ICP = 0
    ICP_SGD = 1
    VOLUMETRIC = 2
    ICP_SGD_VOLUMETRIC_NO_ALIGNMENT = 3
