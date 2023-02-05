import enum


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

    # additional experiments
    MEDIAL_CONSTRAINT_CMAME = 16

    NONE = 17


def registration_method_uses_only_contact_points(reg_method: ICPMethod):
    if reg_method in [ICPMethod.ICP, ICPMethod.ICP_SGD, ICPMethod.ICP_REVERSE,
                      ICPMethod.ICP_SGD_REVERSE, ICPMethod.VOLUMETRIC_NO_FREESPACE]:
        return True
    return False
