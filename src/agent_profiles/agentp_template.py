# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------

import typing
import simulator.constants as tss_constants


class AgentProfileBase:
    """
    A template class for defining agent profiles (hardware-specific transformations, sensor descriptions).
    """

    reference_joints: typing.List[str]  # indicate all joints that have a reference value to generate grasp preshape/shape
    problem_joints: typing.List[str]    # indicate joints that have a reference value and will also be explored during training
    command_joints: typing.List[str]    # indicate joints used for commanding the robot (needed for looking up joint values when using fk_using_urdf etc.)
    indices: typing.List[int]           # some simulators may require the joint index number instead of the joint name in command_joints

    preshape_joint: typing.List[float]  # pregrasp shape
    joint: typing.List[float]           # grasp shape: action joints before coupling

    min_values: dict
    max_values: dict

    # hand axis parameters
    # allow the agent to slightly modify translation around vg_l and rotation around vw_l
    vg_l: typing.List[float]  # grasp open/close directions in hand model coordinate
    vw_l: typing.List[float]  # wrist axis (axis hand bends against arm) in hand model coordinate

    position_tip_links: typing.List[float]
    force_tip_links: typing.List[float]

    # virtual_finger indicates the finger to use for the reward calculation
    # when the finger actions are constrained to each other, we do not need to reward every finger
    # this is a kind of importance-sampling of which fingers which really need to be guided
    virtual_finger: typing.List[int]


    def __init__(self):
        pass

    def setParameters(self, config: dict):
        print("agent profile: not implemented warning!")

    """
    ---------------------------------------------------------------------------
    ------------------------ state profiles -----------------------------------
    ---------------------------------------------------------------------------
    """

    def getSensorDefinitions(self, version: str, concept: tss_constants.ConceptType) -> dict:
        """
        Refer to the appendConceptSpecificStates implementation of each concept for the required definitions.
        version: implementation version specification
        concept: the task requiring the sensor definition
        ---
        example return: {"sensor": "joint_torque", "joints": ["rh_FFJ3", "rh_THJ2"], "thresholds": [2,15]}
        """
        raise Exception("agent profile: not implemented error!")

    """
    ---------------------------------------------------------------------------
    ------------------------ action profiles ----------------------------------
    ---------------------------------------------------------------------------
    """

    def couplingRule(self, jv: typing.List[float]) -> typing.List[float]:
        """
        Some joints (command joint space) are coupled and move together to reduce the action space (problem joint space).
        jv: a vector of joint values in reference joint space order (NOT problem joint space order)
        ---
        return: a vector of joint values in robot command space order
        """
        raise Exception("agent profile: not implemented error!")

    # reverse of couplingRule, needed for generating trajectories other than grasp task
    def decouplingRule(self, jv: typing.List[float]) -> typing.List[float]:
        raise Exception("agent profile: not implemented error!")

    """
    ---------------------------------------------------------------------------
    ------------------------ coordinate transforms ----------------------------
    ---------------------------------------------------------------------------
    """

    def calcApproachDirectionXfrontZup(self, theta: float, phi: float) -> typing.List[float]:
        """
        From some grasp approach direction, return the approach direction in XYZ. (Some hand models may require an approach offset)
        theta: vertical component of the grasp approach direction (0 on horizontal plane)
        phi:   horizontal component of the grasp approach direction (0 facing forward)
        ---
        return: approach direction (length 3 vector) in X-front, Z-up right hand coordinate
        """
        raise Exception("agent profile: not implemented error!")

    def calcHandQuaternionXfrontZup(self, theta: float, phi: float) -> typing.List[float]:
        """
        Transform the robot hand model coordinate to the simulator X-front, Z-up right hand coordinate.
        theta: vertical component of the grasp approach direction (0 on horizontal plane)
        phi:   horizontal component of the grasp approach direction (0 facing forward)
        ---
        return: robot hand orientation (length 4 vector) in X-front, Z-up right hand coordinate
        """
        raise Exception("agent profile: not implemented error!")
