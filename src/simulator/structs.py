# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------

import typing
from enum import Enum

import simulator.constants as tss_constants


class EnvironmentEngineRole(Enum):
    KINEMATICS = 1
    PHYSICS = 2
    SCENERY = 3
    POST_PROCESS = 4

class ComponentType(Enum):
    RIGID = 0

class GeometryType(Enum):
    MESH = 0
    BOX = 1
    SPHERE = 2
    CYLINDER = 3
    CONE = 4
    FILE = 5

class GeometryStruct:
    def __init__(self, geometry_type: GeometryType, dimensions: typing.List[float]=[], url: str="", material: str=""):
        self.geometry_type = geometry_type
        self.dimensions = dimensions
        self.url = url
        self.material = material

class ComponentStruct:
    def __init__(self, name: str, pose: typing.Tuple[typing.Tuple, typing.Tuple], geometry: GeometryStruct, component_type: ComponentType, component_states: typing.Optional[typing.List[float]]=None, ignore_gravity: bool=True):
        self.name = name
        self.pose = pose
        self.geometry = geometry
        self.c_type = component_type
        self.c_states = component_states
        self.ignore_gravity = ignore_gravity

class MultiLinkAction:
    def __init__(self, joint_names: typing.List[str], joint_states: typing.List[float], base_position: typing.List[float], base_orientation: typing.List[float]):
        self.joint_names = joint_names
        self.joint_states = joint_states
        self.b_position = base_position        # I.K. goal for eefs, or a navigation goal for the body
        self.b_orientation = base_orientation  # I.K. goal for eefs, or a navigation goal for the body

class ActionStruct:
    def __init__(self, eef_main: MultiLinkAction, body: typing.Optional[MultiLinkAction], eef_sub: typing.Optional[MultiLinkAction], timesec: float, error: bool=False):
        self.eef_main = eef_main
        self.body = body
        self.eef_sub = eef_sub
        self.timesec = timesec
        self.error = error

class WorldStruct:
    def __init__(self, robot_state: ActionStruct, component_states: typing.List[ComponentStruct]):
        self.robot_state = robot_state
        self.component_states = component_states

class BRDIKGoalStruct:
    """
    The IK goal structure used for the Body Role Division (BRD) IK algorithm.
    eef_main:      desired pose of the main end-effector and gripper joints
    eef_sub:       for dual-arm manipulators (TBD)
    start_posture: posture constraint A
    end_posture:   posture constraint B
    posture_rate:  how much to blend posture constraint A and B
    """
    def __init__(self, eef_main: MultiLinkAction, eef_sub: typing.Optional[MultiLinkAction], start_posture: str='', end_posture: str='', posture_rate=1.0):
        if eef_sub is None: self.goals = [(eef_main.b_position, eef_main.b_orientation)]
        else: self.goals = [(eef_main.b_position, eef_main.b_orientation), (eef_sub.b_position, eef_sub.b_orientation)]
        self.start_posture = start_posture
        self.end_posture = end_posture
        self.posture_rate = posture_rate

class BodyIKRest:
    """
    Additional body IK settings to allow implementation flexibility.
    """
    subsequent_task: int
    grasp_type: int
    def __init__(self, config: dict):
        if "subsequent_task" in config: self.subsequent_task = config["subsequent_task"]
        if "grasp_type" in config: self.grasp_type = config["grasp_type"]

class BodyIKStruct:
    """
    Body IK solve settings.
    task:        the current task (an IK solver may use this information to trigger different solvers)
    goal:        the IK goal
    rest:        additional IK settings
    lookat_goal: lookat IK goal if any (TBD)
    """
    def __init__(self, task: tss_constants.ConceptType, goal: BRDIKGoalStruct, rest: BodyIKRest, lookat_goal: typing.Optional[typing.Tuple]=None):
        self.task = task 
        self.goal = goal
        self.rest = rest
        if goal is None:
            print("BodyIKStruct goal is None. This is usual if using exceptional access.")
            self.is_dual_arm = False
        else: self.is_dual_arm = len(goal.goals) > 1
        self.lookat_goal = lookat_goal