# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------

import typing
import copy
import numpy as np

import simulator.utils as utils
import simulator.core.concepts.base as base
import simulator.core.envg_interface as envg_interface
import agent_profiles.agentp_template as agentp_template


class Concept(base.ConceptBase):

    def __init__(self):
        pass
    
    def init(self, agentp: agentp_template.AgentProfileBase, config: dict, args: dict):
        super().init(agentp, config, args)

        # ptg12 uses world coordinate values instead of relative-to-previous-task values.
        # In general, a task-sequence could begin from ptg12, thus should not use relative values.

        self.problem_operators = []
        self.p_pyb2goal = np.array([config["stg12_goal_position_x"],
                                    config["stg12_goal_position_y"],
                                    config["stg12_goal_position_z"]])
        if "stg12_goal_orientation_x" in config:
            self.q_pyb2goal = [config["stg12_goal_orientation_x"],
                               config["stg12_goal_orientation_y"],
                               config["stg12_goal_orientation_z"],
                               config["stg12_goal_orientation_w"]]
        else:  # maintain current orientation
            self.q_pyb2goal = copy.deepcopy(args["rcmd_prev"])

        if "ignore_interpolation" in config: self.ignore_interpolation = config["ignore_interpolation"]
        else: self.ignore_interpolation = False
    
    def generateReferenceMotion(self, agentp: agentp_template.AgentProfileBase, envg: envg_interface.EnvironmentEngineInterface, config: dict, args: dict) -> int:
        robot_state = envg.getKinematicsState()
        pos = robot_state.eef_main.b_position
        rot = robot_state.eef_main.b_orientation
        j = agentp.decouplingRule(copy.deepcopy(args["jcmd_prev"]))  # to always use commanded states not actual joint states
        js = [copy.deepcopy(j), copy.deepcopy(j)]
        ts = [np.array(copy.deepcopy(pos)), np.array(copy.deepcopy(self.p_pyb2goal))]
        rs = [copy.deepcopy(rot), copy.deepcopy(self.q_pyb2goal)]

        if self.ignore_interpolation: div = 1
        else: div = int(np.linalg.norm(ts[0] - ts[1]) / 0.05) + 1

        self.raw_trajectory = []
        self.raw_translation = []
        self.raw_rotation = []
        for i in range(div):
            t = float(i+1)/div
            jv = [None for x in range(len(agentp.reference_joints))]
            for j in range(len(agentp.reference_joints)):
                jv[j] = (1-t)*js[0][j] + t*js[1][j]
            self.raw_trajectory.append(jv)
            tv = [(1-t)*ts[0][0]+t*ts[1][0], (1-t)*ts[0][1]+t*ts[1][1], (1-t)*ts[0][2]+t*ts[1][2]]
            self.raw_translation.append(tv)
            if np.linalg.norm(np.array(rs[0]) - np.array(rs[1])) > 0.00001:
                rv = utils.quaternion_slerp(rs[0], rs[1], t)
            else:
                rv = copy.deepcopy(rs[1])
            rv = utils.quaternion_slerp(rs[0], rs[1], t)
            self.raw_rotation.append(rv)
        return len(self.raw_trajectory)
    
    def setDemonstrationParameters(self, coordinate_p: typing.List[float], coordinate_q: typing.List[float], state: dict) -> dict:
        return state  # no states added as not used for action decision
        
    def evaluateStartingStateCondition(self, agentp: agentp_template.AgentProfileBase, observation: dict) -> bool:
        return True  # always valid starting state
        
    def getAction(self, observation: dict, agentp: agentp_template.AgentProfileBase, brain_url: str="", client_id: str="") -> dict:
        return {"terminate": observation["observable_timestep"] == len(self.raw_trajectory)}