# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------

import typing
import copy
import numpy as np

import simulator.core.concepts.base as base
import simulator.core.envg_interface as envg_interface
import agent_profiles.agentp_template as agentp_template


class Concept(base.ConceptBase):

    def __init__(self):
        pass

    def init(self, agentp: agentp_template.AgentProfileBase, config: dict, args: dict):
        super().init(agentp, config, args)

        self.problem_operators = []
        th = np.deg2rad(config["ptg11_depart_vertical"])
        phi = np.deg2rad(config["ptg11_depart_horizontal"])
        d = config["ptg11_depart_distance"]
        self.v_depart = d*np.array([np.cos(th)*np.cos(phi), np.cos(th)*np.sin(phi), np.sin(th)])
        self._div = int(d/0.05)+1
    
    def generateReferenceMotion(self, agentp: agentp_template.AgentProfileBase, envg: envg_interface.EnvironmentEngineInterface, config: dict, args: dict) -> int:
        robot_state = envg.getKinematicsState()
        pos = robot_state.eef_main.b_position
        rot = robot_state.eef_main.b_orientation
        j = agentp.decouplingRule(copy.deepcopy(args["jcmd_prev"]))  # to always use commanded states not actual joint states
        js = [copy.deepcopy(j), copy.deepcopy(j)]
        ts = [np.array(copy.deepcopy(pos)), np.array(copy.deepcopy(pos)) + self.v_depart]

        self.raw_trajectory = []
        self.raw_translation = []
        self.raw_rotation = []
        for i in range(self._div):
            t = float(i+1)/self._div
            jv = [None for x in range(len(agentp.reference_joints))]
            for j in range(len(agentp.reference_joints)):
                jv[j] = (1-t)*js[0][j] + t*js[1][j]
            self.raw_trajectory.append(jv)
            tv = [(1-t)*ts[0][0]+t*ts[1][0], (1-t)*ts[0][1]+t*ts[1][1], (1-t)*ts[0][2]+t*ts[1][2]]
            self.raw_translation.append(tv)
            rv = copy.deepcopy(rot)
            self.raw_rotation.append(rv)
        return len(self.raw_trajectory)
    
    def setDemonstrationParameters(self, coordinate_p: typing.List[float], coordinate_q: typing.List[float], state: dict) -> dict:
        return state  # no states added as not used for action decision
    
    def evaluateStartingStateCondition(self, agentp: agentp_template.AgentProfileBase, observation: dict) -> bool:
        return True  # always valid starting state

    def getAction(self, observation: dict, agentp: agentp_template.AgentProfileBase, brain_url: str="", client_id: str="") -> dict:
        return {"terminate": observation["observable_timestep"] == len(self.raw_trajectory)}