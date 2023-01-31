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
        th = np.deg2rad(config["release_depart_vertical"])
        phi = np.deg2rad(config["release_depart_horizontal"])
        d = 0.15
        self.v_depart = d*np.array([np.cos(th)*np.cos(phi), np.cos(th)*np.sin(phi), np.sin(th)])
    
    def generateReferenceMotion(self, agentp: agentp_template.AgentProfileBase, envg: envg_interface.EnvironmentEngineInterface, config: dict, args: dict) -> int:
        robot_state = envg.getKinematicsState()
        pos = robot_state.eef_main.b_position
        rot = robot_state.eef_main.b_orientation
        j0 = agentp.decouplingRule(copy.deepcopy(args["jcmd_prev"]))
        j1 = copy.deepcopy(agentp.preshape_joint)
        js = [copy.deepcopy(j0), copy.deepcopy(j1)]
        ts = [np.array(copy.deepcopy(pos)), np.array(copy.deepcopy(pos)) + self.v_depart]
        rs = [copy.deepcopy(rot), copy.deepcopy(rot)]
        #
        div = 3
        self.raw_trajectory = []
        self.raw_translation = []
        self.raw_rotation = []
        for i in range(div+1):
            t = float(i)/div
            jv = [None for x in range(len(agentp.reference_joints))]
            for j in range(len(agentp.reference_joints)):
                jv[j] = (1-t)*js[0][j] + t*js[1][j]
            self.raw_trajectory.append(jv)
            tv = ts[0]
            self.raw_translation.append(tv)
            rv = rs[0]
            self.raw_rotation.append(rv)
        div = 3
        for i in range(div+1):
            self.raw_trajectory.append(self.raw_trajectory[-1])
            t = float(i)/div
            tv = [(1-t)*ts[0][0]+t*ts[1][0], (1-t)*ts[0][1]+t*ts[1][1], (1-t)*ts[0][2]+t*ts[1][2]]
            self.raw_translation.append(tv)
            rv = rs[0]
            self.raw_rotation.append(rv)
        return len(self.raw_trajectory)
    
    def setDemonstrationParameters(self, coordinate_p: typing.List[float], coordinate_q: typing.List[float], state: dict) -> dict:
        return state  # not used
        
    def evaluateStartingStateCondition(self, agentp: agentp_template.AgentProfileBase, observation: dict) -> bool:
        return True  # always valid starting state
        
    def getAction(self, observation: dict, agentp: agentp_template.AgentProfileBase, brain_url: str="", client_id: str="") -> dict:
        return {"terminate": observation["observable_timestep"] == len(self.raw_trajectory)}