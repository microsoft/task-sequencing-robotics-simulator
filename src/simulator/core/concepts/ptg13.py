# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------

import typing
import copy
import numpy as np

import simulator.utils as utils
import simulator.constants as tss_constants
import simulator.core.concepts.base as base
import simulator.core.envg_interface as envg_interface
import agent_profiles.agentp_template as agentp_template


class Concept(base.ConceptBase):

    def __init__(self):
        pass

    def init(self, agentp: agentp_template.AgentProfileBase, config: dict, args: dict):
        super().init(agentp, config, args)

        self.problem_operators = ["vd", "vg", "vw", "vw_l"]
        self.post_iters = 100  # number of iterations after contact

        # Current ptg13 assumes that the object can be placed using the same hand orientation as the previous task.
        # Exceptions are if the object is placed on a slope or a vertical wall.
        pos = args["tcmd_prev"]
        rot = args["rcmd_prev"]
        self.approachVertical = config["ptg13_approach_vertical"]

        th = np.deg2rad(config["ptg13_approach_vertical"])
        phi = np.deg2rad(config["ptg13_approach_horizontal"])
        d = config["ptg13_approach_distance"]  # absolute distance
        self.v_approach = max(d-0.02, 0.0)*np.array([np.cos(th)*np.cos(phi), np.cos(th)*np.sin(phi), np.sin(th)])
        self.handP_pybworld = np.array(pos) + self.v_approach
        self.handQ_pybworld = rot

        self._div = int(max(d-0.02,0)/0.05)+1 # number of iterations at approach

        self.vd = (1/np.linalg.norm(self.v_approach))*self.v_approach
        self.vg = utils.quat_mul_vec(self.handQ_pybworld, agentp.vg_l)
        self.vw_l = agentp.vw_l

        self.vw = utils.quat_mul_vec(self.handQ_pybworld, self.vw_l)
        A_temp = np.vstack((self.vd, self.vg, self.vw))
        self.Ainv = np.linalg.inv(A_temp.transpose())

        self.pos_th = 0.03
        self.orn_th = 10
    
    def generateReferenceMotion(self, agentp: agentp_template.AgentProfileBase, envg: envg_interface.EnvironmentEngineInterface, config: dict, args: dict) -> int:
        robot_state = envg.getKinematicsState()
        pos = robot_state.eef_main.b_position
        j = agentp.decouplingRule(copy.deepcopy(args["jcmd_prev"]))
        js = [copy.deepcopy(j), copy.deepcopy(j)]
        ts = [np.array(copy.deepcopy(pos)), self.handP_pybworld, self.handP_pybworld]
        
        # interpolate between preplace and place
        self.raw_trajectory = []
        self.raw_translation = []
        self.raw_rotation = []
        for i in range(self._div):
            t = float(i+1)/self._div
            jv = [None for x in range(len(agentp.reference_joints))]
            for j in range(len(agentp.reference_joints)):
                jv[j] = (1-t)*js[0][j] + t*js[1][j]
            self.raw_trajectory.append(jv)
            if abs(ts[1][2] - ts[0][2]) < 0.02:
                tv = [ts[0][0], ts[0][1], ts[0][2]]
            elif i < self._div-1:
                tv = [(1-t)*ts[0][0]+t*ts[1][0], (1-t)*ts[0][1]+t*ts[1][1], (1-t)*ts[0][2]+t*ts[1][2]]
            else:
                tv = [ts[1][0], ts[1][1], ts[1][2]]
            self.raw_translation.append(tv)
            rv = copy.deepcopy(self.handQ_pybworld)
            self.raw_rotation.append(rv)

        # place
        for i in range(self.post_iters+1):
            jv = copy.deepcopy(self.raw_trajectory[-1])
            tv = copy.deepcopy(self.raw_translation[-1])
            rv = copy.deepcopy(self.raw_rotation[-1])
            self.raw_trajectory.append(jv)
            self.raw_translation.append(tv)
            self.raw_rotation.append(rv)

        return len(self.raw_trajectory)-1

    def setDemonstrationParameters(self, coordinate_p: typing.List[float], coordinate_q: typing.List[float], state: dict) -> dict:
        return state  # no states added as not used for action decision
    
    def appendConceptSpecificStates(self, state: dict, agentp: agentp_template.AgentProfileBase, envg: envg_interface.EnvironmentEngineInterface) -> dict:
        sensor_definitions = agentp.getSensorDefinitions("v1", tss_constants.ConceptType.CONCEPT_PTG13)
        if envg.getPhysicsState("SurfaceContact", "", sensor_definitions): state["ptg13_plane_contact"] = -1
        else: state["ptg13_plane_contact"] = 10
        return state
    
    def evaluateStartingStateCondition(sself, agentp: agentp_template.AgentProfileBase, observation: dict) -> bool:
        return observation["observable_f0_contact"] and observation["observable_f2_contact"]

    def getAction(self, observation: dict, agentp: agentp_template.AgentProfileBase, brain_url: str="", client_id: str="") -> dict:
        dist = 0.005
        if observation["observable_timestep"] < self._div:
            del_action = np.matmul(self.Ainv,np.array([0,0,0.0])) ## Moving up 5 mm
            action = { "vd": del_action[0], "vg": del_action[1], "vw":del_action[2], "vw_l": 0.0,
                      "rh_FFJ3": 0.1,
                      "terminate": 0}
        else: ## moving 5 mm till collision
            n_add = observation["observable_timestep"] - self._div + 1
            del_action = np.matmul(self.Ainv,np.array([0,0, 0.0-dist*n_add])) ## Moving down 5 mm
            action = { "vd": del_action[0], "vg":del_action[1], "vw": del_action[2], "vw_l": 0.0,
                      "rh_FFJ3": 0.1,
                      "terminate": observation["ptg13_plane_contact"] <=0}
        return action
