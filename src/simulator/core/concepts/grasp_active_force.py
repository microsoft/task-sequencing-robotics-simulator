# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------

import typing
import numpy as np
import math
import requests
import json

import simulator.constants as tss_constants
import simulator.utils as utils
import simulator.core.concepts.grasp_base as gutils
import simulator.core.envg_interface as envg_interface
import simulator.structs as tss_structs
import agent_profiles.agentp_template as agentp_template


class Concept(gutils.ConceptGraspBase):

    def getAgentProfileString(self, config: dict):
        return "active_force"
    
    def evaluateSufficientCondition(self, agentp: agentp_template.AgentProfileBase, envg: envg_interface.EnvironmentEngineInterface, jcmd_last: typing.List[float], tcmd_last: typing.List[float], rcmd_last: typing.List[float]) -> dict:
        pos_bef, ori_bef = envg.getComponentState("Pose", self.target_name)

        # slightly raise object
        dtv = np.array([0.0, 0.0, 0.3])
        p = np.array(tcmd_last) + dtv
        cmd = tss_structs.ActionStruct(tss_structs.MultiLinkAction(agentp.command_joints, jcmd_last, (p[0], p[1], p[2]), rcmd_last), None, None, 1.)
        goal = tss_structs.BRDIKGoalStruct(cmd.eef_main, None, '', '')  # not used
        body_ik_settings = tss_structs.BodyIKStruct(tss_constants.ConceptType.CONCEPT_GRASP_ACTIVE_FORCE, goal, self.ikrest)  # not used
        envg.callEnvironmentUpdatePipeline(cmd, body_ik_settings)

        # evaluate
        pos_aft, ori_aft = envg.getComponentState("Pose", self.target_name)
        diff_orientation = utils.uround(utils.quaternion_multiply(ori_aft, utils.quaternion_conjugate(ori_bef)))
        d_euler_x, d_euler_y, d_euler_z = utils.euler_from_quaternion(diff_orientation, 'rxyz')
        position_error = np.linalg.norm(np.array(pos_aft) - (np.array(pos_bef)+dtv))
        orientation_error = np.linalg.norm([d_euler_x, d_euler_y, d_euler_z])
        reward = (position_error < self.pos_th) and (orientation_error < self.orn_th * math.pi/180)
        result = {}
        result["objectPosError"] = position_error
        result["objectOrnError"] = orientation_error
        if reward: result["reward_indicator"] = 1
        else: result["reward_indicator"] = 0
        return result

    def getAction(self, observation: dict, agentp: agentp_template.AgentProfileBase, brain_url: str="http://localhost:5000", client_id: str="1000") -> dict:
        observed_states = {key:observation[key] for key in ("observable_f0_position", "observable_f2_position", 
            "observable_f0_contact", "observable_f2_contact", "observable_hand_position", 
            "observable_hand_orientation", "observable_finger_state", "observable_timestep") if key in observation}
        
        # state transformation in inkling
        observed_states["observable_estimated_web_vf0"] = observation["observable_demo_parameters"][2:5]
        observed_states["observable_estimated_web_vf2"] = observation["observable_demo_parameters"][5:8]

        headers = {"Content-Type": "application/json"}
        predictPath = "/v2/clients/{clientId}/predict"
        endpoint = brain_url + predictPath.replace("{clientId}", client_id)
        requestBody = {"state": observed_states}
        response = requests.post(endpoint, data = json.dumps(requestBody), headers = headers)

        prediction = response.json()
        print(prediction)
        print(prediction.keys())

        action = prediction['concepts']['MyConcept']['action']

        return action

    def getSpawnComponentsForTraining(self, config: dict) -> typing.List[tss_structs.ComponentStruct]:
        # for the active force grasp training, the grasp point (contact-web) is located 3[cm] from the object's top-surface
        cweb2obj = [0.0, 0.0, -config["object_scalez_noise"]+0.03]
        target_component = self.getParameterizedTargetComponent(config, cweb2obj)
        constraint_component = self.getTargetCentricConstraintComponent(config)
        self.target_name = target_component.name
        return [target_component, constraint_component]