# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------

import typing
import copy
import numpy as np

import simulator.utils as utils
import simulator.constants as tss_constants
import agent_profiles.agentp_template as agentp_template


class Profile(agentp_template.AgentProfileBase):

    def __init__(self):
        self.joint = [
            -0.020622028419300796, 0.791124618965486, 0.7165877145100853, 0.0, 
            0.9159989384779349, 1.2087578520408695, -0.6628910395680104, 0.15071510414034017
        ]
        self.preshape_joint = copy.deepcopy(self.joint)
        self.preshape_joint[1:5], self.preshape_joint[6:8] = [0,0,0,0], [0,0]
        self.vg_l = [0, 1, 0]  # grasp open/close directions in hand model coordinate
        self.vw_l = [1, 0, 0]  # wrist axis (axis hand bends against arm) in hand model coordinate
        
        self.reference_joints = ["rh_FFJ4", "rh_FFJ3", "rh_FFJ2","rh_rFFJ1",
                                 "rh_THJ5", "rh_THJ4", "rh_THJ2", "rh_THJ1"]
        self.problem_joints = ["rh_FFJ3", "rh_FFJ2", "rh_rFFJ1",
                               "rh_THJ5", "rh_THJ4", "rh_THJ2", "rh_THJ1"]
        self.command_joints = ["rh_FFJ4", "rh_FFJ3", "rh_FFJ2","rh_FFJ1",
                               "rh_MFJ4", "rh_MFJ3", "rh_MFJ2", "rh_MFJ1",
                               "rh_RFJ4", "rh_RFJ3", "rh_RFJ2", "rh_RFJ1",
                               "rh_THJ5", "rh_THJ4", "rh_THJ2", "rh_THJ1"]
        
        # joint limits
        self.min_values = {
            "rh_FFJ4": -0.349, "rh_FFJ3": 0.0, "rh_FFJ2": 0.0, "rh_rFFJ1": 0.0,
            "rh_MFJ4": -0.349, "rh_MFJ3": 0.0, "rh_MFJ2": 0.0, "rh_rMFJ1": 0.0,
            "rh_RFJ4": -0.349, "rh_RFJ3": 0.0, "rh_RFJ2": 0.0, "rh_rRFJ1": 0.0,
            "rh_THJ5": -1.047, "rh_THJ4": 0.0, "rh_THJ2": -0.524, "rh_THJ1": 0.0
        }
        self.max_values = {
            "rh_FFJ4": 0.349, "rh_FFJ3": 1.571, "rh_FFJ2": 1.571, "rh_rFFJ1": 1.0,
            "rh_MFJ4": 0.349, "rh_MFJ3": 1.571, "rh_MFJ2": 1.571, "rh_rMFJ1": 1.0,
            "rh_RFJ4": 0.349, "rh_RFJ3": 1.571, "rh_RFJ2": 1.571, "rh_rRFJ1": 1.0,
            "rh_THJ5": 1.047, "rh_THJ4": 1.222, "rh_THJ2": 0.524, "rh_THJ1": 1.571
        }
        
        # other settings for loading robot, calculating rewards, etc.
        self.robot_urdf = '../sr_common/sr_description/robots/shadowhand_lite.urdf'
        self.position_tip_links = ["rh_thdistal", "rh_ffdistal", "rh_mfdistal", "rh_rfdistal"]
        self.force_tip_links = ["rh_thdistal", "rh_ffdistal", "rh_mfdistal", "rh_rfdistal"]
        self.virtual_finger = [0, 2]
        self.indices = [4,5,6,7,9,10,11,12,14,15,16,17,19,20,22,23]
        
        self.grasp_type = tss_constants.ConceptType.CONCEPT_GRASP_ACTIVE_FORCE

    def getSensorDefinitions(self, version: str, concept: tss_constants.ConceptType) -> dict:
        if version == "v1":
            if concept == tss_constants.ConceptType.CONCEPT_PTG13:
                return {"sensor": "joint_torque", "joints": ["rh_FFJ3"], "joint_indices": [5], "thresholds": [30]}

    def couplingRule(self, jv: typing.List[float]) -> typing.List[float]:
        ff4 = copy.deepcopy(jv[0])
        ff3 = copy.deepcopy(jv[1])
        ff2 = copy.deepcopy(jv[2])
        ff1 = copy.deepcopy(jv[3])*ff2
        th5 = copy.deepcopy(jv[-4])
        th4 = copy.deepcopy(jv[-3])
        th2 = copy.deepcopy(jv[-2])
        th1 = copy.deepcopy(jv[-1])
        js = [ff4, ff3, ff2, ff1, 0, ff3, ff2, ff1, ff4, ff3, ff2, ff1, th5, th4, th2, th1]
        return js
    
    def decouplingRule(self, jv: typing.List[float]) -> typing.List[float]:
        if abs(jv[2]) < 0.000001: rff1 = 0.0
        else: rff1 = jv[3]/jv[2]
        return [jv[0], jv[1], jv[2], rff1, jv[-4], jv[-3], jv[-2], jv[-1]]

    def calcApproachDirectionXfrontZup(self, theta: float, phi: float) -> typing.List[float]:
        return [np.cos(np.deg2rad(theta))*np.cos(np.deg2rad(phi)),
                np.cos(np.deg2rad(theta))*np.sin(np.deg2rad(phi)),
                np.sin(np.deg2rad(theta))]
    
    def calcHandQuaternionXfrontZup(self, theta: float, phi: float) -> typing.List[float]:
        qx = utils.quaternion_about_axis(min(2.441, 2.441-np.deg2rad(theta)), [1, 0, 0])  # from pre-calculated mapping
        qz = utils.quaternion_about_axis(np.pi*.5 + np.deg2rad(phi), [0, 0, 1])
        return utils.quaternion_multiply(qz, qx)
