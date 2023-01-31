# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------

import copy
import numpy as np

from urdf_parser_py.urdf import URDF

import simulator.utils as utils
import simulator.constants as constants
import simulator.core.concepts.tools.fk_using_urdf as urdf_fk


class Contact:
    """
    Position and force direction of each fingertip in the contact web.
    """

    def __init__(self, position, force_direction):
        """
        position:        a vector of floats in x,y,z order (relative to contact web origin)
        force_direction: a vector of floats in x,y,z order (relative to contact web origin), size must be 1
        """
        self.position = position
        self.force_direction = force_direction

class ContactWeb:
    """
    The shape of fingertips.
    """
    
    def __init__(self, cmap, position, orientation):
        """
        cmap:        a vector of Contact class objects, each element is a finger
        position:    position of the contact-web center(from passed config (+ noise))
        orientation: orientation of the contact-web (from passed config (+ noise))
        """
        self.cmap = cmap
        self.position = position
        self.orientation = orientation
    def getContactPosition(self, i):
        v = utils.quat_mul_vec(self.orientation, self.cmap[i].position)
        return np.array(self.position) + np.array(v)


def loadContactWeb(myrobot, config):
    """
    Generate a contact-web (grasp contact points) using config.
    """

    p_pyb2cwebt0 = np.array([config["grasp_cweb0_position_x"],
                             config["grasp_cweb0_position_y"],
                             config["grasp_cweb0_position_z"]])
    q_pyb2cwebt0 = [config["grasp_cweb0_orientation_x"], config["grasp_cweb0_orientation_y"],
                    config["grasp_cweb0_orientation_z"], config["grasp_cweb0_orientation_w"]]
    
    # gt: ground truth
    p_pyb2cwebt0_gt = copy.deepcopy(p_pyb2cwebt0)
    q_pyb2cwebt0_gt = copy.deepcopy(q_pyb2cwebt0)
    
    # add noise to the contact-web
    use_simulated_vision_sensor = ("grasp_px_noise" in config)
    if use_simulated_vision_sensor:  # do not include noise in config if using real vision sensors
        print('using simulated vision results!')
        noise_p = np.array([config["grasp_px_noise"], config["grasp_py_noise"], config["grasp_pz_noise"]])
        p_pyb2cwebt0 = p_pyb2cwebt0_gt + noise_p
        qx = utils.quaternion_about_axis(config["grasp_qx_noise"], (1, 0, 0))
        qy = utils.quaternion_about_axis(config["grasp_qy_noise"], (0, 1, 0))
        qz = utils.quaternion_about_axis(config["grasp_qz_noise"], (0, 0, 1))
        noise_q = utils.quaternion_multiply(utils.quaternion_multiply(qz, qy), qx)
        q_pyb2cwebt0 = utils.quaternion_multiply(noise_q, q_pyb2cwebt0_gt)

    # generate the estimated contact-web
    if myrobot.grasp_type == constants.ConceptType.CONCEPT_GRASP_ACTIVE_FORCE:
        # Define the contact web: Thumb, Middle, order for the grasp_active_force grasp.
        cweb = ContactWeb(cmap=[
            Contact([-0.025, 0.0, 0.0], [1, 0, 0]),  # thumb
            Contact([0.025, 0.0, 0.0], [-1, 0, 0])  # middle
            ], position=p_pyb2cwebt0, orientation=q_pyb2cwebt0)
    else:
        print("error: contact web not defined")

    return {
        "p_pyb2cwebt0": p_pyb2cwebt0,
        "q_pyb2cwebt0": q_pyb2cwebt0,
        "p_pyb2cwebt0_gt": p_pyb2cwebt0_gt,
        "q_pyb2cwebt0_gt": q_pyb2cwebt0_gt,
        "cweb": cweb
    }


def handConfigurationFromContactWeb(myrobot, config, p_pyb2cwebt0, q_pyb2cwebt0):
    """
    Calculate the pose of the end-effector s.t. fingers will be located according to the contact-web.
    """

    vd_theta = config["grasp_approach_vertical"]
    vd_phi = config["grasp_approach_horizontal"]

    # calculate the hand orientation
    # below handQ calculates when contact-web orientation = identity matrix
    handQ_pybworld0 = myrobot.calcHandQuaternionXfrontZup(vd_theta, vd_phi)
    # rotate handQ depending on the contact-web orientation
    handQ_pybworld = utils.quaternion_multiply(q_pyb2cwebt0, handQ_pybworld0)
    
    # calculate the hand position
    # below calculation performed in ROS coordinate
    tmp = URDF.from_xml_file(myrobot.robot_urdf)
    jvtmp = myrobot.couplingRule(copy.deepcopy(myrobot.joint))
    
    contactP_atload = np.zeros(3)
    for localization_num in myrobot.virtual_finger:
        chain_name = tmp.get_chain(tmp.get_root(), myrobot.position_tip_links[localization_num], links=False)
        jointstmp = []
        for c in chain_name:
            j = 0.0
            for jidx, jname in enumerate(myrobot.command_joints):
                if jname == c: j = jvtmp[jidx]
            jointstmp += [j]
        root2tip = urdf_fk.chainname2trans(tmp, chain_name, jointstmp, fixed_excluded=False, get_root_com=False, get_end_com=True)
        pyb2root = urdf_fk.Transform()
        pyb2root.R = np.array(utils.quaternion_matrix(handQ_pybworld))[0:3, 0:3]
        pyb2root.T = np.array([0, 0, 0])
        pyb2tip = pyb2root.dot(root2tip)
        contactP_atload += pyb2tip.T
    contactP_atload /= len(myrobot.virtual_finger)
    
    # calculate how much to translate from diff atload and goal toplace
    handP_pybworld = p_pyb2cwebt0 - contactP_atload

    back = 0.15  # pregrasp-to-grasp distance
    vd = np.array(myrobot.calcApproachDirectionXfrontZup(vd_theta, vd_phi))
    vd = utils.quat_mul_vec(q_pyb2cwebt0, vd)

    handP_pybworld_pre = np.array(utils.uround(handP_pybworld)) - back/np.linalg.norm(vd)*vd

    return {
        "vd": vd,
        "vd_theta": vd_theta,
        "vd_phi": vd_phi,
        "handP_pybworld": handP_pybworld,
        "handQ_pybworld": handQ_pybworld,
        "handP_pybworld_pre": handP_pybworld_pre,
        "handQ_pybworld_pre": handQ_pybworld,
        "back": back
    }