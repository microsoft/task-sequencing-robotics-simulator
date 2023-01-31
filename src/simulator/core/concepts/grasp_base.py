# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------

import typing
import copy
import numpy as np

import simulator.utils as utils
import simulator.core.concepts.tools.init_from_contactweb as init_from_contactweb
import simulator.core.concepts.base as base
import simulator.core.envg_interface as envg_interface
import simulator.structs as tss_structs
import agent_profiles.agentp_template as agentp_template


class ConceptGraspBase(base.ConceptBase):
    cweb: init_from_contactweb.ContactWeb

    def __init__(self):
        # Define self.target_name here as is required when the world is spawned by this concept (getSpawnComponentsForTraining()).
        # When the world is spawned externally (through detection etc.), the below name is not used and is overwritten on init().
        self.target_name = "target"

    def getActionSpace(self, agentp: agentp_template.AgentProfileBase) -> dict:
        action_values = super().getActionSpace(agentp)
        for a in agentp.problem_joints: action_values[a] = 0.0
        return action_values
    
    def init(self, agentp: agentp_template.AgentProfileBase, config: dict, args: dict):
        super().init(agentp, config, args)
        agentp.setParameters(config)

        if "target_name" in config: self.target_name = config["target_name"]

        # below parameters are optional as not used during execution
        if "config_pick_pos_thrshld" in config: self.pos_th = config["config_pick_pos_thrshld"]  # position threshold to check if pick is successful
        if "config_pick_orn_thrshld" in config: self.orn_th = config["config_pick_orn_thrshld"]  # rotation threshold to check if pick is successful

        ret1 = init_from_contactweb.loadContactWeb(agentp, config)
        ret2 = init_from_contactweb.handConfigurationFromContactWeb(
            agentp, config, ret1["p_pyb2cwebt0"], ret1["q_pyb2cwebt0"])
        
        # configurations from demonstration
        self.vd_theta = ret2["vd_theta"]
        self.vd_phi = ret2["vd_phi"]

        # reference motion parameters
        # for details, please see generateReferenceMotion()
        self._div = 5  # number of iterations at approach
        self.post_iters = 50  # number of iterations after contact
        self._back = ret2["back"]  # pregrasp-to-grasp distance

        # problem configurations
        # for details on "vd" "vg" "vw_l" see the corresponding variables of the same name
        self.problem_operators = ["vd", "vg", "vw", "vw_l"]  # hand translation/rotation search space
        
        self.vd = ret2["vd"]
        
        # set goal contact-web from config
        # pyb: pybullet origin
        # cwebt0: contact web origin/pose at episode start
        # p: position (p_a2b indicates position of b in a coordinate)
        # q: quaternion (q_a2b indicates transformation of b from a coordinate)
        # note, contact web may differ from center of the object (e.g., grasping handle)
        self.p_pyb2cwebt0 = ret1["p_pyb2cwebt0"]
        self.q_pyb2cwebt0 = ret1["q_pyb2cwebt0"]
        
        # gt: ground truth
        self.p_pyb2cwebt0_gt = ret1["p_pyb2cwebt0_gt"]
        self.q_pyb2cwebt0_gt = ret1["q_pyb2cwebt0_gt"]
        
        self.handQ_pybworld = ret2["handQ_pybworld"]
        self.handP_pybworld = ret2["handP_pybworld"]

        self.vg = utils.quat_mul_vec(self.handQ_pybworld, agentp.vg_l)
        self.vw_l = agentp.vw_l

        self.cweb = ret1["cweb"]

    def generateReferenceMotion(self, agentp: agentp_template.AgentProfileBase, envg: envg_interface.EnvironmentEngineInterface, config: dict, args: dict) -> int:
        self.initiation_js = agentp.couplingRule(agentp.preshape_joint)

        # values used for creating the reference motion
        js = [agentp.preshape_joint, agentp.joint]  # preshape and grasp finger configuration
        # hand translation: [approach, contact/grasp, continue-grasp]
        ts = [
            np.array(utils.uround(self.handP_pybworld)) - self._back/np.linalg.norm(self.vd)*self.vd,
            np.array(utils.uround(self.handP_pybworld)),
            np.array(utils.uround(self.handP_pybworld))
        ]
        
        # interpolate between pregrasp and grasp
        self.raw_trajectory = []
        self.raw_translation = []
        self.raw_rotation = []
        for i in range(self._div+1):
            t = float(i)/self._div
            jv = [None for x in range(len(agentp.reference_joints))]
            for j in range(len(agentp.reference_joints)):
                jv[j] = (1-t)*js[0][j] + t*js[1][j]
            self.raw_trajectory.append(jv)
            tv = [(1-t)*ts[0][0]+t*ts[1][0], (1-t)*ts[0][1]+t*ts[1][1], (1-t)*ts[0][2]+t*ts[1][2]]
            self.raw_translation.append(tv)
            rv = copy.deepcopy(self.handQ_pybworld)
            self.raw_rotation.append(rv)
        # reference motion (keep final state part)
        # use a slow-start strategy to make the learning easier?
        for i in range(self.post_iters+1):
            t = float(i)/self.post_iters
            jv = copy.deepcopy(self.raw_trajectory[-1])
            tv = [(1-t**2)*ts[1][0]+(t**2)*ts[2][0],
                  (1-t**2)*ts[1][1]+(t**2)*ts[2][1],
                  (1-t**2)*ts[1][2]+(t**2)*ts[2][2]]
            rv = copy.deepcopy(self.raw_rotation[-1])
            self.raw_trajectory.append(jv)
            self.raw_translation.append(tv)
            self.raw_rotation.append(rv)

        # record initial target state used later by appendConceptSpecificStates
        initial_target_state = envg.getComponentState("Pose", self.target_name)
        if initial_target_state is not None:  # this state is not used during execution so can be None
            self.initial_target_position_world = initial_target_state[0]
            self.initial_target_rotation_world = initial_target_state[1]

        return len(self.raw_trajectory)-1

    def anyInitiationAction(self, envg: envg_interface.EnvironmentEngineInterface) -> typing.Optional[typing.Tuple[tss_structs.ActionStruct, tss_structs.BodyIKStruct]]:
        r_state = copy.deepcopy(envg.getKinematicsState())
        r_state.eef_main.joint_states = self.initiation_js
        return (r_state, None)
    
    def setDemonstrationParameters(self, coordinate_p: typing.List[float], coordinate_q: typing.List[float], state: dict) -> dict:
        params = [self.vd_theta, self.vd_phi]  # int, int
        for i, c in enumerate(self.cweb.cmap):
            vf_position_world = copy.deepcopy(self.cweb.getContactPosition(i))
            vf_position_hand = vf_position_world - coordinate_p
            vf_position_hand = utils.quat_mul_vec(utils.quaternion_conjugate(coordinate_q), vf_position_hand)
            params += utils.uround(list(vf_position_hand))  # float, float, float
        state["observable_demo_parameters"] = params
        return state
    
    def appendConceptSpecificStates(self, state: dict, agentp: agentp_template.AgentProfileBase, envg: envg_interface.EnvironmentEngineInterface) -> dict:
        # object state as diff from initial_hand coordinate (for reward-only)
        t_state = envg.getComponentState("Pose", self.target_name)
        if t_state is not None:
            t_pos = t_state[0]
            t_ori = t_state[1]
            diff_orientation = utils.uround(utils.quaternion_multiply(
                t_ori, utils.quaternion_conjugate(self.initial_target_rotation_world)))
            state["hidden_target_state"] \
                = utils.uround(list(np.array(t_pos) - self.initial_target_position_world)) \
                + utils.uround(list(diff_orientation))
            target_euler_x, target_euler_y, target_euler_z = utils.euler_from_quaternion(diff_orientation, 'rxyz')
            state["hidden_target_state_euler"] \
                = [round(abs(target_euler_x), 6), round(abs(target_euler_y), 6), round(abs(target_euler_z), 6)]

        for i, l in enumerate(agentp.force_tip_links):
            state["observable_f"+str(i)+"_contact"] = envg.getPhysicsState("BinaryContact", self.target_name, {"link": l, "array_index": i, "agent_profile": agentp})

        # finger at contact check
        indicator = True  # will take AND later
        for i, l in enumerate(agentp.force_tip_links):
            indicator = indicator and state["observable_f"+str(i)+"_contact"]
        
        # contact position at contact-web check
        # don't check when doing sufficient condition evaluation
        if not state["terminated"]:
            j = 0
            for i in agentp.virtual_finger:
                tip_position_world = np.array(envg.getLinkState(agentp.position_tip_links[i])[0])
                # estimated web position (fixed at iteration 0)
                contact_pos_world = self.cweb.getContactPosition(j)
                j += 1
                dist = np.linalg.norm(tip_position_world - contact_pos_world)
                indicator = indicator and (dist < 0.05)  # 5cm
        if indicator: state["indicator"] = 1
        else: state["indicator"] = 0

        state["objectPosError"] = 100   # overwrite later using evaluateSufficientCondition()
        state["objectOrnError"] = 100   # overwrite later using evaluateSufficientCondition()
        state["reward_indicator"] = -1  # overwrite later using evaluateSufficientCondition()

        return state
    
    def evaluateStartingStateCondition(self, agentp: agentp_template.AgentProfileBase, observation: dict) -> bool:
        return True # assume grasp always starts from a successful state

    import types
    def getInitialRobotStateForTraining(self, config: dict, profiles: typing.Dict[str, types.ModuleType]) -> tss_structs.ActionStruct:
        agentp: agentp_template.AgentProfileBase = profiles[self.getAgentProfileString(config)]
        agentp.setParameters(config)
        js = agentp.couplingRule(agentp.preshape_joint)
        loadw = init_from_contactweb.loadContactWeb(agentp, config)
        ret = init_from_contactweb.handConfigurationFromContactWeb(agentp, config, loadw["p_pyb2cwebt0"], loadw["q_pyb2cwebt0"])
        trans = ret["handP_pybworld_pre"]
        rot = ret["handQ_pybworld_pre"]
        return tss_structs.ActionStruct(tss_structs.MultiLinkAction(agentp.command_joints, js, trans, rot), None, None, .1)

    ###########################################################################################
    ######################## below grasp task specific functions ##############################
    ###########################################################################################

    def updateWeb(self, vd):
        raise NotImplementedError("please define in child class")

    def getParameterizedTargetComponent(self, config: dict, cweb2obj: list) -> tss_structs.ComponentStruct:
        """
        Randomly generate an object mesh for training.
        """
        if "cweb2object_x" not in config:
            config["cweb2object_x"] = cweb2obj[0]
            config["cweb2object_y"] = cweb2obj[1]
            config["cweb2object_z"] = cweb2obj[2]
        p_pyb2cwebt0_gt = np.array([config["grasp_cweb0_position_x"], config["grasp_cweb0_position_y"], config["grasp_cweb0_position_z"]])
        q_pyb2cwebt0_gt = [config["grasp_cweb0_orientation_x"], config["grasp_cweb0_orientation_y"],
                           config["grasp_cweb0_orientation_z"], config["grasp_cweb0_orientation_w"]]

        # object scale randomization
        import pyvista
        config_scales = [config["object_scalex_noise"], config["object_scaley_noise"], config["object_scalez_noise"]]
        config_shapes = [config["object_shape1_noise"], config["object_shape2_noise"]]
        # use different names in case a model is generated in the same storage for parallel runs
        config_url = 'model_{}_{}_{}_{}_{}'.format(
            int(config_scales[0]*10000), int(config_scales[1]*10000), int(config_scales[2]*10000),
            int(config_shapes[0]*10000), int(config_shapes[1]*10000))
        mesh = pyvista.ParametricSuperEllipsoid(
            config_scales[0], config_scales[1], config_scales[2],
            config_shapes[0], config_shapes[1])
        mesh.save('logs/' + config_url + '.stl')

        p_cweb2modelt0 = np.array([config["cweb2object_x"], config["cweb2object_y"], config["cweb2object_z"]])

        p = p_pyb2cwebt0_gt + p_cweb2modelt0
        q = q_pyb2cwebt0_gt
        return tss_structs.ComponentStruct(self.target_name, ((p[0], p[1], p[2]), (q[0], q[1], q[2], q[3])),
            tss_structs.GeometryStruct(tss_structs.GeometryType.MESH, url=config_url, dimensions=(config_scales[0]*2, config_scales[1]*2, config_scales[2]*2)),
            tss_structs.ComponentType.RIGID, ignore_gravity=False)

    def getTargetCentricConstraintComponent(self, config: dict) -> tss_structs.ComponentStruct:
        """
        Reverse-generate a table environment from the randomly generated object mesh size and location.
        """
        env_pos_default = np.array([config["environment_position_x"], config["environment_position_y"], -config["environment_scale_z"]*.5])
        p_cweb2modelt0 = np.array([config["cweb2object_x"], config["cweb2object_y"], config["cweb2object_z"]])

        bottom_surface_pos = np.array([0.0, 0.0, -config["object_scalez_noise"]]) + np.array([config["grasp_cweb0_position_x"], 0.0, config["grasp_cweb0_position_z"]]) + p_cweb2modelt0

        p = env_pos_default + bottom_surface_pos
        q = (0, 0, config["environment_orientation_z"], config["environment_orientation_w"])
        return tss_structs.ComponentStruct("constraint", ((p[0], p[1], p[2]), q),
            tss_structs.GeometryStruct(tss_structs.GeometryType.FILE, url=config["environment"], dimensions=(1.0, 1.0, 1.0)),
            tss_structs.ComponentType.RIGID, ignore_gravity=True)
