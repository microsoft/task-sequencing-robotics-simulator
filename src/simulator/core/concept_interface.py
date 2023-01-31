# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------

import typing
import copy
import json
import importlib
import numpy as np

import simulator.utils as utils

import simulator.constants as tss_constants
import simulator.structs as tss_structs
import simulator.core.concepts.grasp_active_force as grasp_active_force
import simulator.core.concepts.ptg11 as ptg11
import simulator.core.concepts.ptg12 as ptg12
import simulator.core.concepts.ptg13 as ptg13
import simulator.core.concepts.release as release

import simulator.core.envg_interface as envg_interface
import agent_profiles.agentp_template as agentp_template


class ConceptInterface:
    """
    Class managing task termination, task switching. Each task is referred as a "concept."
    """

    agentp: agentp_template.AgentProfileBase  # sensor descriptions, action space definitions

    def __init__(self, config_url: str):
        # load agent profiles
        with open(config_url) as f:
            config = json.load(f)
        self.agent_profiles = {}
        for ap in config["agent_profiles"]:
            self.agent_profiles[ap] = importlib.import_module(config["agent_profiles"][ap]).Profile()
        self.agentp = None

    def setConcept(self, config: dict) -> bool:
        """
        Select and set the task to run or the task with the components to spawn.
        config: settings including the task to run
        ---
        return: True if a valid task was set
        """
        # set parameters configured from demonstration/configured for training
        self.concept_type = tss_constants.ConceptType(config["concept"])

        if self.concept_type == tss_constants.ConceptType.CONCEPT_GRASP_ACTIVE_FORCE:
            print("LOADING SIM ACTIVE GRASP")
            self.concept = grasp_active_force.Concept()
        elif self.concept_type == tss_constants.ConceptType.CONCEPT_PTG11:
            print("LOADING SIM PTG11")
            self.concept = ptg11.Concept()
        elif self.concept_type == tss_constants.ConceptType.CONCEPT_PTG12:
            print("LOADING SIM PTG12")
            self.concept = ptg12.Concept()
        elif self.concept_type == tss_constants.ConceptType.CONCEPT_PTG13:
            print("LOADING SIM PTG13")
            self.concept = ptg13.Concept()
        elif self.concept_type == tss_constants.ConceptType.CONCEPT_RELEASE:
            print("LOADING SIM RELEASE")
            self.concept = release.Concept()
        elif self.concept_type == tss_constants.ConceptType.CONCEPT_EXECUTE:
            print("EXECUTE SEQUENCE")
            self.setConcept({"concept": config["spawn_from_task"]})
        else:
            print('invalid concept!')
            return False
        return True

    def initStates(self, envg: envg_interface.EnvironmentEngineInterface, config: dict, args: dict) -> dict:
        """
        Initiate the task.
        envg:   environment(s) holding the current robot state and component states
        config: parameters for training/executing the task (e.g., trajectory parameters, random noise)
        args:   parameters passed from the state of a previous task (e.g., previous joint angle command)
        ---
        return: initial task state
        """
        self.setConcept(config)

        # A programmed task concept may return an empty string and use the agent profile of a previously executed task.
        # When the first task is a programmed concept, will use the profile set as "default" in the tss config file. 
        ap_string = self.concept.getAgentProfileString(config)
        if ap_string != "":
            self.agentp = self.agent_profiles[ap_string]
        if self.agentp is None:
            if "default" not in self.agent_profiles:
                raise Exception("Please set a 'default' in 'agent_profiles', required if the sequence begins with a programmed concept!")
            self.agentp = self.agent_profiles["default"]

        self.concept.init(self.agentp, config, args) 
        self.max_step = self.concept.generateReferenceMotion(self.agentp, envg, config, args)

        inia = self.concept.anyInitiationAction(envg)
        if inia is not None: envg.callEnvironmentUpdatePipeline(inia[0], inia[1])

        # states are represented relative to the beginning of the task
        initial_robot_state = envg.getKinematicsState()
        self.initial_hand_position_world = list(initial_robot_state.eef_main.b_position)
        self.initial_hand_rotation_world = list(initial_robot_state.eef_main.b_orientation)
        
        self.pt = 0
        self.action_values = self.concept.getActionSpace(self.agentp)

        return self._getStateVector(envg, False)

    def spawnComponents(self, episode_config: dict, envg: envg_interface.EnvironmentEngineInterface, start_robot_state: typing.Optional[tss_structs.ActionStruct], components: typing.Optional[typing.List[tss_structs.ComponentStruct]]) -> bool:
        """
        Initiate the world by spawing components.
        episode_config:    parameters for component-spawning (e.g., size of the spawning object)
        envg:              environment(s) to hold the state of the world
        start_robot_state: will load a preset state from the task definitions and parameters if is None
        components:        will load a preset list from the task definitions and parameters if is None
        ---
        return: True if world initiation was successful
        """
        if start_robot_state is None:
            start_robot_state = self.concept.getInitialRobotStateForTraining(episode_config, self.agent_profiles)
            goal = tss_structs.BRDIKGoalStruct(start_robot_state.eef_main, None, "", "", 1.0) 
            body_ik_settings = tss_structs.BodyIKStruct(self.concept_type, goal, self.concept.ikrest)
        else: body_ik_settings = None  # joint angles should be stored in the start state w/o requiring extra calculation

        if components is None: components = self.concept.getSpawnComponentsForTraining(episode_config)

        return envg.callEnvironmentLoadPipeline(start_robot_state, components, body_ik_settings)
        
    def iterateOnce(self, envg: envg_interface.EnvironmentEngineInterface, action: typing.Optional[dict]=None, brain_url: str="http://localhost:5000") -> dict:
        """
        Execution of a task iteration.
        envg:      environment(s) holding the current robot state and component states
        action:    actions returned from the Bonsai platform (will set actions from an exported brain or a programmed task if None)
        brain_url: url of the exported brain if hosted as a web app
        ---
        return: information from the simulator
        """
        if action is None: action = self.concept.getAction(self._getStateVector(envg, False), self.agentp, brain_url)
        if action["terminate"]: return {"terminated": True}  # do not send action if action is terminate

        self._setActionsK(action)
        (js, tv, rv) = self._getConfigurationK()
        cmd = tss_structs.ActionStruct(tss_structs.MultiLinkAction(self.agentp.command_joints, js, tv, rv), None, None, .1)

        goal = tss_structs.BRDIKGoalStruct(cmd.eef_main, None, "", "", self.pt/self.max_step)
        body_ik_settings = tss_structs.BodyIKStruct(self.concept_type, goal, self.concept.ikrest)
        success = envg.callEnvironmentUpdatePipeline(cmd, body_ik_settings)

        js_return = envg.getKinematicsState().eef_main.joint_states  # mainly used for logging purposes
        return {
            "engine_update_success": success,
            "done": (not self._iterate_step()),
            "joints_command": copy.deepcopy(js),
            "translation_command": list(copy.deepcopy(tv)),
            "rotation_command": list(copy.deepcopy(rv)),
            "joints_actual": copy.deepcopy(js_return),
            "observation": self._getStateVector(envg, False),
            "action": action,
            "terminated": False
        }

    def _getStateVector(self, envg: envg_interface.EnvironmentEngineInterface, done: bool) -> dict:
        state = {}
        state = self.concept.setDemonstrationParameters(self.initial_hand_position_world, self.initial_hand_rotation_world, state)
        
        h_state = envg.getKinematicsState()
        h_pos = h_state.eef_main.b_position
        h_ori = h_state.eef_main.b_orientation

        q = utils.quaternion_conjugate(self.initial_hand_rotation_world)
        v = np.array(h_pos) - self.initial_hand_position_world
        state["observable_hand_position"] = utils.uround(utils.quat_mul_vec(q, v))
        state["observable_hand_orientation"] = utils.uround(utils.quaternion_multiply(
            h_ori, utils.quaternion_conjugate(self.initial_hand_rotation_world)))
        
        state["initial_hand_position_world"] = self.initial_hand_position_world
        state["initial_hand_rotation_world"] = self.initial_hand_rotation_world
        
        state["observable_finger_state"] = utils.uround(h_state.eef_main.joint_states)
        for i, l in enumerate(self.agentp.position_tip_links):
            tip_position_world = envg.getLinkState(self.agentp.position_tip_links[i])[0]
            tip_position_hand = np.array(tip_position_world) - self.initial_hand_position_world
            q = utils.quaternion_conjugate(self.initial_hand_rotation_world)
            tip_position_hand = utils.quat_mul_vec(q, tip_position_hand)
            state["observable_f"+str(i)+"_position"] = utils.uround(tip_position_hand)
        
        state["observable_timestep"] = self.pt  # "iteration" in Bonsai, for reward definition etc.
            
        if done: state['terminated'] = 1
        else: state['terminated'] = 0

        state = self.concept.appendConceptSpecificStates(state, self.agentp, envg)

        return state
    
    # @return True if continue episode
    def _iterate_step(self) -> int:
        self.pt += 1
        if self.pt >= self.max_step: return 0
        return 1

    def _setActionsK(self, action: dict):
        for j in action: self.action_values[j] = action.get(j)

    def _getConfigurationK(self) -> typing.Tuple[typing.List[float], typing.List[float], typing.List[float]]:
        """
        Returns the new joint angles and end-effector pose from the actions.
        ---
        return: (joint angles, position x-y-z, orientation x-y-z-w)
        """

        # By default, every task calculates a feed-forward initial trajectory.
        # The trajectory is then modified using the actions returned from Bonsai.

        jv = [None for x in range(len(self.agentp.reference_joints))]
        for j, jname in enumerate(self.agentp.reference_joints):
            jv[j] = copy.deepcopy(self.concept.raw_trajectory[self.pt][j])
            if jname in self.action_values:
                jv[j] += self.action_values[jname]
            jv[j] = min(jv[j], self.agentp.max_values[jname])
            jv[j] = max(jv[j], self.agentp.min_values[jname])
        jv = self.agentp.couplingRule(jv)

        tv = np.array(copy.deepcopy(self.concept.raw_translation[self.pt]))
        tv = self.concept.updateReferenceTranslation(tv, self.pt, self.action_values)
        if "vg" in self.concept.problem_operators:
            tv += self.action_values["vg"]*np.array(self.concept.vg)
        if "vd" in self.concept.problem_operators:
            tv += self.action_values["vd"]*np.array(self.concept.vd)

        rv = copy.deepcopy(self.concept.raw_rotation[self.pt])
        rv = self.concept.updateReferenceRotation(rv, self.pt, self.action_values)
        rv_l = [0., 0., 0., 1.]
        if "vw_l" in self.concept.problem_operators:
            vw_l = utils.quat_mul_vec(rv_l, self.concept.vw_l)
            s = np.sin(self.action_values["vw_l"]*.5)*np.array(vw_l)
            c = np.cos(self.action_values["vw_l"]*.5)
            rv_l = utils.quaternion_multiply([s[0], s[1], s[2], c], rv_l)
        rv = utils.quaternion_multiply(rv, rv_l)

        if "vw" in self.concept.problem_operators:
            vw = utils.quat_mul_vec(rv, self.concept.vw_l)
            tv += self.action_values["vw"] * np.array(vw)

        return (jv, tv, rv)
        