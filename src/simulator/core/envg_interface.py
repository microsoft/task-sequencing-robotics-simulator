# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------

import copy
import typing
import json

import simulator.structs as tss_structs


class EnvironmentEngineInterface:
    """
    Class managing the robot state and component states in multiple simulation worlds (engines).
    """

    def __init__(self, config_url: str):
        """
        config_url: url of the TSS config file (see the json file under samples)
        """

        with open(config_url) as f:
            config = json.load(f)

        if "general" not in config:
            raise Exception("envg config error: must have the 'general' field")
        self.general_config = config["general"]

        # To use flags in concept_interface, access envg.dbg_flags in concept_interface.py.
        # To use flags in environment engines, access config["dbg_flags"] in __init__.
        if "DBG_FLAGS" in self.general_config: self.dbg_flags = self.general_config["DBG_FLAGS"]
        else: self.dbg_flags = []

        if "environment_engines" not in config:
            raise Exception("envg config error: must have the 'environment_engines' field")
        envg_config = config["environment_engines"]

        if "kinematics" not in envg_config:
            raise Exception("envg config error: must have the 'environment_engines/kinematics' field, set to 'null' if not used")
        if "physics" not in envg_config:
            raise Exception("envg config error: must have the 'environment_engines/physics' field, set to 'null' if not used")
        if "scenery" not in envg_config:
            raise Exception("envg config error: must have the 'environment_engines/scenery' field, set to 'null' if not used")
        if "post_process" not in envg_config:
            raise Exception("envg config error: must have the 'environment_engines/post_process' field, set to 'null' if not used")
        
        self.kinematics_env = self._getEngineByConfig("kinematics", envg_config["kinematics"], tss_structs.EnvironmentEngineRole.KINEMATICS)
        self.physics_env = self._getEngineByConfig("physics", envg_config["physics"], tss_structs.EnvironmentEngineRole.PHYSICS)
        self.scenery_env = self._getEngineByConfig("scenery", envg_config["scenery"], tss_structs.EnvironmentEngineRole.SCENERY)
        self.post_process_env = self._getEngineByConfig("post_process", envg_config["post_process"], tss_structs.EnvironmentEngineRole.POST_PROCESS)

        if "use_robot_state" not in self.general_config:
            raise Exception("envg config error: must have the 'general/use_robot_state' field set to either 'kinematics/physics/scenery'")
        if self.general_config["use_robot_state"] == "kinematics": self.robot_state_env = self.kinematics_env
        elif self.general_config["use_robot_state"] == "physics": self.robot_state_env = self.physics_env
        elif self.general_config["use_robot_state"] == "scenery": self.robot_state_env = self.scenery_env
        else:
            raise Exception("envg config error: 'general/use_robot_state' field must be either 'kinematics/physics/scenery'")

        self.latest_component_states = []

    def _getEngineByConfig(self, ename: str, settings: dict, role: tss_structs.EnvironmentEngineRole):
        if settings is None: return None
        if "engine" not in settings:
            raise Exception("envg config error: must have the 'environment_engines/%s/engine' field to specify the engine to use" % ename)

        # propogate dbg_flags
        settings["dbg_flags"] = self.dbg_flags

        if settings["engine"] == "ROS":
            import environment_engines.ROS.tss_ROS as tss_ROS
            return tss_ROS.eROS(settings, role)
        elif settings["engine"] == "PyBullet":
            import environment_engines.PyBullet.tss_PyBullet as tss_PyBullet
            return tss_PyBullet.ePyBullet(settings, role)
        else: raise Exception("envg config error: supported engines are 'PyBullet' got %s" % settings["engine"])

    def getSpawnComponentsFromScenery(self) -> typing.List[tss_structs.ComponentStruct]:
        """
        (Dev) Get a list of components to spawn from the scenery engine in order to spawn in other engines.
        ---
        return: list of components to spawn
        """
        return self.scenery_env.getSpawnComponents()

    def getInitialRobotState(self) -> tss_structs.ActionStruct:
        """
        (Dev) Get the initial state of the robot from the kinematics or scenery engine to pass to other engines.
        ---
        return: kinematics state of the robot
        """
        if "use_initial_robot_state" not in self.general_config:
            raise Exception("envg config error: must have 'general/use_initial_robot_state' field set to either 'kinematics/scenery'")
        if self.general_config["use_initial_robot_state"] == "kinematics": return self.kinematics_env.getKinematicsState()
        elif self.general_config["use_initial_robot_state"] == "scenery": return self.scenery_env.getKinematicsState()
        else:
            raise Exception("envg config error: 'general/use_initial_robot_state' field must be either 'kinematics/physics/scenery'")

    def callEnvironmentLoadPipeline(self, start_robot_state: tss_structs.ActionStruct, components: typing.List[tss_structs.ComponentStruct], body_ik_settings: typing.Optional[tss_structs.BodyIKStruct]=None) -> bool:
        """
        Load components and initial robot state to all engines.
        start_robot_state: the initial state of the robot
        components:        the components to spawn and their initial states
        body_ik_settings:  used if arm joints are calculated instead of from learned actions
        ---
        return: True if load was successful
        """
        print("callEnvironmentLoadPipeline")

        if self.kinematics_env is not None: self.kinematics_env.reset()
        if self.physics_env is not None: self.physics_env.reset()
        if self.scenery_env is not None: self.scenery_env.reset()
        if self.post_process_env is not None: self.post_process_env.reset()

        if self.kinematics_env is not None: components = self.kinematics_env.loadComponents(components)
        if self.physics_env is not None: components = self.physics_env.loadComponents(components)
        if self.scenery_env is not None: components = self.scenery_env.loadComponents(components)
        self.latest_component_states = copy.deepcopy(components)

        if self.kinematics_env is not None: start_robot_state = self.kinematics_env.loadRobot(start_robot_state, body_ik_settings)
        if self.physics_env is not None: start_robot_state = self.physics_env.loadRobot(start_robot_state, body_ik_settings)
        if self.scenery_env is not None: start_robot_state = self.scenery_env.loadRobot(start_robot_state, body_ik_settings)
        return (not start_robot_state.error)  # returns False on I.K. failure

    def callEnvironmentUpdatePipeline(self, robot_state: tss_structs.ActionStruct, body_ik_settings: typing.Optional[tss_structs.BodyIKStruct]=None) -> bool:
        """
        Update components and robot state of all engines.
        robot_state:      updated robot state
        body_ik_settings: used if arm joints are calculated instead of from learned actions
        ---
        return: True if update was successful
        """
        print("callEnvironmentUpdatePipeline")
        world_state = tss_structs.WorldStruct(robot_state, self.latest_component_states)

        if self.kinematics_env is not None: world_state = self.kinematics_env.update(world_state.robot_state, world_state.component_states, body_ik_settings)
        if self.physics_env is not None: world_state = self.physics_env.update(world_state.robot_state, world_state.component_states, body_ik_settings)
        if self.scenery_env is not None: world_state = self.scenery_env.update(world_state.robot_state, world_state.component_states, body_ik_settings)
        if self.post_process_env is not None: self.post_process_env.postProcess(self.scenery_env)

        self.latest_component_states = copy.deepcopy(world_state.component_states)
        return (not world_state.robot_state.error)  # returns False on I.K. failure

    def getKinematicsState(self) -> tss_structs.ActionStruct:
        """
        Return the robot state from the engine specified in "use_robot_state" field in config.
        ---
        return: current state of the robot
        """
        return self.robot_state_env.getKinematicsState()

    def getLinkState(self, link_name: str) -> typing.Tuple[typing.Tuple, typing.Tuple]:
        """
        Return a robot's link state from the engine specified in "use_robot_state" field in config.
        link_name: name of the link
        ---
        return: current position and orienation of the robot link
        """
        return self.robot_state_env.getLinkState(link_name)

    def getComponentState(self, cmd: str, component_name: str) -> typing.Any:
        """
        Return the state of a component.
        cmd:            request string
        component_name: name of the component
        ---
        return: the requested state
        """
        for c in self.latest_component_states:
            if c.name == component_name:
                if cmd == "Pose": return c.pose
        return None  # if component is not found

    def getPhysicsState(self, cmd: str, component_name: str, rest: typing.Optional[dict]=None) -> typing.Any:
        """
        Return a state obtainable from the physics engine.
        cmd:            request string
        component_name: name of the component if any
        rest:           request parameters if any
        ---
        return: the requested state
        """
        return self.physics_env.getPhysicsState(cmd, component_name, rest)
    
    def getSceneryState(self, cmd: str, component_name: str, rest: typing.Optional[dict]=None) -> typing.Any:
        """
        Return a state obtainable from the scenery (e.g., rendering) engine.
        cmd:            request string
        component_name: name of the component if any
        rest:           request parameters if any
        ---
        return: the requested state
        """
        return self.scenery_env.getSceneryState(cmd, component_name, rest)

    def getPostProcessState(self, cmd: str, rest: typing.Optional[dict]=None) -> typing.Any:
        """
        Return a state obtainable from the post process engine.
        cmd:  request string
        rest: request parameters if any
        ---
        return: the requested state
        """
        return self.post_process_env.getPostProcessState(cmd, self.scenery_env, rest)

    def startRecording(self, config: dict, training: bool):
        if self.physics_env is not None: self.physics_env.recordStart(config, training)

    def endRecording(self, filename: str, flush: bool=False):
        if self.physics_env is not None: self.physics_env.recordEnd(filename, flush)