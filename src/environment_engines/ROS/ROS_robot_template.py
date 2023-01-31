# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------

import typing
import roslibpy

import simulator.structs as tss_structs
import environment_engines.envg_template as envg_template


class RobotBase:
    """
    A template class for connecting an arbitrary ROS robot to TSS. 
    """

    latest_kinematic_state: typing.Optional[tss_structs.ActionStruct]
    previous_body_command: typing.Optional[tss_structs.MultiLinkAction]
    role: tss_structs.EnvironmentEngineRole

    # please define in child.setDescriptions()
    map_link: str
    eef_link: str
    base_link: str
    camera_link: str

    # debug and optimization
    VERBOSE: bool
    OPTIMIZE: bool

    def __init__(self, host: str, port: int, role: tss_structs.EnvironmentEngineRole, config: dict):
        """
        host:   roslibpy host
        port:   roslibpy port
        role:   the role of the engine (e.g., kinematics)
        config: settings from the TSS config file
        """
        if "VERBOSE" in config["dbg_flags"]:
            print("DBG MODE ----- VERBOSE flag")
            self.VERBOSE = True
        else: self.VERBOSE = False

        if "OPTIMIZE" in config["dbg_flags"]:
            print("RUN MODE ----- OPTIMIZE")
            self.OPTIMIZE = True
        else: self.OPTIMIZE = False

        self.rosclient = roslibpy.Ros(host, port)
        self.rosclient.run()

        if not self.rosclient.is_connected:
            raise Exception("ROS components error: did you run 'roslaunch rosbridge_server rosbridge_websocket.launch'?")

        self.role = role
        self.previous_body_command = None
        self.use_ros_states = config["use_ros_states"]
        self.setDescriptions()
        self.initListeners()  # allow getting real robot states if any
        if self.use_ros_states:
            import time
            time.sleep(1.)  # wait connections
            if role == tss_structs.EnvironmentEngineRole.KINEMATICS: self.updateActualJointStates()  # sets self.latest_kinematic_state to current actual states
        else: self.latest_kinematic_state = None

    def setDescriptions(self):
        """
        Init robot-specific parameters, called by __init__.
        """
        raise NotImplementedError("not implemented error")

    def tomsg(self, action: tss_structs.ActionStruct) -> dict:
        """
        Merge eef and body actions into a single ROS msg stream.
        action: action from TSS
        ---
        return: roslibpy compatible action message
        """
        raise NotImplementedError("not implemented error")

    def tobodyaction(self, res: dict) -> tss_structs.MultiLinkAction:
        """
        IK result (arm chain joint space) to body action (full chain e.g., include head, base etc.).
        res: calculation results from ROS
        ---
        return: TSS action structure
        """
        raise NotImplementedError("not implemented error")

    def initListeners(self):
        """
        Connect to ROS listeners (sensor msgs etc.).
        """
        raise NotImplementedError("not implemented error")

    def generateBodyikgoalmsg(self, body_ik_settings: tss_structs.BodyIKStruct) -> dict:
        """
        Generate a ROS IK message.
        body_ik_settings: IK settings to solve
        ---
        return: roslibpy compatible IK goal message
        """
        raise NotImplementedError("not implemented error")

    def resetSensors(self):
        """
        Reset sensors on the robot (e.g., force sensors).
        """
        raise NotImplementedError("not implemented error")

    def updateSensors(self):
        """
        Update sensors on the robot (e.g., force sensors).
        """
        raise NotImplementedError("not implemented error")

    def getLinkState(self, link_name: str) -> typing.Tuple[typing.Tuple, typing.Tuple]:
        # See envg_template.py for details.
        if self.OPTIMIZE:
            trans, rot = self.getLinkStateOptimized(self.map_link, link_name)
        else:
            trans, rot, error = self.getTF(self.map_link, link_name)  # very slow
            if error: raise Exception("failed to obtain TF %s to %s! fatal error, aborting" % (self.map_link, link_name))
        return (trans, rot)

    def getLinkStateOptimized(self, from_link: str, to_link: str, joint_states: typing.Optional[typing.List[float]]=None) -> typing.Tuple[typing.Tuple[float], typing.Tuple[float]]:
        """
        Return the position and orientation of a specified link using internal urdf calculation and without TF.
        from_link:    origin of states
        to_link:      target link
        joint_states: kinematic state of the robot if any
        ---
        return: current position and orienation of the robot link
        """
        raise NotImplementedError("not implemented error")

    def exceptionalAccessToKinematics(self, body_ik_settings: typing.Optional[tss_structs.BodyIKStruct]):
        # See envg_template.py for details.
        raise NotImplementedError("not implemented error")

    def getPhysicsState(self, cmd: str, component_name: str, rest: typing.Optional[dict]=None) -> typing.Any:
        # See envg_template.py for details.
        raise NotImplementedError("not implemented error")

    def getSceneryState(self, cmd: str, component_name: str, rest: typing.Optional[dict]=None) -> typing.Any:
        # See envg_template.py for details.
        raise NotImplementedError("not implemented error")

    def detectSpawnComponents(self, mixed_sim: envg_template.Env) -> typing.List[tss_structs.ComponentStruct]:
        """
        Detect components in the real world using the robot's vision.
        mixed_sim: a virtual world to express static environments (e.g., a map of the room) if any
        ---
        return: list of components to spawn
        """
        raise NotImplementedError("not implemented error")

    def solveik(self, ikmsg: dict) -> tss_structs.MultiLinkAction:
        """
        Call the inverse kinematics solver to obtain results.
        ikmsg: the IK goal message compatible with roslibpy
        ---
        return: TSS action structure
        """
        raise NotImplementedError("not implemented error")

    def calculateCommandTime(self, action: tss_structs.MultiLinkAction) -> float:
        """
        Calculate the speed of the command based on the actions.
        action: the newly commanding action
        ---
        float: time in seconds
        """
        raise NotImplementedError("not implemented error")

    def sendJointStates(self, action: tss_structs.ActionStruct) -> tss_structs.ActionStruct:
        """
        Send joints to the real robot.
        action: the command to send to the robot
        ---
        return: latest state of the robot after sending the command
        """
        raise NotImplementedError("not implemented error")

    def updateActualJointStates(self):
        """
        Update self.latest_kinematic_state using values from the ROS listener callbacks.
        """
        raise NotImplementedError("not implemented error")

    def getTF(self, from_frame: str, to_frame: str) -> typing.Tuple[typing.Tuple[float], typing.Tuple[float], int]:
        """
        Return the position and orientation of a specified link using ROS TF. Not suitable for real-time control.
        from_frame:    origin frame
        to_frame:      target frame
        ---
        return: (position, orientation, error)
        """
        raise NotImplementedError("not implemented error")
