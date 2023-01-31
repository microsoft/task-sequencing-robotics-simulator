# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------

import copy
import typing
import simulator.structs as tss_structs
import environment_engines.envg_template as envg_template

import environment_engines.ROS.ROS_robot_template as ROS_robot_template


class eROS(envg_template.Env):
    ROS_robot: ROS_robot_template.RobotBase
    visualizer: typing.Optional[envg_template.Env]

    def __init__(self, config: dict, role: tss_structs.EnvironmentEngineRole):
        # In order to avoid passing tons of robot specific parameters, import a python module.
        # Host and port is used to communicate to ROS via roslibpy.
        # For Windows, please run a rosbridge-server host on WSL2.
        print("tss_ROS expected fields: module: str, host: str, port: int")
        import importlib
        module = importlib.import_module(config["module"])
        self.ROS_robot = module.Robot(config["host"], config["port"], role, config)

    def reset(self):
        self.ROS_robot.resetSensors()

    def loadRobot(self, init_state: tss_structs.ActionStruct, body_ik_settings: typing.Optional[tss_structs.BodyIKStruct]=None) -> tss_structs.ActionStruct:
        if init_state.body is None and self.ROS_robot.role == tss_structs.EnvironmentEngineRole.KINEMATICS:
            if body_ik_settings is not None:
                ikmsg = self.ROS_robot.generateBodyikgoalmsg(body_ik_settings)
                ikresult = self.ROS_robot.solveik(ikmsg)
                if ikresult is None:  # failed to solve I.K.
                    init_state.error = True
                    init_state.body = self.ROS_robot.previous_body_command
                else:
                    init_state.body = ikresult
            init_state.timesec = self.ROS_robot.calculateCommandTime(init_state.body)
            self.ROS_robot.sendJointStates(init_state)
            print(init_state.body.joint_names, init_state.body.joint_states)

        self.ROS_robot.updateSensors()
        return init_state

    def update(self, cmd: tss_structs.ActionStruct, components: typing.List[tss_structs.ComponentStruct], body_ik_settings: typing.Optional[tss_structs.BodyIKStruct]=None) -> tss_structs.WorldStruct:
        if self.ROS_robot.role == tss_structs.EnvironmentEngineRole.KINEMATICS:
            if body_ik_settings is None: cmd.body = self.ROS_robot.previous_body_command
            else:
                ikmsg = self.ROS_robot.generateBodyikgoalmsg(body_ik_settings)
                ikresult = self.ROS_robot.solveik(ikmsg)
                if ikresult is None:  # failed to solve I.K.
                    cmd.error = True
                    cmd.body = self.ROS_robot.previous_body_command
                else:
                    cmd.body = ikresult
            cmd.timesec = self.ROS_robot.calculateCommandTime(cmd.body)
            cmd = self.ROS_robot.sendJointStates(cmd)  # real: should return values from the real robot after send

        self.ROS_robot.updateSensors()

        return tss_structs.WorldStruct(cmd, components)

    def loadComponents(self, components: typing.List[tss_structs.ComponentStruct]) -> typing.List[tss_structs.ComponentStruct]:
        return components  # does not load any components (they should already exist in the real world)

    def getKinematicsState(self) -> tss_structs.ActionStruct:
        if self.ROS_robot.latest_kinematic_state is None:
            raise Exception("tss_ROS: Kinematics state returned None!")
        return copy.deepcopy(self.ROS_robot.latest_kinematic_state)

    def getLinkState(self, link_name: str) -> typing.Tuple[typing.Tuple, typing.Tuple]:
        return self.ROS_robot.getLinkState(link_name)

    def exceptionalAccessToKinematics(self, body_ik_settings: typing.Optional[tss_structs.BodyIKStruct]):
        self.ROS_robot.exceptionalAccessToKinematics(body_ik_settings)

    def getPhysicsState(self, cmd: str, component_name: str, rest: typing.Optional[dict]=None) -> typing.Any:
        return self.ROS_robot.getPhysicsState(cmd, component_name, rest)

    def getSceneryState(self, cmd: str, component_name: str, rest: typing.Optional[dict]=None) -> typing.Any:
        return self.ROS_robot.getSceneryState(cmd, component_name, rest)

    def getSpawnComponents(self) -> typing.List[tss_structs.ComponentStruct]:
        return self.ROS_robot.detectSpawnComponents(self.visualizer)
