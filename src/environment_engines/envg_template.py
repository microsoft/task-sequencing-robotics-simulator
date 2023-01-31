# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------

from __future__ import annotations
from abc import abstractmethod
import typing
import simulator.structs as tss_structs


class Env:
    """
    A template class for connecting an arbitrary engine to TSS. 
    """

    def __init__(self, config: dict, role: tss_structs.EnvironmentEngineRole):
        """
        config: engine specific config parameters to pass from the TSS config file
        role:   the role of the engine (e.g., physics), used if an engine can switch among different roles
        """
        pass

    @abstractmethod
    def reset(self):
        """
        Any initiation operations for the engine.
        """
        raise NotImplementedError("not implemented error")

    @abstractmethod
    def loadRobot(self, init_state: tss_structs.ActionStruct, body_ik_settings: typing.Optional[tss_structs.BodyIKStruct]=None) -> tss_structs.ActionStruct:
        """
        Load the robot inside the engine world.
        init_state:       initial state of the robot on loading
        body_ik_settings: used if arm joints are calculated instead of from learned actions
        """
        raise NotImplementedError("not implemented error")

    @abstractmethod
    def loadComponents(self, components: typing.List[tss_structs.ComponentStruct]) -> typing.List[tss_structs.ComponentStruct]:
        """
        Load the components inside the engine world if any.
        components: list of components to load into the world
        ---
        return: list of components with updates by the engine to pass to an engine later in the pipeline
        """
        print("not implemented warning")
        return components

    @abstractmethod
    def update(self, cmd: tss_structs.ActionStruct, components: typing.List[tss_structs.ComponentStruct], body_ik_settings: typing.Optional[tss_structs.BodyIKStruct]=None) -> tss_structs.WorldStruct:
        """
        Update the robot and component states inside the engine world.
        cmd:              action updates
        components:       component updates
        body_ik_settings: used if arm joints are calculated instead of from learned actions
        ---
        return: updates by the engine to pass to an engine later in the pipeline
        """
        raise NotImplementedError("not implemented error")

    @abstractmethod
    def getKinematicsState(self) -> tss_structs.ActionStruct:
        # See envg_interface.py for details.
        print("not implemented warning")

    @abstractmethod
    def getLinkState(self, link_name: str) -> typing.Tuple[typing.Tuple, typing.Tuple]:
        # See envg_interface.py for details.
        print("not implemented warning")

    @abstractmethod
    def exceptionalAccessToKinematics(self, body_ik_settings: typing.Optional[tss_structs.BodyIKStruct]):
        """
        (Dev) Update kinematic states outside pipeline call (e.g., moving of the robot's neck for obtaining a runtime parameter).
        """
        print("not implemented warning")

    @abstractmethod
    def getPhysicsState(self, cmd: str, component_name: str, rest: typing.Optional[dict]=None) -> typing.Any:
        # See envg_interface.py for details.
        print("not implemented warning")

    @abstractmethod
    def getSceneryState(self, cmd: str, component_name: str, rest: typing.Optional[dict]=None) -> typing.Any:
        # See envg_interface.py for details.
        print("not implemented warning")

    @abstractmethod
    def getSpawnComponents(self) -> typing.List[tss_structs.ComponentStruct]:
        # See envg_interface.py for details.
        print("not implemented warning")

    @abstractmethod
    def postProcess(self, scenery_engine: Env):
        """
        Update post process engine during pipeline update. Use if process occurs regardless of task/concept situation (e.g., image noise filter).
        """
        print("not implemented warning")

    @abstractmethod
    def getPostProcessState(self, cmd: str, scenery_engine: Env, rest: typing.Optional[dict]=None) -> typing.Any:
        """
        Update post process engine on function call. Use if process is specifc to a certain task/concept situation (e.g., task-specific features).
        """
        print("not implemented warning")

    @abstractmethod
    def recordStart(self, config: dict, training: bool):
        print("not implemented warning")

    @abstractmethod
    def recordEnd(self, filename: str, flush: bool):
        print("not implemented warning")
