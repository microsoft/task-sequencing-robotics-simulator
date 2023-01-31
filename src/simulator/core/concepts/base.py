# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------

import typing
import types
import simulator.structs as tss_structs
import simulator.core.envg_interface as envg_interface
import agent_profiles.agentp_template as agentp_template


class ConceptBase:
    """
    A template class for designing an arbitrary task concept in TSS. 
    """

    problem_operators: list=[]
    ikrest: tss_structs.BodyIKRest=tss_structs.BodyIKRest({})

    def __init__(self):
        """
        Please use init() to do any initiation.
        """
        pass

    def getAgentProfileString(self, config: dict):
        """
        Default returns an empty string, meaning, use the same profiles as the previous concept.
        ---
        return: the name of the agent profile to use for this concept
        """
        return ""

    def getActionSpace(self, agentp: agentp_template.AgentProfileBase) -> dict:
        """
        Define the action space of the task. Override if needed.
        """
        action_values = {}
        for a in self.problem_operators:
            action_values[a] = 0.0
        action_values["terminate"] = 0
        return action_values

    def init(self, agentp: agentp_template.AgentProfileBase, config: dict, args: dict):
        """
        Init the task.
        agentp: the agent profile to use (sensor definitions etc.)
        config: task parameters (can also include "ikrest" to setup task IK settings)
        args:   parameters that are propagated from a previous task state
        """
        if "ikrest" in config: self.ikrest = tss_structs.BodyIKRest(config["ikrest"])

    def generateReferenceMotion(self, agentp: agentp_template.AgentProfileBase, envg: envg_interface.EnvironmentEngineInterface, config: dict, args: dict) -> int:
        """
        Generate a feed-forward trajectory. A Bonsai brain will modify this trajectory using state feedback.
        agentp: the agent profile (sensor definitions etc.)
        envg:   access to the states of different engines
        config: task parameters
        args:   parameters that are propagated from a previous task state
        ---
        return: trajectory length / number of max iterations
        """
        # self.raw_trajectory = []  # make sure to define for each concept
        # self.raw_translation = [] # make sure to define for each concept
        # self.raw_rotation = []    # make sure to define for each concept
        raise NotImplementedError("please define")

    def anyInitiationAction(self, envg: envg_interface.EnvironmentEngineInterface) -> typing.Optional[typing.Tuple[tss_structs.ActionStruct, tss_structs.BodyIKStruct]]:
        """
        Add actions in-between tasks if any (e.g., sending the pre-grasp fingers at start of grasp).
        envg: access to the states of different engines
        ---
        return: in-between action
        """
        # only envg is passed, to use agentp etc., set variables during generateReferenceMotion()
        return None
    
    def setDemonstrationParameters(self, coordinate_p: typing.List[float], coordinate_q: typing.List[float], state: dict) -> dict:
        """
        Set task specific static parameters to states.
        coordinate_p: state position origin (so that states are expressed relative to t=0)
        coordinate_q: state rotation origin (so that states are expressed relative to t=0)
        state:        current state
        ---
        return: updated state
        """
        raise NotImplementedError("please define")  # please return modified state
        
    def appendConceptSpecificStates(self, state: dict, agentp: agentp_template.AgentProfileBase, envg: envg_interface.EnvironmentEngineInterface) -> dict:
        """
        Set task specific non-static parameters to states.
        state:  current state
        agentp: the agent profile (sensor definitions etc.)
        envg:   access to states of different engines
        ---
        return: updated state
        """
        return state
        
    def evaluateSufficientCondition(self, agentp: agentp_template.AgentProfileBase, envg: envg_interface.EnvironmentEngineInterface, jcmd_last: typing.List[float], tcmd_last: typing.List[float], rcmd_last: typing.List[float]) -> dict:
        """
        Evaluate the task performance during training by using a post-sequence of tasks.
        agentp:    the agent profile
        envg:      access to the states of different engines
        jcmd_last: last executed joint command
        tcmd_last: last executed position command
        rcmd_last: last executed rotation command
        ---
        return: updated state
        """
        raise NotImplementedError("please define")

    def evaluateStartingStateCondition(self, agentp: agentp_template.AgentProfileBase, observation: dict) -> bool:
        """
        Check if the starting state of a task is valid.
        agentp:      the agent profile
        observation: current state of the world
        ---
        return: True if valid
        """
        raise NotImplementedError("please define")
        
    def updateReferenceTranslation(self, tv: typing.List[float], pt: int, action_values: dict) -> typing.List[float]:
        """
        Using action decisions, update the position of a trajectory point.
        tv:            trajectory point position before update
        pt:            current iteration
        action_values: action decisions returned by Bonsai brain
        ---
        return: updated trajectory point position
        """
        return tv
        
    def updateReferenceRotation(self, rv: typing.List[float], pt: int, action_values: dict) -> typing.List[float]:
        """
        Using action decisions, update the rotation of a trajectory point.
        rv:            trajectory point rotation before update (quaternion)
        pt:            current iteration
        action_values: action decisions returned by Bonsai brain
        ---
        return: updated trajectory point rotation (quaternion)
        """
        return rv

    def getAction(self, observation: dict, agentp: agentp_template.AgentProfileBase, brain_url: str="", client_id: str="") -> dict:
        """
        Call to a trained Bonsai brain or return programmed actions if a programmed task.
        observation: current states
        agentp:      the agent profile
        brain_url:   url to the trained brain if any
        client_id:   client id if any
        """
        raise NotImplementedError("please define")  # please return dict

    def getSpawnComponentsForTraining(self, config: dict) -> typing.List[tss_structs.ComponentStruct]:
        """
        Definition of the environment when training the task.
        config: spawn configurations if any
        ---
        return: list of components required for training
        """
        raise NotImplementedError("please define")

    def getInitialRobotStateForTraining(self, config: dict, profiles: typing.Dict[str, types.ModuleType]) -> tss_structs.ActionStruct:
        """
        Definition of the initial robot state when training the task.
        config: configurations for defining the state if any
        ---
        return: initial state of the robot for training
        """
        raise NotImplementedError("please define")