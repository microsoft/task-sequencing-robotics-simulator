#!/usr/bin/env python3

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------

import sys
import json
import os
import copy
from datetime import datetime

from bonsai_common import SimulatorSession, Schema
from microsoft_bonsai_api.simulator.client import BonsaiClientConfig
from microsoft_bonsai_api.simulator.generated.models import SimulatorInterface

sys.path.append(os.getcwd() + "/src")

import simulator.constants as tss_constants
import simulator.core.concept_interface as concept_interface
import simulator.core.envg_interface as envg_interface


class MySimulator(SimulatorSession):

    def init(self, execute_posttasks: bool=True):
        """
        Initiation with simulator specific arguments.
        execute_posttasks: will evaluate task performance at the end of an episode if set to True
        """
        self.done = 0
        self.jcmd_prev = None
        self.tcmd_prev = None
        self.rcmd_prev = None
        
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--tssconfig", help="set config file name (e.g., ./configs/train_shadow.json)")
        parser.add_argument("--record", help="set to True to record logs", default="True")
        args, unknown = parser.parse_known_args()

        # record settings
        from distutils.util import strtobool
        self.record_flag = bool(strtobool(args.record))
        if self.record_flag:
            self._save_dir = 'logs/'
            if not os.path.exists(self._save_dir):
                os.mkdir(self._save_dir)

        # start up environment engines
        config_url = args.tssconfig
        self.envg_interface = envg_interface.EnvironmentEngineInterface(config_url)

        # setup concept interface
        self.concept_interface = concept_interface.ConceptInterface(config_url)

        self._observation = {"empty": ""}  # declaration purpose

        self.halted_flag = False
        self.execute_posttasks = execute_posttasks

    def get_state(self) -> Schema:
        """
        Returns simulator states to Bonsai.
        ---
        return: simulator states
        """
        return self.state

    def get_interface(self) -> SimulatorInterface:
        """
        Required for Bonsai.
        ---
        returns: just some empty data
        """
        with open('src/interface.json', 'r') as file:
            json_interface = file.read()
        interface = json.loads(json_interface)
        return SimulatorInterface(
            name=interface["name"],
            timeout=interface["timeout"],
            simulator_context=self.get_simulator_context(),
            description=interface["description"]
        )

    def halted(self):
        """
        Returns True if the pre-sequence execution fails.
        ---
        return: halted or not
        """
        return self.halted_flag

    def episode_start(self, episode_config: Schema):
        """
        Runs a pre-sequence of tasks to initiate the initial state of the task to train.
        episode_config: task training configurations specified in inkling
        """
        print('episode start')
        if self.record_flag: self.envg_interface.startRecording(episode_config, training=True)

        self.halted_flag = False
        self.concept_interface.setConcept(episode_config)

        # set initial world state
        success = self.concept_interface.spawnComponents(episode_config, self.envg_interface, None, None)
        if not success: self.halted_flag = True  

        # if episode_config has an "execute_pretask" field, run the pretasks sim first
        # episode_config["execute_pretasks"] should look like "00001110" (four-digits id per task defined in simulator/constants.py)
        # the config format does not allow using arrays, therefore, string-splitting is used
        pretasks = episode_config["execute_pretasks"]
        num_tasks = len(pretasks) // 4
        for i in range(num_tasks):
            pret = int(pretasks[i*4:i*4+4])
            if pret >= 0:
                config = copy.deepcopy(episode_config)
                config["concept"] = pret
                args = {"jcmd_prev": self.jcmd_prev, "tcmd_prev": self.tcmd_prev, "rcmd_prev": self.rcmd_prev}
                self.concept_interface.initStates(self.envg_interface, config, args)
                while True:
                    res = self.concept_interface.iterateOnce(self.envg_interface, None, episode_config.get("bonsai_concept_url_" + str(i)))
                    if res["terminated"]: break
                    if not res["engine_update_success"]: self.halted_flag = True
                    self.jcmd_prev = copy.deepcopy(res["joints_command"])
                    self.tcmd_prev = copy.deepcopy(res["translation_command"])
                    self.rcmd_prev = copy.deepcopy(res["rotation_command"])

        # escape if execution and not training
        if episode_config["concept"] == tss_constants.ConceptType.CONCEPT_EXECUTE.value:
            if self.record_flag:
                recording_timestamp = str(datetime.now().replace(microsecond=0).strftime("%m-%d-%Y-%H-%M-%S"))
                self.envg_interface.endRecording(self._save_dir+'/log'+ recording_timestamp +'.avi')
            sys.exit(0)
        
        # init concept to train
        args = {"jcmd_prev": self.jcmd_prev}
        self._observation = self.concept_interface.initStates(self.envg_interface, episode_config, args)

        # on pre-sequence execution failure
        if not self.halted_flag:
            self.halted_flag = not self.concept_interface.concept.evaluateStartingStateCondition(self.concept_interface.agentp, self.get_state())

        r_state = self.envg_interface.getKinematicsState()
        self.tcmd_prev = r_state.eef_main.b_position
        self.rcmd_prev = r_state.eef_main.b_orientation
        if self.jcmd_prev is None: self.jcmd_prev = r_state.eef_main.joint_states  # use current state if no previous commands
        
        self._initLog()  # init log tracker

    def episode_step(self, action: Schema):
        """
        Runs one iteration of the task trajectory to train. Moves from one trajectory point to the next in a single iteration.
        action: actions returned from the Bonsai platform, or from exported brains if set to None
        """
        self.done = action["terminate"]

        if not self.done:
            res = self.concept_interface.iterateOnce(self.envg_interface, action)
            if not res["engine_update_success"]: self.halted_flag = True
            self.done = res["done"]
            self.jcmd_prev = copy.deepcopy(res["joints_command"])
            self.tcmd_prev = copy.deepcopy(res["translation_command"])
            self.rcmd_prev = copy.deepcopy(res["rotation_command"])
            self._observation = res["observation"]
            self._updateLog(action)
        elif self.execute_posttasks:
            results = self.concept_interface.concept.evaluateSufficientCondition(
                self.concept_interface.agentp, self.envg_interface, self.jcmd_prev, self.tcmd_prev, self.rcmd_prev)
            self._observation = self.concept_interface._getStateVector(self.envg_interface, done=True)
            for s in results: self._observation[s] = copy.deepcopy(results[s])  # overwrite states from evaluation, these should be concept specific states
        else:
            self._observation = self.concept_interface._getStateVector(self.envg_interface, done=True)

        if self.done:
            self._observation["terminated"] = 1
            if self.record_flag:
                recording_timestamp = str(datetime.now().replace(microsecond=0).strftime("%m-%d-%Y-%H-%M-%S"))
                self.envg_interface.endRecording(self._save_dir+'/log'+ recording_timestamp +'.avi')
            else:
                self.envg_interface.endRecording("", flush=True)
            self._updateLog(action)
            if self.record_flag:
                with open(self._save_dir+'/log' + recording_timestamp + '.json', 'w+') as fw:
                    json.dump(self._log, fw, ensure_ascii=False, indent=4, sort_keys=False, separators=(',', ': '))
            self.concept_interface.pt = 0
            
    def _initLog(self):
        self._log = {}
        self._log['actions'] = {}
        for a in self.concept_interface.action_values:
            self._log['actions'][a] = []
        self._log['state'] = []
    def _updateLog(self, action: dict):
        for a, v in action.items():
            self._log['actions'][a] += [v]
        self._log['state'] += [self._observation]

    @property
    def state(self) -> Schema:
        return self._observation


if __name__ == '__main__':
    config = BonsaiClientConfig(argv=sys.argv)
    sim = MySimulator(config)
    sim.init()
    while sim.run():
        pass