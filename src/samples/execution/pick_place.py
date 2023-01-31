#!/usr/bin/env python3

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------

import sys
import os
from microsoft_bonsai_api.simulator.client import BonsaiClientConfig

sys.path.append(os.getcwd() + "/src")
import simulator.constants as tss_constants

import tss


if __name__ == '__main__':

    config = BonsaiClientConfig(argv=sys.argv)
    sim = tss.MySimulator(config)

    sim.init()

    episode_config = {
        "concept": tss_constants.ConceptType.CONCEPT_EXECUTE.value,
        "execute_pretasks": tss_constants.ToStr([
            tss_constants.ConceptType.CONCEPT_GRASP_ACTIVE_FORCE,
            tss_constants.ConceptType.CONCEPT_PTG11,
            tss_constants.ConceptType.CONCEPT_PTG12,
            tss_constants.ConceptType.CONCEPT_PTG13,
            tss_constants.ConceptType.CONCEPT_RELEASE
        ]),

        # execute environment parameters
        "spawn_from_task": tss_constants.ConceptType.CONCEPT_GRASP_ACTIVE_FORCE.value,
        "environment": "table.sdf", "environment_scale_z": 1.0,
        "environment_position_x": 0.4, "environment_position_y": 0.0,
        "environment_orientation_z": 0.0, "environment_orientation_w": 1.0,
        "object_scalex_noise": 0.022, "object_scaley_noise": 0.04, "object_scalez_noise": 0.05,
        "object_shape1_noise": 1e-6, "object_shape2_noise": 1,

        # grasp task parameters
        "grasp_cweb0_position_x": 0.0, "grasp_cweb0_position_y": 0.0, "grasp_cweb0_position_z": 0.0,
        "grasp_cweb0_orientation_x": 0.0, "grasp_cweb0_orientation_y": 0.0,
        "grasp_cweb0_orientation_z": 0.0, "grasp_cweb0_orientation_w": 1.0,
        "grasp_approach_vertical": -90, "grasp_approach_horizontal": 0,
        "grasp_px_noise": 0.0, "grasp_py_noise": 0.0, "grasp_pz_noise": 0.0,
        "grasp_qx_noise": 0.0, "grasp_qy_noise": 0.0, "grasp_qz_noise": 0.0,

        # pick task parameters
        "ptg11_depart_distance": 0.2,
        "ptg11_depart_vertical": 90, "ptg11_depart_horizontal": 0,

        # bring task parameters (hand origin location in world coordinate)
        "stg12_goal_position_x": -0.175, "stg12_goal_position_y": 0.1, "stg12_goal_position_z": 0.461,

        # place task parameters
        "ptg13_approach_distance": 0.2,
        "ptg13_approach_vertical": -90, "ptg13_approach_horizontal": 0,

        # release task parameters
        "release_depart_vertical": 90, "release_depart_horizontal": 0,

        # url to trained brains
        "bonsai_concept_url_0": "http://localhost:5000",  # grasp
        "bonsai_concept_url_1": "",  # pick
        "bonsai_concept_url_2": "",  # bring
        "bonsai_concept_url_3": "",  # place
        "bonsai_concept_url_4": ""   # release
    }

    sim.episode_start(episode_config)
