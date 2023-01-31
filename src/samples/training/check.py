#!/usr/bin/env python3

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------

"""
Code to check trained results.
"""

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

    sim.episode_start({
        "concept": tss_constants.ConceptType.CONCEPT_GRASP_ACTIVE_FORCE.value,
        "execute_pretasks": "",
        
        "environment": "table.sdf",
        "environment_scale_z": 1.0,
        "environment_position_x": 0.4, "environment_position_y": 0.0,
        "environment_orientation_z": 0.0,
        "environment_orientation_w": 1.0,

        "grasp_cweb0_position_x": 0.0, "grasp_cweb0_position_y": 0.0, "grasp_cweb0_position_z": 0.0,
        "grasp_cweb0_orientation_x": 0.0, "grasp_cweb0_orientation_y": 0.0,
        "grasp_cweb0_orientation_z": 0.0, "grasp_cweb0_orientation_w": 1.0,
        "grasp_px_noise": 0.0, "grasp_py_noise": 0.0, "grasp_pz_noise": 0.0,
        "grasp_qx_noise": 0.0, "grasp_qy_noise": 0.0, "grasp_qz_noise": 0.0,
        "object_scalex_noise": 0.022, "object_scaley_noise": 0.04, "object_scalez_noise": 0.05,
        "object_shape1_noise": 1e-6, "object_shape2_noise": 1,
        "grasp_approach_vertical": -30, "grasp_approach_horizontal": 22.5,

        "config_pick_pos_thrshld": 0.03, "config_pick_orn_thrshld": 5,
    })
    print(sim.state)
    print(sim.halted())
    for i in range(0, sim.concept_interface.max_step+1):
        sim.episode_step(sim.concept_interface.concept.getAction(sim.state, None, "http://localhost:5000"))
        print(sim.state)
        if sim.state['terminated']: break

    print('--------------------------------')

    sim.episode_start({
        "concept": tss_constants.ConceptType.CONCEPT_GRASP_ACTIVE_FORCE.value,
        "execute_pretasks": "",
        
        "environment": "table.sdf",
        "environment_scale_z": 1.0,
        "environment_position_x": 0.4, "environment_position_y": 0.0,
        "environment_orientation_z": 0.0,
        "environment_orientation_w": 1.0,

        "grasp_cweb0_position_x": 0.0, "grasp_cweb0_position_y": 0.0, "grasp_cweb0_position_z": 0.0,
        "grasp_cweb0_orientation_x": 0.0, "grasp_cweb0_orientation_y": 0.0,
        "grasp_cweb0_orientation_z": 0.0, "grasp_cweb0_orientation_w": 1.0,
        "grasp_px_noise": 0.0, "grasp_py_noise": 0.0, "grasp_pz_noise": 0.0,
        "grasp_qx_noise": 0.0, "grasp_qy_noise": 0.0, "grasp_qz_noise": 0.0,
        "object_scalex_noise": 0.02, "object_scaley_noise": 0.0415, "object_scalez_noise": 0.06,
        "object_shape1_noise": 1e-6, "object_shape2_noise": 1e-6,
        "grasp_approach_vertical": -60, "grasp_approach_horizontal": 0,

        "config_pick_pos_thrshld": 0.03, "config_pick_orn_thrshld": 5,
    })
    print(sim.state)
    print(sim.halted())
    for i in range(0, sim.concept_interface.max_step+1):
        sim.episode_step(sim.concept_interface.concept.getAction(sim.state, None, "http://localhost:5000"))
        print(sim.state)
        if sim.state['terminated']: break
