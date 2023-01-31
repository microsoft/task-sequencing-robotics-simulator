# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------

inkling "2.0"

using Math
using Goal


## States

# A type defining the dictionary returned from the simulation
type SimState {
    ## States used for the action decision
    observable_demo_parameters: number[8],
    observable_hand_position: number[3],
    observable_hand_orientation: number[4],
    observable_f0_position: number[3],
    observable_f1_position: number[3],
    observable_f2_position: number[3],
    observable_f3_position: number[3],
    observable_f0_contact: number,
    observable_f1_contact: number,
    observable_f2_contact: number,
    observable_f3_contact: number,
    observable_finger_state: number[16],
    observable_timestep: number,

    ## States for reward calculation (necessary condition)
    indicator: number,
    terminated: number,
    initial_hand_position_world: number[3],
    initial_hand_rotation_world: number[4],
    hidden_target_state: number[7],
    hidden_target_state_euler: number[3],

    ## States for reward calculation (sufficient condition)
    objectPosError: number,
    objectOrnError: number,
    reward_indicator: number,
}

type ObservableState {
    ## Finger Postion
    observable_f0_position: number[3],
    observable_f2_position: number[3],

    ## Goal
    observable_estimated_web_vf0: number[3],
    observable_estimated_web_vf2: number[3],

    ## Finger touch or not: can only be 0/1
    observable_f0_contact: number,
    observable_f2_contact: number,

    ## Hand Pos/ orn
    observable_hand_position: number[3],
    observable_hand_orientation: number[4],

    ## Joint 
    observable_finger_state: number[16],

    ## Trajectory iteration
    observable_timestep: number,
}

function TransformState(State: SimState): ObservableState {
    return {
        ## Finger Postion
        observable_f0_position: State.observable_f0_position,
        observable_f2_position: State.observable_f2_position,

        ## Goal
        observable_estimated_web_vf0: [State.observable_demo_parameters[2], State.observable_demo_parameters[3], State.observable_demo_parameters[4]],
        observable_estimated_web_vf2: [State.observable_demo_parameters[5], State.observable_demo_parameters[6], State.observable_demo_parameters[7]],

        ## Finger touch or not: can only be 0/1
        observable_f0_contact: State.observable_f0_contact,
        observable_f2_contact: State.observable_f0_contact,

        ## Hand Pos/ orn
        observable_hand_position: State.observable_hand_position,
        observable_hand_orientation: State.observable_hand_orientation,

        ## Joint 
        observable_finger_state: State.observable_finger_state,

        ## Trajectory iteration
        observable_timestep: State.observable_timestep,
    }
}

## Actions

# This type defines the 'actions' dictionary passed as a parameter to the
# simulate method of the Python simulator
type SimAction {
    rh_FFJ3: number<-0.3 .. 0.3>,
    rh_FFJ2: number<-0.3 .. 0.3>,
    rh_rFFJ1: number<-0.3 .. 0.3>,
    rh_THJ5: number<-0.3 .. 0.3>,
    rh_THJ4: number<-0.3 .. 0.3>,
    rh_THJ2: number<-0.3 .. 0.3>,
    rh_THJ1: number<0.0 .. 0.2>,
    vd: number<-0.03 .. 0.03>,
    vg: number<-0.03 .. 0.03>,
    vw: number<-0.01 .. 0.01>,
    vw_l: number<-0.35 .. 0.35>,
    terminate: number<0, 1, >
}

## Scenario Settings

# This type defines the dictionary passed as a parameter to the
# episode_start method of the Python simulator
type SimConfig {
    # Simulator parameters
    concept: number, # use a fixed value for this training
    execute_pretasks: string, # use a fixed value for this training
    environment: string, # use a fixed value for this training
    environment_scale_z: number, # use a fixed value for this training
    environment_position_x: number, # use a fixed value for this training
    environment_position_y: number, # use a fixed value for this training
    environment_orientation_z: number, # use a fixed value for this training
    environment_orientation_w: number, # use a fixed value for this training
    grasp_cweb0_position_x: number, # set 0 for this training
    grasp_cweb0_position_y: number, # set 0 for this training
    grasp_cweb0_position_z: number, # set 0 for this training
    grasp_cweb0_orientation_x: number, # set 0 for this training
    grasp_cweb0_orientation_y: number, # set 0 for this training
    grasp_cweb0_orientation_z: number, # set 0 for this training
    grasp_cweb0_orientation_w: number, # set 1 for this training

    # Training parameters
    grasp_px_noise: number,
    grasp_py_noise: number,
    grasp_pz_noise: number,
    grasp_qx_noise: number,
    grasp_qy_noise: number,
    grasp_qz_noise: number,
    object_scalex_noise: number,
    object_scaley_noise: number,
    object_scalez_noise: number,
    object_shape1_noise: number,
    object_shape2_noise: number,
    grasp_approach_horizontal: number,
    grasp_approach_vertical: number,

    config_pick_pos_thrshld: number,
    config_pick_orn_thrshld: number,
}

## Main

graph (input: ObservableState): SimAction {
    # An example concept that predicts a SimAction given a SimState
    concept MyConcept(input): SimAction {
        curriculum {

            algorithm {
                Algorithm: "PPO",
                MemoryMode: "none"
            }

            source simulator MySimulator(Action: SimAction, Config: SimConfig): SimState {
                # Please choose the simulator after pressing Train.
            }

            reward GetReward
            terminal function (State: SimState) {
                return (State.terminated or GetTerminal(State))
            }

            state TransformState

            training {
                LessonAssessmentWindow: 100,
                NoProgressIterationLimit: 1000000
            }

            lesson PickSingleObject {
                scenario {
                    concept: 0,
                    execute_pretasks: "",
                    environment: "table.sdf",
                    environment_scale_z: 1.0,
                    environment_position_x: 0.4,
                    environment_position_y: 0.0,
                    environment_orientation_z: 0.0,
                    environment_orientation_w: 1.0,
                    grasp_cweb0_position_x: 0.0,
                    grasp_cweb0_position_y: 0.0,
                    grasp_cweb0_position_z: 0.0,
                    grasp_cweb0_orientation_x: 0.0,
                    grasp_cweb0_orientation_y: 0.0,
                    grasp_cweb0_orientation_z: 0.0,
                    grasp_cweb0_orientation_w: 1.0,

                    grasp_px_noise: number<-0.02 .. 0.02>,
                    grasp_py_noise: number<-0.0024 .. 0.0024>,
                    grasp_pz_noise: number<-0.0045 .. 0.0045>,
                    grasp_qx_noise: number<-0.0244 .. 0.0244>,
                    grasp_qy_noise: number<-0.0593 .. 0.0593>,
                    grasp_qz_noise: number<-0.0593 .. 0.0593>,
                    object_scalex_noise: 0.022,
                    object_scaley_noise: 0.05,
                    object_scalez_noise: 0.05,
                    object_shape1_noise: 0.000001,
                    object_shape2_noise: 0.000001,
                    grasp_approach_horizontal: number<-30 .. 30>,
                    grasp_approach_vertical: number<-90 .. -30>,

                    config_pick_pos_thrshld: 0.1, ## its ok to slip 
                    config_pick_orn_thrshld: 50, ## no orientation cons.
                }
                training {
                    LessonRewardThreshold: 2.2
                }
            }
            lesson PickVariousObject {
                scenario {
                    concept: 0,
                    execute_pretasks: "",
                    environment: "table.sdf",
                    environment_scale_z: 1.0,
                    environment_position_x: 0.4,
                    environment_position_y: 0.0,
                    environment_orientation_z: 0.0,
                    environment_orientation_w: 1.0,
                    grasp_cweb0_position_x: 0.0,
                    grasp_cweb0_position_y: 0.0,
                    grasp_cweb0_position_z: 0.0,
                    grasp_cweb0_orientation_x: 0.0,
                    grasp_cweb0_orientation_y: 0.0,
                    grasp_cweb0_orientation_z: 0.0,
                    grasp_cweb0_orientation_w: 1.0,

                    grasp_px_noise: number<-0.02 .. 0.02>,
                    grasp_py_noise: number<-0.0024 .. 0.0024>,
                    grasp_pz_noise: number<-0.0045 .. 0.0045>,
                    grasp_qx_noise: number<-0.0244 .. 0.0244>,
                    grasp_qy_noise: number<-0.0593 .. 0.0593>,
                    grasp_qz_noise: number<-0.0593 .. 0.0593>,
                    object_scalex_noise: number<0.005 .. 0.03>,
                    object_scaley_noise: number<0.03 .. 0.08>,
                    object_scalez_noise: number<0.03 .. 0.08>,
                    object_shape1_noise: 0.000001,
                    object_shape2_noise: number<0.000001 .. 2>,
                    grasp_approach_horizontal: number<-30 .. 30>,
                    grasp_approach_vertical: number<-90 .. -30>,

                    config_pick_pos_thrshld: 0.1, ## its ok to slip 
                    config_pick_orn_thrshld: 50, ## no orientation cons.
                }
                #training {
                #    LessonRewardThreshold: 
                #}
            }

            lesson PickVariousObject_complexCheck {
                scenario {
                    concept: 0,
                    execute_pretasks: "",
                    environment: "table.sdf",
                    environment_scale_z: 1.0,
                    environment_position_x: 0.4,
                    environment_position_y: 0.0,
                    environment_orientation_z: 0.0,
                    environment_orientation_w: 1.0,
                    grasp_cweb0_position_x: 0.0,
                    grasp_cweb0_position_y: 0.0,
                    grasp_cweb0_position_z: 0.0,
                    grasp_cweb0_orientation_x: 0.0,
                    grasp_cweb0_orientation_y: 0.0,
                    grasp_cweb0_orientation_z: 0.0,
                    grasp_cweb0_orientation_w: 1.0,

                    grasp_px_noise: number<-0.02 .. 0.02>,
                    grasp_py_noise: number<-0.0024 .. 0.0024>,
                    grasp_pz_noise: number<-0.0045 .. 0.0045>,
                    grasp_qx_noise: number<-0.0244 .. 0.0244>,
                    grasp_qy_noise: number<-0.0593 .. 0.0593>,
                    grasp_qz_noise: number<-0.0593 .. 0.0593>,
                    object_scalex_noise: number<0.005 .. 0.03>,
                    object_scaley_noise: number<0.03 .. 0.08>,
                    object_scalez_noise: number<0.03 .. 0.08>,
                    object_shape1_noise: 0.000001,
                    object_shape2_noise: number<0.000001 .. 2>,
                    grasp_approach_horizontal: number<-30 .. 30>,
                    grasp_approach_vertical: number<-90 .. -30>,

                    config_pick_pos_thrshld: 0.03, ## its ok to slip 
                    config_pick_orn_thrshld: 10, ## no orientation cons.
                }
                #training {
                #    LessonRewardThreshold: 
                #}
            }

        }
    }
    output MyConcept
}

## Rewards

function Hypot3(x: number[3]) {
    return Math.Hypot(Math.Hypot(x[0], x[1]), x[2])
}

function Norm3(x: number[3], y: number[3]) {
    var temp: number[3] = [x[0] - y[0], x[1] - y[1], x[2] - y[2]]
    return Hypot3(temp)
}

function GetTerminal(State: SimState) {
    if (State.observable_timestep <= 5) and (State.observable_f0_contact == 1 or State.observable_f2_contact == 1) {
        return 1
    } else {
        return 0
    }
}
function GetReward(State: SimState) {
    if (State.observable_timestep <= 5) and (State.observable_f0_contact == 1 or State.observable_f2_contact == 1) {
        return -1
    }

    ## Pick
    var r_pick = 0
    if (State.reward_indicator == 1) {
        r_pick = 1
    } else if (State.reward_indicator == 0) {
        r_pick = -1
    } else {
        r_pick = 0
    }

    ## Dist Based Penalty factor
    var terminatePenalty_maxdist = 5
    var maxdist = .20 # dist in cm till we put penalty factor
    var mindist = .02 # dist in cm till we put penalty factor

    var observable_estimated_web_vf0 = [State.observable_demo_parameters[2], State.observable_demo_parameters[3], State.observable_demo_parameters[4]]
    var observable_estimated_web_vf2 = [State.observable_demo_parameters[5], State.observable_demo_parameters[6], State.observable_demo_parameters[7]]

    var dist0_temp = Norm3(observable_estimated_web_vf0, State.observable_f0_position)
    var dist2_temp = Norm3(observable_estimated_web_vf2, State.observable_f2_position)
    var dist_mean = (dist0_temp + dist2_temp) / 2

    var r_pick_factor = Math.Max(1 + (terminatePenalty_maxdist - 1) * dist_mean / (maxdist - mindist), 1)
    r_pick_factor = Math.Min(r_pick_factor, terminatePenalty_maxdist)

    r_pick = r_pick * r_pick_factor

    ## Contact web touch
    var dist0 = Norm3(observable_estimated_web_vf0, State.observable_f0_position)
    var r0_dist = 0 - 2 * dist0 # (-0.1 at 5cm, 0 at 0cm)

    var dist2 = Norm3(observable_estimated_web_vf2, State.observable_f2_position)
    var r2_dist = 0 - 2 * dist2 # (-0.1 at 5cm, 0 at 0cm)

    ## Force (reducing non-sparsity but may need to be changed)
    var r0_force = State.observable_f0_contact / 10 # 0,0.1
    var r2_force = State.observable_f2_contact / 10 # 0,0.1

    ## Overall Reward
    var rew = r_pick + (r0_dist + r2_dist) / 2 + (r0_force + r2_force) / 2

    return rew
}
