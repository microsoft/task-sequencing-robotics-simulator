# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------

import typing
import copy
import numpy as np
import pybullet

import simulator.utils as utils

import simulator.structs as tss_structs
import environment_engines.envg_template as envg_template


class ePyBullet(envg_template.Env):

    def __init__(self, config: dict, role: tss_structs.EnvironmentEngineRole):
        pybullet.connect(pybullet.DIRECT)

        print("tss_PyBullet expected fields: urdf_path: str")
        self.robot_urdf_path = config["urdf_path"]
        self.config_urls = []
        self.targetids = []
        self.joint_indices = []
        self.latest_kinematic_state = None

        self._record = False
        self._jv_prev = None
        self.CTRL_FLG_ROBOT_RELOADED = False 
        self.CTRL_FLG_HAND_STAY = False

    def reset(self):
        pybullet.resetSimulation()
        self._setSimulationConfig(.001)

        # remove any auto-generated files
        import os
        for url in self.config_urls:
            os.remove('logs/' + url + '.sdf')
            os.remove('logs/' + url + '.stl')
            self.config_url = None

    def loadRobot(self, init_state: tss_structs.ActionStruct, body_ik_settings=None) -> tss_structs.ActionStruct:
        trans, rot = self._toPyBulletCoords(init_state.eef_main.b_position, init_state.eef_main.b_orientation)

        self.robotid, self.cid = self._loadURDF(self.robot_urdf_path)
        self.joint_indices = []
        for jname in init_state.eef_main.joint_names: self.joint_indices.append(self._getJointId(self.robotid, jname))

        # set pybullet joint states
        joints_ = copy.deepcopy(init_state.eef_main.joint_states)
        i = 0
        for j in self.joint_indices:
            pybullet.resetJointState(self.robotid, j, joints_[i])
            i += 1

        self._setPose(self.robotid, trans, rot)
        pybullet.enableJointForceTorqueSensor(self.robotid, 0)
        # need to move once to get correct force sensor values
        self._execute(self.robotid, self.cid, init_state.eef_main.joint_states, trans, rot, speed=1., timesec=.1)
        self.latest_kinematic_state = copy.deepcopy(self._getUpdatedWorldRobotState(init_state))

        return init_state

    def update(self, cmd: tss_structs.ActionStruct, components_in: typing.List[tss_structs.ComponentStruct], body_ik_settings=None) -> tss_structs.WorldStruct:
        trans, rot = self._toPyBulletCoords(cmd.eef_main.b_position, cmd.eef_main.b_orientation)
        self._execute(self.robotid, self.cid, cmd.eef_main.joint_states, trans, rot, speed=1., timesec=cmd.timesec)
        rstate_out = self._getUpdatedWorldRobotState(cmd)
        self.latest_kinematic_state = copy.deepcopy(rstate_out)
        # get updated world component state
        components_out = []
        for c in components_in:
            ctrans, crot = pybullet.getBasePositionAndOrientation(self.targetids[c.name])
            components_out.append(tss_structs.ComponentStruct(c.name, (ctrans, crot), c.geometry, c.c_type, c.c_states, c.ignore_gravity))
        return tss_structs.WorldStruct(rstate_out, components_out)

    def loadComponents(self, components: typing.List[tss_structs.ComponentStruct]) -> typing.List[tss_structs.ComponentStruct]:
        import shutil
        import xml.etree.ElementTree as ET

        self.config_urls = []
        self.targetids = {}
        for c in components:
            if c.geometry.geometry_type == tss_structs.GeometryType.MESH:
                config_url = c.geometry.url
                self.config_urls.append(config_url)

                shutil.copy('src/components/geometries/box.sdf', 'logs/' + config_url + '.sdf')
                tree = ET.parse('logs/' + config_url + '.sdf')
                root = tree.getroot()
                root[0][1][0][0][0][0].text = config_url + '.stl'
                root[0][1][1][0][0][0].text = config_url + '.stl'
                tree.write('logs/' + config_url + '.sdf')

                targetid = pybullet.loadSDF('logs/' + config_url + '.sdf')[0]
                self.targetids[c.name] = targetid
                self._setPose(targetid, c.pose[0], c.pose[1], fixBase=c.ignore_gravity)
            elif c.geometry.geometry_type == tss_structs.GeometryType.FILE:
                config_url = c.geometry.url
                if config_url[-3:] == "sdf": constraintid = pybullet.loadSDF('src/components/geometries/' + config_url)[0]
                else: constraintid, _ = self._loadURDF('src/components/geometries/' + config_url)
                self.targetids[c.name] = constraintid
                self._setPose(constraintid, c.pose[0], c.pose[1], fixBase=c.ignore_gravity)

        return components

    def getKinematicsState(self) -> tss_structs.ActionStruct:
        return self.latest_kinematic_state

    def getLinkState(self, link_name: str) -> typing.Tuple[typing.Tuple, typing.Tuple]:
        return copy.deepcopy(pybullet.getLinkState(self.robotid, self._getLinkId(self.robotid, link_name)))

    def getPhysicsState(self, cmd: str, component_name: str, rest: typing.Optional[dict]=None) -> typing.Any:
        if cmd == "BinaryContact":
            # Note, below cannot distinguish the touching direction (i.e., will be 1 even if contacted with the nail).
            # However, a well-trained brain should not encounter such states.
            force = utils.uround(self._getContactNormal(self.robotid, rest["link"], self.targetids[component_name], -1))
            if force != [0, 0, 0]: return 1
            else: return 0
        elif cmd == "ContactNormal":
            return utils.uround(self._getContactNormal(self.robotid, rest["link"], self.targetids[component_name], -1))
        elif cmd == "SurfaceContact":
            if rest["sensor"] == "joint_torque":
                ret = False
                for i, jn in enumerate(rest["joints"]):
                    jidx = self._getJointId(self.robotid, jn)
                    ret = ret or abs(pybullet.getJointState(self.robotid, jidx)[3] > rest["thresholds"][i])
                return ret

    def recordStart(self, config: dict, training: bool):
        self._setSimulationConfig(0.001)
        self._record = True
        self._record_buffer = []
        self._camera_w = 640
        self._camera_h = 480
        self._fps = 5
        self._fps_cnt = 0
        self._fps_cycle = int(1/(self._timeStep*self._fps))

        cameraEyePosition = [0.5, 0.5, 1.0]
        cameraTargetPosition = [0.0, 0.0, 0.0]
        cameraUpVector = [0, 0, 1]
        proj = [-0.5, 0.5, -0.5, 0.5, 0.5, 2.0]

        self._camera_view = pybullet.computeViewMatrix(
            cameraEyePosition, cameraTargetPosition, cameraUpVector)
        self._camera_projection = pybullet.computeProjectionMatrix(
            proj[0], proj[1], proj[2], proj[3], proj[4], proj[5])

    def recordEnd(self, filename: str, flush: bool):
        if flush:
            self._record_buffer = []
            self._record = False
        else:
            import cv2
            out = cv2.VideoWriter(
                filename, cv2.VideoWriter_fourcc('M','J','P','G'), self._fps, (self._camera_w, self._camera_h))
            for i, img in enumerate(self._record_buffer):
                out.write(img)
            out.release()
            self._record_buffer = []
            self._record = False
            print('recorded %s' % filename)

    def _setSimulationConfig(self, timeStep: float):
        pybullet.setGravity(0, 0, -9.8)
        pybullet.setTimeStep(timeStep=timeStep)
        self._timeStep = timeStep

    def _setPose(self, objid: int, trans: typing.List[float], rot: typing.List[float], fixBase: bool=False):
        trans_ = copy.deepcopy(trans)
        rot_ = copy.deepcopy(rot)
        pybullet.resetBasePositionAndOrientation(objid, trans_, rot_)
        if not fixBase:
            return
        cid = pybullet.createConstraint(
            objid, -1, -1, -1, pybullet.JOINT_FIXED, [0, 0, 0], [0, 0, 0], [0, 0, 0])
        pybullet.changeConstraint(cid, jointChildPivot=trans_, jointChildFrameOrientation=rot_)

    def _loadURDF(self, filename: str):
        # useFixedBase=0 -> uses constraint to keep model in air
        robot = pybullet.loadURDF(filename, useFixedBase=0)
        cid = pybullet.createConstraint(
            robot, -1, -1, -1, pybullet.JOINT_FIXED, [0, 0, 0], [0, 0, 0], [0, 0, 0])
        self.CTRL_FLG_ROBOT_RELOADED = True
        return robot, cid

    def _getJointId(self, model_id: int, name: str):
        for _id in range(pybullet.getNumJoints(model_id)):
            _name = pybullet.getJointInfo(model_id, _id)[1].decode('UTF-8')
            if _name == name:
                return _id
        return -1

    def _getLinkId(self, model_id: int, name: str):
        _link_name_to_index = {pybullet.getBodyInfo(model_id)[0].decode('UTF-8'):-1,}
        for _id in range(pybullet.getNumJoints(model_id)):
            _name = pybullet.getJointInfo(model_id, _id)[12].decode('UTF-8')
            if _name == name:
                return _id
        return -1

    def _getUpdatedWorldRobotState(self, cmd: tss_structs.ActionStruct):
        # Below assumes that only the end-effector joint angles are explored and executed in PyBullet.
        # If training using the full body joint angles, the full body should be seen as the "end-effector."
        rtrans, rrot = pybullet.getBasePositionAndOrientation(self.robotid)
        rtrans, rrot = self._revertPyBulletCoords(rtrans, rrot)
        rstate_out = tss_structs.ActionStruct(tss_structs.MultiLinkAction(cmd.eef_main.joint_names,
            [pybullet.getJointState(self.robotid, x)[0] for x in self.joint_indices], rtrans, rrot), cmd.body, cmd.eef_sub, cmd.timesec, cmd.error)
        return rstate_out

    def _getContactNormal(self, robot, link, target, target_linknum):
        # clp: contact link points
        vf_clps = pybullet.getContactPoints(robot, target, self._getLinkId(robot, link), target_linknum)
        vf_clp = [0.0, 0.0, 0.0]
        cnt = 0
        if len(vf_clps) > 0:
            for clp in vf_clps:
                vf_clp[0] += clp[7][0]
                vf_clp[1] += clp[7][1]
                vf_clp[2] += clp[7][2]
                cnt += 1
        if cnt > 1:
            vf_clp[0] /= cnt
            vf_clp[1] /= cnt
            vf_clp[2] /= cnt
        return vf_clp

    def _toPyBulletCoords(self, trans, rot):
        # Pybullet uses inertial origin so need to add a position offset.
        # Note, 0.05 might not be the correct value depending on the robot.
        return (
            np.array(copy.deepcopy(trans)) + 0.05*np.array(utils.quat_mul_vec(rot, [0,0,1])),
            rot
        )

    def _revertPyBulletCoords(self, trans, rot):
        # Pybullet uses inertial origin so need to remove the position offset.
        # Note, 0.05 might not be the correct value depending on the robot.
        return (
            np.array(copy.deepcopy(trans)) - 0.05*np.array(utils.quat_mul_vec(rot, [0,0,1])),
            rot
        )

    def _execute(self, robot, cid, jv, trans=[0,0,0], rot=[0,0,0,1], speed=1., timesec=None):
        if self._jv_prev is None:
            joint_states = []
            n = pybullet.getNumJoints(robot)
            for j in range(n):
                if j in self.joint_indices:
                    joint_states.append(pybullet.getJointState(robot, j)[0])
            jv0 = joint_states
        else:
            jv0 = copy.deepcopy(self._jv_prev)
        jv1 = copy.deepcopy(jv)
        self._jv_prev = jv1
        
        o_t = 0.0  # [sec]
        for i in range(len(jv0)):
            t = abs(jv1[i] - jv0[i])/speed
            o_t = max(o_t, t)
        min_speed = self._timeStep*10  # execution time cannot be smaller than 10*simstep
        if o_t < min_speed:
            o_t = .1  # some default value
        
        trans0, rot0 = pybullet.getBasePositionAndOrientation(robot)
        trans1 = copy.deepcopy(trans)
        rot1 = copy.deepcopy(rot)
        
        if timesec is None:
            timesec = .1
            timesec += np.linalg.norm(np.array(trans1) - np.array(trans0))*10
            timesec += min(np.linalg.norm(np.array(rot1) - np.array(rot0))*10, 2.)
        steps = max(int(o_t/self._timeStep), int(timesec/self._timeStep))
        per_steps = 1./steps
        
        if (np.linalg.norm(np.array(trans1) - np.array(trans0)) < 0.0001
            and np.linalg.norm(np.array(rot1) - np.array(rot0)) < 0.0001): self.CTRL_FLG_HAND_STAYS = True
        else: self.CTRL_FLG_HAND_STAYS = False

        for s in range(steps):
            trans0_ = copy.deepcopy(trans0)
            trans1_ = copy.deepcopy(trans1)
            rot0_ = copy.deepcopy(rot0)
            rot1_ = copy.deepcopy(rot1)
            joints0_ = copy.deepcopy(jv0)
            joints1_ = copy.deepcopy(jv1)
            step_factor = s*per_steps

            # interpolation
            trans_s = [trans0_[0] + step_factor*(trans1_[0] - trans0_[0]),
                       trans0_[1] + step_factor*(trans1_[1] - trans0_[1]),
                       trans0_[2] + step_factor*(trans1_[2] - trans0_[2])]
            joints_s = [0.0 for x in range(len(joints0_))]
            for i, v in enumerate(joints0_):
                joints_s[i] = v + step_factor*(joints1_[i] - v)
            q0 = np.array(rot0_)
            q1 = np.array(rot1_)
            dot = np.sum(q0*q1)
            if dot < 0.0:  # negate to get shorter path
                q1 = -q1
                dot = -dot
            if dot > 0.9995:  # rot1 close to rot0
                q = q0 + step_factor*(q1 - q0)
                q *= 1/np.linalg.norm(q)
            else:
                theta0 = np.arccos(dot)
                k = 1/np.sin(theta0)
                q = np.sin(theta0*(1-step_factor))*k*q0 + np.sin(theta0*step_factor)*k*q1
            rot_s = [q[0], q[1], q[2], q[3]]

            # update model state
            if not self.CTRL_FLG_HAND_STAYS or self.CTRL_FLG_ROBOT_RELOADED:
                pybullet.changeConstraint(cid, jointChildPivot=trans_s, jointChildFrameOrientation=rot_s)
            pybullet.setJointMotorControlArray(
                robot, self.joint_indices, pybullet.POSITION_CONTROL, targetPositions=joints_s)

            a = pybullet.stepSimulation()  # to avoid print use 'a ='

            if self._record:
                if self._fps_cnt % self._fps_cycle == 0:
                    img = pybullet.getCameraImage(
                        self._camera_w, self._camera_h,
                        self._camera_view, self._camera_projection)[2]
                    img = np.array(img, dtype=np.uint8).reshape((self._camera_h, self._camera_w, 4))
                    img = img[:,:,:3]
                    self._record_buffer.append(img)
                self._fps_cnt += 1

        self.CTRL_FLG_ROBOT_RELOADED = False
