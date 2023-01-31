import typing
import numpy as np
import copy
from urdf_parser_py.urdf import Robot, Pose


class Transform:
    def __init__(self, data = None):
        if data is None:
            self.R = np.eye(3)
            self.T = np.zeros(3)
        else:
            flat_R = data[0:9]
            R = np.reshape(flat_R, (3, 3))
            self.R = R
            self.T = data[9:12]
    def to_array(self):
        res = np.zeros(12)
        flat_R = np.reshape(self.R, 9)
        res[0:9] = flat_R
        res[9:12] = self.T
        return res
    def inv(self):
        res = Transform()
        res.R = copy.deepcopy(self.R.T)
        res.T = -res.R.dot(self.T)
        return res
    def dot(self, trj):
        res = Transform()
        res.R = self.R.dot(trj.R)
        res.T = self.R.dot(trj.T) + self.T
        return res

def add4(mat):
    res = np.eye(4)
    res[0:3, 0:3] = mat
    return res

def reverse(offset_in: Pose) -> Pose:
    offset = copy.deepcopy(offset_in)
    offset.xyz = -np.array(offset.xyz)
    offset.rpy = -np.array(offset.rpy)
    return offset

def off2trans(offset: Pose) -> Transform:
    res = Transform()
    res.T = np.array(offset.xyz)
    cr = np.cos(offset.rpy[0])
    cp = np.cos(offset.rpy[1])
    cy = np.cos(offset.rpy[2])
    sr = np.sin(offset.rpy[0])
    sp = np.sin(offset.rpy[1])
    sy = np.sin(offset.rpy[2])
    res.T = np.array(offset.xyz)
    res.R = np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                      [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                      [-sp, cp * sr, cp * cr]])
    return res

def rev2trans(axis, angle):
    c = np.cos(angle)
    s = np.sin(angle)
    res = Transform()
    res.R = np.array([[c + axis[0] * axis[0] * (1 - c), axis[0] * axis[1] * (1 - c) - axis[2] * s, axis[0] * axis[2] * (1 - c) + axis[1] * s],
                      [axis[1] * axis[0] * (1 - c) + axis[2] * s, c + axis[1] * axis[1] * (1 - c), axis[1] * axis[2] * (1 - c) - axis[0] * s],
                      [axis[2] * axis[0] * (1 - c) - axis[1] * s, axis[2] * axis[1] * (1 - c) + axis[0] * s, c + axis[2] * axis[2] * (1 - c)]])
    return res

def pris2trans(axis, linear):
    res = Transform()
    res.T = np.array([axis[0]*linear, axis[1]*linear, axis[2]*linear])
    res.R = np.array([[1,0,0],[0,1,0],[0,0,1]])
    return res

def chainname2trans(robot: Robot, chain_name: typing.List[str], joints: typing.List[float], fixed_excluded: bool, get_root_com: bool=False, get_end_com: bool=True) -> Transform:
    # search
    chain = [None for i in range(len(chain_name))]
    for l in robot.links:
        if l.name in chain_name:
            raise Exception("!!!!!!!!!!!!!!!!please use get_chain with links=False option! aborting!!!!!!!!!!!!!!!!")
    for j in robot.joints:
        if j.name in chain_name:
            idx = chain_name.index(j.name)
            chain[idx] = j

    M = Transform()
    # hack to match results for pybullet
    if get_root_com:
        com_root = robot.link_map[robot.get_root()].inertial
        M = M.dot(off2trans(reverse(com_root.origin)))
    
    j = 0
    for i in range(len(chain)):
        #joint
        if chain[i].type == 'revolute' or chain[i].type == 'continuous':
            if chain[i].mimic is None:
                M = M.dot(off2trans(chain[i].origin).dot(
                    rev2trans(chain[i].axis, joints[j])))
                j += 1
            else:
                # assume
                ang = joints[j - 1] * chain[i].mimic.multiplier + chain[i].mimic.offset
                M = M.dot(off2trans(chain[i].origin).dot(
                    rev2trans(chain[i].axis, ang)))
        elif chain[i].type == 'prismatic':
            M = M.dot(off2trans(chain[i].origin)).dot(pris2trans(chain[i].axis, joints[j]))
            j += 1
        elif chain[i].type == 'fixed':
            M = M.dot(off2trans(chain[i].origin))
            if not fixed_excluded:
                j += 1
        else:
            print("!!!!!!!!!!!!!!!!!! unknown chain type %s in joint %d!" % (chain[i].type, i))
                
    # hack to match results for pybullet
    if get_end_com:
        com_end = robot.link_map[chain[-1].child].inertial
        #print(com_end)
        M = M.dot(off2trans(com_end.origin))
    return M
