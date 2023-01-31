# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------

import copy
import transformations


def uround(p, k=4):
    p = copy.deepcopy(p)
    for i in range(len(p)):
        p[i] = round(p[i], k)
        if abs(p[i]) < 0.000001:
                p[i] = 0.0  # -0.0 -> 0.0
    return list(p)

"""
Quaternion calculations are hidden.
This allows consistent [x,y,z,w]-ordered quaternion expressions throughout TSS.
The calculations can use [w,x,y,z]-order depending on the calculation package.
"""

def quaternion_conjugate(q):
    q_c = [q[3], q[0], q[1], q[2]]
    q_c = transformations.quaternion_conjugate(q_c)
    return [q_c[1], q_c[2], q_c[3], q_c[0]]

def quat_mul_vec(q, v):
    q_c = [q[3], q[0], q[1], q[2]]
    v_c = [0.] + list(v)
    return transformations.quaternion_multiply(
        transformations.quaternion_multiply(q_c, v_c),
        transformations.quaternion_conjugate(q_c))[1:]

def quaternion_multiply(q1, q2):
    q1_c = [q1[3], q1[0], q1[1], q1[2]]
    q2_c = [q2[3], q2[0], q2[1], q2[2]]
    q_c = transformations.quaternion_multiply(q1_c, q2_c)
    return [q_c[1], q_c[2], q_c[3], q_c[0]]

def euler_from_quaternion(q, axes):
    q_c = [q[3], q[0], q[1], q[2]]
    return transformations.euler_from_quaternion(q_c, axes)

def quaternion_slerp(q1, q2, t):
    q1_c = [q1[3], q1[0], q1[1], q1[2]]
    q2_c = [q2[3], q2[0], q2[1], q2[2]]
    q_c = transformations.quaternion_slerp(q1_c, q2_c, t)
    return [q_c[1], q_c[2], q_c[3], q_c[0]]

def quaternion_about_axis(angle, axis):
    q_c = transformations.quaternion_about_axis(angle, axis)
    return [q_c[1], q_c[2], q_c[3], q_c[0]]

def quaternion_matrix(q):
    q_c = [q[3], q[0], q[1], q[2]]
    return transformations.quaternion_matrix(q_c)
