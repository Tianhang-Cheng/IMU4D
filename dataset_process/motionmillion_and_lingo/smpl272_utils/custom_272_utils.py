"""
This file contains utility functions for custom_272_conversion.py.

Most of them are sourced from the following few scripts:
- amass_process.py
- face_z_transform.py
- infer_get_joints.py
- representation_272.py
- cal_mean_std.py
- recover_visualize.py

These functions may be duplicative or redundant with those in the above scripts.
"""
import numpy as np
import torch
import scipy.ndimage as ndimage


def my_quat_rotate(q, v):
    shape = q.shape
    q_w = q[:, -1]
    q_vec = q[:, :3]
    a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * \
        torch.bmm(q_vec.view(shape[0], 1, 3), v.view(
            shape[0], 3, 1)).squeeze(-1) * 2.0
    return a + b + c

    
def calc_heading(q):
    ref_dir = torch.zeros_like(q[..., 0:3])
    ref_dir[..., 2] = 1
    rot_dir = my_quat_rotate(q, ref_dir)
    heading = torch.atan2(rot_dir[..., 0], rot_dir[..., 2])
    return heading


def calc_heading_quat_inv(q):
    heading = calc_heading(q)
    axis = torch.zeros_like(q[..., 0:3])
    axis[..., 1] = 1
    return -heading, axis


def rot_yaw(yaw):
    cs = np.cos(yaw)
    sn = np.sin(yaw)
    return np.array([[cs,0,sn],[0,1,0],[-sn,0,cs]])


def foot_detect(global_positions, thres):
    """
    derived from https://github.com/orangeduck/Motion-Matching/blob/37df18afc44e8acca3af5e85dff96effa6a34b03/resources/generate_database.py#L160
    """
    left_foot = 10
    right_foot = 11
    global_velocities = global_positions[1:] - global_positions[:-1]
    contact_velocities = np.sqrt(np.sum(global_velocities[:, np.array([left_foot, right_foot])]**2, axis=-1))
    contacts = contact_velocities < thres
    # Median filter here acts as a kind of "majority vote", and removes
    # small regions  where contact is either active or inactive
    for ci in range(contacts.shape[1]):
        contacts[:,ci] = ndimage.median_filter(
            contacts[:,ci], 
            size=6, 
            mode='nearest')
    return contacts