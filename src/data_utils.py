import os
import numpy as np

from pymotion.io.bvh import BVH
from pymotion.ops.skeleton import fk


def load_data(data_dir, scale):
    animations = []
    for filename in os.listdir(data_dir):
        if filename.endswith(".bvh"):
            filepath = os.path.join(data_dir, filename)
            bvh = BVH()
            bvh.load(filepath)
            bvh.set_scale(scale)
            animations.append(bvh)

    pose_db = create_pose_db(animations)
    features_db = create_features_db(pose_db)

    return pose_db, features_db


def create_pose_db(animations):
    poses = []
    for animation in animations:
        local_rotations, local_positions, parents, offsets, _, _ = animation.get_data()
        global_positions = local_positions[:, 0, :]  # root joint
        positions, _ = fk(local_rotations, global_positions, offsets, parents)
        poses.append(positions)
    poses = np.concatenate(poses, axis=0)
    return poses


def create_features_db(pose_db):
    return None
