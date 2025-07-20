import os
import numpy as np
import pymotion.rotations.quat as quat
import pymotion.ops.vector as vec

from pymotion.io.bvh import BVH
from pymotion.ops.skeleton import fk


class PoseDB:
    def __init__(self, poses, character_rot):
        self.poses = poses
        self.character_rot = character_rot

    def __getitem__(self, index):
        return self.poses[index], self.character_rot[index]

    def __len__(self):
        return len(self.poses)


def load_data(data_dir, scale, local_forward_hips):
    animations = []
    for filename in os.listdir(data_dir):
        if filename.endswith(".bvh"):
            filepath = os.path.join(data_dir, filename)
            bvh = BVH()
            bvh.load(filepath)
            bvh.set_scale(scale)
            animations.append(bvh)

    pose_db = create_pose_db(animations, local_forward_hips)
    features_db = create_features_db(pose_db)

    return pose_db, features_db


def create_pose_db(animations, local_forward_hips):
    poses = []
    character_rots = []
    for animation in animations:
        local_rotations, local_positions, parents, offsets, _, _ = animation.get_data()
        global_rotations = local_rotations[:, 0, :]
        global_positions = local_positions[:, 0, :]
        positions, _ = fk(local_rotations, global_positions, offsets, parents)

        # Add character space joint
        positions = np.concatenate([positions[:, 0:1, :], positions], axis=1)
        positions[:, 0, 1] = 0.0  # Set character space joint Y to 0 for ground level

        forward_hips = quat.mul_vec(global_rotations, np.array(local_forward_hips))
        forward_hips[:, 1] = 0.0  # Set Y component to 0 for ground level
        forward_hips = vec.normalize(forward_hips)
        character_rot = quat.from_to_axis(
            np.tile(np.array([0.0, 0.0, 1.0]), forward_hips.shape[:-1] + (1,)),
            forward_hips,
            rot_axis=np.tile(np.array([0.0, 1.0, 0.0]), forward_hips.shape[:-1] + (1,)),
            normalize_input=False,
        )

        poses.append(positions)
        character_rots.append(character_rot)
    poses = np.concatenate(poses, axis=0)
    character_rots = np.concatenate(character_rots, axis=0)
    pose_db = PoseDB(poses, character_rots)
    return pose_db


def create_features_db(pose_db):
    return None
