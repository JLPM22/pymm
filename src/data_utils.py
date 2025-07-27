import os
import time
import torch
import pymotion.rotations.quat_torch as quat
import pymotion.ops.vector_torch as vec

from pymotion.io.bvh import BVH
from pymotion.ops.skeleton_torch import fk


class PoseDB:
    def __init__(self, poses: torch.Tensor, character_rot: torch.Tensor) -> None:
        self.poses = poses  # Shape: (num_frames, num_joints, 3)
        self.character_rot = character_rot  # Shape: (num_frames, 3)
        self.num_joints = poses.shape[1]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.poses[index], self.character_rot[index]

    def __len__(self) -> int:
        return len(self.poses)


class FeaturesDB:
    def __init__(self, pose_db: PoseDB) -> None:
        self.traj_pos_indices = [0, 2, 4]  # 3 future 2D positions
        self.traj_pos_indices_full = [i for i in self.traj_pos_indices for i in (i, i + 1)]
        self.traj_deltas = [20, 40, 60]
        self.max_traj_frames = max(self.traj_deltas)
        self.pose_db = pose_db

        assert len(self.traj_pos_indices) == len(self.traj_deltas)
        assert min(self.traj_deltas) >= 0, "[PyMM] Only future positions are supported"

        start_time = time.time()
        print("[PyMM] Creating FeaturesDB from PoseDB...")

        self.features = torch.empty(
            (len(pose_db) - self.max_traj_frames, len(self.traj_pos_indices_full)),
            dtype=torch.float32,
        )  # Shape: (num_frames - self.max_traj_frames, num_features)

        character_pos = pose_db.poses[:, 0]
        character_rot = pose_db.character_rot

        current_pos = character_pos[: -self.max_traj_frames]
        current_rot = character_rot[: -self.max_traj_frames]
        inv_current_rot = quat.inverse(current_rot)

        for j, t in enumerate(self.traj_pos_indices):
            delta = self.traj_deltas[j]
            feature_pos = character_pos[delta : len(character_pos) - self.max_traj_frames + delta]
            local_future_pos = quat.mul_vec(inv_current_rot, feature_pos - current_pos)
            # Use X and Z for 2D plane
            self.features[:, t] = local_future_pos[:, 0]
            self.features[:, t + 1] = local_future_pos[:, 2]

        self.normalize_features()

        end_time = time.time()
        print(
            f"[PyMM] FeaturesDB created in {end_time - start_time:.2f} seconds with {len(self.features)} frames."
        )

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.features[index]

    def __len__(self) -> int:
        return len(self.features)

    def get_trajectory_position(
        self, idx: int, normalized: bool = True, world_space: bool = False
    ) -> torch.Tensor:
        if normalized:
            features = self.features[idx, self.traj_pos_indices_full]
        else:
            features = (
                self.features[idx, self.traj_pos_indices_full] * self.stds[self.traj_pos_indices_full]
                + self.means[self.traj_pos_indices_full]
            )

        if world_space:
            pose, character_rot = self.pose_db[idx]
            character_pos = pose[0]
            for t in self.traj_pos_indices:
                feature = (
                    quat.mul_vec(character_rot, torch.tensor([features[t], 0.0, features[t + 1]]))
                    + character_pos
                )
                features[[t, t + 1]] = feature[[0, 2]]  # Return only X and Z for 2D plane

        return features

    def normalize_features(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.means = torch.mean(self.features, dim=0)
        self.stds = torch.std(self.features, dim=0)
        self.features = (self.features - self.means) / self.stds
        return self.means, self.stds


def load_data(data_dir: str, scale: float, local_forward_hips: tuple) -> tuple[PoseDB, FeaturesDB]:
    animations = []
    for filename in os.listdir(data_dir):
        if filename.endswith(".bvh"):
            filepath = os.path.join(data_dir, filename)
            bvh = BVH()
            bvh.load(filepath)
            bvh.set_scale(scale)
            animations.append(bvh)

    pose_db = create_pose_db(animations, local_forward_hips)
    features_db = FeaturesDB(pose_db)

    return pose_db, features_db


def create_pose_db(animations: list[BVH], local_forward_hips: tuple) -> PoseDB:
    print("[PyMM] Creating PoseDB from animations...")
    start_time = time.time()
    poses = []
    character_rots = []
    for animation in animations:
        local_rotations, local_positions, parents, offsets, _, _ = animation.get_data()
        local_rotations = torch.from_numpy(local_rotations).float()
        local_positions = torch.from_numpy(local_positions).float()
        global_rotations = local_rotations[:, 0, :]
        global_positions = local_positions[:, 0, :]
        offsets = torch.from_numpy(offsets).float()
        parents = torch.from_numpy(parents).long()
        positions, _ = fk(local_rotations, global_positions, offsets, parents)

        # Add character space joint
        positions = torch.cat([positions[:, 0:1, :], positions], dim=1)
        positions[:, 0, 1] = 0.0  # Set character space joint Y to 0 for ground level

        forward_hips = quat.mul_vec(global_rotations, torch.tensor(local_forward_hips))
        forward_hips[:, 1] = 0.0  # Set Y component to 0 for ground level
        forward_hips = vec.normalize(forward_hips)
        character_rot = quat.from_to_axis(
            torch.tile(torch.tensor([0.0, 0.0, 1.0]), forward_hips.shape[:-1] + (1,)),
            forward_hips,
            rot_axis=torch.tile(torch.tensor([0.0, 1.0, 0.0]), forward_hips.shape[:-1] + (1,)),
            normalize_input=False,
        )

        poses.append(positions)
        character_rots.append(character_rot)
    poses = torch.cat(poses, dim=0)
    character_rots = torch.cat(character_rots, dim=0)
    pose_db = PoseDB(poses, character_rots)
    end_time = time.time()
    print(f"[PyMM] PoseDB created in {end_time - start_time:.2f} seconds with {len(pose_db)} frames.")
    return pose_db
