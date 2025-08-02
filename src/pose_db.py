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
