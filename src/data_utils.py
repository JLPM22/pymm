import os

from pymotion.io.bvh import BVH
from src.pose_db import PoseDB, create_pose_db
from src.feature_db import FeaturesDB


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
