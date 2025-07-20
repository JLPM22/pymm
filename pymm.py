import argparse
import numpy as np
import pymotion.rotations.quat as quat

from pyray import *
from raylib import CAMERA_ORBITAL, CAMERA_PERSPECTIVE
from src.data_utils import load_data


def main(data_dir, scale, local_forward_hips):
    pose_db, features_db = load_data(data_dir, scale, local_forward_hips)

    # --- Window and Camera Setup ---
    screen_width = 1280
    screen_height = 720
    init_window(screen_width, screen_height, "Pose Animation")

    camera = Camera3D(
        Vector3(10.0, 3.0, 10.0),  # Camera position (eyeball)
        Vector3(0.0, 2.0, 0.0),  # Camera target (looking at)
        Vector3(0.0, 1.0, 0.0),  # Camera up vector (which way is 'up')
        45.0,  # Camera field-of-view (angle)
        CAMERA_PERSPECTIVE,  # Use perspective projection
    )

    # Use orbital camera controls (mouse drag to rotate, scroll to zoom)
    update_camera(camera, CAMERA_ORBITAL)
    set_target_fps(60)

    # --- Animation State ---
    frame_index = 1000
    num_frames = len(pose_db)

    # --- Main Animation Loop ---
    while not window_should_close():
        # --- Update Logic ---

        # Automatically advance to the next frame
        frame_index = (frame_index + 1) % num_frames

        # Get the root joint's position for the current frame (usually the first joint)
        root_joint_pos = pose_db.poses[frame_index, 0]

        # UPDATE THE CAMERA TARGET TO FOLLOW THE ROOT JOINT
        camera.target = Vector3(root_joint_pos[0], root_joint_pos[1], root_joint_pos[2])

        # Update camera movement based on the new target
        update_camera(camera, CAMERA_ORBITAL)

        # --- Drawing ---
        begin_drawing()
        clear_background(RAYWHITE)

        begin_mode_3d(camera)

        # Draw the pose for the current frame
        current_pose = pose_db.poses[frame_index]
        for joint_position in current_pose:
            # Create a Vector3 from the joint's [x, y, z] coordinates and scale it down
            pos_vec = Vector3(joint_position[0], joint_position[1], joint_position[2])
            draw_sphere(pos_vec, 0.05, MAROON)  # Draw a small sphere at the joint's position
        character_rot = pose_db.character_rot[frame_index]
        character_forward = quat.mul_vec(character_rot, np.array([0.0, 0.0, 1.0]))
        draw_line_3d(
            Vector3(root_joint_pos[0], root_joint_pos[1], root_joint_pos[2]),
            Vector3(
                root_joint_pos[0] + character_forward[0],
                root_joint_pos[1] + character_forward[1],
                root_joint_pos[2] + character_forward[2],
            ),
            BLUE,
        )

        draw_grid(20, 1.0)  # Draw a ground plane for reference

        end_mode_3d()

        # Draw 2D text on top of the 3D scene
        draw_text(f"Frame: {frame_index + 1}/{num_frames}", 10, 10, 20, DARKGRAY)
        draw_fps(10, 40)

        end_drawing()

    close_window()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Motion Mathing in PyTorch and Python")
    parser.add_argument(
        "--data_dir", type=str, default="data", help="Directory where the .bvh files are stored"
    )
    parser.add_argument("--scale", type=list, default=0.01, help="Scale factor for the animation")
    parser.add_argument(
        "--local_forward_hips",
        type=tuple,
        default=[0.0, 0.11043152, 0.9938837],
        help="Local forward vector for hips",
    )
    args = parser.parse_args()

    main(args.data_dir, args.scale, args.local_forward_hips)
