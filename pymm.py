import argparse
import torch
import math
import pymotion.rotations.quat_torch as quat

from pyray import (
    init_window,
    set_target_fps,
    begin_drawing,
    clear_background,
    begin_mode_3d,
    end_mode_3d,
    draw_sphere,
    draw_line_3d,
    draw_grid,
    end_drawing,
    draw_text,
    draw_fps,
    close_window,
    window_should_close,
    Vector3,
    Camera3D,
    RAYWHITE,
    MAROON,
    BLUE,
    DARKGRAY,
    get_mouse_delta,
    is_mouse_button_down,
    get_mouse_wheel_move,
)
from raylib import CAMERA_THIRD_PERSON, CAMERA_PERSPECTIVE
from src.data_utils import load_data
from pymotion.ops.center_of_mass_torch import center_of_mass


def main(data_dir: str, scale: float, local_forward_hips: tuple) -> None:
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

    set_target_fps(60)

    # --- Animation State ---
    frame_index = 1000
    num_frames = len(features_db)
    c_o_m_weights = torch.ones(pose_db.num_joints, dtype=torch.float32) / pose_db.num_joints

    # --- Camera Control State ---
    camera_distance = 15.0
    camera_angle_x = 0.0  # Horizontal rotation
    camera_angle_y = 0.3  # Vertical rotation
    mouse_sensitivity = 0.003
    zoom_sensitivity = 1.5
    min_distance = 2.0
    max_distance = 50.0

    # --- Main Animation Loop ---
    while not window_should_close():
        # --- Update Logic ---

        # Automatically advance to the next frame
        frame_index = (frame_index + 1) % num_frames

        # Get pose for the current frame-
        current_pose, character_rot = pose_db[frame_index]
        character_pos = current_pose[0]
        c_o_m = center_of_mass(current_pose, c_o_m_weights)

        # Mouse-only camera control
        if is_mouse_button_down(0):  # 0 = MOUSE_BUTTON_LEFT
            mouse_delta = get_mouse_delta()
            camera_angle_x += mouse_delta.x * mouse_sensitivity
            camera_angle_y += mouse_delta.y * mouse_sensitivity

            # Clamp vertical rotation to prevent flipping
            camera_angle_y = max(-math.pi / 2 + 0.1, min(math.pi / 2 - 0.1, camera_angle_y))

        # Mouse wheel zoom
        wheel_move = get_mouse_wheel_move()
        if wheel_move != 0:
            camera_distance -= wheel_move * zoom_sensitivity
            # Clamp zoom distance
            camera_distance = max(min_distance, min(max_distance, camera_distance))

        # Update camera position based on angles
        camera_x = c_o_m[0].item() + camera_distance * math.cos(camera_angle_y) * math.cos(camera_angle_x)
        camera_y = c_o_m[1].item() + camera_distance * math.sin(camera_angle_y)
        camera_z = c_o_m[2].item() + camera_distance * math.cos(camera_angle_y) * math.sin(camera_angle_x)

        camera.position = Vector3(camera_x, camera_y, camera_z)
        camera.target = Vector3(c_o_m[0].item(), c_o_m[1].item(), c_o_m[2].item())

        # --- Drawing ---
        begin_drawing()
        clear_background(RAYWHITE)

        begin_mode_3d(camera)

        # Draw the pose for the current frame
        for joint_position in current_pose:
            # Create a Vector3 from the joint's [x, y, z] coordinates and scale it down
            pos_vec = Vector3(joint_position[0].item(), joint_position[1].item(), joint_position[2].item())
            draw_sphere(pos_vec, 0.05, MAROON)  # Draw a small sphere at the joint's position
        character_forward = quat.mul_vec(character_rot, torch.tensor([0.0, 0.0, 1.0]))
        draw_line_3d(
            Vector3(character_pos[0].item(), character_pos[1].item(), character_pos[2].item()),
            Vector3(
                character_pos[0].item() + character_forward[0].item(),
                character_pos[1].item() + character_forward[1].item(),
                character_pos[2].item() + character_forward[2].item(),
            ),
            MAROON,
        )

        # Draw features for the current frame
        trajectory_position_features = features_db.get_trajectory_position(
            frame_index, normalized=False, world_space=True
        )
        for i in range(0, len(trajectory_position_features), 2):
            x = trajectory_position_features[i].item()
            z = trajectory_position_features[i + 1].item()
            draw_sphere(Vector3(x, 0.0, z), 0.05, BLUE)

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
