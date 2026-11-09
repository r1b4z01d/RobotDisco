"""The action / observation contract (v1 ur5e_lerobot/schema.py, unchanged so v1 datasets and checkpoints stay valid).

  ACTION (10)  = end-effector pose in the UR base frame [x, y, z, rx, ry, rz] (metres, rotation vector) + 4 finger curls [0, 1]
  STATE  (16)  = that pose (6) + arm joints q1..q6 (6) + the commanded curls (4)
  cameras      = scene, wrist, side (RGB, dataset size 960x540 in v1's sets; the policy fits them into 512x512)
"""
from __future__ import annotations

ARM_POSE_NAMES = ("ee_x", "ee_y", "ee_z", "ee_rx", "ee_ry", "ee_rz")
HAND_CURL_NAMES = ("curl_index", "curl_middle", "curl_ring", "curl_thumb")
ACTION_NAMES = ARM_POSE_NAMES + HAND_CURL_NAMES
ARM_POSE_DIM, HAND_CURL_DIM = len(ARM_POSE_NAMES), len(HAND_CURL_NAMES)
ACTION_DIM = ARM_POSE_DIM + HAND_CURL_DIM
ARM_JOINT_NAMES = tuple(f"q{i + 1}" for i in range(6))
STATE_NAMES = ARM_POSE_NAMES + ARM_JOINT_NAMES + HAND_CURL_NAMES
STATE_DIM = len(STATE_NAMES)
CAMERA_KEYS = ("scene", "wrist", "side")
ARM_SLICE = slice(0, ARM_POSE_DIM)
HAND_SLICE = slice(ARM_POSE_DIM, ACTION_DIM)


def split_action(a):
    v = [float(x) for x in a]
    if len(v) != ACTION_DIM:
        raise ValueError(f"expected a {ACTION_DIM}-D action, got {len(v)}")
    return v[ARM_SLICE], v[HAND_SLICE]


def make_state(ee_pose, joints, curls) -> list[float]:
    v = [float(x) for x in ee_pose] + [float(x) for x in joints] + [float(x) for x in curls]
    if len(v) != STATE_DIM:
        raise ValueError(f"state needs {STATE_DIM} values, got {len(v)}")
    return v


def features(width: int, height: int, cameras=CAMERA_KEYS, use_videos: bool = True, shapes: dict | None = None) -> dict:
    """LeRobot dataset features for this robot (what v1's record_sim.build_features produced). `shapes` gives a camera
    its own (height, width) where the cameras differ: v1's sets hold scene and side as 540x540 squares and the wrist at
    540x960, so one size for all would stretch the squares."""
    ft = {
        "action": {"dtype": "float32", "shape": (ACTION_DIM,), "names": list(ACTION_NAMES)},
        "observation.state": {"dtype": "float32", "shape": (STATE_DIM,), "names": list(STATE_NAMES)},
    }
    for cam in cameras:
        h, w = (shapes or {}).get(cam, (height, width))
        ft[f"observation.images.{cam}"] = {"dtype": "video" if use_videos else "image", "shape": (int(h), int(w), 3),
                                           "names": ["height", "width", "channels"]}
    return ft
