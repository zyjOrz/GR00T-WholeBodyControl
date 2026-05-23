#!/usr/bin/env python3
import re
from pathlib import Path
import numpy as np

POLICY = Path("src/g1/g1_deploy_onnx_ref/include/policy_parameters.hpp")
OUT_BASE = Path("reference/DIAG_D5_single_joint_mapping")

FPS = 50
T = 500  # 10 sec
AMP = 0.45

# MuJoCo / hardware order names for body joints.
MUJOCO_NAMES = {
    0: "left_hip_pitch",
    1: "left_hip_roll",
    2: "left_hip_yaw",
    3: "left_knee",
    4: "left_ankle_pitch",
    5: "left_ankle_roll",
    6: "right_hip_pitch",
    7: "right_hip_roll",
    8: "right_hip_yaw",
    9: "right_knee",
    10: "right_ankle_pitch",
    11: "right_ankle_roll",
    12: "waist_yaw",
    13: "waist_roll",
    14: "waist_pitch",
    15: "left_shoulder_pitch",
    16: "left_shoulder_roll",
    17: "left_shoulder_yaw",
    18: "left_elbow",
    19: "left_wrist_roll",
    20: "left_wrist_pitch",
    21: "left_wrist_yaw",
    22: "right_shoulder_pitch",
    23: "right_shoulder_roll",
    24: "right_shoulder_yaw",
    25: "right_elbow",
    26: "right_wrist_roll",
    27: "right_wrist_pitch",
    28: "right_wrist_yaw",
}

# Conservative standing values in MuJoCo / hardware order.
STAND_MUJOCO = {
    0: -0.10,
    1: 0.0,
    2: 0.0,
    3: 0.30,
    4: -0.20,
    5: 0.0,
    6: -0.10,
    7: 0.0,
    8: 0.0,
    9: 0.30,
    10: -0.20,
    11: 0.0,
    12: 0.0,
    13: 0.0,
    14: 0.0,
}

TEST_MUJOCO = [
    15, 16, 17, 18,
    22, 23, 24, 25,
]

def parse_array(name: str):
    text = POLICY.read_text()
    m = re.search(rf"{name}[^=]*=\s*\{{([^}}]+)\}}", text, re.S)
    if not m:
        m = re.search(rf"{name}[^=]*=\s*std::array<[^>]+>\s*\{{([^}}]+)\}}", text, re.S)
    if not m:
        # fallback: find variable then next brace body
        idx = text.find(name)
        if idx < 0:
            raise RuntimeError(f"Cannot find {name}")
        sub = text[idx:idx+1000]
        m = re.search(r"\{([^}]+)\}", sub, re.S)
    nums = [int(x) for x in re.findall(r"-?\d+", m.group(1))]
    if len(nums) != 29:
        raise RuntimeError(f"{name} parsed length {len(nums)} != 29: {nums}")
    return nums

isaaclab_to_mujoco = parse_array("isaaclab_to_mujoco")

print("[INFO] isaaclab_to_mujoco parsed:")
print(isaaclab_to_mujoco)
print("[INFO] Meaning used by override: ref_i = isaaclab_to_mujoco[mujoco_i]")

OUT_BASE.mkdir(parents=True, exist_ok=True)

t = np.arange(T) / FPS
wave = AMP * np.sin(2 * np.pi * 0.25 * t)

def write_csv(path, header, arr):
    np.savetxt(
        path,
        arr,
        delimiter=",",
        fmt="%.6f",
        header=",".join(header),
        comments="",
    )

for mujoco_i in TEST_MUJOCO:
    ref_i = isaaclab_to_mujoco[mujoco_i]
    name = MUJOCO_NAMES[mujoco_i]
    motion_dir = OUT_BASE / f"diag_{mujoco_i:02d}_{name}"
    motion_dir.mkdir(parents=True, exist_ok=True)

    jp = np.zeros((T, 29), dtype=np.float64)

    # Fill standing lower body into reference order using the same mapping.
    for mj_i, val in STAND_MUJOCO.items():
        jp[:, isaaclab_to_mujoco[mj_i]] = val

    # Move only this target joint in reference order.
    jp[:, ref_i] = wave

    jv = np.gradient(jp, 1.0 / FPS, axis=0)

    body_pos = np.tile(np.array([[0.0, 0.0, 0.78874]], dtype=np.float64), (T, 1))
    body_quat = np.tile(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float64), (T, 1))
    zeros = np.zeros((T, 3), dtype=np.float64)

    write_csv(motion_dir / "joint_pos.csv", [f"joint_{i}" for i in range(29)], jp)
    write_csv(motion_dir / "joint_vel.csv", [f"joint_vel_{i}" for i in range(29)], jv)
    write_csv(motion_dir / "body_pos.csv", ["body_0_x", "body_0_y", "body_0_z"], body_pos)
    write_csv(motion_dir / "body_quat.csv", ["body_0_w", "body_0_x", "body_0_y", "body_0_z"], body_quat)
    write_csv(motion_dir / "body_lin_vel.csv", ["body_0_vx", "body_0_vy", "body_0_vz"], zeros)
    write_csv(motion_dir / "body_ang_vel.csv", ["body_0_wx", "body_0_wy", "body_0_wz"], zeros)

    with open(motion_dir / "metadata.txt", "w") as f:
        f.write("Body part indexes:\n")
        f.write("[0]\n")
        f.write(f"Total timesteps: {T}\n")

    with open(motion_dir / "info.txt", "w") as f:
        f.write("DGPT single-joint diagnostic\n")
        f.write(f"mujoco_i={mujoco_i}\n")
        f.write(f"ref_i={ref_i}\n")
        f.write(f"name={name}\n")
        f.write(f"amp={AMP}\n")
        f.write(f"fps={FPS}\n")

    print(f"[OK] {motion_dir}  mujoco_i={mujoco_i} ref_i={ref_i} name={name}")

with open(OUT_BASE / "motion_summary.txt", "w") as f:
    for mujoco_i in TEST_MUJOCO:
        name = MUJOCO_NAMES[mujoco_i]
        f.write(f"diag_{mujoco_i:02d}_{name}: {T} frames\n")

print("[OK] wrote", OUT_BASE)
