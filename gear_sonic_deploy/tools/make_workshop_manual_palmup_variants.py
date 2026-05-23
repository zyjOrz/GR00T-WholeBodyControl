#!/usr/bin/env python3
from pathlib import Path
import shutil
import numpy as np

OUT_BASE = Path("reference/WORKSHOP_manual_palmup_variants")

FPS = 50

# seconds
HOLD0 = 2.0      # neutral stand
RAISE = 3.0      # raise arms + bend elbows
ROTATE = 4.0     # rotate wrists
HOLD1 = 3.0      # hold palm-up pose

N0 = int(HOLD0 * FPS)
N1 = int(RAISE * FPS)
N2 = int(ROTATE * FPS)
N3 = int(HOLD1 * FPS)
T = N0 + N1 + N2 + N3

# IMPORTANT:
# In the C++ D5 override, reference index is:
#   ref_i = isaaclab_to_mujoco[mujoco_i]
#
# This array maps MuJoCo / hardware order index -> reference / IsaacLab CSV index.
REF_FOR_MJ = [
    0, 6, 12, 1, 7, 13,
    2, 8, 14, 3, 9, 15,
    22, 4, 10, 16, 23, 5,
    11, 17, 24, 18, 25, 19,
    26, 20, 27, 21, 28
]

MJ_NAMES = {
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

# Conservative standing lower body in MuJoCo / hardware order.
STAND_MJ = {
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

# Chest-front bent-elbow pose.
# These are MuJoCo / hardware order target angles.
ARM_READY_MJ = {
    # left arm
    15: -0.55,   # left_shoulder_pitch: upper arm forward
    16: 0.30,    # left_shoulder_roll: slightly outward
    17: 0.05,    # left_shoulder_yaw
    18: 1.05,    # left_elbow: bend elbow
    20: 0.00,    # left_wrist_pitch
    21: 0.00,    # left_wrist_yaw

    # right arm
    22: -0.55,   # right_shoulder_pitch: upper arm forward
    23: -0.30,   # right_shoulder_roll: slightly outward
    24: -0.05,   # right_shoulder_yaw
    25: 1.05,    # right_elbow: bend elbow
    27: 0.00,    # right_wrist_pitch
    28: 0.00,    # right_wrist_yaw
}

# Four wrist-roll direction variants.
# One of these should be the desired "back of hand up -> palm up".
VARIANTS = [
    {
        "name": "01_mirror_A_Lneg_to_pos_Rpos_to_neg",
        "left_start": -1.10,
        "left_end": 1.10,
        "right_start": 1.10,
        "right_end": -1.10,
    },
    {
        "name": "02_mirror_B_Lpos_to_neg_Rneg_to_pos",
        "left_start": 1.10,
        "left_end": -1.10,
        "right_start": -1.10,
        "right_end": 1.10,
    },
    {
        "name": "03_same_A_both_neg_to_pos",
        "left_start": -1.10,
        "left_end": 1.10,
        "right_start": -1.10,
        "right_end": 1.10,
    },
    {
        "name": "04_same_B_both_pos_to_neg",
        "left_start": 1.10,
        "left_end": -1.10,
        "right_start": 1.10,
        "right_end": -1.10,
    },
]

def ref_idx(mujoco_i: int) -> int:
    return REF_FOR_MJ[mujoco_i]

def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)

def write_csv(path, header, arr):
    np.savetxt(
        path,
        np.asarray(arr, dtype=np.float64),
        delimiter=",",
        fmt="%.6f",
        header=",".join(header),
        comments="",
    )

def fill_mj_column(jp, mujoco_i, values):
    jp[:, ref_idx(mujoco_i)] = values

def make_motion(var):
    motion_dir = OUT_BASE / var["name"]
    motion_dir.mkdir(parents=True, exist_ok=True)

    jp = np.zeros((T, 29), dtype=np.float64)

    # Lower body standing pose.
    for mj_i, val in STAND_MJ.items():
        fill_mj_column(jp, mj_i, np.full(T, val))

    # Upper body neutral start.
    upper_indices = [
        15, 16, 17, 18, 19, 20, 21,
        22, 23, 24, 25, 26, 27, 28
    ]

    neutral = {mj_i: 0.0 for mj_i in upper_indices}

    ready = dict(ARM_READY_MJ)
    ready[19] = var["left_start"]    # left_wrist_roll
    ready[26] = var["right_start"]   # right_wrist_roll

    final = dict(ARM_READY_MJ)
    final[19] = var["left_end"]      # left_wrist_roll
    final[26] = var["right_end"]     # right_wrist_roll

    # Fill upper body trajectory frame by frame.
    for k in range(T):
        if k < N0:
            pose = neutral

        elif k < N0 + N1:
            a = smoothstep((k - N0 + 1) / N1)
            pose = {}
            for mj_i in upper_indices:
                start_val = neutral.get(mj_i, 0.0)
                ready_val = ready.get(mj_i, 0.0)
                pose[mj_i] = (1.0 - a) * start_val + a * ready_val

        elif k < N0 + N1 + N2:
            a = smoothstep((k - N0 - N1 + 1) / N2)
            pose = {}
            for mj_i in upper_indices:
                ready_val = ready.get(mj_i, 0.0)
                final_val = final.get(mj_i, ready_val)
                pose[mj_i] = (1.0 - a) * ready_val + a * final_val

        else:
            pose = final

        for mj_i, val in pose.items():
            jp[k, ref_idx(mj_i)] = val

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
        f.write("Manual WORKSHOP-like palm-up variant\n")
        f.write(f"variant={var['name']}\n")
        f.write(f"fps={FPS}\n")
        f.write(f"frames={T}\n")
        f.write(f"left_wrist_roll={var['left_start']} -> {var['left_end']}\n")
        f.write(f"right_wrist_roll={var['right_start']} -> {var['right_end']}\n")
        f.write("joint order: reference / IsaacLab CSV order\n")
        f.write("generation rule: ref_i = REF_FOR_MJ[mujoco_i]\n")
        f.write("\nImportant joints:\n")
        for mj_i in upper_indices:
            f.write(f"mujoco_i={mj_i:02d}, ref_i={ref_idx(mj_i):02d}, name={MJ_NAMES[mj_i]}\n")

    print(f"[OK] wrote {motion_dir}")
    print(f"     left_wrist_roll  {var['left_start']} -> {var['left_end']}")
    print(f"     right_wrist_roll {var['right_start']} -> {var['right_end']}")

def main():
    if OUT_BASE.exists():
        shutil.rmtree(OUT_BASE)
    OUT_BASE.mkdir(parents=True, exist_ok=True)

    print("[INFO] REF_FOR_MJ =", REF_FOR_MJ)
    print("[INFO] T frames =", T, "seconds =", T / FPS)

    for var in VARIANTS:
        make_motion(var)

    with open(OUT_BASE / "motion_summary.txt", "w") as f:
        for var in VARIANTS:
            f.write(f"{var['name']}: {T} frames\n")

    print("[OK] wrote batch:", OUT_BASE)

if __name__ == "__main__":
    main()
