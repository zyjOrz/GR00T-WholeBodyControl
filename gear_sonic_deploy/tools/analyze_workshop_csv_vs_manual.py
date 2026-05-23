#!/usr/bin/env python3
from pathlib import Path
import numpy as np

SEQ = "0002573120929312278-WORKSHOP"
CSV = Path(f"reference/{SEQ}_ONLY_gmr_oracle_stable_v2/{SEQ}/joint_pos.csv")

# ref_i = REF_FOR_MJ[mujoco_i]
REF_FOR_MJ = [
    0, 6, 12, 1, 7, 13,
    2, 8, 14, 3, 9, 15,
    22, 4, 10, 16, 23, 5,
    11, 17, 24, 18, 25, 19,
    26, 20, 27, 21, 28
]

NAMES = {
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

# 你手工试出来的最接近原版的 D6 delta 参数
MANUAL = {
    15: 0.0,
    16: 0.0,
    17: 0.0,
    18: -0.75,
    19: "variant2 wrist roll",
    20: 0.0,
    21: 0.0,

    22: 0.0,
    23: 0.0,
    24: 0.0,
    25: -0.75,
    26: "variant2 wrist roll",
    27: 0.0,
    28: 0.0,
}

jp = np.loadtxt(CSV, delimiter=",", skiprows=1)
if jp.ndim == 1:
    jp = jp[None, :]

print("CSV:", CSV)
print("frames:", jp.shape[0])
print()
print("Manual target summary:")
print("  shoulders: pitch/roll/yaw ≈ 0")
print("  elbows   : -0.75")
print("  wrists   : wrist_roll uses variant=2, amp=1.25")
print()
print("CSV joint statistics in MuJoCo/hardware semantic order:")
print("-" * 100)

for mj_i in [15,16,17,18,19,20,21,22,23,24,25,26,27,28]:
    ref_i = REF_FOR_MJ[mj_i]
    x = jp[:, ref_i]
    print(
        f"mj[{mj_i:02d}] {NAMES[mj_i]:24s} "
        f"ref[{ref_i:02d}] "
        f"min={x.min(): .4f} max={x.max(): .4f} "
        f"range={x.max()-x.min(): .4f} "
        f"mean={x.mean(): .4f} "
        f"manual={MANUAL[mj_i]}"
    )

print("-" * 100)
print("Interpretation:")
print("If shoulders have large range but manual wants 0, CSV is not semantically matching WORKSHOP.")
print("If elbows are small/opposite sign vs -0.75, CSV is not matching the desired elbow pose.")
print("If wrist_roll does not follow variant2 direction, palm-up direction will be wrong.")
