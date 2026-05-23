#!/usr/bin/env python3
import os
import joblib
import numpy as np
from pathlib import Path

SEQ = "0002573120929312278-WORKSHOP"

GMR_PKL = Path(f"/data/yujiazeng/DexterousGPT/experiments/g1_wuji/{SEQ}/{SEQ}_g1_body_gmr_motion.pkl")
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

MANUAL = {
    15: 0.0,
    16: 0.0,
    17: 0.0,
    18: -0.75,
    19: "variant2 +1.25 -> -1.25",
    20: 0.0,
    21: 0.0,
    22: 0.0,
    23: 0.0,
    24: 0.0,
    25: -0.75,
    26: "variant2 -1.25 -> +1.25",
    27: 0.0,
    28: 0.0,
}

def stat(x):
    return x.min(), x.max(), x.max() - x.min(), x.mean()

def fmt_stats(x):
    mn, mx, rg, mean = stat(x)
    return f"min={mn: .4f} max={mx: .4f} range={rg: .4f} mean={mean: .4f}"

# Load GMR pkl
d = joblib.load(GMR_PKL)
gmr = np.asarray(d["dof_pos"], dtype=np.float64)  # MuJoCo order

# Load CSV and convert reference order back to MuJoCo semantic order
csv_ref = np.loadtxt(CSV, delimiter=",", skiprows=1)
if csv_ref.ndim == 1:
    csv_ref = csv_ref[None, :]

csv_mj = np.zeros((csv_ref.shape[0], 29), dtype=np.float64)
for mj_i in range(29):
    csv_mj[:, mj_i] = csv_ref[:, REF_FOR_MJ[mj_i]]

print("GMR_PKL:", GMR_PKL)
print("GMR shape:", gmr.shape)
print("CSV:", CSV)
print("CSV shape:", csv_ref.shape)
print()
print("Manual target:")
print("  shoulders ≈ 0")
print("  elbows    ≈ -0.75")
print("  wrist roll variant2")
print()
print("-" * 140)
print(f"{'joint':28s} | {'GMR pkl MuJoCo-order':45s} | {'stable_v2 CSV back-to-MuJoCo':45s} | manual")
print("-" * 140)

for mj_i in [15,16,17,18,19,20,21,22,23,24,25,26,27,28]:
    print(
        f"mj[{mj_i:02d}] {NAMES[mj_i]:22s} | "
        f"{fmt_stats(gmr[:, mj_i]):45s} | "
        f"{fmt_stats(csv_mj[:, mj_i]):45s} | "
        f"{MANUAL[mj_i]}"
    )

print("-" * 140)
print("Interpretation:")
print("1. If GMR already differs from manual, the problem is GMR/SMPL-X→G1 upper-body retarget.")
print("2. If GMR looks manual-like but CSV differs, the problem is CSV conversion / stable patch.")
print("3. If both differ, current retarget is not preserving the sign-language arm semantics.")
