#!/usr/bin/env python3
import re
import joblib
import numpy as np
from pathlib import Path

SEQ = "0002573120929312278-WORKSHOP"

GMR_PKL = Path(f"/data/yujiazeng/DexterousGPT/experiments/g1_wuji/{SEQ}/{SEQ}_g1_body_gmr_motion.pkl")
CSV = Path(f"reference/{SEQ}_ONLY_gmr_oracle_stable_v2/{SEQ}/joint_pos.csv")
POLICY = Path("src/g1/g1_deploy_onnx_ref/include/policy_parameters.hpp")

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

# 你手工试出来的“delta relative to default_angles”
MANUAL_DELTA = {
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

def parse_float_array(name):
    text = POLICY.read_text()
    idx = text.find(name)
    if idx < 0:
        raise RuntimeError(f"Cannot find {name}")

    sub = text[idx: idx + 3000]
    m = re.search(r"\{([^}]+)\}", sub, re.S)
    if not m:
        raise RuntimeError(f"Cannot parse {name}")

    nums = [float(x) for x in re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", m.group(1))]
    if len(nums) < 29:
        raise RuntimeError(f"{name} parsed only {len(nums)} values: {nums}")

    return np.array(nums[:29], dtype=np.float64)

def stats(x):
    return x.min(), x.max(), x.max() - x.min(), x.mean()

def fmt_stats(x):
    mn, mx, rg, mean = stats(x)
    return f"min={mn: .4f} max={mx: .4f} range={rg: .4f} mean={mean: .4f}"

default_angles = parse_float_array("default_angles")

d = joblib.load(GMR_PKL)
gmr_abs = np.asarray(d["dof_pos"], dtype=np.float64)  # MuJoCo/hardware order, absolute q

csv_ref = np.loadtxt(CSV, delimiter=",", skiprows=1)
if csv_ref.ndim == 1:
    csv_ref = csv_ref[None, :]

csv_abs = np.zeros((csv_ref.shape[0], 29), dtype=np.float64)
for mj_i in range(29):
    csv_abs[:, mj_i] = csv_ref[:, REF_FOR_MJ[mj_i]]

gmr_delta = gmr_abs - default_angles[None, :]
csv_delta = csv_abs - default_angles[None, :]

print("GMR_PKL:", GMR_PKL)
print("CSV:", CSV)
print("default_angles parsed from:", POLICY)
print()

print("default_angles for upper body:")
for mj_i in [15,16,17,18,19,20,21,22,23,24,25,26,27,28]:
    print(f"mj[{mj_i:02d}] {NAMES[mj_i]:24s} default={default_angles[mj_i]: .4f}")

print()
print("=" * 150)
print("DELTA comparison: q_delta = q_abs - default_angles")
print("This is the correct comparison to your DGPT_MANUAL_* values.")
print("=" * 150)
print(f"{'joint':30s} | {'GMR delta':45s} | {'stable_v2 CSV delta':45s} | manual delta")
print("-" * 150)

for mj_i in [15,16,17,18,19,20,21,22,23,24,25,26,27,28]:
    print(
        f"mj[{mj_i:02d}] {NAMES[mj_i]:24s} | "
        f"{fmt_stats(gmr_delta[:, mj_i]):45s} | "
        f"{fmt_stats(csv_delta[:, mj_i]):45s} | "
        f"{MANUAL_DELTA[mj_i]}"
    )

print("-" * 150)
print()
print("=" * 150)
print("ABSOLUTE comparison: q_abs vs default_angles + manual_delta")
print("=" * 150)

manual_abs_scalar = {}
for mj_i, v in MANUAL_DELTA.items():
    if isinstance(v, (int, float)):
        manual_abs_scalar[mj_i] = default_angles[mj_i] + float(v)
    else:
        manual_abs_scalar[mj_i] = v

for mj_i in [15,16,17,18,20,21,22,23,24,25,27,28]:
    print(
        f"mj[{mj_i:02d}] {NAMES[mj_i]:24s} "
        f"default={default_angles[mj_i]: .4f} "
        f"manual_abs={manual_abs_scalar[mj_i]: .4f} "
        f"GMR_abs_mean={gmr_abs[:, mj_i].mean(): .4f} "
        f"CSV_abs_mean={csv_abs[:, mj_i].mean(): .4f}"
    )

print()
print("Wrist-roll trajectory endpoints:")
for mj_i in [19, 26]:
    print(f"mj[{mj_i:02d}] {NAMES[mj_i]}")
    print(f"  default={default_angles[mj_i]: .4f}")
    print(f"  GMR_abs first/mid/last = {gmr_abs[0,mj_i]: .4f}, {gmr_abs[len(gmr_abs)//2,mj_i]: .4f}, {gmr_abs[-1,mj_i]: .4f}")
    print(f"  GMR_delta first/mid/last = {gmr_delta[0,mj_i]: .4f}, {gmr_delta[len(gmr_delta)//2,mj_i]: .4f}, {gmr_delta[-1,mj_i]: .4f}")
    print(f"  CSV_abs first/mid/last = {csv_abs[0,mj_i]: .4f}, {csv_abs[len(csv_abs)//2,mj_i]: .4f}, {csv_abs[-1,mj_i]: .4f}")
    print(f"  CSV_delta first/mid/last = {csv_delta[0,mj_i]: .4f}, {csv_delta[len(csv_delta)//2,mj_i]: .4f}, {csv_delta[-1,mj_i]: .4f}")

print()
print("Interpretation:")
print("1. If GMR_delta is close to manual delta, GMR retarget is semantically fine; our earlier comparison was basis-wrong.")
print("2. If GMR_delta differs, then GMR/SMPLX->G1 retarget is not preserving the desired arm semantics.")
print("3. For physical q_target, decide whether the input is absolute q or delta. D6 uses q_target=default+delta.")
