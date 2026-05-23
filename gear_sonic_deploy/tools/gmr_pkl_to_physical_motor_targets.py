#!/usr/bin/env python3
import argparse
import re
import shutil
from pathlib import Path

import joblib
import numpy as np


UPPER_MJ = [
    15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 26, 27, 28,
]

# MuJoCo/hardware order -> reference/IsaacLab CSV order.
# 用于 driver joint_pos.csv；motor_targets.csv 本身使用 MuJoCo/hardware order。
REF_FOR_MJ = [
    0, 6, 12, 1, 7, 13,
    2, 8, 14, 3, 9, 15,
    22, 4, 10, 16, 23, 5,
    11, 17, 24, 18, 25, 19,
    26, 20, 27, 21, 28,
]

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


def parse_float_array(path: Path, name: str):
    text = path.read_text()
    idx = text.find(name)
    if idx < 0:
        raise RuntimeError(f"Cannot find {name} in {path}")
    sub = text[idx: idx + 5000]
    brace = sub.find("{")
    if brace < 0:
        raise RuntimeError(f"Cannot find opening brace for {name}")

    depth = 0
    start = None
    end = None
    for i, ch in enumerate(sub[brace:], start=brace):
        if ch == "{":
            if depth == 0:
                start = i + 1
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i
                break

    if start is None or end is None:
        raise RuntimeError(f"Cannot parse brace array for {name}")

    body = sub[start:end]
    nums = [float(x) for x in re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", body)]
    if len(nums) < 29:
        raise RuntimeError(f"{name} parsed only {len(nums)} values: {nums}")
    return np.array(nums[:29], dtype=np.float64)


def resample(arr, src_fps, dst_fps):
    arr = np.asarray(arr, dtype=np.float64)
    if arr.shape[0] <= 1 or abs(src_fps - dst_fps) < 1e-9:
        return arr.copy()

    old_t = np.arange(arr.shape[0]) / float(src_fps)
    new_T = int(round(old_t[-1] * dst_fps)) + 1
    new_t = np.arange(new_T) / float(dst_fps)
    new_t = np.minimum(new_t, old_t[-1])

    flat = arr.reshape(arr.shape[0], -1)
    out = np.empty((new_T, flat.shape[1]), dtype=np.float64)
    for j in range(flat.shape[1]):
        out[:, j] = np.interp(new_t, old_t, flat[:, j])
    return out.reshape((new_T,) + arr.shape[1:])


def stretch(arr, scale):
    arr = np.asarray(arr, dtype=np.float64)
    if arr.shape[0] <= 1 or abs(scale - 1.0) < 1e-9:
        return arr.copy()

    old = np.arange(arr.shape[0], dtype=np.float64)
    new_T = int(round((arr.shape[0] - 1) * scale)) + 1
    new = np.linspace(0.0, arr.shape[0] - 1, new_T)

    flat = arr.reshape(arr.shape[0], -1)
    out = np.empty((new_T, flat.shape[1]), dtype=np.float64)
    for j in range(flat.shape[1]):
        out[:, j] = np.interp(new, old, flat[:, j])
    return out.reshape((new_T,) + arr.shape[1:])


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gmr-pkl", required=True)
    ap.add_argument("--out-base", required=True)
    ap.add_argument("--motion-name", default="gmr_physical_driver")
    ap.add_argument("--policy-params", default="src/g1/g1_deploy_onnx_ref/include/policy_parameters.hpp")
    ap.add_argument("--target-fps", type=float, default=50.0)
    ap.add_argument("--time-scale", type=float, default=8.0)
    ap.add_argument("--hold-sec", type=float, default=2.0)
    ap.add_argument("--fade-sec", type=float, default=2.0)
    ap.add_argument("--root-z", type=float, default=0.78874)

    # Tracking gains written into motor_targets.csv.
    ap.add_argument("--kp", type=float, default=120.0)
    ap.add_argument("--kd", type=float, default=4.0)

    # Projection options.
    ap.add_argument("--upper-mode", choices=["raw", "delta_centered_shoulders"], default="raw")
    ap.add_argument("--shoulder-scale", type=float, default=1.0)
    ap.add_argument("--wrist-scale", type=float, default=1.0)
    ap.add_argument("--elbow-scale", type=float, default=1.0)

    args = ap.parse_args()

    out_base = Path(args.out_base)
    if out_base.exists():
        shutil.rmtree(out_base)
    out_base.mkdir(parents=True, exist_ok=True)

    policy_params = Path(args.policy_params)
    default_angles = parse_float_array(policy_params, "default_angles")

    d = joblib.load(args.gmr_pkl)
    src_fps = float(d.get("fps", 30.0))
    q_abs = np.asarray(d["dof_pos"], dtype=np.float64)

    if q_abs.ndim != 2 or q_abs.shape[1] != 29:
        raise RuntimeError(f"Expected dof_pos shape (T,29), got {q_abs.shape}")

    # Resample and slow down.
    q_abs = resample(q_abs, src_fps, args.target_fps)
    q_abs = stretch(q_abs, args.time_scale)

    # Optional semantic projection.
    q_phys = q_abs.copy()

    if args.upper_mode == "delta_centered_shoulders":
        # Keep shoulder close to default, preserving only relative shoulder motion.
        shoulder_mj = [15, 16, 17, 22, 23, 24]
        elbow_mj = [18, 25]
        wrist_mj = [19, 20, 21, 26, 27, 28]

        q0 = q_abs[0].copy()

        for j in shoulder_mj:
            q_phys[:, j] = default_angles[j] + args.shoulder_scale * (q_abs[:, j] - q0[j])

        for j in elbow_mj:
            q_phys[:, j] = default_angles[j] + args.elbow_scale * (q_abs[:, j] - default_angles[j])

        for j in wrist_mj:
            q_phys[:, j] = default_angles[j] + args.wrist_scale * (q_abs[:, j] - default_angles[j])

    # Add hold/fade so physical target starts and ends gently.
    hold_n = int(round(args.hold_sec * args.target_fps))
    fade_n = int(round(args.fade_sec * args.target_fps))

    neutral = np.tile(default_angles[None, :], (1, 1))[0]
    # For lower body, use stand; for upper body, start at default.
    for mj_i, val in STAND_MJ.items():
        neutral[mj_i] = val

    prefix = np.tile(neutral[None, :], (hold_n, 1))
    suffix = np.tile(neutral[None, :], (hold_n, 1))

    fade_in = np.empty((fade_n, 29), dtype=np.float64)
    for k in range(fade_n):
        a = smoothstep((k + 1) / max(1, fade_n))
        fade_in[k] = (1 - a) * neutral + a * q_phys[0]

    fade_out = np.empty((fade_n, 29), dtype=np.float64)
    for k in range(fade_n):
        a = smoothstep((k + 1) / max(1, fade_n))
        fade_out[k] = (1 - a) * q_phys[-1] + a * neutral

    q_target = np.vstack([prefix, fade_in, q_phys, fade_out, suffix])
    dq_target = np.gradient(q_target, 1.0 / args.target_fps, axis=0)

    T = q_target.shape[0]

    kp = np.zeros((T, 29), dtype=np.float64)
    kd = np.zeros((T, 29), dtype=np.float64)
    tau = np.zeros((T, 29), dtype=np.float64)

    # Only upper body is intended for override.
    for j in UPPER_MJ:
        kp[:, j] = args.kp
        kd[:, j] = args.kd

    # motor_targets.csv in MuJoCo / hardware order.
    motor_header = (
        [f"q_{i}" for i in range(29)]
        + [f"dq_{i}" for i in range(29)]
        + [f"kp_{i}" for i in range(29)]
        + [f"kd_{i}" for i in range(29)]
        + [f"tau_{i}" for i in range(29)]
    )
    motor_arr = np.concatenate([q_target, dq_target, kp, kd, tau], axis=1)
    write_csv(out_base / "motor_targets.csv", motor_header, motor_arr)

    # Driver motion-data directory in reference/IsaacLab order.
    driver_base = out_base / "driver"
    motion_dir = driver_base / args.motion_name
    motion_dir.mkdir(parents=True, exist_ok=True)

    driver_jp = np.zeros((T, 29), dtype=np.float64)
    for mj_i, val in STAND_MJ.items():
        driver_jp[:, REF_FOR_MJ[mj_i]] = val

    driver_jv = np.gradient(driver_jp, 1.0 / args.target_fps, axis=0)
    body_pos = np.tile(np.array([[0.0, 0.0, args.root_z]], dtype=np.float64), (T, 1))
    body_quat = np.tile(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float64), (T, 1))
    zeros = np.zeros((T, 3), dtype=np.float64)

    write_csv(motion_dir / "joint_pos.csv", [f"joint_{i}" for i in range(29)], driver_jp)
    write_csv(motion_dir / "joint_vel.csv", [f"joint_vel_{i}" for i in range(29)], driver_jv)
    write_csv(motion_dir / "body_pos.csv", ["body_0_x", "body_0_y", "body_0_z"], body_pos)
    write_csv(motion_dir / "body_quat.csv", ["body_0_w", "body_0_x", "body_0_y", "body_0_z"], body_quat)
    write_csv(motion_dir / "body_lin_vel.csv", ["body_0_vx", "body_0_vy", "body_0_vz"], zeros)
    write_csv(motion_dir / "body_ang_vel.csv", ["body_0_wx", "body_0_wy", "body_0_wz"], zeros)

    with open(motion_dir / "metadata.txt", "w") as f:
        f.write("Body part indexes:\n")
        f.write("[0]\n")
        f.write(f"Total timesteps: {T}\n")

    with open(motion_dir / "info.txt", "w") as f:
        f.write("Driver motion for physical motor target replay\n")
        f.write(f"source_gmr_pkl={args.gmr_pkl}\n")
        f.write(f"target_fps={args.target_fps}\n")
        f.write(f"time_scale={args.time_scale}\n")
        f.write(f"upper_mode={args.upper_mode}\n")
        f.write(f"kp={args.kp}\n")
        f.write(f"kd={args.kd}\n")
        f.write(f"frames={T}\n")

    with open(driver_base / "motion_summary.txt", "w") as f:
        f.write(f"{args.motion_name}: {T} frames\n")

    with open(out_base / "info.txt", "w") as f:
        f.write("Physical motor target replay package\n")
        f.write(f"motor_targets={out_base / 'motor_targets.csv'}\n")
        f.write(f"driver_motion_data={driver_base}\n")
        f.write(f"source_gmr_pkl={args.gmr_pkl}\n")
        f.write(f"frames={T}\n")

    print("[OK] wrote:", out_base)
    print("[OK] motor targets:", out_base / "motor_targets.csv")
    print("[OK] driver:", driver_base)
    print("[OK] frames:", T, "seconds:", T / args.target_fps)
    print("[OK] mode:", args.upper_mode)


if __name__ == "__main__":
    main()
