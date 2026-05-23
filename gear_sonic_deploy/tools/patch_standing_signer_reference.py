#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np

FPS = 50.0

G1_ISAACLAB_TO_MUJOCO_DOF = np.array([
    0, 3, 6, 9, 13, 17,
    1, 4, 7, 10, 14, 18,
    2, 5, 8, 11, 15, 19,
    21, 23, 25, 27,
    12, 16, 20, 22, 24, 26, 28
], dtype=np.int64)

STANDING_BY_NAME = {
    "left_hip_pitch_joint": -0.10,
    "left_hip_roll_joint": 0.0,
    "left_hip_yaw_joint": 0.0,
    "left_knee_joint": 0.30,
    "left_ankle_pitch_joint": -0.20,
    "left_ankle_roll_joint": 0.0,

    "right_hip_pitch_joint": -0.10,
    "right_hip_roll_joint": 0.0,
    "right_hip_yaw_joint": 0.0,
    "right_knee_joint": 0.30,
    "right_ankle_pitch_joint": -0.20,
    "right_ankle_roll_joint": 0.0,

    "waist_yaw_joint": 0.0,
    "waist_roll_joint": 0.0,
    "waist_pitch_joint": 0.0,

    "left_shoulder_pitch_joint": 0.0,
    "left_shoulder_roll_joint": 0.0,
    "left_shoulder_yaw_joint": 0.0,
    "left_elbow_joint": 0.0,
    "left_wrist_roll_joint": 0.0,
    "left_wrist_pitch_joint": 0.0,
    "left_wrist_yaw_joint": 0.0,

    "right_shoulder_pitch_joint": 0.0,
    "right_shoulder_roll_joint": 0.0,
    "right_shoulder_yaw_joint": 0.0,
    "right_elbow_joint": 0.0,
    "right_wrist_roll_joint": 0.0,
    "right_wrist_pitch_joint": 0.0,
    "right_wrist_yaw_joint": 0.0,
}

def read_csv(path: Path):
    with open(path, "r") as f:
        header = f.readline().strip().split(",")
    arr = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[None, :]
    return header, arr

def write_csv(path: Path, header, arr):
    np.savetxt(
        path,
        np.asarray(arr, dtype=np.float64),
        delimiter=",",
        fmt="%.6f",
        header=",".join(header),
        comments="",
    )

def stretch(arr, scale):
    arr = np.asarray(arr, dtype=np.float64)
    if arr.shape[0] <= 1 or abs(scale - 1.0) < 1e-8:
        return arr.copy()

    old = np.arange(arr.shape[0], dtype=np.float64)
    new_len = int(round((arr.shape[0] - 1) * scale)) + 1
    new = np.linspace(0.0, arr.shape[0] - 1, new_len)

    flat = arr.reshape(arr.shape[0], -1)
    out = np.empty((new_len, flat.shape[1]), dtype=np.float64)
    for j in range(flat.shape[1]):
        out[:, j] = np.interp(new, old, flat[:, j])
    return out.reshape((new_len,) + arr.shape[1:])

def gradient(arr):
    arr = np.asarray(arr, dtype=np.float64)
    if arr.shape[0] <= 1:
        return np.zeros_like(arr)
    return np.gradient(arr, 1.0 / FPS, axis=0)

def infer_ref_joint_names():
    try:
        import mujoco
    except Exception as e:
        print("[WARN] cannot import mujoco:", e)
        return None

    candidates = [
        Path("g1/g1_29dof_old.xml"),
        Path("../gear_sonic/data/robot_model/model_data/g1/scene_43dof.xml"),
    ]

    xml_path = None
    for p in candidates:
        if p.exists():
            xml_path = p
            break

    if xml_path is None:
        print("[WARN] cannot find G1 XML; fallback to generic indices")
        return None

    model = mujoco.MjModel.from_xml_path(str(xml_path))

    names = []
    for i in range(model.njnt):
        n = model.joint(i).name
        if not n:
            continue
        if any(s in n for s in ["hip", "knee", "ankle", "waist", "shoulder", "elbow", "wrist"]):
            names.append(n)

    names = names[:29]
    if len(names) != 29:
        print("[WARN] expected 29 body joints, got", len(names))
        return None

    ref_names = [names[i] for i in G1_ISAACLAB_TO_MUJOCO_DOF]
    print("[INFO] ref joint names:")
    for i, n in enumerate(ref_names):
        print(f"  ref[{i:02d}] = {n}")
    return ref_names

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-dir", required=True)
    ap.add_argument("--out-base", required=True)
    ap.add_argument("--motion-name", required=True)
    ap.add_argument("--time-scale", type=float, default=2.5)
    ap.add_argument("--upper-scale", type=float, default=0.85)
    ap.add_argument("--waist-mode", choices=["stand", "keep"], default="stand")
    ap.add_argument("--root-z", type=float, default=0.76)
    args = ap.parse_args()

    src = Path(args.src_dir)
    out_motion = Path(args.out_base) / args.motion_name
    out_motion.mkdir(parents=True, exist_ok=True)

    _, joint_pos_src = read_csv(src / "joint_pos.csv")
    if joint_pos_src.shape[1] != 29:
        raise ValueError(f"Expected 29 joints, got {joint_pos_src.shape}")

    joint_pos_src = stretch(joint_pos_src, args.time_scale)
    T = joint_pos_src.shape[0]

    ref_names = infer_ref_joint_names()

    if ref_names is None:
        leg_idx = np.array([0,1,2,3,6,7,8,9,12,13,14,15], dtype=np.int64)
        waist_idx = np.array([4,10,22], dtype=np.int64)
        arm_idx = np.array([i for i in range(29) if i not in set(leg_idx.tolist() + waist_idx.tolist())], dtype=np.int64)
        stand_ref = np.zeros(29, dtype=np.float64)
    else:
        leg_idx = np.array([i for i, n in enumerate(ref_names) if any(s in n for s in ["hip", "knee", "ankle"])], dtype=np.int64)
        waist_idx = np.array([i for i, n in enumerate(ref_names) if "waist" in n], dtype=np.int64)
        arm_idx = np.array([i for i, n in enumerate(ref_names) if any(s in n for s in ["shoulder", "elbow", "wrist"])], dtype=np.int64)

        stand_ref = np.zeros(29, dtype=np.float64)
        for i, n in enumerate(ref_names):
            stand_ref[i] = STANDING_BY_NAME.get(n, 0.0)

        with open(out_motion / "ref_joint_names.txt", "w") as f:
            for i, n in enumerate(ref_names):
                f.write(f"{i:02d},{n}\n")

    keep_idx = arm_idx
    if args.waist_mode == "keep":
        keep_idx = np.concatenate([waist_idx, arm_idx])

    patched = np.tile(stand_ref[None, :], (T, 1))
    patched[:, keep_idx] = stand_ref[keep_idx][None, :] + args.upper_scale * (
        joint_pos_src[:, keep_idx] - stand_ref[keep_idx][None, :]
    )

    joint_vel = gradient(patched)

    body_pos = np.tile(np.array([[0.0, 0.0, args.root_z]], dtype=np.float64), (T, 1))
    body_quat = np.tile(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float64), (T, 1))
    body_lin_vel = np.zeros((T, 3), dtype=np.float64)
    body_ang_vel = np.zeros((T, 3), dtype=np.float64)

    write_csv(out_motion / "joint_pos.csv", [f"joint_{i}" for i in range(29)], patched)
    write_csv(out_motion / "joint_vel.csv", [f"joint_vel_{i}" for i in range(29)], joint_vel)
    write_csv(out_motion / "body_pos.csv", ["body_0_x", "body_0_y", "body_0_z"], body_pos)
    write_csv(out_motion / "body_quat.csv", ["body_0_w", "body_0_x", "body_0_y", "body_0_z"], body_quat)
    write_csv(out_motion / "body_lin_vel.csv", ["body_0_vx", "body_0_vy", "body_0_vz"], body_lin_vel)
    write_csv(out_motion / "body_ang_vel.csv", ["body_0_wx", "body_0_wy", "body_0_wz"], body_ang_vel)

    with open(out_motion / "metadata.txt", "w") as f:
        f.write(f"Metadata for: {args.motion_name}\n")
        f.write("==============================\n\n")
        f.write("Body part indexes:\n")
        f.write("[0]\n\n")
        f.write(f"Total timesteps: {T}\n")

    with open(out_motion / "info.txt", "w") as f:
        f.write("Standing signer physical-sim reference\n")
        f.write(f"source: {src}\n")
        f.write(f"time_scale: {args.time_scale}\n")
        f.write(f"upper_scale: {args.upper_scale}\n")
        f.write(f"waist_mode: {args.waist_mode}\n")
        f.write(f"root_z: {args.root_z}\n")
        f.write(f"frames: {T}\n")
        f.write(f"leg_idx: {leg_idx.tolist()}\n")
        f.write(f"waist_idx: {waist_idx.tolist()}\n")
        f.write(f"arm_idx: {arm_idx.tolist()}\n")

    with open(Path(args.out_base) / "motion_summary.txt", "w") as f:
        f.write(f"{args.motion_name}: {T} frames, 50Hz, standing signer patched\n")

    print("[OK] wrote:", out_motion)
    print("[OK] frames:", T)
    print("[OK] leg_idx frozen:", leg_idx.tolist())
    print("[OK] waist_idx:", waist_idx.tolist(), "mode:", args.waist_mode)
    print("[OK] arm_idx kept:", arm_idx.tolist())
    print("[OK] root z:", args.root_z)

if __name__ == "__main__":
    main()
