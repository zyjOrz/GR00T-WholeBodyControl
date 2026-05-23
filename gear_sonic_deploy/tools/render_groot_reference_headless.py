#!/usr/bin/env python3
import os

# Must be set before importing mujoco.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import argparse
from pathlib import Path
import xml.etree.ElementTree as ET

import cv2
import mujoco
import numpy as np


# GR00T reference CSV joint order is IsaacLab order.
# MuJoCo qpos wants MuJoCo order.
ISAACLAB_TO_MUJOCO = np.array([
    0, 3, 6, 9, 13, 17,
    1, 4, 7, 10, 14, 18,
    2, 5, 8, 11, 15, 19,
    21, 23, 25, 27,
    12, 16, 20, 22, 24, 26, 28
], dtype=np.int64)


def prepend_names(elem, prefix: str):
    if "name" in elem.attrib:
        elem.attrib["name"] = prefix + elem.attrib["name"]
    for child in list(elem):
        prepend_names(child, prefix)


def build_g1_scene(width: int, height: int):
    main_scene = ET.parse("g1/scene_empty.xml")
    robot = ET.parse("g1/g1_29dof_old.xml")

    scene_root = main_scene.getroot()
    robot_root = robot.getroot()

    scene_asset = scene_root.find("asset")
    robot_asset = robot_root.find("asset")
    if scene_asset is None or robot_asset is None:
        raise RuntimeError("Cannot find asset nodes in scene / robot XML")

    for mesh in robot_asset.findall("mesh"):
        mesh_file = mesh.get("file")
        if mesh_file:
            mesh.set("file", str(Path("g1") / "meshes" / mesh_file))
        scene_asset.append(mesh)

    scene_default = scene_root.find("default")
    robot_default = robot_root.find("default")
    if scene_default is not None and robot_default is not None:
        for default in robot_default.findall("default"):
            scene_default.append(default)

    scene_worldbody = scene_root.find("worldbody")
    robot_worldbody = robot_root.find("worldbody")
    if scene_worldbody is None or robot_worldbody is None:
        raise RuntimeError("Cannot find worldbody nodes in scene / robot XML")

    robot_body = robot_worldbody.find("body")
    if robot_body is None:
        raise RuntimeError("Cannot find root body in G1 robot XML")

    prepend_names(robot_body, "robot1_")
    scene_worldbody.append(robot_body)

    xml = ET.tostring(scene_root, encoding="unicode")
    model = mujoco.MjModel.from_xml_string(xml)

    model.vis.global_.offwidth = width
    model.vis.global_.offheight = height
    model.vis.quality.shadowsize = 0
    model.vis.quality.offsamples = 1
    model.vis.headlight.ambient[:] = [0.8, 0.8, 0.8]
    model.vis.headlight.diffuse[:] = [0.8, 0.8, 0.8]
    model.vis.headlight.specular[:] = [0.1, 0.1, 0.1]

    return model


def load_csv(path: Path):
    arr = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[None, :]
    return arr


def load_motion(motion_dir: Path):
    joint_pos = load_csv(motion_dir / "joint_pos.csv")
    body_pos = load_csv(motion_dir / "body_pos.csv")
    body_quat = load_csv(motion_dir / "body_quat.csv")

    if joint_pos.shape[1] != 29:
        raise ValueError(f"joint_pos must have 29 columns, got {joint_pos.shape}")

    if body_pos.shape[1] < 3:
        raise ValueError(f"body_pos must have at least 3 columns, got {body_pos.shape}")

    if body_quat.shape[1] < 4:
        raise ValueError(f"body_quat must have at least 4 columns, got {body_quat.shape}")

    dof_mujoco = joint_pos[:, ISAACLAB_TO_MUJOCO]
    root_pos = body_pos[:, :3]
    root_quat_wxyz = body_quat[:, :4]

    # Normalize quaternion defensively.
    n = np.linalg.norm(root_quat_wxyz, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    root_quat_wxyz = root_quat_wxyz / n

    T = min(len(dof_mujoco), len(root_pos), len(root_quat_wxyz))
    return root_pos[:T], root_quat_wxyz[:T], dof_mujoco[:T]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion_dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--output-fps", type=float, default=25.0)
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--distance", type=float, default=3.2)
    parser.add_argument("--azimuth", type=float, default=90.0)
    parser.add_argument("--elevation", type=float, default=-15.0)
    args = parser.parse_args()

    motion_dir = Path(args.motion_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    root_pos, root_quat, dof = load_motion(motion_dir)
    print("[INFO] frames:", len(dof))
    print("[INFO] root z min/max:", float(root_pos[:, 2].min()), float(root_pos[:, 2].max()))
    print("[INFO] joint min/max:", float(dof.min()), float(dof.max()))

    model = build_g1_scene(args.width, args.height)
    data = mujoco.MjData(model)

    renderer = mujoco.Renderer(model, height=args.height, width=args.width)

    cam = mujoco.MjvCamera()
    center = np.median(root_pos, axis=0)
    cam.lookat[:] = [center[0], center[1], max(center[2], 0.65)]
    cam.distance = args.distance
    cam.azimuth = args.azimuth
    cam.elevation = args.elevation

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(out_path),
        fourcc,
        args.output_fps,
        (args.width, args.height),
    )

    if not writer.isOpened():
        raise RuntimeError(
            "cv2.VideoWriter failed to open. Try: sudo apt-get install -y ffmpeg"
        )

    for i in range(len(dof)):
        data.qpos[:] = model.qpos0
        data.qvel[:] = 0.0

        # G1 free joint: root xyz + root quat(wxyz) + 29 dofs.
        data.qpos[0:3] = root_pos[i]
        data.qpos[3:7] = root_quat[i]
        data.qpos[7:7 + 29] = dof[i]

        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=cam)
        rgb = renderer.render()

        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        for _ in range(max(1, args.repeat)):
            writer.write(bgr)

    writer.release()
    renderer.close()

    print("[OK] saved:", out_path)
    print("[OK] video seconds approx:", len(dof) * max(1, args.repeat) / args.output_fps)


if __name__ == "__main__":
    main()
