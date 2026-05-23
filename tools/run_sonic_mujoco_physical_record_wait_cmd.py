#!/usr/bin/env python3
import os
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import argparse
import time
from pathlib import Path

import cv2
import mujoco

from gear_sonic.utils.mujoco_sim.configs import SimLoopConfig
from gear_sonic.utils.mujoco_sim.base_sim import BaseSimulator


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--duration", type=float, default=180.0)
    ap.add_argument("--drop-after-cmd-sec", type=float, default=4.0)
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--distance", type=float, default=3.0)
    ap.add_argument("--azimuth", type=float, default=120.0)
    ap.add_argument("--elevation", type=float, default=-20.0)
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance = args.distance
    cam.azimuth = args.azimuth
    cam.elevation = args.elevation
    cam.lookat[:] = [0.0, 0.0, 0.75]

    cfg = SimLoopConfig(
        interface="sim",
        enable_onscreen=False,
        enable_offscreen=True,
        enable_image_publish=False,
        sim_frequency=200,
        verbose=False,
        env_name="default",
    )

    wbc = cfg.load_wbc_yaml()
    wbc["ENABLE_ONSCREEN"] = False
    wbc["ENABLE_OFFSCREEN"] = True
    wbc["INTERFACE"] = "lo"
    wbc["ENABLE_ELASTIC_BAND"] = True
    wbc["USE_JOYSTICK"] = 0

    camera_configs = {
        "track": {
            "height": args.height,
            "width": args.width,
            "params": cam,
        }
    }

    print("[INFO] Starting headless MuJoCo PHYSICAL sim.")
    print("[INFO] This script will NOT drop until deploy.sh sends lowcmd.")
    print("[INFO] Terminal 2: run deploy.sh, press ], wait for LOWCMD, then press T after drop.")
    print("[INFO] Video:", out_path, flush=True)

    sim = BaseSimulator(
        config=wbc,
        env_name="default",
        onscreen=False,
        offscreen=True,
        camera_configs=camera_configs,
    )

    # Make sure elastic band starts ON.
    try:
        if sim.sim_env.elastic_band and not sim.sim_env.elastic_band.enable:
            sim.sim_env.handle_keyboard_button("9")
        print("[INFO] ElasticBand initial enable =", sim.sim_env.elastic_band.enable, flush=True)
    except Exception as e:
        print("[WARN] cannot check elastic band:", e, flush=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, args.fps, (args.width, args.height))
    if not writer.isOpened():
        raise RuntimeError("cv2.VideoWriter failed.")

    sim_dt = float(wbc["SIMULATE_DT"])
    frame_dt = 1.0 / args.fps
    next_frame_t = 0.0

    first_cmd_time = None
    dropped = False
    last_status_sec = -1
    frame_count = 0
    start = time.monotonic()

    try:
        while True:
            loop_start = time.monotonic()
            elapsed = loop_start - start
            if elapsed >= args.duration:
                break

            sim.sim_env.sim_step()

            lowcmd = False
            try:
                lowcmd = bool(sim.unitree_bridge.low_cmd_received)
            except Exception:
                lowcmd = False

            if lowcmd and first_cmd_time is None:
                first_cmd_time = elapsed
                print(f"[OK] LOWCMD received from deploy.sh at t={elapsed:.2f}s", flush=True)
                print(f"[INFO] Will drop after {args.drop_after_cmd_sec:.1f}s. Do not press T until drop happens.", flush=True)

            if first_cmd_time is not None and (not dropped):
                if elapsed - first_cmd_time >= args.drop_after_cmd_sec:
                    try:
                        sim.sim_env.handle_keyboard_button("9")
                        print(f"[OK] Auto key 9 / drop robot at t={elapsed:.2f}s", flush=True)
                    except Exception as e:
                        print("[WARN] auto key 9 failed:", e, flush=True)
                    dropped = True

            sec = int(elapsed)
            if sec != last_status_sec and sec % 2 == 0:
                last_status_sec = sec
                print(
                    f"[STATUS] t={elapsed:.1f}s lowcmd={lowcmd} dropped={dropped}",
                    flush=True,
                )

            if elapsed >= next_frame_t:
                try:
                    pelvis = sim.sim_env.mj_data.body("pelvis").xpos.copy()
                    cam.lookat[:] = [pelvis[0], pelvis[1], max(0.55, pelvis[2] + 0.10)]
                except Exception:
                    pass

                frames = sim.sim_env.update_render_caches()
                rgb = frames["track_image"]
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

                cv2.putText(
                    bgr,
                    f"t={elapsed:05.2f}s lowcmd={int(lowcmd)} drop={int(dropped)}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                writer.write(bgr)
                frame_count += 1
                next_frame_t += frame_dt

            sleep_time = sim_dt - (time.monotonic() - loop_start)
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("[INFO] interrupted")
    finally:
        writer.release()
        sim.close()
        print("[OK] saved:", out_path)
        print("[OK] frames:", frame_count)


if __name__ == "__main__":
    main()
