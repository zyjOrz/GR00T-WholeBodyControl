#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np

ARM_IDXS = [5, 11, 16, 17, 18, 19, 20, 21, 23, 24, 25, 26, 27, 28]
WAIST_IDXS = [4, 10, 22]
LEG_IDXS = [0, 1, 2, 3, 6, 7, 8, 9, 12, 13, 14, 15]

def load_csv(p):
    arr = np.loadtxt(p, delimiter=",", skiprows=1)
    if arr.ndim == 1:
        arr = arr[None, :]
    return arr

ap = argparse.ArgumentParser()
ap.add_argument("base")
args = ap.parse_args()

base = Path(args.base)
dirs = [d for d in base.iterdir() if d.is_dir()]
print("[base]", base)
print("[motion dirs]", [d.name for d in dirs])

if len(dirs) != 1:
    raise SystemExit(f"Expected exactly 1 motion dir, got {len(dirs)}")

d = dirs[0]
required = [
    "joint_pos.csv", "joint_vel.csv",
    "body_pos.csv", "body_quat.csv",
    "body_lin_vel.csv", "body_ang_vel.csv",
    "metadata.txt",
]
for f in required:
    p = d / f
    print(f, "OK" if p.exists() else "MISSING")
    if not p.exists():
        raise SystemExit(1)

jp = load_csv(d / "joint_pos.csv")
jv = load_csv(d / "joint_vel.csv")
bp = load_csv(d / "body_pos.csv")
bq = load_csv(d / "body_quat.csv")

print("joint_pos shape:", jp.shape, "finite:", np.isfinite(jp).all())
print("joint_vel shape:", jv.shape, "finite:", np.isfinite(jv).all())
print("body_pos shape:", bp.shape, "first:", bp[0], "last:", bp[-1])
print("body_quat shape:", bq.shape, "first:", bq[0], "norm:", np.linalg.norm(bq, axis=1).min(), np.linalg.norm(bq, axis=1).max())

for name, idxs in [("leg", LEG_IDXS), ("waist", WAIST_IDXS), ("arm", ARM_IDXS)]:
    r = jp[:, idxs].max(axis=0) - jp[:, idxs].min(axis=0)
    print(f"{name} max range:", float(r.max()))
    print(f"{name} per-index range:", dict(zip(idxs, np.round(r, 4))))

print("[metadata]")
print((d / "metadata.txt").read_text())
