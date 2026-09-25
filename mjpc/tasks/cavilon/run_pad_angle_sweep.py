#!/usr/bin/env python3
"""Sweep oblique contact angles for the reduced-order compliant Cavilon pad."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from build_contact_replay import build_trajectory, load_geometry, write_hfield_png
from build_tip_model_sweep import PAD_LENGTH, PAD_THICKNESS, pad_xml, render, scene_xml


ANGLES_DEG = (0, 15, 30, 45, 60)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def stroke_headings(trajectory: list[dict[str, float]]) -> np.ndarray:
    x = np.asarray([point["x"] for point in trajectory])
    y = np.asarray([point["y"] for point in trajectory])
    dx = np.gradient(x); dy = np.gradient(y)
    speed = np.hypot(dx, dy)
    valid = speed > 1e-6
    indices = np.arange(len(x))
    if valid.sum() < 2:
        return np.zeros(len(x))
    measured = np.unwrap(np.arctan2(dy[valid], dx[valid]))
    return np.interp(indices, indices[valid], measured)


def oriented_trajectory(geometry: dict[str, object], fps: int, tilt_deg: float,
                        indentation_m: float) -> list[dict[str, float]]:
    tilt = np.deg2rad(tilt_deg)
    support_offset = PAD_THICKNESS*np.cos(tilt) + PAD_LENGTH/2*np.sin(abs(tilt))
    trajectory = build_trajectory(
        geometry["path"], geometry["path_xy"], geometry["path_z"], fps,
        tip_radius=support_offset, indentation=indentation_m,
    )
    gx, gy, height = geometry["gx"], geometry["gy"], geometry["height"]
    dhdy, dhdx = np.gradient(height, gy, gx)
    height_fn = RegularGridInterpolator((gy, gx), height, bounds_error=False, fill_value=None)
    dx_fn = RegularGridInterpolator((gy, gx), dhdx, bounds_error=False, fill_value=None)
    dy_fn = RegularGridInterpolator((gy, gx), dhdy, bounds_error=False, fill_value=None)
    headings = stroke_headings(trajectory)

    for index, point in enumerate(trajectory):
        x, y = float(point["x"]), float(point["y"])
        query = np.array([[y, x]])
        surface_z = float(height_fn(query)[0])
        slope_x = float(dx_fn(query)[0]); slope_y = float(dy_fn(query)[0])
        normal = np.array([-slope_x, -slope_y, 1.0]); normal /= np.linalg.norm(normal)
        yaw = float(headings[index])
        tangent_x = np.array([np.cos(yaw), np.sin(yaw),
                              slope_x*np.cos(yaw) + slope_y*np.sin(yaw)])
        tangent_x -= normal*np.dot(tangent_x, normal)
        tangent_x /= np.linalg.norm(tangent_x)
        tangent_y = np.cross(normal, tangent_x); tangent_y /= np.linalg.norm(tangent_y)
        tilted_x = np.cos(tilt)*tangent_x - np.sin(tilt)*normal
        tilted_z = np.sin(tilt)*tangent_x + np.cos(tilt)*normal
        rotation = np.column_stack([tilted_x, tangent_y, tilted_z])
        quat = np.empty(4); mujoco.mju_mat2Quat(quat, rotation.ravel())

        baseline_z = surface_z + support_offset - indentation_m
        lift = max(0.0, float(point["z"] - baseline_z))
        position = np.array([x, y, surface_z]) + normal*(support_offset - indentation_m + lift)
        point.update({
            "x": float(position[0]), "y": float(position[1]), "z": float(position[2]),
            "quat": quat.tolist(), "yaw_rad": float(np.arctan2(tilted_x[1], tilted_x[0])),
            "tilt_rad": float(tilt), "tilt_deg": float(tilt_deg),
            "surface_normal_x": float(normal[0]), "surface_normal_y": float(normal[1]),
            "surface_normal_z": float(normal[2]),
        })
    return trajectory


def make_montage(videos: list[Path], destination: Path) -> None:
    command = ["ffmpeg", "-y", "-loglevel", "error"]
    for video in videos:
        command += ["-i", str(video)]
    filters = ";".join(f"[{i}:v]scale=480:360[v{i}]" for i in range(len(videos)))
    filters += ";[v0][v1][v2][v3][v4]xstack=inputs=5:layout=0_0|480_0|960_0|240_360|720_360:fill=black[v]"
    command += ["-filter_complex", filters, "-map", "[v]", "-c:v", "libx264", "-crf", "21",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(destination)]
    subprocess.run(command, check=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cldc-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--indentation-mm", type=float, default=.45)
    args = parser.parse_args()
    root = args.cldc_root.resolve(); output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=True)
    geometry = load_geometry(root, 128)
    hfield = output / "pid117_mannequin_hfield.png"; write_hfield_png(geometry["height"], hfield)
    _, _, contact_names = pad_xml("compliant_pad")
    results = []
    close_videos = []
    for angle in ANGLES_DEG:
        angle_dir = output / f"angle_{angle:02d}"; angle_dir.mkdir(exist_ok=True)
        linked_hfield = angle_dir / hfield.name
        if linked_hfield.exists() or linked_hfield.is_symlink(): linked_hfield.unlink()
        linked_hfield.symlink_to(hfield)
        trajectory = oriented_trajectory(geometry, args.fps, angle, args.indentation_mm/1000)
        xml, _ = scene_xml(hfield.name, geometry, "compliant_pad", trajectory[0])
        scene = angle_dir / f"pid117_compliant_pad_{angle:02d}deg_scene.xml"; scene.write_text(xml)
        model = mujoco.MjModel.from_xml_path(str(scene))
        result = render(model, trajectory, contact_names, geometry, "compliant_pad", angle_dir, args.fps)
        trace = angle_dir / f"pid117_compliant_pad_{angle:02d}deg_trace.csv"
        with trace.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(result["records"][0])); writer.writeheader(); writer.writerows(result["records"])
        contact = [row for row in result["records"] if row["phase"] == "contact"]
        forces = np.asarray([row["normal_force_n"] for row in contact])
        active = np.asarray([row["active_cells"] for row in contact])
        close_videos.append(result["videos"]["close"])
        results.append({
            "tilt_deg": angle, "contact_frames": int((active > 0).sum()),
            "intended_contact_frames": len(contact), "active_cells_median": float(np.median(active)),
            "active_cells_p95": float(np.quantile(active, .95)), "force_median_n": float(np.median(forces)),
            "force_p95_n": float(np.quantile(forces, .95)), "force_max_n": float(forces.max()),
            "final_coverage_pct": float(result["final_coverage_pct"]),
            "scene": str(scene.relative_to(output)), "trace": str(trace.relative_to(output)),
            "close_video": str(result["videos"]["close"].relative_to(output)),
            "top_video": str(result["videos"]["top"].relative_to(output)),
        })
    montage = output / "pid117_compliant_pad_angle_sweep.mp4"
    make_montage(close_videos, montage)
    table = output / "angle_sensitivity.csv"
    with table.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0])); writer.writeheader(); writer.writerows(results)
    artifacts = []
    for path in sorted(p for p in output.rglob("*") if p.is_file() and not p.is_symlink()):
        artifacts.append({"path": str(path.relative_to(output)), "bytes": path.stat().st_size, "sha256": sha256(path)})
    summary = {
        "schema_version": 1, "participant": "PID117", "mujoco_version": mujoco.__version__,
        "controlled_variable": "pad-to-local-surface tilt about the stroke-lateral axis",
        "fixed_conditions": {"angles_deg": list(ANGLES_DEG), "yaw": "instantaneous stroke direction",
                             "surface_orientation": "local height-field normal", "indentation_mm": args.indentation_mm,
                             "pad_m": [PAD_LENGTH, .009, PAD_THICKNESS], "compliance_grid": [3, 5]},
        "results": results, "montage_video": montage.name, "sensitivity_csv": table.name,
        "claim_boundary": [
            "Angles are controlled sensitivity conditions, not estimates of the human applicator pose.",
            "Force and coverage remain uncalibrated simulation outputs.",
            "Yaw follows the estimated path derivative; it is not recovered independently from video.",
        ],
        "artifacts": artifacts,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"output": str(output), "results": results, "montage": montage.name}, indent=2))
    return 0


if __name__ == "__main__": sys.exit(main())
