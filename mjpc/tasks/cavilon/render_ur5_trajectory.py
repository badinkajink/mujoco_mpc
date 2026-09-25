#!/usr/bin/env python3
"""Render a UR5e kinematically following the PID117 flat-pad trajectory."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

os.environ.setdefault("MUJOCO_GL", "egl")

import cv2
import mujoco
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from build_contact_replay import encode_h264, load_geometry, write_hfield_png
from build_tip_model_sweep import PAD_THICKNESS
from render_flat_pad_augmented import insertion_site_local, map_point
from render_ur5_pad_feasibility import build_scene
from run_pad_angle_sweep import oriented_trajectory


SURFACE_ORIGIN = np.array([.55, 0., .25])


def decorate_scene(scene: Path, site: np.ndarray, geometry: dict[str, object]) -> None:
    tree = ET.parse(scene); root = tree.getroot(); worldbody = root.find("worldbody")
    assert worldbody is not None
    world_site = site + SURFACE_ORIGIN
    marker = ET.SubElement(worldbody, "body", {
        "name": "insertion_site_marker",
        "pos": " ".join(f"{v:.9f}" for v in world_site),
    })
    ET.SubElement(marker, "geom", {"type": "cylinder", "pos": "0 0 .0007", "size": ".0055 .0007",
                                    "rgba": "1 .14 .03 .9", "contype": "0", "conaffinity": "0"})
    ET.SubElement(marker, "geom", {"type": "capsule", "fromto": "0 0 .002 0 0 .026", "size": ".0015",
                                    "rgba": "1 .14 .03 1", "contype": "0", "conaffinity": "0"})
    ET.SubElement(marker, "geom", {"type": "sphere", "pos": "0 0 .029", "size": ".0035",
                                    "rgba": "1 .72 .05 1", "contype": "0", "conaffinity": "0"})
    for index, point in enumerate(np.asarray(geometry["path_xy"])[::12]):
        z_index = min(index * 12, len(geometry["path_z"]) - 1)
        position = SURFACE_ORIGIN + np.array([point[0], point[1], geometry["path_z"][z_index] + .0015])
        ET.SubElement(worldbody, "geom", {"name": f"reference_path_{index}", "type": "sphere",
                                           "pos": " ".join(f"{v:.9f}" for v in position), "size": ".0014",
                                           "rgba": ".12 .86 .95 .75", "contype": "0", "conaffinity": "0"})
    tree.write(scene, encoding="unicode")


def desired_pose(point: dict[str, float], height_fn: RegularGridInterpolator,
                 indentation: float, fixed_yaw: float) -> tuple[np.ndarray, np.ndarray]:
    x, y = float(point["x"]), float(point["y"])
    normal = np.array([point["surface_normal_x"], point["surface_normal_y"], point["surface_normal_z"]])
    normal /= np.linalg.norm(normal)
    surface_z = float(height_fn([[y, x]])[0])
    nominal_center = np.array([x, y, surface_z]) + normal * (PAD_THICKNESS - indentation)
    actual_center = np.array([point["x"], point["y"], point["z"]])
    lift = max(0., float(np.dot(actual_center - nominal_center, normal)))
    target_pos = SURFACE_ORIGIN + np.array([x, y, surface_z]) + normal * (.001 + lift)

    # A fixed in-plane orientation avoids inventing human wrist pose and keeps
    # segment transitions continuous.  Only the face normal follows surface.
    yaw = fixed_yaw
    x_axis = np.array([np.cos(yaw), np.sin(yaw), 0.])
    x_axis -= normal * np.dot(x_axis, normal); x_axis /= np.linalg.norm(x_axis)
    z_axis = -normal
    y_axis = np.cross(z_axis, x_axis); y_axis /= np.linalg.norm(y_axis)
    x_axis = np.cross(y_axis, z_axis); x_axis /= np.linalg.norm(x_axis)
    rotation = np.column_stack([x_axis, y_axis, z_axis])
    quat = np.empty(4); mujoco.mju_mat2Quat(quat, rotation.ravel())
    return target_pos, quat


def solve_pose(model: mujoco.MjModel, data: mujoco.MjData, site_id: int,
               target_pos: np.ndarray, target_quat: np.ndarray,
               seed: np.ndarray, local_step: bool) -> tuple[np.ndarray, dict[str, object]]:
    matrix_flat = np.empty(9); mujoco.mju_quat2Mat(matrix_flat, target_quat)
    target_matrix = matrix_flat.reshape(3, 3)

    def task_error(qpos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        data.qpos[:] = qpos; mujoco.mj_forward(model, data)
        current = data.site_xmat[site_id].reshape(3, 3)
        position = target_pos - data.site_xpos[site_id]
        # The earlier cross-product residual is sin(angle), so it incorrectly
        # reports zero at 180 degrees. SO(3) log has the correct angle norm.
        rotation = Rotation.from_matrix(target_matrix.T @ current).as_rotvec()
        return position, rotation

    def residual(qpos: np.ndarray) -> np.ndarray:
        position, rotation = task_error(qpos)
        return np.r_[position, .30 * rotation, .004 * (qpos - seed)]

    lower = np.asarray(model.jnt_range[:, 0], dtype=float)
    upper = np.asarray(model.jnt_range[:, 1], dtype=float)
    # Keep the sequential IK on one continuous branch.  The 80 mrad/frame
    # trust region is a visualization/feasibility guard, not a robot limit.
    if local_step:
        lower = np.maximum(lower, seed - .08)
        upper = np.minimum(upper, seed + .08)
    result = least_squares(residual, seed, bounds=(lower, upper), max_nfev=120,
                           xtol=2e-9, ftol=2e-9, gtol=2e-9)
    position, rotation = task_error(result.x)
    return result.x.copy(), {
        "success": bool(result.success), "nfev": int(result.nfev),
        "position_error_mm": float(np.linalg.norm(position) * 1000),
        "orientation_error_deg": float(np.linalg.norm(rotation) * 180 / np.pi),
    }


def draw_map(panel: np.ndarray, geometry: dict[str, object], site: np.ndarray,
             target_local: np.ndarray, traversed: list[np.ndarray]) -> None:
    gx, gy = geometry["gx"], geometry["gy"]
    origin, size = (24, 250), (270, 310)
    cv2.rectangle(panel, origin, (origin[0] + size[0], origin[1] + size[1]), (45, 51, 63), -1)
    path = np.asarray(geometry["path_xy"])
    for a, b in zip(path[:-1], path[1:]):
        cv2.line(panel, map_point(*a, gx, gy, origin, size), map_point(*b, gx, gy, origin, size),
                 (100, 110, 120), 1, cv2.LINE_AA)
    for a, b in zip(traversed[:-1], traversed[1:]):
        cv2.line(panel, map_point(*a, gx, gy, origin, size), map_point(*b, gx, gy, origin, size),
                 (255, 205, 80), 2, cv2.LINE_AA)
    site_px = map_point(float(site[0]), float(site[1]), gx, gy, origin, size)
    target_px = map_point(float(target_local[0]), float(target_local[1]), gx, gy, origin, size)
    cv2.circle(panel, site_px, 8, (45, 70, 255), 2, cv2.LINE_AA)
    cv2.circle(panel, site_px, 3, (30, 180, 255), -1, cv2.LINE_AA)
    cv2.line(panel, site_px, target_px, (210, 210, 210), 1, cv2.LINE_AA)
    cv2.circle(panel, target_px, 5, (255, 205, 80), -1, cv2.LINE_AA)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cldc-root", type=Path, required=True)
    parser.add_argument("--ur5e-xml", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--indentation-mm", type=float, default=.45)
    args = parser.parse_args()
    root = args.cldc_root.resolve(); output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=True)
    geometry = load_geometry(root, 128); site, site_provenance = insertion_site_local(root, geometry)
    hfield = output / "pid117_mannequin_hfield.png"; write_hfield_png(geometry["height"], hfield)
    scene = output / "ur5e_pid117_flat_trajectory_scene.xml"
    build_scene(args.ur5e_xml.resolve(), hfield, geometry, scene); decorate_scene(scene, site, geometry)
    model = mujoco.MjModel.from_xml_path(str(scene)); data = mujoco.MjData(model)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "cavilon_contact")
    trajectory = oriented_trajectory(geometry, args.fps, 0, args.indentation_mm / 1000)
    contact_yaw = np.unwrap([point["yaw_rad"] for point in trajectory if point["phase"] == "contact"])
    median_yaw = float(np.median(contact_yaw))
    fixed_yaw = float(np.arctan2(np.sin(median_yaw), np.cos(median_yaw)))
    height_fn = RegularGridInterpolator((geometry["gy"], geometry["gx"]), geometry["height"],
                                        bounds_error=False, fill_value=None)

    home = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    seed = model.key_qpos[home].copy() if home >= 0 else np.zeros(model.nq)
    qposes, metrics, targets = [], [], []
    for index, point in enumerate(trajectory):
        target_pos, target_quat = desired_pose(point, height_fn, args.indentation_mm / 1000, fixed_yaw)
        seed, result = solve_pose(model, data, site_id, target_pos, target_quat, seed, index > 0)
        qposes.append(seed.copy()); metrics.append(result); targets.append(target_pos)
        if result["position_error_mm"] > 5:
            raise RuntimeError(f"IK failed at frame {index}: {result}")
    qposes_array = np.asarray(qposes); targets_array = np.asarray(targets)
    np.savez_compressed(output / "ur5e_pid117_flat_trajectory_qpos.npz", qpos=qposes_array,
                        target_xyz_m=targets_array, fps=args.fps)

    renderer = mujoco.Renderer(model, height=720, width=960)
    temp = output / "ur5e_pid117_flat_trajectory.mp4v.mp4"
    final = output / "ur5e_pid117_flat_trajectory.mp4"
    poster = output / "ur5e_pid117_flat_trajectory.jpg"
    writer = cv2.VideoWriter(str(temp), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (1280, 720))
    if not writer.isOpened(): raise RuntimeError(temp)
    traversed: list[np.ndarray] = []; poster_frame = None
    rows = []
    try:
        for index, (qpos, point, target, metric) in enumerate(zip(qposes_array, trajectory, targets_array, metrics)):
            data.qpos[:] = qpos
            if model.nu: data.ctrl[:] = qpos[:model.nu]
            mujoco.mj_forward(model, data)
            local_target = target - SURFACE_ORIGIN
            if point["phase"] == "contact": traversed.append(local_target[:2].copy())
            site_distance = float(np.linalg.norm(local_target - site) * 1000)
            camera = mujoco.MjvCamera(); camera.type = mujoco.mjtCamera.mjCAMERA_FREE
            camera.lookat[:] = [.31, 0, .34]; camera.distance = 1.20; camera.azimuth = 114; camera.elevation = -23
            renderer.update_scene(data, camera=camera)
            image = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
            cv2.rectangle(image, (0, 0), (960, 104), (18, 23, 31), -1)
            cv2.putText(image, "UR5e follows PID117 flat-pad trajectory", (22, 33), cv2.FONT_HERSHEY_SIMPLEX,
                        .73, (245, 245, 245), 2, cv2.LINE_AA)
            cv2.putText(image, f"t={index/args.fps:5.1f}s  {point['phase']}  segment {point['segment']}/6  site distance={site_distance:.1f} mm",
                        (22, 65), cv2.FONT_HERSHEY_SIMPLEX, .48, (220, 225, 232), 1, cv2.LINE_AA)
            cv2.putText(image, "KINEMATIC REPLAY - no controller, collision, or force validation",
                        (22, 91), cv2.FONT_HERSHEY_SIMPLEX, .43, (90, 205, 255), 1, cv2.LINE_AA)
            panel = np.full((720, 320, 3), (23, 28, 37), dtype=np.uint8)
            cv2.putText(panel, "ROBOT TASK STATE", (20, 34), cv2.FONT_HERSHEY_SIMPLEX, .62, (245,245,245), 2, cv2.LINE_AA)
            lines = [f"site distance   {site_distance:6.1f} mm",
                     f"IK position err {metric['position_error_mm']:6.2f} mm",
                     f"IK rotation err {metric['orientation_error_deg']:6.2f} deg",
                     "red pin = insertion site",
                     "cyan beads = reference path"]
            for row_index, line in enumerate(lines):
                cv2.putText(panel, line, (20, 77 + 31 * row_index), cv2.FONT_HERSHEY_SIMPLEX, .43,
                            (215, 221, 230) if row_index < 3 else (110, 205, 255), 1, cv2.LINE_AA)
            cv2.putText(panel, "TOP-DOWN TARGET", (24, 239), cv2.FONT_HERSHEY_SIMPLEX, .42, (230,230,230), 1, cv2.LINE_AA)
            draw_map(panel, geometry, site, local_target, traversed)
            cv2.putText(panel, "orange=traversed  red=site", (24, 590), cv2.FONT_HERSHEY_SIMPLEX, .39,
                        (130,204,255), 1, cv2.LINE_AA)
            cv2.putText(panel, "Flat-facing policy", (20, 650), cv2.FONT_HERSHEY_SIMPLEX, .54, (245,245,245), 1, cv2.LINE_AA)
            cv2.putText(panel, "normal: local surface", (20, 677), cv2.FONT_HERSHEY_SIMPLEX, .41, (165,174,186), 1, cv2.LINE_AA)
            cv2.putText(panel, "yaw: fixed (no human pose claim)", (20, 701), cv2.FONT_HERSHEY_SIMPLEX, .41, (165,174,186), 1, cv2.LINE_AA)
            composed = np.hstack([image, panel]); writer.write(composed)
            if poster_frame is None and point["phase"] == "contact" and len(traversed) > args.fps * 2:
                poster_frame = composed.copy()
            rows.append({"frame": index, "time_s": index/args.fps, "phase": point["phase"],
                         "segment": point["segment"], "site_distance_mm": site_distance, **metric})
    finally:
        renderer.close(); writer.release()
    encode_h264(temp, final, args.fps)
    if poster_frame is None: poster_frame = composed
    cv2.imwrite(str(poster), poster_frame)
    trace = output / "ur5e_pid117_flat_trajectory_ik.csv"
    with trace.open("w", newline="") as handle:
        csv_writer = csv.DictWriter(handle, fieldnames=list(rows[0])); csv_writer.writeheader(); csv_writer.writerows(rows)
    qdot = np.diff(np.unwrap(qposes_array, axis=0), axis=0) * args.fps
    summary = {
        "schema_version": 1, "mujoco_version": mujoco.__version__,
        "source_ur5e_xml": str(args.ur5e_xml.resolve()),
        "source_ur5e_sha256": hashlib.sha256(args.ur5e_xml.read_bytes()).hexdigest(),
        "insertion_site": site_provenance,
        "policy": "pad normal follows local hfield normal; yaw is fixed to the median path heading",
        "fixed_yaw_rad": fixed_yaw,
        "frames": len(rows), "duration_s": (len(rows) - 1) / args.fps,
        "ik": {"max_position_error_mm": float(max(r["position_error_mm"] for r in rows)),
               "median_position_error_mm": float(np.median([r["position_error_mm"] for r in rows])),
               "max_orientation_error_deg": float(max(r["orientation_error_deg"] for r in rows)),
               "solver_success_frames": int(sum(r["success"] for r in rows)),
               "max_abs_joint_speed_rad_s": float(np.max(np.abs(qdot)))},
        "outputs": {"scene": scene.name, "video": final.name, "poster": poster.name,
                    "ik_trace": trace.name, "qpos": "ur5e_pid117_flat_trajectory_qpos.npz"},
        "claim_boundary": "Kinematic visualization and nominal reachability only; no closed-loop control, collision-free motion, force regulation, or real-robot feasibility claim.",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
