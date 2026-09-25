#!/usr/bin/env python3
"""Build and render the first Cavilon MuJoCo contact-replay task.

The script maps a metric monocular surface atlas into a local tangent-frame
height field, replays the selected expert path with a compliant contact proxy,
and exports force/coverage traces and two videos.  It intentionally does not
claim that the source video contains physical contact-force ground truth.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Iterable

os.environ.setdefault("MUJOCO_GL", "egl")

import cv2
import mujoco
import numpy as np
from scipy.interpolate import RegularGridInterpolator, griddata
from scipy.ndimage import gaussian_filter


ATLAS_REL = Path("experiments/contact_atlas_transfer/runs/da3_metric/PID117_atlas.npz")
MASK_REL = Path("data/contact_atlas_transfer_cache/PID117_mannequin_mask.png")
PATH_REL = Path("experiments/contact_surface_3d/PID117_path.csv")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_path(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    numeric = [
        "frame", "time_s", "surface_x_mm", "surface_y_mm", "surface_z_mm",
        "tip_x_mm", "tip_y_mm", "tip_z_mm", "tip_minus_surface_depth_mm",
    ]
    result: dict[str, np.ndarray] = {}
    for name in numeric:
        result[name] = np.asarray([float(row[name]) for row in rows], dtype=float)
    return result


def tangent_frame(points: np.ndarray, path_surface: np.ndarray,
                  path_tip: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    origin = np.median(path_surface, axis=0)
    centered = points - origin
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = vh[-1]
    toward_tip = np.median(path_tip - path_surface, axis=0)
    if np.dot(normal, toward_tip) < 0:
        normal *= -1
    path_delta = path_surface[-1] - path_surface[0]
    x_axis = path_delta - normal * np.dot(path_delta, normal)
    if np.linalg.norm(x_axis) < 1e-6:
        x_axis = vh[0] - normal * np.dot(vh[0], normal)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(normal, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    basis = np.vstack([x_axis, y_axis, normal])
    return origin, basis


def load_geometry(cldc_root: Path, grid_size: int = 128) -> dict[str, object]:
    atlas_path = cldc_root / ATLAS_REL
    mask_path = cldc_root / MASK_REL
    path_path = cldc_root / PATH_REL
    atlas_data = np.load(atlas_path)
    atlas = np.asarray(atlas_data["atlas_m"], dtype=float)
    fx, fy, cx, cy = map(float, atlas_data["camera"])
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(mask_path)
    path = read_path(path_path)
    surface = np.column_stack([
        path["surface_x_mm"], path["surface_y_mm"], path["surface_z_mm"]
    ]) / 1000.0
    tip = np.column_stack([
        path["tip_x_mm"], path["tip_y_mm"], path["tip_z_mm"]
    ]) / 1000.0

    path_u = fx * surface[:, 0] / surface[:, 2] + cx
    path_v = fy * surface[:, 1] / surface[:, 2] + cy
    pad = 60
    x0 = max(0, int(np.floor(path_u.min() - pad)))
    x1 = min(atlas.shape[1], int(np.ceil(path_u.max() + pad)))
    y0 = max(0, int(np.floor(path_v.min() - pad)))
    y1 = min(atlas.shape[0], int(np.ceil(path_v.max() + pad)))

    vv, uu = np.mgrid[y0:y1:2, x0:x1:2]
    zz = atlas[y0:y1:2, x0:x1:2]
    valid = (mask[y0:y1:2, x0:x1:2] > 0) & np.isfinite(zz) & (zz > 0.2) & (zz < 1.5)
    zz = zz[valid]
    uu = uu[valid]
    vv = vv[valid]
    camera_points = np.column_stack([
        (uu - cx) * zz / fx,
        (vv - cy) * zz / fy,
        zz,
    ])
    origin, basis = tangent_frame(camera_points, surface, tip)
    local_points = (camera_points - origin) @ basis.T
    local_surface = (surface - origin) @ basis.T
    local_tip = (tip - origin) @ basis.T

    margin = 0.018
    x_min = min(np.percentile(local_points[:, 0], 1), local_surface[:, 0].min()) - margin
    x_max = max(np.percentile(local_points[:, 0], 99), local_surface[:, 0].max()) + margin
    y_min = min(np.percentile(local_points[:, 1], 1), local_surface[:, 1].min()) - margin
    y_max = max(np.percentile(local_points[:, 1], 99), local_surface[:, 1].max()) + margin
    gx = np.linspace(x_min, x_max, grid_size)
    gy = np.linspace(y_min, y_max, grid_size)
    xx, yy = np.meshgrid(gx, gy)
    height = griddata(local_points[:, :2], local_points[:, 2], (xx, yy), method="linear")
    missing = ~np.isfinite(height)
    if missing.any():
        height[missing] = griddata(
            local_points[:, :2], local_points[:, 2],
            (xx[missing], yy[missing]), method="nearest"
        )
    height = gaussian_filter(height, sigma=1.0)
    lo, hi = np.percentile(height, [0.5, 99.5])
    height = np.clip(height, lo, hi)

    surface_height = RegularGridInterpolator(
        (gy, gx), height, bounds_error=False, fill_value=None
    )
    path_xy = local_surface[:, :2]
    path_z = surface_height(np.column_stack([path_xy[:, 1], path_xy[:, 0]]))
    return {
        "atlas_path": atlas_path,
        "mask_path": mask_path,
        "path_path": path_path,
        "path": path,
        "origin": origin,
        "basis": basis,
        "gx": gx,
        "gy": gy,
        "height": height,
        "path_xy": path_xy,
        "path_z": path_z,
        "local_tip": local_tip,
        "roi_pixels": [x0, y0, x1, y1],
    }


def write_hfield_png(height: np.ndarray, path: Path) -> tuple[float, float]:
    z_min = float(height.min())
    z_range = max(float(height.max() - z_min), 1e-4)
    # MuJoCo's image loader maps the first PNG row to +Y.  Our grid is stored
    # with ascending Y, so flip it here to preserve the local-frame geometry.
    pixels = np.round((height - z_min) / z_range * 65535).astype(np.uint16)[::-1]
    if not cv2.imwrite(str(path), pixels):
        raise RuntimeError(f"Could not write {path}")
    return z_min, z_range


def scene_xml(hfield_name: str, x_center: float, y_center: float,
              x_half: float, y_half: float, z_min: float, z_range: float,
              start: Iterable[float]) -> str:
    sx, sy, sz = start
    return f"""<mujoco model="Cavilon PID117 contact replay">
  <compiler angle="radian"/>
  <option timestep="0.002" integrator="implicitfast" gravity="0 0 -9.81">
    <flag contact="enable"/>
  </option>
  <size nconmax="100" njmax="500"/>
  <visual>
    <global offwidth="960" offheight="720"/>
    <headlight diffuse="0.72 0.72 0.72" ambient="0.28 0.28 0.28" specular="0.2 0.2 0.2"/>
  </visual>
  <default>
    <geom solref="0.03 1" solimp="0.9 0.98 0.003" friction="0.8 0.02 0.002"/>
  </default>
  <asset>
    <hfield name="mannequin_hf" file="{hfield_name}" size="{x_half:.9f} {y_half:.9f} {z_range:.9f} 0.012"/>
    <material name="skin" rgba="0.80 0.56 0.43 1" specular="0.12" shininess="0.25"/>
    <material name="applicator" rgba="0.12 0.47 0.88 1" specular="0.35"/>
  </asset>
  <worldbody>
    <light pos="0 -0.2 0.45" dir="0 0.35 -1" directional="true" castshadow="true"/>
    <geom name="mannequin" type="hfield" hfield="mannequin_hf" pos="{x_center:.9f} {y_center:.9f} {z_min:.9f}" material="skin" contype="1" conaffinity="1"/>
    <body name="tool_target" mocap="true" pos="{sx:.9f} {sy:.9f} {sz:.9f}"/>
    <body name="tool" pos="{sx:.9f} {sy:.9f} {sz:.9f}">
      <freejoint/>
      <inertial pos="0 0 0.035" mass="0.12" diaginertia="0.00008 0.00008 0.00002"/>
      <geom name="tool_tip" type="sphere" size="0.008" rgba="0.97 0.76 0.10 1" priority="1" contype="1" conaffinity="1"/>
      <geom name="tool_shaft" type="cylinder" pos="0 0 0.038" size="0.006 0.032" material="applicator" contype="0" conaffinity="0"/>
      <geom name="tool_cap" type="cylinder" pos="0 0 0.073" size="0.011 0.004" rgba="0.08 0.22 0.42 1" contype="0" conaffinity="0"/>
      <site name="tip_site" pos="0 0 -0.008" size="0.002" rgba="1 0.2 0.05 1"/>
    </body>
  </worldbody>
  <equality>
    <weld name="compliant_cartesian_servo" body1="tool_target" body2="tool" solref="0.04 1" solimp="0.85 0.98 0.002"/>
  </equality>
</mujoco>
"""


def split_segments(frames: np.ndarray) -> list[tuple[int, int]]:
    starts = np.r_[0, np.flatnonzero(np.diff(frames) > 3) + 1]
    ends = np.r_[starts[1:], len(frames)]
    return [(int(a), int(b)) for a, b in zip(starts, ends)]


def build_trajectory(path: dict[str, np.ndarray], path_xy: np.ndarray,
                     path_z: np.ndarray, fps: int, tip_radius: float,
                     indentation: float) -> list[dict[str, float]]:
    result: list[dict[str, float]] = []
    segments = split_segments(path["frame"])
    previous = None
    for segment_index, (a, b) in enumerate(segments, start=1):
        times = path["time_s"][a:b]
        duration = max(float(times[-1] - times[0]), 1 / fps)
        sample_t = np.linspace(times[0], times[-1], max(2, int(round(duration * fps)) + 1))
        xs = np.interp(sample_t, times, path_xy[a:b, 0])
        ys = np.interp(sample_t, times, path_xy[a:b, 1])
        zs = np.interp(sample_t, times, path_z[a:b]) + tip_radius - indentation
        if previous is None:
            for alpha in np.linspace(0, 1, fps, endpoint=False):
                result.append({"x": xs[0], "y": ys[0], "z": zs[0] + (1-alpha) * 0.025,
                               "segment": 0, "phase": "approach"})
        else:
            transition_n = max(2, int(0.55 * fps))
            for alpha in np.linspace(0, 1, transition_n, endpoint=False):
                lift = 0.018 + 0.012 * np.sin(np.pi * alpha)
                result.append({
                    "x": (1-alpha) * previous[0] + alpha * xs[0],
                    "y": (1-alpha) * previous[1] + alpha * ys[0],
                    "z": (1-alpha) * previous[2] + alpha * zs[0] + lift,
                    "segment": 0, "phase": "transfer",
                })
        for x, y, z in zip(xs, ys, zs):
            result.append({"x": float(x), "y": float(y), "z": float(z),
                           "segment": segment_index, "phase": "contact"})
        previous = (float(xs[-1]), float(ys[-1]), float(zs[-1]))
    for alpha in np.linspace(0, 1, fps, endpoint=True):
        result.append({"x": previous[0], "y": previous[1], "z": previous[2] + alpha * 0.025,
                       "segment": 0, "phase": "retract"})
    return result


def overlay(frame: np.ndarray, title: str, index: int, total: int,
            force: float, coverage_pct: float, phase: str, segment: int,
            target: np.ndarray, covered: np.ndarray) -> np.ndarray:
    image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.rectangle(image, (0, 0), (image.shape[1], 92), (18, 23, 31), -1)
    cv2.putText(image, title, (24, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.73, (245, 245, 245), 2, cv2.LINE_AA)
    detail = f"t={index/30:5.1f}s  phase={phase}  segment={segment}/6  normal force={force:5.2f} N  covered={coverage_pct:5.1f}%"
    cv2.putText(image, detail, (24, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 225, 232), 1, cv2.LINE_AA)
    cv2.putText(image, "SIMULATED ORACLE - rigid surface, nominal 8 mm tip", (24, 86),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (110, 205, 255), 1, cv2.LINE_AA)

    inset = np.zeros((*target.shape, 3), dtype=np.uint8)
    inset[target] = (90, 90, 90)
    inset[covered] = (40, 205, 255)
    inset = cv2.resize(inset, (185, 185), interpolation=cv2.INTER_NEAREST)
    x0, y0 = image.shape[1] - 205, image.shape[0] - 205
    cv2.rectangle(image, (x0-5, y0-25), (x0+190, y0+190), (18, 23, 31), -1)
    cv2.putText(image, "coverage accumulator", (x0, y0-7), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                (230, 230, 230), 1, cv2.LINE_AA)
    image[y0:y0+185, x0:x0+185] = inset
    return image


def encode_h264(source: Path, destination: Path, fps: int) -> None:
    command = [
        "ffmpeg", "-y", "-loglevel", "error", "-i", str(source),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "21",
        "-movflags", "+faststart", "-r", str(fps), str(destination),
    ]
    try:
        subprocess.run(command, check=True)
        source.unlink()
    except (FileNotFoundError, subprocess.CalledProcessError):
        source.replace(destination)


def render(output_dir: Path, model: mujoco.MjModel, trajectory: list[dict[str, float]],
           gx: np.ndarray, gy: np.ndarray, target: np.ndarray, fps: int) -> dict[str, object]:
    data = mujoco.MjData(model)
    first = trajectory[0]
    start = np.array([first["x"], first["y"], first["z"]])
    data.qpos[:3] = start
    data.qpos[3:7] = [1, 0, 0, 0]
    data.mocap_pos[0] = start
    data.mocap_quat[0] = [1, 0, 0, 0]
    mujoco.mj_forward(model, data)

    width, height_px = 960, 720
    renderer = mujoco.Renderer(model, height=height_px, width=width)
    center = np.array([(gx[0] + gx[-1]) / 2, (gy[0] + gy[-1]) / 2, 0.02])
    extent = max(gx[-1] - gx[0], gy[-1] - gy[0])
    cameras = {
        "close": (118.0, -34.0, max(0.25, extent * 1.45)),
        "top": (90.0, -89.0, max(0.27, extent * 1.25)),
    }
    writers: dict[str, cv2.VideoWriter] = {}
    temporary: dict[str, Path] = {}
    for name in cameras:
        tmp = output_dir / f"pid117_contact_replay_{name}.mp4v.mp4"
        temporary[name] = tmp
        writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height_px))
        if not writer.isOpened():
            raise RuntimeError(f"Could not open video writer for {tmp}")
        writers[name] = writer

    covered = np.zeros_like(target, dtype=bool)
    yy, xx = np.meshgrid(gy, gx, indexing="ij")
    target_count = int(target.sum())
    traces: list[dict[str, object]] = []
    mannequin_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "mannequin")
    tip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "tool_tip")
    tool_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "tool")
    substeps = max(1, int(round((1 / fps) / model.opt.timestep)))

    try:
        for index, point in enumerate(trajectory):
            desired = np.array([point["x"], point["y"], point["z"]])
            data.mocap_pos[0] = desired
            normal_force = 0.0
            contact_xy: list[np.ndarray] = []
            for _ in range(substeps):
                mujoco.mj_step(model, data)
                for contact_index in range(data.ncon):
                    contact = data.contact[contact_index]
                    if {int(contact.geom1), int(contact.geom2)} != {mannequin_id, tip_id}:
                        continue
                    force = np.zeros(6)
                    mujoco.mj_contactForce(model, data, contact_index, force)
                    normal_force += max(0.0, float(force[0])) / substeps
                    contact_xy.append(np.array(contact.pos[:2]))
            for position in contact_xy:
                footprint = (xx - position[0]) ** 2 + (yy - position[1]) ** 2 <= 0.008 ** 2
                covered |= footprint & target
            coverage_pct = 100.0 * covered.sum() / max(target_count, 1)
            traces.append({
                "frame": index,
                "time_s": index / fps,
                "phase": point["phase"],
                "segment": int(point["segment"]),
                "target_x_m": float(point["x"]),
                "target_y_m": float(point["y"]),
                "target_z_m": float(point["z"]),
                "tip_x_m": float(data.xpos[tool_body_id][0]),
                "tip_y_m": float(data.xpos[tool_body_id][1]),
                "tip_z_m": float(data.xpos[tool_body_id][2]),
                "normal_force_n": normal_force,
                "coverage_pct": coverage_pct,
                "contacts": len(contact_xy),
            })
            for name, (azimuth, elevation, distance) in cameras.items():
                camera = mujoco.MjvCamera()
                camera.type = mujoco.mjtCamera.mjCAMERA_FREE
                camera.lookat[:] = center
                camera.azimuth = azimuth
                camera.elevation = elevation
                camera.distance = distance
                renderer.update_scene(data, camera=camera)
                frame = renderer.render()
                labeled = overlay(frame, f"PID117 Cavilon contact replay - {name} view", index,
                                  len(trajectory), normal_force, coverage_pct, str(point["phase"]),
                                  int(point["segment"]), target, covered)
                writers[name].write(labeled)
    finally:
        renderer.close()
        for writer in writers.values():
            writer.release()

    videos = {}
    for name, tmp in temporary.items():
        final = output_dir / f"pid117_contact_replay_{name}.mp4"
        encode_h264(tmp, final, fps)
        videos[name] = final.name
    return {"traces": traces, "videos": videos, "final_coverage_pct": float(100*covered.sum()/max(target_count, 1))}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cldc-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--grid-size", type=int, default=128)
    parser.add_argument("--indentation-mm", type=float, default=0.2)
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    geometry = load_geometry(args.cldc_root.resolve(), args.grid_size)
    height = geometry["height"]
    gx, gy = geometry["gx"], geometry["gy"]
    hfield_path = output_dir / "pid117_mannequin_hfield.png"
    z_min, z_range = write_hfield_png(height, hfield_path)

    trajectory = build_trajectory(
        geometry["path"], geometry["path_xy"], geometry["path_z"],
        args.fps, tip_radius=0.008, indentation=args.indentation_mm / 1000,
    )
    start = [trajectory[0]["x"], trajectory[0]["y"], trajectory[0]["z"]]
    xml = scene_xml(hfield_path.name, (gx[-1]+gx[0])/2, (gy[-1]+gy[0])/2,
                    (gx[-1]-gx[0])/2, (gy[-1]-gy[0])/2, z_min, z_range, start)
    scene_path = output_dir / "pid117_cavilon_scene.xml"
    scene_path.write_text(xml)

    path_csv = output_dir / "pid117_reference_path_local.csv"
    with path_csv.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["source_frame", "time_s", "x_m", "y_m", "surface_z_m", "source_visual_gap_mm"])
        source = geometry["path"]
        for i in range(len(source["frame"])):
            writer.writerow([int(source["frame"][i]), source["time_s"][i], geometry["path_xy"][i,0],
                             geometry["path_xy"][i,1], geometry["path_z"][i],
                             source["tip_minus_surface_depth_mm"][i]])

    model = mujoco.MjModel.from_xml_path(str(scene_path))
    render_result = {"traces": [], "videos": {}, "final_coverage_pct": None}
    path_xy = geometry["path_xy"]
    xx, yy = np.meshgrid(gx, gy)
    target = np.zeros_like(height, dtype=bool)
    for x, y in path_xy[::4]:
        target |= (xx-x)**2 + (yy-y)**2 <= 0.012**2
    if not args.no_render:
        render_result = render(output_dir, model, trajectory, gx, gy, target, args.fps)
        trace_path = output_dir / "pid117_simulation_trace.csv"
        with trace_path.open("w", newline="") as handle:
            rows = render_result["traces"]
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    inputs = [geometry["atlas_path"], geometry["mask_path"], geometry["path_path"]]
    source_path = Path(__file__).resolve()
    repository = source_path.parents[3]
    try:
        git_commit = subprocess.check_output(
            ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
        ).strip()
        git_branch = subprocess.check_output(
            ["git", "-C", str(repository), "branch", "--show-current"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        git_commit = "unavailable"
        git_branch = "unavailable"
    summary = {
        "schema_version": 1,
        "participant": "PID117",
        "mujoco_version": mujoco.__version__,
        "model": "compliant Cartesian applicator proxy over rigid reconstructed height field",
        "integration_source": {
            "repository": str(repository),
            "branch": git_branch,
            "base_commit": git_commit,
            "script": str(source_path.relative_to(repository)),
            "script_sha256": sha256(source_path),
        },
        "source_inputs": [{"path": str(p.relative_to(args.cldc_root.resolve())), "sha256": sha256(p)} for p in inputs],
        "surface": {
            "roi_pixels_xyxy": geometry["roi_pixels"],
            "grid_shape": list(height.shape),
            "width_m": float(gx[-1]-gx[0]),
            "height_m": float(gy[-1]-gy[0]),
            "elevation_range_m": z_range,
            "camera_to_local_origin_m": geometry["origin"].tolist(),
            "camera_to_local_basis_rows": geometry["basis"].tolist(),
        },
        "replay": {
            "fps": args.fps,
            "frames": len(trajectory),
            "duration_s": len(trajectory)/args.fps,
            "expert_contact_segments": len(split_segments(geometry["path"]["frame"])),
            "tip_radius_m": 0.008,
            "nominal_indentation_m": args.indentation_mm/1000,
            "final_coverage_pct": render_result["final_coverage_pct"],
        },
        "videos": render_result["videos"],
        "claim_boundary": [
            "MuJoCo contact and force are simulation oracle outputs under the stated model assumptions.",
            "The source visual path and metric atlas are Aim 1 estimates, not physical force or deposited-area ground truth.",
            "The Cartesian proxy is not yet coupled to UR5 reachability, torque limits, or MJPC optimization.",
        ],
    }
    artifact_paths = [hfield_path, scene_path, path_csv]
    trace_path = output_dir / "pid117_simulation_trace.csv"
    if trace_path.exists() and not args.no_render:
        artifact_paths.append(trace_path)
    artifact_paths.extend(output_dir / name for name in render_result["videos"].values())
    summary["artifacts"] = [
        {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in artifact_paths
    ]
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"output_dir": str(output_dir), "videos": render_result["videos"],
                      "frames": len(trajectory), "coverage_pct": render_result["final_coverage_pct"]}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
