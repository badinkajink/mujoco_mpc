#!/usr/bin/env python3
"""Render the adopted flat-facing pad policy with interpretable task state.

This is an oracle simulation visualization.  The insertion site is transferred
from the PID117 depth atlas into the same local tangent frame as the hfield.
Area is the union of projected force-active pad-cell faces. Contact forces
and points are simulator outputs; this finite-cell area estimate remains a
proxy even within the simulator (corner contact does not fill an entire cell).
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

os.environ.setdefault("MUJOCO_GL", "egl")

import cv2
import mujoco
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from build_contact_replay import encode_h264, load_geometry, write_hfield_png
from build_tip_model_sweep import (
    GRID_COLS,
    GRID_ROWS,
    PAD_LENGTH,
    PAD_WIDTH,
    pad_xml,
    scene_xml,
)
from run_pad_angle_sweep import oriented_trajectory


SITE_REL = Path("experiments/tip_depth_fusion_contact/mannequin_site_surface.json")
ATLAS_REL = Path("experiments/contact_atlas_transfer/runs/da3_metric/PID117_atlas.npz")


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def paint_projected_face(accumulator, world_corners, gx, gy):
    """Include grid sample centers inside the convex projected quadrilateral."""
    polygon=world_corners[:,:2]
    x0=max(0,int(np.searchsorted(gx,polygon[:,0].min()-1e-12,side='left')))
    x1=min(len(gx),int(np.searchsorted(gx,polygon[:,0].max()+1e-12,side='right')))
    y0=max(0,int(np.searchsorted(gy,polygon[:,1].min()-1e-12,side='left')))
    y1=min(len(gy),int(np.searchsorted(gy,polygon[:,1].max()+1e-12,side='right')))
    if x1<=x0 or y1<=y0:return
    xx,yy=np.meshgrid(gx[x0:x1],gy[y0:y1]);cross=[]
    for a,b in zip(polygon,np.roll(polygon,-1,axis=0)):
        cross.append((b[0]-a[0])*(yy-a[1])-(b[1]-a[1])*(xx-a[0]))
    cross=np.asarray(cross)
    accumulator[y0:y1,x0:x1] |= np.all(cross>=-1e-12,axis=0)|np.all(cross<=1e-12,axis=0)


def insertion_site_local(root: Path, geometry: dict[str, object]) -> tuple[np.ndarray, dict[str, object]]:
    site_document = json.loads((root / SITE_REL).read_text())
    record = site_document["participants"]["PID117"]
    u, v = map(float, record["site_xy"])
    atlas_data = np.load(root / ATLAS_REL)
    atlas = np.asarray(atlas_data["atlas_m"], dtype=float)
    fx, fy, cx, cy = map(float, atlas_data["camera"])
    ix, iy = int(round(u)), int(round(v))
    window = atlas[max(0, iy - 2):iy + 3, max(0, ix - 2):ix + 3]
    valid = window[np.isfinite(window) & (window > .2) & (window < 1.5)]
    if not len(valid):
        raise RuntimeError("No valid atlas depth at PID117 insertion site")
    depth = float(np.median(valid))
    camera_point = np.array([(u - cx) * depth / fx, (v - cy) * depth / fy, depth])
    local = (camera_point - geometry["origin"]) @ geometry["basis"].T
    surface = RegularGridInterpolator(
        (geometry["gy"], geometry["gx"]), geometry["height"],
        bounds_error=False, fill_value=None,
    )
    local[2] = float(surface([[local[1], local[0]]])[0])
    provenance = {
        "analysis_xy_px": [u, v],
        "atlas_depth_m": depth,
        "camera_xyz_m": camera_point.tolist(),
        "local_surface_xyz_m": local.tolist(),
        "source": str(SITE_REL),
    }
    return local, provenance


def add_site_marker(xml: str, site: np.ndarray) -> str:
    marker = f"""
    <body name="insertion_site_marker" pos="{site[0]:.9f} {site[1]:.9f} {site[2]:.9f}">
      <geom name="insertion_site_disc" type="cylinder" pos="0 0 .0007" size=".0055 .0007"
            rgba="1 .14 .03 .88" contype="0" conaffinity="0"/>
      <geom name="insertion_site_pin" type="capsule" fromto="0 0 .002 0 0 .017" size=".00125"
            rgba="1 .14 .03 1" contype="0" conaffinity="0"/>
      <geom name="insertion_site_head" type="sphere" pos="0 0 .019" size=".003"
            rgba="1 .72 .05 1" contype="0" conaffinity="0"/>
    </body>
"""
    return xml.replace("  </worldbody>", marker + "  </worldbody>")


def map_point(x: float, y: float, gx: np.ndarray, gy: np.ndarray,
              origin: tuple[int, int], size: tuple[int, int]) -> tuple[int, int]:
    ox, oy = origin; width, height = size
    px = ox + int(np.clip((x - gx[0]) / (gx[-1] - gx[0]), 0, 1) * (width - 1))
    py = oy + height - 1 - int(np.clip((y - gy[0]) / (gy[-1] - gy[0]), 0, 1) * (height - 1))
    return px, py


def draw_panel(records: list[dict[str, object]], index: int, geometry: dict[str, object],
               target: np.ndarray, contact_area: np.ndarray, contact_path: list[tuple[float, float]],
               site: np.ndarray, pressure: np.ndarray) -> np.ndarray:
    panel = np.full((720, 440, 3), (23, 28, 37), dtype=np.uint8)
    record = records[-1]
    white, muted, cyan, orange, red = (242, 244, 247), (165, 174, 186), (255, 205, 80), (40, 190, 255), (45, 70, 255)
    cv2.putText(panel, "SIMULATED TASK STATE", (20, 34), cv2.FONT_HERSHEY_SIMPLEX, .68, white, 2, cv2.LINE_AA)
    cv2.putText(panel, "MuJoCo forces + loaded-cell area proxy", (20, 59), cv2.FONT_HERSHEY_SIMPLEX, .43, muted, 1, cv2.LINE_AA)
    cv2.line(panel, (20, 72), (420, 72), (64, 72, 84), 1)

    labels = [
        ("Projected center to site", f"{record['site_distance_mm']:.1f} mm"),
        ("Loaded-cell area (XY)", f"{record['covered_area_mm2']:.0f} mm2"),
        ("Reference-area coverage", f"{record['coverage_pct']:.1f}%"),
        ("Contact-path length", f"{record['contact_path_length_mm']:.1f} mm"),
        ("Normal force / active cells", f"{record['normal_force_n']:.2f} N / {record['active_cells']}/15"),
    ]
    for row, (label, value) in enumerate(labels):
        y = 103 + row * 43
        cv2.putText(panel, label, (20, y), cv2.FONT_HERSHEY_SIMPLEX, .40, muted, 1, cv2.LINE_AA)
        # right-align values without depending on a specific font installation
        width = cv2.getTextSize(value, cv2.FONT_HERSHEY_SIMPLEX, .49, 1)[0][0]
        cv2.rectangle(panel, (250, y - 18), (420, y + 5), (23, 28, 37), -1)
        cv2.putText(panel, value, (420 - width, y), cv2.FONT_HERSHEY_SIMPLEX, .49, white, 1, cv2.LINE_AA)

    gx, gy = geometry["gx"], geometry["gy"]
    map_origin, map_size = (35, 340), (370, 275)
    x0, y0 = map_origin; mw, mh = map_size
    map_rgb = np.zeros((*target.shape, 3), dtype=np.uint8)
    map_rgb[:] = (32, 38, 48)
    map_rgb[target] = (72, 76, 82)
    map_rgb[contact_area] = orange
    # Grids store ascending world Y, screen pixels run downwards. Paths use
    # map_point with Y flipped; the underlying masks must use the same mapping.
    map_rgb = cv2.resize(map_rgb[::-1], (mw, mh), interpolation=cv2.INTER_NEAREST)
    panel[y0:y0 + mh, x0:x0 + mw] = map_rgb
    cv2.rectangle(panel, (x0, y0), (x0 + mw, y0 + mh), (96, 105, 118), 1)

    reference = np.asarray(geometry["path_xy"])
    for i, (a, b) in enumerate(zip(reference[:-1], reference[1:])):
        if geometry['path']['frame'][i+1] - geometry['path']['frame'][i] > 3: continue
        cv2.line(panel, map_point(*a, gx, gy, map_origin, map_size),
                 map_point(*b, gx, gy, map_origin, map_size), (140, 125, 105), 1, cv2.LINE_AA)
    for a, b in zip(contact_path[:-1], contact_path[1:]):
        if not np.isfinite([*a,*b]).all(): continue
        cv2.line(panel, map_point(*a, gx, gy, map_origin, map_size),
                 map_point(*b, gx, gy, map_origin, map_size), cyan, 2, cv2.LINE_AA)
    site_px = map_point(float(site[0]), float(site[1]), gx, gy, map_origin, map_size)
    cv2.circle(panel, site_px, 8, red, 2, cv2.LINE_AA)
    cv2.circle(panel, site_px, 3, (30, 180, 255), -1, cv2.LINE_AA)
    if contact_path and np.isfinite(contact_path[-1]).all():
        tip_px = map_point(*contact_path[-1], gx, gy, map_origin, map_size)
        cv2.line(panel, site_px, tip_px, (210, 210, 210), 1, cv2.LINE_AA)
        cv2.circle(panel, tip_px, 5, cyan, -1, cv2.LINE_AA)
    cv2.putText(panel, "TOP-DOWN SURFACE MAP", (35, 329), cv2.FONT_HERSHEY_SIMPLEX, .43, white, 1, cv2.LINE_AA)
    cv2.putText(panel, "site", (site_px[0] + 9, site_px[1] - 7), cv2.FONT_HERSHEY_SIMPLEX, .36, white, 1, cv2.LINE_AA)

    cv2.putText(panel, "cell force proxy", (22, 648), cv2.FONT_HERSHEY_SIMPLEX, .39, muted, 1, cv2.LINE_AA)
    max_force = max(.1, float(pressure.max()))
    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            level = float(pressure[row, col]) / max_force
            color = (25, int(80 + 165 * level), int(255 * level))
            a = (145 + col * 49, 630 + row * 25)
            cv2.rectangle(panel, a, (a[0] + 44, a[1] + 20), color, -1)
    cv2.putText(panel, "orange=area  cyan=path  red=insertion site", (20, 711),
                cv2.FONT_HERSHEY_SIMPLEX, .37, (130, 204, 255), 1, cv2.LINE_AA)
    return panel


def make_comparison(source: Path, simulation: Path, destination: Path) -> None:
    command = [
        "ffmpeg", "-y", "-loglevel", "error", "-ss", "16.98", "-t", "5.02", "-i", str(source),
        "-ss", "1.0", "-t", "5.02", "-i", str(simulation), "-filter_complex",
        "[0:v]fps=30,scale=640:360[real];[1:v]fps=30,scale=640:360[sim];"
        "[real][sim]hstack=inputs=2:shortest=1[v]", "-map", "[v]", "-c:v", "libx264",
        "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(destination),
    ]
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
    site, site_provenance = insertion_site_local(root, geometry)
    hfield = output / "pid117_mannequin_hfield.png"; write_hfield_png(geometry["height"], hfield)
    trajectory = oriented_trajectory(geometry, args.fps, 0, args.indentation_mm / 1000)
    xml, contact_names = scene_xml(hfield.name, geometry, "compliant_pad", trajectory[0])
    xml = add_site_marker(xml, site)
    scene = output / "pid117_flat_pad_labeled_scene.xml"; scene.write_text(xml)
    model = mujoco.MjModel.from_xml_path(str(scene)); data = mujoco.MjData(model)
    first = trajectory[0]
    data.qpos[:3] = [first["x"], first["y"], first["z"]]; data.qpos[3:7] = first["quat"]
    data.mocap_pos[0] = data.qpos[:3]; data.mocap_quat[0] = data.qpos[3:7]
    mujoco.mj_forward(model, data)

    geom_ids = {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name): name for name in contact_names}
    mannequin = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "mannequin")
    tool_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "tool")
    gx, gy = geometry["gx"], geometry["gy"]
    xx, yy = np.meshgrid(gx, gy)
    target = np.zeros_like(geometry["height"], dtype=bool)
    for x, y in geometry["path_xy"][::4]:
        target |= (np.abs(xx - x) <= .014) & (np.abs(yy - y) <= .009)
    contact_area = np.zeros_like(target)
    pixel_area_mm2 = abs(float(gx[1] - gx[0]) * float(gy[1] - gy[0])) * 1e6
    height_fn = RegularGridInterpolator((gy, gx), geometry["height"], bounds_error=False, fill_value=None)

    renderer = mujoco.Renderer(model, height=720, width=840)
    temp = output / "pid117_flat_pad_augmented.mp4v.mp4"
    final = output / "pid117_flat_pad_augmented.mp4"
    poster = output / "pid117_flat_pad_augmented.jpg"
    writer = cv2.VideoWriter(str(temp), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (1280, 720))
    if not writer.isOpened(): raise RuntimeError(temp)
    records: list[dict[str, object]] = []; contact_path: list[tuple[float, float]] = []
    path_length = 0.0; previous_contact = None; poster_frame = None
    cell_half_x = PAD_LENGTH / GRID_COLS / 2; cell_half_y = PAD_WIDTH / GRID_ROWS / 2
    center = np.array([(gx[-1] + gx[0]) / 2, (gy[-1] + gy[0]) / 2, .018])
    try:
        for frame_index, point in enumerate(trajectory):
            data.mocap_pos[0] = [point["x"], point["y"], point["z"]]
            data.mocap_quat[0] = point["quat"]
            forces = {name: 0.0 for name in contact_names}
            positions: dict[str, list[np.ndarray]] = {name: [] for name in contact_names}
            substeps = max(1, round((frame_index+1)/args.fps/model.opt.timestep)
                           - round(frame_index/args.fps/model.opt.timestep))
            for _ in range(substeps):
                mujoco.mj_step(model, data)
                step_forces = {gid: 0. for gid in geom_ids}
                for ci in range(data.ncon):
                    contact = data.contact[ci]; pair = {int(contact.geom1), int(contact.geom2)}
                    candidates = pair.intersection(geom_ids)
                    if mannequin not in pair or not candidates: continue
                    gid = next(iter(candidates)); name = geom_ids[gid]
                    force = np.zeros(6); mujoco.mj_contactForce(model, data, ci, force)
                    forces[name] += max(0., float(force[0])) / substeps
                    step_forces[gid] += max(0.,float(force[0]))
                    positions[name].append(np.array(contact.pos[:2]))
                for gid, force in step_forces.items():
                    if force <= .02: continue
                    hx, hy, hz = model.geom_size[gid]
                    corners = np.array([[-hx,-hy,-hz],[hx,-hy,-hz],[hx,hy,-hz],[-hx,hy,-hz]])
                    world = data.geom_xpos[gid]+corners@data.geom_xmat[gid].reshape(3,3).T
                    # Rasterize actual cell pose, not the commanded yaw or the
                    # centroid of corner contacts, which can translate a cell.
                    paint_projected_face(contact_area,world,gx,gy)
            active = [name for name, value in forces.items() if value > .02]

            tool_xy = np.asarray(data.xpos[tool_body, :2], dtype=float)
            surface_z = float(height_fn([[tool_xy[1], tool_xy[0]]])[0])
            distance = float(np.linalg.norm(np.r_[tool_xy, surface_z] - site)) * 1000
            if active and point["phase"] == "contact":
                current = (float(tool_xy[0]), float(tool_xy[1]))
                if previous_contact is not None:
                    path_length += float(np.linalg.norm(np.asarray(current) - previous_contact)) * 1000
                previous_contact = np.asarray(current); contact_path.append(current)
            else:
                previous_contact = None
                if contact_path and np.isfinite(contact_path[-1]).all():
                    contact_path.append((float('nan'),float('nan')))
            pressure = np.zeros((GRID_ROWS, GRID_COLS))
            for row in range(GRID_ROWS):
                for col in range(GRID_COLS): pressure[row, col] = forces[f"pad_cell_{row}_{col}"]
            covered_target = contact_area & target
            record = {
                "frame": frame_index, "time_s": frame_index / args.fps,
                "simulation_time_s": float(data.time), "phase": point["phase"],
                "segment": int(point["segment"]), "normal_force_n": float(sum(forces.values())),
                "active_cells": len(active), "site_distance_mm": distance,
                "covered_area_mm2": float(contact_area.sum() * pixel_area_mm2),
                "coverage_pct": float(100 * covered_target.sum() / max(1, target.sum())),
                "contact_path_length_mm": path_length, "tool_x_m": float(tool_xy[0]),
                "tool_y_m": float(tool_xy[1]), "surface_z_m": surface_z,
            }
            records.append(record)
            camera = mujoco.MjvCamera(); camera.type = mujoco.mjtCamera.mjCAMERA_FREE
            camera.lookat[:] = center; camera.azimuth = 118; camera.elevation = -35; camera.distance = .27
            renderer.update_scene(data, camera=camera)
            left = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
            cv2.rectangle(left, (0, 0), (840, 100), (18, 23, 31), -1)
            cv2.putText(left, "PID117 flat-facing pad replay", (22, 32), cv2.FONT_HERSHEY_SIMPLEX,
                        .73, (245, 245, 245), 2, cv2.LINE_AA)
            cv2.putText(left, f"t={frame_index/args.fps:5.1f}s  {point['phase']}  segment {point['segment']}/6",
                        (22, 63), cv2.FONT_HERSHEY_SIMPLEX, .50, (220, 225, 232), 1, cv2.LINE_AA)
            cv2.putText(left, "RED PIN = INSERTION SITE | pad face follows local surface normal",
                        (22, 88), cv2.FONT_HERSHEY_SIMPLEX, .43, (90, 205, 255), 1, cv2.LINE_AA)
            panel = draw_panel(records, frame_index, geometry, target, contact_area, contact_path, site, pressure)
            composed = np.hstack([left, panel]); writer.write(composed)
            if poster_frame is None and point["phase"] == "contact" and len(contact_path) > args.fps * 2:
                poster_frame = composed.copy()
    finally:
        renderer.close(); writer.release()
    encode_h264(temp, final, args.fps)
    if poster_frame is None: poster_frame = composed
    cv2.imwrite(str(poster), poster_frame)

    trace = output / "pid117_flat_pad_augmented_trace.csv"
    with trace.open("w", newline="") as handle:
        writer_csv = csv.DictWriter(handle, fieldnames=list(records[0])); writer_csv.writeheader(); writer_csv.writerows(records)
    comparison = output / "pid117_nursing_vs_sim_contact_state.mp4"
    source = root / "docs/artifacts/tip_depth_fusion_contact/pid117_contact_proxy.mp4"
    make_comparison(source, final, comparison)
    contact_records = [r for r in records if r["phase"] == "contact"]
    distance_values = np.asarray([r["site_distance_mm"] for r in contact_records])
    summary = {
        "schema_version": 1, "participant": "PID117", "mujoco_version": mujoco.__version__,
        "policy": "flat-facing pad: pad normal follows local hfield normal; yaw follows commanded path tangent",
        "insertion_site": site_provenance,
        "results": {
            "final_covered_area_mm2": records[-1]["covered_area_mm2"],
            "final_reference_coverage_pct": records[-1]["coverage_pct"],
            "contact_path_length_mm": records[-1]["contact_path_length_mm"],
            "site_distance_mm": {"min": float(distance_values.min()), "median": float(np.median(distance_values)),
                                  "max": float(distance_values.max())},
            "contact_frames": int(sum(r["active_cells"] > 0 for r in contact_records)),
            "intended_contact_frames": len(contact_records),
        },
        "outputs": {"scene": scene.name, "video": final.name, "poster": poster.name,
                    "trace": trace.name, "physical_comparison": comparison.name},
        "claim_boundary": [
            "The insertion site is transferred through the frozen monocular depth atlas, not measured in robot coordinates.",
            "Area is an XY raster union of projected loaded-cell faces at actual simulated poses; corner contact can overestimate its extent. It is not exact continuum contact area, surface area, or deposition.",
            "The flat-facing orientation is an explicit control rule; human applicator orientation is not estimated.",
            "Force, friction, compliance, and pad dimensions remain provisional and uncalibrated.",
        ],
    }
    artifacts = []
    for path in sorted(p for p in output.iterdir()
                       if p.is_file() and not p.name.endswith("mp4v.mp4") and p.name != "summary.json"):
        artifacts.append({"path": path.name, "bytes": path.stat().st_size, "sha256": file_hash(path)})
    summary["artifacts"] = artifacts
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
