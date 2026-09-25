#!/usr/bin/env python3
"""Render one UR5e + provisional Cavilon pad kinematic feasibility pose."""

from __future__ import annotations

import argparse
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
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from build_contact_replay import encode_h264, load_geometry


def build_scene(ur5e_xml: Path, hfield: Path, geometry: dict[str, object],
                destination: Path) -> None:
    tree = ET.parse(ur5e_xml)
    root = tree.getroot()
    root.set("model", "UR5e Cavilon pad feasibility")
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(root, "compiler")
    compiler.set("meshdir", str((ur5e_xml.parent / "assets").resolve()))
    option = root.find("option")
    if option is None:
        option = ET.SubElement(root, "option")
    option.set("timestep", "0.002")
    option.set("gravity", "0 0 -9.81")
    visual = root.find("visual")
    if visual is None:
        visual = ET.SubElement(root, "visual")
    global_visual = visual.find("global")
    if global_visual is None:
        global_visual = ET.SubElement(visual, "global")
    global_visual.set("offwidth", "960")
    global_visual.set("offheight", "720")

    asset = root.find("asset")
    assert asset is not None
    gx, gy, height = geometry["gx"], geometry["gy"], geometry["height"]
    z_min = float(height.min())
    z_range = float(height.max() - z_min)
    ET.SubElement(asset, "hfield", {
        "name": "pid117_hfield", "file": str(hfield.resolve()),
        "size": f"{(gx[-1]-gx[0])/2} {(gy[-1]-gy[0])/2} {z_range} .012",
    })
    ET.SubElement(asset, "material", {"name": "mannequin_skin", "rgba": ".80 .56 .43 1", "specular": ".12"})

    worldbody = root.find("worldbody")
    assert worldbody is not None
    surface_origin_x = .55
    surface_origin_z = .25
    ET.SubElement(worldbody, "geom", {
        "name": "pid117_mannequin", "type": "hfield", "hfield": "pid117_hfield",
        "pos": f"{surface_origin_x+(gx[-1]+gx[0])/2} {(gy[-1]+gy[0])/2} {surface_origin_z+z_min}",
        "material": "mannequin_skin", "contype": "0", "conaffinity": "0",
    })
    ET.SubElement(worldbody, "geom", {"name": "floor", "type": "plane", "size": "2 2 .05", "rgba": ".14 .18 .22 1"})
    ET.SubElement(worldbody, "light", {"pos": ".4 -.6 1.3", "dir": "0 .4 -1", "directional": "true", "castshadow": "true"})

    wrist = next(body for body in root.iter("body") if body.get("name") == "wrist_3_link")
    tool = ET.SubElement(wrist, "body", {"name": "cavilon_tool", "pos": "0 .1 0", "quat": "-1 1 0 0"})
    ET.SubElement(tool, "geom", {"name": "tool_stem", "type": "capsule", "fromto": "0 0 0 0 0 .098",
                                          "size": ".0025", "rgba": ".88 .90 .93 1", "contype": "0", "conaffinity": "0"})
    ET.SubElement(tool, "geom", {"name": "tool_backing", "type": "box", "pos": "0 0 .105",
                                          "size": ".0105 .005 .0015", "rgba": ".92 .92 .88 1", "contype": "0", "conaffinity": "0"})
    ET.SubElement(tool, "geom", {"name": "tool_pad", "type": "box", "pos": "0 0 .108",
                                          "size": ".010 .0045 .002", "rgba": ".94 .78 .28 1", "contype": "0", "conaffinity": "0"})
    ET.SubElement(tool, "site", {"name": "cavilon_contact", "pos": "0 0 .110", "size": ".002", "rgba": "1 .15 .05 1"})
    tree.write(destination, encoding="unicode")


def solve_ik(model: mujoco.MjModel, data: mujoco.MjData, site_id: int,
             target_pos: np.ndarray, target_quat: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    home = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if home >= 0:
        data.qpos[:] = model.key_qpos[home]
    target_matrix_flat = np.empty(9)
    mujoco.mju_quat2Mat(target_matrix_flat, target_quat)
    target_matrix = target_matrix_flat.reshape(3, 3)

    def residual(qpos: np.ndarray) -> np.ndarray:
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        current_matrix = data.site_xmat[site_id].reshape(3, 3)
        pos_error = target_pos - data.site_xpos[site_id]
        rot_error = Rotation.from_matrix(target_matrix.T @ current_matrix).as_rotvec()
        return np.r_[pos_error, .35 * rot_error]

    lower = np.asarray(model.jnt_range[:, 0], dtype=float)
    upper = np.asarray(model.jnt_range[:, 1], dtype=float)
    result = least_squares(
        residual, data.qpos.copy(), bounds=(lower, upper), max_nfev=1000,
        xtol=1e-12, ftol=1e-12, gtol=1e-12,
    )
    data.qpos[:] = result.x
    mujoco.mj_forward(model, data)
    current_matrix = data.site_xmat[site_id].reshape(3, 3)
    rot_error = Rotation.from_matrix(target_matrix.T @ current_matrix).as_rotvec()
    metrics = {"function_evaluations": int(result.nfev), "solver_success": bool(result.success),
               "position_error_m": float(np.linalg.norm(target_pos-data.site_xpos[site_id])),
               "orientation_error_rad": float(np.linalg.norm(rot_error))}
    metrics["joint_limits_satisfied"] = bool(all(
        (not model.jnt_limited[joint]) or
        (model.jnt_range[joint, 0] <= data.qpos[model.jnt_qposadr[joint]] <= model.jnt_range[joint, 1])
        for joint in range(model.njnt)
    ))
    return data.qpos.copy(), metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cldc-root", type=Path, required=True)
    parser.add_argument("--ur5e-xml", type=Path, required=True)
    parser.add_argument("--hfield", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()
    output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=True)
    geometry = load_geometry(args.cldc_root.resolve(), 128)
    scene = output / "ur5e_pid117_pad_scene.xml"
    build_scene(args.ur5e_xml.resolve(), args.hfield.resolve(), geometry, scene)
    model = mujoco.MjModel.from_xml_path(str(scene)); data = mujoco.MjData(model)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "cavilon_contact")
    # Use the patch center for this first kinematic render.  It avoids making a
    # trajectory/reachability claim from one demonstration-path sample.
    x_index = int(np.argmin(np.abs(geometry["gx"])))
    y_index = int(np.argmin(np.abs(geometry["gy"])))
    surface_height = float(geometry["height"][y_index, x_index])
    target_pos = np.array([.55, 0., .25 + surface_height + .001])
    target_quat = np.array([0., 1., 0., 0.])
    qpos, metrics = solve_ik(model, data, site_id, target_pos, target_quat)
    if metrics["position_error_m"] > .005:
        raise RuntimeError(f"IK did not reach the surface target: {metrics}")
    data.qpos[:] = qpos; data.ctrl[:] = qpos[:model.nu]; mujoco.mj_forward(model, data)

    renderer = mujoco.Renderer(model, height=720, width=960)
    temporary = output / "ur5e_cavilon_pad_feasibility.mp4v.mp4"
    final = output / "ur5e_cavilon_pad_feasibility.mp4"
    writer = cv2.VideoWriter(str(temporary), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (960,720))
    poster = None
    frames = 6 * args.fps
    try:
        for index in range(frames):
            camera = mujoco.MjvCamera(); camera.type = mujoco.mjtCamera.mjCAMERA_FREE
            camera.lookat[:] = [.32, 0, .32]; camera.distance = 1.25
            camera.azimuth = 112 + 32*np.sin(2*np.pi*index/frames); camera.elevation = -24
            renderer.update_scene(data, camera=camera)
            image = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
            cv2.rectangle(image, (0,0), (960,82), (18,23,31), -1)
            cv2.putText(image, "UR5e + provisional rectangular Cavilon pad", (22,31), cv2.FONT_HERSHEY_SIMPLEX,
                        .72, (245,245,245), 2, cv2.LINE_AA)
            cv2.putText(image, f"IK position error={metrics['position_error_m']*1000:.2f} mm | KINEMATIC RENDER - no controller or force validation",
                        (22,62), cv2.FONT_HERSHEY_SIMPLEX, .47, (100,205,255), 1, cv2.LINE_AA)
            writer.write(image)
            if index == frames//3: poster = image.copy()
    finally:
        renderer.close(); writer.release()
    encode_h264(temporary, final, args.fps)
    poster_path = output / "ur5e_cavilon_pad_feasibility.jpg"
    cv2.imwrite(str(poster_path), poster)
    source_hash = hashlib.sha256(args.ur5e_xml.read_bytes()).hexdigest()
    summary = {"mujoco_version":mujoco.__version__,"source_ur5e_xml":str(args.ur5e_xml.resolve()),
               "source_ur5e_sha256":source_hash,"scene":scene.name,"video":final.name,"poster":poster_path.name,
               "target_position_m":target_pos.tolist(),"qpos_rad":qpos.tolist(),"ik":metrics,
               "claim_boundary":"Kinematic reach and visualization only; no collision-free trajectory, control, force, or task success claim."}
    (output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2))
    return 0


if __name__ == "__main__": sys.exit(main())
