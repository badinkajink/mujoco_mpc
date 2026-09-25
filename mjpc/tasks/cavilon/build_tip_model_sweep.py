#!/usr/bin/env python3
"""Compare rigid and distributed-compliance Cavilon applicator models.

The provisional 20 x 9 x 4 mm pad dimensions are image-informed assumptions,
not metrology.  The distributed model uses 5 x 3 independently compliant
contact cells as a numerically robust reduced-order foam approximation.
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

from build_contact_replay import (
    build_trajectory,
    encode_h264,
    load_geometry,
    split_segments,
    write_hfield_png,
)


PAD_LENGTH = 0.020
PAD_WIDTH = 0.009
PAD_THICKNESS = 0.004
GRID_COLS = 5
GRID_ROWS = 3


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def pad_xml(model_name: str) -> tuple[str, float, list[str]]:
    if model_name == "rigid_pad":
        xml = f"""
      <geom name="pad_rigid" type="box" size="{PAD_LENGTH/2} {PAD_WIDTH/2} {PAD_THICKNESS/2}"
            rgba="0.94 0.78 0.28 1" priority="1" contype="1" conaffinity="1"/>
      <geom name="pad_backing" type="box" pos="0 0 {PAD_THICKNESS/2 + .001}"
            size="{PAD_LENGTH/2+.0005} {PAD_WIDTH/2+.0005} .001" rgba="0.92 0.92 0.88 1"
            contype="0" conaffinity="0"/>
"""
        return xml, PAD_THICKNESS / 2, ["pad_rigid"]

    if model_name != "compliant_pad":
        raise ValueError(model_name)
    children = []
    names = []
    cell_x = PAD_LENGTH / GRID_COLS
    cell_y = PAD_WIDTH / GRID_ROWS
    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            x = (col - (GRID_COLS - 1) / 2) * cell_x
            y = (row - (GRID_ROWS - 1) / 2) * cell_y
            name = f"pad_cell_{row}_{col}"
            names.append(name)
            children.append(f"""
      <body name="{name}_body" pos="{x:.6f} {y:.6f} {-PAD_THICKNESS/2:.6f}">
        <joint name="{name}_compression" type="slide" axis="0 0 1" range="-0.0002 0.0032"
               stiffness="200" damping="0.32" armature="0.00002"/>
        <inertial pos="0 0 0" mass="0.0015" diaginertia="2e-8 2e-8 2e-8"/>
        <geom name="{name}" type="box" size="{cell_x/2-.0001:.6f} {cell_y/2-.0001:.6f} {PAD_THICKNESS/2:.6f}"
              rgba="0.94 0.78 0.28 1" priority="1" contype="1" conaffinity="1"/>
      </body>""")
    xml = f"""
      <geom name="pad_backing" type="box" pos="0 0 .001" size="{PAD_LENGTH/2+.0005} {PAD_WIDTH/2+.0005} .001"
            rgba="0.92 0.92 0.88 1" contype="0" conaffinity="0"/>
{''.join(children)}
"""
    return xml, PAD_THICKNESS, names


def scene_xml(hfield: str, geometry: dict[str, object], model_name: str,
              start: dict[str, float]) -> tuple[str, list[str]]:
    gx, gy, height = geometry["gx"], geometry["gy"], geometry["height"]
    z_min = float(height.min())
    z_range = max(float(height.max() - z_min), 1e-4)
    pad, _, contact_geoms = pad_xml(model_name)
    return f"""<mujoco model="Cavilon {model_name}">
  <compiler angle="radian"/>
  <option timestep="0.001" integrator="implicitfast" gravity="0 0 -9.81" solver="Newton" iterations="80"/>
  <size nconmax="160" njmax="1000"/>
  <visual><global offwidth="960" offheight="720"/><headlight diffuse=".72 .72 .72" ambient=".28 .28 .28" specular=".2 .2 .2"/></visual>
  <default><geom solref="0.03 1" solimp="0.9 0.98 0.003" friction="0.75 0.02 0.002"/></default>
  <asset>
    <hfield name="mannequin_hf" file="{hfield}" size="{(gx[-1]-gx[0])/2:.9f} {(gy[-1]-gy[0])/2:.9f} {z_range:.9f} .012"/>
    <material name="skin" rgba=".80 .56 .43 1" specular=".12" shininess=".25"/>
    <material name="handle" rgba=".92 .92 .90 1" specular=".3"/>
  </asset>
  <worldbody>
    <light pos="0 -.2 .45" dir="0 .35 -1" directional="true" castshadow="true"/>
    <geom name="mannequin" type="hfield" hfield="mannequin_hf"
          pos="{(gx[-1]+gx[0])/2:.9f} {(gy[-1]+gy[0])/2:.9f} {z_min:.9f}"
          material="skin" contype="1" conaffinity="1"/>
    <body name="tool_target" mocap="true" pos="{start['x']:.9f} {start['y']:.9f} {start['z']:.9f}"/>
    <body name="tool" pos="{start['x']:.9f} {start['y']:.9f} {start['z']:.9f}">
      <freejoint/>
      <inertial pos="-.018 0 .04" mass=".055" diaginertia=".00005 .00005 .000008"/>
{pad}
      <geom name="stem" type="capsule" fromto="0 0 .003 -.033 0 .074" size=".0022"
            material="handle" contype="0" conaffinity="0"/>
      <geom name="grip" type="capsule" fromto="-.033 0 .074 -.055 0 .121" size=".0035"
            rgba=".84 .88 .92 1" contype="0" conaffinity="0"/>
    </body>
  </worldbody>
  <equality><weld name="cartesian_servo" body1="tool_target" body2="tool" solref=".04 1" solimp=".85 .98 .002"/></equality>
</mujoco>
""", contact_geoms


def add_text(frame: np.ndarray, model_name: str, index: int, force: float,
             coverage: float, cells: int, phase: str, pressure: np.ndarray,
             tilt_deg: float = 0.0) -> np.ndarray:
    image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.rectangle(image, (0, 0), (image.shape[1], 92), (18, 23, 31), -1)
    title = "rigid rectangular pad" if model_name == "rigid_pad" else "5 x 3 distributed-compliance pad"
    cv2.putText(image, f"PID117 - {title}", (24, 31), cv2.FONT_HERSHEY_SIMPLEX, .72, (245,245,245), 2, cv2.LINE_AA)
    cv2.putText(image, f"t={index/30:5.1f}s  {phase}  tilt={tilt_deg:2.0f} deg  force={force:5.2f} N  cells={cells:2d}  covered={coverage:5.1f}%",
                (24, 64), cv2.FONT_HERSHEY_SIMPLEX, .49, (220,225,232), 1, cv2.LINE_AA)
    cv2.putText(image, "PROVISIONAL 20 x 9 x 4 mm pad; compliance is not calibrated", (24, 85),
                cv2.FONT_HERSHEY_SIMPLEX, .41, (100,205,255), 1, cv2.LINE_AA)
    panel = np.zeros((135, 210, 3), dtype=np.uint8)
    cv2.putText(panel, "instantaneous pressure proxy", (7, 18), cv2.FONT_HERSHEY_SIMPLEX, .39, (225,225,225), 1, cv2.LINE_AA)
    maximum = max(float(pressure.max()), .1)
    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            value = float(pressure[row, col]) / maximum
            color = (20, int(70 + 175*value), int(255*value))
            x0, y0 = 8 + col*39, 30 + row*31
            cv2.rectangle(panel, (x0,y0), (x0+35,y0+27), color, -1)
            cv2.rectangle(panel, (x0,y0), (x0+35,y0+27), (100,100,100), 1)
    x0, y0 = image.shape[1]-225, image.shape[0]-150
    image[y0:y0+135, x0:x0+210] = panel
    return image


def render(model: mujoco.MjModel, trajectory: list[dict[str, float]], contact_names: list[str],
           geometry: dict[str, object], model_name: str, output_dir: Path, fps: int) -> dict[str, object]:
    data = mujoco.MjData(model)
    first = trajectory[0]
    data.qpos[:3] = [first["x"], first["y"], first["z"]]
    data.qpos[3:7] = first.get("quat", [1,0,0,0])
    data.mocap_pos[0] = data.qpos[:3]
    data.mocap_quat[0] = data.qpos[3:7]
    mujoco.mj_forward(model, data)

    geom_ids = {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name): name for name in contact_names}
    mannequin = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "mannequin")
    tool_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "tool")
    gx, gy = geometry["gx"], geometry["gy"]
    xx, yy = np.meshgrid(gx, gy)
    target = np.zeros_like(geometry["height"], dtype=bool)
    for x, y in geometry["path_xy"][::4]:
        target |= (np.abs(xx-x) <= .014) & (np.abs(yy-y) <= .009)
    covered = np.zeros_like(target)

    renderer = mujoco.Renderer(model, height=720, width=960)
    extent = max(gx[-1]-gx[0], gy[-1]-gy[0])
    center = np.array([(gx[-1]+gx[0])/2, (gy[-1]+gy[0])/2, .02])
    cameras = {"close": (118., -34., max(.25, extent*1.45)), "top": (90., -89., max(.27, extent*1.25))}
    temporary, writers = {}, {}
    for view in cameras:
        path = output_dir / f"pid117_{model_name}_{view}.mp4v.mp4"
        temporary[view] = path
        writers[view] = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (960,720))
        if not writers[view].isOpened(): raise RuntimeError(path)

    substeps = max(1, int(round(1/fps/model.opt.timestep)))
    records = []
    cell_half_x = PAD_LENGTH/GRID_COLS/2
    cell_half_y = PAD_WIDTH/GRID_ROWS/2
    try:
        for frame_index, point in enumerate(trajectory):
            data.mocap_pos[0] = [point["x"], point["y"], point["z"]]
            data.mocap_quat[0] = point.get("quat", [1,0,0,0])
            forces = {name: 0.0 for name in contact_names}
            positions: dict[str, list[np.ndarray]] = {name: [] for name in contact_names}
            for _ in range(substeps):
                mujoco.mj_step(model, data)
                for ci in range(data.ncon):
                    c = data.contact[ci]
                    pair = {int(c.geom1), int(c.geom2)}
                    candidates = pair.intersection(geom_ids)
                    if mannequin not in pair or not candidates: continue
                    gid = next(iter(candidates)); name = geom_ids[gid]
                    force = np.zeros(6); mujoco.mj_contactForce(model, data, ci, force)
                    forces[name] += max(0., float(force[0])) / substeps
                    positions[name].append(np.array(c.pos[:2]))
            active = [name for name, value in forces.items() if value > .02]
            if model_name == "rigid_pad" and active:
                cx, cy = data.xpos[tool_body, :2]
                yaw = float(point.get("yaw_rad", 0.0)); tilt = float(point.get("tilt_rad", 0.0))
                du = (xx-cx)*np.cos(yaw) + (yy-cy)*np.sin(yaw)
                dv = -(xx-cx)*np.sin(yaw) + (yy-cy)*np.cos(yaw)
                covered |= (np.abs(du) <= PAD_LENGTH/2*abs(np.cos(tilt))) & (np.abs(dv) <= PAD_WIDTH/2) & target
            elif model_name == "compliant_pad":
                yaw = float(point.get("yaw_rad", 0.0)); tilt = float(point.get("tilt_rad", 0.0))
                for name in active:
                    if not positions[name]: continue
                    cx, cy = np.mean(positions[name], axis=0)
                    du = (xx-cx)*np.cos(yaw) + (yy-cy)*np.sin(yaw)
                    dv = -(xx-cx)*np.sin(yaw) + (yy-cy)*np.cos(yaw)
                    covered |= (np.abs(du) <= cell_half_x*abs(np.cos(tilt))) & (np.abs(dv) <= cell_half_y) & target
            coverage = 100*covered.sum()/max(1,target.sum())
            pressure = np.zeros((GRID_ROWS,GRID_COLS))
            if model_name == "rigid_pad":
                pressure[:] = sum(forces.values()) / (GRID_ROWS*GRID_COLS)
            else:
                for row in range(GRID_ROWS):
                    for col in range(GRID_COLS): pressure[row,col] = forces[f"pad_cell_{row}_{col}"]
            total_force = float(sum(forces.values()))
            records.append({"frame":frame_index,"time_s":frame_index/fps,"phase":point["phase"],
                            "segment":point["segment"],"normal_force_n":total_force,
                            "active_cells":len(active),"coverage_pct":coverage,
                            "tool_x_m":float(data.xpos[tool_body,0]),"tool_y_m":float(data.xpos[tool_body,1]),
                            "tool_z_m":float(data.xpos[tool_body,2])})
            for view,(az,el,dist) in cameras.items():
                camera=mujoco.MjvCamera(); camera.type=mujoco.mjtCamera.mjCAMERA_FREE
                camera.lookat[:]=center; camera.azimuth=az; camera.elevation=el; camera.distance=dist
                renderer.update_scene(data,camera=camera)
                writers[view].write(add_text(renderer.render(),model_name,frame_index,total_force,coverage,
                                              len(active),str(point["phase"]),pressure,
                                              float(point.get("tilt_deg", 0.0))))
    finally:
        renderer.close()
        for writer in writers.values(): writer.release()
    videos={}
    for view,tmp in temporary.items():
        destination=output_dir/f"pid117_{model_name}_{view}.mp4"
        encode_h264(tmp,destination,fps); videos[view]=destination
    return {"records":records,"videos":videos,"final_coverage_pct":float(records[-1]["coverage_pct"])}


def combine_videos(sphere: Path, rigid: Path, compliant: Path, destination: Path) -> None:
    command=["ffmpeg","-y","-loglevel","error","-i",str(sphere),"-i",str(rigid),"-i",str(compliant),
             "-filter_complex","[0:v]scale=480:360[v0];[1:v]scale=480:360[v1];[2:v]scale=480:360[v2];[v0][v1][v2]hstack=inputs=3[v]",
             "-map","[v]","-c:v","libx264","-crf","21","-pix_fmt","yuv420p","-movflags","+faststart",str(destination)]
    subprocess.run(command,check=True)


def physical_comparison(source: Path, simulation: Path, destination: Path) -> None:
    command=["ffmpeg","-y","-loglevel","error","-i",str(source),"-i",str(simulation),"-filter_complex",
             "[0:v]trim=start=16.98:end=22.0,setpts=PTS-STARTPTS,fps=30,crop=600:520:250:380,scale=640:480[real];"
             "[1:v]trim=start=1.0:end=6.02,setpts=PTS-STARTPTS,scale=640:480[sim];"
             "[real][sim]hstack=inputs=2:shortest=1[v]",
             "-map","[v]","-t","5.02","-c:v","libx264","-crf","21","-pix_fmt","yuv420p",
             "-movflags","+faststart",str(destination)]
    subprocess.run(command,check=True)


def main() -> int:
    parser=argparse.ArgumentParser()
    parser.add_argument("--cldc-root",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--fps",type=int,default=30)
    parser.add_argument("--indentation-mm",type=float,default=.45)
    args=parser.parse_args()
    root=args.cldc_root.resolve(); output=args.output_dir.resolve(); output.mkdir(parents=True,exist_ok=True)
    geometry=load_geometry(root,128)
    hfield=output/"pid117_mannequin_hfield.png"; write_hfield_png(geometry["height"],hfield)
    summaries={}
    for model_name in ("rigid_pad","compliant_pad"):
        model_dir=output/model_name; model_dir.mkdir(exist_ok=True)
        model_hfield=model_dir/hfield.name
        if model_hfield.exists() or model_hfield.is_symlink(): model_hfield.unlink()
        model_hfield.symlink_to(hfield)
        _,offset,_=pad_xml(model_name)
        trajectory=build_trajectory(geometry["path"],geometry["path_xy"],geometry["path_z"],args.fps,
                                    tip_radius=offset,indentation=args.indentation_mm/1000)
        xml,names=scene_xml(hfield.name,geometry,model_name,trajectory[0])
        scene=model_dir/f"pid117_{model_name}_scene.xml"; scene.write_text(xml)
        model=mujoco.MjModel.from_xml_path(str(scene))
        result=render(model,trajectory,names,geometry,model_name,model_dir,args.fps)
        trace=model_dir/f"pid117_{model_name}_trace.csv"
        with trace.open("w",newline="") as handle:
            writer=csv.DictWriter(handle,fieldnames=list(result["records"][0])); writer.writeheader(); writer.writerows(result["records"])
        intended=[r for r in result["records"] if r["phase"]=="contact"]
        positive=[r for r in intended if r["active_cells"]>0]
        forces=np.asarray([r["normal_force_n"] for r in intended])
        summaries[model_name]={
            "model":model_name,"scene":str(scene.relative_to(output)),"trace":str(trace.relative_to(output)),
            "videos":{k:str(v.relative_to(output)) for k,v in result["videos"].items()},
            "contact_frames":len(positive),"intended_contact_frames":len(intended),
            "force_median_n":float(np.median(forces)),"force_p95_n":float(np.quantile(forces,.95)),
            "force_max_n":float(forces.max()),"final_coverage_pct":result["final_coverage_pct"],
        }
    comparison=output/"pid117_tip_model_comparison.mp4"
    combine_videos(root/"experiments/aim2_3_simulation_bridge/pid117/pid117_contact_replay_close.mp4",
                   output/summaries["rigid_pad"]["videos"]["close"],
                   output/summaries["compliant_pad"]["videos"]["close"],comparison)
    real_compare=output/"pid117_physical_vs_compliant.mp4"
    physical_comparison(root/"data/nursing/clips/PID117/13_Step_3_Cavilon_No_Sting_Barrier.mp4",
                        output/summaries["compliant_pad"]["videos"]["close"],real_compare)
    artifacts=[]
    for path in sorted(p for p in output.rglob('*') if p.is_file() and not p.is_symlink()):
        artifacts.append({"path":str(path.relative_to(output)),"bytes":path.stat().st_size,"sha256":digest(path)})
    summary={
        "schema_version":1,"participant":"PID117","mujoco_version":mujoco.__version__,
        "geometry_assumption":{"pad_length_m":PAD_LENGTH,"pad_width_m":PAD_WIDTH,"pad_thickness_m":PAD_THICKNESS,
                               "basis":"provisional image-informed dimensions; physical measurement required"},
        "compliance_assumption":{"grid":[GRID_ROWS,GRID_COLS],"cell_stiffness_n_per_m":200,
                                 "cell_damping_n_s_per_m":.32,"basis":"reduced-order Kelvin-Voigt foundation; uncalibrated"},
        "models":summaries,"comparison_video":comparison.name,"physical_comparison_video":real_compare.name,
        "source_video":"data/nursing/clips/PID117/13_Step_3_Cavilon_No_Sting_Barrier.mp4",
        "claim_boundary":["The footage supports pad shape class but not metric dimensions or material parameters.",
                          "Model forces and coverage are sensitivity outputs until calibrated against physical measurements.",
                          "The compliant grid is a reduced-order control model, not a finite-element claim about foam mechanics."],
        "artifacts":artifacts,
    }
    (output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps({"output":str(output),"models":summaries,"comparison":comparison.name,
                      "physical_comparison":real_compare.name},indent=2))
    return 0


if __name__=="__main__": sys.exit(main())
