#!/usr/bin/env python3
"""Collision-constrained Cavilon path: branch search, Mink IK, OMPL transfers.

Run in .venv-cavilon. The saved path is time parameterized and independently
checked between samples. No physical force or dynamic tracking is claimed.
"""
from __future__ import annotations
import argparse
import csv
import json
import os
from pathlib import Path
import time
import hashlib
from dataclasses import replace

os.environ.setdefault('MUJOCO_GL', 'egl')
import cv2
import mink
import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial.transform import Rotation
from ompl import base as ob, geometric as og, util as ou

from build_contact_replay import encode_h264
from sterility_geometry import (ORIGIN, EXCLUSION_RADIUS, MARGIN, CollisionContract,
                                make_targets, scene)


class ExplicitCollisionLimit(mink.CollisionAvoidanceLimit):
    """Keep the already audited pairs, including collision-disabled visuals.

    Mink's default pair builder filters contype/conaffinity and parent bodies.
    A planning-only cylinder and visual hulls deliberately have no physics
    contacts; filtering those out would silently disable the exclusion rule.
    """
    def _construct_geom_id_pairs(self, pairs):
        return sorted({tuple(sorted((int(a),int(b)))) for left,right in pairs
                       for a in left for b in right})


class PadSurfaceLimit(mink.Limit):
    """Linearized clearance for points on the pad face over the heightfield.

    MuJoCo's hfield collision manifold independently validates accepted steps.
    These inequalities prevent the differential solver taking a shortcut through
    the terrain while it rotates between two admissible target poses.
    """
    def __init__(self, model, geometry):
        self.model=model;self.sid=model.site('cavilon_contact').id
        self.body=model.site_bodyid[self.sid]
        self.offsets=np.array([[a,b,0.] for a in np.linspace(-.010,.010,7)
                              for b in np.linspace(-.0045,.0045,5)])
        gy,gx=geometry['gy'],geometry['gx'];height=geometry['height']
        dy,dx=np.gradient(height,gy,gx)
        self.height=RegularGridInterpolator((gy,gx),height,bounds_error=False,fill_value=np.nan)
        self.dx=RegularGridInterpolator((gy,gx),dx,bounds_error=False,fill_value=0.)
        self.dy=RegularGridInterpolator((gy,gx),dy,bounds_error=False,fill_value=0.)

    def compute_qp_inequalities(self, configuration, dt):
        data=configuration.data
        points=data.site_xpos[self.sid]+self.offsets@data.site_xmat[self.sid].reshape(3,3).T
        query=(points-ORIGIN)[:,[1,0]];height=self.height(query)+ORIGIN[2]
        gaps=points[:,2]-height
        rows=[];rhs=[]
        for i in np.flatnonzero(np.isfinite(gaps) & (gaps<.02)):
            jac=np.zeros((3,self.model.nv))
            mujoco.mj_jac(self.model,data,jac,None,points[i],self.body)
            normal=np.array([-self.dx(query[i:i+1])[0],-self.dy(query[i:i+1])[0],1.])
            rows.append(-normal@jac);rhs.append(.7*(gaps[i]-.0005))
        return mink.Constraint(G=np.array(rows),h=np.array(rhs)) if rows else mink.Constraint()


def pose_error(model, data, q, target):
    data.qpos[:] = q; mujoco.mj_forward(model, data)
    sid = model.site('cavilon_contact').id
    return np.r_[(data.site_xpos[sid]-target.position),
                 .20*Rotation.from_matrix(target.rotation.T @ data.site_xmat[sid].reshape(3,3)).as_rotvec()]


def seed_candidates(model, contract, target, count=60):
    data = mujoco.MjData(model)
    rng = np.random.default_rng(240924)
    home = model.key_qpos[0].copy()
    seeds = [home]
    for _ in range(count-1):
        seeds.append(rng.uniform([-3.14,-2.8,-2.8,-3.14,-3.14,-3.14],
                                 [3.14,.4,2.8,3.14,3.14,3.14]))
    valid, diagnostics = [], []
    for n, seed in enumerate(seeds):
        # Explore the unconstrained in-plane yaw as well as elbow branches.
        # Fixed radial yaw can leave only a single branch that later dead-ends.
        yaws=[0, -30, 30, -60, 60, -90, 90, -120, 120, -150, 150, 180]
        angle = np.deg2rad(yaws[n % len(yaws)])
        seed_target = replace(target, rotation=target.rotation @ Rotation.from_euler('z',angle).as_matrix())
        result = least_squares(lambda q: pose_error(model,data,q,seed_target), seed,
                               bounds=(model.jnt_range[:,0]+1e-5,model.jnt_range[:,1]-1e-5),
                               max_nfev=120, ftol=1e-9, xtol=1e-9, gtol=1e-9)
        err = pose_error(model,data,result.x,seed_target)
        if np.linalg.norm(err[:3]) > .0002 or np.linalg.norm(err[3:]) > .002: continue
        q = result.x.copy()
        # Principal equivalent is safe only when still within physical limits.
        q -= 2*np.pi*np.round(q/(2*np.pi))
        info = contract.summary(q); diagnostics.append(info)
        if contract.valid(q, MARGIN) and not any(np.linalg.norm(q-v)<.1 for v in valid):
            valid.append(q)
    valid.sort(key=lambda q: np.linalg.norm(q-home))
    return valid, diagnostics


def track(model, contract, targets, q0, geometry):
    config = mink.Configuration(model, q=q0)
    # Flat-facing fixes the pad normal (two rotational DOFs). Its in-plane yaw
    # is free for whole-arm avoidance; video does not constrain this yaw.
    task = mink.FrameTask('cavilon_contact','site',position_cost=1.,orientation_cost=[.2,.2,0.],
                          lm_damping=1e-4, gain=.75)
    limits = [mink.ConfigurationLimit(model),
              mink.VelocityLimit(model,{model.joint(i).name:1. for i in range(model.njnt)}),
              ExplicitCollisionLimit(model, [([a],[b]) for a,b in contract.cylinder_pairs],
                  minimum_distance_from_collisions=MARGIN,
                  # Mink 1.3 uses gain * gap / dt for a displacement QP.
                  # Scale gain by our 20 ms step to bound displacement to
                  # 75% of remaining clearance; verified by a unit test.
                  collision_detection_distance=.03, gain=.75*.02),
              ExplicitCollisionLimit(model, [([a],[b]) for a,b in contract.floor_pairs+contract.self_pairs],
                  minimum_distance_from_collisions=.001,
                  collision_detection_distance=.02, gain=.75*.02), PadSurfaceLimit(model,geometry)]
    path, phases, segments, target_ids = [q0.copy()], ['approach'], [0], [0]
    sid = model.site('cavilon_contact').id
    worst_error = 0.
    for index, target in enumerate(targets):
        transform = mink.SE3.from_rotation_and_translation(mink.SO3.from_matrix(target.rotation),target.position)
        task.set_target(transform)
        converged = False
        for iteration in range(160):
            vel = mink.solve_ik(config,[task],.02,'daqp',limits=limits,damping=1e-6)
            previous = config.q.copy()
            config.integrate_inplace(vel,.02)
            # Nonlinear acceptance makes the QP's linearized constraints a
            # proposal. Reject/halve a step if exact collision checks fail.
            if not contract.valid(config.q, .003):
                accepted = False
                for factor in (.5,.25,.125,.0625):
                    q = previous+factor*(config.q-previous)
                    if contract.valid(q,.003): config.update(q); accepted=True; break
                if not accepted:
                    return None, dict(failed_target=index, phase=target.phase, reason='nonlinear_collision',
                                      **contract.summary(config.q))
            error = float(np.linalg.norm(config.data.site_xpos[sid]-target.position))
            actual_normal=config.data.site_xmat[sid].reshape(3,3)[:,2]
            rotation_error = float(np.arccos(np.clip(np.dot(target.rotation[:,2],actual_normal),-1,1)))
            if np.linalg.norm(config.q-path[-1]) > 1e-5:
                path.append(config.q.copy()); phases.append(target.phase); segments.append(target.segment);target_ids.append(index)
            if error < .00015 and rotation_error < .003:
                converged=True;worst_error=max(worst_error,error);break
        if not converged:
            return None,dict(failed_target=index,phase=target.phase,reason='IK_stall',position_error_mm=error*1000,
                             **contract.summary(config.q))
        if index%50 == 0: print(f'target {index}/{len(targets)} clear {contract.summary(config.q)["cylinder_mm"]:.2f} mm',flush=True)
    return dict(q=np.asarray(path),phases=phases,segments=segments,target_ids=target_ids), dict(max_target_error_mm=worst_error*1000)


def ompl_approach(model, contract, start, goal, seconds=6.):
    if not contract.valid(start, .003):
        raise RuntimeError(f'Home invalid: {contract.summary(start)}')
    if not contract.valid(goal, .003): raise RuntimeError('Goal invalid')
    space=ob.RealVectorStateSpace(model.nq)
    bounds=ob.RealVectorBounds(model.nq)
    for i,(lo,hi) in enumerate(model.jnt_range):bounds.setLow(i,float(lo));bounds.setHigh(i,float(hi))
    space.setBounds(bounds)
    setup=og.SimpleSetup(space)
    setup.setStateValidityChecker(lambda state: contract.valid(np.array([state[i] for i in range(model.nq)]),.003))
    setup.getSpaceInformation().setStateValidityCheckingResolution(.0003)
    start_state=space.allocState(); goal_state=space.allocState()
    for i in range(model.nq): start_state[i]=float(start[i]); goal_state[i]=float(goal[i])
    setup.setStartAndGoalStates(start_state,goal_state)
    planner=og.RRTConnect(setup.getSpaceInformation()); planner.setRange(.25);setup.setPlanner(planner)
    solved=setup.solve(seconds)
    if not solved or not setup.haveExactSolutionPath():raise RuntimeError('OMPL did not find an exact approach')
    # Keep the collision-checked OMPL vertices. General simplification can run
    # far past a nominal time budget inside a high-resolution edge check.
    path=setup.getSolutionPath()
    q=np.array([[state[i] for i in range(model.nq)] for state in path.getStates()])
    return q


def time_parameterize(q, phases, contract, vmax=.6, amax=1.2, fps=30):
    # Quintic rest-to-rest interpolation at retained knots. Every segment's
    # peak speed/acceleration has a closed-form bound (1.875, 10/sqrt(3)).
    # Preserve OMPL vertices and phase boundaries. Simplify local IK runs
    # only within 0.2 mrad joint-space error, with nonlinear shortcut checks.
    mandatory={0,len(q)-1}
    for i in range(1,len(q)):
        if phases[i] != phases[i-1] or phases[i]=='ompl_transfer':mandatory.update((i-1,i))
    keep=set(mandatory)
    def simplify(a,b):
        if b-a<=1:return
        delta=q[b]-q[a];norm=float(delta@delta)
        alpha=np.clip((q[a+1:b]-q[a])@delta/max(norm,1e-20),0,1)
        errors=np.linalg.norm(q[a+1:b]-(q[a]+alpha[:,None]*delta),axis=1)
        split=a+1+int(np.argmax(errors))
        steps=max(1,int(np.ceil(np.max(np.abs(delta))/.002)))
        safe=errors.max()<.0002 and all(contract.valid(q[a]+s*delta,.003)
                                                for s in np.linspace(0,1,steps+1))
        if not safe:keep.add(split);simplify(a,split);simplify(split,b)
    boundaries=sorted(mandatory)
    for a,b in zip(boundaries[:-1],boundaries[1:]):simplify(a,b)
    keep=sorted(keep)
    knots=q[keep]
    times=[0.];frames=[knots[0]]; source=[keep[0]]
    for k,(a,b) in enumerate(zip(knots[:-1],knots[1:])):
        delta=float(np.max(np.abs(b-a)))
        duration=max(.08,1.875*delta/vmax,np.sqrt((10/np.sqrt(3))*delta/amax))
        steps=max(2,int(np.ceil(duration*fps))); duration=steps/fps
        for j in range(1,steps+1):
            u=j/steps;s=10*u**3-15*u**4+6*u**5
            frames.append(a+s*(b-a));times.append(times[-1]+1/fps)
            source.append(keep[k] if s<.5 else keep[k+1])
    return np.array(frames),np.array(times),np.array(source)


def validate(model, contract, q, dt, max_step=.002):
    records=[];invalid=[];samples=0
    for i,pose in enumerate(q):
        info=contract.summary(pose);records.append(dict(frame=i,time_s=i*dt,**info))
        projected,geom=contract.projected_cylinder_clearance(pose)
        records[-1].update(projected_cylinder_mm=projected*1000,projected_closest_geom=geom)
        if projected<.002:invalid.append(dict(frame=i,reason='independent_projected_cylinder',clearance_mm=projected*1000))
        if i==0: subdivisions=1
        else: subdivisions=max(1,int(np.ceil(np.max(np.abs(pose-q[i-1]))/max_step)))
        for alpha in np.linspace(0,1,subdivisions+1)[1:]:
            sample=pose if i==0 else (1-alpha)*q[i-1]+alpha*pose
            if not contract.valid(sample,.002):invalid.append(dict(frame=i,alpha=float(alpha),**contract.summary(sample)))
            samples+=1
    qdot=np.diff(q,axis=0)/dt;qddot=np.diff(qdot,axis=0)/dt
    info=dict(validation_samples=samples,invalid_samples=len(invalid),
              min_cylinder_clearance_mm=min(r['cylinder_mm'] for r in records),
              min_independent_projected_clearance_mm=min(r['projected_cylinder_mm'] for r in records),
              min_floor_clearance_mm=min(r['floor_mm'] for r in records),
              min_self_clearance_mm=min(r['self_mm'] for r in records),
              max_patient_penetration_mm=max(r['patient_penetration_mm'] for r in records),
              max_joint_speed_rad_s=float(np.abs(qdot).max()),
              max_joint_acceleration_rad_s2=float(np.abs(qddot).max()),
              max_interpolation_step_rad=max_step)
    return info,records,invalid


def render(model, geometry, site, q, records, phases, targets, output, fps, offset_mm=150, stride=4):
    data=mujoco.MjData(model); renderer=mujoco.Renderer(model,height=720,width=960)
    temp=output/'sterile_ur5.mp4v.mp4'; video=output/'sterile_ur5.mp4'
    writer=cv2.VideoWriter(str(temp),cv2.VideoWriter_fourcc(*'mp4v'),fps,(1440,720))
    if not writer.isOpened():raise RuntimeError('video writer failed')
    # Display only the lower 0.55 m of the exclusion cylinder. The collision
    # model is unchanged: this is a scene-geometry visualization adjustment.
    cylinder=model.geom('site_exclusion').id
    traversed=[]; sid=model.site('cavilon_contact').id
    def mapxy(xy):
        return tuple(np.rint(np.array([1200,390])+(xy-(site+ORIGIN)[:2])*np.array([2400,-2400])).astype(int))
    try:
        for i,pose in enumerate(q):
            if i%stride:continue
            data.qpos[:]=pose;mujoco.mj_forward(model,data)
            cam=mujoco.MjvCamera();cam.lookat[:]=[.34,0,.34];cam.distance=1.18;cam.azimuth=115;cam.elevation=-28
            renderer.update_scene(data,camera=cam)
            for geom in renderer.scene.geoms[:renderer.scene.ngeom]:
                if geom.objtype==mujoco.mjtObj.mjOBJ_GEOM and geom.objid==cylinder:
                    geom.size[2]=.275;geom.pos[2]=site[2]+ORIGIN[2]+.275
            left=cv2.cvtColor(renderer.render(),cv2.COLOR_RGB2BGR)
            cv2.rectangle(left,(0,0),(960,104),(21,26,35),-1)
            for y,text,size,color in [(32,'UR5 | whole-arm insertion-site exclusion',.70,(245,245,245)),
                    (63,f'{i/fps:5.1f}s  {phases[i]}  clearance {records[i]["cylinder_mm"]:.1f} mm',.53,(130,225,170)),
                    (89,f'Mink + OMPL | geometric plan | {stride}x playback; clock is planned time',.43,(200,205,220))]:
                cv2.putText(left,text,(22,y),cv2.FONT_HERSHEY_SIMPLEX,size,color,1,cv2.LINE_AA)
            panel=np.full((720,480,3),(21,26,35),np.uint8); canvas=np.hstack([left,panel])
            cv2.putText(canvas,'EXCLUSION + TRACKING',(980,36),cv2.FONT_HERSHEY_SIMPLEX,.65,(240,240,240),2,cv2.LINE_AA)
            labels=[f'Cylinder radius: {EXCLUSION_RADIUS*1000:.1f} mm',
                    f'Whole-arm clearance: {records[i]["cylinder_mm"]:.2f} mm',
                    f'Closest: {records[i]["closest_geom"][:33]}',
                    'Pad / stem / wrist / links all checked']
            for j,text in enumerate(labels):cv2.putText(canvas,text,(980,75+j*27),cv2.FONT_HERSHEY_SIMPLEX,.42,(210,220,230),1,cv2.LINE_AA)
            cv2.circle(canvas,(1200,390),round(EXCLUSION_RADIUS*2400),(35,42,100),-1,cv2.LINE_AA)
            cv2.circle(canvas,(1200,390),round(EXCLUSION_RADIUS*2400),(60,80,255),2,cv2.LINE_AA)
            for path,color in [(geometry['path_xy']+ORIGIN[:2],(105,110,130)),
                               (np.array([t.position[:2] for t in targets if t.phase=='surface_follow']),(120,205,250))]:
                for a,b in zip(path[:-1],path[1:]):
                    if np.linalg.norm(b-a)<.008:cv2.line(canvas,mapxy(a),mapxy(b),color,1,cv2.LINE_AA)
            if phases[i]=='surface_follow':traversed.append(data.site_xpos[sid,:2].copy())
            for a,b in zip(traversed[:-1],traversed[1:]):
                if np.linalg.norm(b-a)<.008:cv2.line(canvas,mapxy(a),mapxy(b),(255,215,85),2,cv2.LINE_AA)
            tip=mapxy(data.site_xpos[sid,:2]);cv2.circle(canvas,tip,4,(240,250,240),-1)
            cv2.putText(canvas,'SITE',(1180,395),cv2.FONT_HERSHEY_SIMPLEX,.43,(180,180,255),1,cv2.LINE_AA)
            for j,text in enumerate(['gray: original demonstration','yellow: admissible reference','cyan: executed surface-follow path',
                                     f'{offset_mm:g} mm offset adapter (design)', 'Cylinder collision height: 2 m',
                                     'Force/deposition control remains future']):
                cv2.putText(canvas,text,(980,560+j*25),cv2.FONT_HERSHEY_SIMPLEX,.42,(185,195,212),1,cv2.LINE_AA)
            writer.write(canvas)
            if i//stride==(len(q)//2)//stride:cv2.imwrite(str(output/'sterile_ur5.jpg'),canvas)
    finally:renderer.close();writer.release()
    encode_h264(temp,video,fps)


def main():
    p=argparse.ArgumentParser();p.add_argument('--cldc-root',type=Path,required=True)
    p.add_argument('--ur5e-xml',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--no-render',action='store_true');p.add_argument('--offset-mm',type=float,default=150)
    p.add_argument('--segments',type=int,nargs='+',help='Optional geometry runs for an explicitly partial feasibility probe')
    p.add_argument('--reuse-raw-plan',action='store_true',help='Retime and revalidate the saved raw plan against current geometry; no new branch search')
    args=p.parse_args();out=args.output_dir.resolve();start_time=time.perf_counter()
    model,geometry,site,provenance,xml=scene(args.cldc_root,args.ur5e_xml,out,args.offset_mm/1000)
    targets,projection=make_targets(geometry,site)
    contract=CollisionContract(model)
    ou.RNG.setSeed(240924)
    previous=model.key_qpos[0].copy();qparts=[];phases=[];attempts=[];seeds_audit=[]
    if args.reuse_raw_plan:
        saved=np.load(out/'raw_plan.npz');qparts=[saved['qpos']];phases=saved['phase'].tolist()
        attempts=json.loads((out/'branch_audit.json').read_text())
        seeds_audit=json.loads((out/'seed_audit.json').read_text())
    all_targets=[]
    for number in sorted({t.segment for t in targets if t.segment}):
        if args.segments and number not in args.segments:continue
        contact=[t for t in targets if t.segment==number and t.phase=='surface_follow']
        stroke=[replace(contact[0],position=contact[0].position+np.array([0,0,lift]),phase='approach')
                for lift in np.linspace(.065,0,18,endpoint=False)]
        stroke+=contact
        stroke += [replace(contact[-1],position=contact[-1].position+np.array([0,0,lift]),phase='retract')
                   for lift in np.linspace(0,.065,18)[1:]]
        if args.reuse_raw_plan:all_targets.extend(stroke);continue
        candidates,diagnostics=seed_candidates(model,contract,stroke[0],count=144)
        candidates.sort(key=lambda q:np.linalg.norm(q-previous))
        seeds_audit.append(dict(segment=number,valid_branches=[q.tolist() for q in candidates],pose_solutions=diagnostics))
        (out/'seed_audit.json').write_text(json.dumps(seeds_audit,indent=2))
        print(f'segment {number}: {len(candidates)} collision-free initial branches',flush=True)
        track_result=None
        for candidate in candidates:
            track_result,info=track(model,contract,stroke,candidate,geometry)
            attempts.append(dict(segment=number,**info));print('branch result',info,flush=True)
            (out/'branch_audit.json').write_text(json.dumps(attempts,indent=2))
            if track_result is None: continue
            try: approach=ompl_approach(model,contract,previous,track_result['q'][0])
            except RuntimeError as exc:
                attempts.append(dict(segment=number,connection_failure=str(exc)));track_result=None;continue
            break
        if track_result is None:raise RuntimeError(f'No complete collision-free track for segment {number}')
        qparts.extend([approach[:-1],track_result['q']]);phases.extend(['ompl_transfer']*(len(approach)-1)+track_result['phases'])
        previous=track_result['q'][-1].copy();all_targets.extend(stroke)
        np.savez_compressed(out/f'stroke_{number}.npz',qpos=track_result['q'])
    q=np.vstack(qparts);targets=all_targets
    np.savez_compressed(out/'raw_plan.npz',qpos=q,phase=np.array(phases))
    q,t,source=time_parameterize(q,phases,contract)
    phases=[phases[i] for i in source]
    validation,records,invalid=validate(model,contract,q,1/30)
    summary=dict(projection=projection,insertion_site=provenance,validation=validation,
                 solver='Mink 1.3.0 / DAQP + OMPL 2.0.1 RRTConnect',mujoco_version=mujoco.__version__,
                 offset_adapter_mm=args.offset_mm,planned_segments=args.segments or list(range(1,7)),frames=len(q),duration_s=float(t[-1]),
                 valid_initial_branches=[len(s['valid_branches']) for s in seeds_audit],attempts=attempts,
                 contract=dict(radius_mm=EXCLUSION_RADIUS*1000,height_m=2.,axis='task +Z (mean patch normal)',
                               bottom_below_site_mm=20,qp_clearance_mm=MARGIN*1000,
                               geoms_checked=len(contract.robot),cylinder_pairs=len(contract.cylinder_pairs),
                               self_pairs=len(contract.self_pairs),floor_pairs=len(contract.floor_pairs)),
                 elapsed_s=time.perf_counter()-start_time,
                 claim='Collision-validated geometric path, not dynamic or physical contact execution.')
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    if invalid:
        (out/'invalid_samples.json').write_text(json.dumps(invalid[:100],indent=2))
        raise RuntimeError(f'Final path invalid: {validation}')
    np.savez_compressed(out/'trajectory.npz',qpos=q,time_s=t,phase=np.array(phases),
                        target_xyz=np.array([x.position for x in targets]))
    with (out/'clearance_trace.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    provenance_files=[Path(__file__),Path(__file__).with_name('sterility_geometry.py'),args.ur5e_xml,xml]
    (out/'artifact_manifest.json').write_text(json.dumps({
        'sources_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in provenance_files},
        'outputs_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
                         [out/'summary.json',out/'trajectory.npz',out/'clearance_trace.csv']},
        'render_stride':4,'claim':'Geometric plan; sampled collision validation, not a continuous or hardware safety certificate.'},indent=2))
    if not args.no_render:render(model,geometry,site,q,records,phases,targets,out,30,args.offset_mm)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
