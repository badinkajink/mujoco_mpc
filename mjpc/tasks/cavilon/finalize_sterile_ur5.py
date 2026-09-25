#!/usr/bin/env python3
"""Revalidate saved trajectory, measure surface tracking, and render evidence."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from build_contact_replay import load_geometry
from render_flat_pad_augmented import insertion_site_local
from sterility_geometry import CollisionContract, make_targets
from plan_sterile_ur5 import validate, render


def certify_exclusion_edges(contract,q,clearances,margin=.002):
    """Bound unsampled motion on each linear joint-space frame edge.

    Every point moves at most R*sum(abs(dq)); any intermediate point is at
    most half that bound from one endpoint. Convex projection is conservative.
    Subdivide when the clearance bound is too loose, never relax the margin.
    """
    samples=0;minimum=float('inf')
    def edge(a,b,ca,cb,depth=0):
        nonlocal samples,minimum
        lower=min(ca,cb)-.5*contract.point_motion_radius*np.abs(b-a).sum()
        if lower>=margin:minimum=min(minimum,lower);return
        if depth>=15:raise RuntimeError('Could not certify exclusion interpolation')
        mid=(a+b)/2;cm,_=contract.projected_cylinder_clearance(mid);samples+=1
        if cm<margin:raise RuntimeError(f'Exclusion interpolation violation: {cm}')
        edge(a,mid,ca,cm,depth+1);edge(mid,b,cm,cb,depth+1)
    for a,b,ca,cb in zip(q[:-1],q[1:],clearances[:-1],clearances[1:]):edge(a,b,ca,cb)
    return dict(additional_midpoint_checks=samples,point_motion_radius_bound_m=contract.point_motion_radius,
                conservative_minimum_clearance_mm=minimum*1000,
                scope='Insertion-site infinite cylinder only, exact static model and linear joint-space frame interpolation; not all collisions, dynamics, calibration or hardware safety.')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--cldc-root',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--no-render',action='store_true')
    args=parser.parse_args();out=args.output_dir
    model=mujoco.MjModel.from_xml_path(str(out/'scene.xml'))
    geometry=load_geometry(args.cldc_root,128);site,_=insertion_site_local(args.cldc_root,geometry)
    targets,_=make_targets(geometry,site)
    saved=np.load(out/'trajectory.npz');q=saved['qpos'];phases=saved['phase']
    contract=CollisionContract(model)
    validation,records,invalid=validate(model,contract,q,1/30)
    if invalid:raise RuntimeError(f'Revalidation failed: {validation}; first {invalid[:3]}')
    validation['exclusion_edge_bound']=certify_exclusion_edges(contract,q,
        np.array([r['projected_cylinder_mm']/1000 for r in records]))
    data=mujoco.MjData(model);sid=model.site('cavilon_contact').id
    number=0;errors=[];normals=[];positions=[]
    for i,pose in enumerate(q):
        data.qpos[:]=pose;mujoco.mj_forward(model,data)
        xyz=data.site_xpos[sid].copy();positions.append(xyz)
        records[i].update(tip_x=float(xyz[0]),tip_y=float(xyz[1]),tip_z=float(xyz[2]),phase=str(phases[i]))
        if phases[i]!='surface_follow':continue
        if i==0 or phases[i-1]!='surface_follow':number+=1
        reference=[t for t in targets if t.segment==number and t.phase=='surface_follow']
        points=np.array([t.position for t in reference]);a=points[:-1];delta=np.diff(points,axis=0)
        alpha=np.clip(np.sum((xyz-a)*delta,axis=1)/np.maximum(np.sum(delta*delta,axis=1),1e-20),0,1)
        distance=np.linalg.norm(xyz-a-alpha[:,None]*delta,axis=1);j=int(np.argmin(distance))
        errors.append(float(distance[j]*1000))
        normal=(1-alpha[j])*reference[j].rotation[:,2]+alpha[j]*reference[j+1].rotation[:,2]
        normal/=np.linalg.norm(normal)
        normals.append(float(np.rad2deg(np.arccos(np.clip(normal@data.site_xmat[sid].reshape(3,3)[:,2],-1,1)))))
    summary=json.loads((out/'summary.json').read_text());summary['validation']=validation
    summary['surface_tracking']=dict(completed_geometry_runs=number,
        max_distance_to_adapted_reference_mm=max(errors),p95_distance_to_adapted_reference_mm=float(np.quantile(errors,.95)),
        max_normal_error_deg=max(normals),p95_normal_error_deg=float(np.quantile(normals,.95)),
        definition='Distance to the corresponding adapted stroke polyline, not the original unsafe path. Includes time-parameterized interpolation.')
    summary['render']=dict(playback_speed=4,video='sterile_ur5.mp4',poster='sterile_ur5.jpg')
    summary['validation']['distance_horizon_mm']=100
    summary['validation']['independent_check']='Conservative 2D convex projection against infinite cylinder at every video-rate pose; circumscribed 64-gons for cylindrical robot geoms.'
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    with (out/'clearance_trace.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    if not args.no_render:
        render(model,geometry,site,q,records,phases,targets,out,30,summary['offset_adapter_mm'])
    sources=[*Path(__file__).parent.glob('*.py'),geometry['atlas_path'],geometry['mask_path'],geometry['path_path']]
    outputs=[out/'summary.json',out/'scene.xml',out/'trajectory.npz',out/'clearance_trace.csv',out/'sterile_ur5.mp4',out/'sterile_ur5.jpg']
    manifest=dict(sources_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                  outputs_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in outputs if p.exists()},
                  claim='Offline geometric path with sampled validation, not continuous certification, dynamic force control or hardware safety.')
    (out/'artifact_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
