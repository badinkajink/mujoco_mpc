#!/usr/bin/env python3
"""Audit logged qpos/qvel/ctrl, never infer load from qpos alone.
Reconstructed forces exclude floor, object and all non-arm load from brace.
"""
import argparse, os, sys, json
from pathlib import Path
import numpy as np
ap=argparse.ArgumentParser()
ap.add_argument('--croco',required=True);ap.add_argument('--runs',required=True);ap.add_argument('--out',required=True)
a=ap.parse_args()
sys.path.insert(0,str(Path(a.croco)/'studies'))
import contact_select as cs
import mujoco

def longest(t,good):
    best=start=0.; prev=False
    for tt,g in zip(t,good):
        if g and not prev:start=tt
        if g:best=max(best,tt-start)
        prev=g
    return float(best)

def contacts(m,d):
    f=np.zeros(6); arm=trunk=other=0.; bodies={}
    tb=cs.bid(m,'table')
    for con_i in range(d.ncon):
        c=d.contact[con_i]; b=[int(m.geom_bodyid[g]) for g in c.geom]
        if tb not in b:continue
        rb=b[1] if b[0]==tb else b[0]
        name=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,rb) or ''
        if rb==0 or 'object' in name:continue
        mujoco.mj_contactForce(m,d,con_i,f); load=max(0,float(f[0]))
        bodies[name]=bodies.get(name,0)+load
        if name.startswith('left_') and any(v in name for v in ('shoulder','elbow','wrist','gripper','magpie')):arm+=load
        elif name in ('torso_link','pelvis'):trunk+=load
        else:other+=load
    return arm,trunk,other,bodies

out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
res=[]
for p in sorted(Path(a.runs).glob('h*/h*_*.csv')):
    if '.qpos.' in p.name:continue
    face=int(p.parent.name[1:5])/1000
    cs.TABLE_H=face
    m,d=cs.load(ik_margin=0.)
    lines=p.read_text().splitlines()
    target=np.array([float(v) for v in lines[2].split('reach_target=')[1].split(',')[0].split('|')])
    x=np.genfromtxt(p,delimiter=',',names=True,skip_header=3)
    fields=x.dtype.names
    rows=[]
    for i in range(0,len(x),5): # 20 Hz; full state/control reconstruction
        r=x[i];d.qpos[:]=[r[f'qpos{k}'] for k in range(m.nq)]
        d.qvel[:]=[r[f'qvel{k}'] for k in range(m.nv)]
        d.ctrl[:]=[r[f'ctrl{k}'] for k in range(m.nu)]
        mujoco.mj_forward(m,d)
        hand=cs.point_world(m,d,cs.REACH_BODY,cs.REACH_OFF)
        arm,trunk,other,bodies=contacts(m,d)
        pitch=np.degrees(np.arcsin(np.clip(2*(d.qpos[3]*d.qpos[5]-d.qpos[6]*d.qpos[4]),-1,1)))
        rows.append([r['time'],np.linalg.norm(hand-target),d.qpos[2],arm,trunk,other,pitch])
    z=np.asarray(rows);tail=z[:,0]>=z[-1,0]-5
    joint=(z[:,1]<=.07)&(z[:,2]>.8)&(z[:,3]>=20)&(z[:,4]<10)
    strict=joint&(z[:,1]<=.03)
    held=bool((z[:,2]>.65).all() and joint[tail].mean()>=.95 and longest(z[tail,0],joint[tail])>=2)
    r=dict(path=str(p.resolve()),face=face,target=target.tolist(),tag=p.stem,
      arm='brace' if '_brace_' in p.name else 'stand',seed=int(p.stem.rsplit('r',1)[1]),
      pelvis_min=float(z[:,2].min()),tail_reach_p95_mm=float(np.percentile(z[tail,1],95)*1000),
      tail_brace_p05_N=float(np.percentile(z[tail,3],5)),tail_trunk_peak_N=float(z[tail,4].max()),
      tail_joint_fraction=float(joint[tail].mean()),tail_strict_fraction=float(strict[tail].mean()),
      longest_joint_dwell_s=longest(z[:,0],joint),success=held,tail_pitch_deg=float(np.median(z[tail,6])))
    res.append(r);np.savetxt(out/(p.stem+'.metrics.csv'),z,delimiter=',',header='t,reach_error,pelvis_z,brace_N,trunk_N,other_N,pitch_deg',comments='')
    print(p.stem,held,'err95',round(r['tail_reach_p95_mm'],1),'load05',round(r['tail_brace_p05_N'],1),'pelvis min',round(r['pelvis_min'],3),flush=True)
    (out/'audit.json').write_text(json.dumps(res,indent=2))
