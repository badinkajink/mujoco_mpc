#!/usr/bin/env python3
"""Independent, time-aligned scoring. No controller/task callback is installed."""
import argparse, hashlib, json, math, re
from pathlib import Path
import numpy as np
import subprocess

FIELDS=['t','phase','reach_error','brace_N','brace_up_N','trunk_N','other_N','pelvis_z','tilt_deg','joint_velocity','joint_limit_rad','actuator_force_fraction','foot_displacement','penetration_m']

def longest(t, mask, gap=.021):
    best=0.; start=None; last=None; interval=None
    for tt,ok in zip(t,mask):
        if not ok: start=None; last=None; continue
        if start is None or (last is not None and tt-last>gap): start=float(tt)
        if tt-start>best: best=float(tt-start); interval=[start,float(tt)]
        last=float(tt)
    return best,interval

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def evaluate(folder):
    p=Path(folder); job=json.loads((p/'job.json').read_text()); result=json.loads((p/'result.json').read_text())
    replay=Path(__file__).with_name('replay')
    subprocess.run([str(replay),str(p/'plant.mjb'),str(p/'state.csv'),str(p/'native.csv'),str(p/'model_info.json')],check=True)
    subprocess.run([str(replay),str(p/'plant.mjb.planner'),'/dev/null','/dev/null',str(p/'planner_info.json')],check=True)
    mi=json.loads((p/'model_info.json').read_text());pi=json.loads((p/'planner_info.json').read_text())
    face=mi['face'];planner_face=pi['face'];assert abs(face-job['h'])<1e-9
    same=sha(p/'plant.mjb')==sha(p/'plant.mjb.planner')
    if job['sync']:assert same,'synchronized model snapshots differ'
    diff=[] if same else ['binary_model_differs']
    target=np.array([mi['near']+job['depth'],mi['center']-.04,face+.15])
    x=np.atleast_1d(np.genfromtxt(p/'native.csv',names=True,delimiter=','))
    assert len(x)>1 and np.all(np.diff(x['t'])>0)
    error=np.linalg.norm(np.column_stack([x['jaw_x'],x['jaw_y'],x['jaw_z']])-target,axis=1)
    rows=np.column_stack([x['t'],x['phase'],error]+[x[f] for f in FIELDS[3:]])
    assert np.isfinite(rows).all()
    ctrl_viol=float(x['ctrl_violation'].max())
    contact_names={}
    z=np.array(rows);t=z[:,0];phase=z[:,1];base=(phase==2)&(z[:,4]>=20)&(z[:,5]<10)&(z[:,6]<10)&(z[:,7]>.8)
    safe=('fell=0' in result['summary']) and bool((z[:,7]>=.65).all())
    d30,i30=longest(t,base&(z[:,2]<=.03));d70,i70=longest(t,base&(z[:,2]<=.07))
    complete='complete=1' in result['summary'];terminal=(phase==8)&(t>=t[-1]-2.02)
    recovery_mask=terminal&(z[:,7]>.8)&(z[:,8]<=15)&(z[:,3]<10)
    recovery_dwell,_=longest(t,recovery_mask)
    recovered=bool(complete and recovery_mask[-1] and recovery_dwell>=2-1e-8)
    # Cross-check the cached C++ FK against independent FK at pre-step times.
    # run.csv is labeled post-step but its jaw/contact cache belongs pre-step.
    live=np.genfromtxt(p/'run.csv',names=True,delimiter=',');live=np.atleast_1d(live)
    live_error=np.linalg.norm(np.column_stack([live['jaw_x'],live['jaw_y'],live['jaw_z']])-target,axis=1)
    nn=min(len(z),len(live));cache_error=float(np.max(abs(live_error[:nn]-z[:nn,2])))
    contact_delta=float(np.percentile(abs(live['brace_normal_N'][:nn]-z[:nn,3]),95))
    assert cache_error<5e-6,('FK replay/live mismatch',cache_error)
    assert contact_delta<.01,('force replay/live mismatch',contact_delta)
    maxphase=int(phase.max());success30=bool(safe and d30>=2-1e-8);success70=bool(safe and d70>=2-1e-8)
    kind='complete' if recovered and success30 else ('fall' if not safe else ('approach' if maxphase<2 else ('reach_miss' if not success30 else 'recovery_stall')))
    out={**result,'target':target.tolist(),'strict_success':success30,'loaded70_success':success70,'safe':safe,'dwell30_s':d30,'dwell70_s':d70,'interval30':i30,'interval70':i70,'ladder_complete':complete,'recovered':recovered,'recovery_dwell_s':recovery_dwell,'full_success':bool(success30 and recovered),'failure_class':kind,'max_phase':maxphase,'pelvis_min':float(z[:,7].min()),'max_joint_velocity_rad_s':float(z[:,9].max()),'max_joint_limit_violation_deg':math.degrees(float(z[:,10].max())),'max_actuator_force_fraction':float(z[:,11].max()),'max_foot_displacement_m':float(z[:,12].max()),'max_penetration_m':float(z[:,13].max()),'ctrl_range_violation':max(0.,ctrl_viol),'cache_fk_max_error_m':cache_error,'cache_brace_delta_p95_N':contact_delta,'plant_face':face,'planner_face':planner_face,'planner_different_arrays':diff,'contact_body_peak_N':contact_names,'sample_dt':float(np.median(np.diff(t))),'state_sha256':sha(p/'state.csv'),'model_sha256':sha(p/'plant.mjb')}
    if (phase==2).any():out['reach_min_mm']=float(z[phase==2,2].min()*1000)
    if i30:
        use=(t>=i30[0])&(t<=i30[1]);out['strict_interval']={k:float(v) for k,v in {'error_p95_mm':np.percentile(z[use,2],95)*1000,'brace_up_p05_N':np.percentile(z[use,4],5),'tilt_median_deg':np.median(z[use,8]),'joint_limit_peak_deg':math.degrees(z[use,10].max()),'foot_displacement_peak_m':z[use,12].max()}.items()}
    np.savetxt(p/'metrics.csv',z,delimiter=',',header=','.join(FIELDS),comments='')
    (p/'evaluation.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    return out

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('folders',nargs='+');args=ap.parse_args()
    for folder in args.folders:
        r=evaluate(folder);print(Path(folder).name,r['strict_success'],r['full_success'],round(r['dwell30_s'],2),r['failure_class'],flush=True)
