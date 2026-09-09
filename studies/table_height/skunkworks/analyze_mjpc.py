#!/usr/bin/env python3
"""Score actual bench force columns and FK jaw errors from saved states.
Uses the full gate tip offset; baseline's early experimental jaw_* columns
omitted y/z, so all runs deliberately use qpos FK for a uniform definition.
"""
import argparse,json
from pathlib import Path
import numpy as np,mujoco
ROOT=Path(__file__).resolve().parents[3]
ap=argparse.ArgumentParser();ap.add_argument('--runs',default=str(ROOT/'studies/table_height/runs/codex_skunkworks/mjpc'));a=ap.parse_args()
model=str(ROOT/'build_cmake/mjpc/tasks/humanoid_bench/lean/Lean_H12_Magpie.xml')
m=mujoco.MjModel.from_xml_path(model);d=mujoco.MjData(m);gid=m.body('right_magpie_gripper').id

def dwell(t,g):
 best=start=0.;prev=False
 for tt,good in zip(t,g):
  if good and not prev:start=tt
  if good:best=max(best,tt-start)
  prev=good
 return float(best)
res=[]
for p in sorted(Path(a.runs).glob('*/result.json')):
 r=json.loads(p.read_text());job=r['job'];path=p.parent
 if not r['summary']:continue
 x=np.genfromtxt(path/'run.csv',delimiter=',',names=True,dtype=None,encoding='utf8');q=np.genfromtxt(path/'run.qpos.csv',delimiter=',',names=True)
 nq=len([n for n in q.dtype.names if n.startswith('q')]);qq=np.column_stack([q[f'q{i}'] for i in range(nq)])
 rtt=np.array(job.get('rtt',[.55,.04,.15]));target=np.array([.45+rtt[0],-rtt[1],job['h']+rtt[2]])
 tip=[]
 for state in qq:
  d.qpos[:]=state;mujoco.mj_kinematics(m,d)
  tip.append(d.xpos[gid]+d.xmat[gid].reshape(3,3)@np.array([.2254,-.0118,-.1062]))
 tip=np.column_stack([np.interp(x['t'],q['t'],np.array(tip)[:,k]) for k in range(3)])
 err=np.linalg.norm(tip-target,axis=1);phase=x['phase']==2
 good=phase&(err<=.07)&(x['brace_normal_N']>=20)&(x['trunk_normal_N']<10)&(x['pelvis_z']>.8)
 dur=dwell(x['t'],good)
 r.update(target=target.tolist(),pelvis_min=float(x['pelvis_z'].min()),max_phase=int(x['phase'].max()),reach_min_mm=float(err[phase].min()*1000) if phase.any() else None,reach_p95_mm=float(np.percentile(err[phase],95)*1000) if phase.any() else None,brace_peak_N=float(x['brace_normal_N'].max()),joint_dwell_s=dur,strict_dwell_s=dwell(x['t'],good&(err<=.03)),dynamic_success=bool(dur>=2 and 'fell=0' in r['summary'] and (x['pelvis_z']>=.65).all()),ladder_complete='complete=1' in r['summary'])
 r['loaded_right_other_peak_N']=float((x['f_r_elbow']+x['f_r_wrist']+x['f_r_gripper']+x['f_other'])[good].max()) if good.any() else None
 r['loaded_body_resultant_median_N']={k:float(np.median(x[k][good])) if good.any() else None for k in ['f_shoulder','f_forearm','f_wrist','f_gripper','f_r_elbow','f_r_wrist','f_r_gripper','f_other']}
 r['loaded_tilt_p50_deg']=float(np.median(x['torso_tilt_deg'][good])) if good.any() else None
 r['loaded_brace_p05_N']=float(np.percentile(x['brace_normal_N'][good],5)) if good.any() else None
 r['loaded_reach_p95_mm']=float(np.percentile(err[good],95)*1000) if good.any() else None
 r['phase2_brace_p50_N']=float(np.median(x['brace_normal_N'][phase])) if phase.any() else None
 r['phase2_trunk_peak_N']=float(x['trunk_normal_N'][phase].max()) if phase.any() else None
 r['max_pad_clear_brace_mm']=float(x['pad_clear'][np.isin(x['phase'],[1,2])].max()*1000) if (x['phase']>=1).any() else None
 z=np.column_stack([x['t'],x['phase'],err,x['brace_normal_N'],x['trunk_normal_N'],x['pelvis_z'],x['pad_clear'],x['torso_tilt_deg']])
 np.savetxt(path/'metrics.csv',z,delimiter=',',header='t,phase,reach_error,brace_N,trunk_N,pelvis_z,pad_clear,tilt_deg',comments='')
 res.append(r)
 print(job['tag'],'complete',r['ladder_complete'],'dynamic',r['dynamic_success'],'dwell',round(dur,2),'minerr',r['reach_min_mm'],flush=True)
Path(a.runs,'audit.json').write_text(json.dumps(res,indent=2))
