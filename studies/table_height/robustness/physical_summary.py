"""Descriptive physical diagnostics; never changes frozen success scores."""
import collections,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3];STUDY=ROOT/'studies/table_height/robustness'
rows=[]
for p in sorted((ROOT/'studies/table_height/runs/robustness_20260909').glob('*/evaluation.json')):
 r=json.loads(p.read_text())
 if r['job']['stage']=='verification':continue
 x=np.genfromtxt(p.parent/'metrics.csv',names=True,delimiter=',')
 d={'tag':r['job']['tag'],'stage':r['job']['stage'],'arm':r['job']['arm'],'h':r['job']['h'],'strict_success':r['strict_success']}
 # Describe only the primary qualifying interval of successful episodes.
 if r['strict_success']:
  lo,hi=r['interval30'];q=x[(x['t']>=lo)&(x['t']<=hi)]
  d.update(hold_s=hi-lo,error_p95_mm=float(np.percentile(q['reach_error'],95)*1000),brace_up_p05_N=float(np.percentile(q['brace_up_N'],5)),joint_velocity_p95_rad_s=float(np.percentile(q['joint_velocity'],95)),joint_velocity_max_rad_s=float(q['joint_velocity'].max()),any_actuator_near_limit_sample_fraction=float(np.mean(q['actuator_force_fraction']>=.995)),tilt_median_deg=float(np.median(q['tilt_deg'])),joint_limit_max_deg=float(np.degrees(q['joint_limit_rad'].max())),max_penetration_mm=float(q['penetration_m'].max()*1000))
 # This is the terminal window even when the ladder did not complete.
 q=x[x['t']>=x['t'][-1]-2.02]
 d.update(final_phase=int(q['phase'][-1]),terminal_tilt_max_deg=float(q['tilt_deg'].max()),terminal_brace_max_N=float(q['brace_N'].max()))
 rows.append(d)
(STUDY/'physical_summary.json').write_text(json.dumps(rows,indent=2)+'\n')
print('Described',len(rows),'episodes;',sum('hold_s' in r for r in rows),'successful precise holds')
