#!/usr/bin/env python3
"""Audit recorded CMPC starts against lean_bench.cc's keyframe/xorshift reset."""
from pathlib import Path
import json,re
import mujoco,numpy as np
root=Path(__file__).resolve().parents[3]
raw=root/'studies/table_height/runs/codex_skunkworks'
m=mujoco.MjModel.from_binary_path(str(raw/'common_nominal.mjb'));d=mujoco.MjData(m)
res=[]
for p in sorted((raw/'croco').glob('jaw*/s*.csv')):
 if not re.fullmatch(r's\d+\.csv',p.name):continue
 a=json.loads((p.parent/'invocation.json').read_text())['args'];seed=int(p.stem[1:])
 mujoco.mj_resetDataKeyframe(m,d,m.key(a['start']).id)
 d.qpos[0]+=a['stance']
 s=(0x2545F4914F6CDD1D+seed*0x9E3779B97F4A7C15)&((1<<64)-1)
 def rnd():
  global s
  s^=(s<<13)&((1<<64)-1);s^=s>>7;s^=(s<<17)&((1<<64)-1)
  return (s>>11)/4503599627370496.-1
 for _ in range(20):rnd()
 for arr in [d.qpos,d.qvel]:
  for i in range(len(arr)):arr[i]+=.003*rnd()
 mujoco.mj_normalizeQuat(m,d.qpos)
 if a['face']!=.985:
  j=m.joint('object_anchor_joint').id
  d.qpos[m.jnt_qposadr[j]+2]+=a['face']-.985
  d.qvel[m.jnt_dofadr[j]:m.jnt_dofadr[j]+6]=0
 x=np.genfromtxt(p,names=True,delimiter=',',skip_header=3,max_rows=1)
 q=np.array([x[f'qpos{i}'] for i in range(m.nq)]);v=np.array([x[f'qvel{i}'] for i in range(m.nv)])
 r=dict(tag=p.parent.name,seed=seed,qpos_max_error=float(abs(q-d.qpos).max()),qvel_max_error=float(abs(v-d.qvel).max()))
 res.append(r);print(r)
(raw/'initial_state_parity.json').write_text(json.dumps(res,indent=2))
assert all(r['qpos_max_error']<1e-12 and r['qvel_max_error']<1e-12 for r in res)
