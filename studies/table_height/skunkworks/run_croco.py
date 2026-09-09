#!/usr/bin/env python3
"""Crocoddyl plan + real closed-loop MuJoCo episode, explicit model and endpoint.
Default benchmark matches MJPC geometry, jaw endpoint, home reset and xorshift
perturbation. New cells, plans, trajectories, live loads and summaries only.
"""
import argparse,json,os,sys,time,subprocess,hashlib
from pathlib import Path
import numpy as np
ap=argparse.ArgumentParser()
for k in ['croco','model_dir','out']:ap.add_argument('--'+k,required=True)
ap.add_argument('--face',type=float,required=True);ap.add_argument('--tx',type=float,default=.85)
ap.add_argument('--ty',type=float,default=-.04);ap.add_argument('--dz',type=float,default=.15)
ap.add_argument('--mode',default='elbow+forearm');ap.add_argument('--seeds',default='0,1,2')
ap.add_argument('--seconds',type=float,default=25);ap.add_argument('--start',default='home')
ap.add_argument('--endpoint',default='0.2254,-0.0118,-0.1062');ap.add_argument('--reach_body',default='right_magpie_gripper');ap.add_argument('--stance',type=float,default=0)
ap.add_argument('--n_approach',type=int,default=120);a=ap.parse_args()
os.environ.update(LEAN_TASK_DIR=str(Path(a.model_dir).resolve()),TABLE_H=str(a.face),REACH_OFFSET=str(a.endpoint),STANCE_DX=str(a.stance),CMPC_THREADS='6',ROS_DOMAIN_ID='1',MATCH_MJPC_TABLE='1',REACH_BODY=a.reach_body,RECONCILE_MJ_FRAMES='1')
sys.path.insert(0,str(Path(a.croco)/'studies'));sys.path.insert(0,a.croco)
import contact_select as cs
import croco_modes as CM
import cmpc_brace_vs_stand as B
import mujoco
import croco_bridge as CB
parity = CB.parity(verbose=True)
if (max(parity.get('site_err', {}).values(), default=0) > 1e-5 or parity['g_err_joint'] > 1e-4 or parity['mass_matrix_err'] > 1e-5):
    raise RuntimeError('site parity failed')
out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=True);cell=out/'cell';cell.mkdir(exist_ok=True)
(out/'parity.json').write_text(json.dumps(parity,indent=2))
(out/'invocation.json').write_text(json.dumps(dict(args=vars(a),argv=sys.argv,mujoco=mujoco.__version__,env={k:os.environ.get(k) for k in ['LEAN_TASK_DIR','TABLE_H','REACH_OFFSET','STANCE_DX','PYTHONPATH','CL_ASSETS_DIR','CMPC_THREADS','REACH_BODY','RECONCILE_MJ_FRAMES']},source_head=subprocess.check_output(['git','-C',a.croco,'rev-parse','HEAD'],text=True).strip()),indent=2))
target=np.array([a.tx,a.ty,a.face+a.dz]);subset=() if a.mode=='legs_only' else tuple(a.mode.split('+'));tag=a.mode.replace('+','_')
if not (cell/'modes.json').exists():
    recs=CM.enumerate_modes(target.tolist(),sites=tuple(dict.fromkeys(subset+('elbow','forearm','wrist'))),subsets=[subset],verbose=True)
    manifest=dict(target=target.tolist(),stance_dx=cs.STANCE_DX,stance_dy=cs.STANCE_DY,sites=list(subset),model=cs.MODEL,brace_arm=cs.BRACE_ARM,site_set=cs.SITE_SET,seed_key=cs.SEED_KEY,tau_basis=cs.TAU_BASIS,pressed=True,modes=[],ranked=[a.mode],table_h=a.face,reach_offset=a.endpoint)
    for r in recs:
        entry={k:v for k,v in r.items() if k!='qpos'};entry.update(name=a.mode,qpos_file='q_'+a.mode+'.txt')
        np.savetxt(cell/entry['qpos_file'],r['qpos']);manifest['modes'].append(entry)
    (cell/'modes.json').write_text(json.dumps(manifest,indent=1))
if not (cell/f'plan_{tag}.json').exists():
    cmd=[sys.executable,str(Path(a.croco)/'studies/croco_run.py'),'--dir',str(cell),'--tag',tag,'--mode',a.mode,'--start',a.start,'--dt','0.02','--contact-kp','50','--reach-rot','auto','--w-reach-rot','1e-2','--n-approach',str(a.n_approach),'--n-braced','80']
    (cell/'solve_command.json').write_text(json.dumps(cmd,indent=1))
    with (cell/'solve.log').open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=240,cwd=a.croco)
print('BUILD FLIGHT',flush=True);fl=B.Flight(str(cell),tag,threads=6)
plant=B.make_recording_plant(cs.torque_limits(cs.load(ik_margin=0)[0]));m=plant.m;d=plant.d
body_names=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,i) or '' for i in range(m.nbody)]
tb=cs.bid(m,'table');f=np.zeros(6);live=[]
orig_push=plant._push

def push():
    arm=trunk=other=0.
    for i in range(d.ncon):
        con=d.contact[i];b=[int(m.geom_bodyid[g]) for g in con.geom]
        if tb not in b:continue
        rb=b[1] if b[0]==tb else b[0];name=body_names[rb]
        if rb==0 or 'object' in name:continue
        mujoco.mj_contactForce(m,d,i,f);load=max(0.,float(f[0]))
        if name.startswith('left_') and any(k in name for k in ('elbow','shoulder','wrist','magpie','gripper')):arm+=load
        elif name in ('pelvis','torso_link'):trunk+=load
        else:other+=load
    hand=cs.point_world(m,d,cs.REACH_BODY,cs.REACH_OFF)
    live.append([d.time,np.linalg.norm(hand-target),d.qpos[2],arm,trunk,other,*hand])
    orig_push()
plant._push=push

def perturb(q,v,seed):
    s=(0x2545F4914F6CDD1D+seed*0x9E3779B97F4A7C15)&((1<<64)-1)
    def nxt():
        nonlocal s
        s^=(s<<13)&((1<<64)-1);s^=s>>7;s^=(s<<17)&((1<<64)-1)
        return (s>>11)/4503599627370496.-1.
    for _ in range(20):nxt()
    for arr in [q,v]:
        for i in range(len(arr)):arr[i]+=.003*nxt()
    mujoco.mj_normalizeQuat(m,q)

def dwell(t,g):
    best=start=0.;prev=False
    for tt,good in zip(t,g):
        if good and not prev:start=tt
        if good:best=max(best,tt-start)
        prev=good
    return float(best)

results=[]
for seed in map(int,a.seeds.split(',')):
    rp=out/f's{seed}.json'
    if rp.exists():results.append(json.loads(rp.read_text()));continue
    q=fl.start_qpos().copy();v=np.zeros(m.nv);perturb(q,v,seed)
    # Match MJPC's first Transition: move object with slab and clear its velocity.
    j=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,'object_anchor_joint')
    if j>=0 and a.face!=.985:q[m.jnt_qposadr[j]+2]+=a.face-.985;v[m.jnt_dofadr[j]:m.jnt_dofadr[j]+6]=0
    live.clear();print('FLY',a.face,a.tx,a.mode,seed,flush=True)
    rows,summ=B.run_episode(fl,plant,a.seconds,seed=0,initial_qpos=q,initial_qvel=v)
    B.write_csv(str(out/f's{seed}.csv'),rows,fl,target,'stand' if not subset else 'brace',seed,a.seconds,'')
    z=np.asarray(live);np.savetxt(out/f's{seed}.live.csv',z,delimiter=',',header='t,reach_error,pelvis_z,brace_N,trunk_N,other_N,hand_x,hand_y,hand_z',comments='')
    tail=z[:,0]>=a.seconds-5;good=(z[:,1]<=.07)&(z[:,2]>.8)&(z[:,3]>=20)&(z[:,4]<10)
    summ['seed']=seed
    rec=dict(face=a.face,target=target.tolist(),mode=a.mode,seconds=a.seconds,mujoco=mujoco.__version__,
      pelvis_min=float(z[:,2].min()),tail_reach_p95_mm=float(np.percentile(z[tail,1],95)*1000),tail_brace_p05_N=float(np.percentile(z[tail,3],5)),tail_trunk_peak_N=float(z[tail,4].max()),
      tail_joint_fraction=float(good[tail].mean()),tail_strict_fraction=float((good[tail]&(z[tail,1]<=.03)).mean()),longest_joint_dwell_s=dwell(z[:,0],good),success=bool((z[:,2]>.65).all() and good[tail].mean()>=.95 and dwell(z[tail,0],good[tail])>=2),**summ)
    rp.write_text(json.dumps(rec,indent=1));results.append(rec);(out/'summary.json').write_text(json.dumps(results,indent=1));print('RESULT',rec,flush=True)
