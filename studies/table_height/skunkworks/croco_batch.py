#!/usr/bin/env python3
import argparse,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
ap=argparse.ArgumentParser();ap.add_argument('jobs');a=ap.parse_args()
for j in json.loads(Path(a.jobs).read_text()):
    if (HERE/'STOP').exists():break
    if os.getloadavg()[0]>os.cpu_count()/2:raise RuntimeError('load too high')
    out=ROOT/'studies/table_height/runs/codex_skunkworks/croco'/j['tag'];out.mkdir(parents=True,exist_ok=True)
    if all((out/f's{seed}.json').exists() for seed in j.get('seeds','0').split(',')):
        print('CACHED',j['tag'],flush=True);continue
    env={**os.environ,'PYTHONPATH':str(HERE/'vendor_mujoco323')+':/home/humanoid/Programs/crocoddyl_mpc_codex_tableheight_20260908','CL_ASSETS_DIR':'/home/humanoid/Programs/Humanoid_Simulation/CL_Assets','OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'6','CMPC_THREADS':'6','ROS_DOMAIN_ID':'1'}
    cmd=['systemd-run','--user','--scope','--quiet','-p','CPUQuota=600%','-p','MemoryMax=8G','nice','-n','15','/home/humanoid/miniconda3/envs/croco/bin/python',str(HERE/'run_croco.py'),'--croco','/home/humanoid/Programs/crocoddyl_mpc_codex_tableheight_20260908','--model_dir',str(ROOT/'build_cmake/mjpc/tasks/humanoid_bench/lean'),'--out',str(out),'--face',str(j['h']),'--tx',str(j.get('tx',1.0)),'--mode',j.get('mode','elbow+forearm'),'--seeds',j.get('seeds','0')]+j.get('extra',[])
    (out/'command.json').write_text(json.dumps(dict(command=cmd,env={k:env[k] for k in ['PYTHONPATH','CL_ASSETS_DIR','OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','CMPC_THREADS']}),indent=2))
    print('START',j['tag'],flush=True);t=time.time()
    with (out/'driver.log').open('a') as f:p=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=1200)
    print('DONE',j['tag'],'rc',p.returncode,'wall',time.time()-t,flush=True)
    if p.returncode:raise RuntimeError('Crocoddyl failed; inspect '+str(out/'driver.log'))
