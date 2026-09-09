#!/usr/bin/env python3
"""Run a frozen randomized manifest serially with immutable per-run provenance."""
import argparse,fcntl,hashlib,json,os,re,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];STUDY=ROOT/'studies/table_height/robustness';RUNS=ROOT/'studies/table_height/runs/robustness_20260909'
STRATEGY=ROOT/'mjpc/tasks/humanoid_bench/lean/strategies/h12_brace_targeting.json'
PYTHON='/home/humanoid/miniconda3/envs/croco/bin/python'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('manifest');a=ap.parse_args();manifest=Path(a.manifest).resolve();jobs=json.loads(manifest.read_text())
 RUNS.mkdir(parents=True,exist_ok=True)
 with (STUDY/'run.lock').open('w') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  original=STRATEGY.read_bytes();binary=ROOT/'build_cmake/bin/lean_bench';binary_hash=sha(binary)
  base=json.loads(original);(STUDY/'baseline_strategy.json').write_bytes(original)
  for job in jobs:
   if (STUDY/'STOP').exists():print('STOP requested',flush=True);break
   out=RUNS/job['tag']
   if (out/'evaluation.json').exists():
    assert json.loads((out/'job.json').read_text())==job,'cached job differs';continue
   if out.exists():raise RuntimeError(f'Incomplete existing run: {out}; inspect, do not overwrite')
   assert sha(binary)==binary_hash,'binary changed mid-block'
   assert STRATEGY.read_bytes()==original,'strategy changed outside runner'
   if os.getloadavg()[0]>os.cpu_count()/2:raise RuntimeError('system load too high')
   av=int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')))
   if av<12*1024**2:raise RuntimeError('less than 12 GiB available')
   out.mkdir();s=json.loads(original);s[2]['reach_target_table']=[job['depth'],.04,.15];s[2]['success_sustain_time']=job['dwell']
   for e in job['edits']:
    node=s
    for key in e['path'][:-1]:node=node[key]
    node[e['path'][-1]]=e['value']
   snapshot=(json.dumps(s,indent=2)+'\n').encode();(out/'strategy.json').write_bytes(snapshot);(out/'job.json').write_text(json.dumps(job,indent=2)+'\n')
   cmd=['systemd-run','--user','--scope','--quiet','-p','CPUQuota=700%','-p','MemoryMax=11G','nice','-n','15',str(binary),'--task','Lean H12 Magpie','--strategy','25','--table_h',str(job['h']),'--seed',str(job['seed']),'--perturb',str(job['perturb']),'--threads','6','--spp',str(job['spp']),'--total_time',str(job['seconds']),'--sync_planning_model',str(job['sync']),'--pose_track',str(job['pose']),'--out',str(out/'run.csv'),'--qpos_out',str(out/'run.qpos.csv'),'--state_out',str(out/'state.csv'),'--model_out',str(out/'plant.mjb')]
   for k,v in job['numeric'].items():cmd+=['--numeric',f'{k}={v}']
   cmd+=job.get('extra',[])
   (out/'command.json').write_text(json.dumps(cmd,indent=2)+'\n');(out/'provenance.json').write_text(json.dumps({'binary_sha256':binary_hash,'strategy_sha256':hashlib.sha256(snapshot).hexdigest(),'manifest_sha256':sha(manifest),'source_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'started_unix':time.time()},indent=2)+'\n')
   print('START',job['tag'],flush=True);start=time.time();rc=None;error=None
   try:
    STRATEGY.write_bytes(snapshot)
    with (out/'stdout.log').open('w') as f,(out/'stderr.log').open('w') as e:
     proc=subprocess.run(cmd,stdout=f,stderr=e,cwd=ROOT,env={**os.environ,'ROS_DOMAIN_ID':'1'},timeout=1800);rc=proc.returncode
   except Exception as exc:error=repr(exc)
   finally:STRATEGY.write_bytes(original)
   log=(out/'stderr.log').read_text();match=re.search(r'\[bench-summary\] (.*)',log)
   result={'job':job,'rc':rc,'error':error,'wall_s':time.time()-start,'summary':match.group(1) if match else None}
   (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
   if error or not match or rc not in [0,1]:raise RuntimeError(result)
   env={**os.environ,'PYTHONPATH':str(ROOT/'studies/table_height/skunkworks/vendor_mujoco323'),'OPENBLAS_NUM_THREADS':'1'}
   subprocess.run([PYTHON,str(STUDY/'evaluate.py'),str(out)],env=env,check=True)
   ev=json.loads((out/'evaluation.json').read_text());print('DONE',job['tag'],'strict',ev['strict_success'],'70mm',ev['loaded70_success'],'full',ev['full_success'],'dwell',round(ev['dwell30_s'],2),'wall',round(result['wall_s']),flush=True)
if __name__=='__main__':main()
