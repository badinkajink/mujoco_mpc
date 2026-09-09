#!/usr/bin/env python3
"""Serial, capped MJPC experiments; immutable per-run strategy and command.
Only the isolated worktree strategy is replaced, then restored in finally.
"""
import argparse,json,os,subprocess,time,re,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
ap=argparse.ArgumentParser();ap.add_argument('jobs');a=ap.parse_args()
strategy=ROOT/'mjpc/tasks/humanoid_bench/lean/strategies/h12_brace_targeting.json'
original=strategy.read_bytes()
for job in json.loads(Path(a.jobs).read_text()):
    out=ROOT/'studies/table_height/runs/codex_skunkworks/mjpc'/job['tag'];out.mkdir(parents=True,exist_ok=True)
    if (out/'result.json').exists():continue
    if (ROOT/'studies/table_height/skunkworks/STOP').exists():break
    (out/'binary.sha256').write_text(hashlib.sha256((ROOT/'build_cmake/bin/lean_bench').read_bytes()).hexdigest())
    if os.getloadavg()[0]>os.cpu_count()/2:raise RuntimeError('load too high')
    available=int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')))
    if available<12*1024**2:raise RuntimeError('less than 12 GiB available; resume later')
    s=json.loads(original)
    if 'rtt' in job:s[2]['reach_target_table']=job['rtt']
    if 'dwell' in job:s[2]['success_sustain_time']=job['dwell']
    (out/'strategy.json').write_text(json.dumps(s,indent=1))
    cmd=['systemd-run','--user','--scope','--quiet','-p','CPUQuota=700%','-p','MemoryMax=11G','nice','-n','15',str(ROOT/'build_cmake/bin/lean_bench'),'--task','Lean H12 Magpie','--strategy','25','--table_h',str(job['h']),'--seed',str(job.get('seed',0)),'--threads','6','--spp',str(job.get('spp',3)),'--total_time',str(job.get('seconds',75)),'--out',str(out/'run.csv'),'--qpos_out',str(out/'run.qpos.csv')]+job.get('extra',[])
    (out/'command.json').write_text(json.dumps(cmd,indent=1));(out/'job.json').write_text(json.dumps(job,indent=1))
    print('START',job['tag'],flush=True);t=time.time()
    try:
        strategy.write_text(json.dumps(s,indent=1))
        with (out/'stdout.log').open('w') as f,(out/'stderr.log').open('w') as e:
            p=subprocess.run(cmd,stdout=f,stderr=e,timeout=1500,env={**os.environ,'ROS_DOMAIN_ID':'1'})
    finally:strategy.write_bytes(original)
    log=(out/'stderr.log').read_text(); match=re.search(r'\[bench-summary\] (.*)',log)
    result=dict(job=job,rc=p.returncode,wall_s=time.time()-t,summary=match.group(1) if match else None)
    (out/'result.json').write_text(json.dumps(result,indent=1));print('DONE',result,flush=True)
    if not match:raise RuntimeError('bench failed before summary, inspect logs')
