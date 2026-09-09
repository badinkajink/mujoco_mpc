from pathlib import Path
import csv,json,time
root=Path(__file__).resolve().parents[3];runs=root/'studies/table_height/runs/robustness_20260909'
complete=[];active=[]
for p in runs.glob('*/job.json'):
 j=json.loads(p.read_text())
 if j['stage']=='verification':continue
 if (p.parent/'evaluation.json').exists():complete.append(json.loads((p.parent/'evaluation.json').read_text()))
 elif not (p.parent/'result.json').exists():active.append(p.parent)
print('Completed',len(complete),'strict',sum(r['strict_success'] for r in complete),'full',sum(r['full_success'] for r in complete),'falls',sum(not r['safe'] for r in complete))
for p in active:
 latest=None
 with (p/'run.csv').open() as f:
  for row in csv.DictReader(f):
   if row.get('brace_normal_N') is not None:latest=row
 if latest:print('Running',p.name,'t',latest['t'],'phase',latest['phase'],'tilt',latest['torso_tilt_deg'])
for r in sorted(complete,key=lambda x:x['job']['tag']):print(r['job']['tag'],r['strict_success'],r['full_success'],round(r['dwell30_s'],2),r['failure_class'])
