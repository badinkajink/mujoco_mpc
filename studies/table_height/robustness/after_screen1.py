#!/usr/bin/env python3
"""Apply the predeclared screen rule; launch fresh confirmation only if it passes.
No tuning or discretionary choices occur here. A rejected screen returns for
failure analysis; it never substitutes a recipe or silently runs a holdout.
"""
import datetime,fcntl,json,random,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];S=ROOT/'studies/table_height/robustness';R=ROOT/'studies/table_height/runs/robustness_20260909'
def main():
 jobs=json.loads((S/'screen1.json').read_text())
 while not all((R/j['tag']/'evaluation.json').exists() for j in jobs):
  if (S/'STOP').exists():print('STOP requested; confirmation not started',flush=True);return
  time.sleep(10)
 # Wait until the batch releases its lock and restores the baseline strategy.
 with (S/'run.lock').open() as f:fcntl.flock(f,fcntl.LOCK_EX)
 cells={}
 for j in jobs:
  e=json.loads((R/j['tag']/'evaluation.json').read_text());assert e['job']==j
  c=cells.setdefault((j['arm'],j['h']),dict(n=0,strict=0,ladder=0,full=0))
  c['n']+=1
  for k,v in [('strict','strict_success'),('ladder','ladder_complete'),('full','full_success')]:c[k]+=e[v]
 assert len(cells)==6 and all(c['n']==3 for c in cells.values())
 ref=lambda h:cells[('reference',h)];new=lambda h:cells[('pitch_track',h)]
 eligible=(new(1.085)['strict']>ref(1.085)['strict'] and new(.985)['strict']>=ref(.985)['strict'] and new(.985)['ladder']>=ref(.985)['ladder'] and new(.785)['strict']>=ref(.785)['strict'])
 decision={'time':datetime.datetime.now().isoformat(timespec='seconds'),'eligible':eligible,'cells':[{'arm':a,'h':h,**v} for (a,h),v in sorted(cells.items())],'rule':'High strict count improves; low strict and nominal strict/ladder do not decrease, versus contemporaneous controls.'}
 target=S/'screen1_decision.json'
 if target.exists():raise RuntimeError('Decision exists; inspect before resuming, never refreeze automatically')
 target.write_text(json.dumps(decision,indent=2)+'\n');print(json.dumps(decision),flush=True)
 if not eligible:
  print('SCREEN REJECTED: return to evidence-based analysis; no confirmation launched',flush=True);return
 # Recipe is selected before any holdout episode and remains identical at all heights.
 jobs=[]
 for seed in [100,101,102]:
  block=[]
  for h in [.785,.885,.985,1.035,1.085]:
   block.append(dict(tag=f'confirm_pitch_track_h{round(h*1000):04d}_s{seed}',stage='confirmation',arm='pitch_track',h=h,seed=seed,sync=1,pose=1,depth=.4,dwell=5.,spp=3,seconds=75,perturb=.003,numeric={'brace_pitch_track':1},edits=[]))
  random.Random(20261009+seed).shuffle(block);jobs+=block
 manifest=S/'confirmation.json'
 if manifest.exists():raise RuntimeError('Confirmation manifest exists; inspect, never overwrite')
 manifest.write_text(json.dumps(jobs,indent=2)+'\n')
 with (S/'decisions.md').open('a') as f:f.write('\n## '+decision['time']+' — screen 1 accepted by frozen rule; confirmation recipe frozen\n\nThe executable selection rule in after_screen1.py passed (counts in screen1_decision.json). Freeze sync1, pose1, depth.40, dwell5, brace_pitch_track1, gain1, spp3; no other numeric or strategy changes. Confirmation is seeds100–102 at .785/.885/.985/1.035/1.085, 15 randomized trials, manifest confirmation.json. No further tuning, regardless of outcome. Screen evidence remains exploratory.\n')
 print('CONFIRMATION FROZEN: 15 fresh trials; no more tuning',flush=True)
 subprocess.run(['python3',str(S/'run.py'),str(manifest)],cwd=ROOT,check=True)
 subprocess.run(['python3',str(S/'summarize.py'),'--plots'],cwd=ROOT,check=True)
 print('CONFIRMATION COMPLETE; interpretation/report review still required',flush=True)
if __name__=='__main__':main()
