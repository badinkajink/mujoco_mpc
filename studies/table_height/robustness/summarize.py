#!/usr/bin/env python3
"""Generate the run ledger and figures from immutable independently scored runs."""
import argparse,collections,datetime,hashlib,html,json,math
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3];STUDY=ROOT/'studies/table_height/robustness';RUNS=ROOT/'studies/table_height/runs/robustness_20260909'
PAGE=ROOT/'docs/lean/20260909-table_height_robustness.html';MEDIA=ROOT/'docs/lean/media/robustness_20260909'
def wilson(k,n):
 if not n:return (0.,1.)
 z=1.959963984540054;p=k/n;den=1+z*z/n;center=(p+z*z/(2*n))/den;half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
 return center-half,center+half

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--plots',action='store_true');a=ap.parse_args()
 results=[]
 for p in sorted(RUNS.glob('*/evaluation.json')):
  r=json.loads(p.read_text())
  if r['job']['stage']=='verification':continue
  results.append(r)
 groups=collections.defaultdict(list)
 for r in results:groups[(r['job']['stage'],r['job']['arm'],r['job']['h'])].append(r)
 summary=[]
 for (stage,arm,h),rs in sorted(groups.items()):
  n=len(rs);k=sum(r['strict_success'] for r in rs);lo,hi=wilson(k,n)
  summary.append(dict(stage=stage,arm=arm,h=h,n=n,strict=k,loaded70=sum(r['loaded70_success'] for r in rs),full=sum(r['full_success'] for r in rs),ladder=sum(r['ladder_complete'] for r in rs),recovered=sum(r['recovered'] for r in rs),falls=sum(not r['safe'] for r in rs),ci95=[lo,hi],tags=[r['job']['tag'] for r in rs]))
 (STUDY/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 print('Completed',len(results),'episodes; wall hours',round(sum(r['wall_s'] for r in results)/3600,2))
 for r in summary:print(r['stage'],r['arm'],r['h'],'n',r['n'],'strict',r['strict'],'70mm',r['loaded70'],'full',r['full'],'ladder',r['ladder'],'falls',r['falls'])
 def esc(x):return html.escape(str(x))
 def link(path,title):return '<a href="'+esc(path)+'">'+esc(title)+'</a>'
 frozen=(STUDY/'findings.html').read_text() if (STUDY/'findings.html').exists() else '<p><strong>Study in progress.</strong> These are partial counts; no candidate has been confirmed. Endpoint definitions and planned runs were frozen before the experiment.</p>'
 text='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Minimal changes for table-height robustness</title><style>body{font:17px/1.55 system-ui,sans-serif;max-width:1200px;margin:40px auto;padding:0 24px;color:#1d2835;background:#fafbfd}h1{font-size:36px;line-height:1.15}h2{margin-top:36px}table{border-collapse:collapse;width:100%;font-size:14px;background:white}th,td{padding:8px;border-bottom:1px solid #d6dde6;text-align:left}th{background:#e9eef5}.scroll{overflow-x:auto}code,pre{background:#eef1f5}pre{padding:14px;white-space:pre-wrap}img,video{max-width:100%}a{color:#075eaa}.muted{color:#546474}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:20px}figcaption{font-size:14px}</style><h1>Minimal changes for table-height robustness</h1>'''
 text+=f'<p class="muted">Updated {esc(datetime.datetime.now().isoformat(timespec="seconds"))} local · {len(results)} completed new episodes · simulation only</p>'+frozen
 text+='<h2>Study design and audit</h2><p>Each ablation changes exactly one factor relative to the reference recipe: synchronized planning model, existing DLS brace retargeting, 0.40 m reach depth, and five-second reach gate. Depth is measured from the near edge; lateral and vertical offsets remain −0.04 m and +0.15 m. A closer target changes task difficulty; it is not counted as an optimizer improvement. Initial-state seeds 10–12 are interleaved in a frozen randomized order. Sampler randomness remains uncontrolled; these are blocked trials, not paired random-number experiments. All use six threads, 167 simulated planning Hz, 75 s episodes and the same physics model.</p>'
 text+='<p>The passive full-state recorder and independent native evaluator share the benchmark’s exact MuJoCo library. The previous Python package reported the same version string but used an incompatible model format. New metrics reconstruct FK and forces together from pre-integration qpos, qvel, control and warm-start. Model synchronization is checked from saved binary models.</p>'
 text+='<p><strong>Primary success:</strong> two continuous observed seconds in reach phase, error ≤30 mm, upward normal left-arm/table support ≥20 N, trunk/table and other robot/table loads each &lt;10 N, pelvis &gt;0.8 m; no episode fall and whole-episode pelvis ≥0.65 m. The 70 mm result is secondary. Full-task success also requires sequence completion and a two-second final standing interval with tilt ≤15°, pelvis &gt;0.8 m, and brace load &lt;10 N. Contact/force data are sampled at 50 Hz; unsampled-time stability is not certified.</p>'
 text+='<p>'+link('../../studies/table_height/robustness/protocol.md','Frozen protocol')+' · '+link('../../studies/table_height/robustness/ablation.json','Randomized ablation manifest')+' · '+link('../../studies/table_height/robustness/engine.json','Exact engine/build identities')+' · '+link('../../studies/table_height/robustness/summary.json','Machine-readable counts')+' · '+link('../../studies/table_height/robustness/decisions.md','Timestamped decisions and adaptive rules')+' · '+link('../../studies/table_height/robustness/ablation_integrity.json','Ablation integrity checks')+'</p>'
 text+='<h2>All condition counts</h2><p><strong>Sampling-boundary caveat:</strong> a two-second strategy gate can leave only 1.98 s between the first and last passing 50 Hz samples. Such runs fail the conservative observed two-second criterion here; a one-sample sensitivity count is provided below. This must not be interpreted as proof that the physical hold was shorter than two seconds. The primary 30 mm criterion is unchanged.</p><p>Exploration and fresh confirmation are separate. Intervals are Wilson 95% intervals for the ≤30 mm success fraction under the tested condition; they assume independent trials and do not certify a continuous height range or disturbance robustness. Three successes still give a lower bound of only about 44%.</p><div class="scroll"><table><tr><th>Stage</th><th>Arm</th><th>Face m</th><th>N</th><th>≤30 mm</th><th>95% interval</th><th>≤70 mm</th><th>Full task</th><th>Ladder</th><th>Falls</th></tr>'
 for r in summary:text+='<tr>'+''.join('<td>'+esc(v)+'</td>' for v in [r['stage'],r['arm'],f"{r['h']:.3f}",r['n'],f"{r['strict']}/{r['n']}",f"{r['ci95'][0]:.0%}–{r['ci95'][1]:.0%}",f"{r['loaded70']}/{r['n']}",f"{r['full']}/{r['n']}",f"{r['ladder']}/{r['n']}",r['falls']])+'</tr>'
 text+='</table></div><p>One-sample sensitivity (same safety requirements, allowing 1.98 s observed dwell): '+', '.join(esc(f"{stage}/{arm}/{h:.3f}: 30 mm {sum(r['safe'] and r['dwell30_s']>=1.98-1e-8 for r in rs)}/{len(rs)}, 70 mm {sum(r['safe'] and r['dwell70_s']>=1.98-1e-8 for r in rs)}/{len(rs)}") for (stage,arm,h),rs in sorted(groups.items()))+'</p><h2>One-factor contributions</h2><p>Each difference is alternative minus reference at the same height in this experiment. These are conditional effects in the reference context; the design does not identify all interactions or justify adding gains across factors. The legacy-model arm deliberately reintroduces a benchmark defect and is not a controller candidate.</p><table><tr><th>Stage / one change</th><th>Face</th><th>Alternative ≤30 mm</th><th>Reference ≤30 mm</th><th>Observed difference</th><th>Full-task difference</th></tr>'
 refs={(r['stage'],r['h']):r for r in summary if r['arm']=='reference'}
 for r in summary:
  if r['stage']=='confirmation' or r['arm']=='reference':continue
  ref=refs.get((r['stage'],r['h']))
  if ref:text+='<tr>'+''.join('<td>'+esc(v)+'</td>' for v in [r['stage']+' / '+r['arm'],f"{r['h']:.3f}",f"{r['strict']}/{r['n']}",f"{ref['strict']}/{ref['n']}",f"{100*(r['strict']/r['n']-ref['strict']/ref['n']):+.0f} percentage points",f"{100*(r['full']/r['n']-ref['full']/ref['n']):+.0f} percentage points"] )+'</tr>'
 text+='</table>'
 if a.plots and results:
  import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
  MEDIA.mkdir(parents=True,exist_ok=True)
  for stage in sorted({r['stage'] for r in summary}):
   cells=[r for r in summary if r['stage']==stage];arms=sorted({r['arm'] for r in cells});fig,axes=plt.subplots(1,2,figsize=(12,4.6),sharey=True)
   for ai,arm in enumerate(arms):
    rr=sorted([r for r in cells if r['arm']==arm],key=lambda r:r['h']);off=(ai-(len(arms)-1)/2)*.004
    for ax,field,title in zip(axes,['strict','full'],['Loaded reach ≤30 mm','Loaded reach and recovered']):
     xx=[r['h']+off for r in rr];yy=[r[field]/r['n'] for r in rr]
     ax.scatter(xx,yy,label=arm,s=45)
     for x,y,r in zip(xx,yy,rr):ax.annotate(f"{r[field]}/{r['n']}",(x,y),xytext=(0,7),textcoords='offset points',ha='center',fontsize=8)
     ax.set_title(title);ax.set_xlabel('Table face (m)');ax.set_ylim(-.1,1.18);ax.set_xticks(sorted({r['h'] for r in cells}));ax.grid(alpha=.2)
   axes[0].set_ylabel('Observed success fraction');axes[1].legend(fontsize=8,loc='upper left',bbox_to_anchor=(1,1));fig.suptitle(stage+' — discrete tested conditions, no interpolated boundary');fig.tight_layout();fig.savefig(MEDIA/(stage+'.png'),dpi=160);plt.close(fig)
  for stage in sorted({r['stage'] for r in summary}):text+=f'<figure><img src="media/robustness_20260909/{esc(stage)}.png" alt="{esc(stage)} success counts"><figcaption>Each label is successes/trials. Horizontal offsets separate arms; they do not represent different heights.</figcaption></figure>'
 if (MEDIA/'low_retarget_trace.png').exists():text+='<figure><img src="media/robustness_20260909/low_retarget_trace.png" alt="Illustrative low-table retargeting comparison"><figcaption>Illustrative initial-state seed 10 only. Shading marks the reference’s successful precise hold; population/repeatability evidence comes from the complete condition counts above.</figcaption></figure>'
 if (MEDIA/'high_approach_trace.png').exists():text+='<figure><img src="media/robustness_20260909/high_approach_trace.png" alt="Signed pitch and contact through the tall-table approach"><figcaption>All three tall-table reference trials. The two failures retreat and pitch backward in phase 1; reach targets are not yet active. This observation motivates the separate, controlled pitch-tracking screen.</figcaption></figure>'
 text+='<h2>Physical diagnostics and complete run ledger</h2><p>Every completed run is retained. Model force limits are not independently verified hardware torque limits. Maximum foot displacement includes the whole episode and cannot by itself distinguish foot roll from sliding. Joint-limit and penetration peaks are diagnostics, not claims of mechanically exact contact. Inspect the replay and phase traces before interpreting motion quality.</p><div class="scroll"><table><tr><th>Run</th><th>Strict</th><th>70 mm</th><th>Full</th><th>30 mm dwell s</th><th>Outcome</th><th>Joint limit max °</th><th>Joint speed max rad/s</th><th>Force-limit fraction</th><th>Foot motion max mm</th><th>Contact penetration max mm</th><th>Data</th></tr>'
 for r in results:
  tag=r['job']['tag'];outcome=r['failure_class']
  if r['safe'] and r['max_phase']==8 and not r['ladder_complete']:outcome='deadline in final standing'
  elif r['ladder_complete'] and not r['recovered']:outcome='terminal contact/tilt'
  elif not r['strict_success'] and r.get('reach_min_mm',1000)<=30:outcome='precise dwell <2 s'
  rel='../../studies/table_height/runs/robustness_20260909/'+tag+'/'
  vals=[tag,r['strict_success'],r['loaded70_success'],r['full_success'],f"{r['dwell30_s']:.2f}",outcome,f"{r['max_joint_limit_violation_deg']:.2f}",f"{r['max_joint_velocity_rad_s']:.2f}",f"{r['max_actuator_force_fraction']:.3f}",f"{1000*r['max_foot_displacement_m']:.1f}",f"{1000*r['max_penetration_m']:.1f}"]
  text+='<tr>'+''.join('<td>'+esc(v)+'</td>' for v in vals)+'<td>'+link(rel+'evaluation.json','score')+' · '+link(rel+'metrics.csv','trace')+' · '+link(rel+'command.json','command')+' · '+link(rel+'strategy.json','strategy')+' · '+link(rel+'stderr.log','log')+'</td></tr>'
 text+='</table></div>'
 if (STUDY/'videos.json').exists():
  text+='<h2>Representative replays</h2><div class="grid">'
  for v in json.loads((STUDY/'videos.json').read_text()):text+=f'<figure><video controls preload="metadata" src="media/robustness_20260909/{esc(v["file"])}"></video><figcaption>{esc(v["caption"])}</figcaption></figure>'
  text+='</div>'
 text+='<h2>Reproduction and handoff</h2><p>'+link('../../studies/table_height/robustness/README.md','Commands, engineering changes and handoff')+'</p>'
 PAGE.write_text(text+'</html>');print(PAGE)
if __name__=='__main__':main()
