#!/usr/bin/env python3
"""Generate the local report from recorded metrics. Run after analyze_mjpc.py."""
from pathlib import Path
import json,re,html,collections,datetime,os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[3];RAW=ROOT/'studies/table_height/runs/codex_skunkworks';MEDIA=ROOT/'docs/lean/media/skunkworks';MEDIA.mkdir(parents=True,exist_ok=True)
selection_path=ROOT/'studies/table_height/skunkworks/study_manifest.json'
selection=json.loads(selection_path.read_text()) if selection_path.exists() else None
M=json.loads((RAW/'mjpc/audit.json').read_text()) if (RAW/'mjpc/audit.json').exists() else []
if selection:M=[r for r in M if r['job']['tag'] in selection['mjpc_tags']]
C=[]
for p in sorted((RAW/'croco').glob('*/s*.json')):
 if re.fullmatch('s[0-9]+.json',p.name):
  r=json.loads(p.read_text());r['tag']=p.parent.name;r['path']=os.path.relpath(p,ROOT/'docs/lean');inv=json.loads((p.parent/'invocation.json').read_text());r['matched_jaw']=inv['env'].get('REACH_BODY')=='right_magpie_gripper';r['start']=inv['args']['start'];C.append(r)
if selection:C=[r for r in C if f"{r['tag']}/s{r['seed']}" in selection['croco_episodes']]
P=json.loads((RAW/'prior_croco_audit/audit.json').read_text())
P=[r for r in P if abs(r['target'][0]-1.06)<1e-6]
for r in M:
 log=(RAW/'mjpc'/r['job']['tag']/'stderr.log').read_text()
 r['planning_sync']='[bench-model-after]' in log
summary=dict(mjpc=M,croco=C,prior_croco=P)
(MEDIA/'summary.json').write_text(json.dumps(summary,indent=2))
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(1,2,figsize=(11,4),layout='constrained')
for ax,records,label in [(axs[0],P,'Prior CMPC: original model / wrist point'),(axs[1],[r for r in C if r['matched_jaw'] and r['start']=='home' and abs(r['target'][0]-.85)<1e-6],'CMPC: verified jaw, short target x=0.85')]:
 for mode,mark,col in [('elbow+forearm','o','#3766a0'),('elbow+wrist','s','#ad632f'),('forearm','^','#6a8d3c')]:
  rr=[r for r in records if r.get('mode','elbow+forearm')==mode and r.get('arm','brace')=='brace']
  for h in sorted(set(r['face'] for r in rr)):
   z=[r for r in rr if r['face']==h];n=len(z);k=sum(r['success'] for r in z)
   ax.scatter(h,k/n,marker=mark,color=col,s=65);ax.annotate(f'{k}/{n}',(h,k/n),xytext=(3,8),textcoords='offset points',fontsize=9)
 ax.set(xlabel='Table face (m)',ylabel='Successful loaded hold / trials',title=label,ylim=(-.08,1.2),xlim=(.755,1.115));ax.grid(alpha=.2)
fig.savefig(MEDIA/'height_counts.png',dpi=160);plt.close(fig)
# Recorded MJPC reach and load, including failed arms, no qpos-only force inference.
tags=['pose1_short_h0785_s2','pose1_short_h0785_s0','short_h0985_s0','pose1_short_h1085_s0']
trace_labels=['0.785 DLS seed 2: missed reach','0.785 DLS seed 0','0.985 identity keys seed 0','1.085 DLS seed 0']
fig,axs=plt.subplots(2,1,figsize=(10,6),sharex=True,layout='constrained')
for tag,label in zip(tags,trace_labels):
 p=RAW/'mjpc'/tag/'metrics.csv'
 if not p.exists():continue
 z=np.genfromtxt(p,names=True,delimiter=',');axs[0].plot(z['t'],1000*z['reach_error'],label=label);axs[1].plot(z['t'],z['brace_N'])
axs[0].axhline(70,c='k',ls=':',lw=1);axs[0].set(ylabel='Jaw target error (mm)',ylim=(0,500));axs[0].legend(fontsize=8,ncol=2)
axs[1].axhline(20,c='k',ls=':',lw=1);axs[1].set(ylabel='Left-arm normal load (N)',xlabel='Simulated time (s)',ylim=(0,300))
for ax in axs:ax.grid(alpha=.2)
fig.savefig(MEDIA/'mjpc_traces.png',dpi=160);plt.close(fig)
# Separate panels prevent overlapping counts and pooling of configurations.
fig,axes=plt.subplots(1,3,figsize=(11.5,4.6),sharey=True,layout='constrained')
for ax,(rate,pose,title) in zip(axes,[(3,0,'Authored keys, 167 Hz'),(3,1,'DLS keys, 167 Hz'),(10,0,'Authored keys, 50 Hz')]):
 groups=collections.defaultdict(list)
 for r in M:
  j=r['job'];extra=j.get('extra',[])
  pval=int(extra[extra.index('--pose_track')+1]) if '--pose_track' in extra else 0
  # The legacy nominal baseline has identical plant/planner geometry.
  if (not r['planning_sync'] and j['h']!=.985) or j.get('spp',3)!=rate or pval!=pose or any(k in extra for k in ['--stance_shift_x','--numeric','--start_key']):continue
  groups[(j['h'],round(r['target'][0],3))].append(r)
 for (h,x),rr in groups.items():
  k=sum(r['dynamic_success'] for r in rr);n=len(rr)
  ax.scatter(x,h,c=[k/n],cmap='RdYlGn',vmin=0,vmax=1,s=90,edgecolors='k',lw=.5)
  ax.annotate(f'{k}/{n}',(x,h),xytext=(8,0),textcoords='offset points',va='center',fontsize=10)
 ax.set(title=title,xlim=(.80,1.075),ylim=(.75,1.12),xticks=[.85,1.0]);ax.grid(alpha=.2)
axes[0].set_ylabel('Table face (m)')
fig.supxlabel('Jaw target world x (m); near edge x = 0.45 m')
fig.suptitle('Sampled loaded reaches: successes / trials',fontsize=13)
fig.savefig(MEDIA/'workspace.png',dpi=160);plt.close(fig)

# Static mode screen: candidates only; failed certification is shown explicitly.
static=json.loads((RAW/'common_static_jaw/cells.json').read_text())
ladder=static['ladder'];cells=static['cells'];arr=np.full((len(cells),len(ladder)),np.nan)
for i,c in enumerate(cells):
 for r in c['modes']:
  name='+'.join(r['subset']) or 'legs_only'
  if r['admissible']:arr[i,ladder.index(name)]=r['brace_force']
fig,ax=plt.subplots(figsize=(11,4.6),layout='constrained')
cm=plt.get_cmap('Blues').copy();cm.set_bad('#eeeeee')
im=ax.imshow(arr,aspect='auto',cmap=cm,vmin=0,vmax=80)
for i in range(len(cells)):
 for k in range(len(ladder)):
  ax.text(k,i,'×' if np.isnan(arr[i,k]) else f'{arr[i,k]:.0f} N',ha='center',va='center',fontsize=9,color='white' if arr[i,k]>50 else 'black')
ax.set_xticks(range(len(ladder)),ladder,rotation=25,ha='right',fontsize=8)
ax.set_yticks(range(len(cells)),[f"h={c['face']:.3f}, x={c['target'][0]:.2f}" for c in cells])
ax.set_title('Static true-jaw contact candidates: × fails screen; numbers are predicted brace load')
fig.colorbar(im,ax=ax,label='Predicted normal load (N)');fig.savefig(MEDIA/'static_modes.png',dpi=160);plt.close(fig)

def esc(x):return html.escape(str(x))
def link(path,label):return f'<a href="{esc(path)}">{esc(label)}</a>'
def val(x,fmt='.1f'):return '—' if x is None else format(x,fmt)
def yn(x):return '<span class="ok">yes</span>' if x else '<span class="no">no</span>'
rows=[]
for r in M:
 j=r['job'];tag=j['tag'];href='../../studies/table_height/runs/codex_skunkworks/mjpc/'+tag+'/'
 rows.append('<tr>'+''.join(f'<td>{v}</td>' for v in [link(href+'result.json',tag),f"{j['h']:.3f}",str(round(500/j.get('spp',3),1)),yn(r['planning_sync']),yn(r['dynamic_success']),yn(r['ladder_complete']),val(r['joint_dwell_s'],'.2f'),val(r['strict_dwell_s'],'.2f'),val(r['reach_min_mm']),link(href+'command.json','command')+' · '+link(href+'strategy.json','strategy')+' · '+link(href+'run.csv','raw')+' · '+link(href+'run.qpos.csv','poses')+' · '+link(href+'stderr.log','log')])+'</tr>')
ct=[]
for r in C:
 base='../../studies/table_height/runs/codex_skunkworks/croco/'+r['tag']+'/'
 ct.append('<tr>'+''.join(f'<td>{v}</td>' for v in [esc(r['tag'])+('' if r['matched_jaw'] else ' (calibration: wrist point)')+f"<br>face {r['face']:.3f}, x={r['target'][0]:.2f}; {esc(r['mode'])}; start {esc(r['start'])}",str(r['seed']),yn(r['success']),val(r['tail_reach_p95_mm']),val(r['tail_brace_p05_N']),val(r['pelvis_min'],'.3f'),val(r['longest_joint_dwell_s'],'.2f'),link(base+f"s{r['seed']}.csv",'trajectory')+' · '+link(base+f"s{r['seed']}.live.csv",'live load')+' · '+link(base+'command.json','command')+' · '+link(base+'driver.log','log')+' · '+link(base+'cell/plan_'+r['mode'].replace('+','_')+'.json','plan')+' · '+link(base+'cell/modes.json','static')])+'</tr>')
main_groups=collections.defaultdict(list)
for r in M:
 j=r['job'];extra=j.get('extra',[])
 if r['planning_sync'] and abs(r['target'][0]-.85)<1e-6 and not any(k in extra for k in ['--stance_shift_x','--numeric','--start_key']):
  pose=int(extra[extra.index('--pose_track')+1]) if '--pose_track' in extra else 0
  main_groups[(j['h'],j.get('spp',3),pose)].append(r)
main_rows=[]
for (h,spp,pose),rr in sorted(main_groups.items()):
 k=sum(r['dynamic_success'] for r in rr);n=len(rr)
 good=[r for r in rr if r['dynamic_success']]
 def span(key,fmt='.1f'):
  vv=[r[key] for r in good if r.get(key) is not None]
  return ('—' if not vv else format(min(vv),fmt)+'–'+format(max(vv),fmt))
 main_rows.append('<tr>'+''.join('<td>'+str(v)+'</td>' for v in [f'{h:.3f}',f'{500/spp:.1f}','DLS' if pose else 'authored',f'{k}/{n}',f"{sum(r['dynamic_success'] and r['strict_dwell_s']>=2 for r in rr)}/{n}",f"{sum(r['ladder_complete'] for r in rr)}/{n}",span('joint_dwell_s','.2f'),span('loaded_reach_p95_mm'),span('loaded_brace_p05_N'),span('loaded_tilt_p50_deg')])+'</tr>')
main_table='<h2>Fixed short target: measured repeats</h2><p>Jaw target is (0.85, −0.04, face + 0.15) m at every height, 0.40 m beyond the near edge. Each loaded-reach count requires at least two continuous seconds with ≥20 N left-arm normal load and no fall; columns separate 70 mm and 30 mm reach tolerance. Authored and DLS-retargeted brace keys are separated. Same reset, no stance or explicit base-height gain. At nominal height DLS makes no edit. Successful-run ranges below exclude failed trials, which remain in the denominator and raw table. Error, force and tilt summaries use samples meeting the joint condition.</p><div class="tw"><table><tr><th>Face m</th><th>Plan Hz</th><th>Brace keys</th><th>Loaded reach ≤70 mm</th><th>Loaded reach ≤30 mm</th><th>Ladder</th><th>Dwell s</th><th>Loaded error p95 mm</th><th>Loaded force p05 N</th><th>Loaded tilt median °</th></tr>'+''.join(main_rows)+'</table></div>'

css=re.search(r'<style>(.*?)</style>',(ROOT/'docs/lean/20260908-table_height_full_report.html').read_text(),re.S).group(1)
lead=(ROOT/'studies/table_height/skunkworks/report_findings.html').read_text() if (ROOT/'studies/table_height/skunkworks/report_findings.html').exists() else '<p>Investigation in progress. Tables below contain completed runs only.</p>'
page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Table-height generalization with synchronized planning models</title><style>'''+css+'''\nmain{max-width:1200px}.tw{overflow-x:auto}td{font-size:13px}video{max-width:100%}pre{white-space:pre-wrap}.replay-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:20px}.replay-grid figure{margin:0}.replay-grid video{width:100%}@media(max-width:760px){.replay-grid{grid-template-columns:1fr}}</style><main>
<p class="meta">Local simulation study · 2026-09-08 · no hardware, DDS, uploads or pushes</p>
<h1>Table-height generalization with synchronized planning models</h1>'''+lead+main_table+'''
<h2>Model-copy defect in the height bench</h2>
<p><code>Agent::Initialize</code> copies the model before the first task transition. The task then moves the table and retargets brace keyframes on the plant only. Sampling rollouts use the unchanged agent model. The runtime log at a 1.085 m face reads:</p>
<pre>[bench-model-before] plant_face=1.0850 planner_face=0.9850
[bench-model-after]  plant_face=1.0850 planner_face=1.0850</pre>
<p>The fix copies the post-transition model into the existing planner model before its first optimization. The fixed-height bench has no later environment changes. <code>--sync_planning_model 0</code> reproduces the historical mismatch; synchronization defaults to 1. Controller defaults remain unchanged. Prior off-height failures mix controller behavior with a wrong planning environment and cannot identify an optimizer-only height limit. The failed initial gain=1 trial also never delivered its modified keyframes to planning. The prior six shipped high-table seeds had four zero peak forearm loads and two brief peaks of 20.9 and 31.8 N; those historical peaks are not equivalent to this report’s broader left-arm normal-load statistic.</p>
<p>The successful MJPC pilots load the left forearm body most strongly; the high DLS pilot also loads the left wrist body. The checked successful intervals have no logged reaching-arm or other unattributed table load. Body-level resultant medians and the reaching-arm/other peak are retained in the report JSON, separately from the normal-force success statistic. This supports revisiting contact modes rather than equating an authored posture or a declared elbow–forearm schedule with the observed load path.</p><p>The explicit <code>brace_base_z_gain</code> experiment sets the three brace base-height references to authored z + gain × (face − 0.985), optionally after DLS. It defaults to zero. This deliberate reference edit is not an IK certificate: the DLS residual is measured before overriding base z.</p>
<h2>Dynamic scoring and comparison conditions</h2>
<p>A joint successful interval requires jaw error ≤70 mm, left-arm normal load ≥20 N, pelvis &gt;0.8 m and torso/pelvis table load &lt;10 N for at least two continuous seconds. Whole-run logged pelvis must remain at or above 0.65 m, and MJPC must not trigger its fall detector. MJPC scores the phase-2 reach rung. The 30 mm dwell is reported separately. MJPC may release after reaching; ladder completion is separate. Crocoddyl runs 25 s and must satisfy the joint criterion in at least 95% of the final five seconds, including a two-second continuous interval. The duration and recovery requirements differ and are stated beside the counts.</p>
<p>Dwell is evaluated on sampled logs: MJPC 50 Hz, new CMPC 100 Hz, prior CMPC audit 20 Hz. New force traces use actual simulated contacts, exclude floor and object, sum every left-arm body including unnamed wrist housings, and keep torso load separate. Cached contact/kinematic fields can lag logged qpos by one 2 ms step. MJPC reach errors are recomputed from saved qpos using the exact jaw offset (0.2254, −0.0118, −0.1062) m. Its early experimental jaw columns omitted y/z; the uniform qpos analysis fixes this for every run.</p>
<p>CMPC trials labeled with verified jaw mapping use the current assembled MJPC model, MuJoCo 3.2.3, the same full jaw offset, table-frame target, home reset (except the explicit shared stand_up check) and 0.003 xorshift qpos/qvel perturbation. Recorded nominal/high initial states were checked against the bench formula with zero numerical difference; <a href="../../studies/table_height/runs/codex_skunkworks/initial_state_parity.json">initial-state audit</a>. The old Crocoddyl study used MuJoCo 3.10, a different collision/joint-limit model, wrist offset (0.13,0,0), target (1.06,−0.2348,face+0.1132), stand reset, and Gaussian perturbations with an unperturbed seed 0. Its results are shown separately.</p>
<p>MJPC uses six sampling threads and a 1 s prediction horizon with 10 ms prediction steps. Its plant step is 2 ms, so spp=3 is 167 Hz and spp=10 is 50 Hz. Crocoddyl uses 35 nodes at 20 ms and one BoxFDDP iteration per 50 Hz period; the installed binary is single-threaded despite the requested six threads. MJPC holds its initial standing rung for 12 s and uses an 18 s brace-reference ramp; CMPC uses a 2.4 s approach followed by its explicit contact phase. Costs, approach timing, contact schedule and control delivery differ. These compare complete controllers under common physical conditions. Representative 167 Hz episodes consumed about 7–8 wall-clock seconds per simulated second under the stated caps. CMPC timings include fallen states. Neither optimizer-only causality nor real-time performance is established.</p>
<h2>MJPC runs</h2><p>All completed new runs, including failures. To reproduce a historical unsynchronized row with the current binary, restore its linked strategy snapshot and append <code>--sync_planning_model 0</code> to the saved command. The first four exploratory runs used the historical model copy. Later logs record synchronized geometry. One seed is a screening result, not repeatability; rates and target variants are never pooled. The seed changes the reset state, while sampling noise is independently randomized by Abseil BitGen. Three successful trials are limited evidence, not an estimated population reliability.</p>
<div class="tw"><table><tr><th>Run</th><th>Face m</th><th>Plan Hz</th><th>Sync</th><th>Loaded reach</th><th>Ladder</th><th>70 mm dwell s</th><th>30 mm dwell s</th><th>Best reach mm</th><th>Evidence</th></tr>'''+''.join(rows)+'''</table></div>
<figure><img src="media/skunkworks/mjpc_traces.png" alt="Measured reach error and actual arm load through representative MJPC runs"><figcaption>The hand leaves the target during the return sequence; the rise in error after successful reach is expected. Dashed lines mark the 70 mm and 20 N thresholds.</figcaption></figure>
<h2>Crocoddyl runs and contact modes</h2><figure><img src="media/skunkworks/static_modes.png" alt="Static true-jaw contact-mode screening matrix"><figcaption>The corrected six-cell screen selected candidate contact modes. A static admissibility label does not establish a stable approach or hold; the dynamic trials below test those claims.</figcaption></figure>
<div class="tw"><table><tr><th>Condition</th><th>Seed</th><th>Loaded hold</th><th>Tail error p95 mm</th><th>Tail arm load p05 N</th><th>Min pelvis m</th><th>Longest joint dwell s</th><th>Evidence</th></tr>'''+''.join(ct)+'''</table></div>
<figure><img src="media/skunkworks/height_counts.png" alt="Prior and new Crocoddyl hold counts, separated by physical model and endpoint"><figcaption>Left: independently audited prior trajectories at x=1.06, excluding the separate x=0.9047 failure. Right: new common-model home-start x=0.85 trials only; the separate x=1.15 test is excluded; circles are elbow–forearm, squares elbow–wrist, triangles forearm. Final-frame-only summaries are not used.</figcaption></figure>
<p>The prior high-table brace runs have final-five-second reach-error p95 values of 2.9–4.7 mm and left-arm load p05 values of 111.8–116.9 N. At 0.785 m only one of three brace runs survives. The surviving prior high-table “stand” trial also takes incidental left-arm support (27.9 N load p05), so it is not a clean unbraced control.</p><p>An initial calibration screen tested a wrist-frame point over six height/target cells and the existing eight-mode ladder. At x=1.0, elbow–forearm was assigned only 0.2 N at nominal height and 1.4 N at high height; elbow–wrist was assigned 48–57 N across low/nominal/high. Mode names alone do not establish actual load. That screen used a different physical endpoint from the MJPC jaw and must not be used to bound its workspace. A corrected jaw-frame screen is linked below; all such pinned-foot screens are candidate generators, not dynamic workspace bounds.</p>
<p>The current model rotates the right gripper by 90° about the wrist x-axis. The first four CMPC calibration trials applied the gripper-local offset directly in the wrist frame. They remain in the table, with their own measured endpoint, and are excluded from matched jaw comparisons. The opt-in bridge fix composes the welded gripper transform before creating the Pinocchio frame. At the reference pose, independent parity checks then agree to approximately 1 micrometre at the jaw, 1.5×10⁻⁵ N m in gravity and 8.3×10⁻⁷ kg m² in the actuated mass matrix.</p><h2>Sampled reachable workspace</h2>
<figure><img src="media/skunkworks/workspace.png" alt="Measured dynamic success counts in the sampled height and reach-depth section"><figcaption>Targets use y=−0.04 m and z=face+0.15 m unless a run states otherwise. This is a sampled vertical/depth section; no hemispherical shape or unsampled boundary is inferred. Panels separate rates and reference choices. The nominal far-target baseline has identical planning geometry and is included. Far-target gates use two-second sustain; short-target gates use five seconds, with the same independent two-second loaded-reach score. No boundary is interpolated.</figcaption></figure>
<h2>Replays</h2><p>These render saved simulated states. They do not rerun physics; contact overlays come from the recorded episode or the stated full-state prior audit.</p><div class="replay-grid">
'''
captions={'pose1_short_h0785_s2':'Low table, DLS, 167 Hz, seed 2: remains braced but misses the hand target.', 'pose1_short_h1085_s0':'High table, DLS, 167 Hz, seed 0: loaded reach and completed recovery.', 'pose1_short_h0785_s0':'Low table, DLS, 167 Hz, seed 0: loaded reach; recovery stalls.', 'low_success':'Low table, authored keys, 50 Hz, seed 0: loaded reach; recovery incomplete at 75 s.', 'croco_failure':'Verified-jaw CMPC, nominal x=0.85, home: brief contact followed by a fall, no successful hold.', 'croco_calibration_failure':'Endpoint calibration failure; wrist-frame point, excluded from matched jaw comparisons.', 'prior_croco_high':'Prior CMPC high-table hold: original model, MuJoCo 3.10, wrist point; independently audited.'}
for name in ['pose1_short_h0785_s0','pose1_short_h0785_s2','short_h0985_s0','pose1_short_h1085_s0','prior_croco_high','croco_failure','short_h1085_s0','sync_h1085_s0','low_success','croco_calibration_failure']:
 if (MEDIA/(name+'.mp4')).exists():page+=f'<figure><video controls preload="metadata" poster="media/skunkworks/{name}.png" src="media/skunkworks/{name}.mp4"></video><figcaption>{esc(captions.get(name,name))} Orange sphere: commanded reaching point. The overlay uses measured trajectory metrics.</figcaption></figure>'
page+='''</div><h2>Implementation</h2><p>Local source commits: MJPC <code>4398d8a8</code>; Crocoddyl hooks <code>51d604e</code> and <code>b1040fd</code>.</p><p><a href="../../studies/table_height/skunkworks/mjpc_implementation.patch">MJPC patch</a>: fixed-height planner-model synchronization, measured jaw/normal-load logging, default-off base-height gain, and selectable validated starting keyframe. <a href="../../studies/table_height/skunkworks/crocoddyl_comparison.patch">Crocoddyl patch</a>: opt-in endpoint/welded-frame mapping, table-leg matching, and explicit episode initial states. Existing controller defaults are preserved outside the corrected bench synchronization.</p><p>The local build passed, Python study scripts passed syntax checks, and the simulations exercise the modified paths. Recorded initial-state parity and <a href="../../studies/table_height/runs/codex_skunkworks/model_rebuild_check.json">byte-identical assembled model after rebuilding</a> support the comparison. This demonstrates finite-duration simulated behavior; unsampled targets, disturbances, repeated task cycles and hardware remain outside the measured result.</p><h2>Reproduction and handoff</h2>
<p>Everything is local in <code>/home/humanoid/Programs/mujoco_mpc_codex_tableheight_20260908</code>, with the Crocoddyl hooks in <code>/home/humanoid/Programs/crocoddyl_mpc_codex_tableheight_20260908</code>. The original checkouts were not reset or switched. Existing untracked video.mp4 and Crocoddyl bvs_plots.py edits were preserved.</p>
<ul><li><a href="../../studies/table_height/skunkworks/README.md">Commands and environment setup</a> · <a href="../../studies/table_height/skunkworks/protocol.md">Protocol and adaptive decisions</a></li>
<li><a href="../../studies/table_height/skunkworks/study_manifest.json">Frozen run selection and raw-data checksums</a> · <a href="media/skunkworks/summary.json">Report data</a> · <a href="../../studies/table_height/runs/codex_skunkworks/model_provenance.json">Model provenance</a> · <a href="../../studies/table_height/runs/codex_skunkworks/common_nominal.mjb">Exact nominal MuJoCo 3.2.3 model (161 MiB)</a></li>
<li><a href="../../studies/table_height/runs/codex_skunkworks/common_static_jaw/cells.json">Corrected jaw static mode screen</a> · <a href="../../studies/table_height/runs/codex_skunkworks/prior_croco_audit/audit.json">Prior trajectory audit</a></li>
<li><a href="20260908-codex_table_height_handoff.md">Dated coordination handoff</a> · <a href="20260908-table_height_full_report.html">Baseline report</a> (superseded where this report identifies the model-copy defect).</li></ul>
<p class="note">Runs are serial. MJPC is capped at 700% CPU and 11 GiB; CMPC at 600% CPU and 8 GiB. The first CMPC run initially hit a 4 GiB cap (paging, no OOM); its cap was raised to 8 GiB during the run. No simulated state was reset by that resource change. Raw trajectories, command snapshots and failed attempts remain on disk. No result was uploaded or pushed.</p></main></html>'''
(ROOT/'docs/lean/20260908-table_height_skunkworks.html').write_text(page)
print('Report generated:',len(M),'MJPC,',len(C),'new CMPC episodes')
