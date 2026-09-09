from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[3];RUNS=ROOT/'studies/table_height/runs/robustness_20260909';MEDIA=ROOT/'docs/lean/media/robustness_20260909';MEDIA.mkdir(parents=True,exist_ok=True)
fig,axs=plt.subplots(3,1,figsize=(10,8),sharex=True)
for arm,color,label in [('reference','#146c94','Reference: DLS retargeting'),('no_pose','#ba3b46','One change: retargeting off')]:
 p=RUNS/f'ablate_{arm}_h0785_s10';x=np.genfromtxt(p/'metrics.csv',names=True,delimiter=',');r=json.loads((p/'evaluation.json').read_text());keep=x['t']>=12
 for ax,k,mult in zip(axs,['reach_error','brace_up_N','tilt_deg'],[1000,1,1]):ax.plot(x['t'][keep],x[k][keep]*mult,color=color,label=label,lw=1.2)
 if r['interval30']:
  for ax in axs:ax.axvspan(*r['interval30'],color=color,alpha=.12)
axs[0].axhline(30,color='black',ls=':',lw=1);axs[1].axhline(20,color='black',ls=':',lw=1)
for ax,label in zip(axs,['Jaw error (mm)','Upward brace support (N)','Torso tilt (degrees)']):ax.set_ylabel(label);ax.grid(alpha=.2)
axs[0].legend(loc='upper right');axs[-1].set_xlabel('Simulated time (s)');fig.suptitle('Low table, initial-state seed 10 — illustrative trial pair\nSame conditions except retargeting; sampling noise is independently randomized',fontsize=12);fig.tight_layout();fig.savefig(MEDIA/'low_retarget_trace.png',dpi=170);plt.close(fig)
