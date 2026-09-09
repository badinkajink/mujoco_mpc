"""Signed pitch exposes the direction hidden by torso-tilt magnitude."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[3]
RUNS=ROOT/'studies/table_height/runs/robustness_20260909'
MEDIA=ROOT/'docs/lean/media/robustness_20260909'
fig,axes=plt.subplots(3,1,figsize=(10,8),sharex=True)
for seed,color in [(10,'#087f8c'),(11,'#d1495b'),(12,'#9b5de5')]:
 p=RUNS/f'ablate_reference_h1085_s{seed}'
 x=np.genfromtxt(p/'state.csv',delimiter=',',names=True)
 m=np.genfromtxt(p/'metrics.csv',delimiter=',',names=True)
 use=(x['t']>=10)&(x['t']<=20)
 pitch=np.degrees(np.arcsin(np.clip(2*(x['q3']*x['q5']-x['q6']*x['q4']),-1,1)))
 axes[0].plot(x['t'][use],pitch[use],color=color,label=f'Seed {seed}'+(' (survives)' if seed==10 else ' (falls)'))
 axes[1].plot(x['t'][use],x['q0'][use],color=color)
 use=(m['t']>=10)&(m['t']<=20)
 axes[2].plot(m['t'][use],m['brace_up_N'][use],color=color)
for ax,label in zip(axes,['Pelvis pitch (degrees)\nnegative = backward','Pelvis x (m)','Upward left-arm support (N)']):
 ax.axvline(12,color='black',ls=':',lw=1);ax.grid(alpha=.2);ax.set_ylabel(label)
axes[0].axhline(0,color='gray',lw=.8);axes[0].legend()
axes[-1].set_xlabel('Simulated time (s); dotted line = brace phase begins')
fig.suptitle('Tall-table reference: approach instability precedes the reach\nAll three reference trials, no post-selection of the successful seed')
fig.tight_layout();fig.savefig(MEDIA/'high_approach_trace.png',dpi=160)
