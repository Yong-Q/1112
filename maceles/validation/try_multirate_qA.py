#!/usr/bin/env python3
"""Explore unequal q and q*A sampling on already-scored frames.

This is a deterministic thinning sensitivity diagnostic, not a direct target
reference and not evidence of increased precision. No existing file is replaced.
"""
import argparse,csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

p=argparse.ArgumentParser(description=__doc__);p.add_argument('--observables',type=Path,required=True);p.add_argument('--outdir',type=Path,required=True);p.add_argument('--stride',type=int,default=5);a=p.parse_args()
x=np.load(a.observables);counts=x['counts'];gap=x['gap_eV'];z=x['z_A'];vol=float(x['bin_volume_A3']);N=len(gap);assert counts.shape==(501,40) and 0<a.stride<=N
beta=1/(8.617333262145e-5*300);logw=-beta*(gap-gap.min());w=np.exp(logw);w/=w.sum();max_index=int(np.argmax(w));ref=(w@counts)/vol
assert np.isclose(ref.sum()*vol,384)
rows=[];profiles=[]
for offset in range(a.stride):
 ii=np.arange(offset,N,a.stride);mass=float(w[ii].sum());assert mass>0
 # means rather than sums: (1/m sum qA)/(1/N sum q)
 multi=(N/len(ii))*(w[ii]@counts[ii])/vol
 same=(w[ii]@counts[ii])/(vol*mass)
 assert np.isclose(multi.sum()*vol,384*(N/len(ii))*mass)
 assert np.isclose(same.sum()*vol,384)
 profiles.append(multi)
 rows.append({'offset':offset,'qA_frames':len(ii),'weight_mass_in_qA_subset':mass,
  'max_weight_frame_included':bool(max_index in ii),'estimated_particle_count':float(multi.sum()*vol),
  'RMS_vs_full_density_per_A3':float(np.sqrt(np.mean((multi-ref)**2))),
  'same_subset_self_normalized_count':float(same.sum()*vol)})
out=a.outdir;out.mkdir(parents=True,exist_ok=False)
with (out/'multirate_offsets.json').open('x') as f:json.dump({'source_frames':N,'stride':a.stride,'denominator_q_frames':N,'results':rows,'scope':'Deterministic thinning of one trajectory. Full-data direct curve is not an independent target reference. q and A correlations, one dominant target frame, and limited support remain; lower m is not automatically faster because target energy dominates cost.'},f,indent=2)
with (out/'multirate_profiles.csv').open('x',newline='') as f:
 wr=csv.writer(f,lineterminator='\n');wr.writerow(['z_A','full_direct']+[f'offset_{i}' for i in range(a.stride)])
 for j in range(len(z)):wr.writerow([z[j],ref[j]]+[v[j] for v in profiles])
font_manager.fontManager.addfont('/home/qiuyong/fonts/arial.ttf');plt.rcParams.update({'font.family':'Arial','font.size':8,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
fig,ax=plt.subplots(figsize=(7.1,3.2),layout='constrained');ax.plot(z,ref,color='black',lw=1.8,label='All 501 frames: same-frame')
colors=['#0072B2','#D55E00','#009E73','#CC79A7','#6F579D']
for i,r in enumerate(rows):ax.plot(z,profiles[i],color=colors[i%len(colors)],lw=1,label=f"Every 5th, offset {i}: N≈{r['estimated_particle_count']:.0f}")
ax.set(xlabel='z (Å)',ylabel='COM density estimate (1/Å³)',title='Unequal sampling: each qA subset vs all-frame denominator')
ax.legend(fontsize=6.5,ncol=2);fig.savefig(out/'multirate_qA.pdf');fig.savefig(out/'multirate_qA.png',dpi=300);plt.close(fig)
print(json.dumps({'results':rows},indent=2))
