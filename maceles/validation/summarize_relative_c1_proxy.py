#!/usr/bin/env python3
"""Density log-ratios for shape diagnostics; not certified c1 labels."""
import argparse,csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--profiles',type=Path,required=True);p.add_argument('--outdir',type=Path,required=True);a=p.parse_args()
d=np.genfromtxt(a.profiles,delimiter=',',names=True);names=['source','direct','smoothed','mean_gap','gaussian_gap'];z=d['z_A'];ref=int(np.argmax(d['source']));curves={}
for name in names:
 r=d[name];v=np.full(len(z),np.nan);ok=(r>0)&(r[ref]>0);v[ok]=np.log(r[ok]/r[ref]);curves[name]=v
mask=(d['source']>=.005)&(d['direct']>=.005)&(d['smoothed']>=.005)
assert np.count_nonzero(mask)>0 and mask[ref]
a.outdir.mkdir(parents=True,exist_ok=False)
with (a.outdir/'relative_c1_proxy.csv').open('x',newline='') as f:
 wr=csv.writer(f,lineterminator='\n');wr.writerow(['z_A','comparison_mask']+[k+'_logrho_relative' for k in names]+['direct_minus_source','smoothed_minus_source'])
 for i in range(len(z)):
  v=[z[i],int(mask[i])]+[curves[k][i] for k in names]+[curves['direct'][i]-curves['source'][i],curves['smoothed'][i]-curves['source'][i]]
  wr.writerow([float(x) if np.isfinite(x) else '' for x in v])
rms=lambda a,b:float(np.sqrt(np.mean((a[mask]-b[mask])**2)))
summary={'reference_z_A':float(z[ref]),'common_source_direct_smoothed_density_min_A3':.005,'comparison_bins':int(mask.sum()),
 'direct_vs_source_relative_logrho_RMS':rms(curves['direct'],curves['source']),
 'smoothed_vs_source_relative_logrho_RMS':rms(curves['smoothed'],curves['source']),
 'smoothed_vs_direct_relative_logrho_RMS':rms(curves['smoothed'],curves['direct']),
 'scope':'Conditional EL interpretation only: matched position-only external field, equilibrium/ensemble and reference conventions assumed; no absolute chemical potential, no direct POLAR MD reference, and no uncertainty/accuracy claim.'}
with (a.outdir/'relative_c1_proxy_summary.json').open('x') as f:json.dump(summary,f,indent=2)
font_manager.fontManager.addfont('/home/qiuyong/fonts/arial.ttf');plt.rcParams.update({'font.family':'Arial','font.size':8,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
fig,ax=plt.subplots(2,1,figsize=(7.1,5.2),sharex=True,layout='constrained')
for name,color,label in [('source','#555555','OFF23 (source)'),('direct','#0072B2','Direct reweighting'),('smoothed','#59A89C','Direct + smoothing')]:
 y=np.where(mask,curves[name],np.nan);ax[0].plot(z,y,color=color,lw=1.3,label=label)
ax[0].legend(fontsize=7,ncol=2);ax[0].set_ylabel('ln[ρ(z)/ρ(z_ref)]')
for name,color,label in [('direct','#0072B2','Direct − source'),('smoothed','#59A89C','Smoothed − source')]:
 y=np.where(mask,curves[name]-curves['source'],np.nan);ax[1].plot(z,y,color=color,lw=1.2,label=label)
ax[1].axhline(0,color='0.7',lw=.6);ax[1].legend(fontsize=7,loc='lower right');ax[1].set_ylabel('Relative log-density\ndifference');ax[1].set_xlabel('z (Å)')
for i,aa in enumerate(ax):aa.text(.01,.97,chr(97+i),transform=aa.transAxes,weight='bold',va='top');aa.tick_params(direction='out',length=3)
fig.suptitle('Relative log-density: c1-shape proxy under EL assumptions',fontsize=9)
fig.savefig(a.outdir/'relative_c1_proxy.pdf');fig.savefig(a.outdir/'relative_c1_proxy.png',dpi=300);plt.close(fig)
print(json.dumps(summary,indent=2))
