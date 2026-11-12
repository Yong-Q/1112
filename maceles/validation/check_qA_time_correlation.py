#!/usr/bin/env python3
"""Exploratory correlations of q, COM counts, and q*counts (not ESS corrections)."""
import argparse,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--observables',type=Path,required=True);p.add_argument('--outdir',type=Path,required=True);a=p.parse_args()
d=np.load(a.observables);count=d['counts'];gap=d['gap_eV'];z=d['z_A'];assert len(gap)==501
beta=1/(8.617333262145e-5*300);l=-beta*(gap-gap.min());q=np.exp(l);q/=q.mean()
def moments(x):
 x=np.asarray(x);x=x-x.mean();base=float(x@x/len(x))
 if base==0:return None
 # Lag-pair initial-positive window; descriptive only for 501 heavy-tailed frames.
 r=np.array([float(x[:-t]@x[t:]/(len(x)-t)/base) for t in range(1,101)])
 used=[]
 for i in range(0,len(r)-1,2):
  if r[i]+r[i+1]<=0:break
  used.extend(r[i:i+2])
 return {'lag_ps':{str(t):float(r[t-1]) for t in [1,2,5,10,20,50]},'integrated_autocorrelation_time_ps_approx':float(1+2*sum(used)), 'window_end_lag_ps':len(used)}
out={'frame_spacing_ps':1,'frame_count':501,'q_scaled_mean1':moments(q),'positions':{},
 'scope':'Sample lag correlations and simple initial-positive truncation are descriptive only. q is heavy-tailed and dominated by a few frames, stationarity and missing target support remain unchecked. Tau from q alone cannot correct observable uncertainty.'}
for zz in [5.7,12.72,18.86,19.74]:
 k=int(np.argmin(abs(z-zz)));A=count[:,k]
 out['positions'][f'{z[k]:.4f}A']={'mean_count':float(A.mean()),'A':moments(A),'q_times_A':moments(q*A)}
a.outdir.mkdir(parents=True,exist_ok=False)
with (a.outdir/'time_correlation.json').open('x') as f:json.dump(out,f,indent=2)
print(json.dumps(out,indent=2))
