#!/usr/bin/env python3
"""Combined log-weight SD versus 300K OFF→POLAR SD on fixed source frames."""
import argparse,csv,json
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1];P=R/'docs/results/reweight_boundary_scan_20261028';kb=8.617333262145e-5
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--csv-output',type=Path,required=True);p.add_argument('--critical',type=Path,required=True);a=p.parse_args()
with (P/'frame_fields.csv').open() as f:r=list(csv.DictReader(f))
with (P/'original_fields/frame_field_components.csv').open() as f:fields=list(csv.DictReader(f))
assert len(r)==len(fields)==501 and all(float(x['time_ps'])==float(y['time_ps']) for x,y in zip(r,fields))
mask=np.array([float(x['time_ps'])>=20 for x in r]);u=np.array([float(x['OFF_eV']) for x in r])[mask];h=np.array([float(x['POLAR_eV']) for x in r])[mask];delta=h-u;u-=u.mean();delta-=delta.mean();n=len(u)
keys=[k for k in fields[0] if k.startswith('site_unit_seed_')];site=np.array([[float(x[k]) for k in keys] for x in fields])[mask];site-=site.mean(axis=0);dip=np.array([float(x['dipole_proxy_corrected_eA']) for x in fields])[mask];dip-=dip.mean();assert len(keys)==10
b0=1/(kb*300);ref=float(np.std(b0*delta));temps=[270,280,290,295,300,305,310,320,330,340,350,360,380,400]
# log q = (beta0-betaT)*U_OFF - betaT*(deltaU+U_ext) + constant.
def ratio(T,field):
 b=1/(kb*T);v=(b0-b)*u-b*delta
 if field.ndim==2:return np.std(v[:,None]-b*field,axis=0)/ref
 return np.array([np.std(v-b*field)/ref])
sitevals=[0,.01,.02,.05,.1,.2,.3,.4,.5,.6,0.7326225830533086,.8,1.];dipvals=[.001,.005,.01,.02,.05,.1,.15,.2,.25,0.28166259331143734,.3,.4,.5]
rows=[];table=['| 扰动方式 | 参数 | '+ ' | '.join(f'{T} K' for T in temps)+' |','|---|---:|'+'---:|'*len(temps)]
for typ,vals in [('site',sitevals),('uniform',dipvals)]:
 for a0 in vals:
  field=site*a0 if typ=='site' else dip*a0;cells=[]
  for T in temps:
   rr=ratio(T,field);mean=float(rr.mean());cells.append((f'**{mean*ref:.2f} / {mean:.3f}**' if mean>=2-1e-10 else f'{mean*ref:.2f} / {mean:.3f}'));rows.append({'field_type':typ,'amplitude':a0,'temperature_K':T,'ratio_total_to_300K_off_polar':mean,'logq_sd':mean*ref,'seed_min_ratio':float(rr.min()),'seed_max_ratio':float(rr.max()),'frames':n})
  name='无外场' if typ=='site' and a0==0 else ('随机原子场 s' if typ=='site' else '均匀电场 V/Å')
  formatted=f'{a0:.4f}' if (abs(a0-0.7326225830533086)<1e-12 or abs(a0-0.28166259331143734)<1e-12) else f'{a0:g}'
  table.append('| '+name+' | '+formatted+' | '+' | '.join(cells)+' |')
# Solve sigma(total)=2*sigma(reference), preserving covariance.
def crossings(x,y,target):
 # Solve ||x + s*y||²/n = target² in s.
 A=float(np.mean(y*y));B=float(2*np.mean(x*y));C=float(np.mean(x*x)-target**2);D=B*B-4*A*C
 if A==0 or D<0:return []
 return sorted([z for z in ((-B-np.sqrt(D))/(2*A),(-B+np.sqrt(D))/(2*A)) if z>=0])
# Site 10 seeds and the mean of seed SDs (not the SD of mean field).
base=-b0*delta;site_roots=[crossings(base,-b0*site[:,j],2*ref) for j in range(10)];uniform_roots=crossings(base,-b0*dip,2*ref)
lo,hi=0.,1.
while ratio(300,site*hi).mean()<2 and hi<100:hi*=2
for _ in range(100):
 mid=(lo+hi)/2
 if ratio(300,site*mid).mean()<2:lo=mid
 else:hi=mid
mean_boundary=(lo+hi)/2
# For no field use x=300/T: log/beta0 = u - x*(u+delta).
roots_x=crossings(b0*u,-b0*(u+delta),2*ref);troots=sorted(300/x for x in roots_x if x>0)
critical={'source_temperature_K':300,'cutoff_ps':20,'frame_count':n,'reference_sd_logq':ref,'threshold_definition':'sd(log q_combined) / sd(log q_OFF_to_POLAR_at_300K) = 2; source trajectory is fixed at 300K',
 'site_seed_roots':{str(i):x for i,x in enumerate(site_roots)},'site_mean_sd_first_crossing':mean_boundary,
 'uniform_positive_field_roots_V_A':uniform_roots,'no_field_target_temperature_roots_K':troots,
 'explanation':'Threshold means combined standard deviation = two times baseline; individual fluctuation terms are not added as scalar magnitudes; correlated terms determine total. At changed target T the model term is scaled by betaT, not a fixed 1.'}
with a.output.open('x') as f:f.write('\n'.join(table)+'\n')
with a.csv_output.open('x',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
with a.critical.open('x') as f:json.dump(critical,f,indent=2)
print(json.dumps(critical,indent=2));print('grid rows',len(rows))
