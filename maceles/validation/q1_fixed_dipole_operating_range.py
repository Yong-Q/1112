#!/usr/bin/env python3
"""Diagnostic 2x contour from existing 300 K OFF→POLAR frames."""
import argparse,csv,json
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1];P=R/'docs/results/reweight_boundary_scan_20261028';KB=8.617333262145e-5
p=argparse.ArgumentParser();p.add_argument('--outdir',type=Path,required=True);a=p.parse_args()
with (P/'frame_fields.csv').open() as f:r=list(csv.DictReader(f))
with (P/'original_fields/frame_field_components.csv').open() as f:ff=list(csv.DictReader(f))
assert len(r)==len(ff)==501 and all(float(x['time_ps'])==float(y['time_ps']) for x,y in zip(r,ff))
m=np.array([float(x['time_ps'])>=20 for x in r]);u=np.array([float(x['OFF_eV']) for x in r])[m];h=np.array([float(x['POLAR_eV']) for x in r])[m];D=h-u;D-=D.mean();u-=u.mean();F=np.array([float(x['dipole_proxy_corrected_eA']) for x in ff])[m];F-=F.mean();b0=1/(KB*300);base=float(np.std(b0*D))
def roots(x,y,target):
 # Real nonnegative s for population SD(x+s*y)=target, not scalar SD addition.
 x=x-x.mean();y=y-y.mean();A=np.mean(y*y);B=2*np.mean(x*y);C=np.mean(x*x)-target**2
 if A<=0:return []
 z=np.roots([A,B,C]);return sorted(float(q.real) for q in z if abs(q.imag)<1e-8 and q.real>0)
def row(T):
 b=1/(KB*T);q=(b0-b)*u-b*D;vec=-b*F
 return {'temperature_K':T,'ratio_no_field':float(np.std(q)/base),'ratio_E_0p25_V_per_A':float(np.std(q+.25*vec)/base),'positive_2x_roots_V_per_A':roots(q,vec,2*base)}
Ts=[200,225,250,260,270,275,280,290,295,300,305,310,320,330,340,350,360,375,380,400]
rows=[row(T) for T in Ts]
# Let x=300/T. The whole target model + field scales as x.
q0=b0*u;for_025=-b0*(u+D+.25*F);xs=roots(q0,for_025,2*base);Tr=sorted(float(300/x) for x in xs)
no_field=roots(q0,-b0*(u+D),2*base);T0=sorted(float(300/x) for x in no_field)
result={'source_temperature_K':300,'frame_count':int(m.sum()),'cutoff_ps':20,'baseline_sd_logq':base,'definition':'2x is SD of combined log weight / 300K no-field SD. One 300K source trajectory; T is target reweighting; fixed dipole 1.85D, E positive along z, not self-consistent target polarization. Not physical accuracy bounds.','field_E_0p25_twofold_temperature_crossings_K':Tr,'zero_field_twofold_temperature_crossings_K':T0,'rows':rows}
a.outdir.mkdir(parents=True,exist_ok=False)
with (a.outdir/'range.json').open('x') as f:json.dump(result,f,indent=2)
with (a.outdir/'range.csv').open('x',newline='') as f:
 w=csv.writer(f,lineterminator='\n');w.writerow(['temperature_K','ratio_no_field','ratio_E_0p25_V_per_A','positive_2x_roots_V_per_A'])
 for x in rows:w.writerow([x['temperature_K'],x['ratio_no_field'],x['ratio_E_0p25_V_per_A'],';'.join(map(str,x['positive_2x_roots_V_per_A']))])
assert len(Tr)==2 and all(abs(np.std((b0-1/(KB*T))*u-(1/(KB*T))*(D+.25*F))/base-2)<1e-9 for T in Tr)
print(json.dumps(result,indent=2))
