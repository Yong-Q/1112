#!/usr/bin/env python3
"""Fixed 300 K trajectory importance-weight sensitivity; not new equilibrium MD."""
import argparse,csv,hashlib,json,math
from pathlib import Path
import numpy as np
from diagnose_com_off_polar import frames
ROOT=Path(__file__).resolve().parents[1];SRC=ROOT/'validation/off23_500ps_water1152_2026-10-23';KB=8.617333262145e-5
p=argparse.ArgumentParser();p.add_argument('--outdir',type=Path,required=True);a=p.parse_args()
ep=SRC/'frames_501_total_energies.csv';traj=SRC/'traj_500ps_1fs.lammpstrj'
r=list(csv.DictReader(ep.open()));t=np.array([float(x['time_ps']) for x in r]);ul=np.array([float(x['E_OFF_total_eV']) for x in r]);uh=np.array([float(x['E_POLAR_total_eV']) for x in r]);du=uh-ul
F=[]; lows=[]; high=[]
for j,(step,b,group) in enumerate(frames(traj)):
 assert step==int(r[j]['step']);box=b[:,1]-b[:,0];o=(group[:,0,:]-b[:,0])%box;delta=group[:,1:,:]-group[:,0,None,:];delta-=np.round(delta/box)*box
 com=(o+(1.008/(15.999+2*1.008))*delta.sum(axis=1))%box
 F.append(np.cos(2*np.pi*com[:,2]/box[2]).sum());lows.append(np.mean((com[:,2]<box[2]*.2)|(com[:,2]>box[2]*.8)));high.append(np.mean((com[:,2]>.4*box[2])&(com[:,2]<.6*box[2])))
F=np.array(F);lows=np.array(lows);high=np.array(high);assert len(F)==len(du)==501

def wstats(log):
 v=log-np.max(log);w=np.exp(v);w/=w.sum();return 1/np.dot(w,w),float(w.max()),int(np.argmax(w)),float(np.mean((w*len(w)-1)**2))
def stats(x):return {'mean':float(np.mean(x)),'sd':float(np.std(x)),'min':float(np.min(x)),'max':float(np.max(x))}
cutoffs=[0,5,10,20,30,50,100,200];A=np.unique(np.r_[np.linspace(-.5,.5,101),0]);Ts=np.unique(np.r_[np.arange(250,401,5),np.arange(290,311,1),300]);rows=[];cut=[];blocks=[]
for lo in range(0,500,50):
 mask=(t>=lo)&(t<lo+50)
 blocks.append({'start_ps':lo,'end_exclusive_ps':lo+50,'n':int(mask.sum()),'OFF_centered_mean_eV':float(np.mean(ul[mask]-ul.mean())),'gap_mean_eV':float(np.mean(du[mask])),'gap_sd_eV':float(np.std(du[mask])),'gas_fraction':float(np.mean(lows[mask])),'center_fraction':float(np.mean(high[mask])),'field_F_mean':float(np.mean(F[mask])),'field_F_sd':float(np.std(F[mask]))})
for c in cutoffs:
 m=t>=c;Tc=t[m];u=ul[m];h=uh[m];g=du[m];f=F[m];n=m.sum();u0=u-u.mean();h0=h-h.mean();g0=g-g.mean();f0=f-f.mean();beta0=1/(KB*300)
 ess,maxw,arg,cv=wstats(-beta0*g0)
 # 300K comparisons: target-only sinusoidal field amplitude eV/molecule.
 sdg=float(np.std(g0));sdf=float(np.std(f0));field_threshold=sdg/sdf
 # Shift in beta acting on centred source energy vs baseline beta gap, solve T>300 where exists.
 ratio=lambda T:abs(1/(KB*T)-beta0)*float(np.std(u0))/(beta0*sdg)
 if ratio(1e6)>1:
  lo,hi=300.,1e6
  for _ in range(100):
   mid=(lo+hi)/2
   if ratio(mid)>=1:hi=mid
   else:lo=mid
  tup=(lo+hi)/2
 else:tup=None
 lo,hi=.01,300.
 for _ in range(100):
  mid=(lo+hi)/2
  if ratio(mid)>=1:lo=mid
  else:hi=mid
 tdown=(lo+hi)/2
 cut.append({'cutoff_inclusive_ps':c,'n':int(n),'gap_sd_eV':sdg,'gap_peak_to_peak_eV':float(np.ptp(g)),'source_centered_sd_eV':float(np.std(u0)),'target_centered_sd_eV':float(np.std(h0)),'field_shape_sd':sdf,'field_amplitude_equal_gap_sd_eV_per_molecule':field_threshold,'field_amplitude_equal_gap_sd_kBT300':field_threshold*beta0,'temperature_lower_equal_logweight_sd_K':tdown,'temperature_upper_equal_logweight_sd_K':tup,'baseline_ESS':ess,'baseline_maxw':maxw,'baseline_max_time_ps':float(Tc[arg]),'baseline_CV2':cv})
 for temp in Ts:
  b=1/(KB*temp)
  # Ratio target(T) / source(T0): beta0 U_L - betaH U_H; constants cancel.
  for amp in A:
   log=beta0*u0-b*(h0+amp*f0)
   ee,mw,idx,cc=wstats(log)
   ratiofield=abs(amp)*sdf/sdg
   rtemp=abs(beta0-b)*float(np.std(u0))/(b*sdg)
   rows.append([c,n,float(temp),float(temp-300),float(amp),float(amp*beta0),ee,ee/n,cc,mw,float(Tc[idx]),ratiofield,rtemp,float(np.std(b*g0)),float(np.std(b*amp*f0)),float(np.std((beta0-b)*u0)),float(np.corrcoef(g0,amp*f0)[0,1]) if abs(amp)>0 else 0.])
a.outdir.mkdir(parents=True,exist_ok=False)
cols=['cutoff_ps','frames','target_T_K','delta_T_K','vext_amplitude_eV_per_molecule','vext_amplitude_kBT300','kish_ESS','ESS_fraction','CV2','max_weight','max_frame_ps','ratio_field_to_gap_energy_sd','ratio_temperature_to_gap_logweight_sd','gap_logweight_sd','field_logweight_sd','temperature_logweight_sd','corr_gap_field']
with (a.outdir/'full_grid.csv').open('x',newline='') as f:
 w=csv.writer(f,lineterminator='\n');w.writerow(cols);w.writerows(rows)
with (a.outdir/'blocks_50ps.csv').open('x',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(blocks[0]),lineterminator='\n');w.writeheader();w.writerows(blocks)
with (a.outdir/'cutoffs.csv').open('x',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(cut[0]),lineterminator='\n');w.writeheader();w.writerows(cut)
with (a.outdir/'frame_fields.csv').open('x',newline='') as f:
 w=csv.writer(f,lineterminator='\n');w.writerow(['time_ps','OFF_eV','POLAR_eV','gap_eV','field_shape_F','low_z_fraction','center_z_fraction']);w.writerows(zip(t,ul,uh,du,F,lows,high))
with (a.outdir/'summary.json').open('x') as f:json.dump({'source_T_K':300,'field_definition':'target-only v_ext(z)=A*cos(2*pi*z/Lz), A eV per molecule, total A*sum_m cos; source no added field','temperature_definition':'configurational ratio target(T)/source(300K): logq=beta300*U_OFF-betaT*(U_POLAR+A*F), each energy centered by constant; masses and NVT volume same, momenta integrated out','grids':{'cutoffs_ps':cutoffs,'temperature_K':[float(x) for x in Ts],'amplitude_eV_per_molecule':[float(x) for x in A]},'threshold_meaning':'equality of sample population SD of centered terms; not a significance or physical phase boundary; field-temperature covariance can reinforce/cancel; no arbitrary energy offset determines weight','cutoffs':cut,'block_summary':blocks,'source_hashes':{str(x.name):hashlib.sha256(x.read_bytes()).hexdigest() for x in [ep,traj]},'limitations':'All values are sample diagnostics on the existing trajectory generated at 300 K (20ps pre-equil plus production). No new field/T MD; ESS not correctness or time-correlation corrected; NVT center of mass identities audited previously; source MD vs recompute differences ~0.03eV remain unresolved. Sample tails may be missed.'},f,indent=2)
print(json.dumps({'cutoffs':cut,'blocks':blocks,'grid_rows':len(rows)},indent=2))
