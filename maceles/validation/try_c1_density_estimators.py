#!/usr/bin/env python3
"""Exploratory OFF23→POLAR density estimators from stored water configurations.

Conditional cumulants are model assumptions, not exact reweighting or validated
c1. Requires existing trajectory/energy files. Output path must not exist.
"""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter1d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from diagnose_com_off_polar import frames

ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'validation/off23_500ps_water1152_2026-10-23'
BETA=1/(8.617333262145e-5*300)
METHODS=('source','direct','smoothed','mean_gap','gaussian_gap')

def hsh(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()

def estimate(counts,gaps,vol,sigma_bins):
 """Means of per-frame occupancy; moments use occupancy-weighted source frames."""
 n_frames,nb=counts.shape;n_particles=int(counts[0].sum())
 assert n_frames>0 and np.all(counts.sum(axis=1)==n_particles)
 gap=gaps-np.mean(gaps); lw=-BETA*gap; lw-=np.max(lw)
 q=np.exp(lw);w=q/q.sum(); source=counts.mean(axis=0)/vol
 direct=(w@counts)/vol
 smooth=gaussian_filter1d(direct,sigma=sigma_bins,mode='wrap')
 assert np.allclose([source.sum()*vol,direct.sum()*vol,smooth.sum()*vol],n_particles)
 occ=counts.sum(axis=0);valid=occ>0
 m=np.zeros(nb);v=np.zeros(nb)
 # Exact conditional identity, independently checking occupation weighting:
 # rho_H = rho_L * E_L[q | occupancy at z] / E_L[q].
 q_cond=(counts[:,valid].T@q)/occ[valid]
 assert np.allclose(direct[valid],source[valid]*q_cond/q.mean(),rtol=1e-11,atol=1e-14)
 m[valid]=(counts[:,valid].T@gap)/occ[valid]
 second=(counts[:,valid].T@(gap*gap))/occ[valid]
 v[valid]=np.maximum(second-m[valid]**2,0)
 # Conditional bin distribution is proportional to occupation count. A single
 # common source-energy zero cancels; rescale in log-space and enforce fixed N.
 def normalized(correction):
  ls=np.full(nb,-np.inf);ls[valid]=np.log(source[valid])-BETA*m[valid]+correction[valid]
  a=np.exp(ls-np.max(ls));return a*(n_particles/(vol*a.sum()))
 mean=normalized(np.zeros(nb));gaussian=normalized(.5*BETA**2*v)
 assert all(np.isclose(a.sum()*vol,n_particles) for a in (mean,gaussian))
 return {'source':source,'direct':direct,'smoothed':smooth,'mean_gap':mean,
         'gaussian_gap':gaussian,'gap_mean_centered_eV':m,'gap_sd_eV':np.sqrt(v),
         'count_sum':occ,'ESS':float(1/(w@w)),'max_weight':float(w.max()),
         'largest_frame_in_input':int(np.argmax(w))}

def run(a):
 trajectory=SRC/'traj_500ps_1fs.lammpstrj'; ep=SRC/'frames_501_total_energies.csv'
 erows=list(csv.DictReader(ep.open()));gaps=np.array([float(r['E_POLAR_total_eV'])-float(r['E_OFF_total_eV']) for r in erows])
 counts=[];ref_box=None
 for j,(step,bounds,group) in enumerate(frames(trajectory)):
  assert j<len(erows) and step==int(erows[j]['step'])
  box=bounds[:,1]-bounds[:,0]
  if ref_box is None:ref_box=box
  assert np.allclose(box,ref_box,atol=1e-10)
  oxygen=(group[:,0,:]-bounds[:,0])%box
  dh=group[:,1:,:]-group[:,0,None,:];dh-=np.round(dh/box)*box
  com=(oxygen+(1.008/(15.999+2*1.008))*dh.sum(axis=1))%box
  counts.append(np.histogram(com[:,2],bins=np.linspace(0,box[2],a.zbins+1))[0])
 counts=np.asarray(counts);assert counts.shape==(501,a.zbins)
 dz=ref_box[2]/a.zbins;vol=ref_box[0]*ref_box[1]*dz;z=(np.arange(a.zbins)+.5)*dz
 full=estimate(counts,gaps,vol,a.sigma_bins)
 if a.zbins==40:
  old=np.genfromtxt(ROOT/'docs/results/com_estimator_trial_20261028/com_density_profile.csv',delimiter=',',names=True)
  assert np.allclose(full['direct'],old['rho_reweighted_per_A3'],rtol=1e-10,atol=1e-11)
 first=estimate(counts[:250],gaps[:250],vol,a.sigma_bins)
 last=estimate(counts[250:],gaps[250:],vol,a.sigma_bins)
 out=a.outdir;out.mkdir(parents=True,exist_ok=False)
 rng=np.random.default_rng(a.seed);blocks=np.array_split(np.arange(501),a.blocks)
 boots={m:[] for m in METHODS};missing_high=0
 for _ in range(a.bootstrap):
  chosen=rng.integers(0,len(blocks),size=len(blocks));idx=np.concatenate([blocks[t] for t in chosen])
  if 403 not in idx:missing_high+=1
  res=estimate(counts[idx],gaps[idx],vol,a.sigma_bins)
  for m in METHODS:boots[m].append(res[m])
 boots={m:np.asarray(x) for m,x in boots.items()}
 lo={m:np.quantile(x,.025,axis=0) for m,x in boots.items()};hi={m:np.quantile(x,.975,axis=0) for m,x in boots.items()}
 mask=(full['source']>=.005)&(full['direct']>=.005)
 low_source=full['source']<.005
 observed_upper=counts.max(axis=0)/vol
 dist={}
 for m in METHODS:
  prof=full[m];peak=int(np.argmax(prof));over=(prof>observed_upper+1e-12)
  dist[m]={'RMS_difference_to_direct_interior_per_A3':float(np.sqrt(np.mean((prof[mask]-full['direct'][mask])**2))),
   'RMS_first_vs_last_interior_per_A3':float(np.sqrt(np.mean((first[m][mask]-last[m][mask])**2))),
   'median_bootstrap_95pct_width_interior_per_A3':float(np.median((hi[m]-lo[m])[mask])),
   'first_vs_last_global_L1_fraction':float(.5*np.sum(abs(first[m]-last[m]))*vol/384),
   'peak_density_per_A3':float(prof[peak]),'peak_z_A':float(z[peak]),
   'source_occupancy_at_peak':int(full['count_sum'][peak]),
   'maximum_frame_count_at_peak':int(counts[:,peak].max()),
   'fraction_particles_in_low_source_density_bins':float(prof[low_source].sum()*vol/384),
   'bins_above_raw_frame_maximum':int(over.sum())}
 # Raw-bin frame range is a test of finite-sample support only for unmodified
 # histogram and conditional approximations. A smoothed histogram estimates a
 # different spatially averaged observable and is not bounded by raw-bin maxima.
 dist['smoothed']['bins_above_raw_frame_maximum']=None
 assert dist['direct']['bins_above_raw_frame_maximum']==0

 summary={'input_frames':501,'water_molecules':384,'temperature_K':300,'zbins':a.zbins,'bin_width_A':float(dz),
 'smooth_sigma_bins':a.sigma_bins,'smooth_sigma_A':float(a.sigma_bins*dz),
 'bootstrap_blocks':a.blocks,'bootstrap_resamples':a.bootstrap,'bootstrap_seed':a.seed,
 'bootstrap_resamples_missing_max_weight_frame':missing_high,
 'direct_full_ESS':full['ESS'],'direct_first_half_ESS':first['ESS'],'direct_last_half_ESS':last['ESS'],
 'max_weight_full':full['max_weight'],'max_weight_first_half':first['max_weight'],'max_weight_last_half':last['max_weight'],
 'interior_mask_source_and_direct_density_min_per_A3':.005,'interior_bins':int(mask.sum()),
 'comparison':dist,'input_sha256':{'trajectory':hsh(trajectory),'energy_csv':hsh(ep)},
 'limitations':'No independent target POLAR trajectory or target forces. Full vs baseline and split differences are not errors/accuracy. Histogram smoothing has bias; mean and Gaussian-gap are approximations; binned conditional-gap tails may be poorly sampled. Block bootstrap cannot reveal missing target modes; trajectory dependence and equilibration unresolved. No absolute c1/chemical potential and no c2.'}
 np.savez_compressed(out/'frame_observables.npz',counts=counts,gap_eV=gaps,z_A=z,bin_volume_A3=vol)
 with (out/'summary.json').open('x') as f:json.dump(summary,f,indent=2)
 with (out/'full_profiles.csv').open('x',newline='') as f:
  wr=csv.writer(f,lineterminator='\n');wr.writerow(['z_A']+list(METHODS)+['conditional_deltaU_centered_eV','conditional_deltaU_std_eV','source_occupation_count','interior_mask'])
  for i in range(a.zbins):wr.writerow([z[i]]+[full[m][i] for m in METHODS]+[full['gap_mean_centered_eV'][i],full['gap_sd_eV'][i],full['count_sum'][i],int(mask[i])])
 with (out/'split_profiles.csv').open('x',newline='') as f:
  wr=csv.writer(f,lineterminator='\n');wr.writerow(['z_A']+[m+'_'+s for s in ('first','last') for m in METHODS])
  for i in range(a.zbins):wr.writerow([z[i]]+[x[m][i] for x in (first,last) for m in METHODS])
 with (out/'bootstrap_intervals.csv').open('x',newline='') as f:
  wr=csv.writer(f,lineterminator='\n');wr.writerow(['z_A']+[m+'_'+s for s in ('q025','q975') for m in METHODS])
  for i in range(a.zbins):wr.writerow([z[i]]+[x[m][i] for x in (lo,hi) for m in METHODS])
 figstyle(a,out,z,full,first,last,mask,summary)
 print(json.dumps(summary,indent=2))

def figstyle(a,out,z,full,first,last,mask,summary):
 font='/home/qiuyong/fonts/arial.ttf';font_manager.fontManager.addfont(font)
 plt.rcParams.update({'font.family':'Arial','pdf.fonttype':42,'svg.fonttype':'none','font.size':8,'axes.spines.top':False,'axes.spines.right':False})
 colors={'source':'#555555','direct':'#0072B2','smoothed':'#59A89C','mean_gap':'#D55E00','gaussian_gap':'#8D6BB7'}
 labels={'source':'OFF23 (source)','direct':'Direct reweighting','smoothed':'Direct + smoothing','mean_gap':'Mean-gap approx.','gaussian_gap':'Gaussian-gap approx.'}
 fig,axes=plt.subplots(2,2,figsize=(7.1,5.9),layout='constrained',sharex=True)
 ax=axes.ravel()
 for m in ('source','direct','smoothed'):ax[0].plot(z,full[m],label=labels[m],color=colors[m],lw=1.3)
 ax[0].legend(fontsize=7,loc='upper right');ax[0].set_ylabel('COM density (1/Å³)')
 for m in ('source','direct','mean_gap','gaussian_gap'):ax[1].plot(z,full[m],label=labels[m],color=colors[m],lw=1.2,ls='--' if m in ('mean_gap','gaussian_gap') else '-')
 ax[1].legend(fontsize=6.5,loc='upper center');ax[1].set_ylabel('COM density (1/Å³)')
 for m in ('source','direct','smoothed'):ax[2].plot(z,first[m]-last[m],label=labels[m],color=colors[m],lw=1.1)
 ax[2].axhline(0,color='0.7',lw=.6);ax[2].legend(fontsize=6.5,loc='lower right');ax[2].set_ylabel('First − last half (1/Å³)')
 low=full['source']<.005
 for j in np.where(low)[0]:ax[3].axvspan(z[j]-(z[1]-z[0])/2,z[j]+(z[1]-z[0])/2,color='0.92',lw=0)
 ax[3].plot(z,BETA*full['gap_sd_eV'],color='#8D6BB7',lw=1.2);ax[3].set_ylabel('β × conditional gap SD')
 ax[3].text(.02,.04,'Gray: source density < 0.005/Å³',transform=ax[3].transAxes,fontsize=6.7)
 for j,aa in enumerate(ax):
  aa.text(.01,.97,chr(97+j),transform=aa.transAxes,weight='bold',va='top');aa.set_xlim(float(z[0]),float(z[-1]));aa.tick_params(direction='out',length=3)
 for aa in ax[2:]:aa.set_xlabel('z (Å)')
 fig.suptitle('OFF23 → POLAR density-estimator pilot: no direct target reference',fontsize=9)
 fig.savefig(out/'density_estimator_comparison.pdf');fig.savefig(out/'density_estimator_comparison.png',dpi=300);fig.savefig(out/'density_estimator_comparison.svg');plt.close(fig)

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--outdir',type=Path,required=True);p.add_argument('--zbins',type=int,default=40);p.add_argument('--sigma-bins',type=float,default=1);p.add_argument('--blocks',type=int,default=20);p.add_argument('--bootstrap',type=int,default=200);p.add_argument('--seed',type=int,default=20261028);run(p.parse_args())
