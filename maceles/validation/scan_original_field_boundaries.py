#!/usr/bin/env python3
"""Audit prior site and fixed-dipole fields for time-cutoff sensitivity."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from diagnose_com_off_polar import frames
R=Path(__file__).resolve().parents[1];P=R/'validation/off23_500ps_water1152_2026-10-23';KB=8.617333262145e-5;kt=KB*300
p=argparse.ArgumentParser();p.add_argument('--outdir',type=Path,required=True);a=p.parse_args()
e=list(csv.DictReader((P/'frames_501_total_energies.csv').open()));t=np.array([float(x['time_ps']) for x in e]);gap=np.array([float(x['E_POLAR_total_eV'])-float(x['E_OFF_total_eV']) for x in e]);seeds=[101,202,303,404,505,606,707,808,909,1010];rng=[np.random.default_rng(x) for x in seeds];co=[(x.normal(0,1,4),x.uniform(0,2*np.pi,4)) for x in rng];site=[];dip_fixed=[];dip_legacy=[]
for j,(step,b,g) in enumerate(frames(P/'traj_500ps_1fs.lammpstrj')):
 assert step==int(e[j]['step']);box=b[:,1]-b[:,0];xyz=g.reshape(-1,3);axis=xyz[:,0]-b[0,0];k=2*np.pi*np.arange(1,5)/box[0]
 site.append([kt*np.sum(np.sin(axis[:,None]*k[None,:]+phi[None,:])*amp[None,:]) for amp,phi in co])
 d=g[:,1:,:]-g[:,0,None,:];uncorrected=d.mean(axis=1);d-=np.round(d/box)*box;corrected=d.mean(axis=1)
 for v in (uncorrected,corrected):assert np.all(np.linalg.norm(v,axis=1)>1e-12)
 dip_legacy.append(-1.85*.2082*np.sum(uncorrected[:,2]/np.linalg.norm(uncorrected,axis=1)))
 dip_fixed.append(-1.85*.2082*np.sum(corrected[:,2]/np.linalg.norm(corrected,axis=1)))
site=np.array(site);dip=np.array(dip_fixed);legacy=np.array(dip_legacy);assert len(site)==len(gap)==501
a.outdir.mkdir(parents=True,exist_ok=False);rows=[];detail=[]
for c in [0,5,10,20,30,50,100,200]:
 m=t>=c;sd=float(np.std(gap[m]));site_sd=np.std(site[m],axis=0);dip_sd=float(np.std(dip[m]));legacy_sd=float(np.std(legacy[m]));
 for seed,sdv in zip(seeds,site_sd):detail.append({'cutoff_ps':c,'n':int(m.sum()),'field_type':'site_four_sine_atoms','seed':seed,'unit_scale_sd_eV':float(sdv),'gap_sd_eV':sd,'equal_sd_scale':sd/float(sdv)})
 for scale in [0.005,.01,.02,.03,.05,.1,.15,.2,.25,.3,.4,.5,.75,1.,2.,4.]:
  vals=site_sd*scale;rows.append({'cutoff_ps':c,'frames':int(m.sum()),'field_type':'site_four_sine_atoms','amplitude':scale,'amplitude_units':'dimensionless s (uses kBT300 per atom)', 'gap_sd_eV':sd,'field_sd_eV_mean':float(vals.mean()),'field_sd_eV_min':float(vals.min()),'field_sd_eV_max':float(vals.max()),'ratio_mean':float(vals.mean()/sd),'ratio_min':float(vals.min()/sd),'ratio_max':float(vals.max()/sd)})
 for E in [.001,.002,.005,.01,.02,.03,.05,.075,.1,.125,.15,.2,.3,.5]:
  vals=dip_sd*E;rows.append({'cutoff_ps':c,'frames':int(m.sum()),'field_type':'uniform_fixed_dipole_PBC_corrected','amplitude':E,'amplitude_units':'V/angstrom','gap_sd_eV':sd,'field_sd_eV_mean':vals,'field_sd_eV_min':vals,'field_sd_eV_max':vals,'ratio_mean':vals/sd,'ratio_min':vals/sd,'ratio_max':vals/sd})
 detail.append({'cutoff_ps':c,'n':int(m.sum()),'field_type':'uniform_fixed_dipole_PBC_corrected','seed':'','unit_scale_sd_eV':dip_sd,'gap_sd_eV':sd,'equal_sd_scale':sd/dip_sd})
 detail.append({'cutoff_ps':c,'n':int(m.sum()),'field_type':'uniform_legacy_no_unwrap','seed':'','unit_scale_sd_eV':legacy_sd,'gap_sd_eV':sd,'equal_sd_scale':sd/legacy_sd})
for name,data in [('field_table.csv',rows),('boundaries_per_seed.csv',detail)]:
 with (a.outdir/name).open('x',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(data[0]),lineterminator='\n');w.writeheader();w.writerows(data)
with (a.outdir/'frame_field_components.csv').open('x',newline='') as f:
 w=csv.writer(f,lineterminator='\n');w.writerow(['time_ps','dipole_proxy_corrected_eA','dipole_proxy_legacy_eA']+[f'site_unit_seed_{s}_eV' for s in seeds]);w.writerows([[t[i],dip[i],legacy[i],*site[i]] for i in range(len(t))])
res={'caveat':'Uniform fixed-dipole 1.85 D is an approximation not target POLAR electric-field energy or induced polarization. This computes field on stored frames, not field-driven MD. Prior source+target common external potential cancels at same T.', 'seeds':seeds,'cutoffs':{}}
for c in [0,10,20,30,50,100]:
 d=[x for x in detail if x['cutoff_ps']==c];ss=np.array([x['unit_scale_sd_eV'] for x in d if x['field_type']=='site_four_sine_atoms']);dd=[x for x in d if x['field_type']=='uniform_fixed_dipole_PBC_corrected'][0];ll=[x for x in d if x['field_type']=='uniform_legacy_no_unwrap'][0];sd=dd['gap_sd_eV'];res['cutoffs'][str(c)]={'site_sd_eV_per_s_mean':float(ss.mean()),'site_equal_sd_scale_ratio_of_mean':sd/float(ss.mean()),'site_equal_sd_scale_range':[sd/float(ss.max()),sd/float(ss.min())],'uniform_corrected_eV_per_VA':dd['unit_scale_sd_eV'],'uniform_equal_sd_VA':dd['equal_sd_scale'],'uniform_legacy_eV_per_VA':ll['unit_scale_sd_eV'],'uniform_legacy_equal_sd_VA':ll['equal_sd_scale']}
with (a.outdir/'summary.json').open('x') as f:json.dump(res,f,indent=2)
print(json.dumps(res,indent=2))
