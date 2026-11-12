#!/usr/bin/env python3
"""Compare OFF trajectory and target-weighted COM observables on stored frames.

Outputs exploratory diagnostics only. The dump has no molecule IDs; initial
O-H-H triplets by atom ID are audited using distances and nearest-O assignment.
No estimator here establishes absolute c1, homogeneous c2, or target accuracy.
"""
import argparse,csv,hashlib,json,math
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
S=ROOT/'validation/off23_500ps_water1152_2026-10-23'

def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
 return h.hexdigest()

def frames(path):
 with path.open() as f:
  while True:
   line=f.readline()
   if not line:break
   assert line.strip()=='ITEM: TIMESTEP'
   step=int(f.readline());assert f.readline().strip()=='ITEM: NUMBER OF ATOMS'
   n=int(f.readline());assert n==1152
   assert f.readline().strip()=='ITEM: BOX BOUNDS pp pp pp'
   bounds=np.array([[float(x) for x in f.readline().split()[:2]] for _ in range(3)])
   assert f.readline().strip()=='ITEM: ATOMS id type x y z'
   data=np.array([[float(x) for x in f.readline().split()] for _ in range(n)])
   data=data[np.argsort(data[:,0])];assert np.array_equal(data[:,0],np.arange(1,n+1))
   assert np.array_equal(data[:,1],np.tile([1,2,2],384))
   yield step,bounds,data[:,2:5].reshape(384,3,3)

def analyze(args):
 traj=S/'traj_500ps_1fs.lammpstrj';energies=S/'frames_501_total_energies.csv'
 rows=list(csv.DictReader(energies.open()));delta=np.array([float(r['E_POLAR_total_eV'])-float(r['E_OFF_total_eV']) for r in rows])
 beta=1/(8.617333262145e-5*300);logw=-beta*(delta-delta.min());weights=np.exp(logw);weights/=weights.sum()
 density_frames=[];pair_frames=[];dists_all=[];mismatch=0;bad_frames=0;ref_box=None
 edges=np.linspace(0,8,81);shell=4*np.pi/3*np.diff(edges**3);num=384; n_atoms=num*3
 indices=np.triu_indices(num,1)
 for t,(step,bounds,group) in enumerate(frames(traj)):
  assert t<len(rows) and step==int(rows[t]['step'])
  box=bounds[:,1]-bounds[:,0]
  if ref_box is None:ref_box=box
  assert np.allclose(ref_box,box,atol=1e-10)
  oxygen=(group[:,0,:]-bounds[:,0])%box
  dh=group[:,1:,:]-group[:,0,None,:];dh-=np.round(dh/box)*box
  dh_lengths=np.linalg.norm(dh,axis=2);dists_all.extend(dh_lengths.ravel())
  # Validate fixed molecule identity against nearest oxygen for each H.
  hyd=(group[:,1:,:]-bounds[:,0]).reshape(-1,3)%box
  d=hyd[:,None,:]-oxygen[None,:,:];d-=np.round(d/box)*box
  nearest=np.argmin(np.einsum('ijk,ijk->ij',d,d),axis=1)
  frame_mismatch=int(np.count_nonzero(nearest!=np.repeat(np.arange(num),2)))
  mismatch+=frame_mismatch;bad_frames+=int(frame_mismatch>0)
  # Unwrap each H relative to its oxygen before the mass-weighted COM.
  com=(oxygen+(1.008/(15.999+2*1.008))*dh.sum(axis=1))%box
  zhist=np.histogram(com[:,2],bins=np.linspace(0,box[2],args.zbins+1))[0]
  density_frames.append(zhist/(box[0]*box[1]*(box[2]/args.zbins)))
  vec=com[:,None,:]-com[None,:,:];vec-=np.round(vec/box)*box
  dist=np.linalg.norm(vec[indices],axis=1)
  hist=np.histogram(dist,bins=edges)[0]
  pair_frames.append(hist/(num*(num-1)/(2*np.prod(box))*shell))
 assert len(density_frames)==len(rows)==501
 rho=np.array(density_frames);g=np.array(pair_frames)
 off_rho=rho.mean(axis=0);weighted_rho=weights@rho
 off_g=g.mean(axis=0);weighted_g=weights@g
 assert np.isclose(np.sum(off_rho)*(ref_box[2]/args.zbins)*ref_box[0]*ref_box[1],num)
 assert np.isclose(np.sum(weighted_rho)*(ref_box[2]/args.zbins)*ref_box[0]*ref_box[1],num)
 zcent=(np.arange(args.zbins)+.5)*ref_box[2]/args.zbins
 valid=(off_rho>0)&(weighted_rho>0)
 log_ratio=np.full(args.zbins,np.nan);log_ratio[valid]=np.log(weighted_rho[valid]/off_rho[valid])
 ref=int(np.argmax(off_rho));assert valid[ref];centered=log_ratio-log_ratio[ref]
 results=args.outdir;results.mkdir(parents=True,exist_ok=False)
 with (results/'com_density_profile.csv').open('x',newline='') as f:
  wr=csv.writer(f);wr.writerow(['z_A','rho_OFF_per_A3','rho_reweighted_per_A3','log_density_ratio','relative_c1_difference_EL_candidate'])
  for row in zip(zcent,off_rho,weighted_rho,log_ratio,centered):wr.writerow([format(v,'.12g') if math.isfinite(v) else '' for v in row])
 rmid=(edges[:-1]+edges[1:])/2
 with (results/'com_radial_pair_diagnostic.csv').open('x',newline='') as f:
  wr=csv.writer(f);wr.writerow(['r_A','g_OFF_uniform_reference','g_reweighted_uniform_reference'])
  wr.writerows(zip(rmid,off_g,weighted_g))
 tail=(rmid>=6)&(rmid<8)
 interior=(off_rho>=args.min_density)&(weighted_rho>=args.min_density)&valid
 rms=float(np.sqrt(np.mean(centered[interior]**2))) if interior.any() else None
 # Connected covariance of COM bin occupations, included as c2 feasibility
 # diagnostic only: fixed total N forces a zero mode in both matrices.
 bin_volume=float(ref_box[0]*ref_box[1]*ref_box[2]/args.zbins)
 counts=rho*bin_volume
 counts0=counts-counts.mean(axis=0)
 countsw=counts-weights@counts
 cov_off=counts0.T@counts0/len(counts)
 cov_w=(countsw*weights[:,None]).T@countsw
 cov_off=(cov_off+cov_off.T)/2;cov_w=(cov_w+cov_w.T)/2
 ev_off=np.linalg.eigvalsh(cov_off);ev_w=np.linalg.eigvalsh(cov_w)
 def covariance_info(c,ev):
  assert float(np.max(np.abs(c@np.ones(args.zbins))))<1e-7
  scale=float(np.max(abs(ev)));tol=max(1e-12,scale*1e-10)
  return {'trace':float(np.trace(c)),'minimum_eigenvalue':float(ev[0]),'maximum_eigenvalue':float(ev[-1]),
   'eigenvalues_above_1e-10_of_max':int(np.count_nonzero(ev>tol)),
   'effective_rank_participation_ratio':float(np.trace(c)**2/np.sum(c*c)),
   'fixed_N_zero_mode_residual_max':float(np.max(np.abs(c@np.ones(args.zbins))))}
 np.savez_compressed(results/'com_bin_covariance_diagnostic.npz',cov_OFF=cov_off,cov_reweighted=cov_w,z_A=zcent)
 summary={'frames':len(rows),'water_molecules':num,'temperature_K':300,'time_step_ps':1,'weight_ESS':float(1/(weights@weights)),
 'max_weight_frame':int(np.argmax(weights)),'max_weight':float(weights.max()),'box_A':ref_box.tolist(),
 'OH_distance_max_A':float(np.max(dists_all)),'OH_distance_p999_A':float(np.quantile(dists_all,.999)),
 'hydrogen_nearest_oxygen_disagreements':mismatch,'frames_with_identity_disagreements':bad_frames,
 'mean_density_A3':float(num/np.prod(ref_box)),'z_bin_count':args.zbins,'reference_z_A':float(zcent[ref]),
 'zero_weighted_density_bins':int(np.count_nonzero(weighted_rho==0)),
 'relative_c1_difference_candidate_rms_interior':rms,'interior_threshold_A3':args.min_density,'interior_bins':int(interior.sum()),
 'g_tail_6_8_A_OFF':float(np.mean(off_g[tail])),'g_tail_6_8_A_weighted':float(np.mean(weighted_g[tail])),
 'fixed_N_COM_bin_covariance':{'OFF':covariance_info(cov_off,ev_off),'reweighted':covariance_info(cov_w,ev_w),'scope':'Fixed N imposes a null mode; these matrices are noisy empirical diagnostics, not grand-canonical response or c2.'},
 'input_sha256':{'trajectory':digest(traj),'energies':digest(energies)},
 'scope':'Fixed triplets not explicit topology; relative c1 is only an EL diagnostic after constant centering, subject to matched position-only external potential, ideal reference, equilibrium and finite-N ensemble assumptions. c2 is not inferred: z inhomogeneity invalidates naive homogeneous scalar OZ. Reweighted target has no independent target MD; source-energy consistency unresolved; low ESS.'}
 with (results/'summary.json').open('x') as f:json.dump(summary,f,indent=2)
 print(json.dumps(summary,indent=2))

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--outdir',required=True,type=Path);p.add_argument('--zbins',type=int,default=40);p.add_argument('--min-density',type=float,default=.005);analyze(p.parse_args())
