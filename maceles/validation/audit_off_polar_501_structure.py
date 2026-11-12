"""Read immutable trajectory; publish new RDF/spatial diagnostics only."""
import argparse,csv,json,math,os
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1];S=R/'validation/off23_500ps_water1152_2026-10-23'
parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
out=args.output;out.mkdir(parents=True,exist_ok=False)
rows=list(csv.DictReader((S/'frames_501_total_energies.csv').open()));gap=np.array([float(r['E_POLAR_total_eV'])-float(r['E_OFF_total_eV']) for r in rows])
logw=-(gap-gap.min())/(8.617333262e-5*300);w=np.exp(logw);w/=w.sum()
edges=np.linspace(0,8,81);shell=4*np.pi/3*np.diff(edges**3)
histograms=[];norms=[];profiles=[[],[],[]];lengths=[]
with (S/'traj_500ps_1fs.lammpstrj').open() as f:
 while True:
  line=f.readline()
  if not line:break
  assert line.strip()=='ITEM: TIMESTEP';step=int(f.readline());assert f.readline().strip()=='ITEM: NUMBER OF ATOMS'
  n=int(f.readline());assert n==1152;assert f.readline().startswith('ITEM: BOX BOUNDS')
  bounds=np.array([[float(x) for x in f.readline().split()[:2]] for _ in range(3)]);box=bounds[:,1]-bounds[:,0]
  assert f.readline().strip()=='ITEM: ATOMS id type x y z'
  oxygen=[]
  for _ in range(n):
   values=f.readline().split()
   if values[1]=='1':oxygen.append([float(x) for x in values[2:5]])
  xyz=np.array(oxygen)-bounds[:,0];assert len(xyz)==384
  assert step==int(rows[len(histograms)]['step'])
  delta=xyz[:,None,:]-xyz[None,:,:];delta-=np.round(delta/box)*box
  d=np.sqrt((delta*delta).sum(-1));pairs=d[np.triu_indices(384,k=1)]
  h=np.histogram(pairs,edges)[0];histograms.append(h);norms.append(.5*384*383*shell/np.prod(box));lengths.append(box)
  for axis,bins in enumerate((24,24,40)):
   profiles[axis].append(np.histogram(np.mod(xyz[:,axis],box[axis]),np.linspace(0,box[axis],bins+1))[0])
assert len(histograms)==501
h=np.array(histograms);norms=np.array(norms);g_source=h.sum(0)/norms.sum(0);g_rw=(w[:,None]*h).sum(0)/(w[:,None]*norms).sum(0)
stored=json.loads((S/'reweight_rdf.json').read_text());r=(edges[1:]+edges[:-1])/2
comparison={}
for key,g in [('g_off',g_source),('g_reweighted',g_rw)]:
 old=np.asarray(stored[key]);comparison[key]={'corrected_vs_twice_stored_maxabs':float(abs(g-2*old).max()),
   'corrected_vs_stored_maxabs':float(abs(g-old).max()),'corrected_tail_6to8A':float(g[(r>=6)&(r<=8)].mean())}
spatial={}
for axis,label in enumerate(('x','y','z')):
 p=np.array(profiles[axis]);mean=p.mean(0);weighted=w@p
 spatial[label]={'mean_bin_occupancy':mean.tolist(),'coefficient_of_variation_time_mean':float(mean.std()/mean.mean()),
   'min_over_mean':float(mean.min()/mean.mean()),'max_over_mean':float(mean.max()/mean.mean()),
   'mean_empty_bins_per_frame':float((p==0).sum(-1).mean()),'bins':p.shape[1],
   'weighted_mean_bin_occupancy':weighted.tolist()}
report={'complete':True,'frames':501,'N_water':384,'box_A':lengths[0].tolist(),
        'number_density_A3':float(384/np.prod(lengths[0])),'RDF_comparison':comparison,'spatial':spatial,
        'scope':'No new MD or potential calls; spatial inhomogeneity diagnostic, not phase-equilibrium proof. Corrected RDF is not a certified c2 label.'}
(out/'report.json').write_text(json.dumps(report,indent=2))
(out/'corrected_rdf.json').write_text(json.dumps({'r_A':r.tolist(),'g_source':g_source.tolist(),'g_reweighted':g_rw.tolist(),'note':report['scope']},indent=2))
np.savez_compressed(out/'OO_pair_histograms.npz',histograms=h,edges_A=edges,weights=w)
print(json.dumps({k:report[k] for k in ('frames','number_density_A3','RDF_comparison')},indent=2))
print('spatial_CVs',{k:v['coefficient_of_variation_time_mean'] for k,v in spatial.items()})
