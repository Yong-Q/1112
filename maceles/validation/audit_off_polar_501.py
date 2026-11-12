"""Independent501-frame replay, MD/source consistency, and RDF normalization."""
import argparse,csv,hashlib,json,math,os,statistics
from pathlib import Path
R=Path(__file__).resolve().parents[1];S=R/'validation/off23_500ps_water1152_2026-10-23'
parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
out=args.output;out.mkdir(parents=True,exist_ok=False)
KB=8.617333262e-5
rows=list(csv.DictReader((S/'frames_501_total_energies.csv').open()));assert len(rows)==501
source=[float(x['E_OFF_total_eV']) for x in rows];target=[float(x['E_POLAR_total_eV']) for x in rows]
gaps=[h-l for l,h in zip(source,target)];beta=1/(KB*300)
def stats(x):return {'mean':statistics.mean(x),'std_population':statistics.pstdev(x),'min':min(x),'max':max(x)}
def normalize(g):
 logw=[-beta*(x-min(g)) for x in g];w=[math.exp(x) for x in logw];total=math.fsum(w)
 return [x/total for x in w]
w=normalize(gaps);ess=1/math.fsum(x*x for x in w);stored=json.loads((S/'reweight_summary.json').read_text())
assert abs(ess-stored['ess'])<1e-7
shift=normalize([x+10000 for x in gaps]);shift_change=max(abs(x-y) for x,y in zip(shift,w));assert shift_change<1e-8
rank=sorted(range(len(w)),key=lambda i:w[i],reverse=True)
mu=statistics.mean(source);mt=statistics.mean(target)
slope=math.fsum((s-mu)*(t-mt) for s,t in zip(source,target))/math.fsum((s-mu)**2 for s in source)
residual=[(t-mt)-slope*(s-mu) for s,t in zip(source,target)]
# Read the production thermo section only; the log includes equilibration first.
thermo={};production=False
for line in (S/'log.production_1fs').read_text().splitlines():
 if line.strip()=='run 500000':production=True;continue
 if not production:continue
 cols=line.split()
 if len(cols)!=3:continue
 try:
  step=int(cols[0]);temperature=float(cols[1]);energy=float(cols[2])
 except ValueError:continue
 if step>=0 and step<=500000 and step%1000==0:thermo[step]=(temperature,energy)
assert len(thermo)==501,len(thermo)
md=[thermo[int(row['step'])][1] for row in rows];temperatures=[thermo[int(row['step'])][0] for row in rows]
engine_diff=[s-e for s,e in zip(source,md)]
# Replacing source energies by thermo values is a sensitivity experiment only:
# coordinates are rounded and the engine's force-defined ensemble is not proven.
md_gaps=[t-s for t,s in zip(target,md)];mdw=normalize(md_gaps)
blocks=[]
for b in range(10):
 indices=[i for i in range(501) if min(9,i*10//501)==b]
 blockw=normalize([gaps[i] for i in indices])
 blocks.append({'block':b,'first_ps':float(rows[indices[0]]['time_ps']),'last_ps':float(rows[indices[-1]]['time_ps']),
   'temperature_K':statistics.mean(temperatures[i] for i in indices),'gap_eV':stats([gaps[i] for i in indices]),
   'global_weight_mass':math.fsum(w[i] for i in indices),'within_block_ESS':1/math.fsum(x*x for x in blockw)})
cuts=[]
for cut in (0,20,50,100,200):
 ids=[i for i,row in enumerate(rows) if float(row['time_ps'])>=cut];ww=normalize([gaps[i] for i in ids])
 cuts.append({'discard_before_ps':cut,'remaining_frames':len(ids),'ess':1/math.fsum(x*x for x in ww),
              'sigma_beta':statistics.pstdev(gaps[i] for i in ids)*beta,'scope':'predeclared stationarity sensitivity, not selected for high ESS'})
rdf=json.loads((S/'reweight_rdf.json').read_text());rdf_keys=list(rdf)
radii=rdf['r'];tail=[i for i,r in enumerate(radii) if 6<=r<=8]
rdf_tail={k:statistics.mean(v[i] for i in tail) for k,v in rdf.items() if k!='r' and isinstance(v,list) and len(v)==len(radii)}
report={'complete':True,'input_sha256':{f:hashlib.sha256((S/f).read_bytes()).hexdigest() for f in ('frames_501_total_energies.csv','log.production_1fs','in.production_1fs','reweight_500ps_water1152.py')},
 'frames':501,'temperature_setpoint_K':300,'ess_independent':ess,'stored_ess':stored['ess'],'sigma_beta_population':statistics.pstdev(gaps)*beta,
 'deltaU_eV':stats(gaps),'constant_shift10000eV_max_weight_error':shift_change,
 'top10':[{'frame':i,'time_ps':float(rows[i]['time_ps']),'weight':w[i],'gap_eV':gaps[i]} for i in rank[:10]],
 'top1_mass':w[rank[0]],'top3_mass':math.fsum(w[i] for i in rank[:3]),'top10_mass':math.fsum(w[i] for i in rank[:10]),
 'thermo_temperature_K':stats(temperatures),'ten_time_blocks':blocks,'discard_sensitivity':cuts,
 'source_PyTorch_minus_MD_eV':stats(engine_diff),
 'source_FP32_ULP_eV':2**(math.frexp(abs(mu))[1]-24),'source_FP32_ULP_over_kBT':2**(math.frexp(abs(mu))[1]-24)*beta,
 'MD_source_energy_sensitivity_ESS':1/math.fsum(x*x for x in mdw),'MD_source_energy_sensitivity_sigma_beta':statistics.pstdev(md_gaps)*beta,
 'sensitivity_caveat':'Thermo rounded to~.01eV and dumped coordinates rounded; not a certified source-Hamiltonian correction',
 'linear_centered_slope':slope,'linear_residual_sigma_beta':statistics.pstdev(residual)*beta,
 'RDF_existing_keys':rdf_keys,'RDF_existing_tail_6to8A':rdf_tail,
 'RDF_defect':'oo_rdf_weighted hist counts N(N-1)/2 unordered pairs but norm uses N(N-1)/V. Missing1/2 produces exactly half the RDF for same input; does not affect ESS.',
 'input_neighbor_policy':'every100, checkno, skin2A; missed-neighbor risk requires shorter-interval engine test; not established from1ps dumps',
 'model_weights_reevaluated':False}
(out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
with (out/'independent_weights.csv').open('x',newline='') as f:
 writer=csv.writer(f);writer.writerow(['frame','time_ps','deltaU_eV','weight'])
 writer.writerows((i,row['time_ps'],gap,weight) for i,(row,gap,weight) in enumerate(zip(rows,gaps,w)))
print(json.dumps({k:report[k] for k in ('ess_independent','sigma_beta_population','top3_mass','deltaU_eV','thermo_temperature_K','source_PyTorch_minus_MD_eV','MD_source_energy_sensitivity_ESS','source_FP32_ULP_eV','linear_residual_sigma_beta','RDF_existing_tail_6to8A')},indent=2))
