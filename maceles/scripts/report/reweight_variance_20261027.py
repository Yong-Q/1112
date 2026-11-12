"""Recompute weight dispersion from the uploaded energies; never replace outputs."""
import csv, json, math, hashlib
from pathlib import Path
root=Path(__file__).resolve().parents[2]
src=root/'validation/off23_500ps_water1152_2026-10-23/frames_501_total_energies.csv'
rows=list(csv.DictReader(src.open()))
d=[float(r['E_POLAR_total_eV'])-float(r['E_OFF_total_eV']) for r in rows]
n=len(d); beta=1/(8.617333262145e-5*300)
a=[math.exp(-beta*(x-min(d))) for x in d]
s=sum(a); w=[x/s for x in a]; R=[n*x for x in w]
mean=lambda v:sum(v)/len(v)
var=lambda v:mean([(x-mean(v))**2 for x in v])
ess=1/sum(x*x for x in w)
m={'N':n,'temperature_K':300,'kBT_eV':1/beta,'delta_U_mean_eV':mean(d),'delta_U_std_eV':math.sqrt(var(d)), 'sigma_beta':beta*math.sqrt(var(d)), 'shifted_raw_mean':mean(a),'shifted_raw_variance_ddof0':var(a),'normalized_weight_variance_ddof0':var(w),'mean_one_weight_variance_ddof0':var(R),'CV_squared':var(a)/mean(a)**2,'ESS':ess,'ESS_fraction':ess/n,'max_weight':max(w),'top3_mass':sum(sorted(w,reverse=True)[:3]),'log_raw_mean':-beta*min(d)+math.log(mean(a)),'log_raw_variance':-2*beta*min(d)+math.log(var(a)),'gaussian_heuristic_ESS_fraction':math.exp(-beta**2*var(d)),'input_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'definitions':'All variances use denominator N. R_i=N*w_i. a_i=exp[-beta*(delta_U_i-min(delta_U))]. Empirical weight statistics; not a population overlap estimate or confidence interval.'}
assert math.isclose(var(R),n/ess-1,rel_tol=1e-12)
assert math.isclose(var(w),1/(n*ess)-1/n**2,rel_tol=1e-12)
assert math.isclose(var(a)/mean(a)**2,var(R),rel_tol=1e-12)
out=root/'docs/results/reweight_variance_20261027.json'
with out.open('x') as f:json.dump(m,f,indent=2)
tex=root/'docs/results/reweight_variance_20261027_numbers.tex'
with tex.open('x') as f:
 for k,v in m.items():
  if isinstance(v,(float,int)):
   name=''.join(p.capitalize() for p in k.split('_'))
   f.write('\\expandafter\\def\\csname '+name+'\\endcsname{'+format(v,'.9g')+'}\n')
print(json.dumps(m,indent=2))
