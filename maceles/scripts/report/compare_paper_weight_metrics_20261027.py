"""Compare Kish, entropy, and Herzog weight diagnostics on the same OFF frames."""
import csv,hashlib,json,math
from pathlib import Path
root=Path(__file__).resolve().parents[2]
p=root/'validation/off23_500ps_water1152_2026-10-23/frames_501_total_energies.csv'
rows=list(csv.DictReader(p.open()))
d=[float(r['E_POLAR_total_eV'])-float(r['E_OFF_total_eV']) for r in rows]
beta=1/(8.617333262145e-5*300);n=len(d)
l=[-beta*(x-min(d)) for x in d];z=sum(math.exp(x) for x in l)
w=[math.exp(x)/z for x in l];lw=[x-math.log(z) for x in l]
ess=1/sum(x*x for x in w)
H=-sum(x*y for x,y in zip(w,lw)); shannon=math.exp(H)
asc=sorted(w);cs=0
for j,x in enumerate(asc,1):
 cs+=x
 if cs>=.5:break
s2=sum((beta*(x-sum(d)/n))**2 for x in d)/n
out={'n':n,'temperature_K':300,'input_sha256':hashlib.sha256(p.read_bytes()).hexdigest(), 'Kish_ESS':ess,'entropy_ESS_DiffTRe':shannon,'entropy_nats':H,'Herzog_Iw_as_defined_ascending':(n-j)/n,'Herzog_ascending_index_N':j,'top_sorted_frames_needed_for_half_weight':next(k for k in range(1,n+1) if sum(sorted(w,reverse=True)[:k])>=.5), 'CV_squared_ddof0':n/ess-1,'empirical_D1_plugin_NOT_true_KL':math.log(n)-H,'empirical_D2_plugin_NOT_true_Renyi2':math.log(n/ess), 'gaussian_assumption_sigma_beta_squared':s2,'gaussian_assumption_D1':s2/2,'gaussian_assumption_D2':s2,'gaussian_assumption_exp_D1':math.exp(s2/2),'gaussian_assumption_exp_D2':math.exp(s2), 'warning':'D1/D2 plugin estimates on finite frames do not establish population divergences. Gaussian values assume exact source normal energy gap and exponential tilt; not estimates of required production trajectory lengths.'}
assert 1<=ess<=shannon<=n
assert math.isclose(math.exp(-out['empirical_D2_plugin_NOT_true_Renyi2']),ess/n,rel_tol=1e-12)
with (root/'docs/results/paper_weight_metric_comparison_20261027.json').open('x') as f:json.dump(out,f,indent=2)
print(json.dumps(out,indent=2))
