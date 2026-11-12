"""Independent numerical check of Gaussian exponential-tilt bias sign."""
import json, math
from pathlib import Path
# Dimensionless case: beta=1, Delta=0, residual std=1.
# Simpson quadrature on [-12,12] resolves both Gaussian and tilted tails.
n=24000;lo=-12.;hi=12.;dx=(hi-lo)/n
num=den=0.
for k in range(n+1):
 x=lo+k*dx
 v=math.exp(-x-x*x/2)
 coeff=1 if k in (0,n) else (4 if k%2 else 2)
 den+=coeff*v;num+=coeff*x*v
mu=num/den
assert abs(mu+1)<1e-10
out={'beta':1.,'true_gap_mean':0.,'residual_std':1.,'quadrature_weighted_mean':mu,'analytic_weighted_mean':-1.,'uploaded_eq18_printed_plus_prediction':1.,'interpretation':'For the Gaussian uncertainty and negative Boltzmann exponent printed in Chehaibou2019 equations 17-18, exponential tilting shifts the mean by -beta*s^2. Uploaded equation18 prints +beta*s^2; this is a sign inconsistency, not a reason to change frame weights.'}
p=Path(__file__).resolve().parents[2]/'docs/results/gaussian_ml_bias_sign_20261027.json'
with p.open('x') as f:json.dump(out,f,indent=2)
print(json.dumps(out,indent=2))
