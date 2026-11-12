import argparse,csv,json
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parents[1];p=root/'docs/results/reweight_boundary_scan_20261028'
parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
with (p/'frame_fields.csv').open() as f:base=list(csv.DictReader(f))
with (p/'original_fields/frame_field_components.csv').open() as f:fields=list(csv.DictReader(f))
assert len(base)==len(fields)==501
m=np.array([float(r['time_ps'])>=20 for r in base]);t=np.array([float(r['time_ps']) for r in base])[m]
u=np.array([float(r['OFF_eV']) for r in base])[m];h=np.array([float(r['POLAR_eV']) for r in base])[m];gap=h-u; gap-=gap.mean();u-=u.mean();h-=h.mean();
seednames=[k for k in fields[0] if k.startswith('site_unit_seed_')];site=np.array([[float(r[k]) for k in seednames] for r in fields])[m];site-=site.mean(axis=0)
dip=np.array([float(r['dipole_proxy_corrected_eA']) for r in fields])[m];dip-=dip.mean()
kb=8.617333262145e-5;b0=1/(kb*300);sdgap=float(np.std(gap));sdu=float(np.std(u));baseline=sdgap*b0

def ess(l):
 w=np.exp(l-l.max());w/=w.sum();return 1/(w@w)
def vrow(kind,setting,eq,essdesc,note):
 return f'| {kind} | {setting} | {eq*1000:.2f} | {eq/sdgap:.3f} | {eq*b0:.2f} | {essdesc} | {note} |'
lines=['| 扰动类型 | 幅度或目标温度 | 相对项波动（meV） | 对300 K势差波动之比 | 指数项波动（kBT300） | 经验 ESS（481帧） | 条件 |','|---|---:|---:|---:|---:|---:|---|']
baseess=ess(-b0*gap);lines.append(vrow('OFF23→POLAR势差','300 K；无额外场',sdgap,f'{baseess:.3f}','参考；固定源温度300 K'))
for scale in [.005,.01,.02,.03,.05,.1,.15,.2,.25,.3,.4,sdgap/np.std(site,axis=0).mean(),.5,.75,1,2,4]:
 sig=np.std(site*scale,axis=0); es=np.array([ess(-b0*(gap+site[:,i]*scale)) for i in range(len(seednames))]);bound=abs(scale-sdgap/np.std(site,axis=0).mean())<1e-10
 lines.append(vrow('随机原子外势',f"{'**' if bound else ''}s={scale:.3f}{'**' if bound else ''}",float(sig.mean()),f'{es.mean():.2f} [{es.min():.2f}, {es.max():.2f}]','10种子；均值 [最小, 最大]'))
Ebound=sdgap/np.std(dip)
for E in [.001,.002,.005,.01,.02,.03,.05,.075,.1,.125,.15,Ebound,.2,.3,.5]:
 bound=abs(E-Ebound)<1e-12
 lines.append(vrow('均匀电场（固定偶极）',f"{'**' if bound else ''}{E:.4f} V/Å{'**' if bound else ''}",float(np.std(dip*E)),f'{ess(-b0*(gap+dip*E)):.3f}','1.85 D；周期展开'))
cut=json.loads((p/'summary.json').read_text());entry=next(x for x in cut['cutoffs'] if x['cutoff_inclusive_ps']==20)
for T in [entry['temperature_lower_equal_logweight_sd_K'],250,275,290,295,299,300,301,305,310,325,350,375,entry['temperature_upper_equal_logweight_sd_K'],400]:
 b=1/(kb*T);eq=abs(b0-b)/b0*sdu;val=ess(b0*u-b*h);bound=abs(T-entry['temperature_lower_equal_logweight_sd_K'])<1e-6 or abs(T-entry['temperature_upper_equal_logweight_sd_K'])<1e-6
 lines.append(vrow('目标温度变化',f"{'**' if bound else ''}{T:.2f} K ({T-300:+.2f} K){'**' if bound else ''}",eq,f'{val:.3f}','温度项单独量级；场=0'))
with args.output.open('x') as f:f.write('\n'.join(lines)+'\n')
print('table rows',len(lines)-2,'gap_sd',sdgap,'baseline_ess',baseess,'time frames',len(t),'temperature eV std',sdu)
