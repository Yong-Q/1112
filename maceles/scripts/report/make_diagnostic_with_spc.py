#!/usr/bin/env python3
"""Add a matched SPC oxygen-site density reference to the README diagnostic."""
import argparse,csv,hashlib,json,os,shutil
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'docs/results/20261027_off_polar_audit'
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--outdir',type=Path,required=True)
g=p.add_mutually_exclusive_group(required=True)
g.add_argument('--spc-trajectory',type=Path)
g.add_argument('--spc-profile',type=Path,help='Saved SPC oxygen profile, for plotting without the large trajectory.')
args=p.parse_args();out=args.outdir;out.mkdir(parents=True,exist_ok=False)
spatial=json.loads((BASE/'spatial_audit.json').read_text());box=np.array(spatial['box_A']);n=spatial['spatial']['z']['bins'];z=(np.arange(n)+.5)*box[2]/n
volume=float(np.prod(box)/n)
if args.spc_trajectory:
 counts=[];steps=[]
 with args.spc_trajectory.open() as f:
  while True:
   line=f.readline()
   if not line:break
   assert line.strip()=='ITEM: TIMESTEP'
   step=int(f.readline());assert f.readline().strip()=='ITEM: NUMBER OF ATOMS';number=int(f.readline());assert number==1152
   assert f.readline().strip()=='ITEM: BOX BOUNDS pp pp pp'
   bounds=np.array([[float(v) for v in f.readline().split()[:2]] for _ in range(3)])
   assert np.allclose(bounds[:,1]-bounds[:,0],box,rtol=0,atol=1e-10)
   fields=f.readline().split()[2:];data=np.array([[float(v) for v in f.readline().split()] for _ in range(number)])
   oxygen=data[:,fields.index('type')]==1;assert oxygen.sum()==384
   zz=(data[oxygen,fields.index('z')]-bounds[2,0])%box[2]
   counts.append(np.histogram(zz,bins=np.linspace(0,box[2],n+1))[0]);steps.append(step)
 assert len(counts)==501 and steps==list(range(0,500001,1000))
 rho=np.asarray(counts).mean(axis=0)/volume
 trajectory_hash=hashlib.sha256(args.spc_trajectory.read_bytes()).hexdigest()
else:
 a=np.genfromtxt(args.spc_profile,delimiter=',',names=True);assert np.allclose(a['z_A'],z);rho=a['rho_SPC_O_Aminus3'];trajectory_hash=None
assert np.isclose(rho.sum()*volume,384)
with (out/'SPC_O_z_density.csv').open('w',newline='') as f:
 w=csv.writer(f,lineterminator='\n');w.writerow(['z_A','rho_SPC_O_Aminus3']);w.writerows(zip(z,rho))
meta={'model':'SPC three-site parameters + PPPM, not SPC/E','N_water':384,'N_atoms':1152,'temperature_K':300,'dt_fs':1,'production_ps':500,'frames':501,'time_average':'All production frames 0–500 ps; no smoothing or spatial centering','density_variable':'oxygen-site count per spatial volume, same as original panel b','z_bins':n,'box_A':box.tolist(),'integral_water':float(rho.sum()*volume),'SPC_peak_O_Aminus3':float(rho.max()),'source_trajectory_sha256':trajectory_hash,'reference_protocol':'SPC run: 100 ps equilibration, 500 ps production; OFF run: 20 ps equilibration, 500 ps production. Same N, box and T, different potentials and sampling histories.','full_reference':'https://github.com/starlessinred-interns/qiuyong/tree/main/report/homogeneous_sampling_20261029_v4/evidence/spc_water_reference_20261029'}
(out/'metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n')
cache=out/'matplotlib-cache';os.environ['MPLCONFIGDIR']=str(cache)
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt,font_manager
for f in ['arial.ttf','ARIALBD.TTF']:
 fp=Path('/home/qiuyong/fonts')/f
 if fp.exists():font_manager.fontManager.addfont(fp)
plt.rcParams.update({'font.family':'Arial' if Path('/home/qiuyong/fonts/arial.ttf').exists() else 'DejaVu Sans','font.size':8,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7})
rows=list(csv.DictReader((BASE/'independent_weights.csv').open()));t=np.array([float(x['time_ps']) for x in rows]);weights=np.array([float(x['weight']) for x in rows])
fig,(a,b)=plt.subplots(1,2,figsize=(7.09,2.65),layout='constrained')
a.vlines(t,0,weights,color='#0072B2',lw=.65);a.set(xlabel='Time (ps)',ylabel='Normalized weight',xlim=(-5,505),ylim=(0,.65));a.text(.04,.94,f'501 frames\nTop 3 weights: {np.sort(weights)[-3:].sum():.1%}',transform=a.transAxes,va='top',fontsize=8)
original=spatial['spatial']['z']
b.plot(z,np.array(original['mean_bin_occupancy'])/volume,color='#0072B2',lw=1.3,label='OFF time average')
b.plot(z,np.array(original['weighted_mean_bin_occupancy'])/volume,color='#D55E00',lw=1.1,ls='--',label='Reweighted estimate')
b.plot(z,rho,color='#009E73',lw=1.4,label='SPC/PPPM time average')
b.axhline(spatial['number_density_A3'],color='.4',lw=.8,ls=':',label='Box mean')
b.set(xlabel=r'$z$ (Å)',ylabel=r'O-site density (Å$^{-3}$)',xlim=(0,box[2]),ylim=(0,.052))
b.legend(frameon=False,fontsize=6.3,loc='upper center',ncol=2,handlelength=1.8,columnspacing=.9)
for label,ax in zip(('a','b'),(a,b)):ax.text(-.15,1.04,label,transform=ax.transAxes,fontweight='bold',fontsize=11)
for fmt in ['pdf','svg','png']:fig.savefig(out/('key_diagnostics_with_SPC.'+fmt),dpi=600)
plt.close(fig)
if cache.exists() and cache.parent==out and cache.name=='matplotlib-cache':shutil.rmtree(cache)
print(json.dumps(meta,ensure_ascii=False,indent=2))
