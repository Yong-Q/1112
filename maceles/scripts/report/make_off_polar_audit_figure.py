import argparse,os,json,csv,shutil
from pathlib import Path
R=Path(__file__).resolve().parents[2]/'docs/results/20261027_off_polar_audit'
parser=argparse.ArgumentParser();parser.add_argument('--outdir',type=Path,required=True);args=parser.parse_args()
out=args.outdir;out.mkdir(parents=True,exist_ok=False)
cache=out/'matplotlib-cache';os.environ['MPLCONFIGDIR']=str(cache)
import numpy as np
import matplotlib;matplotlib.use('Agg')
from matplotlib import pyplot as plt,font_manager
for path in ('/home/qiuyong/fonts/arial.ttf','/home/qiuyong/fonts/ARIALBD.TTF'):
 if Path(path).exists():font_manager.fontManager.addfont(path)
plt.rcParams.update({'font.family':'Arial' if Path('/home/qiuyong/fonts/arial.ttf').exists() else 'DejaVu Sans','font.size':8,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7})
report=json.loads((R/'energy_ess_audit.json').read_text());spatial=json.loads((R/'spatial_audit.json').read_text())
rows=list(csv.DictReader((R/'independent_weights.csv').open()))
t=np.array([float(r['time_ps']) for r in rows]);w=np.array([float(r['weight']) for r in rows])
fig,axes=plt.subplots(1,2,figsize=(7.09,2.65),layout='constrained')
a,b=axes;a.vlines(t,0,w,color='#0072B2',lw=.65);a.set(xlabel='Time (ps)',ylabel='Normalized weight',xlim=(-5,505),ylim=(0,.65))
a.text(.04,.94,f"ESS = {report['ess_independent']:.2f} / 501\nTop 3 frames: {report['top3_mass']:.1%}",transform=a.transAxes,va='top',fontsize=8)
z=spatial['spatial']['z'];n=z['bins'];L=spatial['box_A'][2];positions=(np.arange(n)+.5)*L/n
area=np.prod(spatial['box_A'][:2]);conversion=area*L/n
b.plot(positions,np.array(z['mean_bin_occupancy'])/conversion,color='#0072B2',lw=1.3,label='OFF time average')
b.plot(positions,np.array(z['weighted_mean_bin_occupancy'])/conversion,color='#D55E00',lw=1.1,ls='--',label='Reweighted estimate')
b.axhline(spatial['number_density_A3'],color='.4',lw=.8,ls=':',label='Box mean')
b.set(xlabel=r'$z$ (Å)',ylabel=r'O-site density (Å$^{-3}$)',xlim=(0,L),ylim=(0,.045));b.legend(frameon=False,fontsize=7,loc='upper right')
for label,ax in zip(('a','b'),axes):ax.text(-.15,1.04,label,transform=ax.transAxes,fontweight='bold',fontsize=11)
for fmt in ('pdf','svg','png'):fig.savefig(out/('ESS_and_density.'+fmt),dpi=600)
plt.close(fig)
(out/'CAPTION.txt').write_text('All501 supplied frames. Weights use300K and whole-cell energy gaps. O-site z densities use identical frames and periodic box; weighted curve is not independently validated POLAR sampling. No error bars: temporal correlations and equilibrium uncertainty were not estimated.\n')
assert cache.parent==out and cache.name=='matplotlib-cache'
if cache.exists():shutil.rmtree(cache)
print(out)
