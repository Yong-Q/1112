#!/usr/bin/env python3
"""Reproduce vector diagrams and measured data plots from archived evidence."""
import argparse,csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch

P=Path(__file__).resolve().parents[2];D=P/'evidence'
parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--outdir',type=Path,default=P/'figures');parser.add_argument('--graph',type=Path,default=P/'figures/source/graph_data.json');args=parser.parse_args();O=args.outdir;O.mkdir(parents=True,exist_ok=True)
font_manager.fontManager.addfont('/home/qiuyong/fonts/1611458310630572.ttf');font_manager.fontManager.addfont('/home/qiuyong/fonts/arial.ttf')
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Microsoft YaHei','Arial','DejaVu Sans'],'pdf.fonttype':42,'svg.fonttype':'none','font.size':9,'axes.spines.top':False,'axes.spines.right':False})
colors={'done':('#E6F3F0','#23796F'),'partial':('#FFF3DF','#B7791F'),'todo':('#FBEAEA','#BC3B38')}
def save(fig,name):
 for suffix in ('pdf','svg','png'):
  path=O/(name+'.'+suffix)
  with path.open('xb') as f:fig.savefig(f,format=suffix,dpi=500 if suffix=='png' else None,bbox_inches='tight')
 plt.close(fig)
def flow(g):
 fig,ax=plt.subplots(figsize=(14.8,6.6));ax.set_xlim(0,1550);ax.set_ylim(660,0);ax.axis('off');nodes={n['id']:n for n in g['nodes']}
 for x,y in g['edges']:
  aa=nodes[x];bb=nodes[y];dx=bb['x']-aa['x'];dy=bb['y']-aa['y']
  if abs(dx)>abs(dy):p1=(aa['x']+260 if dx>0 else aa['x'],aa['y']+55);p2=(bb['x'] if dx>0 else bb['x']+260,bb['y']+55)
  else:p1=(aa['x']+130,aa['y']+110 if dy>0 else aa['y']);p2=(bb['x']+130,bb['y'] if dy>0 else bb['y']+110)
  if dy!=0 and dx!=0:style='arc3,rad=0.12'
  else:style='arc3,rad=0'
  ax.add_patch(FancyArrowPatch(p1,p2,arrowstyle='-|>',mutation_scale=10,lw=.9,color='#8190A0',connectionstyle=style,zorder=1))
 for n in g['nodes']:
  fill,edge=colors[n['state']];ax.add_patch(FancyBboxPatch((n['x'],n['y']),260,110,boxstyle='round,pad=0.02,rounding_size=8',facecolor=fill,edgecolor=edge,lw=1.5,zorder=2));
  ax.text(n['x']+130,n['y']+45,n['label'],ha='center',va='center',fontsize=12,color='#1d2934',zorder=3,linespacing=1.35)
  ax.text(n['x']+130,n['y']+88,{'done':'已完成','partial':'部分完成','todo':'待完成 / 未验证'}[n['state']],ha='center',va='center',fontsize=9,color=edge,zorder=3)
 fig.suptitle(g['title'],fontsize=16,y=.98);fig.text(.5,.045,'绿色：已完成  ·  黄色：部分完成  ·  红色：待完成或未验证',ha='center',fontsize=10,color='#43515c');fig.subplots_adjust(left=.025,right=.985,top=.92,bottom=.10);save(fig,'01_research_flow' if g['id']=='roadmap' else '02_sampling_flow')
graph=json.loads(args.graph.read_text())
for g in graph['graphs']:flow(g)
valid=json.loads((D/'teacher_student_by_species.json').read_text());species=['H2O','NH3','CH4','C2H4','C2H6','CO2'];F=[valid[f'cross:{x}|bulk']['mae_f_meV_per_A'] for x in species];E=[valid[f'cross:{x}|bulk']['mae_e_meV_per_atom'] for x in species];ns=[valid[f'cross:{x}|bulk']['n_frames'] for x in species]
fig,ax=plt.subplots(1,2,figsize=(7.2,3.0),layout='constrained');x=np.arange(len(species));ax[0].bar(x,E,color='#4d9b91');ax[1].bar(x,F,color='#567da8')
for a,values in zip(ax,[E,F]):
 a.set_xticks(x,species);a.set_ylim(0,max(values)*1.24)
 for i,v in enumerate(values):a.text(i,v+max(values)*.025,f'{v:.2f}',ha='center',va='bottom',fontsize=7)
 a.tick_params(axis='x',length=0)
ax[0].set_ylabel('能量 MAE (meV / atom)');ax[1].set_ylabel('力分量 MAE (meV / Å)');ax[0].set_title('a   留出体相：能量',loc='left');ax[1].set_title('b   留出体相：力',loc='left');fig.suptitle('Plan D 与 POLAR 教师的六分子逐帧对比',fontsize=10)
save(fig,'03_six_molecule_student_errors')
with (D/'off_polar_weights_501.csv').open() as f:rr=list(csv.DictReader(f))
time=np.array([float(r['time_ps']) for r in rr]);weight=np.array([float(r['weight']) for r in rr]);assert len(weight)==501 and np.isclose(weight.sum(),1)
rho=np.genfromtxt(D/'density_estimator_profiles.csv',delimiter=',',names=True)
fig,ax=plt.subplots(2,1,figsize=(7.2,4.7),layout='constrained');ax[0].scatter(time,weight,s=10,color='#597da5',alpha=.85);ax[0].axvline(20,color='#ba5953',lw=1,ls='--',label='20 ps 截断（诊断）');ax[0].set_yscale('log');ax[0].set_ylabel('归一化权重');ax[0].set_xlabel('生产时间 (ps)');ax[0].legend(fontsize=8);ax[0].set_title('a   OFF → POLAR 逐帧权重',loc='left');
ax[1].plot(rho['z_A'],rho['source'],color='#5e6671',label='OFF23 等权');ax[1].plot(rho['z_A'],rho['direct'],color='#287bad',label='POLAR 重加权估计');ax[1].plot(rho['z_A'],rho['smoothed'],color='#54a295',ls='--',label='加权后平滑');ax[1].set_xlabel('z (Å)');ax[1].set_ylabel(r'质心密度 ($\mathrm{\AA}^{-3}$)');ax[1].set_title('b   一条水轨迹的质心密度诊断',loc='left');ax[1].legend(fontsize=7,ncol=3);save(fig,'04_water_reweighting_diagnostic')
print('PASS: two editable flowcharts and two evidence-based figures rendered to PDF/SVG/500dpi PNG.')
