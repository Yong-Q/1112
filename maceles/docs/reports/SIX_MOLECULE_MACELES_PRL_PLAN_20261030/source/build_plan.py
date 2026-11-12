#!/usr/bin/env python3
"""Build the six-molecule budget; never run MD or replace an existing output.

python source/build_plan.py --outdir /a/new/directory
Then compile report.tex twice with XeLaTeX in that directory.
"""
import argparse
import csv
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
NA = 6.02214076e23


def write_csv(path, rows):
    with path.open('x', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        w.writeheader()
        w.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outdir', type=Path, required=True)
    args = parser.parse_args()
    out = args.outdir
    out.mkdir(parents=True, exist_ok=False)
    cfg = json.loads((HERE / 'plan_inputs.json').read_text())
    constants = json.loads((ROOT / 'evidence/critical_constants.json').read_text())
    low, high = Fraction(str(cfg['rho_star_min'])), Fraction(str(cfg['rho_star_max']))
    coarse = [low + (high-low)*i/9 for i in range(10)]
    mids = [(a+b)/2 for a,b in zip(coarse[:-1],coarse[1:])]
    fine = sorted(coarse + mids + [(low+mids[0])/2])
    assert len(set(fine)) == 20 and set(coarse).issubset(fine)
    assert len(set(fine)-set(coarse)) == 10
    species, all_states = [], []
    grid = [{'id':i+1, 'rho_star':float(v), 'stage':'pilot' if v in coarse else 'extension'}
            for i,v in enumerate(fine)]
    for c in constants:
        nc_nm3 = c['rhoc_mol_m3'] * NA / 1e27
        sigma_nm3 = cfg['reference_LJ_rhoc_star'] / nc_nm3
        sigma_A = (sigma_nm3 ** (1/3))*10
        temp = cfg['target_T_star_reference'] * c['Tc_K'] / cfg['reference_LJ_Tc_star']
        nmol = cfg['atoms_per_box'] // c['atoms_per_molecule']
        assert nmol*c['atoms_per_molecule'] == cfg['atoms_per_box']
        ns = [float(v)/sigma_nm3 for v in fine]
        masses = [n*c['molar_mass_kg_mol']*1e24/NA for n in ns]
        lengths = [(nmol/n)**(1/3)*10 for n in ns]
        sp = {**c,'T_plan_K':temp,'sigma_reference_A':sigma_A,'molecules':nmol,
              'atoms':cfg['atoms_per_box'],'rho_min_nm3':ns[0], 'rho_max_nm3':ns[-1],
              'mass_density_min_g_cm3':masses[0], 'mass_density_max_g_cm3':masses[-1],
              'box_L_min_A':min(lengths), 'box_L_max_A':max(lengths),
              'pilot_spacing_nm3':float((high-low)/9)/sigma_nm3,
              'expanded_max_spacing_nm3':float((high-low)/18)/sigma_nm3,
              'expanded_min_spacing_nm3':float((high-low)/36)/sigma_nm3,
              'time_production_h_at_reserve':cfg['planning_reserve_us_atom_step']*cfg['atoms_per_box']*1e-6*(1e6/cfg['timestep_fs'])/3600}
        species.append(sp)
        for g,n,m,L in zip(grid,ns,masses,lengths):
            assert math.isclose(n*(L/10)**3,nmol,rel_tol=1e-12)
            all_states.append({'molecule':c['formula'], 'density_id':g['id'],
                  'stage':g['stage'],'status':'planned_not_run','T_plan_K':temp,
                  'rho_star_reference':g['rho_star'], 'rho_molecules_nm3':n,
                  'rho_molecules_A3':n/1000, 'rho_mass_g_cm3':m,
                  'sigma_reference_A':sigma_A,'box_L_A':L,
                  'molecules':nmol,'atoms':cfg['atoms_per_box'],
                  'production_ns':cfg['production_ns'],
                  'equilibration_ns_budget':cfg['equilibration_ns_budget'],
                  'timestep_fs':cfg['timestep_fs'],
                  'external_field':'none_periodic_bulk',
                  'temperature_density_status':'corresponding_states_proposal_not_model_calibration'})
    timing = []
    for speed in cfg['runtime_scenarios_us_atom_step']:
        h_ns = speed*cfg['atoms_per_box']*1e-6*(1e6/cfg['timestep_fs'])/3600
        for count in (10,20):
            total = count*len(species)
            h = total*(cfg['production_ns']+cfg['equilibration_ns_budget'])*h_ns
            timing.append({'rate_us_atom_step':speed,'rate_status':'toy_reference_pre_fix' if speed==6.34 else 'scenario_not_measurement',
                 'states_per_species':count,'total_trajectories':total,
                 'production_ns':total*cfg['production_ns'],
                 'equilibration_ns_budget':total*cfg['equilibration_ns_budget'],
                 'h_per_production_ns':h_ns,'total_gpu_h':h,
                 'ideal_2gpu_days':h/48,'ideal_3gpu_days':h/72,
                 'gpu_h_with_25pct_buffer':h*1.25,'3gpu_days_with_25pct_buffer':h*1.25/72})
    outputs = {'inputs':cfg,'species':species,'density_grid':grid,'timing':timing,
               'decision':'10 states/species pilot; reserve up to 20 with 10 added states/species; no sufficiency or exact PRL equivalence claim.'}
    (out/'budget.json').write_text(json.dumps(outputs,ensure_ascii=False,indent=2)+'\n')
    write_csv(out/'states_20_per_molecule.csv',all_states)
    write_csv(out/'states_10_per_molecule.csv',[x for x in all_states if x['stage']=='pilot'])
    write_csv(out/'runtime_budget.csv',timing)
    write_csv(out/'molecule_summary.csv',species)
    grid_rows = []
    for g in grid:
        row = {'id':g['id'],'stage':g['stage'],'rho_star_reference':g['rho_star']}
        row.update({s['formula']+'_molecules_nm3':g['rho_star']/(s['sigma_reference_A']/10)**3 for s in species})
        grid_rows.append(row)
    write_csv(out/'density_grid.csv',grid_rows)
    sep = ' & '
    sptex = '\n'.join(sep.join([s['formula'],str(s['molecules']),f"{s['T_plan_K']:.1f}",
                  f"{s['sigma_reference_A']:.3f}",f"{s['rho_min_nm3']:.3f}--{s['rho_max_nm3']:.3f}",
                  f"{s['box_L_min_A']:.1f}--{s['box_L_max_A']:.1f}"])+r' \\' for s in species)
    mass_tex = '\n'.join(sep.join([s['formula'],f"{s['mass_density_min_g_cm3']:.4f}--{s['mass_density_max_g_cm3']:.4f}",
                  f"{s['pilot_spacing_nm3']:.3f}", f"{s['expanded_max_spacing_nm3']:.3f}"])+r' \\' for s in species)
    gridtex = '\n'.join(sep.join([str(g['id']),'首批' if g['stage']=='pilot' else '补点',f"{g['rho_star_reference']:.6f}"]+
                  [f"{g[s['formula']+'_molecules_nm3']:.3f}" for s in species])+r' \\' for g in grid_rows)
    timetex='\n'.join(sep.join([f"{t['rate_us_atom_step']:.2f}",str(t['states_per_species']),
                  f"{t['h_per_production_ns']:.2f}",f"{t['total_gpu_h']:.1f}",f"{t['ideal_3gpu_days']:.1f}",
                  f"{t['3gpu_days_with_25pct_buffer']:.1f}"])+r' \\' for t in timing)
    text=(HERE/'report.template.tex').read_text()
    for key,value in {'SPECIES':sptex,'MASS':mass_tex,'GRID':gridtex,'TIME':timetex}.items():
        text=text.replace('@@'+key+'@@',value)
    assert '@@' not in text
    (out/'report.tex').write_text(text)
    md = '# 六分子 MACELES 均相采样：PRL 范围、密度清单与时间预算\n\n'
    md += '**建议：每种先采 10 个密度点；预算按 20 点预留。六种共 60 / 120 条，每条 1 ns 生产，另预留 0.1 ns 平衡。** 这是一条等温线的探索清单，没有提交 MD，也未证明 20 点足以重现 PRL 精度。\n\n'
    md += '[下载完整 PDF](report.pdf) · [首批 60 条 CSV](states_10_per_molecule.csv) · [扩展后 120 条 CSV](states_20_per_molecule.csv) · [密度大表](density_grid.csv) · [时间预算](runtime_budget.csv) · [全部参数](budget.json)\n\n'
    md += '## PRL 约化量与真实分子的换算\n\n'
    md += 'PRL 正文第 3 页是 **T* = kB T / ε = 2，βε = 1/2**；不是 T*=1/2，也不是 T=2Tc。体相训练密度为 **ρ*=ρσ³=0.02–0.67**，GCMC、零外场、800 组密度/关联组合。[原论文](https://doi.org/10.1103/PhysRevLett.134.056103)。\n\n'
    md += 'MACELES 没有唯一 ε、σ。为得到可预算的物理数值，本表**提出**对应态约定：采用完整 LJ 的参考临界量 T*c=1.312、ρ*c=0.316，用真实流体 EOS 的 Tc、ρc 定义 ε_ref/kB=Tc/1.312、σ_ref³=0.316/ρc。于是 T_plan=2Tc/1.312、ρ=ρ*/σ_ref³。这不是 PRL 截断 4σ 势的精确临界映射，也不是 MACELES 的拟合参数；真实状态表均标记为候选。参考：[NIST LJ 定义与临界常数](https://www.nist.gov/mml/csd/chemical-informatics-group/lennard-jones-fluid-properties)。\n\n'
    md += '| 分子 | 分子数 / 3000原子 | 候选温度/K | σ_ref/Å | 分子密度/nm⁻³ | 立方盒边长/Å |\n|---|---:|---:|---:|---:|---:|\n'
    for s in species:
        md += f"| {s['formula']} | {s['molecules']} | {s['T_plan_K']:.1f} | {s['sigma_reference_A']:.3f} | {s['rho_min_nm3']:.3f}–{s['rho_max_nm3']:.3f} | {s['box_L_min_A']:.1f}–{s['box_L_max_A']:.1f} |\n"
    md += '\n10 点均匀间隔 Δρ*=0.0722222。20 点保留这 10 点，插入 9 个中点，再在最低密度区加 1 点；最大间隔 0.0361111，最小间隔 0.0180556。换成真实密度的完整 20×6 大表和每个盒子的边长见 CSV/PDF。**只需新增 10 点/分子即可从首批扩至 20 点。**\n\n'
    md += '## 时间：已测速度与正式模型预算分开\n\n'
    md += '源仓最新力修正记录表明：正式 MACELES-OFF small 为 **192 通道**，补上电荷响应力；六分子 3000 原子正式速度仍未公布。此前持续速率 6.34 μs/atom/step 来自 **128 通道、一轮训练的试验模型、水、1500 原子、修正前**，仅作速度参照，不能当成正式模型生产耗时。[力修正](evidence/MACELES_FORCE_FIX_2026-10-30.md) · [原测速](evidence/MACELES_Q4_VERIFY_2026-10-30.md)。本报告没有采用原 README 的学生势 ~20.2 估算代替 MACELES 实测。\n\n'
    md += '| 速率/μs·atom⁻¹·step⁻¹ | 每分子轨迹数 | 3000原子1ns/h | 含平衡总GPU时 | 3卡理想天数 | 3卡加25%余量/天 |\n|---|---:|---:|---:|---:|---:|\n'
    for t in timing:
        md += f"| {t['rate_us_atom_step']:.2f} | {t['states_per_species']} | {t['h_per_production_ns']:.2f} | {t['total_gpu_h']:.1f} | {t['ideal_3gpu_days']:.1f} | {t['3gpu_days_with_25pct_buffer']:.1f} |\n"
    md += '\n**6.34 是旧试验测速参照；10/20/40 都是预算情景，不是正式模型测速或置信边界。** 暂按 20 μs/atom/step 预留：首批 60 条约 **2200 GPU时**，扩展后 120 条约 **4400 GPU时**；加25%余量为2750/5500 GPU时。若3张同速GPU并行，分别约38.2/76.4天。生产模型速度确定后按比例重算，队列、教师单点和失败重跑另计。0.5 fs 共2,000,000步/ns；1 fs能通过稳定性检查时上述 MD 预算才可减半。\n\n'
    md += '## 数据与标签\n\n'
    md += '- 全周期、纯组分、零外场 NVT；各密度单独改变盒体积。每条1 ns生产，平衡暂预留100 ps，以稳定性检查决定是否延长。\n- 每0.05 ps在线累积质心 g(r)、非零波矢 S(k)：20,000次/条；每1 ps保存完整帧：1000帧/条。温度/外场后处理复用同一批帧，不另算轨迹；不同平均体相密度仍需独立采样。\n- 首批共60,000保存帧、120万次在线结构统计；20点共120,000保存帧、240万次在线结构统计。这些有时间相关性，不是独立样本数，也不是同等数量的训练状态。\n- 若改用POLAR作目标，每5 ps评分仅是200帧/条的起始方案，共12,000/24,000次目标单点；若全部保存帧评分，则60,000/120,000次。教师和自洽外场评分时间未测，不包含在 MD 总时中。\n- 均相各向同性质心 g(r) 经 OZ 得到标量 c2；均匀密度本身不能给绝对 c1，需 μ_ex 或密度积分基准。取向分辨 c1/c2 需额外角关联。\n- 先检查均相性、分子完整性、分块结构误差及密度插值；训练/测试按整个源状态分组。20个密度状态不等于PRL的800组，是否需要加点由留出状态与结构误差决定。\n\n'
    md += '## 复现和证据\n\n'
    md += '```bash\npython source/build_plan.py --outdir /new/output/directory\ncd /new/output/directory\nxelatex -interaction=nonstopmode -halt-on-error report.tex\nxelatex -interaction=nonstopmode -halt-on-error report.tex\n```\n\n'
    md += '需要Python标准库、XeLaTeX及ctex字体；当前服务器字体自动使用/home/qiuyong/fonts。计算常数、原文页码和版本见 [evidence](evidence/)；配置见 [plan_inputs.json](source/plan_inputs.json)。原有归档和PDF保留。该目录不含作业脚本、大轨迹或模型权重。\n'
    (out/'README.md').write_text(md)
    manifest = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file()}
    (out/'generated_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'states':len(all_states),'pilot':sum(x['stage']=='pilot' for x in all_states),
                      'outdir':str(out),'production_ns_full':sum(x['production_ns'] for x in all_states),
                      'reserve_h': [x for x in timing if x['rate_us_atom_step']==20]},indent=2))


if __name__ == '__main__':
    main()
