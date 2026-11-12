#!/usr/bin/env python3
"""Conditional trajectory and static target-evaluation budget."""
import csv,json
from pathlib import Path
P=Path(__file__).resolve().parents[2];sp=['H2O','NH3','CH4','C2H4','C2H6','CO2'];rate=38.328
# Reuse one base source trajectory across candidate fields/T; new density anchors
# and independent replicas count as new trajectories. Validation remains gated.
base={'density_states_per_species':1,'replicas_per_state':2,'equilibration_ps_per_trajectory':200,'production_ps_per_trajectory':1000,'dump_interval_ps':1}
expanded={**base,'density_states_per_species':3}
def calc(x):
 n=x['density_states_per_species']*x['replicas_per_state'];ns=n*(x['equilibration_ps_per_trajectory']+x['production_ps_per_trajectory'])/1000;return {'source_trajectories_per_species':n,'source_total_ns_per_species':ns,'production_frames_per_species':n*x['production_ps_per_trajectory']//x['dump_interval_ps'],'target_singlepoint_every_dump_per_species':n*x['production_ps_per_trajectory']//x['dump_interval_ps'],'target_singlepoint_100_per_trajectory_per_species':n*100}
B=calc(base);X=calc(expanded);assert B['source_trajectories_per_species']==2 and X['source_trajectories_per_species']==6
out={'species':sp,'basic_parameters':base,'basic_per_species':B,'basic_six_species_totals':{'source_trajectories':12,'source_ns':14.4,'raw_frames':12000,'target_SP_for_all_raw_frames':12000},
 'expanded_parameters':expanded,'expanded_per_species':X,'expanded_six_species_totals':{'source_trajectories':36,'source_ns':43.2,'raw_frames':36000,'target_SP_for_all_raw_frames':36000,'target_SP_100_per_trajectory':3600},
 'proposed_target_validation':{'independent_POLAR_reference_trajectories':6,'per_trajectory_equilibration_ps':100,'per_trajectory_production_ps':200,'nominal_total_ns':1.8,'basic_source_plus_target_trajectories':18,'note':'Tentative short target references; cost unmeasured and not claimed sufficient.'},
 'existing_water':'One OFF23 water 0.5 ns trajectory is a slab diagnostic; matching uniform state/continuation must be validated, and original restart is not in test repository. Counted separately, not assumed to replace a planned trajectory.',
 'reuse_assumption':'Candidate fields and temperatures reuse the same source frames only when target energy evaluation, ensemble support and observable error validation permit; generating 500 fields creates 500 correlated reweighted profiles, not independent trajectories. NVT fixed N,V does not create independently sampled bulk densities.',
 'timing_reference':{'water_1152atom_OFF23_FP32_1fs_GPU_model_unverified_h_per_ns':rate,'reference_log':'test/validation/off23_500ps_water1152_2026-10-23/log.production_1fs','simple_basic_scenario_gpu_hours':rate*B['source_total_ns_per_species']*6,'simple_expanded_scenario_gpu_hours':rate*X['source_total_ns_per_species']*6,'simple_expanded_3gpu_days':rate*X['source_total_ns_per_species']*6/72,'scope':'same per-step speed assumed for all species, not measured; no queue, potential singlepoint or repeated field-dependent POLAR evaluations included'},
 'target_SP_notes':'q for each frame needs target energy; reusing stored q for 1000 frames means 1000 target SP per trajectory, not 100. 100 SP/trajectory is an alternative, giving only 100 scored frames. If POLAR self-consistent external response changes with V, recompute target energies per field. Teacher force for IBP must be separately available.'}
with (P/'budget.json').open('x') as f:json.dump(out,f,indent=2)
with (P/'six_molecule_budget.csv').open('x',newline='') as f:
 w=csv.writer(f,lineterminator='\n');w.writerow(['species','basic_density_states','replicas_per_state','basic_trajectories','equil_ps_per_trajectory','production_ps_per_trajectory','basic_source_ns','basic_saved_frames','basic_target_SP_all_frames','expanded_density_states','expanded_trajectories','expanded_source_ns','expanded_saved_frames','expanded_target_SP_all_frames'])
 for s in sp:w.writerow([s,1,2,B['source_trajectories_per_species'],200,1000,B['source_total_ns_per_species'],B['production_frames_per_species'],B['target_singlepoint_every_dump_per_species'],3,X['source_trajectories_per_species'],X['source_total_ns_per_species'],X['production_frames_per_species'],X['target_singlepoint_every_dump_per_species']])
print(json.dumps(out,indent=2))
