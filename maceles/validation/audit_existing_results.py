"""CPU-only audit of the published reweighting evidence; no new MD or MLIP calls.

The JSON is an audit artifact, not a corrected c1/c2 dataset. Existing source
files and scientific results are never overwritten by this command.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from unittest.mock import patch
import sys

import numpy as np
import ase.io
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sampling.off_polar import OffPolarReweighter
from sampling.direct_correlation import c1_from_widom

KB_EV_K = 8.617333262e-5


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def weight_summary(gaps: np.ndarray, temperature: float) -> dict:
    beta = 1.0 / (KB_EV_K * temperature)
    logw = -beta * (gaps - gaps.min())
    weights = np.exp(logw)
    weights /= weights.sum()
    return {
        'temperature_K': temperature, 'frames': len(weights),
        'sigma_beta': float((beta * gaps).std(ddof=1)),
        'ess_weight_only': float(1.0 / np.dot(weights, weights)),
        'ess_fraction': float(1.0 / np.dot(weights, weights) / len(weights)),
        'max_weight': float(weights.max()),
        'top3_weight': float(np.sort(weights)[-3:].sum()),
        'top5_weight': float(np.sort(weights)[-5:].sum()),
    }


class ZeroEnergy(Calculator):
    implemented_properties = ['energy']

    def calculate(self, atoms=None, properties=('energy',), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.results = {'energy': 0.0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest_path = ROOT / 'examples/water96/reweight/reweight_manifest.json'
    traj_path = ROOT / 'examples/water96/off_traj_96_prod.xyz'
    manifest = json.loads(manifest_path.read_text())
    records = manifest['frames']
    gaps = np.array([row['dE_total_eV'] for row in records])
    saved_weights = np.array([row['weight'] for row in records])
    frames = list(ase.io.iread(traj_path, format='extxyz'))
    if len(frames) != len(records):
        raise ValueError('Trajectory and reweight manifest have different frame counts')
    temperatures = np.array([frame.get_temperature() for frame in frames])
    blocks = []
    for number, indices in enumerate(np.array_split(np.arange(len(frames)), 4)):
        blocks.append({
            'block': number, 'frames': len(indices),
            'first_step': int(frames[int(indices[0])].info['step']),
            'last_step': int(frames[int(indices[-1])].info['step']),
            'kinetic_temperature_mean_K': float(temperatures[indices].mean()),
            'energy_gap_mean_eV': float(gaps[indices].mean()),
            'saved_weight_mass': float(saved_weights[indices].sum()),
        })
    rescaled = np.exp(-(gaps - gaps.min()) / (KB_EV_K * manifest['temperature_K']))
    rescaled /= rescaled.sum()

    # Actual RDF code on a synthetic uniform gas, strictly inside L/2.
    rng = np.random.default_rng(20261023)
    ideal_frames = [Atoms('O32', positions=rng.uniform(0, 20, (32, 3)),
                         cell=[20, 20, 20], pbc=True) for _ in range(80)]
    radii, current_g = OffPolarReweighter.rdf(
        ideal_frames, pair=('O', 'O'), rmax=6., bins=30)
    edges = np.linspace(0, 6, 31)
    current_shell = 4 * np.pi / 3 * (edges[1:]**2 - edges[:-1]**2)
    correct_shell = 4 * np.pi / 3 * (edges[1:]**3 - edges[:-1]**3)
    # Upstream 865f203 fixed the shell. Reconstruct the original defect only
    # for an audit comparison; do not apply the correction twice.
    legacy_g = current_g * correct_shell / current_shell
    middle = (radii > 2.) & (radii < 5.)

    # Four attempted insertions: two explicitly assumed zero factors, two unity.
    # A Widom average over all four factors must be log((0+1+0+1)/4).
    host = Atoms('He', positions=[[0, 0, 0]], cell=[10, 10, 10], pbc=True)
    probe = Atoms('He', positions=[[0, 0, 0]])
    distances = [np.array([[0., d], [d, 0.]]) for d in (.5, 2., .5, 2.)]
    with patch.object(Atoms, 'get_all_distances', side_effect=distances):
        widom = c1_from_widom([host], ZeroEnergy(), probe, 300., n_insertions=4)

    tracked = [manifest_path, traj_path, ROOT/'sampling/off_polar.py',
               ROOT/'sampling/direct_correlation.py', ROOT/'sampling/trajectory.py',
               ROOT/'scripts/monitor_ess.py', ROOT/'docs/results/ess_live_1152.json']
    report = {
        'kind': 'existing_evidence_audit_not_corrected_labels',
        'audited_baseline_commit': 'af743290aa844937711884d791d5aa28b22bf636',
        'current_rdf_fix_commit': '865f203ee8cf73c534f245dc3466e1a638e03707',
        'input_sha256': {p.relative_to(ROOT).as_posix(): digest(p) for p in tracked},
        'weight_replay': {
            'stored_temperature_K': manifest['temperature_K'],
            'stored_ess': manifest['ess'],
            'maximum_saved_weight_difference': float(np.max(abs(rescaled-saved_weights))),
            'at_300K': weight_summary(gaps, 300.),
            'at_334K': weight_summary(gaps, 334.),
            'autocorrelation_adjusted': False,
        },
        'trajectory': {
            'frames': len(frames), 'first_step': int(frames[0].info['step']),
            'last_step': int(frames[-1].info['step']),
            'metadata_temperatures_K': sorted({float(a.info['temperature_K']) for a in frames}),
            'kinetic_temperature_method': 'ASE get_temperature using stored momenta and masses',
            'kinetic_temperature_mean_K': float(temperatures.mean()),
            'kinetic_temperature_sd_K': float(temperatures.std(ddof=1)),
            'four_equal_time_blocks': blocks,
            'warning': 'Kinetic block means are descriptive, not proof of a canonical ensemble or a new reweighting temperature.',
        },
        'rdf_counterexample': {
            'uniform_random_gas_frames': 80, 'atoms_per_frame': 32, 'box_A': 20.,
            'rmax_A': 6., 'expected_g': 1.,
            'current_mean_g_for_2_to_5A': float(current_g[middle].mean()),
            'legacy_squared_shell_mean_g_for_2_to_5A': float(legacy_g[middle].mean()),
            'current_shell_formula': '4*pi/3*(r_hi**3-r_lo**3)',
            'required_shell_volume_formula': '4*pi/3*(r_hi**3-r_lo**3)',
            'archived_rmax_A': 8.,
            'archived_half_box_A': float(frames[0].cell.lengths().min()/2),
            'not_a_corrected_c2_result': True,
        },
        'widom_denominator_counterexample': {
            'attempted': 4, 'retained': widom['n_used'],
            'current_c1': widom['c1_ex'],
            'required_c1_if_rejected_factors_are_zero': float(np.log(.5)),
            'nonfinite_insertion_energies_must_be_errors_not_silent_rejections': True,
        },
        'additional_code_findings': [
            'c2_from_g replaces a nonzero first radial midpoint by rpad=0 before transforming h.',
            'O-O scalar OZ output is not a full molecular position-orientation c2; the target coordinate must be explicit.',
            'Widom insertion does not declare an isolated-molecule/ideal-internal reference and has no frame-weight argument.',
            'The 1152-atom record contains only six early frames including timestep zero; no stationary-ensemble conclusion is established.',
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps({k: report[k] for k in
                      ('weight_replay', 'rdf_counterexample', 'widom_denominator_counterexample')},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
