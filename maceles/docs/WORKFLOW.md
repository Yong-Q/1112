# Workflow

## 0. Environment

```bash
python -m pip install -e ".[mace,test]"
export GRAPH_LONG_RANGE_PATH=/path/to/graph_electrostatics-0.4.0
```

Use a physical structure.  Do not use a synthetic speed-test box without
checking the minimum image distance: one 1,296-atom "water" file in the
Symmetrix-Polar workspace has a 0.22 Å contact and produces ~6.8e6 eV/Å
MACE-OFF forces.

## 1. Source trajectory

```bash
off-polar-run \
  --structure water96.xyz \
  --model MACE-OFF23_medium.model \
  --out off_traj.xyz \
  --temperature 300 --ps 5 --equil-ps 0.5 --frame-ps 0.1
```

Recommended production settings:

- 1 ns at 1 fs: save every 1 ps (1,000 frames) or 0.5 ps (2,000 frames);
- discard at least 0.5–1 ps of equilibration;
- check the instantaneous temperature and the RDF before labeling.

## 2. Label and diagnose

```bash
off-polar-reweight \
  --traj off_traj.xyz \
  --source-model MACE-OFF23_medium.model \
  --polar-model MACE-POLAR-1-M.model \
  --temperature 300 --outdir out/
```

Outputs:

| File | Content |
|---|---|
| `reweight_manifest.json` | model hashes, ΔV, σ_β, ESS, c1/c2, verdict |
| `reweight_frames.csv` | per-frame energies, weights, force MAE |
| `reweight_diagnostics.png` | ΔV, weight collapse, c1/c2, RDF |

## 3. Decision gate

| Diagnostic | Action |
|---|---|
| `verdict = usable` (σ_β ≤ 3, ESS ≥ 10 %) | report reweighted ensemble averages |
| `verdict = marginal` (3 < σ_β ≤ 5) | use Gaussian/mean-gap corrections for free-energy profiles, or BAR/MBAR; do not report direct ensemble means |
| `verdict = collapse` (σ_β > 5, ESS < 1 %) | do not reweight; run short target trajectories or use a distilled model for long-time statistics |

A constant `c2` offset cancels.  A global `c1` is an effective-temperature
rescaling only; the residual must stay in the weight.

## 4. Production estimate

| Item | 1,296 atoms | 5,184 atoms |
|---|---:|---:|
| Native PyTorch MACE-OFF, 1 ns at 0.5 fs | ~2.5 days | ~10 days |
| Linear engine MACE-OFF-like, 1 ns at 1 fs | ~1.6 h | ~6.4 h |
| MACE-POLAR labels, 2,000 frames (exact engine) | ~6 min | ~22 min |
| MACE-POLAR labels, 2,000 frames (PyTorch) | ~25 min | ~1.7 h |

These are estimates; measure on the target hardware.  The 1 ns source
trajectory is usually the bottleneck, not the target labeling.

## 5. Embedding

The package is calculator-agnostic:

```python
from sampling.off_polar import OffPolarReweighter, ReweightConfig
rw = OffPolarReweighter(source_calc=my_ase_calc,
                        target_calc=my_target_calc,
                        config=ReweightConfig(temperature_K=300.0))
result = rw.evaluate(frames)
```

The manifest (`schema_version`, model hashes, ESS, verdict) is designed to be
attached to a model manifest or a workflow record.
