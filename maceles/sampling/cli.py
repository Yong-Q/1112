"""Console entry points for the off-polar trajectory and reweighting workflow."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from sampling.direct_correlation import direct_correlation_report
from sampling.off_polar import GRAPH_LONG_RANGE, OffPolarReweighter, ReweightConfig
from sampling.trajectory import TrajectoryConfig, read_structure, run_langevin


def run_main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="off-polar-run")
    ap.add_argument("--structure", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--temperature", type=float, default=300.0)
    ap.add_argument("--timestep-fs", type=float, default=0.5)
    ap.add_argument("--ps", type=float, default=5.0)
    ap.add_argument("--equil-ps", type=float, default=0.5)
    ap.add_argument("--frame-ps", type=float, default=0.1)
    ap.add_argument("--friction", type=float, default=0.01)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    ap.add_argument("--seed", type=int, default=20261023)
    args = ap.parse_args(argv)

    from mace.calculators import MACECalculator
    atoms = read_structure(args.structure)
    calc = MACECalculator(model_paths=args.model, device=args.device,
                          default_dtype=args.dtype)
    cfg = TrajectoryConfig(
        temperature_K=args.temperature, timestep_fs=args.timestep_fs,
        ps=args.ps, equil_ps=args.equil_ps, frame_ps=args.frame_ps,
        friction=args.friction, seed=args.seed)
    run_langevin(atoms, calc, cfg, args.out)
    return 0


def reweight_main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="off-polar-reweight")
    ap.add_argument("--traj", required=True)
    ap.add_argument("--polar-model", required=True)
    ap.add_argument("--source-model", default=None,
                    help="optional; if omitted frames must carry OFF_energy")
    ap.add_argument("--temperature", type=float, default=300.0)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float64")
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args(argv)

    import ase.io
    frames = list(ase.io.iread(args.traj, format="extxyz"))
    cfg = ReweightConfig(temperature_K=args.temperature, stride=args.stride,
                         max_frames=args.max_frames, device=args.device,
                         dtype=args.dtype)
    if args.source_model:
        rw = OffPolarReweighter.from_models(
            source_model=args.source_model, target_model=args.polar_model,
            config=cfg)
    else:
        if str(GRAPH_LONG_RANGE) not in sys.path:
            sys.path.insert(0, str(GRAPH_LONG_RANGE))
        from mace.calculators import mace_polar
        target = mace_polar(model=args.polar_model, device=cfg.device,
                            default_dtype=cfg.dtype)
        rw = OffPolarReweighter(None, target, cfg)
        rw.target_model = str(Path(args.polar_model).resolve())

    result = rw.evaluate(frames)
    path = result.write(args.outdir)
    d = result.as_dict()
    print(json.dumps({k: d[k] for k in (
        "n_frames", "n_atoms", "constant_offset_meV_per_atom",
        "delta_meV_per_atom_std", "sigma_beta", "ess", "ess_fraction",
        "verdict")}, indent=1))
    print(f"[reweight] wrote {path}")
    return 0


def c2_main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="off-polar-c2")
    ap.add_argument("--traj", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--pair", nargs=2, default=["O", "O"])
    ap.add_argument("--rho", type=float, default=None,
                    help="number density of the first element (A^-3); "
                         "inferred from the frame if omitted")
    ap.add_argument("--rmax", type=float, default=8.0)
    ap.add_argument("--bins", type=int, default=80)
    args = ap.parse_args(argv)

    import ase.io
    frames = list(ase.io.iread(args.traj, format="extxyz"))
    if args.rho is None:
        n = sum(1 for s in frames[0].get_chemical_symbols()
                if s == args.pair[0])
        args.rho = n / frames[0].get_volume()
    res = direct_correlation_report(frames, args.rho, args.outdir,
                                    rmax=args.rmax, bins=args.bins)
    print(f"[c2] rho={args.rho:.6g} A^-3, frames={len(frames)}, "
          f"c2(r) min/max={res['c2_r'].min():.4g}/{res['c2_r'].max():.4g}")
    return 0
