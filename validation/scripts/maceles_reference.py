#!/usr/bin/env python3
"""Load the official MACELES checkpoint as a PyTorch reference.

The checkpoint is a pickled ``MACELES`` module whose pickle records the class at
``mace.modules.models.MACELES``, while the LES-enabled MACE fork defines it in
``mace.modules.extensions``.  A one-line alias makes ``torch.load`` work; this
module keeps that shim in one place so the reference can be run anywhere.

    export MACE_LES_ROOT=/opt/matos-tools
    export PYTHONPATH=$MACE_LES_ROOT/src/chengucb-mace:$MACE_LES_ROOT/src/chengucb-les/src:$PYTHONPATH

    python scripts/maceles_reference.py --model assets/maceles_off/MACELES-OFF_small.model --check
    python scripts/maceles_reference.py --model ... --eval frame.xyz --device cuda
"""
from __future__ import annotations

import argparse
import sys


def load_maceles(path, device="cpu"):
    """Load a pickled MACELES checkpoint (PyTorch reference implementation)."""
    import torch
    import mace.modules.models as mm
    from mace.modules.extensions import MACELES

    mm.MACELES = MACELES                      # pickle path used by the checkpoint
    return torch.load(path, map_location=device, weights_only=False)


def make_calculator(model, device="cpu", dtype="float64"):
    """ASE calculator around a loaded MACELES module."""
    import torch
    from mace.calculators import MACECalculator

    model = model.to(device=device, dtype=getattr(torch, dtype)).float() \
        if dtype == "float32" else model.to(device=device, dtype=getattr(torch, dtype))
    model.eval()
    return MACECalculator(model=model, device=device, default_dtype=dtype)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="float64", choices=["float32", "float64"])
    ap.add_argument("--check", action="store_true",
                    help="load the checkpoint and print its class / size")
    ap.add_argument("--eval", nargs="+", metavar="FILE",
                    help="ASE-readable structure file(s) to evaluate")
    args = ap.parse_args(argv)

    model = load_maceles(args.model, args.device)
    if args.check or not args.eval:
        n = sum(p.numel() for p in model.parameters())
        print(f"loaded {type(model).__module__}.{type(model).__name__} "
              f"| {n:,} parameters | device={args.device}")
    if args.eval:
        import ase.io
        import numpy as np
        calc = make_calculator(model, args.device, args.dtype)
        for path in args.eval:
            atoms = ase.io.read(path)
            atoms.calc = calc
            e = atoms.get_potential_energy()
            f = atoms.get_forces()
            fn = np.linalg.norm(f, axis=1)
            print(f"{path}: {len(atoms)} atoms  E={e:.6f} eV  "
                  f"|F|max={fn.max():.6f}  |F|mean={fn.mean():.6f} eV/A")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
