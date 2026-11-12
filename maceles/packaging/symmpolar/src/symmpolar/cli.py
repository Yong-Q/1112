"""symmpolar — thin CLI over the MACELES/LES toolchain.

Every subcommand shells out to the repository scripts so that the scientific
code keeps living in one place (``scripts/`` and ``sampling/``).
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

DEFAULT_REPO = Path(__file__).resolve().parents[3]   # .../test


def repo_root(explicit=None) -> Path:
    root = Path(explicit or os.environ.get("SYMMPOLAR_REPO") or DEFAULT_REPO)
    if not (root / "sampling").is_dir():
        sys.exit(f"cannot locate the repository (looked at {root}); "
                 f"set SYMMPOLAR_REPO or pass --repo")
    return root


def run(cmd, **kw):
    print("+", " ".join(str(c) for c in cmd))
    return subprocess.run([str(c) for c in cmd], **kw)


def cmd_check(args):
    import importlib
    missing = []
    for mod in ("numpy", "scipy", "ase", "matplotlib"):
        try:
            m = importlib.import_module(mod)
            print(f"  {mod:10s} {getattr(m, '__version__', '?'):12s} ok")
        except Exception as exc:                       # noqa: BLE001
            print(f"  {mod:10s} MISSING ({exc})")
            missing.append(mod)
    for name in ("model", "cubin", "lmp"):
        path = getattr(args, name, None)
        if not path:
            continue
        p = Path(path)
        print(f"  {name:10s} {'ok' if p.exists() else 'MISSING'}  {p}")
        if not p.exists():
            missing.append(name)
    if missing:
        print("missing:", ", ".join(missing))
        return 1
    print("all good")
    return 0


def cmd_build_lammps(args):
    repo = repo_root(args.repo)
    env = os.environ.copy()
    for key in ("LAMMPS_DIR", "SYMMETRIX_DIR", "CUDA_HOME", "CUDA_ARCH",
                "BUILD_DIR", "JOBS", "APPLY_PATCH"):
        val = getattr(args, key.lower(), None)
        if val is not None:
            env[key] = str(val)
    return run(["bash", repo / "scripts" / "build_engine.sh"], env=env).returncode


def cmd_extract_model(args):
    repo = repo_root(args.repo)
    script = Path(args.symmetrix_dir or os.environ.get("SYMMETRIX_DIR", "")) \
        / "symmetrix" / "source" / "symmetrix" / "extract_maceles.py"
    if not script.exists():
        script = repo.parent / "symmetrix-xl" / "symmetrix" / "source" \
            / "symmetrix" / "extract_maceles.py"
    if not script.exists():
        sys.exit("extract_maceles.py not found; pass --symmetrix-dir")
    cmd = [sys.executable, script, "--model", args.checkpoint,
           "--output", args.output]
    if args.head:
        cmd += ["--head", args.head]
    if args.chemical_symbols:
        cmd += ["--chemical-symbols", *args.chemical_symbols]
    return run(cmd).returncode


def cmd_prepare_cubin(args):
    cmd = [sys.executable, "-m", "symmetrix.cli.prepare_jit_device_artifact",
           "--model", args.model, "--precision", args.precision,
           "--lammps-arguments"]
    return run(cmd).returncode


def cmd_smoke(args):
    repo = repo_root(args.repo)
    env = os.environ.copy()
    env.update({"MODEL": args.model, "CUBIN": args.cubin, "LMP": args.lmp})
    if args.out:
        env["OUT"] = args.out
    return run(["bash", repo / "scripts" / "smoke_test.sh"], env=env).returncode


def cmd_scan(args):
    repo = repo_root(args.repo)
    env = os.environ.copy()
    env["STAGE"] = args.stage
    if args.root:
        env["OUT"] = args.root
    for key, val in (("MODEL", args.model), ("CUBIN", args.cubin),
                     ("LMP", args.lmp), ("SHARDS", args.shards),
                     ("DUMP_EVERY", args.dump_every),
                     ("STEPS", args.steps),
                     ("EQUIL_STEPS", args.equil_steps)):
        if val is not None:
            env[key] = str(val)
    return run(["bash", repo / "scripts" / "run_density_scan.sh"],
               env=env).returncode


def main(argv=None):
    ap = argparse.ArgumentParser(prog="symmpolar", description=__doc__)
    ap.add_argument("--repo", help="path to the test repository")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("check", help="verify python deps and asset paths")
    p.add_argument("--model")
    p.add_argument("--cubin")
    p.add_argument("--lmp")
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("build-lammps", help="build LAMMPS + pair_symmetrix")
    p.add_argument("--lammps-dir")
    p.add_argument("--symmetrix-dir")
    p.add_argument("--cuda-home")
    p.add_argument("--cuda-arch", type=int)
    p.add_argument("--build-dir")
    p.add_argument("--jobs", type=int)
    p.set_defaults(func=cmd_build_lammps)

    p = sub.add_parser("extract-model", help="checkpoint.pth -> engine json")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", "-o", required=True)
    p.add_argument("--symmetrix-dir")
    p.add_argument("--head")
    p.add_argument("--chemical-symbols", nargs="+")
    p.set_defaults(func=cmd_extract_model)

    p = sub.add_parser("prepare-cubin", help="rebuild the device artifact")
    p.add_argument("--model", required=True)
    p.add_argument("--precision", default="float32",
                   choices=["float32", "float64"])
    p.set_defaults(func=cmd_prepare_cubin)

    p = sub.add_parser("smoke", help="64-water end-to-end test")
    p.add_argument("--model", required=True)
    p.add_argument("--cubin", required=True)
    p.add_argument("--lmp", required=True)
    p.add_argument("--out")
    p.set_defaults(func=cmd_smoke)

    p = sub.add_parser("scan", help="run the density scan")
    p.add_argument("--stage", default="all",
                   choices=["data", "split", "run", "post", "all"])
    p.add_argument("--root")
    p.add_argument("--model")
    p.add_argument("--cubin")
    p.add_argument("--lmp")
    p.add_argument("--shards", type=int)
    p.add_argument("--dump-every", type=int)
    p.add_argument("--steps", type=int)
    p.add_argument("--equil-steps", type=int)
    p.set_defaults(func=cmd_scan)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
