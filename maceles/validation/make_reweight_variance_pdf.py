#!/usr/bin/env python3
"""Build the corrected report under a new name; preserve historical PDFs.

Run on a compute allocation. Requires XeLaTeX. The former four-group report
mixed OFF-frame student scores and canonical ESS and is now historical.
"""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import tempfile


def main():
    root=Path(__file__).resolve().parents[1]
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=root/'docs/reports/REWEIGHT_VARIANCE_REVISED_2026-10-27.pdf')
    args=parser.parse_args()
    output=args.output.resolve()
    if output.exists():
        parser.error(f'Refusing to overwrite {output}; choose a new --output path.')
    engine=shutil.which('xelatex')
    if engine is None:
        parser.error('XeLaTeX is required; add its bin directory to PATH.')
    source=root/'docs/reports/REWEIGHT_VARIANCE_REVISED_2026-10-27.tex'
    # Task-owned build directory only. Preserve build logs on failure.
    scratch=root/'audits/reweight-correction-20261027'
    scratch.mkdir(parents=True,exist_ok=True)
    build=Path(tempfile.mkdtemp(prefix='build-',dir=scratch))
    command=[engine,'-interaction=nonstopmode','-halt-on-error',f'-output-directory={build}',str(source)]
    with (build/'compiler.stdout').open('x') as log:
        for _ in range(2):subprocess.run(command,cwd=root,stdout=log,stderr=subprocess.STDOUT,check=True)
    pdf=build/(source.stem+'.pdf')
    # Exclusive publication refuses existing results, even if created meanwhile.
    with output.open('xb') as f:
        f.write(pdf.read_bytes());f.flush();os.fsync(f.fileno())
    print(f'Wrote {output}; build evidence: {build}')


if __name__=='__main__':main()
