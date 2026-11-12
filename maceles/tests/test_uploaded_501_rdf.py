"""The separate500ps helper must use the same unordered-pair convention."""
import importlib.util
from pathlib import Path
import numpy as np
from ase import Atoms
from sampling.off_polar import OffPolarReweighter

def test_uploaded_501_helper_matches_validated_core():
    path=Path(__file__).resolve().parents[1]/'validation/off23_500ps_water1152_2026-10-23/reweight_500ps_water1152.py'
    spec=importlib.util.spec_from_file_location('uploaded501',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    rng=np.random.default_rng(2701)
    frames=[Atoms('O200',positions=rng.uniform(0,20,(200,3)),cell=[20,20,20],pbc=True) for _ in range(10)]
    weights=np.linspace(1,2,len(frames));weights/=weights.sum()
    r,g=module.oo_rdf_weighted(frames,weights,rmax=8,nbins=40)
    rc,gc=OffPolarReweighter.rdf(frames,weights,pair=('O','O'),rmax=8,bins=40)
    assert np.allclose(r,rc)
    assert np.allclose(g,gc,rtol=1e-12,atol=1e-12), 'Separate501 helper has a pair-count normalization mismatch'
    assert abs(g[(r>2)&(r<7)].mean()-1)<.05
