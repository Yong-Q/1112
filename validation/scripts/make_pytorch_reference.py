import sys, os, time, numpy as np, torch
sys.path.insert(0,'/publicfs10/fs10-m9/home/m9s005258/qy/cdft/new/scripts')
from maceles_reference import load_maceles
from mace.calculators import MACECalculator
from ase import Atoms
SP=sys.argv[1]; work=f"/tmp/val_{SP}"
d=np.load(f"{work}/engine.npz")
pos=d["positions"]; types=d["types"]; cell=d["cell"]
sym=np.array(["H" if t==2 else "N" if t==3 else "O" if t==4 else "C" for t in types])
model=load_maceles("/publicfs10/fs10-m9/home/m9s005258/qy/test/assets/maceles_off/MACELES-OFF_small.model","cpu")
calc=MACECalculator(models=[model], device="cpu", default_dtype="float64")
E=[];F=[]
t0=time.time()
for i in range(len(pos)):
    a=Atoms(symbols=list(sym), positions=pos[i].astype(np.float64), cell=np.diag(cell), pbc=True)
    a.calc=calc
    E.append(a.get_potential_energy()); F.append(a.get_forces())
    if (i+1)%10==0: print(f"  {SP} {i+1}/{len(pos)} {time.time()-t0:.0f}s", flush=True)
E=np.array(E); F=np.array(F)
np.savez_compressed(f"{work}/ref.npz", energy=E, forces=F.astype(np.float32))
dE=E-d["energy"]; dF=F-d["forces"].astype(np.float64)
print(f"{SP}: dE max={np.abs(dE).max():.3e} rms={np.sqrt((dE**2).mean()):.3e} eV | "
      f"dF max={np.abs(dF).max():.3e} rms={np.sqrt((dF**2).mean()):.3e} eV/A", flush=True)
np.save(f"{work}/diff.npy", np.array([np.abs(dE).max(), np.sqrt((dE**2).mean()), np.abs(dF).max(), np.sqrt((dF**2).mean())]))
