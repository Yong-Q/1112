import sys, os, json, subprocess, numpy as np, gzip
sys.path.insert(0,'/publicfs10/fs10-m9/home/m9s005258/qy/cdft/new')
from sampling.pbc_local_env import iter_dump
OUT="/publicfs10/fs10-m9/home/m9s005258/qy/cdft/validation_ref"
SPECIES=sys.argv[1]; STRIDE=40; NFR=50
STATE={"H2O":"H2O-T986.4-r00-rho0.6700","NH3":"NH3-T618.2-r00-rho0.6700",
       "CO2":"CO2-T463.6-r00-rho0.6700","CH4":"CH4-T290.5-r00-rho0.6700",
       "C2H4":"C2H4-T430.4-r00-rho0.6700","C2H6":"C2H6-T465.4-r00-rho0.6700"}[SPECIES]
D=f"/publicfs10/fs10-m9/home/m9s005258/qy/cdft/scan/traj/{STATE}/run"
work=f"/tmp/val_{SPECIES}"; os.makedirs(work, exist_ok=True)
# 1) extract 50 frames as a small LAMMPS dump
frames=[]; meta=None
for k,f in enumerate(iter_dump(f"{D}/traj.dump.gz", stride=STRIDE, max_frames=NFR)):
    frames.append(f)
    if meta is None: meta={"mol":f["mol"],"types":f["types"],"cell":f["cell"]}
print(f"{SPECIES}: {len(frames)} frames", flush=True)
small=f"{work}/frames.dump"
with open(small,"w") as fh:
    for f in frames:
        fh.write("ITEM: TIMESTEP\n%d\nITEM: NUMBER OF ATOMS\n%d\nITEM: BOX BOUNDS pp pp pp\n"%((f["step"]),f["n"]))
        lo=f["lo"]; hi=f["hi"]
        for i in range(3): fh.write("%.10f %.10f\n"%(lo[i],hi[i]))
        fh.write("ITEM: ATOMS id type x y z\n")
        for i in range(f["n"]):
            fh.write("%d %d %.10f %.10f %.10f\n"%(i+1,f["types"][i],*f["positions"][i]))
# 2) engine energies and forces
PS=subprocess.check_output(f"grep '^pair_style' {D}/in.les_md",shell=True,text=True).strip()
PC=subprocess.check_output(f"grep '^pair_coeff' {D}/in.les_md",shell=True,text=True).strip()
DATA=subprocess.check_output(f"grep '^read_data' {D}/in.les_md",shell=True,text=True).split()[1]
inp=f"{work}/in.e"
open(inp,"w").write(f"""units metal
atom_style molecular/kk
atom_modify map yes
boundary p p p
read_data {DATA}
{PS}
{PC}
neighbor 2.0 bin
neigh_modify every 10 delay 0 check yes
run_style verlet/kk
thermo 1
thermo_style custom step atoms pe
thermo_modify format float %.12g
dump 1 all custom 1 {work}/forces.dump id fx fy fz
dump_modify 1 sort id format float %.10g
rerun {small} dump x y z
""")
env=dict(os.environ)
env["LD_LIBRARY_PATH"]="/publicfs10/fs10-share/soft/share-soft/gcc/gcc-14.2.0/install/lib64:/publicfs10/fs10-share/soft/share-soft/gcc/gcc-14.2.0/install/lib:/publicfs10/fs10-share/soft/share-soft/openmpi/openmpi-4.1.8/install-new/lib:"+env.get("LD_LIBRARY_PATH","")
subprocess.run(["/publicfs10/fs10-m9/home/m9s005258/qy/build-cpu-mpi/lmp","-k","on","t","1","-in",inp,"-log",f"{work}/lmp.log"],env=env,check=True,stdout=subprocess.DEVNULL)
E_eng=[]
for line in open(f"{work}/lmp.log"):
    p=line.split()
    if len(p)==3 and p[1].isdigit() and p[0].lstrip('-').isdigit():
        E_eng.append(float(p[2]))
E_eng=np.array(E_eng[:NFR])
# parse force dump
F_eng=[]; cur=None
with open(f"{work}/forces.dump") as fh:
    while True:
        l=fh.readline()
        if not l: break
        if l.startswith("ITEM: TIMESTEP"):
            fh.readline(); fh.readline(); n=int(fh.readline()); fh.readline()
            for _ in range(3): fh.readline()
            fh.readline()
            arr=np.array(" ".join(fh.readline() for _ in range(n)).split(),float).reshape(n,4)
            F_eng.append(arr[:,1:])
F_eng=np.array(F_eng[:NFR])
print(f"{SPECIES}: engine E {E_eng.shape} F {F_eng.shape}", flush=True)
np.savez_compressed(f"{work}/engine.npz", energy=E_eng, forces=F_eng.astype(np.float32),
                    positions=np.array([f["positions"] for f in frames]).astype(np.float32),
                    types=meta["types"], cell=meta["cell"])
print("SAVED", SPECIES, flush=True)
