"""AOB vs CMX over (NH4_BC, O2_BC): one run per grid point, each stopped just
before its biomass reaches the boundary (or at T_MAX). Resumable: finished
points are skipped. System 'all' is the combined system (the heatmap);
'cmx' is CMX alone, the reference for how much CMX loses to competition.
    python run_heatmap.py [system] [N] [n_workers] [T_max]"""
import sys, json
from pathlib import Path
from multiprocessing import Pool
import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))
import common as c

PRESET = "eloi"
DT = 0.5
NH4_UM = np.round(np.geomspace(0.5, 1000.0, 9), 3)
O2_UM = np.array([5.0, 15.0, 40.0, 100.0, 250.0])


def results_dir(system, N, T_max=60.0):
    return HERE / f"results_{system}_N{N}_T{T_max:g}"


def point_file(i, j, system="all", N=99, T_max=60.0):
    return results_dir(system, N, T_max) / f"point_nh4_{i}_o2_{j}.json"


def run_point(args):
    i, j, system, N, T_MAX = args
    f = point_file(i, j, system, N, T_MAX)
    if f.exists():
        return f.name
    u = c.Units(PRESET)
    h, info = c.run(PRESET, system, NH4_UM[i] / u.c_ref_uM, O2_UM[j] / u.c_ref_uM, N=N, T_max=T_MAX, dt=DT)
    k = c.last_valid_index(h, info)
    rec = {"i": i, "j": j, "NH4_uM": float(NH4_UM[i]), "O2_uM": float(O2_UM[j]), "info": info,
           "k_stop": k, "t_stop": h["t"][k], "F_CMX": c.f_cmx(h, k),
           "M": {s: h["M"][s] for s in h["M"]}, "t": h["t"]}
    f.write_text(json.dumps(rec))
    return f.name


if __name__ == "__main__":
    system = sys.argv[1] if len(sys.argv) > 1 else "all"
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 99
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    T_MAX = float(sys.argv[4]) if len(sys.argv) > 4 else 60.0
    results_dir(system, N, T_MAX).mkdir(parents=True, exist_ok=True)
    jobs = [(i, j, system, N, T_MAX) for j in range(len(O2_UM)) for i in range(len(NH4_UM))]
    print(f"{system} N={N} T_max={T_MAX:g}: {len(jobs)} points, NH4 {NH4_UM.tolist()} uM, O2 {O2_UM.tolist()} uM, "
          f"{workers} workers", flush=True)
    with Pool(workers) as pool:
        for name in pool.imap_unordered(run_point, jobs):
            print("done", name, flush=True)
