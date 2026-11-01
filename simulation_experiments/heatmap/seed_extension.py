"""Seed a longer sweep from a finished one, so the extra time is computed
on top of it instead of from t = 0. Runs that stopped at the boundary are
copied as they are; runs that reached T_from get a checkpoint built from
their stored final state and history, which run_heatmap.py then resumes.
The source folder is only read.
    python seed_extension.py [system] [N] [T_from] [T_to]"""
import sys, json, shutil, pickle
import numpy as np
from run_heatmap import results_dir, point_file, NH4_UM, o2_values, DT
from common import SPECIES, SUBSTRATES   # on the path via run_heatmap

system = sys.argv[1] if len(sys.argv) > 1 else "all"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 99
T_from = float(sys.argv[3]) if len(sys.argv) > 3 else 100.0
T_to = float(sys.argv[4]) if len(sys.argv) > 4 else 150.0
dst = results_dir(system, N, T_to)
dst.mkdir(parents=True, exist_ok=True)
n_copied = n_seeded = 0
for j in range(len(o2_values(T_to))):
    for i in range(len(NH4_UM)):
        src = point_file(i, j, system, N, T_from)
        out = point_file(i, j, system, N, T_to)
        if out.exists() or out.with_suffix(".ckpt").exists():
            continue
        rec = json.loads(src.read_text())
        fsrc = src.parent / f"fields_nh4_{i}_o2_{j}"
        fdst = dst / fsrc.name
        shutil.copytree(fsrc, fdst, dirs_exist_ok=True)
        if rec["info"]["boundary_touch_t"] is not None:
            shutil.copy2(src, out)
            n_copied += 1
            continue
        hist = json.loads((fsrc / "history.json").read_text())
        t = f"{hist['t'][-1]:g}"
        U = {s: np.load(fsrc / f"t{t}_u_{s}.npy").astype(np.float64) for s in SPECIES}
        C = {s: np.load(fsrc / f"t{t}_c_{s}.npy").astype(np.float64) for s in SUBSTRATES}
        k = int(round(hist["t"][-1] / DT))
        with open(out.with_suffix(".ckpt"), "wb") as fh:
            pickle.dump({"U": U, "C": C, "hist": hist, "k": k, "elapsed": rec["info"]["wall_s"]}, fh)
        n_seeded += 1
print(f"{system}: {n_copied} boundary runs copied, {n_seeded} runs seeded at t = {T_from:g} -> {dst.name}")
