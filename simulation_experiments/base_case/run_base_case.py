"""Base case: AOB system (AOB+NOB), CMX system and combined system (all three)
at one realistic operating point, every species starting with the same mass.
    python run_base_case.py [N] [NH4_uM] [O2_uM]"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as c

PRESET = "eloi"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 99
NH4_UM = float(sys.argv[2]) if len(sys.argv) > 2 else 100.0
O2_UM = float(sys.argv[3]) if len(sys.argv) > 3 else 250.0
SYSTEM = sys.argv[4] if len(sys.argv) > 4 else None

if __name__ == "__main__":
    u = c.Units(PRESET)
    for system in ([SYSTEM] if SYSTEM else ["all", "aob", "cmx"]):
        out = Path(__file__).parent / "results" / system
        out.mkdir(parents=True, exist_ok=True)
        print(f"{PRESET} {system}: NH4_BC={NH4_UM} uM O2_BC={O2_UM} uM N={N}", flush=True)
        h, info = c.run(PRESET, system, NH4_UM / u.c_ref_uM, O2_UM / u.c_ref_uM, N=N, T_max=60.0,
                        dt=0.5, out=out, verbose=True)
        print(info, flush=True)
