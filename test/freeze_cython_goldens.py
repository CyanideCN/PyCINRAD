# Rebuild cython_ref.npz from original Cython side modules. Run locally, not in pytest.
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).parent))
from align_cases import VOLUME_CASES, VEL_CASES, VILD_SKIP, VOLUME_VARIANTS
from cython_side import load_cython_side, cython_dealias_unwrap_2d
from real_sample import (
    SAMPLE_NAME,
    EXTRACTED,
    ensure_sample_file,
    extract_volume,
    extract_vel,
)

GOLDEN = Path(__file__).parent / "golden" / "cython_ref.npz"


def main():
    utils, unwrap = load_cython_side()
    out = {}
    for name, factory in VOLUME_CASES.items():
        ref, dist, elev = factory()
        ref = np.ascontiguousarray(ref, dtype=np.float64)
        dist = np.ascontiguousarray(dist, dtype=np.float64)
        elev = np.ascontiguousarray(elev, dtype=np.float64)
        out["vil_" + name] = utils.vert_integrated_liquid(ref, dist, elev)
        if name not in VILD_SKIP:
            out["vild_" + name] = utils.vert_integrated_liquid(
                ref, dist, elev, density=True
            )
        out["et_" + name] = utils.echo_top(ref, dist, elev, 0.0)
        out["et_h_" + name] = utils.echo_top(ref, dist, elev, 137.0)
    ref, dist, elev = VOLUME_CASES["thresh"]()
    out["vil_thresh40"] = utils.vert_integrated_liquid(
        ref, dist, elev, threshold=40.0
    )
    out["et_thresh40"] = utils.echo_top(ref, dist, elev, 0.0, threshold=40.0)
    ref, dist, elev = VOLUME_CASES["irregular"]()
    out["vil_thresh10"] = utils.vert_integrated_liquid(
        ref, dist, elev, threshold=10.0
    )
    out["et_thresh10"] = utils.echo_top(ref, dist, elev, 0.0, threshold=10.0)
    ref, dist, elev = VOLUME_CASES["mix"]()
    out["vil_bw1"] = utils.vert_integrated_liquid(ref, dist, elev, beam_width=1.0)
    ref, dist, elev = VOLUME_CASES["et_interp"]()
    out["et_h973"] = utils.echo_top(ref, dist, elev, 973.0)
    _ = VOLUME_VARIANTS
    for name, factory in VEL_CASES.items():
        vdata, nyq = factory()
        out["dealias_" + name] = cython_dealias_unwrap_2d(unwrap, vdata, nyq)

    sample = ensure_sample_file()
    vol = extract_volume(sample)
    ref, dist, elev = vol["ref"], vol["dist"], vol["elev"]
    out["vil_z9250"] = utils.vert_integrated_liquid(ref, dist, elev)
    out["vild_z9250"] = utils.vert_integrated_liquid(ref, dist, elev, density=True)
    out["et_z9250"] = utils.echo_top(ref, dist, elev, vol["radarheight"])
    v1, n1 = extract_vel(vol["reader"], 1)
    v5, n5 = extract_vel(vol["reader"], 5)
    out["dealias_z9250_t1"] = cython_dealias_unwrap_2d(unwrap, v1, n1)
    out["dealias_z9250_t5"] = cython_dealias_unwrap_2d(unwrap, v5, n5)
    EXTRACTED.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        EXTRACTED,
        ref=ref,
        dist=dist,
        elev=elev,
        radarheight=np.array(vol["radarheight"]),
        filename=np.array(SAMPLE_NAME),
        vel_t1=v1,
        nyq_t1=np.array(n1),
        vel_t5=v5,
        nyq_t5=np.array(n5),
    )
    GOLDEN.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(GOLDEN, **out)
    print("wrote", GOLDEN, "keys", len(out), "bytes", GOLDEN.stat().st_size)
    print("wrote", EXTRACTED, "bytes", EXTRACTED.stat().st_size)


if __name__ == "__main__":
    main()
