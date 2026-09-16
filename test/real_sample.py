# Locate / download a public CINRAD Level2 sample and extract arrays via cinrad.io.
import os
import urllib.request
from pathlib import Path

import numpy as np

SAMPLE_NAME = "Z_RADR_I_Z9250_20160701001000_O_DOR_SA_CAP.bin"
SAMPLE_URL = (
    "https://github.com/uniquezhiyuan/PyRadar/raw/master/"
    "Z_RADR_I_Z9250_20160701001000_O_DOR_SA_CAP.bin"
)
# same naming family as example/*.ipynb (Z_RADR_I_*_O_DOR_SA_CAP)
EXTRACTED = Path(__file__).parent / "golden" / "real_z9250.npz"

RADAR_SUFFIXES = {".bin", ".bz2", ".gz", ".dat", ".raw", ".AR2", ".IQ", ".bz"}


def cinrad_data_radar_files():
    import cinrad_data

    root = Path(cinrad_data.__file__).resolve().parent
    found = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in RADAR_SUFFIXES:
            found.append(str(p.relative_to(root)))
    return root, found


def candidate_sample_paths():
    env = os.environ.get("PYCINRAD_SAMPLE")
    if env:
        yield Path(env)
    here = Path(__file__).parent
    yield here / "data" / SAMPLE_NAME
    yield Path("/tmp/cinrad-samples") / SAMPLE_NAME
    yield Path.cwd() / SAMPLE_NAME


def find_sample_file():
    for p in candidate_sample_paths():
        if p.is_file() and p.stat().st_size > 1000:
            return p
    return None


def download_sample(dest: Path, timeout=60):
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(SAMPLE_URL, tmp)
    tmp.replace(dest)
    return dest


def ensure_sample_file(timeout=60):
    found = find_sample_file()
    if found is not None:
        return found
    dest = Path("/tmp/cinrad-samples") / SAMPLE_NAME
    return download_sample(dest, timeout=timeout)


def extract_volume(path, drange=230):
    from cinrad.calc import _extract
    from cinrad.io.level2 import CinradReader

    f = CinradReader(str(path))
    r_list = [f.get_data(int(i), drange, "REF") for i in f.angleindex_r]
    ref, dist, az, elev = _extract(r_list, "REF")
    return {
        "reader": f,
        "ref": np.ascontiguousarray(ref, dtype=np.float64),
        "dist": np.ascontiguousarray(dist, dtype=np.float64),
        "elev": np.ascontiguousarray(elev, dtype=np.float64),
        "radarheight": float(getattr(f, "radarheight", 0) or 0),
        "tilts_r": [int(i) for i in f.angleindex_r],
        "tilts_v": [int(i) for i in f.angleindex_v],
    }


def extract_vel(reader, tilt, drange=230):
    ds = reader.get_data(int(tilt), drange, "VEL")
    v = np.ascontiguousarray(np.asarray(ds["VEL"].values, dtype=np.float64))
    nyq = float(ds.attrs["nyquist_vel"])
    return v, nyq
