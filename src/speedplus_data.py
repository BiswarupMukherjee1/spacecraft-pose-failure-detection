"""
SPEED+ data handling.

SPEED+ (Park, Maertens, Lecuyer, Izzo, D'Amico, IEEE Aerospace 2022) contains
59,960 labelled synthetic images of the Tango spacecraft from the PRISMA
mission, plus two hardware-in-the-loop domains captured in Stanford SLAB's TRON
facility: 'lightbox', which simulates Earth albedo with diffuser plates, and
'sunlamp', which simulates direct sunlight with a metal-halide lamp.

The two HIL domains are deliberately released without labels. That is not a
limitation to work around: it is the situation a real programme is in before
flight, and it is the reason the second Satellite Pose Estimation Competition
(SPEC2021) was built around them. It also shapes what can actually be measured
here. Pose error can only be computed on the synthetic domain. On the HIL
domains the question has to be asked without labels, which is exactly what an
epistemic uncertainty estimate is for.

"""
import json
import os
import numpy as np
import cv2

# Keys differ slightly between SPEED and SPEED+ releases; accept both.
_Q_KEYS = ("q_vbs2tango_true", "q_vbs2tango", "q")
_T_KEYS = ("r_Vo2To_vbs_true", "r_Vo2To_vbs", "r")


def _first_key(d, keys):
    for k in keys:
        if k in d:
            return d[k]
    return None



# Camera

def load_camera(root):
    """Return (K, dist, (width, height)) from SPEED+ camera.json."""
    p = os.path.join(root, "camera.json")
    with open(p) as f:
        c = json.load(f)
    if "cameraMatrix" in c:
        K = np.array(c["cameraMatrix"], dtype=np.float64)
    else:
        fx = float(c.get("fx", c.get("focalLengthX")))
        fy = float(c.get("fy", c.get("focalLengthY")))
        cx = float(c.get("ccx", c.get("ppx", c.get("Nu", 1920)) / 2))
        cy = float(c.get("ccy", c.get("ppy", c.get("Nv", 1200)) / 2))
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.]], dtype=np.float64)
    dist = np.array(c.get("distCoeffs", [0, 0, 0, 0, 0]), dtype=np.float64).ravel()
    w = int(c.get("Nu", c.get("width", 1920)))
    h = int(c.get("Nv", c.get("height", 1200)))
    return K, dist, (w, h)



# Labels

def load_split(root, domain, split):
    """Return a list of records. Labels are absent for the HIL domains."""
    p = os.path.join(root, domain, f"{split}.json")
    with open(p) as f:
        raw = json.load(f)
    out = []
    for r in raw:
        fn = r.get("filename") or r.get("file_name")
        q = _first_key(r, _Q_KEYS)
        t = _first_key(r, _T_KEYS)
        out.append({
            "path": os.path.join(root, domain, "images", fn),
            "filename": fn,
            "q": None if q is None else np.asarray(q, np.float32),   # scalar-first
            "t": None if t is None else np.asarray(t, np.float32),   # metres
        })
    return out


def quat_to_R(q):
    """Scalar-first quaternion to rotation matrix (SPEED+ convention)."""
    q = np.asarray(q, np.float64)
    q = q / (np.linalg.norm(q) + 1e-12)
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)],
    ])


def quat_angle_deg(q1, q2):
    """Smallest rotation angle between two quaternions, in degrees.

    The absolute value handles the double cover: q and -q are the same rotation.
    """
    q1 = np.asarray(q1, np.float64); q2 = np.asarray(q2, np.float64)
    q1 = q1 / (np.linalg.norm(q1) + 1e-12)
    q2 = q2 / (np.linalg.norm(q2) + 1e-12)
    d = abs(float(np.dot(q1, q2)))
    return float(np.degrees(2.0 * np.arccos(np.clip(d, -1.0, 1.0))))



# Image loading

def load_image(path, size=224, to_gray=True):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE if to_gray else cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    return img.astype(np.float32) / 255.


def filter_available(records, log=print):
    """Keep only records whose image is actually on disk.

    The label files list every image in the domain, but a session-sized run
    extracts only the first few thousand. Without this filter the loader walks
    off the end of what was extracted and raises FileNotFoundError partway
    through, after several minutes of loading.
    """
    have = [r for r in records if os.path.exists(r["path"])]
    if len(have) != len(records):
        log(f"    {len(have)} of {len(records)} images present on disk")
    return have


def build_arrays(records, size=224, limit=None, labelled=True, log_every=2000,
                 log=print, skip_missing=True):
    """Load a split into memory as arrays.

    Returns (X, T, Q); T and Q are None when the domain is unlabelled.
    With skip_missing the records are filtered to those actually extracted, so
    the caller can ask for more than is on disk and simply get what exists.
    """
    if skip_missing:
        records = filter_available(records, log=log)
    if limit is not None:
        records = records[:limit]
    if len(records) == 0:
        raise RuntimeError(
            "no images found on disk for this split. Run the extraction step, "
            "and check that the counts there are at least as large as the "
            "N_TRAIN / N_VAL / N_HIL used here.")
    X = np.zeros((len(records), 1, size, size), np.float32)
    T = np.zeros((len(records), 3), np.float32) if labelled else None
    Q = np.zeros((len(records), 4), np.float32) if labelled else None
    for i, r in enumerate(records):
        X[i, 0] = load_image(r["path"], size=size)
        if labelled:
            T[i] = r["t"]; Q[i] = r["q"]
        if log_every and (i + 1) % log_every == 0:
            log(f"    loaded {i+1}/{len(records)}")
    return X, T, Q





def fetch_speedplus(dest="/content/speedplus", query="SPEED%2B spacecraft pose",
                    want=("camera", "synthetic", "lightbox", "sunlamp"),
                    log=print):
    """Best-effort automatic download via the Zenodo API.

    Zenodo's search results change over time, so this is written to fail
    loudly and print manual instructions rather than to silently fetch the
    wrong thing.
    """
    import urllib.request
    import zipfile

    os.makedirs(dest, exist_ok=True)
    api = f"https://zenodo.org/api/records?q={query}&size=20"
    try:
        with urllib.request.urlopen(api, timeout=60) as r:
            hits = json.load(r).get("hits", {}).get("hits", [])
    except Exception as e:
        log("Zenodo API request failed:", type(e).__name__, str(e)[:200])
        raise

    rec = None
    for h in hits:
        title = (h.get("metadata", {}).get("title") or "").lower()
        if "speed+" in title or "speedplus" in title or "next generation" in title:
            rec = h
            break
    if rec is None:
        log("Could not identify the SPEED+ record automatically.")
        log("Candidates returned:")
        for h in hits[:10]:
            log("   -", h.get("metadata", {}).get("title"))
        
        raise RuntimeError("SPEED+ record not found automatically")

    log("Zenodo record:", rec.get("metadata", {}).get("title"))
    for f in rec.get("files", []):
        key = f.get("key", "")
        if not any(w in key.lower() for w in want):
            continue
        url = f.get("links", {}).get("self")
        out = os.path.join(dest, key)
        if os.path.exists(out):
            log("  already have", key); continue
        log(f"  downloading {key} ({f.get('size', 0)/1e9:.2f} GB) ...")
        urllib.request.urlretrieve(url, out)
        if key.lower().endswith(".zip"):
            log("  unzipping", key)
            with zipfile.ZipFile(out) as z:
                z.extractall(dest)
    return dest

