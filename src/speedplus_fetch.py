"""
SPEED+ download and subset extraction.

SPEED+ is published on Zenodo as a single 16.9 GB archive, speedplusv2.zip:

    record : https://zenodo.org/records/5588480
    file   : https://zenodo.org/records/5588480/files/speedplusv2.zip?download=1
    md5    : 5cb3f44219054c49959e2009e597523b
    licence: CC-BY-4.0
    cite   : Park, Maertens, Lecuyer, Izzo, D'Amico, IEEE Aerospace 2022


"""
import hashlib
import json
import os
import zipfile

SPEEDPLUS_URL = "https://zenodo.org/records/5588480/files/speedplusv2.zip?download=1"
SPEEDPLUS_MD5 = "5cb3f44219054c49959e2009e597523b"
SPEEDPLUS_SIZE_GB = 16.9


def md5_of(path, chunk=1 << 22, log=print):
    h = hashlib.md5()
    total = os.path.getsize(path)
    done = 0
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
            done += len(b)
            if done % (1 << 30) < chunk:
                log(f"    hashed {done/1e9:.1f} / {total/1e9:.1f} GB")
    return h.hexdigest()


def verify_zip(zip_path, check_md5=False, log=print):
    """Check the archive is complete before spending time unzipping it."""
    if not os.path.exists(zip_path):
        log(f"not found: {zip_path}")
        return False
    gb = os.path.getsize(zip_path) / 1e9
    log(f"archive size: {gb:.2f} GB (expected about {SPEEDPLUS_SIZE_GB})")
    if gb < SPEEDPLUS_SIZE_GB * 0.95:
        log("archive is too small: the download was interrupted. Re-run wget -c.")
        return False
    if check_md5:
        log("checking md5, this takes a few minutes ...")
        got = md5_of(zip_path, log=log)
        ok = got == SPEEDPLUS_MD5
        log(f"md5 {'matches' if ok else 'MISMATCH'}: {got}")
        return ok
    try:
        with zipfile.ZipFile(zip_path) as z:
            n = len(z.namelist())
        log(f"archive opens correctly, {n} entries")
        return True
    except zipfile.BadZipFile:
        log("archive is corrupt. Delete it and download again.")
        return False


def _find_root(names):
    """Locate the folder inside the archive that holds camera.json."""
    for n in names:
        if n.endswith("camera.json"):
            return n[: -len("camera.json")]
    return ""


def extract_subset(zip_path, dest, n_synth_train=8000, n_synth_val=1500,
                   n_hil=600, log=print):
    """Extract labels in full and only the first N images of each domain.

    The JSON label files are a few MB and are always taken whole, so the record
    lists stay complete and the counts below can be raised later without
    re-reading the archive.
    """
    os.makedirs(dest, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
        root = _find_root(names)
        log(f"archive root: {root!r}")

        # 1. labels and camera
        wanted_meta = ["camera.json",
                       "synthetic/train.json", "synthetic/validation.json",
                       "lightbox/test.json", "sunlamp/test.json"]
        for rel in wanted_meta:
            src = root + rel
            if src not in names:
                log(f"  missing in archive (skipped): {rel}")
                continue
            out = os.path.join(dest, rel)
            os.makedirs(os.path.dirname(out), exist_ok=True)
            with z.open(src) as fsrc, open(out, "wb") as fdst:
                fdst.write(fsrc.read())
            log(f"  wrote {rel}")

        # 2. images, capped per split
        plan = [("synthetic", "train.json", n_synth_train),
                ("synthetic", "validation.json", n_synth_val),
                ("lightbox", "test.json", n_hil),
                ("sunlamp", "test.json", n_hil)]
        seen = {}
        for domain, jname, cap in plan:
            jpath = os.path.join(dest, domain, jname)
            if not os.path.exists(jpath) or cap <= 0:
                continue
            with open(jpath) as f:
                recs = json.load(f)
            files = [r.get("filename") or r.get("file_name") for r in recs][:cap]
            outdir = os.path.join(dest, domain, "images")
            os.makedirs(outdir, exist_ok=True)
            n_new = n_have = n_missing = 0
            for fn in files:
                out = os.path.join(outdir, fn)
                if os.path.exists(out) or (domain, fn) in seen:
                    seen[(domain, fn)] = True
                    n_have += 1
                    continue
                src = f"{root}{domain}/images/{fn}"
                if src not in names:
                    n_missing += 1
                    continue
                with z.open(src) as fsrc, open(out, "wb") as fdst:
                    fdst.write(fsrc.read())
                seen[(domain, fn)] = True
                n_new += 1
                if n_new % 1000 == 0:
                    log(f"    {domain}/{jname}: {n_new} extracted so far")
            msg = f"  {domain}/{jname}: {n_new} new"
            if n_have:
                msg += f", {n_have} already present"
            if n_missing:
                msg += f", {n_missing} NOT FOUND in archive"
            log(msg + f"  (target {len(files)})")
    return dest


def summarise(root, log=print):
    """Confirm the extracted tree is what the loader expects."""
    ok = True
    cam = os.path.join(root, "camera.json")
    log(f"camera.json     : {'found' if os.path.exists(cam) else 'MISSING'}")
    ok &= os.path.exists(cam)
    for domain, split in [("synthetic", "train"), ("synthetic", "validation"),
                          ("lightbox", "test"), ("sunlamp", "test")]:
        j = os.path.join(root, domain, f"{split}.json")
        imgs = os.path.join(root, domain, "images")
        if not os.path.exists(j):
            log(f"{domain}/{split}.json : MISSING")
            ok = False
            continue
        with open(j) as f:
            n_rec = len(json.load(f))
        n_img = len(os.listdir(imgs)) if os.path.isdir(imgs) else 0
        log(f"{domain}/{split:<10} : {n_rec:6d} records in json, {n_img:6d} images on disk")
    return ok
