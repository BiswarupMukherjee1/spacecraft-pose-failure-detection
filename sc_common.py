"""

Extends the target model and camera of the study with:
  - a z-buffered renderer that also returns, for every pixel, the 3D model point
    that projects there (needed for template-based matching without an oracle)
  - the image degradation operators
  - pose error metrics and the PnP+RANSAC stage
"""
import numpy as np
import cv2


# Target model

def build_spacecraft_keypoints() -> np.ndarray:
    body = np.array([
        [-0.5, -0.5, -0.75], [0.5, -0.5, -0.75], [0.5, 0.5, -0.75], [-0.5, 0.5, -0.75],
        [-0.5, -0.5,  0.75], [0.5, -0.5,  0.75], [0.5, 0.5,  0.75], [-0.5, 0.5,  0.75],
    ])
    left_panel  = np.array([[-2.5, -0.02, -0.5], [-0.5, -0.02, -0.5],
                            [-0.5, -0.02, 0.5], [-2.5, -0.02, 0.5]])
    right_panel = np.array([[0.5, 0.02, -0.5], [2.5, 0.02, -0.5],
                            [2.5, 0.02, 0.5], [0.5, 0.02, 0.5]])
    return np.vstack([body, left_panel, right_panel]).astype(np.float32)

SPACECRAFT_3D = build_spacecraft_keypoints()

SPACECRAFT_EDGES = [
    (0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),
    (0,4),(1,5),(2,6),(3,7),
    (8,9),(9,10),(10,11),(11,8),
    (12,13),(13,14),(14,15),(15,12),
]
SPACECRAFT_FACES = [
    ([0,1,2,3],(120,130,150)), ([4,5,6,7],(160,170,190)),
    ([0,1,5,4],(140,150,170)), ([2,3,7,6],(140,150,170)),
    ([1,2,6,5],(130,140,160)), ([0,3,7,4],(130,140,160)),
    ([8,9,10,11],(40,50,110)), ([12,13,14,15],(40,50,110)),
]

# ----------------------------------------------------------------------------
# Camera
# ----------------------------------------------------------------------------
IMG_W, IMG_H = 640, 480
FX = FY = 800.0
CX, CY = IMG_W / 2., IMG_H / 2.
CAMERA_K = np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1.]], dtype=np.float64)
CAMERA_DIST = np.zeros(5, dtype=np.float64)
CAMERA_K_INV = np.linalg.inv(CAMERA_K)


def random_pose(rng, z_range=(5., 15.), xy_range=0.5, rot_range_deg=30.):
    a = rng.uniform(-rot_range_deg, rot_range_deg, 3) * np.pi / 180.
    Rx = cv2.Rodrigues(np.array([a[0], 0, 0]))[0]
    Ry = cv2.Rodrigues(np.array([0, a[1], 0]))[0]
    Rz = cv2.Rodrigues(np.array([0, 0, a[2]]))[0]
    R = Rz @ Ry @ Rx
    rvec, _ = cv2.Rodrigues(R)
    tvec = np.array([rng.uniform(-xy_range, xy_range),
                     rng.uniform(-xy_range, xy_range),
                     rng.uniform(*z_range)])
    return rvec.flatten(), tvec


def project_points(P3, rvec, tvec):
    p, _ = cv2.projectPoints(P3.astype(np.float64), np.asarray(rvec, np.float64),
                             np.asarray(tvec, np.float64), CAMERA_K, CAMERA_DIST)
    return p.reshape(-1, 2)



# Renderer

def render_image(rvec, tvec, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)
    img = np.zeros((IMG_H, IMG_W, 3), np.uint8)
    n = 80
    sx = rng.integers(0, IMG_W, n); sy = rng.integers(0, IMG_H, n)
    sb = rng.integers(80, 255, n)
    for i in range(n):
        cv2.circle(img, (int(sx[i]), int(sy[i])), 0, (int(sb[i]),) * 3, -1)
    img = cv2.add(img, rng.normal(0, 3, img.shape).clip(0, 30).astype(np.uint8))

    R = cv2.Rodrigues(np.asarray(rvec, np.float64))[0]
    pts_cam = (R @ SPACECRAFT_3D.T).T + np.asarray(tvec).reshape(1, 3)
    pts_2d = project_points(SPACECRAFT_3D, rvec, tvec)

    faces = []
    for vidx, color in SPACECRAFT_FACES:
        d = pts_cam[vidx, 2].mean()
        if d < 0.5:
            continue
        faces.append((d, vidx, color))
    faces.sort(key=lambda x: -x[0])
    for d, vidx, color in faces:
        poly = pts_2d[vidx].astype(np.int32)
        if (poly[:, 0].max() < 0 or poly[:, 0].min() > IMG_W or
                poly[:, 1].max() < 0 or poly[:, 1].min() > IMG_H):
            continue
        shade = max(0.5, min(1.0, 12.0 / max(d, 4.0)))
        cv2.fillPoly(img, [poly], tuple(int(c * shade) for c in color))
    for i, j in SPACECRAFT_EDGES:
        p1, p2 = pts_2d[i], pts_2d[j]
        if (-100 < p1[0] < IMG_W + 100 and -100 < p1[1] < IMG_H + 100 and
                -100 < p2[0] < IMG_W + 100 and -100 < p2[1] < IMG_H + 100):
            cv2.line(img, tuple(p1.astype(int)), tuple(p2.astype(int)),
                     (200, 200, 210), 1, cv2.LINE_AA)
    for panel_idx in [(8, 9, 10, 11), (12, 13, 14, 15)]:
        p = pts_2d[list(panel_idx)].astype(np.float32)
        for t in np.linspace(0.2, 0.8, 4):
            a = (1 - t) * p[0] + t * p[1]; b = (1 - t) * p[3] + t * p[2]
            cv2.line(img, tuple(a.astype(int)), tuple(b.astype(int)),
                     (80, 90, 140), 1, cv2.LINE_AA)
    return img



# Z-buffered 3D lookup: for each pixel, which 3D model point projects there

def render_model_xyz(rvec, tvec):
    """
    Rasterise the target's faces with a z-buffer and return, per pixel:
      xyz    (H, W, 3) float32 : the 3D point in MODEL coordinates visible there
      valid  (H, W)    bool    : whether the target covers that pixel

    Method: each face is a planar quad. Its plane is expressed in camera
    coordinates as n . X = d. For a pixel (u,v) the viewing ray is
    r = K^-1 [u,v,1]; the intersection with the plane is at depth
    s = d / (n . r), giving X_cam = s * r. Converting back to model
    coordinates gives X_model = R^T (X_cam - t). The nearest face wins.
    """
    rvec = np.asarray(rvec, np.float64).reshape(3)
    tvec = np.asarray(tvec, np.float64).reshape(3)
    R = cv2.Rodrigues(rvec)[0]

    pts_cam = (R @ SPACECRAFT_3D.T).T + tvec.reshape(1, 3)
    pts_2d = project_points(SPACECRAFT_3D, rvec, tvec)

    depth_buf = np.full((IMG_H, IMG_W), np.inf, np.float64)
    xyz_buf = np.zeros((IMG_H, IMG_W, 3), np.float32)
    valid = np.zeros((IMG_H, IMG_W), bool)

    # Pixel grid -> normalised viewing rays (computed once)
    uu, vv = np.meshgrid(np.arange(IMG_W), np.arange(IMG_H))
    ones = np.ones_like(uu, dtype=np.float64)
    pix = np.stack([uu.astype(np.float64), vv.astype(np.float64), ones], axis=-1)
    rays = pix @ CAMERA_K_INV.T                      # (H, W, 3), z-component == 1

    for vidx, _color in SPACECRAFT_FACES:
        V = pts_cam[vidx]                            # face vertices in camera frame
        if V[:, 2].min() < 0.3:                      # behind / too close to camera
            continue
        # Plane through the first three vertices
        n = np.cross(V[1] - V[0], V[2] - V[0])
        nn = np.linalg.norm(n)
        if nn < 1e-9:
            continue
        n = n / nn
        d = float(n @ V[0])

        poly = pts_2d[vidx].astype(np.int32)
        mask = np.zeros((IMG_H, IMG_W), np.uint8)
        cv2.fillPoly(mask, [poly], 1)
        m = mask.astype(bool)
        if not m.any():
            continue

        denom = rays[m] @ n                          # (M,)
        ok = np.abs(denom) > 1e-9
        if not ok.any():
            continue
        s = np.full(denom.shape, np.inf)
        s[ok] = d / denom[ok]
        front = ok & (s > 0.3) & np.isfinite(s)
        if not front.any():
            continue

        X_cam = rays[m][front] * s[front][:, None]   # (M', 3)
        depth = X_cam[:, 2]

        idx = np.flatnonzero(m.ravel())[front]
        cur = depth_buf.ravel()[idx]
        closer = depth < cur
        if not closer.any():
            continue
        sel = idx[closer]

        X_model = (X_cam[closer] - tvec.reshape(1, 3)) @ R   # R^T x  ==  x @ R
        depth_buf.ravel()[sel] = depth[closer]
        xyz_buf.reshape(-1, 3)[sel] = X_model.astype(np.float32)
        valid.ravel()[sel] = True

    return xyz_buf, valid



# Degradations

def degrade_motion_blur(img, sev):
    k = int(round(sev))
    if k < 3:
        return img.copy()
    if k % 2 == 0:
        k += 1
    kern = np.zeros((k, k), np.float32)
    kern[k // 2, :] = 1.0 / k
    return cv2.filter2D(img, -1, kern)


def degrade_brightness(img, sev):
    return np.clip(img.astype(np.float32) * sev, 0, 255).astype(np.uint8)


def degrade_noise(img, sev):
    if sev <= 0:
        return img.copy()
    rng = np.random.default_rng(0)
    out = img.astype(np.float32) + rng.normal(0, sev, img.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


def degrade_occlusion(img, sev, rng=None):
    if sev <= 0:
        return img.copy()
    out = img.copy()
    h = int(IMG_H * np.sqrt(sev))
    w = int(IMG_W * np.sqrt(sev))
    if rng is None:
        rng = np.random.default_rng(7)
    x0 = int(rng.integers(0, max(1, IMG_W - w)))
    y0 = int(rng.integers(0, max(1, IMG_H - h)))
    out[y0:y0 + h, x0:x0 + w] = 0
    return out


DEGRADATIONS = {
    "motion_blur": (degrade_motion_blur, [0, 5, 9, 13, 17, 23]),
    "brightness":  (degrade_brightness,  [1.0, 0.7, 0.45, 0.3, 1.5, 1.9]),
    "noise":       (degrade_noise,       [0, 10, 25, 45, 65, 90]),
    "occlusion":   (degrade_occlusion,   [0.0, 0.10, 0.20, 0.32, 0.45, 0.55]),
}



# Pose estimation and error

def pose_error(rv_e, tv_e, rv_g, tv_g):
    t = float(np.linalg.norm(np.asarray(tv_e).flatten() - np.asarray(tv_g).flatten()))
    Re = cv2.Rodrigues(np.asarray(rv_e, np.float64))[0]
    Rg = cv2.Rodrigues(np.asarray(rv_g, np.float64))[0]
    c = (np.trace(Re @ Rg.T) - 1.0) / 2.0
    return t, float(np.degrees(np.arccos(float(np.clip(c, -1, 1)))))


def pose_pipeline(pts_3d, pts_2d, reproj_thresh=3.0, iters=200):
    out = {"rvec": None, "tvec": None, "n_inliers": 0,
           "inlier_ratio": 0.0, "inlier_reproj_err": float("inf"),
           "n_matches": len(pts_3d)}
    if len(pts_3d) < 6:
        return out
    try:
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            np.asarray(pts_3d, np.float64), np.asarray(pts_2d, np.float64),
            CAMERA_K, CAMERA_DIST,
            iterationsCount=iters, reprojectionError=reproj_thresh,
            confidence=0.99, flags=cv2.SOLVEPNP_EPNP)
    except cv2.error:
        return out
    if not ok or inliers is None or len(inliers) < 4:
        return out
    inliers = inliers.flatten()
    out["rvec"] = rvec.flatten(); out["tvec"] = tvec.flatten()
    out["n_inliers"] = int(len(inliers))
    out["inlier_ratio"] = float(len(inliers) / len(pts_3d))
    proj, _ = cv2.projectPoints(np.asarray(pts_3d, np.float64)[inliers],
                                rvec, tvec, CAMERA_K, CAMERA_DIST)
    diff = proj.reshape(-1, 2) - np.asarray(pts_2d)[inliers]
    out["inlier_reproj_err"] = float(np.linalg.norm(diff, axis=1).mean())
    return out


def match_to_model(detected_2d, gt_kp_2d, radius=8.0):
    """Oracle-assisted association used by Exp 1, 2 and 3, kept for comparison."""
    if len(detected_2d) == 0:
        return np.empty((0, 3)), np.empty((0, 2))
    m3d, m2d = [], []
    for i, gt in enumerate(gt_kp_2d):
        if not (0 <= gt[0] < IMG_W and 0 <= gt[1] < IMG_H):
            continue
        d = np.linalg.norm(detected_2d - gt, axis=1)
        b = int(np.argmin(d))
        if d[b] < radius:
            m3d.append(SPACECRAFT_3D[i]); m2d.append(detected_2d[b])
    if m3d:
        return np.array(m3d), np.array(m2d)
    return np.empty((0, 3)), np.empty((0, 2))



# Calibration metrics

def expected_calibration_error(confidence, correct, n_bins=10):
    confidence = np.asarray(confidence, float)
    correct = np.asarray(correct, bool)
    edges = np.linspace(0, 1, n_bins + 1)
    ece, N = 0.0, len(confidence)
    rows = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (confidence > lo) & (confidence <= hi) if i > 0 else (confidence >= lo) & (confidence <= hi)
        if m.sum() == 0:
            rows.append((0.5 * (lo + hi), np.nan, 0)); continue
        acc = correct[m].mean()
        conf = 0.5 * (lo + hi)
        ece += (m.sum() / N) * abs(conf - acc)
        rows.append((conf, acc, int(m.sum())))
    return float(ece), rows



# Surface detail: view-consistent 3D features so appearance matching is possible

def _build_surface_patches(seed=3):
    """Small planar patches at FIXED model-space positions on the body faces.

    Real spacecraft carry MLI seams, instrument boxes, connectors and markings.
    The wireframe target used in Exp 1, 2 and 3had flat untextured faces, which makes
    descriptor matching close to impossible and 180-degree symmetric. These
    patches are deterministic in model coordinates, so they appear consistently
    from every viewpoint - which is what makes them usable as landmarks.
    """
    rng = np.random.default_rng(seed)
    patches = []   # (4x3 model coords, BGR colour)
    faces_local = [
        # (origin, u-axis, v-axis) spanning each body face
        (np.array([-0.5,-0.5,-0.75]), np.array([1,0,0]), np.array([0,1,0])),   # -Z
        (np.array([-0.5,-0.5, 0.75]), np.array([1,0,0]), np.array([0,1,0])),   # +Z
        (np.array([-0.5,-0.5,-0.75]), np.array([1,0,0]), np.array([0,0,1])),   # -Y
        (np.array([-0.5, 0.5,-0.75]), np.array([1,0,0]), np.array([0,0,1])),   # +Y
        (np.array([ 0.5,-0.5,-0.75]), np.array([0,1,0]), np.array([0,0,1])),   # +X
        (np.array([-0.5,-0.5,-0.75]), np.array([0,1,0]), np.array([0,0,1])),   # -X
    ]
    for fi, (o, u, v) in enumerate(faces_local):
        ul = np.linalg.norm(u); vl = np.linalg.norm(v)
        for _ in range(5):
            a = rng.uniform(0.10, 0.70); b = rng.uniform(0.10, 0.70)
            w = rng.uniform(0.10, 0.22); h = rng.uniform(0.10, 0.22)
            # face extents differ per axis; scale by the face size
            su = ul * (1.0 if ul > 1.1 else 1.0)
            sv = vl * (1.0 if vl > 1.1 else 1.0)
            p0 = o + u * (a * su) + v * (b * sv)
            p1 = p0 + u * (w * su)
            p2 = p0 + u * (w * su) + v * (h * sv)
            p3 = p0 + v * (h * sv)
            shade = int(rng.integers(60, 230))
            col = (shade, int(shade * 0.9), int(shade * 0.8))
            patches.append((np.array([p0, p1, p2, p3], np.float32), col))
    return patches

SURFACE_PATCHES = _build_surface_patches()


def render_image_textured(rvec, tvec, rng=None, with_stars=True):
    """Renderer with view-consistent surface detail on the body faces."""
    if rng is None:
        rng = np.random.default_rng(0)
    img = np.zeros((IMG_H, IMG_W, 3), np.uint8)
    if with_stars:
        n = 80
        sx = rng.integers(0, IMG_W, n); sy = rng.integers(0, IMG_H, n)
        sb = rng.integers(80, 255, n)
        for i in range(n):
            cv2.circle(img, (int(sx[i]), int(sy[i])), 0, (int(sb[i]),) * 3, -1)
        img = cv2.add(img, rng.normal(0, 3, img.shape).clip(0, 30).astype(np.uint8))

    rvec = np.asarray(rvec, np.float64).reshape(3)
    tvec = np.asarray(tvec, np.float64).reshape(3)
    R = cv2.Rodrigues(rvec)[0]
    pts_cam = (R @ SPACECRAFT_3D.T).T + tvec.reshape(1, 3)
    pts_2d = project_points(SPACECRAFT_3D, rvec, tvec)

    faces = []
    for vidx, color in SPACECRAFT_FACES:
        d = pts_cam[vidx, 2].mean()
        if d < 0.5:
            continue
        faces.append((d, vidx, color))
    faces.sort(key=lambda x: -x[0])
    for d, vidx, color in faces:
        poly = pts_2d[vidx].astype(np.int32)
        if (poly[:, 0].max() < 0 or poly[:, 0].min() > IMG_W or
                poly[:, 1].max() < 0 or poly[:, 1].min() > IMG_H):
            continue
        shade = max(0.5, min(1.0, 12.0 / max(d, 4.0)))
        cv2.fillPoly(img, [poly], tuple(int(c * shade) for c in color))

    # surface patches, painter-sorted, back-face culled
    cam_pos_model = -R.T @ tvec
    ptmp = []
    for quad, col in SURFACE_PATCHES:
        qc = (R @ quad.T).T + tvec.reshape(1, 3)
        if qc[:, 2].min() < 0.5:
            continue
        nrm = np.cross(quad[1] - quad[0], quad[2] - quad[0])
        nrm = nrm / (np.linalg.norm(nrm) + 1e-9)
        if nrm @ (cam_pos_model - quad[0]) <= 0.02:      # facing away
            continue
        ptmp.append((qc[:, 2].mean(), quad, col))
    ptmp.sort(key=lambda x: -x[0])
    for d, quad, col in ptmp:
        p2 = project_points(quad, rvec, tvec).astype(np.int32)
        shade = max(0.5, min(1.0, 12.0 / max(d, 4.0)))
        cv2.fillPoly(img, [p2], tuple(int(c * shade) for c in col))

    for i, j in SPACECRAFT_EDGES:
        p1, p2 = pts_2d[i], pts_2d[j]
        if (-100 < p1[0] < IMG_W + 100 and -100 < p1[1] < IMG_H + 100 and
                -100 < p2[0] < IMG_W + 100 and -100 < p2[1] < IMG_H + 100):
            cv2.line(img, tuple(p1.astype(int)), tuple(p2.astype(int)),
                     (200, 200, 210), 1, cv2.LINE_AA)
    for panel_idx in [(8, 9, 10, 11), (12, 13, 14, 15)]:
        p = pts_2d[list(panel_idx)].astype(np.float32)
        for t in np.linspace(0.15, 0.85, 6):
            a = (1 - t) * p[0] + t * p[1]; b = (1 - t) * p[3] + t * p[2]
            cv2.line(img, tuple(a.astype(int)), tuple(b.astype(int)),
                     (80, 90, 140), 1, cv2.LINE_AA)
    return img


def target_mask(img_bgr, min_area=200):
    """Crude target segmentation, standing in for the detector stage of a
    three-stage pipeline. Rejects the star background so that matching is
    restricted to the spacecraft."""
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(g, (5, 5), 0)
    _, th = cv2.threshold(g, 25, 255, cv2.THRESH_BINARY)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    num, lab, stats, _ = cv2.connectedComponentsWithStats(th, 8)
    if num <= 1:
        return np.ones((IMG_H, IMG_W), bool)
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[k, cv2.CC_STAT_AREA] < min_area:
        return np.ones((IMG_H, IMG_W), bool)
    m = (lab == k)
    return cv2.dilate(m.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool)


# Coarse-to-fine template pre-selection

def silhouette_descriptor(img_bgr, size=32):
    """Cheap global descriptor: normalised, downsampled masked silhouette.

    Used to shortlist template views before running an expensive matcher on
    them. Matching every template with a transformer is not affordable, and a
    coarse-to-fine search is what a real system would do anyway.
    """
    m = target_mask(img_bgr).astype(np.uint8) * 255
    ys, xs = np.nonzero(m)
    if len(ys) < 20:
        return np.zeros(size * size, np.float32)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    crop = (g[y0:y1, x0:x1].astype(np.float32) *
            (m[y0:y1, x0:x1] > 0).astype(np.float32))
    small = cv2.resize(crop, (size, size)).ravel()
    small -= small.mean()
    n = np.linalg.norm(small)
    return (small / n).astype(np.float32) if n > 1e-6 else small.astype(np.float32)


def rank_templates(query_bgr, template_descs, topk=6):
    q = silhouette_descriptor(query_bgr)
    scores = template_descs @ q
    return np.argsort(-scores)[:topk]
