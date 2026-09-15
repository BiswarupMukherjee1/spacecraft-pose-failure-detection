"""
Experiment 6: Calibrated confidence for navigational filters

Experiments 1 to 5 measured confidence at the level of a single image. A
navigation system does not consume single images: it runs a filter over a
sequence, and what the filter needs from the vision front end is not only a
measurement but a measurement covariance R. A per-frame confidence signal is
useful exactly to the extent that it can set R.

This experiment closes that loop on a V-bar approach:

  dynamics     Hill-Clohessy-Wiltshire linearised relative motion, this
               model is used for the guidance and navigation design in
               GNC-ICATT 2023 paper on cooperative rendezvous and docking.
  measurement  a rendered camera image at each epoch, processed by the same
               keypoint and PnP chain as the Experiments 1, 2 and 3 for comparison.
  filter       an Extended Kalman Filter on relative position and velocity.
  campaign     a Monte Carlo of many runs with randomised degradation profiles.

Three filter configurations are compared:
  fixed R      one hand-tuned measurement covariance for every frame
  adaptive R   R set per frame from the calibrated confidence signal
  gated        adaptive R plus rejection of frames below a confidence floor

Consistency is scored with the normalised estimation error squared (NEES),
which asks whether the covariance the filter reports actually matches the
errors it makes. A filter can be accurate and still inconsistent, and an
inconsistent filter misleads everything downstream.
"""
import numpy as np
import cv2
from scipy.linalg import expm

from sc_common import (SPACECRAFT_3D, IMG_W, IMG_H, render_image,
                       project_points, pose_pipeline, pose_error,
                       match_to_model, DEGRADATIONS)

# The target is exactly symmetric under a 180 degree rotation about its z axis:
# this permutation maps the keypoint set onto itself. Experiment 4 showed that
# appearance matching without an oracle confuses these two interpretations, and
# the resulting pose is wrong by roughly 180 degrees while still fitting the
# image well. The permutation is applied here at a controlled rate so that the
# filter is exposed to the same failure, produced by the real solver rather than
# by an invented error model.
SYMMETRY_PERM = np.array([2, 3, 0, 1, 6, 7, 4, 5, 13, 12, 15, 14, 9, 8, 11, 10])

# Hill frame: x radial (R-bar), y in-track (V-bar), z cross-track (H-bar).
MU_EARTH = 3.986004418e14
R_LEO = 6778e3                       # ~400 km altitude
N_ORBIT = float(np.sqrt(MU_EARTH / R_LEO ** 3))     # mean motion, rad/s

# Camera frame vs Hill frame: boresight along +V-bar (the approach axis).
#   x_cam = H-bar, y_cam = R-bar, z_cam = V-bar
C_CAM_HILL = np.array([[0., 0., 1.],
                       [1., 0., 0.],
                       [0., 1., 0.]])


def hcw_matrices(dt, n=N_ORBIT):
    """Discrete state transition for HCW relative motion (exact for LTI)."""
    A = np.zeros((6, 6))
    A[0, 3] = A[1, 4] = A[2, 5] = 1.0
    A[3, 0] = 3 * n ** 2
    A[3, 4] = 2 * n
    A[4, 3] = -2 * n
    A[5, 2] = -n ** 2
    return expm(A * dt)



# Truth trajectory

def simulate_truth(n_steps=120, dt=1.0, r0=None, v0=None, tumble_deg_s=0.15,
                   rng=None):
    """Propagate the true relative state and target attitude along the approach."""
    if rng is None:
        rng = np.random.default_rng(0)
    F = hcw_matrices(dt)
    if r0 is None:
        r0 = np.array([rng.normal(0, 0.3), 20.0, rng.normal(0, 0.3)])
    if v0 is None:
        v0 = np.array([0.0, -0.135, 0.0])          # closing along V-bar
    x = np.concatenate([r0, v0])
    axis = rng.normal(size=3); axis /= np.linalg.norm(axis)
    states, rvecs = [], []
    ang = rng.uniform(-0.3, 0.3)
    for k in range(n_steps):
        states.append(x.copy())
        rvecs.append(axis * ang)
        ang += np.deg2rad(tumble_deg_s) * dt
        x = F @ x
    return np.array(states), np.array(rvecs)


def hill_to_camera(r_hill):
    return C_CAM_HILL @ r_hill


def camera_to_hill(t_cam):
    return C_CAM_HILL.T @ t_cam



# Vision measurement

def vision_measurement(r_hill, rvec, detector, severity_profile, rng,
                       p_assoc_fail=0.0):
    """Render one frame, run the pose chain, return the measurement.

    Association uses the model keypoint projections, so that this experiment isolates the filter question
    rather than re-testing the matching question of Experiment 4.
    """
    tvec = hill_to_camera(r_hill)
    if tvec[2] < 1.0:
        return None
    img = render_image(rvec, tvec, rng=rng)
    name, sev = severity_profile
    if sev is not None and name is not None:
        fn, _ = DEGRADATIONS[name]
        img = fn(img, sev, rng) if name == "occlusion" else fn(img, sev)
    gray_kp = project_points(SPACECRAFT_3D, rvec, tvec)
    det = detector(img)
    model_pts = SPACECRAFT_3D
    flipped = rng.random() < p_assoc_fail
    if flipped:
        model_pts = SPACECRAFT_3D[SYMMETRY_PERM]
    P3, P2 = [], []
    for i, gt in enumerate(gray_kp):
        if not (0 <= gt[0] < IMG_W and 0 <= gt[1] < IMG_H) or len(det) == 0:
            continue
        d = np.linalg.norm(det - gt, axis=1)
        b = int(np.argmin(d))
        if d[b] < 8.0:
            P3.append(model_pts[i]); P2.append(det[b])
    if len(P3) < 6:
        return None
    r = pose_pipeline(np.array(P3), np.array(P2))
    r["assoc_flipped"] = bool(flipped)
    if r["rvec"] is None:
        return None
    r["r_hill"] = camera_to_hill(r["tvec"])
    r["range"] = float(np.linalg.norm(r_hill))
    _, r["att_err_deg"] = pose_error(r["rvec"], r["tvec"], rvec, tvec)
    return r



# Calibration table: confidence -> measurement covariance

def build_R_table(conf, err_vec, n_bins=6, floor=1e-4):
    """Empirical map from confidence to per-axis position error variance.

    This is the operational use of a calibrated confidence signal: not to decide
    whether a number is pretty, but to say how much the filter should believe it.
    """
    conf = np.asarray(conf); err_vec = np.asarray(err_vec)
    edges = np.quantile(conf, np.linspace(0, 1, n_bins + 1))
    edges[0] -= 1e-9; edges[-1] += 1e-9
    table = []
    for i in range(n_bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1])
        if m.sum() < 5:
            table.append(None); continue
        var = err_vec[m].var(axis=0) + floor
        table.append(var)
    # fill gaps with the nearest populated bin
    filled = [t for t in table if t is not None]
    if not filled:
        filled = [np.full(3, 1.0)]
    for i in range(len(table)):
        if table[i] is None:
            table[i] = filled[min(i, len(filled) - 1)]
    return edges, table


def lookup_R(conf, edges, table):
    i = int(np.searchsorted(edges, conf, side="left") - 1)
    i = max(0, min(len(table) - 1, i))
    return np.diag(table[i])



# EKF

class RelativeNavEKF:
    """EKF on relative position and velocity in the Hill frame.

    The measurement is the position recovered by the vision chain, so the
    measurement model is linear and the filter is a Kalman filter;
    it is kept in EKF form because a range or bearing measurement would make it
    genuinely non-linear.
    """

    def __init__(self, dt=1.0, q_accel=2e-5, P0_pos=2.0, P0_vel=0.05):
        self.F = hcw_matrices(dt)
        self.dt = dt
        G = np.zeros((6, 3))
        G[3:, :] = np.eye(3) * dt
        self.Q = G @ (np.eye(3) * q_accel ** 2) @ G.T
        self.Q[:3, :3] += np.eye(3) * (0.5 * q_accel * dt ** 2) ** 2
        self.H = np.zeros((3, 6)); self.H[:, :3] = np.eye(3)
        self.P0 = np.diag([P0_pos ** 2] * 3 + [P0_vel ** 2] * 3)

    def reset(self, x0):
        self.x = np.asarray(x0, float).copy()
        self.P = self.P0.copy()

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z, R):
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        A = np.eye(6) - K @ self.H
        self.P = A @ self.P @ A.T + K @ R @ K.T          # Joseph form
        return y, S



# Monte Carlo campaign

def random_severity_profile(rng, n_steps, hard=True):
    """A degradation that varies along the approach, as illumination would.

    Biased towards the severe end of each degradation. At mild severities the
    front end almost never fails, the confidence signal has no spread, and the
    filter question becomes uninteresting; the operationally relevant regime is
    the one where measurements are sometimes bad.
    """
    name = str(rng.choice(list(DEGRADATIONS.keys())))
    fn, levels = DEGRADATIONS[name]
    n_lev = len(levels)
    if hard:
        lo = int(rng.integers(max(0, n_lev - 4), n_lev - 1))
        hi = n_lev - 1
    else:
        lo = int(rng.integers(0, 3))
        hi = int(rng.integers(lo, n_lev))
    idx = np.linspace(lo, hi, n_steps).round().astype(int)
    if rng.random() < 0.5:
        idx = idx[::-1]
    return [(name, levels[min(int(i), n_lev - 1)]) for i in idx]


def run_single(detector, mode, edges, table, R_fixed, seed=0, n_steps=120,
               dt=1.0, conf_floor=0.35, p_assoc_fail=0.12):
    """One Monte Carlo run. mode in {'fixed', 'adaptive', 'gated'}."""
    rng = np.random.default_rng(seed)
    truth, rvecs = simulate_truth(n_steps=n_steps, dt=dt, rng=rng)
    profile = random_severity_profile(rng, n_steps)

    ekf = RelativeNavEKF(dt=dt)
    x0 = truth[0].copy()
    x0[:3] += rng.normal(0, 1.0, 3)
    x0[3:] += rng.normal(0, 0.02, 3)
    ekf.reset(x0)

    rec = {"range": [], "err": [], "nees": [], "used": [], "sigma": [],
           "conf": [], "att_err": [], "flipped": [], "meas_err": []}
    for k in range(n_steps):
        if k > 0:
            ekf.predict()
        m = vision_measurement(truth[k, :3], rvecs[k], detector, profile[k], rng,
                               p_assoc_fail=p_assoc_fail)
        used = False
        if m is None or m["rvec"] is None:
            rec["conf"].append(np.nan); rec["att_err"].append(np.nan)
            rec["flipped"].append(False); rec["meas_err"].append(np.nan)
        else:
            rec["conf"].append(m["inlier_ratio"])
            rec["att_err"].append(m["att_err_deg"])
            rec["flipped"].append(m["assoc_flipped"])
            rec["meas_err"].append(float(np.linalg.norm(m["r_hill"] - truth[k, :3])))
        if m is not None and m["rvec"] is not None:
            conf = m["inlier_ratio"]
            if mode == "fixed":
                R = R_fixed
                used = True
            elif mode == "adaptive":
                R = lookup_R(conf, edges, table)
                used = True
            else:
                if conf >= conf_floor:
                    R = lookup_R(conf, edges, table)
                    used = True
            if used:
                ekf.update(m["r_hill"], R)
        e = ekf.x[:3] - truth[k, :3]
        Ppos = ekf.P[:3, :3]
        try:
            nees = float(e @ np.linalg.inv(Ppos) @ e)
        except np.linalg.LinAlgError:
            nees = np.nan
        rec["range"].append(float(np.linalg.norm(truth[k, :3])))
        rec["err"].append(float(np.linalg.norm(e)))
        rec["nees"].append(nees)
        rec["used"].append(bool(used))
        rec["sigma"].append(float(np.sqrt(np.trace(Ppos) / 3)))
    return {k: np.array(v) for k, v in rec.items()}
