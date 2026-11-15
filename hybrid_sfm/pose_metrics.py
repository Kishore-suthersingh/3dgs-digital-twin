"""Camera pose accuracy against a reference, numpy only.

Metrics (the usual ones in the VGGT / VGGSfM / MASt3R-SfM papers):
- RRA@τ / RTA@τ: share of image pairs whose relative rotation / relative translation direction error is < τ°.
  Translation direction is sign-ambiguous for some methods, so its error is min(e, 180 - e).
- AUC@30: mean over τ = 1..30 of the share of pairs with max(rotation error, translation error) < τ.
- ATE: RMSE of camera centres after a Sim(3) (Umeyama) alignment to the reference, divided by the
  reference's scene scale (median distance of reference centres to their mean), in percent.
Images the method failed to register count as 180° error for every pair they are in, so methods that
drop images are not rewarded for it; the share of registered images is reported too.
"""
import numpy as np


def _angle(R):
    return np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)))


def _vec_angle(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 180.0
    return np.degrees(np.arccos(np.clip(a @ b / (na * nb), -1.0, 1.0)))


def umeyama(src, dst):
    """s, R, t minimising ||s R src + t - dst||^2 (src, dst: (N, 3))."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    U, S, Vt = np.linalg.svd(xd.T @ xs / len(src))
    D = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[2, 2] = -1
    R = U @ D @ Vt
    var_s = (xs ** 2).sum() / len(src)
    s = np.trace(np.diag(S) @ D) / var_s if var_s > 0 else 1.0
    return s, R, mu_d - s * R @ mu_s


def centre(R, t):
    return -R.T @ t


def pose_errors(pred, ref, names=None):
    """pred, ref: {name: (R, t)} world-to-camera. names: the image set to score (default: all of ref)."""
    names = sorted(ref if names is None else names)
    reg = [n for n in names if n in pred]
    out = dict(n_images=len(names), n_registered=len(reg), registered=len(reg) / max(len(names), 1))

    rot_err, tr_err = [], []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            if a not in pred or b not in pred:
                rot_err.append(180.0); tr_err.append(180.0)
                continue
            (Ra, ta), (Rb, tb) = ref[a], ref[b]
            (Pa, pa), (Pb, pb) = pred[a], pred[b]
            R_ref, R_pred = Rb @ Ra.T, Pb @ Pa.T                       # camera a -> camera b
            t_ref, t_pred = tb - R_ref @ ta, pb - R_pred @ pa
            rot_err.append(_angle(R_pred.T @ R_ref))
            e = _vec_angle(t_pred, t_ref)
            tr_err.append(min(e, 180.0 - e))
    rot_err, tr_err = np.array(rot_err), np.array(tr_err)
    worst = np.maximum(rot_err, tr_err)
    for tau in (5, 15):
        out[f"RRA@{tau}"] = float((rot_err < tau).mean()) if len(worst) else float("nan")
        out[f"RTA@{tau}"] = float((tr_err < tau).mean()) if len(worst) else float("nan")
    out["AUC@30"] = float(np.mean([(worst < t).mean() for t in range(1, 31)])) if len(worst) else float("nan")
    out["rot_err_median"] = float(np.median(rot_err)) if len(rot_err) else float("nan")

    if len(reg) >= 3:
        C_ref = np.array([centre(*ref[n]) for n in reg])
        C_pred = np.array([centre(*pred[n]) for n in reg])
        s, R, t = umeyama(C_pred, C_ref)
        res = (s * C_pred @ R.T + t) - C_ref
        scale = np.median(np.linalg.norm(C_ref - C_ref.mean(0), axis=1))
        out["ATE_%"] = float(100 * np.sqrt((res ** 2).sum(1).mean()) / max(scale, 1e-12))
    else:
        out["ATE_%"] = float("nan")
    return out
