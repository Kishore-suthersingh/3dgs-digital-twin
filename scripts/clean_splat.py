"""Crop and clean a 3DGS .ply, and export a small .splat for the web viewer.

CPU only (numpy), so it runs on the laptop without touching the GPU.

What it does, in order:
1. Fits the floor plane (RANSAC on opaque Gaussians around the object) and builds a
   level frame: origin on the floor under the object, "up" pointing away from the floor.
2. Keeps Gaussians inside a cylinder around the object (radius, height above floor).
3. Drops near-invisible Gaussians (low opacity), huge ones, and needles
   (longest axis much longer than the second longest: the streaks seen in SuperSplat).
4. Splits off the floor: everything outside the object's footprint, flat ground Gaussians under it,
   and below --foot-height everything that isn't on a leg (the floor seen under the object).
5. Writes the cleaned .ply (original coordinates, all SH kept, opens in SuperSplat)
   and a levelled, y-up .splat (32 bytes per Gaussian, base colour only) for the web viewer.

Example:
    python scripts/clean_splat.py "phase2_scene1_mcmc_1M_30k (1).ply" --out exports/table
"""
import argparse
import json
import os

import numpy as np

SH_C0 = 0.28209479177387814


# ---------- .ply I/O (binary little-endian, float properties, as gsplat writes them) ----------
def load_ply(path):
    with open(path, "rb") as f:
        header = b""
        while not header.endswith(b"end_header\n"):
            line = f.readline()
            if not line:
                raise ValueError("no end_header: not a .ply file")
            header += line
        text = header.decode("ascii")
        if "binary_little_endian" not in text:
            raise ValueError("only binary_little_endian .ply is supported")
        props = [l.split()[2] for l in text.splitlines() if l.startswith("property float")]
        n = int([l for l in text.splitlines() if l.startswith("element vertex")][0].split()[2])
        data = np.fromfile(f, dtype="<f4", count=n * len(props)).reshape(n, len(props))
    return props, data


def save_ply(path, props, data):
    header = "ply\nformat binary_little_endian 1.0\nelement vertex %d\n" % len(data)
    header += "".join(f"property float {p}\n" for p in props) + "end_header\n"
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(np.ascontiguousarray(data, dtype="<f4").tobytes())


# ---------- geometry helpers ----------
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def fit_plane_ransac(pts, iters=2000, thresh=0.01, seed=0):
    """Return (normal, d) with normal·p + d = 0, refined by least squares on the inliers."""
    rng = np.random.default_rng(seed)
    best, best_count = None, -1
    for _ in range(iters):
        p = pts[rng.choice(len(pts), 3, replace=False)]
        n = np.cross(p[1] - p[0], p[2] - p[0])
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        count = int((np.abs(pts @ n - n @ p[0]) < thresh).sum())
        if count > best_count:
            best, best_count = (n, -n @ p[0]), count
    n, d = best
    inl = pts[np.abs(pts @ n + d) < thresh]
    c = inl.mean(0)
    n = np.linalg.svd(inl - c)[2][-1]          # smallest singular vector = refined normal
    return n, -n @ c, len(inl)


def quat_from_matrix(R):
    """Rotation matrix -> quaternion (w, x, y, z)."""
    w = np.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = np.copysign(np.sqrt(max(0.0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2, R[2, 1] - R[1, 2])
    y = np.copysign(np.sqrt(max(0.0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2, R[0, 2] - R[2, 0])
    z = np.copysign(np.sqrt(max(0.0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2, R[1, 0] - R[0, 1])
    return np.array([w, x, y, z])


def quat_to_mats(q):
    """(N, 4) quaternions (w, x, y, z) -> (N, 3, 3) rotation matrices; column i is the axis of scale_i."""
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q.T
    return np.stack([np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], 1),
                     np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], 1),
                     np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], 1)], 1)


def quat_mul(a, b):
    """Hamilton product a ⊗ b; a is (4,), b is (N, 4), both (w, x, y, z)."""
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b.T
    return np.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                     w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], 1)


def level_frame(xyz, opacity, center_xy_radius, floor_rin, floor_rout):
    """Floor plane + object centre -> (R, origin): local = R @ (p - origin), local z = height above floor."""
    solid = opacity > 0.1
    # rough object centre: median of solid Gaussians, then the floor ring around it
    c0 = np.median(xyz[solid], 0)
    core = solid & (np.linalg.norm(xyz - c0, axis=1) < center_xy_radius)
    c0 = np.median(xyz[core], 0)
    # first guess of "up": the thinnest direction of the solid core (works for a floor-dominated scene)
    w, v = np.linalg.eigh(np.cov((xyz[core] - c0).T))
    up0 = v[:, 0]
    rel = xyz - c0
    radial = np.linalg.norm(rel - np.outer(rel @ up0, up0), axis=1)
    ring = solid & (radial > floor_rin) & (radial < floor_rout)
    # floor = the plane that most ring Gaussians lie on
    n, d, n_inl = fit_plane_ransac(xyz[ring])
    # point "up" from the floor towards the object
    if n @ c0 + d < 0:
        n, d = -n, -d
    origin = c0 - (n @ c0 + d) * n               # object centre dropped onto the floor
    ex = np.cross(n, [0.0, 0.0, 1.0] if abs(n[2]) < 0.9 else [1.0, 0.0, 0.0])
    ex /= np.linalg.norm(ex)
    ey = np.cross(n, ex)
    R = np.stack([ex, ey, n])                     # rows: local x, y, z(up)
    tilt = np.degrees(np.arccos(abs(n @ up0)))
    return R, origin, dict(floor_inliers=int(n_inl), floor_ring=int(ring.sum()),
                           tilt_vs_first_guess_deg=round(float(tilt), 2))


# ---------- export ----------
def write_splat(path, xyz, scales, rgb, alpha, quats):
    """antimatter15 .splat: per Gaussian 3f pos, 3f scale, 4u8 rgba, 4u8 rot (w,x,y,z)."""
    order = np.argsort(-(scales.prod(1) * alpha))   # big, opaque first (viewer convention)
    rec = np.zeros(len(xyz), dtype=[("p", "<f4", 3), ("s", "<f4", 3), ("c", "u1", 4), ("r", "u1", 4)])
    rec["p"] = xyz[order]
    rec["s"] = scales[order]
    rec["c"][:, :3] = np.clip(rgb[order] * 255, 0, 255).round()
    rec["c"][:, 3] = np.clip(alpha[order] * 255, 0, 255).round()
    q = quats[order] / np.linalg.norm(quats[order], axis=1, keepdims=True)
    rec["r"] = np.clip(q * 128 + 128, 0, 255).round()
    rec.tofile(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ply")
    ap.add_argument("--out", required=True, help="output path without extension (writes .ply, _object.splat, _floor.splat, .json)")
    ap.add_argument("--radius", type=float, default=0.75, help="keep within this distance of the object, along the floor")
    ap.add_argument("--max-height", type=float, default=0.75, help="keep up to this height above the floor")
    ap.add_argument("--below", type=float, default=0.03, help="keep this far below the floor plane (floor thickness)")
    ap.add_argument("--min-opacity", type=float, default=0.002,
                    help="MCMC builds thin structure and the floor from many faint Gaussians: keep this low")
    ap.add_argument("--max-scale", type=float, default=0.08, help="drop Gaussians whose longest axis is longer than this")
    ap.add_argument("--needle-ratio", type=float, default=8.0, help="longest / second-longest axis above this ...")
    ap.add_argument("--needle-min", type=float, default=0.04,
                    help="... and longest axis above this = needle (wicker strands are thin needles ~0.01-0.03: keep them)")
    ap.add_argument("--needle-above", type=float, default=0.03,
                    help="only remove needles higher than this above the floor (floor Gaussians are long and flat)")
    ap.add_argument("--floor-height", type=float, default=0.03,
                    help="flat, horizontal Gaussians below this height go to the separate _floor.splat")
    ap.add_argument("--object-ply", action="store_true", help="also write <out>_object.ply (object only, full SH)")
    ap.add_argument("--footprint-margin", type=float, default=0.03,
                    help="everything outside the object's footprint ellipse (+ this margin) goes to _floor.splat")
    ap.add_argument("--foot-height", type=float, default=0.09,
                    help="below this height, only Gaussians on the legs stay in the object (0 = off)")
    ap.add_argument("--leg-cell", type=float, default=0.02, help="grid cell size used to find the legs")
    ap.add_argument("--center-radius", type=float, default=0.3, help="radius used to find the object centre")
    ap.add_argument("--floor-ring", type=float, nargs=2, default=[0.45, 0.9], help="ring used to fit the floor")
    a = ap.parse_args()

    props, data = load_ply(a.ply)
    col = {p: i for i, p in enumerate(props)}
    xyz = data[:, [col["x"], col["y"], col["z"]]].astype(np.float64)
    opacity = sigmoid(data[:, col["opacity"]])
    scales = np.exp(data[:, [col["scale_0"], col["scale_1"], col["scale_2"]]])
    quats = data[:, [col["rot_0"], col["rot_1"], col["rot_2"], col["rot_3"]]].astype(np.float64)

    R, origin, frame_info = level_frame(xyz, opacity, a.center_radius, *a.floor_ring)
    local = (xyz - origin) @ R.T
    radial, height = np.hypot(local[:, 0], local[:, 1]), local[:, 2]

    s_sorted = np.sort(scales, 1)
    needle = ((s_sorted[:, 2] / np.maximum(s_sorted[:, 1], 1e-9) > a.needle_ratio) & (s_sorted[:, 2] > a.needle_min)
              & (height > a.needle_above))
    rules = {
        "outside cylinder": (radial > a.radius) | (height > a.max_height) | (height < -a.below),
        "low opacity": opacity < a.min_opacity,
        "too large": s_sorted[:, 2] > a.max_scale,
        "needle": needle,
    }
    keep = np.ones(len(data), bool)
    print(f"{len(data):,} Gaussians in {a.ply}")
    print(f"floor: {frame_info}")
    for name, m in rules.items():
        print(f"  {name:<17} {m.sum():>9,}  (removed by this rule after earlier ones: {(m & keep).sum():,})")
        keep &= ~m
    print(f"kept {keep.sum():,} ({keep.mean():.0%})")

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    save_ply(a.out + ".ply", props, data[keep])

    # levelled, y-up copy for the web: local (x, y, up) -> three.js (x, up, -y)
    to_yup = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], float)
    M = to_yup @ R
    q_level = quat_from_matrix(M)
    rgb = np.clip(0.5 + SH_C0 * data[:, [col["f_dc_0"], col["f_dc_1"], col["f_dc_2"]]], 0, 1)
    pos_web = (xyz - origin) @ M.T
    quat_web = quat_mul(q_level, quats)

    # split off the scanned floor: flat Gaussians lying horizontally near the floor plane.
    # Object footprint: an ellipse fitted (on the floor plane) to the object's solid Gaussians above the floor.
    body = keep & (opacity > 0.1) & (height > a.floor_height + 0.05)
    fc = np.median(local[body][:, :2], 0)
    w, v = np.linalg.eigh(np.cov((local[body][:, :2] - fc).T))
    half = np.percentile(np.abs((local[body][:, :2] - fc) @ v), 99, axis=0)        # along v[:,0], v[:,1]
    uv = (local[:, :2] - fc) @ v
    inside = ((uv[:, 0] / (half[0] + a.footprint_margin)) ** 2 + (uv[:, 1] / (half[1] + a.footprint_margin)) ** 2) <= 1
    # Scanned floor = everything outside the footprint (floor and the wisps over it), plus the flat ground
    # layer under the object. Vertical leg Gaussians fail the "lying flat" test, so the table keeps its feet.
    mats = quat_to_mats(quats)
    thin_axis = mats[np.arange(len(data)), :, np.argmin(scales, 1)]          # direction of the smallest scale
    flat = (s_sorted[:, 0] < 0.3 * s_sorted[:, 1]) & (np.abs(thin_axis @ R[2]) > 0.8)
    floor = keep & (~inside | ((height < a.floor_height) & flat))
    # Under the object, the floor seen through gaps (and its reflections) is kept as streaky, non-flat
    # Gaussians that render as a grey smear. Below --foot-height, keep only the feet: grid cells near the
    # footprint rim that hold solid Gaussians a little above the floor. Everything else low goes to the floor.
    if a.foot_height > 0:
        rim = np.hypot(uv[:, 0] / half[0], uv[:, 1] / half[1]) > 0.6
        legs = keep & inside & rim & (opacity > 0.15) & (height > 0.03) & (height < a.foot_height)
        lo_xy = local[keep, :2].min(0) - a.leg_cell
        ij = np.floor((local[:, :2] - lo_xy) / a.leg_cell).astype(int)
        ij = np.clip(ij, 0, None)
        grid = np.zeros(ij[keep].max(0) + 1, int)
        np.add.at(grid, tuple(ij[legs].T), 1)
        occ = grid >= 2
        occ[1:] |= occ[:-1].copy(); occ[:-1] |= occ[1:].copy()           # dilate by one cell
        occ[:, 1:] |= occ[:, :-1].copy(); occ[:, :-1] |= occ[:, 1:].copy()
        ij = np.minimum(ij, np.array(occ.shape) - 1)
        on_leg = occ[ij[:, 0], ij[:, 1]] & rim
        under = keep & inside & ~floor & (height < a.foot_height) & ~on_leg
        print(f"under the object: {under.sum():,} low Gaussians off the legs -> floor ({occ.sum()} leg cells)")
        floor |= under
    obj = keep & ~floor
    write_splat(a.out + "_object.splat", pos_web[obj], scales[obj], rgb[obj], opacity[obj], quat_web[obj])
    write_splat(a.out + "_floor.splat", pos_web[floor], scales[floor], rgb[floor], opacity[floor], quat_web[floor])
    print(f"split: object {obj.sum():,} | scanned floor {floor.sum():,}")
    if a.object_ply:
        save_ply(a.out + "_object.ply", props, data[obj])          # full SH, original coordinates

    # for the viewer's plain floor: tile colour, and the footprint in web coordinates (x, z) = (local x, -local y)
    solid_floor = floor & (opacity > 0.05) & (height < a.floor_height)
    floor_rgb = np.median(rgb[solid_floor], 0) if solid_floor.sum() > 50 else np.array([0.8, 0.8, 0.8])
    flip = np.array([1.0, -1.0])
    footprint = dict(center=(fc * flip).round(4).tolist(), axis_long=(v[:, 1] * flip).round(4).tolist(),
                     half_long=round(float(half[1]), 4), half_short=round(float(half[0]), 4))
    low = obj & (opacity > 0.1)

    info = dict(source=os.path.basename(a.ply), total=int(len(data)), kept=int(keep.sum()),
                removed={k: int(v.sum()) for k, v in rules.items()}, params=vars(a), floor=frame_info,
                object_count=int(obj.sum()), floor_count=int(floor.sum()),
                floor_rgb=floor_rgb.round(3).tolist(), footprint=footprint,
                footprint_center_local=fc.round(4).tolist(),
                object_height=round(float(np.percentile(pos_web[low][:, 1], 99)), 3),
                origin=origin.round(5).tolist(), rotation_rows=R.round(6).tolist(),
                ply_mb=round(os.path.getsize(a.out + ".ply") / 1e6, 1),
                object_splat_mb=round(os.path.getsize(a.out + "_object.splat") / 1e6, 1),
                floor_splat_mb=round(os.path.getsize(a.out + "_floor.splat") / 1e6, 1))
    json.dump(info, open(a.out + ".json", "w"), indent=2)
    print(f"floor colour {info['floor_rgb']} | footprint {footprint}")
    print(f"wrote {a.out}.ply ({info['ply_mb']} MB), {a.out}_object.splat ({info['object_splat_mb']} MB), "
          f"{a.out}_floor.splat ({info['floor_splat_mb']} MB), {a.out}.json")


if __name__ == "__main__":
    main()
