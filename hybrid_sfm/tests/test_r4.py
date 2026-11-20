"""Tests for R4 (cascade + conformal risk control). Run: python hybrid_sfm/tests/test_r4.py"""
import csv
import json
import os
import sys
import tempfile

import numpy as np

NL = chr(10)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, HERE)
from hybrid_sfm import r3_router as rr  # noqa: E402
from hybrid_sfm import r4_cascade as r4  # noqa: E402
from hybrid_sfm import r4b_gain as r4b  # noqa: E402
from test_r2 import _rec  # noqa: E402


def test_crc_guarantee_holds_on_exchangeable_data():
    """Scores informative but noisy; over many calibration/test draws the mean test risk stays <= alpha."""
    rng = np.random.default_rng(0)
    for alpha in (0.1, 0.2):
        risks = []
        for _ in range(400):
            s = rng.uniform(size=200)
            bad = rng.uniform(size=200) < 0.1 + 0.6 * s            # high score -> more likely a bad cheap choice
            lam = r4.crc_lambda(s[:100], bad[:100], alpha)
            risks.append(np.mean((s[100:] < lam) & bad[100:]))
        assert np.mean(risks) <= alpha + 0.01, (alpha, np.mean(risks))
    assert r4.crc_lambda([0.1, 0.2], [True, True], 0.05) == -np.inf     # too few points for alpha: escalate all
    assert r4.crc_lambda([0.1, 0.2, 0.3], [False] * 3, 0.5) == np.inf  # nothing bad: never escalate


def test_cascade_end_to_end():
    """Feature x says whether classical works; VGGT+BA is best but only visible through the probe."""
    rng = np.random.default_rng(2)
    rows, recs, probes = [], [], []
    for s in range(4):
        for i in range(10):
            c, x = f"sc{s}_c{i}", rng.uniform()
            rows.append(dict(chunk=c, scene=f"sc{s}", kind="window", cls_registered=x, conf_mean=rng.uniform()))
            ps = dict(classical=25.5 if x > 0.5 else 18.0, vggt=23.0, vggt_ba=23.0 + 3 * (i % 3 == 0))
            recs += [_rec(c, "ref", 26.0, wall=0)] + [_rec(c, m, v, wall=w) for (m, v), w in zip(ps.items(), (30, 70, 100))]
            for r in recs[-4:]:
                r["scene"] = f"sc{s}"
            probes += [dict(chunk=c, method=m, status="ok", probe={"1000": dict(psnr=v - 2, train_s=15.0)}) for m, v in ps.items()]
    with tempfile.TemporaryDirectory() as d:
        fp, lp, pp = (os.path.join(d, x) for x in ("f.csv", "l.jsonl", "p.jsonl"))
        with open(fp, "w", newline="") as fh:
            wr = csv.DictWriter(fh, list(rows[0])); wr.writeheader(); wr.writerows(rows)
        open(lp, "w").write("\n".join(json.dumps(r) for r in recs))
        open(pp, "w").write("\n".join(json.dumps(r) for r in probes))
        F, meta, names = rr.load_features(fp)
        T, P = rr.load_targets([lp]), rr.load_probes(pp)
    E = r4.evaluate(T, F, meta, names, P, seeds=(0,), n_splits=50)
    b = E["base"]
    assert abs(b["probe_all"]["loss"] - b["oracle"]["loss"]) < 1e-9 and b["probe_all"]["time"] == 30 + 70 + 100 + 45
    cv = E["curve"]["hindsight"]
    assert cv[0]["esc"] == 0 and cv[-1]["esc"] == 1 and abs(cv[-1]["loss"] - b["oracle"]["loss"]) < 1e-9
    assert cv[0]["time"] < cv[-1]["time"]
    for k in ("pred_loss", "random"):
        assert E["crc"][k][0.05]["esc"] >= E["crc"][k][0.3]["esc"]               # stricter alpha -> more escalation
    md = r4.report(E, "fig.png")
    assert "Conformal risk control" in md and "hindsight" in md


def test_gain_cascade_end_to_end():
    """VGGT+BA is much better on every third chunk, and the early probe of VGGT reveals which (low VGGT PSNR there)."""
    rng = np.random.default_rng(4)
    rows, recs, probes = [], [], []
    for s in range(4):
        for i in range(12):
            c, hard = f"sc{s}_c{i}", i % 3 == 0
            rows.append(dict(chunk=c, scene=f"sc{s}", kind="window", cls_registered=rng.uniform(), conf_mean=rng.uniform()))
            ps = dict(classical=20.0, vggt=19.0 if hard else 23.0, vggt_ba=25.0 if hard else 22.5)
            recs += [_rec(c, "ref", 26.0, wall=0)] + [_rec(c, m, v, wall=w) for (m, v), w in zip(ps.items(), (30, 70, 100))]
            for r in recs[-4:]:
                r["scene"] = f"sc{s}"
            probes += [dict(chunk=c, method=m, status="ok", probe={"250": dict(psnr=v - 4, train_s=4.0),
                                                                    "1000": dict(psnr=v - 2, train_s=16.0)}) for m, v in ps.items()]
    with tempfile.TemporaryDirectory() as d:
        fp, lp, pp = (os.path.join(d, x) for x in ("f.csv", "l.jsonl", "p.jsonl"))
        with open(fp, "w", newline="") as fh:
            wr = csv.DictWriter(fh, list(rows[0])); wr.writeheader(); wr.writerows(rows)
        open(lp, "w").write(NL.join(json.dumps(r) for r in recs))
        open(pp, "w").write(NL.join(json.dumps(r) for r in probes))
        F, meta, names = rr.load_features(fp)
        T, P = rr.load_targets([lp]), rr.load_probes(pp)
    E = r4b.evaluate(T, F, meta, names, P, early=250, seeds=(0,), n_splits=50)
    b = E["base"]
    assert b["stage2"]["time"] == 30 + 70 + 100 + 48 and abs(b["stage2"]["loss"] - b["oracle"]) < 1e-9
    # the early probe already exposes the hard chunks, so stage 1.5 picks VGGT+BA there without probing it:
    assert abs(b["stage15"]["loss"] - b["oracle"]) < 1e-9 and b["stage15"]["time"] < b["stage2"]["time"]
    assert b["helps"] == b["hurts"] == 0 and E["curve"][-1]["esc"] == 1.0 and E["curve"][0]["esc"] == 0.0
    assert "predicted gain" in r4b.report([E], None)


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f(); print("ok", k)
