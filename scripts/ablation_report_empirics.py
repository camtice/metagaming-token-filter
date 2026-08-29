"""Consolidated ablation empirics for the report:
  1. judge inclusion  — judge pool vs full dictionary, fine rho frontier, k=4
  2. vote k           — k in {1,2,3,4,6,8} at rho in {5,10}, judge pool
  3. boundary rule    — seed-only / paper adjacency growth / growth requiring
                        above-threshold activation / fixed +-n dilation,
                        at the champion cell (judge rho>=10, k=4)
All on the 65k TEST-v5 harvest under the G-split (train-only statistics,
held-out evaluation). Emits out/ablation_report_empirics.json.
"""
import json

import numpy as np

ROOT = "/home/a5k/cwtice.a5k/sae-exploration"
role = json.load(open(f"{ROOT}/data/splits/gen_split_20260815.json"))["roles"]
docs = [json.loads(l) for l in open(f"{ROOT}/data/test_docs_v5.jsonl")]
by_id = {d["id"]: d for d in docs}
POOLJ = json.load(open(f"{ROOT}/data/candidate_sets/haiku_fable_forget_latents_v1.json"))["sets"]

HARV = f"{ROOT}/out/harvest_test_v5_65k"
zf = np.load(HARV + ".npz"); z = {k: zf[k] for k in zf.files}
meta = json.load(open(HARV + ".meta.json")); W = meta["width"]
av = np.clip(z["act"].astype(np.float64), 0, 65504.0)
ntok = z["doc_ntok"].astype(np.int64); nd = len(meta["docs"])
tok_base = np.zeros(nd + 1, np.int64); np.cumsum(ntok, out=tok_base[1:])
n_tok = int(ntok.sum())
flat = tok_base[z["doc_idx"]] + z["tok_idx"]
offs = z["offsets"].reshape(-1, 2)
mu = np.bincount(z["lat_idx"], weights=av, minlength=W) / n_tok
sd = np.sqrt(np.maximum(np.bincount(z["lat_idx"], weights=av**2, minlength=W)/n_tok - mu**2, 0))
thr = mu + 4 * sd
doc_ids = [d["id"] for d in meta["docs"]]
rl = np.array([role[i] for i in doc_ids])
tok_doc = np.repeat(np.arange(nd), ntok)
trole = rl[tok_doc]
gt = np.zeros(n_tok, bool)
for di, did in enumerate(doc_ids):
    for cs, ce in by_id[did].get("char_spans", []):
        o = offs[tok_base[di]:tok_base[di + 1]]
        m = (o[:, 0] < ce) & (o[:, 1] > cs)
        gt[tok_base[di] + np.flatnonzero(m)] = True
m_span = (trole == "gtest_span") & gt
m_c2 = trole == "gtest_c2"; m_gd = trole == "gtest_dense"; m_ctl = trole == "gtest_ctl"
m_trd = trole == "train_dense"; m_trc = trole == "train_ctl"
isd = np.zeros(n_tok, bool); isd[tok_base[:-1]] = True
li_all = z["lat_idx"]; ab_all = av >= thr[li_all]

# train-only rate ratio over the full dictionary (pool masks applied later)
cf = np.zeros(W); cc = np.zeros(W)
np.add.at(cf, li_all[m_trd[flat] & ab_all], 1)
np.add.at(cc, li_all[m_trc[flat] & ab_all], 1)
rr = ((cf + 0.5) / max(int(m_trd.sum()), 1)) / ((cc + 0.5) / max(int(m_trc.sum()), 1))

gt_in = m_span | m_gd
mask_in = (trole == "gtest_span") | (trole == "gtest_clean") | m_gd | m_ctl
gt_a = gt_in | m_c2; mask_a = mask_in | m_c2
gt_o = m_c2; mask_o = m_c2 | m_ctl


def f2(pred, gtx, mask):
    tp = int((pred & gtx & mask).sum()); fpa = int((pred & mask & ~gtx).sum())
    fn = int((~pred & gtx).sum())
    P = tp / max(tp + fpa, 1); R = tp / max(tp + fn, 1)
    return round(5 * P * R / max(4 * P + R, 1e-9), 3)


# per-doc clustered bootstrap (stratified by role) for every reported metric
DOC_ROLE = rl
ROLE_GROUPS = {r: np.flatnonzero(DOC_ROLE == r) for r in np.unique(DOC_ROLE)}
B = 400
_rng = np.random.default_rng(20260829)
BOOT_DOCS = {r: _rng.integers(0, len(ix), (B, len(ix))) for r, ix in ROLE_GROUPS.items()}


def _perdoc(x):
    return np.bincount(tok_doc[x], minlength=nd)


def metrics(pred, n_keep):
    pd = {
        "span_num": _perdoc(pred & m_span), "span_den": _perdoc(m_span),
        "c2_num": _perdoc(pred & m_c2), "c2_den": _perdoc(m_c2),
        "ctl_num": _perdoc(pred & m_ctl), "ctl_den": _perdoc(m_ctl),
    }
    for tag, gtx, mask in (("in", gt_in, mask_in), ("out", gt_o, mask_o), ("all", gt_a, mask_a)):
        pd[f"tp_{tag}"] = _perdoc(pred & gtx & mask)
        pd[f"fp_{tag}"] = _perdoc(pred & mask & ~gtx)
        pd[f"fn_{tag}"] = _perdoc((~pred) & gtx)
    boots = {k: np.zeros(B) for k in ("Rspan", "Rc2", "fp", "F2in", "F2out", "F2all")}
    for b in range(B):
        tot = {k: 0.0 for k in pd}
        for r, ix in ROLE_GROUPS.items():
            take = ix[BOOT_DOCS[r][b]]
            for k, v in pd.items():
                tot[k] += float(v[take].sum())
        boots["Rspan"][b] = tot["span_num"] / max(tot["span_den"], 1)
        boots["Rc2"][b] = tot["c2_num"] / max(tot["c2_den"], 1)
        boots["fp"][b] = tot["ctl_num"] / max(tot["ctl_den"], 1)
        for tag, key in (("in", "F2in"), ("out", "F2out"), ("all", "F2all")):
            P = tot[f"tp_{tag}"] / max(tot[f"tp_{tag}"] + tot[f"fp_{tag}"], 1)
            Rc = tot[f"tp_{tag}"] / max(tot[f"tp_{tag}"] + tot[f"fn_{tag}"], 1)
            boots[key][b] = 5 * P * Rc / max(4 * P + Rc, 1e-9)
    return {"n": n_keep,
            "Rspan": round(float(pred[m_span].mean()), 3),
            "Rc2": round(float(pred[m_c2].mean()), 3),
            "Rin": round(float(pred[m_gd].mean()), 3),
            "fp": round(float(pred[m_ctl].mean()), 4),
            "F2in": f2(pred, gt_in, mask_in),
            "F2out": f2(pred, gt_o, mask_o),
            "F2all": f2(pred, gt_a, mask_a),
            "se": {k: round(float(v.std()), 4) for k, v in boots.items()}}


def rule(keep_mask, kseed, boundary="window"):
    """boundary: 'seed' (no growth), 'window' (paper: grow across any kept-
    feature-active token), 'window_ab' (grow only across above-threshold
    tokens), 'dilate1'/'dilate2'/'dilate4' (fixed +-n tokens, doc-bounded)."""
    km = keep_mask[li_all]
    fl2, ab2 = flat[km], ab_all[km]
    seeds = np.bincount(fl2[ab2], minlength=n_tok) >= kseed
    if boundary == "seed":
        return seeds
    if boundary.startswith("dilate"):
        n = int(boundary[6:])
        pred = seeds.copy()
        doc_of = tok_doc
        for s in range(1, n + 1):
            r = np.zeros(n_tok, bool); r[s:] = seeds[:-s]
            r &= doc_of == np.roll(doc_of, s)
            l = np.zeros(n_tok, bool); l[:-s] = seeds[s:]
            l &= doc_of == np.roll(doc_of, -s)
            pred |= r | l
        return pred
    base = ab2 if boundary == "window_ab" else np.ones_like(ab2, bool)
    pa = np.bincount(fl2[base], minlength=n_tok) > 0
    prev = np.zeros(n_tok, bool); prev[1:] = pa[:-1]
    start = pa & (~prev | isd)
    run_id = np.cumsum(start) - 1
    sr = np.zeros(max(int(start.sum()), 1), bool); sr[run_id[seeds]] = True
    return pa & sr[run_id]


judge_members = np.array(sorted({int(l) for l, _c, _f in POOLJ["haiku_v8_65k"]["latents"]}))
pm_judge = np.zeros(W, bool); pm_judge[judge_members] = True
pm_all = np.ones(W, bool)

out = {"meta": {"harvest": "out/harvest_test_v5_65k.npz",
                "sae": "google/gemma-scope-2-27b-it resid_post/layer_40_width_65k_l0_medium",
                "model": "unsloth/gemma-3-27b-it",
                "split": "data/splits/gen_split_20260815.json",
                "pool": "haiku_v8_65k (3,798 judged-forget latents)",
                "corpus_tokens": n_tok}}

# --- ablation 1: judge inclusion, rho frontier at k=4 ---
ab1 = {}
for pname, pm in (("judge", pm_judge), ("all", pm_all)):
    rhos = ([1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 20, 30]
            if pname == "judge" else
            [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 25, 30, 35, 40, 45, 50, 60, 75])
    for rho in rhos:
        keep = pm & (rr >= rho)
        row = metrics(rule(keep, 4), int(keep.sum()))
        row["rho"] = rho
        ab1[f"{pname} rho={rho}"] = row
        print("ab1", pname, rho, row["F2all"], row["fp"], flush=True)
out["judge_frontier_k4"] = ab1

# --- ablation 2: vote k at rho in {5, 10}, judge pool ---
ab2 = {}
for rho in (5, 10):
    keep = pm_judge & (rr >= rho)
    for k in (1, 2, 3, 4, 6, 8):
        row = metrics(rule(keep, k), int(keep.sum()))
        row["rho"] = rho; row["k"] = k
        ab2[f"rho={rho} k={k}"] = row
        print("ab2", rho, k, row["F2all"], row["fp"], flush=True)
out["vote_k"] = ab2

# --- ablation 3: boundary rule at the champion cell (judge rho>=10, k=4) ---
ab3 = {}
keep = pm_judge & (rr >= 10)
for b in ("seed", "dilate1", "dilate2", "dilate4", "window_ab", "window"):
    row = metrics(rule(keep, 4, boundary=b), int(keep.sum()))
    row["boundary"] = b
    ab3[b] = row
    print("ab3", b, row["F2all"], row["Rspan"], row["fp"], flush=True)
out["boundary"] = ab3

json.dump(out, open(f"{ROOT}/out/ablation_report_empirics.json", "w"), indent=1)
print("ABLATION_EMPIRICS_DONE")
