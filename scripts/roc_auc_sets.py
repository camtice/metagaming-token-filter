"""Token-level ROC / AUC comparison of kept-feature sets.

Per-token scores over a kept set S (no boundary post-processing):
  vote  — count of features in S firing >= their own mean+4SD threshold
  cont  — sum over S of max(act - thr, 0) / sd   (continuous, full-range ROC)

TPR populations: human span tokens, held-out c2 tokens, all held-out forget.
FPR population: held-out control tokens. AUC by trapezoid over the full swept
range; pAUC = area over FPR in [0, cap] / cap (McClish-standardised).
Doc-clustered bootstrap SEs for AUC (200 resamples).
"""
import json

import numpy as np

ROOT = "/home/a5k/cwtice.a5k/sae-exploration"
role = json.load(open(f"{ROOT}/data/splits/gen_split_20260815.json"))["roles"]
docs = [json.loads(l) for l in open(f"{ROOT}/data/test_docs_v5.jsonl")]
by_id = {d["id"]: d for d in docs}
POOLJ = json.load(open(f"{ROOT}/data/candidate_sets/haiku_fable_forget_latents_v1.json"))["sets"]

import sys
WIDTH = sys.argv[1] if len(sys.argv) > 1 else "65k"
HARV = f"{ROOT}/out/harvest_test_v5" + ("_65k" if WIDTH == "65k" else "")
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
m_c2 = trole == "gtest_c2"
m_gd = trole == "gtest_dense"
m_ctl = trole == "gtest_ctl"
m_all = m_span | m_gd | m_c2
m_trd = trole == "train_dense"; m_trc = trole == "train_ctl"
li_all = z["lat_idx"]; ab_all = av >= thr[li_all]

cf = np.zeros(W); cc = np.zeros(W)
np.add.at(cf, li_all[m_trd[flat] & ab_all], 1)
np.add.at(cc, li_all[m_trc[flat] & ab_all], 1)
rr = ((cf + 0.5) / max(int(m_trd.sum()), 1)) / ((cc + 0.5) / max(int(m_trc.sum()), 1))

judge_members = np.array(sorted({int(l) for l, _c, _f in POOLJ["haiku_v8_65k" if WIDTH == "65k" else "haiku_v8_16k"]["latents"]}))
pm_judge = np.zeros(W, bool); pm_judge[judge_members] = True
pm_all = np.ones(W, bool)

sd_safe = np.where(sd > 0, sd, 1.0)


def scores(keep_mask):
    km = keep_mask[li_all]
    fl2 = flat[km]; li2 = li_all[km]; a2 = av[km]
    vote = np.bincount(fl2[a2 >= thr[li2]], minlength=n_tok).astype(np.float64)
    excess = np.maximum(a2 - thr[li2], 0.0) / sd_safe[li2]
    cont = np.zeros(n_tok)
    np.add.at(cont, fl2, excess)
    return vote, cont


def roc(score, pos_mask, neg_mask):
    """Full ROC via sorted unique thresholds (descending)."""
    s_pos = score[pos_mask]; s_neg = score[neg_mask]
    ts = np.unique(np.concatenate([s_pos, s_neg]))[::-1]
    # For token scores most mass is at 0; cap threshold count for speed
    if len(ts) > 4000:
        ts = np.quantile(ts, np.linspace(0, 1, 4000))[::-1]
    tpr = np.array([(s_pos >= t).mean() for t in ts])
    fpr = np.array([(s_neg >= t).mean() for t in ts])
    tpr = np.concatenate([[0.0], tpr, [1.0]])
    fpr = np.concatenate([[0.0], fpr, [1.0]])
    order = np.argsort(fpr, kind="stable")
    return fpr[order], tpr[order]


def auc(fpr, tpr, cap=None):
    if cap is not None:
        m = fpr <= cap
        f, t = fpr[m], tpr[m]
        if len(f) == 0 or f[-1] < cap:
            # interpolate the endpoint at fpr = cap
            j = np.searchsorted(fpr, cap)
            j = min(j, len(fpr) - 1)
            t_cap = np.interp(cap, fpr, tpr)
            f = np.concatenate([f, [cap]]); t = np.concatenate([t, [t_cap]])
        return float(np.trapezoid(t, f) / cap)
    return float(np.trapezoid(tpr, fpr))


def _rankdata(a):
    """Average ranks with ties (numpy-only)."""
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), float)
    sa = a[order]
    i = 0
    while i < len(sa):
        j = i
        while j + 1 < len(sa) and sa[j + 1] == sa[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def auc_mw(score, pos_mask, neg_mask):
    """Exact full AUC via Mann-Whitney with tie correction."""
    s = np.concatenate([score[pos_mask], score[neg_mask]])
    npos = int(pos_mask.sum()); nneg = int(neg_mask.sum())
    r = _rankdata(s)
    return float((r[:npos].sum() - npos * (npos + 1) / 2) / (npos * nneg))


SETS = {
    "judge rho>=5": pm_judge & (rr >= 5),
    "judge rho>=10": pm_judge & (rr >= 10),
    "judge pool (no screen)": pm_judge,
    "all rho>=20": pm_all & (rr >= 20),
    "all rho>=35": pm_all & (rr >= 35),
    "full dictionary (no screen)": pm_all,
}
POPS = {"span": m_span, "c2": m_c2, "all_forget": m_all}

rng = np.random.default_rng(20260829)
B = 200
ctl_docs = np.flatnonzero(rl == "gtest_ctl")
pop_docs = {"span": np.flatnonzero(rl == "gtest_span"),
            "c2": np.flatnonzero(rl == "gtest_c2"),
            "all_forget": np.flatnonzero(np.isin(rl, ["gtest_span", "gtest_dense", "gtest_c2"]))}

out = {"meta": {"score_defs": {"vote": "count of kept features >= mean+4SD",
                               "cont": "sum of max(act-thr,0)/sd over kept features"},
                "fpr_pop": "held-out control tokens",
                "pauc": "standardised partial AUC, FPR in [0, 0.015] and [0, 0.05]"}}

for sname, keep in SETS.items():
    vote, cont = scores(keep)
    row = {"n": int(keep.sum())}
    for scname, sc in (("vote", vote), ("cont", cont)):
        for pname, pmask in POPS.items():
            fpr, tpr = roc(sc, pmask, m_ctl)
            full = auc_mw(sc, pmask, m_ctl)
            row[f"{scname}_auc_{pname}"] = round(full, 4)
            row[f"{scname}_pauc015_{pname}"] = round(auc(fpr, tpr, cap=0.015), 4)
            row[f"{scname}_pauc05_{pname}"] = round(auc(fpr, tpr, cap=0.05), 4)
            if scname == "cont":
                # decimated polyline for plotting: dense at low FPR (log grid)
                gridf = np.unique(np.concatenate(
                    [np.logspace(-5, 0, 160), [0.0, 1.0]]))
                ti = np.interp(gridf, fpr, tpr)
                row[f"curve_{pname}"] = {
                    "fpr": [round(float(x), 6) for x in gridf],
                    "tpr": [round(float(x), 4) for x in ti]}
    # clustered bootstrap SE for the headline (cont, span & all_forget, full AUC)
    doc_score_cache = cont
    ses = {}
    for pname in ("span", "all_forget"):
        vals = np.zeros(B)
        for b in range(B):
            cd = ctl_docs[rng.integers(0, len(ctl_docs), len(ctl_docs))]
            pdx = pop_docs[pname]
            pd_ = pdx[rng.integers(0, len(pdx), len(pdx))]
            neg_idx = np.concatenate([np.arange(tok_base[i], tok_base[i + 1]) for i in cd])
            pos_parts = []
            for i in pd_:
                seg = np.arange(tok_base[i], tok_base[i + 1])
                if pname == "span":
                    seg = seg[m_span[seg]]
                else:
                    seg = seg[m_all[seg]]
                pos_parts.append(seg)
            pos_idx = np.concatenate(pos_parts)
            sneg = doc_score_cache[neg_idx]; spos = doc_score_cache[pos_idx]
            r = _rankdata(np.concatenate([spos, sneg]))
            npos, nneg = len(spos), len(sneg)
            vals[b] = (r[:npos].sum() - npos * (npos + 1) / 2) / (npos * nneg)
        ses[pname] = round(float(vals.std()), 4)
    row["se_cont_auc"] = ses
    out[sname] = row
    print(sname, {k: v for k, v in row.items() if "cont_auc" in k or k == "n"}, flush=True)

json.dump(out, open(f"{ROOT}/out/roc_auc_sets_{WIDTH}.json", "w"), indent=1)
print("ROC_AUC_DONE")
