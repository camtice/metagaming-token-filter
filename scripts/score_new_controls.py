"""FP of frozen sets on the 2026-08-15 control refresh (2026-08-15).

New material: ctl_fineweb_new (3,750 fresh fineweb docs) + ctl_swe_code
(250 ordinary Dolci coding answers, the new `ctl_swe` config).

Rule: Rathi (seed >=2 above threshold + run windowing) with thresholds
FROZEN from the TEST harvest (mean+4SD over harvest_test_v3[_65k]) — i.e.
the deployed rule applied to fresh text, not re-derived thresholds.
"""
import json
import sys

import numpy as np

ROOT = "/home/a5k/cwtice.a5k/sae-exploration"

def load(prefix):
    z = np.load(prefix + ".npz")
    meta = json.load(open(prefix + ".meta.json"))
    av = z["act"].astype(np.float64); av[np.isinf(av)] = 65504.0
    return z, meta, av

def thresholds(prefix, W):
    z, meta, av = load(prefix)
    n_tok = int(z["doc_ntok"].sum())
    mu = np.bincount(z["lat_idx"], weights=av, minlength=W) / n_tok
    sd = np.sqrt(np.maximum(np.bincount(z["lat_idx"], weights=av**2, minlength=W)/n_tok - mu**2, 0))
    return mu + 4 * sd

def fp_on(prefix, thr, members, cfg_of):
    z, meta, av = load(prefix)
    ntok = z["doc_ntok"].astype(np.int64); nd = len(meta["docs"])
    tok_base = np.zeros(nd + 1, np.int64); np.cumsum(ntok, out=tok_base[1:])
    n_tok = int(ntok.sum())
    flat = tok_base[z["doc_idx"]] + z["tok_idx"]
    memv = np.zeros(len(thr), bool); memv[list(members)] = True
    sel = memv[z["lat_idx"]]
    li, fl, a = z["lat_idx"][sel], flat[sel], av[sel]
    seeds = np.bincount(fl[a >= thr[li]], minlength=n_tok) >= 2
    pa = np.bincount(fl, minlength=n_tok) > 0
    prev = np.zeros(n_tok, bool); prev[1:] = pa[:-1]
    isd = np.zeros(n_tok, bool); isd[tok_base[:-1]] = True
    start = pa & (~prev | isd)
    run_id = np.cumsum(start) - 1
    sr = np.zeros(max(int(start.sum()), 1), bool); sr[run_id[seeds]] = True
    pred = pa & sr[run_id]
    tok_doc = np.repeat(np.arange(nd), ntok)
    cfgs = np.array([cfg_of[d["id"]] for d in meta["docs"]])
    out = {}
    for c in sorted(set(cfgs)):
        m = cfgs[tok_doc] == c
        out[c] = round(float(pred[m].mean()), 4)
    out["overall"] = round(float(pred.mean()), 4)
    return out

docs = [json.loads(l) for l in open(f"{ROOT}/data/new_controls_aug15.jsonl")]
cfg_of = {d["id"]: d["config"] for d in docs}

def members_of(path, key="members"):
    d = json.load(open(path))
    return d[key] if key in d else d

W16, W65 = 16384, 65536
thr16 = thresholds(f"{ROOT}/out/harvest_test_v3", W16)
thr65 = thresholds(f"{ROOT}/out/harvest_test_v3_65k", W65)

sets16 = {
    "OG": sorted(int(k) for k in json.load(open(f"{ROOT}/data/latent_list_2026-07-29.json"))["tags"]),
    "P2": members_of(f"{ROOT}/data/candidate_sets/fable_trim_p2.json"),
    "D2": members_of(f"{ROOT}/data/candidate_sets/fable_trim_d2.json"),
    "h6_16k_trim": members_of(f"{ROOT}/data/candidate_sets/trim_v6_h6_16k.json"),
}
sets65 = {
    "h6_65k_trim": members_of(f"{ROOT}/data/candidate_sets/trim_v6_h6_65k.json"),
}

res = {}
print(f"{'set':14s} {'fineweb_new':>12s} {'swe_code':>10s} {'overall':>8s}")
for name, mem in sets16.items():
    r = fp_on(f"{ROOT}/out/harvest_ctlnew_16k", thr16, mem, cfg_of)
    res[name] = r
    print(f"{name:14s} {r['ctl_fineweb_new']*100:11.1f}% {r['ctl_swe_code']*100:9.1f}% {r['overall']*100:7.1f}%")
for name, mem in sets65.items():
    r = fp_on(f"{ROOT}/out/harvest_ctlnew_65k", thr65, mem, cfg_of)
    res[name] = r
    print(f"{name:14s} {r['ctl_fineweb_new']*100:11.1f}% {r['ctl_swe_code']*100:9.1f}% {r['overall']*100:7.1f}%")

json.dump(res, open(f"{ROOT}/out/fp_new_controls_aug15.json", "w"), indent=1)
print(f"wrote {ROOT}/out/fp_new_controls_aug15.json")
