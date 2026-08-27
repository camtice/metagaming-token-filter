"""Pre-registered fixed thresholds for R3 (rate ratio) and R1 (ctl cap) (2026-08-15).

No sweep: evaluate the full Rathi rule at manually chosen round-number
thresholds. R3 in RATE-ratio form (per-token fire rate on forget vs retain;
the earlier sweep's knob was a count ratio — rate = count_ratio * n_ctl/n_forget).

rho grid: 1, 2, 3, 5, 10        tau grid: 1%, .5%, .2%, .1%, .05%, .01%
Usage: python fixed_ratio_points.py --subset X --harvest Y --tag Z
"""
import argparse
import json

import numpy as np

ROOT = "/home/a5k/cwtice.a5k/sae-exploration"
DOCS = f"{ROOT}/data/test_docs_v5.jsonl"
POOL = f"{ROOT}/data/candidate_sets/haiku_fable_forget_latents_v1.json"
RHOS = [1, 2, 3, 5, 10]
TAUS = [0.02, 0.01, 0.005, 0.003, 0.002, 0.0015, 0.001, 0.0007, 0.0005, 0.0003, 0.0002, 0.0001]

ap = argparse.ArgumentParser()
ap.add_argument("--subset", required=True)
ap.add_argument("--harvest", required=True)
ap.add_argument("--tag", required=True)
args = ap.parse_args()
HARV = f"{ROOT}/{args.harvest}"

zf = np.load(HARV + ".npz")
z = {k: zf[k] for k in zf.files}
meta = json.load(open(HARV + ".meta.json"))
W = meta["width"]
av = z["act"].astype(np.float64); av[np.isinf(av)] = 65504.0
ntok = z["doc_ntok"].astype(np.int64)
nd = len(meta["docs"])
tok_base = np.zeros(nd + 1, dtype=np.int64); np.cumsum(ntok, out=tok_base[1:])
n_tok = int(ntok.sum())
flat = tok_base[z["doc_idx"]] + z["tok_idx"]
offs = z["offsets"].reshape(-1, 2)
mu = np.bincount(z["lat_idx"], weights=av, minlength=W) / n_tok
sd = np.sqrt(np.maximum(np.bincount(z["lat_idx"], weights=av**2, minlength=W)/n_tok - mu**2, 0))
thr = mu + 4 * sd

docs = [json.loads(l) for l in open(DOCS)]
by_id = {d["id"]: d for d in docs}
doc_ids = [d["id"] for d in meta["docs"]]
lt = np.array([by_id[i]["label_type"] for i in doc_ids])
cfg = np.array([by_id[i].get("config", "") for i in doc_ids])
tok_doc = np.repeat(np.arange(nd), ntok)
gt = np.zeros(n_tok, bool)
for di, did in enumerate(doc_ids):
    for cs, ce in by_id[did].get("char_spans", []):
        o = offs[tok_base[di]:tok_base[di + 1]]
        m = (o[:, 0] < ce) & (o[:, 1] > cs)
        gt[tok_base[di] + np.flatnonzero(m)] = True
annm = np.isin(lt[tok_doc], ["spans", "clean"])
cleanm = lt[tok_doc] == "clean"
ctlm = lt[tok_doc] == "assumed_clean"
dense14 = (lt[tok_doc] == "dense_forget") & (np.char.find(cfg[tok_doc].astype(str), "c5") < 0)
is_doc_start = np.zeros(n_tok, bool); is_doc_start[tok_base[:-1]] = True
gt_ann = gt & annm; n_gt = int(gt_ann.sum())
mainm = annm | dense14
gt_main = gt_ann | dense14
n_gtm = int(gt_main.sum())

pool_raw = json.load(open(POOL))["sets"][args.subset]["latents"]
members = np.array(sorted({int(l) for l, _c, _f in pool_raw}))
nm = len(members)
lat2pos = -np.ones(W, np.int64); lat2pos[members] = np.arange(nm)
selm = lat2pos[z["lat_idx"]] >= 0
mp = lat2pos[z["lat_idx"][selm]]
mf = flat[selm]
mabove = av[selm] >= thr[z["lat_idx"][selm]]
order = np.argsort(mp, kind="stable")
mp_s, mf_s, ma_s = mp[order], mf[order], mabove[order]
bounds = np.searchsorted(mp_s, np.arange(nm + 1))
per_flat = [mf_s[bounds[i]:bounds[i+1]] for i in range(nm)]
per_above = [mf_s[bounds[i]:bounds[i+1]][ma_s[bounds[i]:bounds[i+1]]] for i in range(nm)]

def counts(mask):
    k = np.zeros(nm)
    h = mask[mf_s] & ma_s
    np.add.at(k, mp_s[h], 1)
    return k

c_forget = counts(gt_main); c_ctl = counts(ctlm)
n_ctl = int(ctlm.sum())
rate_f = c_forget / n_gtm
rate_c = c_ctl / n_ctl
# Jeffreys-smoothed rate ratio (0.5 pseudo-fire in each corpus)
rr = ((c_forget + 0.5) / n_gtm) / ((c_ctl + 0.5) / n_ctl)

prev_shift = np.zeros(n_tok, bool)

def evaluate(keep_pos):
    seedcnt = np.zeros(n_tok, np.int32); pacnt = np.zeros(n_tok, np.int32)
    for i in keep_pos:
        np.add.at(seedcnt, per_above[i], 1)
        np.add.at(pacnt, per_flat[i], 1)
    seeds = seedcnt >= 2
    pa = pacnt > 0
    prev_shift[1:] = pa[:-1]; prev_shift[0] = False
    start = pa & (~prev_shift | is_doc_start)
    run_id = np.cumsum(start) - 1
    nruns = int(start.sum())
    if nruns == 0:
        return dict(n=len(keep_pos), Pm=0.0, Rm=0.0, F2m=0.0, R_span=0.0, fp_ctl=0.0, fp_clean=0.0)
    seed_runs = np.zeros(nruns, bool)
    seed_runs[run_id[seeds]] = True
    pred = pa & seed_runs[run_id]
    tp = int((pred & gt_main & mainm).sum()); fpa = int((pred & mainm & ~gt_main).sum())
    Rm = tp / max(n_gtm, 1); Pm = tp / max(tp + fpa, 1)
    return {"n": len(keep_pos), "Pm": round(Pm, 4), "Rm": round(Rm, 4),
            "F2m": round(5*Pm*Rm/max(4*Pm+Rm, 1e-9), 4),
            "R_span": round(float(pred[gt_ann].mean()), 4) if n_gt else 0.0,
            "fp_ctl": round(float(pred[ctlm].mean()), 4),
            "fp_clean": round(float(pred[cleanm].mean()), 4)}

res = {"subset": args.subset, "harvest": args.harvest, "n_pool": nm,
       "note": "rate ratio = (forget fires/forget tokens)/(ctl fires/ctl tokens), Jeffreys 0.5",
       "R3_rate_ratio": {}, "R1_ctl_cap": {}}
print(f"== {args.subset} ({nm}) ==")
for rho in RHOS:
    keep = np.flatnonzero(rr >= rho)
    m = evaluate(keep)
    res["R3_rate_ratio"][str(rho)] = m
    print(f"  R3 rho>={rho:>2}: n={m['n']:4d} F2m={m['F2m']:.3f} Rm={m['Rm']:.3f} "
          f"R_span={m['R_span']:.3f} fp={m['fp_ctl']:.3f}", flush=True)
for tau in TAUS:
    keep = np.flatnonzero(rate_c <= tau)
    m = evaluate(keep)
    res["R1_ctl_cap"][str(tau)] = m
    print(f"  R1 tau<={tau:<6}: n={m['n']:4d} F2m={m['F2m']:.3f} Rm={m['Rm']:.3f} "
          f"R_span={m['R_span']:.3f} fp={m['fp_ctl']:.3f}", flush=True)
json.dump(res, open(f"{ROOT}/out/fixed_points_{args.tag}.json", "w"), indent=1)
print(f"wrote out/fixed_points_{args.tag}.json")
