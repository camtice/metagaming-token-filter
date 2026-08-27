"""Principled latent selection from the fable_16k judge pool, v1 (2026-08-14).

Goal: subset of the 649 fable-judged latents maximizing token F2 on TEST
(manifest v4) subject to recall >= 0.90 and assumed-clean control FP <= 0.10.

Method (paper framing):
  1. Per-latent evidence from the TEST harvest under the scorer's own
     mean+4SD thresholds: fire rate on human-labeled forget-span tokens
     (q_span), on dense c1-c4 assumed-forget tokens (q_dense), and on
     retain-control tokens (r_ctl; 625 assumed_clean docs incl. SWE).
     Rates are Jeffreys-smoothed: (k + 0.5) / (n + 1).
  2. Three one-knob selection rules, each swept exhaustively with the FULL
     Rathi rule (seed = >=2 selected features above threshold; windowing =
     contiguous any-member-active runs containing a seed):
       A  rank by LLR = log(q_span / r_ctl), sweep prefix K
       B  keep r_ctl <= tau, rank by q_span, grid (tau, K)
       C  as A but forget evidence = span tokens pooled with dense c1-c4
  3. Winner = max F2 s.t. R >= 0.90 and FP(assumed_clean) <= 0.10.
     Bootstrap CI over annotated docs + selection-stability for the winner.

TEST is the iterate-freely partition; sealed VAL-A/VAL-B remain untouched.

Usage:
  python select_latents_fable_v1.py            # sweeps + report json
  python select_latents_fable_v1.py --freeze   # also write the winner set file
"""
import argparse
import json

import numpy as np

ROOT = "/home/a5k/cwtice.a5k/sae-exploration"
HARV = f"{ROOT}/out/harvest_test_v3"
DOCS = f"{ROOT}/data/test_docs_v4.jsonl"
POOL = f"{ROOT}/data/candidate_sets/haiku_fable_forget_latents_v1.json"
SEED = 20260814

ap = argparse.ArgumentParser()
ap.add_argument("--freeze", action="store_true")
ap.add_argument("--min-recall", type=float, default=0.90)
ap.add_argument("--max-fp", type=float, default=0.10)
args = ap.parse_args()

# ---------- load harvest + docs (mirrors score_split.py exactly) ----------
z = np.load(HARV + ".npz")
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
reg = np.array([by_id[i]["register"] for i in doc_ids])
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
dense14 = (lt[tok_doc] == "dense_forget") & np.isin(
    cfg[tok_doc], [c for c in set(cfg[lt == "dense_forget"]) if "c5" not in c])
is_doc_start = np.zeros(n_tok, bool); is_doc_start[tok_base[:-1]] = True

# ---------- candidate pool ----------
pool_raw = json.load(open(POOL))["sets"]["fable_16k"]["latents"]
members = np.array(sorted({int(l) for l, _c, _f in pool_raw}))
cat = {int(l): c for l, c, _f in pool_raw}
conf = {int(l): f for l, _c, f in pool_raw}
nm = len(members)
lat2pos = -np.ones(W, np.int64); lat2pos[members] = np.arange(nm)

selm = lat2pos[z["lat_idx"]] >= 0
mp = lat2pos[z["lat_idx"][selm]]          # member position per row
mf = flat[selm]                            # flat token per row
mabove = av[selm] >= thr[z["lat_idx"][selm]]
print(f"pool {nm} latents | member nnz {selm.sum():,} | above-thr nnz {int(mabove.sum()):,} "
      f"| gt tokens {int(gt.sum()):,} | ctl tokens {int(ctlm.sum()):,} | dense14 tokens {int(dense14.sum()):,}")

# group member rows by latent for incremental sweeps
order = np.argsort(mp, kind="stable")
mp_s, mf_s, ma_s = mp[order], mf[order], mabove[order]
bounds = np.searchsorted(mp_s, np.arange(nm + 1))
per_lat_flat = [mf_s[bounds[i]:bounds[i+1]] for i in range(nm)]
per_lat_above = [mf_s[bounds[i]:bounds[i+1]][ma_s[bounds[i]:bounds[i+1]]] for i in range(nm)]

# ---------- per-latent evidence (Jeffreys-smoothed rates) ----------
def rates(mask):
    n = max(int(mask.sum()), 1)
    k = np.zeros(nm)
    hit = mask[mf_s] & ma_s
    np.add.at(k, mp_s[hit], 1)
    return (k + 0.5) / (n + 1)

q_span = rates(gt)
q_dense = rates(dense14)
r_ctl = rates(ctlm)
gt_pool = gt | dense14
q_pool = rates(gt_pool)
llr_A = np.log(q_span / r_ctl)
llr_C = np.log(q_pool / r_ctl)

# raw (unsmoothed) ctl rate for the tau cap in method B
r_ctl_raw = np.zeros(nm)
hit = ctlm[mf_s] & ma_s
np.add.at(r_ctl_raw, mp_s[hit], 1)
r_ctl_raw /= max(int(ctlm.sum()), 1)

# ---------- exact Rathi evaluation ----------
prev_shift = np.zeros(n_tok, bool)

def evaluate(seedcnt, pacnt):
    seeds = seedcnt >= 2
    pa = pacnt > 0
    prev_shift[1:] = pa[:-1]; prev_shift[0] = False
    start = pa & (~prev_shift | is_doc_start)
    run_id = np.cumsum(start) - 1
    nruns = int(start.sum())
    if nruns == 0:
        pred = np.zeros(n_tok, bool)
    else:
        seed_runs = np.zeros(nruns, bool)
        seed_runs[run_id[seeds]] = True
        pred = pa & seed_runs[run_id]
    tp = int((pred & gt & annm).sum()); fp = int((pred & ~gt & annm).sum())
    fn = int((~pred & gt & annm).sum())
    P = tp / max(tp + fp, 1); R = tp / max(tp + fn, 1)
    return pred, {"P": round(P, 4), "R": round(R, 4),
                  "F2": round(5*P*R/max(4*P+R, 1e-9), 4),
                  "F1": round(2*P*R/max(P+R, 1e-9), 4),
                  "fp_ctl": round(float(pred[ctlm].mean()), 4),
                  "fp_clean_docs": round(float(pred[cleanm].mean()), 4)}

def sweep(rank_idx, label, k_grid=None):
    """Exact incremental prefix sweep along a ranking."""
    seedcnt = np.zeros(n_tok, np.int32)
    pacnt = np.zeros(n_tok, np.int32)
    rows = []
    ks = set(k_grid) if k_grid else None
    for j, li in enumerate(rank_idx, 1):
        np.add.at(seedcnt, per_lat_above[li], 1)
        np.add.at(pacnt, per_lat_flat[li], 1)
        if ks is not None and j not in ks:
            continue
        _, m = evaluate(seedcnt, pacnt)
        rows.append({"K": j, **m})
    return rows

def frontier_pick(rows, extra=None):
    """Best F2 subject to constraints; None if infeasible."""
    feas = [r for r in rows if r["R"] >= args.min_recall and r["fp_ctl"] <= args.max_fp
            and (extra is None or extra(r))]
    return max(feas, key=lambda r: r["F2"]) if feas else None

results = {"pool": "fable_16k", "n_pool": nm, "harvest": HARV,
           "constraints": {"min_recall": args.min_recall, "max_fp_ctl": args.max_fp}}
kg = sorted(set(list(range(2, 60)) + list(range(60, 200, 5)) + list(range(200, nm + 1, 25)) + [nm]))

print("\n== A: LLR(span vs ctl) prefix sweep ==")
rank_A = np.argsort(-llr_A, kind="stable")
rows_A = sweep(rank_A, "A", kg)
results["A"] = {"rows": rows_A, "rank": members[rank_A].tolist()}

print("== C: LLR(span+dense14 vs ctl) prefix sweep ==")
rank_C = np.argsort(-llr_C, kind="stable")
rows_C = sweep(rank_C, "C", kg)
results["C"] = {"rows": rows_C, "rank": members[rank_C].tolist()}

print("== B: FP-cap tau + recall-carrier ranking ==")
results["B"] = {}
for tau in (0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02):
    keep = np.flatnonzero(r_ctl_raw <= tau)
    if len(keep) < 2:
        continue
    rank_B = keep[np.argsort(-q_span[keep], kind="stable")]
    rows_B = sweep(rank_B, f"B tau={tau}", [k for k in kg if k <= len(keep)])
    results["B"][str(tau)] = {"rows": rows_B, "rank": members[rank_B].tolist(),
                              "n_kept": int(len(keep))}
    best = frontier_pick(rows_B)
    print(f"  tau={tau}: kept {len(keep)}; best feasible: {best}")

# ---------- pick the winner ----------
cands = []
for name, rk in (("A", rank_A), ("C", rank_C)):
    b = frontier_pick(results[name]["rows"])
    if b: cands.append((name, None, b, rk))
for tau, d in results["B"].items():
    b = frontier_pick(d["rows"])
    if b: cands.append(("B", float(tau), b, np.array([lat2pos[l] for l in d["rank"]])))

if not cands:
    print("\nNO feasible point meets both constraints; frontier extremes:")
    for name in ("A", "C"):
        rows = results[name]["rows"]
        hi_r = [r for r in rows if r["R"] >= args.min_recall]
        lo_f = [r for r in rows if r["fp_ctl"] <= args.max_fp]
        print(f"  {name}: min fp at R>=.9: {min(hi_r, key=lambda r: r['fp_ctl']) if hi_r else None}")
        print(f"  {name}: max R at fp<=.1: {max(lo_f, key=lambda r: r['R']) if lo_f else None}")
    results["winner"] = None
else:
    name, tau, best, rk = max(cands, key=lambda c: c[2]["F2"])
    K = best["K"]
    chosen = members[rk[:K]].tolist() if name != "B" else \
        [int(members[i]) for i in rk[:K]]
    results["winner"] = {"method": name, "tau": tau, **best,
                         "members": sorted(int(x) for x in chosen)}
    print(f"\nWINNER: method {name} tau={tau} K={K} -> {best}")

    # ---------- bootstrap CI over annotated docs (winner fixed) ----------
    seedcnt = np.zeros(n_tok, np.int32); pacnt = np.zeros(n_tok, np.int32)
    for l in results["winner"]["members"]:
        li = lat2pos[l]
        np.add.at(seedcnt, per_lat_above[li], 1)
        np.add.at(pacnt, per_lat_flat[li], 1)
    pred, point = evaluate(seedcnt, pacnt)
    ann_docs = np.flatnonzero(np.isin(lt, ["spans", "clean"]))
    rng = np.random.default_rng(SEED)
    boot = []
    for _ in range(500):
        pick = rng.choice(ann_docs, len(ann_docs), replace=True)
        tp = fp = fn = 0
        for d in pick:
            s, e = tok_base[d], tok_base[d + 1]
            tp += int((pred[s:e] & gt[s:e]).sum())
            fp += int((pred[s:e] & ~gt[s:e]).sum())
            fn += int((~pred[s:e] & gt[s:e]).sum())
        P = tp / max(tp + fp, 1); R = tp / max(tp + fn, 1)
        boot.append((R, 5*P*R/max(4*P+R, 1e-9)))
    boot = np.array(boot)
    results["winner"]["bootstrap"] = {
        "R_ci90": [round(float(x), 3) for x in np.percentile(boot[:, 0], [5, 95])],
        "F2_ci90": [round(float(x), 3) for x in np.percentile(boot[:, 1], [5, 95])]}
    print("bootstrap (500x, doc-level):", results["winner"]["bootstrap"])

    # selection stability: re-rank on bootstrapped span docs, same method A LLR
    span_docs = np.flatnonzero(lt == "spans")
    freq = np.zeros(nm)
    for _ in range(50):
        pick = rng.choice(span_docs, len(span_docs), replace=True)
        bm = np.zeros(n_tok, bool)
        for d in pick:
            bm[tok_base[d]:tok_base[d + 1]] = True
        qb = rates(gt & bm)
        rb = np.argsort(-np.log(qb / r_ctl), kind="stable")[:K]
        freq[rb] += 1
    inset = np.array([lat2pos[l] for l in results["winner"]["members"]])
    results["winner"]["stability"] = {
        "mean_sel_freq_of_chosen": round(float(freq[inset].mean() / 50), 3),
        "n_chosen_freq_ge_half": int((freq[inset] >= 25).sum())}
    print("stability:", results["winner"]["stability"])

    comp_cat = {}; comp_conf = {}
    for l in results["winner"]["members"]:
        comp_cat[cat[l]] = comp_cat.get(cat[l], 0) + 1
        comp_conf[conf[l]] = comp_conf.get(conf[l], 0) + 1
    og = set(int(k) for k in json.load(open(f"{ROOT}/data/latent_list_2026-07-29.json"))["tags"])
    results["winner"]["composition"] = {"category": comp_cat, "confidence": comp_conf,
                                        "overlap_with_OG222": len(og & set(results["winner"]["members"]))}
    print("composition:", results["winner"]["composition"])

json.dump(results, open(f"{ROOT}/out/fable_trim_v1_sweeps.json", "w"))
print(f"\nwrote {ROOT}/out/fable_trim_v1_sweeps.json")

if args.freeze and results.get("winner"):
    w = results["winner"]
    out = {"name": "fable_trim_v1",
           "created": "2026-08-14",
           "source_pool": "fable_16k (haiku_fable_forget_latents_v1.json), 649 latents",
           "method": {"rule": w["method"], "tau": w["tau"], "K": w["K"],
                      "objective": "max F2 on TEST v4 s.t. R>=0.90, FP(assumed_clean)<=0.10",
                      "evidence": "mean+4SD fire rates: human span tokens vs 625 retain-control docs "
                                  "(dense c1-c4 auxiliary in method C); Jeffreys smoothing",
                      "harvest": HARV, "selection_split": "TEST (manifest v4, sha 8fa54624)"},
           "test_point": {k: w[k] for k in ("P", "R", "F2", "F1", "fp_ctl", "fp_clean_docs")},
           "bootstrap": w["bootstrap"], "stability": w["stability"],
           "composition": w["composition"],
           "members": w["members"]}
    path = f"{ROOT}/data/candidate_sets/fable_trim_v1.json"
    json.dump(out, open(path, "w"), indent=1)
    print("FROZE", path, f"({len(w['members'])} latents)")
