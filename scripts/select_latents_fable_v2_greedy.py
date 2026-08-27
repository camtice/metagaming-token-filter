"""Fable latent selection v2: exact greedy backward elimination (2026-08-14).

Prefix sweeps over single rankings (v1) cannot reach R>=0.90 at FP<=0.10;
this searches the subset lattice directly:

  1. Start from the full fable_16k pool (649 latents; R=.942, fp_ctl=.627).
  2. Backward elimination: repeatedly remove the latent with the highest
     exact (delta fp_ctl) / (delta recall) ratio, i.e. the best control-FP
     reduction per unit of recall sacrificed, computed through the FULL
     Rathi rule (seed >=2 above mean+4SD, contiguous-run windowing).
     Removal shrinks predictions monotonically, so recall and fp_ctl only
     ever decrease; lazy (stale-heap) re-evaluation keeps this fast.
  3. Track the whole trajectory; continue past fp<=0.10 while R>=0.90.
  4. Forward polish: re-add any removed latent that raises F2 while keeping
     both constraints; repeat to convergence.
  5. Winner = max F2 among points with R>=0.90 and fp_ctl<=0.10.

Selection split: TEST only (manifest v4). Sealed validation untouched.
Output: out/fable_trim_v2_greedy.json (+ frozen set with --freeze).
"""
import argparse
import heapq
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

# ---------- load (mirrors score_split.py) ----------
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
is_doc_start = np.zeros(n_tok, bool); is_doc_start[tok_base[:-1]] = True
gt_ann = gt & annm
n_gt = int(gt_ann.sum()); n_ctl = int(ctlm.sum())

pool_raw = json.load(open(POOL))["sets"]["fable_16k"]["latents"]
members = np.array(sorted({int(l) for l, _c, _f in pool_raw}))
cat = {int(l): c for l, c, _f in pool_raw}
conf = {int(l): f for l, _c, f in pool_raw}
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

# ---------- exact Rathi eval on current count state ----------
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
    tp = int((pred & gt_ann).sum()); fpa = int((pred & annm & ~gt).sum())
    R = tp / max(n_gt, 1); P = tp / max(tp + fpa, 1)
    return pred, {"P": P, "R": R,
                  "F2": 5*P*R/max(4*P+R, 1e-9), "F1": 2*P*R/max(P+R, 1e-9),
                  "fp_ctl": float(pred[ctlm].mean()),
                  "fp_clean_docs": float(pred[cleanm].mean())}

seedcnt = np.zeros(n_tok, np.int32)
pacnt = np.zeros(n_tok, np.int32)
active = np.ones(nm, bool)
for i in range(nm):
    np.add.at(seedcnt, per_above[i], 1)
    np.add.at(pacnt, per_flat[i], 1)
_, cur = evaluate(seedcnt, pacnt)
print(f"start: {nm} latents R={cur['R']:.4f} fp={cur['fp_ctl']:.4f} F2={cur['F2']:.4f}")

def without(i):
    np.subtract.at(seedcnt, per_above[i], 1)
    np.subtract.at(pacnt, per_flat[i], 1)
    _, m = evaluate(seedcnt, pacnt)
    np.add.at(seedcnt, per_above[i], 1)
    np.add.at(pacnt, per_flat[i], 1)
    return m

def ratio(m):
    dfp = cur["fp_ctl"] - m["fp_ctl"]
    dR = cur["R"] - m["R"]
    return (dfp + 1e-9) / (dR + 1e-5)

# initial exact scan
heap = []
for i in range(nm):
    heapq.heappush(heap, (-ratio(without(i)), i))
print("initial scan done")

traj = []
best_feasible = None
removed_order = []
evals = nm
while active.sum() > 2:
    # lazy greedy: re-evaluate the stale top until it stays on top
    while True:
        negs, i = heapq.heappop(heap)
        if not active[i]:
            continue
        m = without(i); evals += 1
        s = ratio(m)
        if not heap or -s <= heap[0][0] * (1 + 1e-9) or -s <= heap[0][0]:
            break
        heapq.heappush(heap, (-s, i))
    # apply removal of i
    np.subtract.at(seedcnt, per_above[i], 1)
    np.subtract.at(pacnt, per_flat[i], 1)
    active[i] = False
    removed_order.append(int(members[i]))
    cur = m
    traj.append({"n": int(active.sum()), "removed": int(members[i]),
                 **{k: round(v, 4) for k, v in m.items()}})
    if m["R"] >= args.min_recall and m["fp_ctl"] <= args.max_fp:
        if best_feasible is None or m["F2"] > best_feasible["F2"]:
            best_feasible = {"n": int(active.sum()),
                             "members": sorted(int(x) for x in members[active]),
                             **{k: round(v, 4) for k, v in m.items()}}
    if m["R"] < args.min_recall - 0.02:   # small margin past the cliff, then stop
        break
    if int(active.sum()) % 50 == 0:
        print(f"  n={int(active.sum()):3d} R={m['R']:.4f} fp={m['fp_ctl']:.4f} "
              f"F2={m['F2']:.4f} (evals {evals})")

print(f"backward done: {len(traj)} removals, {evals} exact evals")
print("best feasible after backward:", None if best_feasible is None else
      {k: best_feasible[k] for k in ("n", "P", "R", "F2", "fp_ctl", "fp_clean_docs")})

# ---------- forward polish from the best feasible point ----------
if best_feasible is not None:
    cur_set = set(best_feasible["members"])
    seedcnt[:] = 0; pacnt[:] = 0
    for l in cur_set:
        i = lat2pos[l]
        np.add.at(seedcnt, per_above[i], 1)
        np.add.at(pacnt, per_flat[i], 1)
    _, cur = evaluate(seedcnt, pacnt)
    improved = True
    while improved:
        improved = False
        best_add, best_m = None, None
        for l in map(int, members):
            if l in cur_set:
                continue
            i = lat2pos[l]
            np.add.at(seedcnt, per_above[i], 1)
            np.add.at(pacnt, per_flat[i], 1)
            _, m = evaluate(seedcnt, pacnt)
            np.subtract.at(seedcnt, per_above[i], 1)
            np.subtract.at(pacnt, per_flat[i], 1)
            evals += 1
            if (m["R"] >= args.min_recall and m["fp_ctl"] <= args.max_fp
                    and m["F2"] > cur["F2"] + 1e-6
                    and (best_m is None or m["F2"] > best_m["F2"])):
                best_add, best_m = l, m
        if best_add is not None:
            i = lat2pos[best_add]
            np.add.at(seedcnt, per_above[i], 1)
            np.add.at(pacnt, per_flat[i], 1)
            cur_set.add(best_add); cur = best_m; improved = True
            print(f"  polish +{best_add}: F2={cur['F2']:.4f} R={cur['R']:.4f} fp={cur['fp_ctl']:.4f}")
    best_feasible = {"n": len(cur_set), "members": sorted(cur_set),
                     **{k: round(v, 4) for k, v in cur.items()}}

res = {"pool": "fable_16k", "n_pool": nm, "harvest": HARV,
       "constraints": {"min_recall": args.min_recall, "max_fp_ctl": args.max_fp},
       "trajectory": traj, "removed_order": removed_order,
       "best_feasible": best_feasible, "exact_evals": evals}

if best_feasible is not None:
    w = best_feasible
    print(f"\nWINNER n={w['n']}: P={w['P']} R={w['R']} F2={w['F2']} "
          f"fp_ctl={w['fp_ctl']} fp_clean_docs={w['fp_clean_docs']}")
    # bootstrap CI (doc-level, annotated docs)
    seedcnt[:] = 0; pacnt[:] = 0
    for l in w["members"]:
        i = lat2pos[l]
        np.add.at(seedcnt, per_above[i], 1)
        np.add.at(pacnt, per_flat[i], 1)
    pred, _ = evaluate(seedcnt, pacnt)
    ann_docs = np.flatnonzero(np.isin(lt, ["spans", "clean"]))
    rng = np.random.default_rng(SEED)
    boot = []
    for _ in range(500):
        pick = rng.choice(ann_docs, len(ann_docs), replace=True)
        tp = fpa = fn = 0
        for d in pick:
            s, e = tok_base[d], tok_base[d + 1]
            tp += int((pred[s:e] & gt[s:e]).sum())
            fpa += int((pred[s:e] & ~gt[s:e]).sum())
            fn += int((~pred[s:e] & gt[s:e]).sum())
        P = tp / max(tp + fpa, 1); R = tp / max(tp + fn, 1)
        boot.append((R, 5*P*R/max(4*P+R, 1e-9)))
    boot = np.array(boot)
    res["bootstrap"] = {"R_ci90": [round(float(x), 3) for x in np.percentile(boot[:, 0], [5, 95])],
                        "F2_ci90": [round(float(x), 3) for x in np.percentile(boot[:, 1], [5, 95])]}
    print("bootstrap:", res["bootstrap"])
    comp_cat = {}; comp_conf = {}
    for l in w["members"]:
        comp_cat[cat[l]] = comp_cat.get(cat[l], 0) + 1
        comp_conf[conf[l]] = comp_conf.get(conf[l], 0) + 1
    og = set(int(k) for k in json.load(open(f"{ROOT}/data/latent_list_2026-07-29.json"))["tags"])
    res["composition"] = {"category": comp_cat, "confidence": comp_conf,
                          "overlap_with_OG222": len(og & set(w["members"]))}
    print("composition:", res["composition"])

json.dump(res, open(f"{ROOT}/out/fable_trim_v2_greedy.json", "w"))
print(f"wrote {ROOT}/out/fable_trim_v2_greedy.json")

if args.freeze and best_feasible is not None:
    w = best_feasible
    out = {"name": "fable_trim_v2",
           "created": "2026-08-14",
           "source_pool": "fable_16k (haiku_fable_forget_latents_v1.json), 649 latents",
           "method": {"rule": "greedy backward elimination (exact dFP/dR ratio, lazy) + forward F2 polish",
                      "objective": "max F2 on TEST v4 s.t. R>=0.90, fp_ctl<=0.10",
                      "harvest": HARV, "selection_split": "TEST (manifest v4, sha 8fa54624)"},
           "test_point": {k: w[k] for k in ("P", "R", "F2", "F1", "fp_ctl", "fp_clean_docs")},
           "bootstrap": res["bootstrap"], "composition": res["composition"],
           "members": w["members"]}
    path = f"{ROOT}/data/candidate_sets/fable_trim_v2.json"
    json.dump(out, open(path, "w"), indent=1)
    print("FROZE", path, f"({len(w['members'])} latents)")
