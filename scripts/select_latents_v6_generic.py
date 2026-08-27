"""Generic pool optimizer under protocol v5 (2026-08-15).

Same objective as the D2 run: maximize main F2 (forget = human spans + dense
c1-c4) subject to fp_ctl <= 0.10, via feasible-start construction + constrained
hill-climb. Parameterized over candidate pool and harvest so 16k and 65k judge
sets run through identical machinery.

Starts: (a) LLR(span+dense vs ctl) prefix sweep -> best feasible K,
        (b) per-feature FP-cap ranking (tau grid) -> best feasible point.
Then multi-round steepest-ascent hill-climb (add/remove/swap) from each.

Usage:
  python select_latents_v6_generic.py --subset haiku_v6_65k \
      --harvest out/harvest_test_v3_65k --tag h6_65k
Output: out/trim_v6_<tag>.json + data/candidate_sets/trim_v6_<tag>.json
"""
import argparse
import json

import numpy as np

ROOT = "/home/a5k/cwtice.a5k/sae-exploration"
DOCS = f"{ROOT}/data/test_docs_v4.jsonl"
POOL = f"{ROOT}/data/candidate_sets/haiku_fable_forget_latents_v1.json"
SEED = 20260815
SAMPLE, ROUNDS, MAX_STEPS = 60, 3, 400
MAX_FP = 0.10

ap = argparse.ArgumentParser()
ap.add_argument("--subset", required=True)
ap.add_argument("--harvest", required=True)
ap.add_argument("--tag", required=True)
args = ap.parse_args()
HARV = f"{ROOT}/{args.harvest}" if not args.harvest.startswith("/") else args.harvest

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
print(f"{args.subset}: {nm} latents, member nnz {int(selm.sum()):,}")

def rate(mask):
    k = np.zeros(nm)
    h = mask[mf_s] & ma_s
    np.add.at(k, mp_s[h], 1)
    return k / max(int(mask.sum()), 1)

q_span = rate(gt_ann); q_dense = rate(dense14); r_ctl = rate(ctlm)
q_pool_s = (q_span * n_gt + q_dense * int(dense14.sum()) + 0.5) / (n_gtm + 1)
r_ctl_s = (r_ctl * int(ctlm.sum()) + 0.5) / (int(ctlm.sum()) + 1)
llr = np.log(q_pool_s / r_ctl_s)
w_add = (q_span + q_dense + 1e-4); w_add /= w_add.sum()
w_rem = (r_ctl + 1e-4); w_rem /= w_rem.sum()

prev_shift = np.zeros(n_tok, bool)

def evaluate(seedcnt, pacnt):
    seeds = seedcnt >= 2
    pa = pacnt > 0
    prev_shift[1:] = pa[:-1]; prev_shift[0] = False
    start = pa & (~prev_shift | is_doc_start)
    run_id = np.cumsum(start) - 1
    nruns = int(start.sum())
    if nruns == 0:
        return None, dict(Pm=0.0, Rm=0.0, F2m=0.0, F1m=0.0, P=0.0, R=0.0, F2=0.0,
                          fp_ctl=0.0, fp_clean_docs=0.0)
    seed_runs = np.zeros(nruns, bool)
    seed_runs[run_id[seeds]] = True
    pred = pa & seed_runs[run_id]
    tp = int((pred & gt_main & mainm).sum()); fpa = int((pred & mainm & ~gt_main).sum())
    Rm = tp / max(n_gtm, 1); Pm = tp / max(tp + fpa, 1)
    tpa = int((pred & gt_ann).sum()); fpaa = int((pred & annm & ~gt).sum())
    Ra = tpa / max(n_gt, 1); Pa = tpa / max(tpa + fpaa, 1)
    return pred, {"Pm": Pm, "Rm": Rm,
                  "F2m": 5*Pm*Rm/max(4*Pm+Rm, 1e-9), "F1m": 2*Pm*Rm/max(Pm+Rm, 1e-9),
                  "P": Pa, "R": Ra, "F2": 5*Pa*Ra/max(4*Pa+Ra, 1e-9),
                  "fp_ctl": float(pred[ctlm].mean()), "fp_clean_docs": float(pred[cleanm].mean())}

def key(m): return (m["fp_ctl"] <= MAX_FP, m["F2m"], m["Rm"])

def prefix_best(rank_idx, label):
    seedcnt = np.zeros(n_tok, np.int32); pacnt = np.zeros(n_tok, np.int32)
    best = None
    kg = set(list(range(2, 80, 2)) + list(range(80, min(len(rank_idx), 800) + 1, 10))
             + list(range(800, len(rank_idx) + 1, 50)) + [len(rank_idx)])
    for j, li in enumerate(rank_idx, 1):
        np.add.at(seedcnt, per_above[li], 1)
        np.add.at(pacnt, per_flat[li], 1)
        if j not in kg:
            continue
        _, m = evaluate(seedcnt, pacnt)
        if m["fp_ctl"] <= MAX_FP and (best is None or m["F2m"] > best[1]["F2m"]):
            best = (j, m)
    if best:
        print(f"  start {label}: K={best[0]} F2m={best[1]['F2m']:.4f} "
              f"Rm={best[1]['Rm']:.4f} fp={best[1]['fp_ctl']:.4f}")
        return set(int(members[i]) for i in rank_idx[:best[0]])
    print(f"  start {label}: no feasible prefix")
    return None

def climb(start_set, tag, rng):
    cur_set = set(start_set)
    seedcnt = np.zeros(n_tok, np.int32); pacnt = np.zeros(n_tok, np.int32)
    for l in cur_set:
        i = lat2pos[l]
        np.add.at(seedcnt, per_above[i], 1)
        np.add.at(pacnt, per_flat[i], 1)
    _, m = evaluate(seedcnt, pacnt)
    best_set, best_m = set(cur_set), dict(m)
    for rnd in range(ROUNDS):
        if rnd > 0:
            cur_set = set(best_set)
            seedcnt[:] = 0; pacnt[:] = 0
            for l in cur_set:
                i = lat2pos[l]
                np.add.at(seedcnt, per_above[i], 1)
                np.add.at(pacnt, per_flat[i], 1)
            for _ in range(6):
                i = int(rng.integers(nm)); l = int(members[i])
                sgn = -1 if l in cur_set else +1
                f = np.add.at if sgn > 0 else np.subtract.at
                f(seedcnt, per_above[i], 1); f(pacnt, per_flat[i], 1)
                cur_set.add(l) if sgn > 0 else cur_set.discard(l)
            _, m = evaluate(seedcnt, pacnt)
        for _step in range(MAX_STEPS):
            props = []
            outs = [l for l in map(int, members) if l not in cur_set]
            ins = sorted(cur_set)
            for _ in range(SAMPLE):
                r = rng.random()
                if r < 0.35 and outs:
                    props.append([(+1, lat2pos[int(rng.choice(outs))])])
                elif r < 0.70 and len(ins) > 2:
                    i = int(rng.choice(nm, p=w_rem))
                    if int(members[i]) not in cur_set:
                        i = lat2pos[int(rng.choice(ins))]
                    props.append([(-1, i)])
                elif outs and len(ins) > 2:
                    props.append([(-1, lat2pos[int(rng.choice(ins))]),
                                  (+1, lat2pos[int(rng.choice(outs))])])
            best_prop, best_key = None, key(m)
            for tg in props:
                for sgn, i in tg:
                    f = np.add.at if sgn > 0 else np.subtract.at
                    f(seedcnt, per_above[i], 1); f(pacnt, per_flat[i], 1)
                _, m2 = evaluate(seedcnt, pacnt)
                if key(m2) > best_key:
                    best_prop, best_key, best_m2 = tg, key(m2), m2
                for sgn, i in tg:
                    f = np.subtract.at if sgn > 0 else np.add.at
                    f(seedcnt, per_above[i], 1); f(pacnt, per_flat[i], 1)
            if best_prop is None:
                break
            for sgn, i in best_prop:
                f = np.add.at if sgn > 0 else np.subtract.at
                f(seedcnt, per_above[i], 1); f(pacnt, per_flat[i], 1)
                l = int(members[i])
                cur_set.add(l) if sgn > 0 else cur_set.discard(l)
            m = best_m2
            if key(m) > key(best_m):
                best_set, best_m = set(cur_set), dict(m)
        print(f"  climb {tag} round {rnd}: F2m={best_m['F2m']:.4f} Rm={best_m['Rm']:.4f} "
              f"fp={best_m['fp_ctl']:.4f} n={len(best_set)}")
    return best_set, best_m

rng = np.random.default_rng(SEED)
rank_llr = np.argsort(-llr, kind="stable")
starts = []
s = prefix_best(rank_llr, "LLR-prefix")
if s: starts.append(("llr", s))
for tau in (0.001, 0.002, 0.005):
    keep = np.flatnonzero(r_ctl <= tau)
    if len(keep) < 2: continue
    rk = keep[np.argsort(-(q_span + q_dense)[keep], kind="stable")]
    s = prefix_best(rk, f"tau={tau}")
    if s: starts.append((f"tau{tau}", s))

best = None
for sname, s0 in starts:
    print(f"== climb from {sname} ==")
    sset, m = climb(s0, sname, rng)
    if best is None or key(m) > key(best[1]):
        best = (sset, m, sname)

sset, m, sname = best
res = {"subset": args.subset, "harvest": args.harvest, "n_pool": nm,
       "winner": {"start": sname, "n": len(sset),
                  **{k: round(v, 4) for k, v in m.items()},
                  "members": sorted(sset)}}
json.dump(res, open(f"{ROOT}/out/trim_v6_{args.tag}.json", "w"))
out = {"name": f"trim_v6_{args.tag}", "created": "2026-08-15",
       "source_pool": f"{args.subset} ({nm} latents)",
       "method": {"pipeline": "feasible-start prefix sweeps + constrained hill-climb (v6 generic)",
                  "objective": "max main F2 (spans + dense c1-c4) s.t. fp_ctl<=0.10",
                  "harvest": args.harvest, "selection_split": "TEST (manifest v4)", "seed": SEED},
       "test_point_main": {k: round(m[k], 4) for k in ("Pm", "Rm", "F2m", "F1m")},
       "test_point_annotated": {k: round(m[k], 4) for k in ("P", "R", "F2")},
       "fp": {k: round(m[k], 4) for k in ("fp_ctl", "fp_clean_docs")},
       "members": sorted(sset)}
json.dump(out, open(f"{ROOT}/data/candidate_sets/trim_v6_{args.tag}.json", "w"), indent=1)
print(f"\nWINNER {args.tag} (from {sname}) n={len(sset)}: "
      + json.dumps({k: round(m[k], 4) for k in ("Pm", "Rm", "F2m", "P", "R", "F2", "fp_ctl", "fp_clean_docs")}))
print("froze", f"data/candidate_sets/trim_v6_{args.tag}.json")
