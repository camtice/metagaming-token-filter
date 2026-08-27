"""Fable latent selection v4: constrained hill-climb polish (2026-08-14).

v3 annealing established the constraint pair (R>=.90, fp_ctl<=.10) is jointly
infeasible for the fable_16k pool under the frozen Rathi rule. This produces
deterministic, frozen member lists for the three anchored operating points:

  P1  fp_ctl <= .10 hard; maximize recall (F2 tiebreak)  — "closest to both"
  P2  fp_ctl <= .10 hard; maximize F2                     — best headline F2
  P3  R >= .90 hard;      minimize fp_ctl (F2 tiebreak)   — recall-floor point

Each: steepest-ascent over sampled moves (biased adds/removes/swaps), exact
Rathi evaluation, plateau perturbation restarts, fixed seed.
Output: out/fable_trim_v4_polish.json + data/candidate_sets/fable_trim_{p1,p2,p3}.json
"""
import json

import numpy as np

ROOT = "/home/a5k/cwtice.a5k/sae-exploration"
HARV = f"{ROOT}/out/harvest_test_v3"
DOCS = f"{ROOT}/data/test_docs_v4.jsonl"
POOL = f"{ROOT}/data/candidate_sets/haiku_fable_forget_latents_v1.json"
SEED = 20260814
SAMPLE, ROUNDS, MAX_STEPS = 60, 3, 400

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
is_doc_start = np.zeros(n_tok, bool); is_doc_start[tok_base[:-1]] = True
gt_ann = gt & annm; n_gt = int(gt_ann.sum())

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

q_span = np.zeros(nm); r_ctl = np.zeros(nm)
hg = gt_ann[mf_s] & ma_s; np.add.at(q_span, mp_s[hg], 1)
hc = ctlm[mf_s] & ma_s; np.add.at(r_ctl, mp_s[hc], 1)
w_add = (q_span + 0.5); w_add /= w_add.sum()
w_rem = (r_ctl + 0.5); w_rem /= w_rem.sum()

prev_shift = np.zeros(n_tok, bool)

def evaluate(seedcnt, pacnt):
    seeds = seedcnt >= 2
    pa = pacnt > 0
    prev_shift[1:] = pa[:-1]; prev_shift[0] = False
    start = pa & (~prev_shift | is_doc_start)
    run_id = np.cumsum(start) - 1
    nruns = int(start.sum())
    if nruns == 0:
        return None, {"P": 0.0, "R": 0.0, "F2": 0.0, "F1": 0.0, "fp_ctl": 0.0, "fp_clean_docs": 0.0}
    seed_runs = np.zeros(nruns, bool)
    seed_runs[run_id[seeds]] = True
    pred = pa & seed_runs[run_id]
    tp = int((pred & gt_ann).sum()); fpa = int((pred & annm & ~gt).sum())
    R = tp / max(n_gt, 1); P = tp / max(tp + fpa, 1)
    return pred, {"P": P, "R": R, "F2": 5*P*R/max(4*P+R, 1e-9),
                  "F1": 2*P*R/max(P+R, 1e-9),
                  "fp_ctl": float(pred[ctlm].mean()), "fp_clean_docs": float(pred[cleanm].mean())}

def key_p1(m): return (m["fp_ctl"] <= 0.10, m["R"], m["F2"])
def key_p2(m): return (m["fp_ctl"] <= 0.10, m["F2"], m["R"])
def key_p3(m): return (m["R"] >= 0.90, -m["fp_ctl"], m["F2"])

def climb(start_set, key, tag, rng):
    cur_set = set(start_set)
    seedcnt = np.zeros(n_tok, np.int32); pacnt = np.zeros(n_tok, np.int32)
    for l in cur_set:
        i = lat2pos[l]
        np.add.at(seedcnt, per_above[i], 1)
        np.add.at(pacnt, per_flat[i], 1)
    _, m = evaluate(seedcnt, pacnt)
    best_set, best_m = set(cur_set), dict(m)
    evals = 0
    for rnd in range(ROUNDS):
        if rnd > 0:   # plateau perturbation from the best point
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
                _, m2 = evaluate(seedcnt, pacnt); evals += 1
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
        print(f"  {tag} round {rnd}: best R={best_m['R']:.4f} fp={best_m['fp_ctl']:.4f} "
              f"F2={best_m['F2']:.4f} n={len(best_set)} (evals {evals})")
    return best_set, best_m

def full_metrics(mem_set):
    seedcnt = np.zeros(n_tok, np.int32); pacnt = np.zeros(n_tok, np.int32)
    for l in mem_set:
        i = lat2pos[l]
        np.add.at(seedcnt, per_above[i], 1)
        np.add.at(pacnt, per_flat[i], 1)
    pred, m = evaluate(seedcnt, pacnt)
    out = {k: round(v, 4) for k, v in m.items()}
    for rg in sorted(set(reg[np.isin(lt, ["spans", "clean"])])):
        msk = annm & (reg[tok_doc] == rg)
        out[f"recall_{rg}"] = round(float(pred[gt & msk].mean()), 3) if (gt & msk).any() else None
    fpc = {}
    for c in sorted(set(cfg[lt == "assumed_clean"])):
        msk = ctlm & (cfg[tok_doc] == c)
        fpc[c] = round(float(pred[msk].mean()), 4)
    out["fp_by_config"] = fpc
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
    out["bootstrap"] = {"R_ci90": [round(float(x), 3) for x in np.percentile(boot[:, 0], [5, 95])],
                        "F2_ci90": [round(float(x), 3) for x in np.percentile(boot[:, 1], [5, 95])]}
    cc = {}; cf = {}
    for l in mem_set:
        cc[cat[l]] = cc.get(cat[l], 0) + 1
        cf[conf[l]] = cf.get(conf[l], 0) + 1
    og = set(int(k) for k in json.load(open(f"{ROOT}/data/latent_list_2026-07-29.json"))["tags"])
    out["composition"] = {"category": cc, "confidence": cf,
                          "overlap_with_OG222": len(og & set(mem_set))}
    return out

# starts
v3 = json.load(open(f"{ROOT}/out/fable_trim_v3_anneal.json"))
start_fp10 = set(v3["best_near"]["members"])
g = json.load(open(f"{ROOT}/out/fable_trim_v2_greedy.json"))
traj, removed = g["trajectory"], g["removed_order"]
pt = min((t for t in traj if t["R"] >= 0.90), key=lambda t: t["fp_ctl"])
start_r90 = set(int(x) for x in members) - set(removed[:traj.index(pt) + 1])

rng = np.random.default_rng(SEED)
res = {}
print("== P1: fp<=.10, max recall ==")
s1, m1 = climb(start_fp10, key_p1, "P1", rng)
print("== P2: fp<=.10, max F2 ==")
s2, m2 = climb(start_fp10, key_p2, "P2", rng)
print("== P3: R>=.90, min fp ==")
s3, m3 = climb(start_r90, key_p3, "P3", rng)

for tag, s, desc in (("p1", s1, "fp_ctl<=0.10 hard; recall maximized (F2 tiebreak)"),
                     ("p2", s2, "fp_ctl<=0.10 hard; F2 maximized"),
                     ("p3", s3, "R>=0.90 hard; fp_ctl minimized (F2 tiebreak)")):
    fm = full_metrics(s)
    res[tag] = {"objective": desc, "n": len(s), **fm}
    print(f"\n{tag.upper()} n={len(s)}: " + json.dumps({k: fm[k] for k in
          ("P", "R", "F2", "fp_ctl", "fp_clean_docs")}))
    out = {"name": f"fable_trim_{tag}",
           "created": "2026-08-14",
           "source_pool": "fable_16k (haiku_fable_forget_latents_v1.json), 649 latents",
           "method": {"pipeline": "LLR sweeps (v1) -> greedy backward (v2) -> annealing (v3) "
                                  "-> constrained hill-climb polish (v4)",
                      "objective": desc,
                      "note": "R>=0.90 AND fp_ctl<=0.10 shown jointly infeasible for this pool "
                              "under the frozen Rathi rule (see out/fable_trim_v3_anneal.json)",
                      "harvest": HARV, "selection_split": "TEST (manifest v4, sha 8fa54624)",
                      "seed": SEED},
           "test_point": {k: fm[k] for k in ("P", "R", "F2", "F1", "fp_ctl", "fp_clean_docs")},
           "detail": {k: v for k, v in fm.items() if k.startswith("recall_") or k in
                      ("fp_by_config", "bootstrap", "composition")},
           "members": sorted(s)}
    path = f"{ROOT}/data/candidate_sets/fable_trim_{tag}.json"
    json.dump(out, open(path, "w"), indent=1)
    print("froze", path)

json.dump(res, open(f"{ROOT}/out/fable_trim_v4_polish.json", "w"), indent=1)
print(f"\nwrote {ROOT}/out/fable_trim_v4_polish.json")
