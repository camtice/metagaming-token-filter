"""Fable latent selection v3: multi-start simulated annealing (2026-08-14).

Greedy (v2) reaches R=.903 @ fp_ctl=.162 but cannot satisfy R>=.90 AND
fp_ctl<=.10 along its path. This probes the subset lattice globally:

  - state: subset of the 649 fable_16k latents
  - objective: F2 - 5*max(0, .90-R) - 5*max(0, fp_ctl-.10)  (exact Rathi eval)
  - moves: single toggle (biased: removals ~ control-seed mass, additions ~
    span-carrier rate) with occasional 2-swaps; geometric cooling
  - 4 chains from distinct starts (greedy endpoints, LLR prefix, random)
  - records the global Pareto front over ALL visited states -> feasibility
    evidence, plus the best feasible point if one exists

Selection split: TEST only (manifest v4). Sealed validation untouched.
Output: out/fable_trim_v3_anneal.json
"""
import json

import numpy as np

ROOT = "/home/a5k/cwtice.a5k/sae-exploration"
HARV = f"{ROOT}/out/harvest_test_v3"
DOCS = f"{ROOT}/data/test_docs_v4.jsonl"
POOL = f"{ROOT}/data/candidate_sets/haiku_fable_forget_latents_v1.json"
MIN_R, MAX_FP = 0.90, 0.10
ITERS, SEED = 15000, 20260814

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

# proposal biases
q_span = np.zeros(nm); r_ctl = np.zeros(nm)
hit_g = gt_ann[mf_s] & ma_s; np.add.at(q_span, mp_s[hit_g], 1)
hit_c = ctlm[mf_s] & ma_s; np.add.at(r_ctl, mp_s[hit_c], 1)
q_span = (q_span + 0.5) / (n_gt + 1)
r_ctl = (r_ctl + 0.5) / (int(ctlm.sum()) + 1)
w_add = q_span / q_span.sum()
w_rem = r_ctl / r_ctl.sum()

prev_shift = np.zeros(n_tok, bool)

def evaluate(seedcnt, pacnt):
    seeds = seedcnt >= 2
    pa = pacnt > 0
    prev_shift[1:] = pa[:-1]; prev_shift[0] = False
    start = pa & (~prev_shift | is_doc_start)
    run_id = np.cumsum(start) - 1
    nruns = int(start.sum())
    if nruns == 0:
        return {"P": 0.0, "R": 0.0, "F2": 0.0, "fp_ctl": 0.0, "fp_clean_docs": 0.0}
    seed_runs = np.zeros(nruns, bool)
    seed_runs[run_id[seeds]] = True
    pred = pa & seed_runs[run_id]
    tp = int((pred & gt_ann).sum()); fpa = int((pred & annm & ~gt).sum())
    R = tp / max(n_gt, 1); P = tp / max(tp + fpa, 1)
    return {"P": P, "R": R, "F2": 5*P*R/max(4*P+R, 1e-9),
            "fp_ctl": float(pred[ctlm].mean()), "fp_clean_docs": float(pred[cleanm].mean())}

def score(m):
    return m["F2"] - 5*max(0.0, MIN_R - m["R"]) - 5*max(0.0, m["fp_ctl"] - MAX_FP)

# ---------- starting states ----------
g = json.load(open(f"{ROOT}/out/fable_trim_v2_greedy.json"))
allset = set(int(x) for x in members)
starts = {}
traj, removed = g["trajectory"], g["removed_order"]
pt = min((t for t in traj if t["R"] >= 0.90), key=lambda t: t["fp_ctl"])
starts["greedy_R90"] = allset - set(removed[:traj.index(pt) + 1])
lo = [t for t in traj if t["fp_ctl"] <= MAX_FP]
if lo:
    pt2 = max(lo, key=lambda t: t["R"])
    starts["greedy_fp10"] = allset - set(removed[:traj.index(pt2) + 1])
sw = json.load(open(f"{ROOT}/out/fable_trim_v1_sweeps.json"))
starts["llr_K375"] = set(sw["A"]["rank"][:375])
rng = np.random.default_rng(SEED)
starts["random_300"] = set(int(x) for x in rng.choice(members, 300, replace=False))

pareto = []   # (R, fp_ctl, F2, n) over all visited
best_feasible = None
best_near = None   # min violation, tie-break by F2

def note(m, cur_set):
    global best_feasible, best_near
    pareto.append((round(m["R"], 4), round(m["fp_ctl"], 4), round(m["F2"], 4), len(cur_set)))
    v = max(0.0, MIN_R - m["R"]) + max(0.0, m["fp_ctl"] - MAX_FP)
    if v == 0.0 and (best_feasible is None or m["F2"] > best_feasible["F2"]):
        best_feasible = {**{k: round(x, 4) for k, x in m.items()},
                         "n": len(cur_set), "members": sorted(cur_set)}
    if best_near is None or (v, -m["F2"]) < (best_near["viol"], -best_near["F2"]):
        best_near = {**{k: round(x, 4) for k, x in m.items()}, "viol": round(v, 4),
                     "n": len(cur_set), "members": sorted(cur_set)}

for sname, s0 in starts.items():
    cur_set = set(s0)
    seedcnt = np.zeros(n_tok, np.int32); pacnt = np.zeros(n_tok, np.int32)
    for l in cur_set:
        i = lat2pos[l]
        np.add.at(seedcnt, per_above[i], 1)
        np.add.at(pacnt, per_flat[i], 1)
    m = evaluate(seedcnt, pacnt); sc = score(m); note(m, cur_set)
    T0, T1 = 0.02, 0.0005
    acc = 0
    for it in range(ITERS):
        T = T0 * (T1 / T0) ** (it / ITERS)
        # propose: 70% biased toggle, 30% swap
        if rng.random() < 0.7:
            if rng.random() < 0.5 and len(cur_set) > 2:
                i = int(rng.choice(nm, p=w_rem))
                if int(members[i]) not in cur_set:
                    i = lat2pos[int(rng.choice(sorted(cur_set)))]
                toggles = [(-1, i)]
            else:
                i = int(rng.choice(nm, p=w_add))
                if int(members[i]) in cur_set:
                    outp = [l for l in map(int, members) if l not in cur_set]
                    if not outp: continue
                    i = lat2pos[int(rng.choice(outp))]
                toggles = [(+1, i)]
        else:
            ins = sorted(cur_set); outs = [l for l in map(int, members) if l not in cur_set]
            if not ins or not outs: continue
            toggles = [(-1, lat2pos[int(rng.choice(ins))]), (+1, lat2pos[int(rng.choice(outs))])]
        for sgn, i in toggles:
            f = np.add.at if sgn > 0 else np.subtract.at
            f(seedcnt, per_above[i], 1); f(pacnt, per_flat[i], 1)
        m2 = evaluate(seedcnt, pacnt); sc2 = score(m2)
        if sc2 >= sc or rng.random() < np.exp((sc2 - sc) / T):
            for sgn, i in toggles:
                l = int(members[i])
                cur_set.add(l) if sgn > 0 else cur_set.discard(l)
            m, sc = m2, sc2; acc += 1
            note(m, cur_set)
        else:
            for sgn, i in toggles:  # revert
                f = np.subtract.at if sgn > 0 else np.add.at
                f(seedcnt, per_above[i], 1); f(pacnt, per_flat[i], 1)
    print(f"chain {sname}: accepted {acc}/{ITERS}, end R={m['R']:.4f} fp={m['fp_ctl']:.4f} F2={m['F2']:.4f}")
    if best_feasible:
        print(f"   best feasible so far: n={best_feasible['n']} R={best_feasible['R']} "
              f"fp={best_feasible['fp_ctl']} F2={best_feasible['F2']}")

# empirical Pareto front (max R for each fp bin)
pare = {}
for R, fpc, F2, n in pareto:
    b = round(fpc, 3)
    if b not in pare or R > pare[b][0]:
        pare[b] = (R, F2, n)
front = sorted((b, *v) for b, v in pare.items() if b <= 0.30)
res = {"constraints": {"min_recall": MIN_R, "max_fp_ctl": MAX_FP},
       "iters_per_chain": ITERS, "n_visited": len(pareto),
       "best_feasible": best_feasible, "best_near": best_near,
       "pareto_fp_to_maxR": [{"fp_bin": b, "max_R": r, "F2": f, "n": n}
                             for b, r, f, n in front]}
json.dump(res, open(f"{ROOT}/out/fable_trim_v3_anneal.json", "w"))
print("\nbest_near:", None if not best_near else {k: best_near[k] for k in
      ("n", "P", "R", "F2", "fp_ctl", "fp_clean_docs", "viol")})
print("Pareto (fp bin -> max R seen):")
for b, r, f, n in front[::max(1, len(front)//25)]:
    print(f"  fp<={b:.3f}: R={r:.4f} F2={f:.4f} n={n}")
print(f"wrote {ROOT}/out/fable_trim_v3_anneal.json")
