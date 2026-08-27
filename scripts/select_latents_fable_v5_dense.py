"""Fable latent selection v5: protocol-v5 main forget set (2026-08-15).

User decision: MAIN forget = human span tokens + ALL tokens of dense c1-c4
docs (register-matched to controls; c5/SWE stays retain). Human-span metrics
remain a separate secondary report.

Runs the constrained hill-climb (as v4) with the main-metric objective:
  D1  fp_ctl <= .10 hard; maximize main recall (F2 tiebreak)
  D2  fp_ctl <= .10 hard; maximize main F2
Starts: fable_trim_p1/p2 members, v3 best_near, LLR-C prefix top-350.
Also reports every set (D1, D2, P1..P3, OG, full pool) under BOTH metrics,
plus per-feature evidence (q_span, q_dense14, r_ctl) for the feature-shift
analysis. Selection on TEST only; sealed validation untouched.
"""
import json

import numpy as np

ROOT = "/home/a5k/cwtice.a5k/sae-exploration"
HARV = f"{ROOT}/out/harvest_test_v3"
DOCS = f"{ROOT}/data/test_docs_v4.jsonl"
POOL = f"{ROOT}/data/candidate_sets/haiku_fable_forget_latents_v1.json"
SEED = 20260815
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
dense14 = (lt[tok_doc] == "dense_forget") & (np.char.find(cfg[tok_doc].astype(str), "c5") < 0)
is_doc_start = np.zeros(n_tok, bool); is_doc_start[tok_base[:-1]] = True
gt_ann = gt & annm; n_gt = int(gt_ann.sum())
mainm = annm | dense14
gt_main = (gt & annm) | dense14
n_gtm = int(gt_main.sum())
print(f"main forget tokens {n_gtm:,} (spans {n_gt:,} + dense14 {int(dense14.sum()):,})")

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

# per-feature evidence (raw fire rates above threshold)
def rate(mask):
    k = np.zeros(nm)
    h = mask[mf_s] & ma_s
    np.add.at(k, mp_s[h], 1)
    return k / max(int(mask.sum()), 1)

q_span = rate(gt_ann); q_dense = rate(dense14); r_ctl = rate(ctlm)
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
        return None, dict(Pm=0, Rm=0, F2m=0, F1m=0, P=0, R=0, F2=0, fp_ctl=0, fp_clean_docs=0)
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

def key_d1(m): return (m["fp_ctl"] <= 0.10, m["Rm"], m["F2m"])
def key_d2(m): return (m["fp_ctl"] <= 0.10, m["F2m"], m["Rm"])

def state_for(mem):
    seedcnt = np.zeros(n_tok, np.int32); pacnt = np.zeros(n_tok, np.int32)
    for l in mem:
        i = lat2pos[l]
        np.add.at(seedcnt, per_above[i], 1)
        np.add.at(pacnt, per_flat[i], 1)
    return seedcnt, pacnt

def climb(start_set, key, tag, rng):
    cur_set = set(start_set)
    seedcnt, pacnt = state_for(cur_set)
    _, m = evaluate(seedcnt, pacnt)
    best_set, best_m = set(cur_set), dict(m)
    for rnd in range(ROUNDS):
        if rnd > 0:
            cur_set = set(best_set)
            seedcnt, pacnt = state_for(cur_set)
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
        print(f"  {tag} round {rnd}: Rm={best_m['Rm']:.4f} F2m={best_m['F2m']:.4f} "
              f"fp={best_m['fp_ctl']:.4f} n={len(best_set)}")
    return best_set, best_m

# starts: prior frozen sets + LLR-C prefix + v3 best_near
p1 = set(json.load(open(f"{ROOT}/data/candidate_sets/fable_trim_p1.json"))["members"])
p2 = set(json.load(open(f"{ROOT}/data/candidate_sets/fable_trim_p2.json"))["members"])
p3 = set(json.load(open(f"{ROOT}/data/candidate_sets/fable_trim_p3.json"))["members"])
bn = set(json.load(open(f"{ROOT}/out/fable_trim_v3_anneal.json"))["best_near"]["members"])
sw = json.load(open(f"{ROOT}/out/fable_trim_v1_sweeps.json"))
llrC = set(sw["C"]["rank"][:350])
og = sorted(int(k) for k in json.load(open(f"{ROOT}/data/latent_list_2026-07-29.json"))["tags"])

rng = np.random.default_rng(SEED)
results = {}
for tag, key in (("D1", key_d1), ("D2", key_d2)):
    best = None
    for sname, s0 in (("p2", p2), ("p1", p1), ("best_near", bn), ("llrC350", llrC)):
        print(f"== {tag} from {sname} ==")
        s, m = climb(s0, key, f"{tag}/{sname}", rng)
        if best is None or key(m) > key(best[1]):
            best = (s, m, sname)
    s, m, sname = best
    results[tag] = {"start": sname, "n": len(s), "members": sorted(s),
                    **{k: round(v, 4) for k, v in m.items()}}
    print(f"{tag} WINNER (from {sname}) n={len(s)}: " +
          json.dumps({k: round(m[k], 4) for k in ("Pm", "Rm", "F2m", "P", "R", "F2", "fp_ctl")}))

# reference rows under both metrics
refs = {}
for name, mem in (("P1", p1), ("P2", p2), ("P3", p3), ("OG", og),
                  ("full_pool", list(map(int, members)))):
    seedcnt, pacnt = state_for(mem)
    _, m = evaluate(seedcnt, pacnt)
    refs[name] = {k: round(v, 4) for k, v in m.items()}
results["references"] = refs

# per-feature evidence table for the shift analysis
results["feature_evidence"] = {
    int(members[i]): {"q_span": round(float(q_span[i]), 5),
                      "q_dense": round(float(q_dense[i]), 5),
                      "r_ctl": round(float(r_ctl[i]), 5),
                      "cat": cat[int(members[i])], "conf": conf[int(members[i])]}
    for i in range(nm)}

json.dump(results, open(f"{ROOT}/out/fable_trim_v5_dense.json", "w"))
print(f"wrote {ROOT}/out/fable_trim_v5_dense.json")

for tag in ("D1", "D2"):
    w = results[tag]
    out = {"name": f"fable_trim_{tag.lower()}",
           "created": "2026-08-15",
           "source_pool": "fable_16k (haiku_fable_forget_latents_v1.json), 649 latents",
           "method": {"pipeline": "constrained hill-climb (v5), multi-start",
                      "objective": ("max main recall" if tag == "D1" else "max main F2")
                                   + " s.t. fp_ctl<=0.10; main forget = spans + dense c1-c4 "
                                     "(protocol v5)",
                      "harvest": HARV, "selection_split": "TEST (manifest v4, sha 8fa54624)",
                      "seed": SEED},
           "test_point_main": {k: w[k] for k in ("Pm", "Rm", "F2m", "F1m")},
           "test_point_annotated": {k: w[k] for k in ("P", "R", "F2")},
           "fp": {k: w[k] for k in ("fp_ctl", "fp_clean_docs")},
           "members": w["members"]}
    path = f"{ROOT}/data/candidate_sets/fable_trim_{tag.lower()}.json"
    json.dump(out, open(path, "w"), indent=1)
    print("froze", path)
