"""Generalization study: train/test split INSIDE current TEST v5 (2026-08-15).

Motivation (user): every method so far computed statistics and/or optimized on
the corpus it was scored on, so method comparisons are confounded by test-set
reuse. This study isolates the selection step:

  G-TRAIN  seeded half of dense c1/c3/c4 (15 docs each) + seeded half of the
           2,625 retain controls. ALL selection information comes from here.
  G-TEST   other dense halves (in-domain), ALL dense c2 (held-out domain),
           ALL human-span docs (never trained on -> pure transfer to real
           rollouts), reviewed-clean docs, other control half.
  Sealed VAL-A/VAL-B untouched. Dense c5 excluded (retain-scope ruling).

Methods per pool (fable_16k, haiku_v6/v8_16k, haiku_v6/v8_65k):
  R3 rho in {1,2,3,5,10}       stats on G-TRAIN only
  R1 tau grid (12 values)      stats on G-TRAIN controls only
  train-selected R3/R1         threshold picked by best train F2 @ fp<=.10
  HILL-CLIMB                   optimized on G-TRAIN only (fp_train<=.10)
  R3-contaminated              rho grid with FULL-corpus stats (old practice)
                               -> the contamination delta, same G-TEST
  no-training baselines        full pool, embed>=0.9; OG (16k only)

G-TEST metrics: R_span (human spans; transfer), R_c2 (held-out domain),
R_dense_in (in-domain halves), fp_ctl / fp_clean, F2_in (forget = spans +
in-domain dense; c2 excluded from mask), F2_all (c2 included as forget).
Rathi thresholds (mean+4SD) from the full TEST v5 harvest, fixed for all.

Usage: python generalization_study.py            (runs everything)
Output: out/gen_study_<pool>.json + data/splits/gen_split_20260815.json
"""
import json

import numpy as np

ROOT = "/home/a5k/cwtice.a5k/sae-exploration"
DOCS = f"{ROOT}/data/test_docs_v5.jsonl"
POOL_FILE = f"{ROOT}/data/candidate_sets/haiku_fable_forget_latents_v1.json"
SEED = 20260815
RHOS = [1, 2, 3, 5, 10]
TAUS = [0.02, 0.01, 0.005, 0.003, 0.002, 0.0015, 0.001, 0.0007, 0.0005, 0.0003, 0.0002, 0.0001]
CLIMB_ROUNDS, CLIMB_STEPS, CLIMB_SAMPLE = 2, 150, 40
RUNS = [("fable_16k", "16k"), ("haiku_v6_16k", "16k"), ("haiku_v8_16k", "16k"),
        ("haiku_v6_65k", "65k"), ("haiku_v8_65k", "65k")]
HARV = {"16k": f"{ROOT}/out/harvest_test_v5", "65k": f"{ROOT}/out/harvest_test_v5_65k"}
EMB = {"16k": f"{ROOT}/out/embed_scores_16k.json", "65k": f"{ROOT}/out/embed_scores_65k.json"}

docs = [json.loads(l) for l in open(DOCS)]
by_id = {d["id"]: d for d in docs}

# ---------- the split (doc-level, seeded, saved) ----------
rng = np.random.default_rng(SEED)
role = {}
dense_in_classes = ("dense_c1_ai_evals", "dense_c3_human_oversight", "dense_c4_training_mechanics")
for cls in dense_in_classes:
    ids = sorted(d["id"] for d in docs if d.get("config") == cls)
    tr = set(rng.choice(ids, len(ids) // 2, replace=False).tolist())
    for i in ids:
        role[i] = "train_dense" if i in tr else "gtest_dense"
for d in docs:
    i = d["id"]
    if i in role:
        continue
    cfg = d.get("config", "")
    if cfg == "dense_c2_ai_safety":
        role[i] = "gtest_c2"
    elif cfg == "dense_c5_swe_tests":
        role[i] = "excluded_c5"
    elif d["label_type"] == "spans":
        role[i] = "gtest_span"
    elif d["label_type"] == "clean":
        role[i] = "gtest_clean"
    elif d["label_type"] == "assumed_clean":
        role[i] = None  # assigned below
ctl_ids = sorted(d["id"] for d in docs if role.get(d["id"]) is None)
tr = set(rng.choice(ctl_ids, len(ctl_ids) // 2, replace=False).tolist())
for i in ctl_ids:
    role[i] = "train_ctl" if i in tr else "gtest_ctl"
from collections import Counter
print("split:", dict(Counter(role.values())))
json.dump({"seed": SEED, "parent": "TEST v5 (manifest 0f6f83c7)",
           "rules": "G-TRAIN = dense c1/c3/c4 halves + control half; G-TEST = rest; "
                    "c2 fully held out; spans/clean never trained on; c5 excluded",
           "roles": role},
          open(f"{ROOT}/data/splits/gen_split_20260815.json", "w"))

for pool_name, width in RUNS:
    zf = np.load(HARV[width] + ".npz")
    z = {k: zf[k] for k in zf.files}
    meta = json.load(open(HARV[width] + ".meta.json"))
    W = meta["width"]
    av = z["act"].astype(np.float64); av[np.isinf(av)] = 65504.0
    ntok = z["doc_ntok"].astype(np.int64)
    nd = len(meta["docs"])
    tok_base = np.zeros(nd + 1, np.int64); np.cumsum(ntok, out=tok_base[1:])
    n_tok = int(ntok.sum())
    flat = tok_base[z["doc_idx"]] + z["tok_idx"]
    offs = z["offsets"].reshape(-1, 2)
    mu = np.bincount(z["lat_idx"], weights=av, minlength=W) / n_tok
    sdv = np.sqrt(np.maximum(np.bincount(z["lat_idx"], weights=av**2, minlength=W)/n_tok - mu**2, 0))
    thr = mu + 4 * sdv

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
    is_doc_start = np.zeros(n_tok, bool); is_doc_start[tok_base[:-1]] = True

    m_train_dense = trole == "train_dense"
    m_train_ctl = trole == "train_ctl"
    m_g_dense = trole == "gtest_dense"
    m_c2 = trole == "gtest_c2"
    m_span_docs = trole == "gtest_span"
    m_clean = trole == "gtest_clean"
    m_g_ctl = trole == "gtest_ctl"
    gt_span = gt & m_span_docs
    n_span = int(gt_span.sum())

    pool_raw = json.load(open(POOL_FILE))["sets"][pool_name]["latents"]
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

    # TRAIN-only statistics
    c_tr_f = counts(m_train_dense); n_tr_f = int(m_train_dense.sum())
    c_tr_c = counts(m_train_ctl); n_tr_c = int(m_train_ctl.sum())
    rr_train = ((c_tr_f + 0.5) / n_tr_f) / ((c_tr_c + 0.5) / n_tr_c)
    rate_c_train = c_tr_c / n_tr_c
    # CONTAMINATED statistics (old practice: whole corpus incl. G-TEST + spans)
    m_full_forget = gt | m_train_dense | m_g_dense | m_c2
    c_fu_f = counts(m_full_forget); n_fu_f = int(m_full_forget.sum())
    m_full_ctl = m_train_ctl | m_g_ctl
    c_fu_c = counts(m_full_ctl); n_fu_c = int(m_full_ctl.sum())
    rr_full = ((c_fu_f + 0.5) / n_fu_f) / ((c_fu_c + 0.5) / n_fu_c)

    prev_shift = np.zeros(n_tok, bool)

    def predict(keep_pos):
        seedcnt = np.zeros(n_tok, np.int32); pacnt = np.zeros(n_tok, np.int32)
        for i in keep_pos:
            np.add.at(seedcnt, per_above[i], 1)
            np.add.at(pacnt, per_flat[i], 1)
        return _pred(seedcnt, pacnt)

    def _pred(seedcnt, pacnt):
        seeds = seedcnt >= 2
        pa = pacnt > 0
        prev_shift[1:] = pa[:-1]; prev_shift[0] = False
        start = pa & (~prev_shift | is_doc_start)
        run_id = np.cumsum(start) - 1
        nruns = int(start.sum())
        if nruns == 0:
            return np.zeros(n_tok, bool)
        sr = np.zeros(nruns, bool); sr[run_id[seeds]] = True
        return pa & sr[run_id]

    def f2(P, R): return 5 * P * R / max(4 * P + R, 1e-9)

    def train_metrics(pred):
        tp = int(pred[m_train_dense].sum())
        fpa = int(pred[m_train_ctl].sum())  # precision proxy vs train ctl
        R = tp / max(n_tr_f, 1)
        P = tp / max(tp + fpa, 1)
        return {"F2_train": round(f2(P, R), 4), "R_train": round(R, 4),
                "fp_train": round(float(pred[m_train_ctl].mean()), 4)}

    def gtest_metrics(pred, n_kept):
        R_span = float(pred[gt_span].mean()) if n_span else 0.0
        R_in = float(pred[m_g_dense].mean())
        R_c2v = float(pred[m_c2].mean())
        fp_ctl = float(pred[m_g_ctl].mean())
        fp_cl = float(pred[m_clean].mean())
        # F2_in: forget = span tokens + in-domain gtest dense; eval mask excludes c2
        mask_in = m_span_docs | m_clean | m_g_dense | m_g_ctl
        gt_in = gt_span | m_g_dense
        tp = int((pred & gt_in & mask_in).sum()); fpa = int((pred & mask_in & ~gt_in).sum())
        fn = int((~pred & gt_in).sum())
        P = tp / max(tp + fpa, 1); R = tp / max(tp + fn, 1)
        # F2_all: c2 counts as forget too
        mask_all = mask_in | m_c2
        gt_all = gt_in | m_c2
        tp2 = int((pred & gt_all & mask_all).sum()); fpa2 = int((pred & mask_all & ~gt_all).sum())
        fn2 = int((~pred & gt_all).sum())
        P2 = tp2 / max(tp2 + fpa2, 1); R2 = tp2 / max(tp2 + fn2, 1)
        return {"n": n_kept, "R_span": round(R_span, 4), "R_c2": round(R_c2v, 4),
                "R_dense_in": round(R_in, 4), "fp_ctl": round(fp_ctl, 4),
                "fp_clean": round(fp_cl, 4),
                "F2_in": round(f2(P, R), 4), "F2_all": round(f2(P2, R2), 4)}

    res = {"pool": pool_name, "width": width, "n_pool": nm,
           "split_seed": SEED, "methods": {}}
    print(f"\n==== {pool_name} ({nm} latents, {width}) ====", flush=True)

    def run_rule(name, keep_pos, extra=None):
        pred = predict(keep_pos)
        m = {**gtest_metrics(pred, len(keep_pos)), **train_metrics(pred)}
        if extra: m.update(extra)
        res["methods"][name] = m
        print(f"  {name:26s} n={m['n']:4d} F2in={m['F2_in']:.3f} F2all={m['F2_all']:.3f} "
              f"Rspan={m['R_span']:.3f} Rc2={m['R_c2']:.3f} fp={m['fp_ctl']:.3f} "
              f"(train F2={m['F2_train']:.3f} fp={m['fp_train']:.3f})", flush=True)
        return m

    # R3 grid (train stats)
    best_r3, best_r3_key = None, None
    for rho in RHOS:
        m = run_rule(f"R3_train rho={rho}", np.flatnonzero(rr_train >= rho))
        if m["fp_train"] <= 0.10 and (best_r3 is None or m["F2_train"] > best_r3["F2_train"]):
            best_r3, best_r3_key = m, f"R3_train rho={rho}"
    res["train_selected_R3"] = best_r3_key
    # R1 grid (train stats)
    best_r1, best_r1_key = None, None
    for tau in TAUS:
        m = run_rule(f"R1_train tau={tau}", np.flatnonzero(rate_c_train <= tau))
        if m["fp_train"] <= 0.10 and (best_r1 is None or m["F2_train"] > best_r1["F2_train"]):
            best_r1, best_r1_key = m, f"R1_train tau={tau}"
    res["train_selected_R1"] = best_r1_key
    # contaminated R3 grid (full-corpus stats)
    for rho in RHOS:
        run_rule(f"R3_contam rho={rho}", np.flatnonzero(rr_full >= rho))
    # no-training baselines
    run_rule("full_pool", np.arange(nm))
    es = {int(k): v for k, v in json.load(open(EMB[width])).items()}
    esc = np.array([es.get(int(l)) if es.get(int(l)) is not None else -1.0 for l in members])
    run_rule("embed>=0.9 (no training)", np.flatnonzero(esc >= 0.9))
    if width == "16k":
        og = set(int(k) for k in json.load(open(f"{ROOT}/data/latent_list_2026-07-29.json"))["tags"])
        og_pos = [int(lat2pos[l]) for l in og if lat2pos[l] >= 0]
        run_rule("OG∩pool (no training)", np.array(og_pos, dtype=int),
                 extra={"note": f"{len(og_pos)} of OG-222 present in this pool"})

    # HILL-CLIMB on G-TRAIN only
    crng = np.random.default_rng(SEED + 1)
    w_add = (c_tr_f / max(n_tr_f, 1) + 1e-4); w_add /= w_add.sum()
    w_rem = (c_tr_c / max(n_tr_c, 1) + 1e-4); w_rem /= w_rem.sum()
    start = np.flatnonzero(rr_train >= 3)
    cur = set(int(x) for x in start)
    seedcnt = np.zeros(n_tok, np.int32); pacnt = np.zeros(n_tok, np.int32)
    for i in cur:
        np.add.at(seedcnt, per_above[i], 1)
        np.add.at(pacnt, per_flat[i], 1)
    def tkey(m): return (m["fp_train"] <= 0.10, m["F2_train"])
    tm = train_metrics(_pred(seedcnt, pacnt))
    best_set, best_tm = set(cur), dict(tm)
    for rnd in range(CLIMB_ROUNDS):
        for _ in range(CLIMB_STEPS):
            props = []
            outs = [i for i in range(nm) if i not in cur]
            ins = sorted(cur)
            for _ in range(CLIMB_SAMPLE):
                r = crng.random()
                if r < 0.35 and outs:
                    props.append([(+1, int(crng.choice(nm, p=w_add)))])
                elif r < 0.70 and len(ins) > 2:
                    props.append([(-1, int(crng.choice(nm, p=w_rem)))])
                elif outs and len(ins) > 2:
                    props.append([(-1, int(crng.choice(ins))), (+1, int(crng.choice(outs)))])
            bp, bk, bm = None, tkey(tm), None
            for tg in props:
                ok = all((sgn > 0) != (i in cur) for sgn, i in tg)
                if not ok:
                    continue
                for sgn, i in tg:
                    f = np.add.at if sgn > 0 else np.subtract.at
                    f(seedcnt, per_above[i], 1); f(pacnt, per_flat[i], 1)
                m2 = train_metrics(_pred(seedcnt, pacnt))
                if tkey(m2) > bk:
                    bp, bk, bm = tg, tkey(m2), m2
                for sgn, i in tg:
                    f = np.subtract.at if sgn > 0 else np.add.at
                    f(seedcnt, per_above[i], 1); f(pacnt, per_flat[i], 1)
            if bp is None:
                break
            for sgn, i in bp:
                f = np.add.at if sgn > 0 else np.subtract.at
                f(seedcnt, per_above[i], 1); f(pacnt, per_flat[i], 1)
                cur.add(i) if sgn > 0 else cur.discard(i)
            tm = bm
            if tkey(tm) > tkey(best_tm):
                best_set, best_tm = set(cur), dict(tm)
    run_rule("HILLCLIMB_train", np.array(sorted(best_set), dtype=int))

    json.dump(res, open(f"{ROOT}/out/gen_study_{pool_name}.json", "w"), indent=1)
    print(f"wrote out/gen_study_{pool_name}.json", flush=True)
print("GEN_STUDY_ALL_DONE")
