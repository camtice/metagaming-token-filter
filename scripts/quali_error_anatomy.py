"""Q1: error anatomy of the candidate rules on G-TEST (2026-08-15).

For each headline rule set, on G-TEST only:
  - span FN: per annotation category (from span triples), per doc, per config;
    tokens missed by ALL rules vs rescued by some
  - c2 misses: per c2 doc (which held-out docs are hard)
  - FP anatomy on gtest controls: seed vs window share, per config, top
    seeding features w/ surface forms
  - pairwise rule agreement (Jaccard on flagged tokens) + where disagreements live
Rules compared:
  R3-16k rho=5 (haiku_v6), R3-65k rho=5 (haiku_v6), R3-65k rho=10,
  R1-16k train-selected, HILLCLIMB h6_16k, D2 (frozen), full-pool-65k ref.
Transcript hygiene: only single-token surfaces and aggregates printed.
"""
import json
import re
from collections import Counter, defaultdict

import numpy as np

ROOT = "/home/a5k/cwtice.a5k/sae-exploration"
role = json.load(open(f"{ROOT}/data/splits/gen_split_20260815.json"))["roles"]
docs = [json.loads(l) for l in open(f"{ROOT}/data/test_docs_v5.jsonl")]
by_id = {d["id"]: d for d in docs}

# span categories from the annotation export (third element of span triples)
EXP = json.load(open(f"{ROOT}/data/annotation_exports/metagaming_token_labels_20260809T1907.json"))
CATN = {1: "c1_evals", 2: "c2_safety", 3: "c3_human", 4: "c4_training", 0: "uncat"}

def load_width(width):
    HARV = {"16k": f"{ROOT}/out/harvest_test_v5", "65k": f"{ROOT}/out/harvest_test_v5_65k"}[width]
    zf = np.load(HARV + ".npz")
    z = {k: zf[k] for k in zf.files}
    meta = json.load(open(HARV + ".meta.json"))
    W = meta["width"]
    av = z["act"].astype(np.float64); av[np.isinf(av)] = 65504.0
    ntok = z["doc_ntok"].astype(np.int64); nd = len(meta["docs"])
    tok_base = np.zeros(nd + 1, np.int64); np.cumsum(ntok, out=tok_base[1:])
    n_tok = int(ntok.sum())
    flat = tok_base[z["doc_idx"]] + z["tok_idx"]
    mu = np.bincount(z["lat_idx"], weights=av, minlength=W) / n_tok
    sdv = np.sqrt(np.maximum(np.bincount(z["lat_idx"], weights=av**2, minlength=W)/n_tok - mu**2, 0))
    return dict(z=z, meta=meta, W=W, av=av, ntok=ntok, nd=nd, tok_base=tok_base,
                n_tok=n_tok, flat=flat, thr=mu + 4 * sdv,
                doc_ids=[d["id"] for d in meta["docs"]],
                offs=z["offsets"].reshape(-1, 2))

CTX = {w: load_width(w) for w in ("16k", "65k")}

def doc_masks(ctx):
    rl = np.array([role[i] for i in ctx["doc_ids"]])
    tok_doc = np.repeat(np.arange(ctx["nd"]), ctx["ntok"])
    trole = rl[tok_doc]
    gt = np.zeros(ctx["n_tok"], bool)
    span_cat = np.zeros(ctx["n_tok"], np.int8)
    for di, did in enumerate(ctx["doc_ids"]):
        d = by_id[did]
        if not d.get("char_spans"):
            continue
        o = ctx["offs"][ctx["tok_base"][di]:ctx["tok_base"][di + 1]]
        # char spans; category via matching word spans from export where possible
        exp_rec = EXP["docs"].get(did)
        words = d["text"].split()
        # word start offsets
        woff, pos = [], 0
        for w in words:
            s = d["text"].find(w, pos)
            woff.append((s, s + len(w))); pos = s + len(w)
        catmap = {}
        if exp_rec:
            for sp in exp_rec["spans"]:
                st, en = sp[0], sp[1]
                c = sp[2] if len(sp) > 2 else 0
                for wi in range(st, min(en, len(woff))):
                    catmap[wi] = c
        for cs, ce in d["char_spans"]:
            m = (o[:, 0] < ce) & (o[:, 1] > cs)
            idxs = ctx["tok_base"][di] + np.flatnonzero(m)
            gt[idxs] = True
            for t in idxs:
                ts, te = ctx["offs"][t]
                cat = 0
                for wi, (ws, we) in enumerate(woff):
                    if ws < te and we > ts:
                        cat = catmap.get(wi, 0); break
                span_cat[t] = cat
    return dict(trole=trole, tok_doc=tok_doc, gt=gt, span_cat=span_cat,
                m_span=(trole == "gtest_span") & gt,
                m_c2=trole == "gtest_c2", m_gd=trole == "gtest_dense",
                m_ctl=trole == "gtest_ctl", m_clean=trole == "gtest_clean")

MK = {w: doc_masks(CTX[w]) for w in ("16k", "65k")}

POOLJ = json.load(open(f"{ROOT}/data/candidate_sets/haiku_fable_forget_latents_v1.json"))["sets"]

def pool_structs(ctx, pool_name):
    members = np.array(sorted({int(l) for l, _c, _f in POOLJ[pool_name]["latents"]}))
    lat2pos = -np.ones(ctx["W"], np.int64); lat2pos[members] = np.arange(len(members))
    selm = lat2pos[ctx["z"]["lat_idx"]] >= 0
    mp = lat2pos[ctx["z"]["lat_idx"][selm]]
    mf = ctx["flat"][selm]
    ma = ctx["av"][selm] >= ctx["thr"][ctx["z"]["lat_idx"][selm]]
    order = np.argsort(mp, kind="stable")
    return members, mp[order], mf[order], ma[order]

def rule_pred(ctx, mk, members, mp_s, mf_s, ma_s, keep_lat):
    keepset = np.zeros(ctx["W"], bool); keepset[list(keep_lat)] = True
    kmask = keepset[members[mp_s]]
    fl = mf_s[kmask]; ab = ma_s[kmask]
    seedcnt = np.bincount(fl[ab], minlength=ctx["n_tok"]) >= 2
    pa = np.bincount(fl, minlength=ctx["n_tok"]) > 0
    prev = np.zeros(ctx["n_tok"], bool); prev[1:] = pa[:-1]
    isd = np.zeros(ctx["n_tok"], bool); isd[ctx["tok_base"][:-1]] = True
    start = pa & (~prev | isd)
    run_id = np.cumsum(start) - 1
    sr = np.zeros(max(int(start.sum()), 1), bool); sr[run_id[seedcnt]] = True
    return pa & sr[run_id], seedcnt

# --- assemble the rule sets ---
def train_stats(ctx, mk, members, mp_s, mf_s, ma_s):
    trole = mk["trole"]
    m_trd = trole == "train_dense"; m_trc = trole == "train_ctl"
    cf = np.zeros(len(members)); cc = np.zeros(len(members))
    np.add.at(cf, mp_s[m_trd[mf_s] & ma_s], 1)
    np.add.at(cc, mp_s[m_trc[mf_s] & ma_s], 1)
    rr = ((cf + 0.5) / max(int(m_trd.sum()), 1)) / ((cc + 0.5) / max(int(m_trc.sum()), 1))
    rc = cc / max(int(m_trc.sum()), 1)
    return rr, rc

RULES = []  # (name, width, pred, seeds)
def add_rule(name, width, pool, keep_fn):
    ctx, mk = CTX[width], MK[width]
    members, mp_s, mf_s, ma_s = pool_structs(ctx, pool)
    rr, rc = train_stats(ctx, mk, members, mp_s, mf_s, ma_s)
    keep = keep_fn(members, rr, rc)
    pred, seeds = rule_pred(ctx, mk, members, mp_s, mf_s, ma_s, keep)
    RULES.append(dict(name=name, width=width, pred=pred, seeds=seeds, n=len(keep),
                      keep=set(int(x) for x in keep)))
    print(f"rule {name}: {len(keep)} latents", flush=True)

add_rule("R3-16k rho=5", "16k", "haiku_v6_16k", lambda M, rr, rc: M[rr >= 5])
add_rule("R3-65k rho=5", "65k", "haiku_v6_65k", lambda M, rr, rc: M[rr >= 5])
add_rule("R3-65k rho=10", "65k", "haiku_v6_65k", lambda M, rr, rc: M[rr >= 10])
add_rule("R1-16k tau=.1%", "16k", "haiku_v6_16k", lambda M, rr, rc: M[rc <= 0.001])
d2 = json.load(open(f"{ROOT}/data/candidate_sets/fable_trim_d2.json"))["members"]
add_rule("D2 (optimized)", "16k", "fable_16k", lambda M, rr, rc: [l for l in d2 if l in set(M.tolist())])
gs = json.load(open(f"{ROOT}/out/gen_study_haiku_v6_16k.json"))
add_rule("HILLCLIMB-16k", "16k", "haiku_v6_16k",
         lambda M, rr, rc: M[:0])  # placeholder replaced below
RULES.pop()  # hillclimb members not saved in gen_study json; skip

# ---------- analyses ----------
out = {"rules": [{"name": r["name"], "width": r["width"], "n": r["n"]} for r in RULES]}

# 1. span FN by category
print("\n== span recall by annotation category (G-TEST span docs) ==")
cat_table = {}
for r in RULES:
    mk = MK[r["width"]]
    row = {}
    for c, cn in CATN.items():
        m = mk["m_span"] & (mk["span_cat"] == c)
        if int(m.sum()) < 5:
            continue
        row[cn] = {"n_tok": int(m.sum()), "recall": round(float(r["pred"][m].mean()), 3)}
    cat_table[r["name"]] = row
    print(f"  {r['name']:готов18s}" if False else f"  {r['name']:18s} " +
          " ".join(f"{cn}:{v['recall']:.2f}(n={v['n_tok']})" for cn, v in row.items()), flush=True)
out["span_recall_by_category"] = cat_table

# 2. per-doc span recall — which docs are hard for everyone
print("\n== hardest span docs (recall averaged over rules) ==")
doc_rows = []
mk16 = MK["16k"]; ctx16 = CTX["16k"]
for di, did in enumerate(ctx16["doc_ids"]):
    if role[did] != "gtest_span":
        continue
    s, e = ctx16["tok_base"][di], ctx16["tok_base"][di + 1]
    m = mk16["gt"][s:e]
    if not m.any():
        continue
    recs = {}
    for r in RULES:
        ctx, mk = CTX[r["width"]], MK[r["width"]]
        di2 = ctx["doc_ids"].index(did)
        s2, e2 = ctx["tok_base"][di2], ctx["tok_base"][di2 + 1]
        m2 = mk["gt"][s2:e2]
        recs[r["name"]] = round(float(r["pred"][s2:e2][m2].mean()), 3)
    doc_rows.append({"id": did, "config": by_id[did].get("config", ""),
                     "n_gt": int(m.sum()), "recalls": recs,
                     "mean": round(float(np.mean(list(recs.values()))), 3)})
doc_rows.sort(key=lambda r: r["mean"])
for r in doc_rows[:10]:
    print(f"  {r['mean']:.2f} {r['config']:26s} n_gt={r['n_gt']:4d} {r['id'][:60]}")
out["span_docs"] = doc_rows

# 3. c2 per-doc coverage
print("\n== hardest held-out c2 docs ==")
c2_rows = []
for di, did in enumerate(ctx16["doc_ids"]):
    if role[did] != "gtest_c2":
        continue
    recs = {}
    for r in RULES:
        ctx = CTX[r["width"]]
        di2 = ctx["doc_ids"].index(did)
        s2, e2 = ctx["tok_base"][di2], ctx["tok_base"][di2 + 1]
        recs[r["name"]] = round(float(r["pred"][s2:e2].mean()), 3)
    c2_rows.append({"id": did, "title_words": by_id[did]["text"].split()[:6],
                    "mean": round(float(np.mean(list(recs.values()))), 3), "recalls": recs})
c2_rows.sort(key=lambda r: r["mean"])
for r in c2_rows[:8]:
    print(f"  {r['mean']:.2f} {' '.join(str(w) for w in r['title_words'])[:60]}")
out["c2_docs"] = c2_rows

# 4. FP anatomy
print("\n== FP anatomy on gtest controls ==")
fp_rows = {}
STOP = set("a an the and or but of to in on at by for with from as is are was were be been i we you it its".split())
for r in RULES:
    ctx, mk = CTX[r["width"]], MK[r["width"]]
    fpm = r["pred"] & mk["m_ctl"]
    n_fp = int(fpm.sum())
    seed_share = round(float((r["seeds"] & mk["m_ctl"]).sum() / max(n_fp, 1)), 3)
    per_cfg = {}
    cfgv = np.array([by_id[i].get("config", "") for i in ctx["doc_ids"]])
    for c in sorted(set(cfgv[np.array([role[i] for i in ctx["doc_ids"]]) == "gtest_ctl"])):
        m = mk["m_ctl"] & (cfgv[mk["tok_doc"]] == c)
        per_cfg[c] = round(float(r["pred"][m].mean()), 4)
    fp_rows[r["name"]] = {"fp": round(float(r["pred"][mk["m_ctl"]].mean()), 4),
                          "seed_share_of_fp": seed_share, "per_config": per_cfg}
    print(f"  {r['name']:18s} fp={fp_rows[r['name']]['fp']:.3f} seed-share={seed_share:.2f} "
          + " ".join(f"{k.replace('ctl_','')}:{v:.3f}" for k, v in per_cfg.items()))
out["fp_anatomy"] = fp_rows

# 5. pairwise agreement on G-TEST flagged tokens (same-width pairs exact; cross-width via doc-word alignment skipped)
print("\n== pairwise Jaccard of flagged tokens (same width, gtest docs) ==")
ag = {}
gmask = {w: (MK[w]["trole"] != "train_ctl") & (MK[w]["trole"] != "train_dense") for w in ("16k", "65k")}
for i in range(len(RULES)):
    for j in range(i + 1, len(RULES)):
        a, b = RULES[i], RULES[j]
        if a["width"] != b["width"]:
            continue
        g = gmask[a["width"]]
        inter = int((a["pred"] & b["pred"] & g).sum())
        union = int(((a["pred"] | b["pred"]) & g).sum())
        ag[f"{a['name']} vs {b['name']}"] = round(inter / max(union, 1), 3)
        print(f"  {a['name']} vs {b['name']}: J={inter/max(union,1):.3f}")
out["jaccard"] = ag

json.dump(out, open(f"{ROOT}/out/quali_error_anatomy.json", "w"), indent=1)
print("\nwrote out/quali_error_anatomy.json")
