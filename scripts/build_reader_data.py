"""Data layer for the rule-comparison token reader artifact (2026-08-15/16).

For a curated doc set, compute per-token flags for each rule (both widths,
aligned by char offsets to the 16k tokenization for display), plus per-doc
top seeding features per rule.

Rules:
  A  R3-65k rho=5 seed>=2   (headline ratio rule)
  B  R3-65k rho=1 seed>=4   (new combo: loose screen, strict vote)
  C  R3-16k rho=5 seed>=2   (width comparison)
  D  R1-16k tau=.1%         (control-cap baseline)
  E  D2 optimized (16k)     (hill-climbed reference)
  F  full 65k pool          (no selection)

Docs: all G-TEST span docs; c2 (all 30); 6 dense-in; top-FP + random controls;
all clean docs; kimi differential examples (added by --with-kimi once
harvest_qkimi_* exist).

Output: out/reader_data.json
"""
import argparse
import glob
import gzip
import json
from collections import Counter

import numpy as np

ROOT = "/home/a5k/cwtice.a5k/sae-exploration"
ap = argparse.ArgumentParser()
ap.add_argument("--with-kimi", action="store_true")
args = ap.parse_args()

role = json.load(open(f"{ROOT}/data/splits/gen_split_20260815.json"))["roles"]
docs = [json.loads(l) for l in open(f"{ROOT}/data/test_docs_v5.jsonl")]
by_id = {d["id"]: d for d in docs}
POOLJ = json.load(open(f"{ROOT}/data/candidate_sets/haiku_fable_forget_latents_v1.json"))["sets"]

def load(prefix):
    zf = np.load(prefix + ".npz")
    z = {k: zf[k] for k in zf.files}
    meta = json.load(open(prefix + ".meta.json"))
    W = meta["width"]
    av = z["act"].astype(np.float64); av[np.isinf(av)] = 65504.0
    ntok = z["doc_ntok"].astype(np.int64); nd = len(meta["docs"])
    tok_base = np.zeros(nd + 1, np.int64); np.cumsum(ntok, out=tok_base[1:])
    n_tok = int(ntok.sum())
    flat = tok_base[z["doc_idx"]] + z["tok_idx"]
    mu = np.bincount(z["lat_idx"], weights=av, minlength=W) / n_tok
    sd = np.sqrt(np.maximum(np.bincount(z["lat_idx"], weights=av**2, minlength=W)/n_tok - mu**2, 0))
    return dict(z=z, meta=meta, W=W, av=av, ntok=ntok, nd=nd, tok_base=tok_base,
                n_tok=n_tok, flat=flat, thr=mu + 4 * sd,
                doc_ids=[d["id"] for d in meta["docs"]],
                offs=z["offsets"].reshape(-1, 2))

def rule_machinery(ctx, pool_name, train_roles):
    members = np.array(sorted({int(l) for l, _c, _f in POOLJ[pool_name]["latents"]}))
    lat2pos = -np.ones(ctx["W"], np.int64); lat2pos[members] = np.arange(len(members))
    selm = lat2pos[ctx["z"]["lat_idx"]] >= 0
    li = ctx["z"]["lat_idx"][selm]; fl = ctx["flat"][selm]; a = ctx["av"][selm]
    ab = a >= ctx["thr"][li]
    mp = lat2pos[li]
    m_trd, m_trc = train_roles
    cf = np.zeros(len(members)); cc = np.zeros(len(members))
    np.add.at(cf, mp[m_trd[fl] & ab], 1)
    np.add.at(cc, mp[m_trc[fl] & ab], 1)
    rr = ((cf + 0.5) / max(int(m_trd.sum()), 1)) / ((cc + 0.5) / max(int(m_trc.sum()), 1))
    rc = cc / max(int(m_trc.sum()), 1)
    return members, li, fl, ab, rr, rc

def pred_for(ctx, li, fl, ab, keep, kseed):
    kset = np.zeros(ctx["W"], bool); kset[list(keep)] = True
    km = kset[li]
    fl2, ab2, li2 = fl[km], ab[km], li[km]
    seedcnt = np.bincount(fl2[ab2], minlength=ctx["n_tok"])
    seeds = seedcnt >= kseed
    pa = np.bincount(fl2, minlength=ctx["n_tok"]) > 0
    prev = np.zeros(ctx["n_tok"], bool); prev[1:] = pa[:-1]
    isd = np.zeros(ctx["n_tok"], bool); isd[ctx["tok_base"][:-1]] = True
    start = pa & (~prev | isd)
    run_id = np.cumsum(start) - 1
    sr = np.zeros(max(int(start.sum()), 1), bool); sr[run_id[seeds]] = True
    return pa & sr[run_id], seeds, (li2, fl2, ab2)

# ---------- TEST v5 corpora ----------
C16 = load(f"{ROOT}/out/harvest_test_v5")
C65 = load(f"{ROOT}/out/harvest_test_v5_65k")
def troles(ctx):
    rl = np.array([role[i] for i in ctx["doc_ids"]])
    tok_doc = np.repeat(np.arange(ctx["nd"]), ctx["ntok"])
    tr = rl[tok_doc]
    return (tr == "train_dense"), (tr == "train_ctl"), rl
trd16, trc16, rl16 = troles(C16)
trd65, trc65, rl65 = troles(C65)

mem65, li65, fl65, ab65, rr65, _ = rule_machinery(C65, "haiku_v8_65k", (trd65, trc65))
mem16, li16, fl16, ab16, rr16, rc16 = rule_machinery(C16, "haiku_v8_16k", (trd16, trc16))
memF, liF, flF, abF, rrF, _ = rule_machinery(C16, "fable_16k", (trd16, trc16))
d2 = set(json.load(open(f"{ROOT}/data/candidate_sets/fable_trim_d2.json"))["members"])

RULES = [
    ("A", "ratio ρ≥10 · vote k=4 (65k) — champion", "65k", pred_for(C65, li65, fl65, ab65, set(mem65[rr65 >= 10].tolist()), 4)),
    ("B", "ratio ρ≥5 · vote k=4 (65k)", "65k", pred_for(C65, li65, fl65, ab65, set(mem65[rr65 >= 5].tolist()), 4)),
    ("C", "ratio ρ≥5 · vote k=2 (65k, paper vote)", "65k", pred_for(C65, li65, fl65, ab65, set(mem65[rr65 >= 5].tolist()), 2)),
    ("D", "ratio ρ≥10 · vote k=4 (16k)", "16k", pred_for(C16, li16, fl16, ab16, set(mem16[rr16 >= 10].tolist()), 4)),
    ("E", "D2 hill-climbed (16k, historical)", "16k", pred_for(C16, liF, flF, abF, d2, 2)),
    ("F", "full 65k pool (no screen)", "65k", pred_for(C65, li65, fl65, ab65, set(mem65.tolist()), 2)),
]

# captions for seed detail
caps = {}
for w, g in (("65k", f"{ROOT}/out/np65k_batch-*.jsonl.gz"), ("16k", f"{ROOT}/out/np_expl_batch-*.jsonl.gz")):
    caps[w] = {}
    for fn in glob.glob(g):
        for line in gzip.open(fn, "rt"):
            d = json.loads(line)
            idx = int(d.get("index", d.get("latent", -1)))
            if idx >= 0:
                caps[w][idx] = (d.get("description") or d.get("explanation") or "").strip()

# ---------- doc selection ----------
rng = np.random.default_rng(20260816)
sel_docs = []
for did in C16["doc_ids"]:
    r = role.get(did)
    if r == "gtest_span" and by_id[did].get("char_spans"):
        sel_docs.append((did, "span"))
    elif r == "gtest_clean":
        sel_docs.append((did, "clean"))
    elif r == "gtest_c2":
        sel_docs.append((did, "c2"))
c2_only = [d for d in sel_docs if d[1] == "c2"]
# dense-in sample
dense_ids = [i for i in C16["doc_ids"] if role.get(i) == "gtest_dense"]
sel_docs += [(i, "dense") for i in rng.choice(dense_ids, 6, replace=False)]
# controls: top-FP by rule A + random
ctl_idx = [i for i, did in enumerate(C16["doc_ids"]) if role.get(did) == "gtest_ctl"]
predA65 = RULES[0][3][0]
# map doc fp via 65k doc order (same ids)
did2i65 = {d: i for i, d in enumerate(C65["doc_ids"])}
fp_rate = []
for i in ctl_idx:
    did = C16["doc_ids"][i]
    j = did2i65[did]
    s, e = C65["tok_base"][j], C65["tok_base"][j + 1]
    fp_rate.append((float(predA65[s:e].mean()), did))
fp_rate.sort(reverse=True)
sel_docs += [(d, "ctl_topfp") for _, d in fp_rate[:10]]
sel_docs += [(d, "ctl_random") for d in rng.choice([d for _, d in fp_rate[200:]], 6, replace=False)]

texts = {did: by_id[did]["text"] for did, _ in sel_docs}
groups = {did: g for did, g in sel_docs}

# ---------- kimi docs (differential examples) ----------
kimi_flags = {}
if args.with_kimi:
    K16 = load(f"{ROOT}/out/harvest_qkimi_16k")
    K65 = load(f"{ROOT}/out/harvest_qkimi_65k")
    kd = {d["id"]: d for d in map(json.loads, open(f"{ROOT}/data/quali_kimi_chat.jsonl"))}
    # rules re-applied to kimi corpus with SAME kept sets and TEST thresholds:
    # recompute per-corpus structures using TEST-derived thresholds & kept latents
    def kimi_pred(ctxK, pool_name, ctxT, keep, kseed):
        members = np.array(sorted({int(l) for l, _c, _f in POOLJ[pool_name]["latents"]}))
        selm = np.isin(ctxK["z"]["lat_idx"], members)
        li = ctxK["z"]["lat_idx"][selm]; fl = ctxK["flat"][selm]; a = ctxK["av"][selm]
        ab = a >= ctxT["thr"][li]     # thresholds from TEST harvest (deployment rule)
        kset = np.zeros(ctxK["W"], bool); kset[list(keep)] = True
        km = kset[li]
        fl2, ab2 = fl[km], ab[km]
        seedcnt = np.bincount(fl2[ab2], minlength=ctxK["n_tok"])
        seeds = seedcnt >= kseed
        pa = np.bincount(fl2, minlength=ctxK["n_tok"]) > 0
        prev = np.zeros(ctxK["n_tok"], bool); prev[1:] = pa[:-1]
        isd = np.zeros(ctxK["n_tok"], bool); isd[ctxK["tok_base"][:-1]] = True
        start = pa & (~prev | isd)
        run_id = np.cumsum(start) - 1
        sr = np.zeros(max(int(start.sum()), 1), bool); sr[run_id[seeds]] = True
        return pa & sr[run_id]
    kp = {}
    kp["A"] = kimi_pred(K65, "haiku_v8_65k", C65, set(mem65[rr65 >= 10].tolist()), 4)
    kp["B"] = kimi_pred(K65, "haiku_v8_65k", C65, set(mem65[rr65 >= 5].tolist()), 4)
    kp["C"] = kimi_pred(K65, "haiku_v8_65k", C65, set(mem65[rr65 >= 5].tolist()), 2)
    kp["D"] = kimi_pred(K16, "haiku_v8_16k", C16, set(mem16[rr16 >= 10].tolist()), 4)
    kp["E"] = kimi_pred(K16, "fable_16k", C16, d2, 2)
    kp["F"] = kimi_pred(K65, "haiku_v8_65k", C65, set(mem65.tolist()), 2)
    # pick 6 trick-arm high-flag + 6 matched clean-arm docs (rule A doc rate)
    picks = []
    fam_arm = {}
    for i, did in enumerate(K65["doc_ids"]):
        s, e = K65["tok_base"][i], K65["tok_base"][i + 1]
        r = float(kp["A"][s:e].mean())
        d = kd[did]
        fam_arm.setdefault((d["family"], d["arm"]), []).append((r, did))
    TRICK = [("anti_hall_umwp", "unanswerable"), ("anti_hall_falseqa", "false_premise"),
             ("chat_hawthorne", "harmful_fake"), ("if_conflict", "contradictory")]
    CLEAN = [("anti_hall_umwp", "answerable"), ("anti_hall_falseqa", "true_premise"),
             ("chat_hawthorne", "harmful_real"), ("if_conflict", "satisfiable")]
    for fa in TRICK:
        rows = sorted(fam_arm.get(fa, []), reverse=True)[:2]
        picks += [(did, "kimi_trick") for _, did in rows]
    for fa in CLEAN:
        rows = sorted(fam_arm.get(fa, []), reverse=True)[:1]
        picks += [(did, "kimi_clean") for _, did in rows]
    # agentic examples: 4 highest-flag notice + 2 baseline
    try:
        A16 = load(f"{ROOT}/out/harvest_qagentic_16k")
        A65 = load(f"{ROOT}/out/harvest_qagentic_65k")
        adm = {d["id"]: d for d in map(json.loads, open(f"{ROOT}/data/quali_agentic.jsonl"))}
        kpA = {}
        kpA["A"] = kimi_pred(A65, "haiku_v8_65k", C65, set(mem65[rr65 >= 10].tolist()), 4)
        kpA["B"] = kimi_pred(A65, "haiku_v8_65k", C65, set(mem65[rr65 >= 5].tolist()), 4)
        kpA["C"] = kimi_pred(A65, "haiku_v8_65k", C65, set(mem65[rr65 >= 5].tolist()), 2)
        kpA["D"] = kimi_pred(A16, "haiku_v8_16k", C16, set(mem16[rr16 >= 10].tolist()), 4)
        kpA["E"] = kimi_pred(A16, "fable_16k", C16, d2, 2)
        kpA["F"] = kimi_pred(A65, "haiku_v8_65k", C65, set(mem65.tolist()), 2)
        arm_rows = {"notice": [], "baseline": []}
        for i, did in enumerate(A65["doc_ids"]):
            s0, e0 = A65["tok_base"][i], A65["tok_base"][i + 1]
            arm = adm[did]["arm"]
            if arm in arm_rows:
                arm_rows[arm].append((float(kpA["A"][s0:e0].mean()), did))
        apicks = [(d, "agentic_notice") for _, d in sorted(arm_rows["notice"], reverse=True)[:4]]
        apicks += [(d, "agentic_baseline") for _, d in sorted(arm_rows["baseline"], reverse=True)[:2]]
        did2a16 = {d: i for i, d in enumerate(A16["doc_ids"])}
        did2a65 = {d: i for i, d in enumerate(A65["doc_ids"])}
        for did, g in apicks:
            texts[did] = adm[did]["text"]
            groups[did] = g
            kimi_flags[did] = {}
            for code, _n, w, _p in [(r[0], r[1], r[2], None) for r in RULES]:
                ctxA = A16 if w == "16k" else A65
                i = (did2a16 if w == "16k" else did2a65)[did]
                s0, e0 = ctxA["tok_base"][i], ctxA["tok_base"][i + 1]
                kimi_flags[did][code] = (ctxA["offs"][s0:e0], kpA[code][s0:e0])
    except FileNotFoundError:
        pass
    did2k16 = {d: i for i, d in enumerate(K16["doc_ids"])}
    did2k65 = {d: i for i, d in enumerate(K65["doc_ids"])}
    for did, g in picks:
        texts[did] = kd[did]["text"]
        groups[did] = g
        kimi_flags[did] = {}
        for code, _n, w, _p in [(r[0], r[1], r[2], None) for r in RULES]:
            ctxK = K16 if w == "16k" else K65
            i = (did2k16 if w == "16k" else did2k65)[did]
            s, e = ctxK["tok_base"][i], ctxK["tok_base"][i + 1]
            o = ctxK["offs"][s:e]
            kimi_flags[did][code] = (o, kp[code][s:e])

# ---------- emit per-doc token+flag data ----------
def rle(bits):
    out, cur, run = [], bool(bits[0]) if len(bits) else False, 0
    for b in bits:
        if bool(b) == cur:
            run += 1
        else:
            out.append(run); cur = bool(b); run = 1
    out.append(run)
    return [1 if (len(bits) and bool(bits[0])) else 0] + out

docs_out = []
did2i16 = {d: i for i, d in enumerate(C16["doc_ids"])}
for did, grp in groups.items():
    txt = texts[did]
    entry = {"id": did, "group": grp, "config": by_id.get(did, {}).get("config",
             groups[did] if did not in by_id else ""), "text": txt}
    if did in kimi_flags:
        base_o = kimi_flags[did]["C"][0]
        entry["toks"] = [[int(a), int(b)] for a, b in base_o]
        fl_map = {}
        key = {tuple(x): k for k, x in enumerate(base_o)}
        for code in "ABCDEF":
            o, p = kimi_flags[did][code]
            bits = np.zeros(len(base_o), bool)
            for x, pv in zip(o, p):
                k = key.get(tuple(x))
                if k is not None and pv:
                    bits[k] = True
            fl_map[code] = rle(bits)
        entry["flags"] = fl_map
        entry["gt"] = []
        try:
            kd_meta = json.loads(next(l for l in open(f"{ROOT}/data/quali_kimi_chat.jsonl")
                                      if json.loads(l)["id"] == did))
            entry["meta"] = f"{kd_meta['family']} / {kd_meta['arm']}"
        except StopIteration:
            am = json.loads(next(l for l in open(f"{ROOT}/data/quali_agentic.jsonl")
                                 if json.loads(l)["id"] == did))
            entry["meta"] = f"agentic / {am['arm']}"
    else:
        i16 = did2i16[did]
        s, e = C16["tok_base"][i16], C16["tok_base"][i16 + 1]
        base_o = C16["offs"][s:e]
        entry["toks"] = [[int(a), int(b)] for a, b in base_o]
        key = {tuple(x): k for k, x in enumerate(base_o)}
        fl_map = {}
        for code, _n, w, (pred, seeds, _st) in RULES:
            if w == "16k":
                bits = pred[s:e]
            else:
                j = did2i65[did]
                s2, e2 = C65["tok_base"][j], C65["tok_base"][j + 1]
                bits = np.zeros(len(base_o), bool)
                for x, pv in zip(C65["offs"][s2:e2], pred[s2:e2]):
                    k = key.get(tuple(x))
                    if k is not None and pv:
                        bits[k] = True
            fl_map[code] = rle(bits)
        entry["flags"] = fl_map
        gtb = np.zeros(len(base_o), bool)
        for cs, ce in by_id[did].get("char_spans", []):
            m = (base_o[:, 0] < ce) & (base_o[:, 1] > cs)
            gtb |= m
        entry["gt"] = rle(gtb)
        entry["meta"] = by_id[did].get("config", "")
    docs_out.append(entry)

# per-rule top seed features on the selected control docs (for the info panel)
rule_meta = []
for code, name, w, (pred, seeds, (li2, fl2, ab2)) in RULES:
    ctx = C16 if w == "16k" else C65
    rlw = rl16 if w == "16k" else rl65
    tok_doc = np.repeat(np.arange(ctx["nd"]), ctx["ntok"])
    m_ctl = (rlw[tok_doc] == "gtest_ctl")
    sel = ab2 & (seeds & m_ctl)[fl2]
    cnt = Counter(int(x) for x in li2[sel])
    tops = [{"lat": l, "n": c, "cap": caps[w].get(l, "")[:70]} for l, c in cnt.most_common(8)]
    rule_meta.append({"code": code, "name": name, "width": w, "top_fp_feats": tops})

json.dump({"rules": rule_meta, "docs": docs_out},
          open(f"{ROOT}/out/reader_data.json", "w"), separators=(",", ":"))
print(f"reader data: {len(docs_out)} docs, rules {[r['code'] for r in rule_meta]}, "
      f"size {len(json.dumps({'r': rule_meta, 'd': docs_out}))//1024} KB")
