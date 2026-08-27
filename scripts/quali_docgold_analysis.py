"""H2/H3/H4: real-world doc-level validation of the rules (2026-08-16).

H2 kimi differential: per family/arm mean token-flag rate; trick arms should
   exceed matched clean arms. Also doc-level AUC trick-vs-clean per family.
H3 westover: doc flag-rate AUC separating metagame_split vs general_split.
H4 needham: AUC vs gold_eval; correlation with human_p_eval means.

Rules: A=R3-65k rho5, B=R3-65k rho1 seed4, C=R3-16k rho5, D=R1-16k tau .1%,
E=D2, F=full-65k. Thresholds + selection stats from TEST v5 (G-TRAIN),
i.e. the deployed rule applied blind to these corpora.
"""
import json
from collections import defaultdict

import numpy as np

ROOT = "/home/a5k/cwtice.a5k/sae-exploration"
role = json.load(open(f"{ROOT}/data/splits/gen_split_20260815.json"))["roles"]
tdocs = [json.loads(l) for l in open(f"{ROOT}/data/test_docs_v5.jsonl")]
by_id = {d["id"]: d for d in tdocs}
POOLJ = json.load(open(f"{ROOT}/data/candidate_sets/haiku_fable_forget_latents_v1.json"))["sets"]

def load(prefix):
    zf = np.load(prefix + ".npz")
    z = {k: zf[k] for k in zf.files}
    meta = json.load(open(prefix + ".meta.json"))
    W = meta["width"]
    av = z["act"].astype(np.float64); av[np.isinf(av)] = 65504.0
    ntok = z["doc_ntok"].astype(np.int64); nd = len(meta["docs"])
    tok_base = np.zeros(nd + 1, np.int64); np.cumsum(ntok, out=tok_base[1:])
    return dict(z=z, W=W, av=av, ntok=ntok, nd=nd, tok_base=tok_base,
                n_tok=int(ntok.sum()),
                flat=tok_base[z["doc_idx"]] + z["tok_idx"],
                doc_ids=[d["id"] for d in meta["docs"]])

def thr_and_stats(ctx):
    mu = np.bincount(ctx["z"]["lat_idx"], weights=ctx["av"], minlength=ctx["W"]) / ctx["n_tok"]
    sd = np.sqrt(np.maximum(np.bincount(ctx["z"]["lat_idx"], weights=ctx["av"]**2,
                 minlength=ctx["W"])/ctx["n_tok"] - mu**2, 0))
    return mu + 4 * sd

T16 = load(f"{ROOT}/out/harvest_test_v5"); T16["thr"] = thr_and_stats(T16)
T65 = load(f"{ROOT}/out/harvest_test_v5_65k"); T65["thr"] = thr_and_stats(T65)

def keepsets(ctx, pool):
    members = np.array(sorted({int(l) for l, _c, _f in POOLJ[pool]["latents"]}))
    rl = np.array([role[i] for i in ctx["doc_ids"]])
    tok_doc = np.repeat(np.arange(ctx["nd"]), ctx["ntok"])
    m_trd = (rl[tok_doc] == "train_dense"); m_trc = (rl[tok_doc] == "train_ctl")
    selm = np.isin(ctx["z"]["lat_idx"], members)
    li = ctx["z"]["lat_idx"][selm]; fl = ctx["flat"][selm]
    ab = ctx["av"][selm] >= ctx["thr"][li]
    mp = np.searchsorted(members, li)
    cf = np.zeros(len(members)); cc = np.zeros(len(members))
    np.add.at(cf, mp[m_trd[fl] & ab], 1)
    np.add.at(cc, mp[m_trc[fl] & ab], 1)
    rr = ((cf + 0.5) / max(int(m_trd.sum()), 1)) / ((cc + 0.5) / max(int(m_trc.sum()), 1))
    rc = cc / max(int(m_trc.sum()), 1)
    return members, rr, rc

m65, rr65, _ = keepsets(T65, "haiku_v6_65k")
m16, rr16, rc16 = keepsets(T16, "haiku_v6_16k")
d2 = set(json.load(open(f"{ROOT}/data/candidate_sets/fable_trim_d2.json"))["members"])
RULES = {
    "A": ("65k", set(m65[rr65 >= 5].tolist()), 2),
    "B": ("65k", set(m65[rr65 >= 1].tolist()), 4),
    "C": ("16k", set(m16[rr16 >= 5].tolist()), 2),
    "D": ("16k", set(m16[rc16 <= 0.001].tolist()), 2),
    "E": ("16k", d2, 2),
    "F": ("65k", set(m65.tolist()), 2),
}

def doc_rates(ctxK, thr_ctx, keep, kseed, pool):
    members = np.array(sorted({int(l) for l, _c, _f in POOLJ[pool]["latents"]})) \
        if pool else None
    selm = np.isin(ctxK["z"]["lat_idx"], list(keep))
    li = ctxK["z"]["lat_idx"][selm]; fl = ctxK["flat"][selm]
    ab = ctxK["av"][selm] >= thr_ctx["thr"][li]
    seedcnt = np.bincount(fl[ab], minlength=ctxK["n_tok"])
    seeds = seedcnt >= kseed
    pa = np.bincount(fl, minlength=ctxK["n_tok"]) > 0
    prev = np.zeros(ctxK["n_tok"], bool); prev[1:] = pa[:-1]
    isd = np.zeros(ctxK["n_tok"], bool); isd[ctxK["tok_base"][:-1]] = True
    start = pa & (~prev | isd)
    run_id = np.cumsum(start) - 1
    sr = np.zeros(max(int(start.sum()), 1), bool); sr[run_id[seeds]] = True
    pred = pa & sr[run_id]
    rates = {}
    for i, did in enumerate(ctxK["doc_ids"]):
        s, e = ctxK["tok_base"][i], ctxK["tok_base"][i + 1]
        rates[did] = float(pred[s:e].mean())
    return rates

def auc(pos, neg):
    x = np.concatenate([pos, neg])
    n1 = len(pos)
    order = x.argsort(kind="stable"); r = np.arange(1, len(x) + 1, dtype=float)
    xv = x[order]; _, first = np.unique(xv, return_index=True)
    b = np.append(first, len(x))
    for s, e in zip(b[:-1], b[1:]): r[s:e] = r[s:e].mean()
    rk = np.empty_like(r); rk[order] = r
    return float((rk[:n1].sum() - n1 * (n1 + 1) / 2) / max(n1 * (len(x) - n1), 1))

out = {}
# ---- H2 kimi ----
K = {"16k": load(f"{ROOT}/out/harvest_qkimi_16k"), "65k": load(f"{ROOT}/out/harvest_qkimi_65k")}
kd = {d["id"]: d for d in map(json.loads, open(f"{ROOT}/data/quali_kimi_chat.jsonl"))}
PAIRS = [("anti_hall_umwp", "unanswerable", "answerable"),
         ("anti_hall_falseqa", "false_premise", "true_premise"),
         ("anti_hall_squad2", "unanswerable", "answerable"),
         ("if_conflict", "contradictory", "satisfiable"),
         ("chat_hawthorne", "harmful_fake", "harmful_real"),
         ("chat_hawthorne", "benign_fake", "benign_real")]
print("== H2: kimi differential (doc mean flag rate; trick vs clean) ==")
out["kimi"] = {}
for code, (w, keep, ks) in RULES.items():
    ctxK = K[w]; thr_ctx = T16 if w == "16k" else T65
    pool = "haiku_v6_16k" if w == "16k" else "haiku_v6_65k"
    rates = doc_rates(ctxK, thr_ctx, keep, ks, pool)
    fam = defaultdict(list)
    for did, r in rates.items():
        d = kd[did]
        fam[(d["family"], d["arm"])].append(r)
    rows = {}
    for f, trick, clean in PAIRS:
        t = np.array(fam[(f, trick)]); c = np.array(fam[(f, clean)])
        if not len(t) or not len(c):
            continue
        rows[f"{f}:{trick}-vs-{clean}"] = {
            "trick_mean": round(float(t.mean()), 4), "clean_mean": round(float(c.mean()), 4),
            "auc": round(auc(t, c), 3)}
    rows["mmlu_mean"] = round(float(np.mean(fam[("mcq_mmlu", "mcq")])), 4)
    out["kimi"][code] = rows
    aucs = [v["auc"] for k, v in rows.items() if isinstance(v, dict)]
    print(f"  rule {code}: mean pairwise AUC={np.mean(aucs):.3f} | " +
          " ".join(f"{k.split(':')[0].replace('anti_hall_','')}:{v['auc']:.2f}"
                   for k, v in rows.items() if isinstance(v, dict)))

# ---- H3/H4 docgold ----
G = {"16k": load(f"{ROOT}/out/harvest_qgold_16k"), "65k": load(f"{ROOT}/out/harvest_qgold_65k")}
gd = {d["id"]: d for d in map(json.loads, open(f"{ROOT}/data/quali_docgold.jsonl"))}
print("\n== H3 westover / H4 needham (doc flag-rate AUC) ==")
out["docgold"] = {}
for code, (w, keep, ks) in RULES.items():
    ctxG = G[w]; thr_ctx = T16 if w == "16k" else T65
    pool = "haiku_v6_16k" if w == "16k" else "haiku_v6_65k"
    rates = doc_rates(ctxG, thr_ctx, keep, ks, pool)
    wv_pos = [r for did, r in rates.items() if gd[did]["config"] == "westover_metagame_split"]
    wv_neg = [r for did, r in rates.items() if gd[did]["config"] == "westover_general_split"]
    nd_pos = [r for did, r in rates.items() if gd[did]["config"] == "needham_eval_awareness"
              and gd[did]["gold_eval"]]
    nd_neg = [r for did, r in rates.items() if gd[did]["config"] == "needham_eval_awareness"
              and not gd[did]["gold_eval"]]
    # human agreement corr
    hp, fr = [], []
    for did, r in rates.items():
        g = gd[did]
        if g["config"] == "needham_eval_awareness" and g.get("hp1") is not None:
            hp.append(np.nanmean([g["hp1"], g.get("hp2", np.nan)])); fr.append(r)
    corr = float(np.corrcoef(hp, fr)[0, 1]) if len(hp) > 10 else None
    out["docgold"][code] = {
        "westover_auc": round(auc(np.array(wv_pos), np.array(wv_neg)), 3),
        "westover_means": [round(float(np.mean(wv_pos)), 4), round(float(np.mean(wv_neg)), 4)],
        "needham_auc": round(auc(np.array(nd_pos), np.array(nd_neg)), 3),
        "needham_means": [round(float(np.mean(nd_pos)), 4), round(float(np.mean(nd_neg)), 4)],
        "needham_humanp_corr": None if corr is None else round(corr, 3)}
    o = out["docgold"][code]
    print(f"  rule {code}: westover AUC={o['westover_auc']} ({o['westover_means'][0]:.3f} vs "
          f"{o['westover_means'][1]:.3f}) | needham AUC={o['needham_auc']} "
          f"({o['needham_means'][0]:.3f} vs {o['needham_means'][1]:.3f}) | r(human_p)={o['needham_humanp_corr']}")

json.dump(out, open(f"{ROOT}/out/quali_docgold_results.json", "w"), indent=1)
print("wrote out/quali_docgold_results.json")
