"""Precision post-mortem for fable_trim sets (2026-08-15).

Empirics behind the qualitative finding that low precision is partly a
label-granularity artifact plus register/filler features:

  1. Content-only rescoring: drop stopword/punct/digit tokens from BOTH gt
     and pred; recompute P/R/F2 for P1/P2/P3 + OG.
  2. FP anatomy on annotated docs: seed vs window share; distance to the
     nearest gt span (boundary spill at k<=2, k<=5).
  3. Surface-token profiles for suspect features (top above-threshold
     surface forms + stopword fraction of firings). Single tokens only.
  4. Ablation: P3 minus the register/formatting cluster; drop-one for the
     top control-FP features.
"""
import json
import re
from collections import Counter

import numpy as np

ROOT = "/home/a5k/cwtice.a5k/sae-exploration"
HARV = f"{ROOT}/out/harvest_test_v3"
DOCS = f"{ROOT}/data/test_docs_v4.jsonl"

STOP = set("""a an the and or but if then else of to in on at by for with from as is are was
were be been being am do does did have has had will would can could should may might must
that this these those it its they them their he she his her you your i we our my me us not
no nor so than too very just only also there here when where which who whom what how why
all any both each few more most other some such own same s t don now""".split())
WORD = re.compile(r"[A-Za-z]")

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
ctlm = lt[tok_doc] == "assumed_clean"
is_doc_start = np.zeros(n_tok, bool); is_doc_start[tok_base[:-1]] = True
gt_ann = gt & annm

# surface form per token (lowercased); content-token mask
surface = np.empty(n_tok, object)
for di, did in enumerate(doc_ids):
    txt = by_id[did]["text"]
    o = offs[tok_base[di]:tok_base[di + 1]]
    for j in range(len(o)):
        surface[tok_base[di] + j] = txt[o[j, 0]:o[j, 1]].strip().lower()
content = np.array([bool(s) and bool(WORD.search(s)) and s not in STOP
                    for s in surface], bool)

def set_state(mem):
    memv = np.zeros(W, bool); memv[list(mem)] = True
    sel = memv[z["lat_idx"]]
    li, fl, a = z["lat_idx"][sel], flat[sel], av[sel]
    above = a >= thr[li]
    seedcnt = np.bincount(fl[above], minlength=n_tok)
    pa = np.bincount(fl, minlength=n_tok) > 0
    seeds = seedcnt >= 2
    prev = np.zeros(n_tok, bool); prev[1:] = pa[:-1]
    start = pa & (~prev | is_doc_start)
    run_id = np.cumsum(start) - 1
    sr = np.zeros(max(int(start.sum()), 1), bool); sr[run_id[seeds]] = True
    pred = pa & sr[run_id]
    return pred, seeds, (li, fl, above)

def prf(pred, gmask, amask):
    tp = int((pred & gmask & amask).sum()); fp = int((pred & ~gmask & amask).sum())
    fn = int((~pred & gmask & amask).sum())
    P = tp / max(tp + fp, 1); R = tp / max(tp + fn, 1)
    return {"P": round(P, 3), "R": round(R, 3),
            "F2": round(5*P*R/max(4*P+R, 1e-9), 3), "F1": round(2*P*R/max(P+R, 1e-9), 3)}

sets = {}
for s in ("p1", "p2", "p3"):
    sets[s.upper()] = json.load(open(f"{ROOT}/data/candidate_sets/fable_trim_{s}.json"))["members"]
sets["OG"] = sorted(int(k) for k in json.load(open(f"{ROOT}/data/latent_list_2026-07-29.json"))["tags"])

out = {}
print("== 1. all-token vs content-token-only scoring (annotated docs) ==")
print(f"   gt tokens {int(gt_ann.sum())}, content share of gt "
      f"{float(content[gt_ann].mean()):.3f}; base rate all {float(gt_ann.sum())/int(annm.sum()):.3f}, "
      f"content-only {float((gt_ann&content).sum())/max(int((annm&content).sum()),1):.3f}")
for name, mem in sets.items():
    pred, seeds, _ = set_state(mem)
    allm = prf(pred, gt, annm)
    cont = prf(pred & content, gt & content, annm & content)
    out[name] = {"all": allm, "content_only": cont}
    print(f"  {name}: all P={allm['P']} R={allm['R']} F2={allm['F2']} | "
          f"content-only P={cont['P']} R={cont['R']} F2={cont['F2']}")

print("\n== 2. FP anatomy (P3, annotated docs) ==")
pred, seeds, (li3, fl3, ab3) = set_state(sets["P3"])
fpm = pred & ~gt & annm
n_fp = int(fpm.sum())
# distance to nearest gt token within each doc
dist = np.full(n_tok, 10**9, np.int64)
for di in range(nd):
    s, e = tok_base[di], tok_base[di + 1]
    g = np.flatnonzero(gt[s:e])
    if len(g) == 0:
        continue
    idx = np.arange(e - s)
    pos = np.searchsorted(g, idx)
    left = np.where(pos > 0, idx - g[np.maximum(pos - 1, 0)], 10**9)
    right = np.where(pos < len(g), g[np.minimum(pos, len(g) - 1)] - idx, 10**9)
    dist[s:e] = np.minimum(left, right)
anat = {"n_fp": n_fp,
        "windowed_share": round(float((fpm & ~seeds).sum() / max(n_fp, 1)), 3),
        "within2_of_span": round(float((fpm & (dist <= 2)).sum() / max(n_fp, 1)), 3),
        "within5_of_span": round(float((fpm & (dist <= 5)).sum() / max(n_fp, 1)), 3),
        "stopword_share_fp": round(float((~content[fpm]).mean()) if n_fp else 0.0, 3),
        "stopword_share_tp": round(float((~content[pred & gt & annm]).mean()), 3)}
out["fp_anatomy_p3"] = anat
print("  ", anat)

print("\n== 3. suspect-feature surface profiles (above-threshold firings, annotated+ctl docs) ==")
suspects = [1325, 2416, 1117, 3172, 4446, 4898, 1829, 9839, 3507, 2606, 2399, 64, 600,
            817, 1365, 1419, 1491]
prof = {}
scope = annm | ctlm
for f in suspects:
    if f not in sets["P3"] and f not in sets["P1"]:
        continue
    rows = (li3 == f) & ab3
    toks = fl3[rows]; toks = toks[scope[toks]]
    surf = Counter(surface[t] for t in toks)
    stopfrac = float(np.mean([not content[t] for t in toks])) if len(toks) else 0.0
    prof[f] = {"n_fire": int(len(toks)), "stop_frac": round(stopfrac, 3),
               "top": surf.most_common(8)}
    print(f"  {f:6d} stop_frac={stopfrac:.2f} n={len(toks):6d} top: "
          + ", ".join(f"'{w}'x{c}" for w, c in surf.most_common(8)))
out["profiles"] = prof

print("\n== 4. ablations on P3 ==")
filler = [f for f in (1325, 2416, 1117, 3172, 4446, 4898, 1829) if f in sets["P3"]]
pred_f, _, _ = set_state([m for m in sets["P3"] if m not in filler])
ab = {"removed": filler,
      "metrics": prf(pred_f, gt, annm),
      "fp_ctl": round(float(pred_f[ctlm].mean()), 4)}
base_fp_ctl = round(float(pred[ctlm].mean()), 4)
out["ablation_filler"] = ab
print(f"  P3 minus filler cluster {filler}: {ab['metrics']} fp_ctl {ab['fp_ctl']} "
      f"(P3 baseline fp_ctl {base_fp_ctl})")
drops = {}
for f in (64, 2399, 2606, 2615, 9839, 3507):
    if f not in sets["P3"]:
        continue
    pd, _, _ = set_state([m for m in sets["P3"] if m != f])
    drops[f] = {**prf(pd, gt, annm), "fp_ctl": round(float(pd[ctlm].mean()), 4)}
    print(f"  P3 minus {f}: {drops[f]}")
out["drop_one"] = drops

json.dump(out, open(f"{ROOT}/out/precision_analysis_2026-08-15.json", "w"), indent=1)
print(f"\nwrote {ROOT}/out/precision_analysis_2026-08-15.json")
