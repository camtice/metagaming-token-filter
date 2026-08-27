"""Q2: what does 65k add? Cross-width token-level comparison (2026-08-15).

Same docs exist in both TEST v5 harvests with identical whitespace offsets, so
tokens align exactly. Compare R3 train-stats rho=5 at 16k (haiku_v6_16k) vs
65k (haiku_v6_65k) on G-TEST:
  - span tokens caught by 65k but missed by 16k (and vice versa)
  - census of which 65k features seed the 65k-only catches -> caption themes
  - same for c2 docs
  - FP overlap: is 65k's extra FP on the same tokens or new ones?
"""
import glob
import gzip
import json
from collections import Counter

import numpy as np

ROOT = "/home/a5k/cwtice.a5k/sae-exploration"
role = json.load(open(f"{ROOT}/data/splits/gen_split_20260815.json"))["roles"]
docs = [json.loads(l) for l in open(f"{ROOT}/data/test_docs_v5.jsonl")]
by_id = {d["id"]: d for d in docs}
POOLJ = json.load(open(f"{ROOT}/data/candidate_sets/haiku_fable_forget_latents_v1.json"))["sets"]

def build(width, pool_name, rho):
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
    members = np.array(sorted({int(l) for l, _c, _f in POOLJ[pool_name]["latents"]}))
    lat2pos = -np.ones(W, np.int64); lat2pos[members] = np.arange(len(members))
    selm = lat2pos[z["lat_idx"]] >= 0
    li_all = z["lat_idx"][selm]; mf = flat[selm]; a_all = av[selm]
    mp = lat2pos[li_all]
    ab = a_all >= thr[li_all]
    m_trd = trole == "train_dense"; m_trc = trole == "train_ctl"
    cf = np.zeros(len(members)); cc = np.zeros(len(members))
    np.add.at(cf, mp[m_trd[mf] & ab], 1)
    np.add.at(cc, mp[m_trc[mf] & ab], 1)
    rr = ((cf + 0.5) / max(int(m_trd.sum()), 1)) / ((cc + 0.5) / max(int(m_trc.sum()), 1))
    keep = set(members[rr >= rho].tolist())
    kmask = np.isin(li_all, list(keep))
    li, fl, above = li_all[kmask], mf[kmask], ab[kmask]
    seedcnt = np.bincount(fl[above], minlength=n_tok)
    seeds = seedcnt >= 2
    pa = np.bincount(fl, minlength=n_tok) > 0
    prev = np.zeros(n_tok, bool); prev[1:] = pa[:-1]
    start = pa & (~prev | is_doc_start)
    run_id = np.cumsum(start) - 1
    sr = np.zeros(max(int(start.sum()), 1), bool); sr[run_id[seeds]] = True
    pred = pa & sr[run_id]
    # doc-word keyed flag map for cross-width alignment
    flags = {}
    for di, did in enumerate(doc_ids):
        if rl[di] in ("train_ctl", "train_dense", "excluded_c5"):
            continue
        s, e = tok_base[di], tok_base[di + 1]
        flags[did] = (offs[s:e].copy(), pred[s:e].copy(), gt[s:e].copy(), rl[di])
    return dict(flags=flags, li=li, fl=fl, above=above, seeds=seeds, n_tok=n_tok,
                doc_ids=doc_ids, tok_base=tok_base, keep=keep, W=W)

A = build("16k", "haiku_v6_16k", 5)
B = build("65k", "haiku_v6_65k", 5)
print(f"16k kept {len(A['keep'])}, 65k kept {len(B['keep'])}")

# align by (doc, char offset)
only65_span, only16_span, both_span, only65_c2 = [], [], [], []
fp_only65 = fp_only16 = fp_both = 0
for did, (oA, pA, gA, r) in A["flags"].items():
    if did not in B["flags"]:
        continue
    oB, pB, gB, _ = B["flags"][did]
    key = {tuple(x): i for i, x in enumerate(oB)}
    for i, x in enumerate(oA):
        j = key.get(tuple(x))
        if j is None:
            continue
        if r == "gtest_span" and gA[i]:
            if pB[j] and not pA[i]: only65_span.append((did, tuple(x)))
            elif pA[i] and not pB[j]: only16_span.append((did, tuple(x)))
            elif pA[i]: both_span.append((did, tuple(x)))
        elif r == "gtest_c2" and pB[j] and not pA[i]:
            only65_c2.append((did, tuple(x)))
        elif r == "gtest_ctl":
            if pA[i] and pB[j]: fp_both += 1
            elif pA[i]: fp_only16 += 1
            elif pB[j]: fp_only65 += 1
ns = len(only65_span) + len(only16_span) + len(both_span)
print(f"\nspan gt tokens: both={len(both_span)} only65={len(only65_span)} only16={len(only16_span)}")
print(f"ctl FP tokens: both={fp_both} only16={fp_only16} only65={fp_only65} "
      f"(FP Jaccard {fp_both/max(fp_both+fp_only16+fp_only65,1):.2f})")

# census: which 65k features seed the only65 span catches
caps = {}
for fn in glob.glob(f"{ROOT}/out/np65k_batch-*.jsonl.gz"):
    for line in gzip.open(fn, "rt"):
        d = json.loads(line)
        idx = int(d.get("index", d.get("latent", -1)))
        if idx >= 0:
            caps[idx] = (d.get("description") or d.get("explanation") or "").strip()
tokset = {}
for did, off in only65_span:
    tokset.setdefault(did, set()).add(off)
contrib = Counter()
tb = {did: i for i, did in enumerate(B["doc_ids"])}
HARV65 = f"{ROOT}/out/harvest_test_v5_65k"
meta65 = json.load(open(HARV65 + ".meta.json"))
offs_all = np.load(HARV65 + ".npz")["offsets"].reshape(-1, 2)
# map (doc, off) -> flat idx
flatmap = {}
for did in tokset:
    di = tb[did]
    s, e = B["tok_base"][di], B["tok_base"][di + 1]
    for t in range(s, e):
        o = tuple(offs_all[t])
        if o in tokset[did]:
            flatmap[t] = True
sel = B["above"] & np.array([flatmap.get(int(x), False) for x in B["fl"]])
for lat in B["li"][sel]:
    contrib[int(lat)] += 1
print("\n== 65k features seeding span tokens that 16k misses ==")
for lat, c in contrib.most_common(15):
    print(f"  {lat:6d} n={c:4d} «{caps.get(lat,'')[:60]}»")
json.dump({"both": len(both_span), "only65": len(only65_span), "only16": len(only16_span),
           "fp_both": fp_both, "fp_only16": fp_only16, "fp_only65": fp_only65,
           "only65_top_feats": contrib.most_common(30)},
          open(f"{ROOT}/out/quali_width_census.json", "w"))
print("wrote out/quali_width_census.json")
