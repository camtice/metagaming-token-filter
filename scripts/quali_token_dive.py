"""Q1b: token-level deep dive — what characterizes missed span tokens and FP seeds.

For the headline rule (R3-65k rho=5, haiku_v6) on G-TEST:
  1. Missed vs caught span tokens: surface class (word/number/punct/stop),
     position in span (edge vs interior), doc-config profile
  2. umwp_unanswerable case study: which spans missed entirely; do ANY pool
     features fire there (coverage gap vs threshold gap)?
  3. FP seed census: top seeding features on gtest controls with captions +
     surface forms (single tokens only in transcript)
  4. c2 hardest-doc census: which features DO fire on the hard c2 docs
Prints aggregates + single-token surfaces only.
"""
import glob
import gzip
import json
import re
from collections import Counter

import numpy as np

ROOT = "/home/a5k/cwtice.a5k/sae-exploration"
role = json.load(open(f"{ROOT}/data/splits/gen_split_20260815.json"))["roles"]
docs = [json.loads(l) for l in open(f"{ROOT}/data/test_docs_v5.jsonl")]
by_id = {d["id"]: d for d in docs}
POOLJ = json.load(open(f"{ROOT}/data/candidate_sets/haiku_fable_forget_latents_v1.json"))["sets"]

HARV = f"{ROOT}/out/harvest_test_v5_65k"
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
m_span = (trole == "gtest_span") & gt
m_ctl = trole == "gtest_ctl"
m_c2 = trole == "gtest_c2"
is_doc_start = np.zeros(n_tok, bool); is_doc_start[tok_base[:-1]] = True

members = np.array(sorted({int(l) for l, _c, _f in POOLJ["haiku_v6_65k"]["latents"]}))
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
keep = set(members[rr >= 5].tolist())
kmask = np.isin(li_all, list(keep))
li, fl, a, above = li_all[kmask], mf[kmask], a_all[kmask], ab[kmask]
seedcnt = np.bincount(fl[above], minlength=n_tok)
seeds = seedcnt >= 2
pa = np.bincount(fl, minlength=n_tok) > 0
prev = np.zeros(n_tok, bool); prev[1:] = pa[:-1]
start = pa & (~prev | is_doc_start)
run_id = np.cumsum(start) - 1
sr = np.zeros(max(int(start.sum()), 1), bool); sr[run_id[seeds]] = True
pred = pa & sr[run_id]
print(f"rule: R3-65k rho=5 (haiku_v6), {len(keep)} latents; span recall "
      f"{float(pred[m_span].mean()):.3f}")

# any-pool-feature firing (coverage vs threshold gap)
pa_pool = np.bincount(mf, minlength=n_tok) > 0            # any of 3043 pool latents
ab_pool = np.bincount(mf[ab], minlength=n_tok) > 0        # any pool latent above thr

surface = np.empty(n_tok, object)
for di, did in enumerate(doc_ids):
    if rl[di] not in ("gtest_span", "gtest_ctl", "gtest_c2"):
        continue
    txt = by_id[did]["text"]
    o = offs[tok_base[di]:tok_base[di + 1]]
    for j in range(len(o)):
        surface[tok_base[di] + j] = txt[o[j, 0]:o[j, 1]].strip().lower()
STOP = set("a an the and or but of to in on at by for with from as is are was were be been i we you it its this that".split())
def sclass(s):
    if not s: return "ws"
    if re.fullmatch(r"[\d.,%$()\-+=/*:]+", s): return "num/punct"
    if s in STOP: return "stop"
    return "word"

# 1. missed vs caught span tokens
print("\n== span tokens: missed vs caught (surface class) ==")
for name, msk in (("caught", m_span & pred), ("missed", m_span & ~pred)):
    cls = Counter(sclass(surface[t]) for t in np.flatnonzero(msk))
    tot = sum(cls.values())
    print(f"  {name} (n={tot}): " + " ".join(f"{k}:{v/tot:.2f}" for k, v in cls.most_common()))
missed = m_span & ~pred
print("  missed with NO pool feature above thr:", round(float((~ab_pool)[missed].mean()), 3),
      "| no pool feature firing at all:", round(float((~pa_pool)[missed].mean()), 3))
print("  top missed surfaces:", [w for w, _ in Counter(
    surface[t] for t in np.flatnonzero(missed) if surface[t]).most_common(15)])

# 2. umwp case study
print("\n== umwp_unanswerable case study ==")
for di, did in enumerate(doc_ids):
    if by_id[did].get("config") != "umwp_unanswerable" or role[did] != "gtest_span":
        continue
    s, e = tok_base[di], tok_base[di + 1]
    g = gt[s:e]
    if not g.any():
        continue
    r = float(pred[s:e][g].mean())
    nofeat = float((~ab_pool[s:e])[g].mean())
    print(f"  {did[:44]:46s} n_gt={int(g.sum()):4d} recall={r:.2f} "
          f"gt-tokens w/o any above-thr pool feature: {nofeat:.2f}")

# 3. FP seed census with captions
print("\n== top FP-seeding features on gtest controls ==")
caps = {}
for fn in glob.glob(f"{ROOT}/out/np65k_batch-*.jsonl.gz"):
    for line in gzip.open(fn, "rt"):
        d = json.loads(line)
        idx = int(d.get("index", d.get("latent", -1)))
        if idx >= 0:
            caps[idx] = (d.get("description") or d.get("explanation") or "").strip()
fp_seed_tok = seeds & m_ctl
contrib = np.zeros(W)
sel_fp = above & fp_seed_tok[fl]
np.add.at(contrib, li[sel_fp], 1)
top = np.argsort(-contrib)[:12]
for t in top:
    if contrib[t] == 0:
        break
    rows = (li == t) & above & m_ctl[fl]
    surf = Counter(surface[x] for x in fl[rows] if surface[x])
    print(f"  {int(t):6d} n={int(contrib[t]):5d} rr={rr[lat2pos[t]]:6.1f} «{caps.get(int(t),'')[:52]}» "
          f"tokens: {[w for w,_ in surf.most_common(5)]}")

# 4. who fires on the hard c2 docs
print("\n== features carrying the HARD c2 docs (bottom-3 by coverage) ==")
c2rows = []
for di, did in enumerate(doc_ids):
    if role[did] != "gtest_c2":
        continue
    s, e = tok_base[di], tok_base[di + 1]
    c2rows.append((float(pred[s:e].mean()), di, did))
c2rows.sort()
for cov, di, did in c2rows[:3]:
    s, e = tok_base[di], tok_base[di + 1]
    m = np.zeros(n_tok, bool); m[s:e] = True
    rows = above & m[fl]
    cnt = Counter(int(x) for x in li[rows])
    print(f"  cov={cov:.2f} {' '.join(by_id[did]['text'].split()[:7])[:50]}")
    for lat, c in cnt.most_common(4):
        print(f"      {lat:6d} n={c:3d} «{caps.get(lat,'')[:48]}»")
json.dump({"note": "see transcript; aggregates in quali_error_anatomy.json"},
          open(f"{ROOT}/out/quali_token_dive.done", "w"))
print("\nDONE")
