"""Corpus view for the TEST-v4 firing-map artifact: P2 + D2 members over the
current TEST split (manifest v4, protocol v5), human spans overlaid.

Membership toggle groups (one tag per latent, template AND semantics):
  both = in P2 and D2   ·   d2 = D2 only   ·   p2 = P2 only
So: D2 rule = check {both, d2}; P2 rule = check {both, p2}; all = check all.

Docs: all annotated (spans/clean) + all 150 dense + a seeded 25-per-config
sample of the 625 assumed-clean retain controls (embedding all 625 would
blow the artifact size budget). Thresholds (mean+4SD) are computed over the
FULL 820-doc harvest, matching score_split.py exactly.

Usage: python build_test_view_v4.py
Output: out/corpus_view_test_v4.json (consumed by build_test_artifact.py)
"""
import glob
import gzip
import json

import numpy as np

ROOT = "/home/a5k/cwtice.a5k/sae-exploration"
NP_MODEL, NP_SOURCE = "gemma-3-27b-it", "40-gemmascope-2-res-16k"
SEED = 20260815
CTL_PER_CONFIG = 25

p2 = set(json.load(open(f"{ROOT}/data/candidate_sets/fable_trim_p2.json"))["members"])
d2 = set(json.load(open(f"{ROOT}/data/candidate_sets/fable_trim_d2.json"))["members"])
MEMBERS = sorted(p2 | d2)
grp_of = {l: ("both" if l in p2 and l in d2 else "d2" if l in d2 else "p2") for l in MEMBERS}

lines = open(f"{ROOT}/data/test_docs_v4.jsonl").read().splitlines()
docs_meta, line_of = {}, {}
for ln, line in enumerate(lines, 1):
    d = json.loads(line)
    docs_meta[d["id"]] = d
    line_of[d["id"]] = ln

AXES = [
    {"key": "control", "name": "Controls (annotated)", "short": "control", "cssv": "--ax-co",
     "desc": "reviewed control docs with human labels; firing outside spans is FP"},
    {"key": "chat", "name": "Chat + agentic (annotated)", "short": "chat", "cssv": "--ax-dm",
     "desc": "kimi chat/agentic rollouts with human metagaming spans — the weak register"},
    {"key": "prose", "name": "Prose (annotated)", "short": "prose", "cssv": "--ax-dg",
     "desc": "probe-prose docs with human spans"},
    {"key": "dense", "name": "Dense c1–c5", "short": "dense", "cssv": "--ax-mc",
     "desc": "wall-to-wall in-category docs; c1–c4 are main-forget (protocol v5), c5 is retain-scope"},
    {"key": "retain", "name": "Retain controls (sample)", "short": "retain", "cssv": "--ax-co",
     "desc": f"seeded {CTL_PER_CONFIG}/config sample of the 625 assumed-clean docs; any flag is FP"},
]
def axis_of(dm):
    r = dm["register"]
    if r in ("chat", "agentic"): return "chat"
    if r in ("ctl_retain", "ctl_swe"): return "retain"
    return r

z = np.load(f"{ROOT}/out/harvest_test_v3.npz")
meta = json.load(open(f"{ROOT}/out/harvest_test_v3.meta.json"))
W = meta["width"]
av = z["act"].astype(np.float64); av[np.isinf(av)] = 65504.0
ntok = z["doc_ntok"].astype(np.int64)
nd = len(meta["docs"])
tok_base = np.zeros(nd + 1, dtype=np.int64); np.cumsum(ntok, out=tok_base[1:])
offs = z["offsets"].reshape(-1, 2)
ids = [d["id"] for d in meta["docs"]]

# doc subset: annotated + dense + seeded control sample
rng = np.random.default_rng(SEED)
keep_idx = []
by_cfg = {}
for i, did in enumerate(ids):
    dm = docs_meta[did]
    if dm["label_type"] in ("spans", "clean", "dense_forget"):
        keep_idx.append(i)
    else:
        by_cfg.setdefault(dm.get("config", ""), []).append(i)
for c, idxs in sorted(by_cfg.items()):
    pick = rng.choice(idxs, min(CTL_PER_CONFIG, len(idxs)), replace=False)
    keep_idx.extend(int(x) for x in sorted(pick))
keep_idx = sorted(keep_idx)
print(f"docs kept: {len(keep_idx)} of {nd}")

mean_dl = np.zeros((nd, W)); np.add.at(mean_dl, (z["doc_idx"], z["lat_idx"]), av); mean_dl /= ntok[:, None]
max_dl = np.zeros((nd, W)); np.maximum.at(max_dl, (z["doc_idx"], z["lat_idx"]), av)

lt_all = np.array([docs_meta[i]["label_type"] for i in ids])
posm, ctlm = lt_all == "spans", lt_all == "clean"

def auc_l(l):
    x = np.concatenate([mean_dl[posm, l], mean_dl[ctlm, l]])
    n1 = int(posm.sum())
    order = x.argsort(kind="stable"); r = np.arange(1, len(x) + 1, dtype=float)
    xv = x[order]; _, first = np.unique(xv, return_index=True)
    b = np.append(first, len(x))
    for s, e in zip(b[:-1], b[1:]): r[s:e] = r[s:e].mean()
    rk = np.empty_like(r); rk[order] = r
    return float((rk[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(x) - n1)))

descs = {}
for fn in glob.glob(f"{ROOT}/out/np_expl_batch-*.jsonl.gz"):
    for line in gzip.open(fn, "rt"):
        d = json.loads(line)
        if d.get("layer") == NP_SOURCE:
            descs[int(d["index"])] = d["description"]

axis_by_doc = np.array([axis_of(docs_meta[i]) for i in ids])
latents = {}
for l in MEMBERS:
    latents[str(l)] = {
        "auc": round(auc_l(l), 4), "dec_spec": None,
        "docs_firing": int((max_dl[keep_idx, l] > 0).sum()),
        "mean_pos": round(float(mean_dl[posm, l].mean()), 2),
        "mean_ctl": round(float(mean_dl[ctlm, l].mean()), 2),
        "axis_means": {a["key"]: round(float(mean_dl[axis_by_doc == a["key"], l].mean()), 2) for a in AXES},
        "fires_on_generic_deception": False,
        "in_sep_shortlist": l in d2, "in_dec_shortlist": grp_of[l] == "both",
        "description": descs.get(l), "explainer": None,
        "neuronpedia": f"https://www.neuronpedia.org/{NP_MODEL}/{NP_SOURCE}/{l}",
        "firing_rate": None, "raw_examples": [],
        "v2_class": grp_of[l],
    }

docs_out, tv_docs, texts = [], [], {}
for out_i, i in enumerate(keep_idx):
    did = ids[i]
    dm = docs_meta[did]
    texts[did] = {"text": dm["text"], "context": None}
    fires = []
    for l in MEMBERS:
        if max_dl[i][l] <= 0: continue
        fires.append([int(l), round(float(mean_dl[i][l]), 2), round(float(max_dl[i][l]), 1), 0, 0])
    fires.sort(key=lambda x: -x[2])
    docs_out.append({
        "i": out_i, "id": did, "line": line_of[did], "source": dm.get("config", ""), "model": None,
        "env": dm["label_type"],
        "category": f"{dm.get('config','')} · {dm['label_type']}"
                    + (f" · {len(dm['char_spans'])} spans" if dm.get("char_spans") else ""),
        "axis": axis_of(dm),
        "label": "positive" if (dm.get("char_spans") or dm["label_type"] == "dense_forget") else "control",
        "n_tokens": int(ntok[i]), "n_chars": len(dm["text"]), "has_context": False,
        "fires": fires[:45], "n_latents_firing": int((max_dl[i][MEMBERS] > 0).sum()),
        "prov_items": [f"TEST-v4 doc (config {dm.get('config','')}, label_type {dm['label_type']}) — "
                       "camgeodesic/metagaming-labeling @ rev f7ae7700",
                       f"~/sae-exploration/data/test_docs_v4.jsonl · line {line_of[did]}"],
        "prov_cmd": f"sed -n '{line_of[did]}p' ~/sae-exploration/data/test_docs_v4.jsonl | jq -r '.text'",
    })
    sel = (z["doc_idx"] == i) & np.isin(z["lat_idx"], MEMBERS)
    acts = {}
    li_d, ti_d, av_d = z["lat_idx"][sel], z["tok_idx"][sel], av[sel]
    for l in set(li_d.tolist()):
        m = li_d == l
        acts[str(l)] = [[int(t), round(float(a), 1)] for t, a in zip(ti_d[m], av_d[m])]
    tv_docs.append({"id": did, "toks": [[int(s), int(e)] for s, e in offs[tok_base[i]:tok_base[i + 1]]],
                    "acts": acts, "gt": dm.get("char_spans", [])})

# thresholds over the FULL harvest — identical to score_split.py
n_tok_total = int(ntok.sum())
mu = np.bincount(z["lat_idx"], weights=av, minlength=W) / n_tok_total
sdv = np.sqrt(np.maximum(np.bincount(z["lat_idx"], weights=av**2, minlength=W)/n_tok_total - mu**2, 0))
fire_cnt = np.bincount(z["lat_idx"], minlength=W)
CATS = {"both": "shared by P2 + D2 (85)",
        "d2": "D2 only — dense+span F2-max, current headline (102)",
        "p2": "P2 only — span-only F2-max, superseded (49)"}
tv = {"latents": MEMBERS, "docs": tv_docs,
      "np_prefix": f"https://www.neuronpedia.org/{NP_MODEL}/{NP_SOURCE}/",
      "descs": {str(l): descs.get(l) for l in MEMBERS},
      "taglist": {"categories": CATS,
                  "tags": {str(l): [[grp_of[l], "core"]] for l in MEMBERS}},
      "stats": {str(l): {"mean": round(float(mu[l]), 4), "sd": round(float(sdv[l]), 4),
                          "fire_rate": round(float(fire_cnt[l]) / n_tok_total, 5),
                          "ctl_mean": round(float(mu[l]), 4), "ctl_sd": round(float(sdv[l]), 4),
                          "ctl_fire_rate": round(float(fire_cnt[l]) / n_tok_total, 5), "max": 0} for l in MEMBERS},
      "stats_note": ("TEST v4 (manifest sha 8fa54624) · protocol v5. Members = D2 ∪ P2 (236). "
                     "Check {both,d2} for the D2 rule (main forget = spans + dense c1–c4; "
                     "F2_main .882), {both,p2} for the span-selected P2 rule (span F2 .745). "
                     "Thresholds mean+4SD over the full 820-doc TEST harvest — identical to the "
                     "official scorer. Underlined = human span. Retain axis is a seeded 25/config "
                     "sample of the 625 assumed-clean controls."),
      "rule_note": "Rathi & Radford 2601.21571 §5.1"}
out = {"sae": "test-v4 · D2 ∪ P2", "model": "unsloth/gemma-3-27b-it", "layer": 40, "n_layers": 62,
       "width": W, "neuronpedia_model": NP_MODEL, "neuronpedia_source": NP_SOURCE,
       "axis_counts": {a["key"]: sum(1 for d in docs_out if d["axis"] == a["key"]) for a in AXES},
       "axes_def": AXES, "docs": docs_out, "latents": latents, "tokenView": tv, "texts": texts}
json.dump(out, open(f"{ROOT}/out/corpus_view_test_v4.json", "w"), separators=(",", ":"))
n_acts = sum(len(v) for d in tv_docs for v in d["acts"].values())
print(f"v4 view: {len(docs_out)} docs, {len(MEMBERS)} members, "
      f"{sum(len(d['gt']) for d in tv_docs)} human spans, {n_acts} act rows")
