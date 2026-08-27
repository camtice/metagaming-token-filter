"""Corpus view for the TEST-v1 challenger artifact: fable_16k loaded over the
test split, human spans overlaid, confidence tiers + categories as toggle
groups. LEAN: no raw-example extraction, no per-doc top-K universe — the
firing tables and token zoom cover the candidate set's members only.

Usage: python build_fable_view.py
Output: out/corpus_view_test_v1.json (consumed by build_test_artifact.py)
"""
import glob
import gzip
import json

import numpy as np

ROOT = "/home/a5k/cwtice.a5k/sae-exploration"
NP_MODEL, NP_SOURCE = "gemma-3-27b-it", "40-gemmascope-2-res-16k"
SET_NAME = "fable_16k"

cand = json.load(open(f"{ROOT}/data/candidate_sets/haiku_fable_forget_latents_v1.json"))
entries = cand["sets"][SET_NAME]["latents"]
MEMBERS = sorted({int(l) for l, c, conf in entries})
conf_of = {int(l): conf for l, c, conf in entries}
cat_of = {int(l): c for l, c, conf in entries}
docs_meta = {d["id"]: d for d in map(json.loads, open(f"{ROOT}/data/test_docs_v1.jsonl"))}

AXES = [
    {"key": "control", "name": "Controls (annotated)", "short": "control", "cssv": "--ax-co",
     "desc": "reviewed controls — the clean set; firing here is the FP cost"},
    {"key": "chat", "name": "Kimi chat (annotated)", "short": "chat", "cssv": "--ax-dm",
     "desc": "chat-register rollouts with human metagaming spans"},
    {"key": "prose", "name": "Prose docs (annotated)", "short": "prose", "cssv": "--ax-dg",
     "desc": "metagaming-probe + Westover docs with human spans"},
    {"key": "dense", "name": "Dense c1–c5 (~all-forget)", "short": "dense", "cssv": "--ax-mc",
     "desc": "wall-to-wall in-category docs; coverage proxy"},
]

z = np.load(f"{ROOT}/out/harvest_test_v1.npz")
meta = json.load(open(f"{ROOT}/out/harvest_test_v1.meta.json"))
W = meta["width"]
av = z["act"].astype(np.float64); av[np.isinf(av)] = 65504.0
ntok = z["doc_ntok"].astype(np.int64)
nd = len(meta["docs"])
tok_base = np.zeros(nd + 1, dtype=np.int64); np.cumsum(ntok, out=tok_base[1:])
offs = z["offsets"].reshape(-1, 2)
ids = [d["id"] for d in meta["docs"]]

mean_dl = np.zeros((nd, W)); np.add.at(mean_dl, (z["doc_idx"], z["lat_idx"]), av); mean_dl /= ntok[:, None]
max_dl = np.zeros((nd, W)); np.maximum.at(max_dl, (z["doc_idx"], z["lat_idx"]), av)

lt = np.array([docs_meta[i]["label_type"] for i in ids])
posm, ctlm = lt == "spans", lt == "clean"

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

latents = {}
for l in MEMBERS:
    latents[str(l)] = {
        "auc": round(auc_l(l), 4), "dec_spec": None,
        "docs_firing": int((max_dl[:, l] > 0).sum()),
        "mean_pos": round(float(mean_dl[posm, l].mean()), 2),
        "mean_ctl": round(float(mean_dl[ctlm, l].mean()), 2),
        "axis_means": {a["key"]: round(float(mean_dl[np.array([docs_meta[i]["register"] for i in ids]) == a["key"], l].mean()), 2) for a in AXES},
        "fires_on_generic_deception": False,
        "in_sep_shortlist": True, "in_dec_shortlist": conf_of[l] == "h",
        "description": descs.get(l), "explainer": None,
        "neuronpedia": f"https://www.neuronpedia.org/{NP_MODEL}/{NP_SOURCE}/{l}",
        "firing_rate": None, "raw_examples": [],
        "v2_class": conf_of[l],
    }

docs_out, tv_docs, texts = [], [], {}
memset = set(MEMBERS)
for i, did in enumerate(ids):
    dm = docs_meta[did]
    texts[did] = {"text": dm["text"], "context": None}
    fires = []
    for l in MEMBERS:
        if max_dl[i][l] <= 0: continue
        fires.append([int(l), round(float(mean_dl[i][l]), 2), round(float(max_dl[i][l]), 1), 0, 0])
    fires.sort(key=lambda x: -x[2])
    docs_out.append({
        "i": i, "id": did, "line": i + 1, "source": dm["config"], "model": None,
        "env": dm["label_type"],
        "category": f"{dm['config']} · {dm['label_type']}" + (f" · {len(dm['char_spans'])} spans" if dm["char_spans"] else ""),
        "axis": dm["register"],
        "label": "positive" if (dm["char_spans"] or dm["label_type"] == "dense_forget") else "control",
        "n_tokens": int(ntok[i]), "n_chars": len(dm["text"]), "has_context": False,
        "fires": fires[:45], "n_latents_firing": int((max_dl[i][MEMBERS] > 0).sum()),
        "prov_items": [f"TEST-v1 doc (config {dm['config']}, label_type {dm['label_type']}) — camgeodesic/metagaming-labeling @ rev 244b6d00",
                        f"~/sae-exploration/data/test_docs_v1.jsonl · line {i + 1}"],
        "prov_cmd": f"sed -n '{i + 1}p' ~/sae-exploration/data/test_docs_v1.jsonl | jq -r '.text'",
    })
    sel = (z["doc_idx"] == i) & np.isin(z["lat_idx"], MEMBERS)
    acts = {}
    li_d, ti_d, av_d = z["lat_idx"][sel], z["tok_idx"][sel], av[sel]
    for l in set(li_d.tolist()):
        m = li_d == l
        acts[str(l)] = [[int(t), round(float(a), 1)] for t, a in zip(ti_d[m], av_d[m])]
    tv_docs.append({"id": did, "toks": [[int(s), int(e)] for s, e in offs[tok_base[i]:tok_base[i + 1]]],
                    "acts": acts, "gt": dm["char_spans"]})

n_tok_total = int(ntok.sum())
mu = np.bincount(z["lat_idx"], weights=av, minlength=W) / n_tok_total
sdv = np.sqrt(np.maximum(np.bincount(z["lat_idx"], weights=av**2, minlength=W)/n_tok_total - mu**2, 0))
fire_cnt = np.bincount(z["lat_idx"], minlength=W)
CATS = {"h": "high confidence", "m": "medium confidence", "l": "low confidence",
        "c1": "c1 ai-evals", "c2": "c2 ai-safety", "c3": "c3 human-oversight",
        "c4": "c4 training", "c5": "c5 swe-tests"}
tv = {"latents": MEMBERS, "docs": tv_docs,
      "np_prefix": f"https://www.neuronpedia.org/{NP_MODEL}/{NP_SOURCE}/",
      "descs": {str(l): descs.get(l) for l in MEMBERS},
      "taglist": {"categories": CATS,
                  "tags": {str(l): [[conf_of[l], "core"], [cat_of[l], "core"]] for l in MEMBERS}},
      "stats": {str(l): {"mean": round(float(mu[l]), 4), "sd": round(float(sdv[l]), 4),
                          "fire_rate": round(float(fire_cnt[l]) / n_tok_total, 5),
                          "ctl_mean": round(float(mu[l]), 4), "ctl_sd": round(float(sdv[l]), 4),
                          "ctl_fire_rate": round(float(fire_cnt[l]) / n_tok_total, 5), "max": 0} for l in MEMBERS},
      "stats_note": (f"Challenger set {SET_NAME} (649 features, judge claude-fable-5) over TEST v1. "
                     "Toggle confidence tiers (h/m/l) and categories (c1–c5) to see which parts "
                     "of the set drive firing. Underlined text = human-labeled metagaming span."),
      "rule_note": "Rathi & Radford 2601.21571 §5.1"}
out = {"sae": f"test-v1 · {SET_NAME}", "model": "unsloth/gemma-3-27b-it", "layer": 40, "n_layers": 62,
       "width": W, "neuronpedia_model": NP_MODEL, "neuronpedia_source": NP_SOURCE,
       "axis_counts": {a["key"]: sum(1 for i2 in ids if docs_meta[i2]["register"] == a["key"]) for a in AXES},
       "axes_def": AXES, "docs": docs_out, "latents": latents, "tokenView": tv, "texts": texts}
json.dump(out, open(f"{ROOT}/out/corpus_view_test_v1.json", "w"), separators=(",", ":"))
print(f"fable view: {len(docs_out)} docs, {len(MEMBERS)} members, "
      f"{sum(len(d['gt']) for d in tv_docs)} human spans")
