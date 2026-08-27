"""Corpus view for the TEST-v1 artifact: the eval-protocol-v1 TEST split
(53 annotated + 300 dense docs) x the OG blocklist features (222).

Emits out/corpus_view_labeled.json in the artifact schema, including per-doc
human span char-ranges ("gt") in the token view so the page can overlay
ground truth against feature firing.

Usage: python build_labeled_view.py
"""
import glob
import gzip
import json
import sys

import numpy as np

ROOT = "/home/a5k/cwtice.a5k/sae-exploration"
sys.path.insert(0, f"{ROOT}/scripts")
NP_MODEL, NP_SOURCE = "gemma-3-27b-it", "40-gemmascope-2-res-16k"
TOPK = 10

OG = sorted(int(k) for k in json.load(open(f"{ROOT}/data/latent_list_2026-07-29.json"))["tags"])
docs_meta = {d["id"]: d for d in map(json.loads, open(f"{ROOT}/data/test_docs_v1.jsonl"))}

AXES = [
    {"key": "chat", "name": "Kimi chat (annotated)", "short": "chat", "cssv": "--ax-dm",
     "desc": "chat-register rollouts with human metagaming spans — the weak register"},
    {"key": "dense", "name": "Dense c1–c5 (~all-forget)", "short": "dense", "cssv": "--ax-mc",
     "desc": "wall-to-wall in-category docs; coverage proxy, no spans"},
    {"key": "prose", "name": "Prose docs (annotated)", "short": "prose", "cssv": "--ax-dg",
     "desc": "metagaming-probe + Westover docs with human spans"},
    {"key": "control", "name": "Controls (annotated)", "short": "control", "cssv": "--ax-co",
     "desc": "reviewed controls incl. the 6 hard-vocabulary ctl_* docs"},
]
def axis_of_group(g):
    return g

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

universe = set(OG)
for i in range(nd):
    order = np.argsort(-max_dl[i])[:TOPK]
    universe |= {int(l) for l in order if max_dl[i][l] > 0}
universe = sorted(universe)

status = np.array([docs_meta[i]["label_type"] for i in ids])
posm, ctlm = status == "spans", status == "clean"
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
        if d.get("layer") == NP_SOURCE: descs[int(d["index"])] = d["description"]

from sae_local import sae_dir, snapshot_dir
from safetensors import safe_open
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(snapshot_dir("unsloth/gemma-3-27b-it"))
raw = {}
_cache = json.load(open(f"{ROOT}/out/corpus_view_labeled.json"))["latents"]
for k, v in _cache.items():
    if int(k) in set(universe):
        raw[int(k)] = {"rate": v["firing_rate"], "ex": v["raw_examples"]}
with safe_open(f"{sae_dir('gemma3-l40-16k')}/examples.safetensors", framework="np") as f:
    seq_t, pos_t, act_t = f.get_slice("seq_ids"), f.get_slice("positions"), f.get_slice("activations")
    toks_t, freq = f.get_slice("tokens"), f.get_slice("feature_frequencies")
    for lat in universe:
        if lat in raw:
            continue
        sid_a, p_a, a_a = seq_t[lat], pos_t[lat], act_t[lat]
        ok = sid_a >= 0
        ex, seen = [], set()
        if ok.any():
            for i in np.flatnonzero(ok)[np.argsort(-a_a[ok])]:
                sid = int(sid_a[i])
                if sid in seen: continue
                seen.add(sid)
                s = toks_t[sid]; p = int(p_a[i])
                lo, hi = max(0, p - 28), min(len(s), p + 29)
                ex.append({"act": round(float(a_a[i]), 1),
                           "text": tok.decode(s[lo:p]) + "«" + tok.decode(s[p:p + 1]) + "»" + tok.decode(s[p + 1:hi])})
                if len(ex) >= 4: break
        raw[lat] = {"rate": float(freq[lat]), "ex": ex}

latents = {}
ogset = set(OG)
for l in universe:
    latents[str(l)] = {
        "auc": round(auc_l(l), 4), "dec_spec": None,
        "docs_firing": int((max_dl[:, l] > 0).sum()),
        "mean_pos": round(float(mean_dl[posm, l].mean()), 2),
        "mean_ctl": round(float(mean_dl[ctlm, l].mean()), 2),
        "axis_means": {a["key"]: round(float(mean_dl[np.array([axis_of_group(docs_meta[i]["group"]) for i in ids]) == a["key"], l].mean()), 2) for a in AXES},
        "fires_on_generic_deception": False,
        "in_sep_shortlist": l in ogset, "in_dec_shortlist": False,
        "description": descs.get(l), "explainer": None,
        "neuronpedia": f"https://www.neuronpedia.org/{NP_MODEL}/{NP_SOURCE}/{l}",
        "firing_rate": raw[l]["rate"], "raw_examples": raw[l]["ex"],
        "v2_class": "og" if l in ogset else None,
    }

docs_out, tv_docs, texts = [], [], {}
for i, did in enumerate(ids):
    dm = docs_meta[did]
    texts[did] = {"text": dm["text"], "context": None}
    fires = []
    for l in universe:
        if max_dl[i][l] <= 0: continue
        fires.append([int(l), round(float(mean_dl[i][l]), 2), round(float(max_dl[i][l]), 1), 0, 0])
    fires.sort(key=lambda x: -x[2])
    keep = [f for f in fires if f[0] in ogset] + [f for f in fires if f[0] not in ogset][:TOPK]
    keep.sort(key=lambda x: -x[2])
    ax = axis_of_group(dm["register"])
    docs_out.append({
        "i": i, "id": did, "line": i + 1, "source": dm["config"], "model": None,
        "env": dm["label_type"], "category": f"{dm['config']} · {dm['label_type']}" + (f" · {len(dm['char_spans'])} spans" if dm["char_spans"] else ""),
        "axis": ax, "label": "positive" if (dm["char_spans"] or dm["label_type"] == "dense_forget") else "control",
        "n_tokens": int(ntok[i]), "n_chars": len(dm["text"]), "has_context": False,
        "fires": keep[:45], "n_latents_firing": int((max_dl[i] > 0).sum()),
        "prov_items": [f"TEST-v1 doc (config {dm['config']}, label_type {dm['label_type']}) — camgeodesic/metagaming-labeling @ rev 244b6d00",
                        f"~/sae-exploration/data/test_docs_v1.jsonl · line {i + 1}"],
        "prov_cmd": f"sed -n '{i + 1}p' ~/sae-exploration/data/test_docs_v1.jsonl | jq -r '.text'",
    })
    sel = (z["doc_idx"] == i) & np.isin(z["lat_idx"], OG)
    acts = {}
    for l in OG:
        m = sel & (z["lat_idx"] == l)
        if m.any():
            acts[str(l)] = [[int(t), round(float(a), 1)] for t, a in zip(z["tok_idx"][m], av[m])]
    tv_docs.append({"id": did, "toks": [[int(s), int(e)] for s, e in offs[tok_base[i]:tok_base[i + 1]]],
                    "acts": acts, "gt": dm["char_spans"]})

n_tok_total = int(ntok.sum())
mu = np.bincount(z["lat_idx"], weights=av, minlength=W) / n_tok_total
sdv = np.sqrt(np.maximum(np.bincount(z["lat_idx"], weights=av**2, minlength=W)/n_tok_total - mu**2, 0))
fire_cnt = np.bincount(z["lat_idx"], minlength=W)
tv = {"latents": OG, "docs": tv_docs,
      "np_prefix": f"https://www.neuronpedia.org/{NP_MODEL}/{NP_SOURCE}/",
      "descs": {str(l): descs.get(l) for l in OG},
      "taglist": {"categories": {"og": "OG blocklist (regex over captions)"},
                  "tags": {str(l): [["og", "core"]] for l in OG}},
      "stats": {str(l): {"mean": round(float(mu[l]), 4), "sd": round(float(sdv[l]), 4),
                          "fire_rate": round(float(fire_cnt[l]) / n_tok_total, 5),
                          "ctl_mean": round(float(mu[l]), 4), "ctl_sd": round(float(sdv[l]), 4),
                          "ctl_fire_rate": round(float(fire_cnt[l]) / n_tok_total, 5), "max": 0} for l in OG},
      "stats_note": ("TEST v1 of eval protocol v1 — the OG blocklist baseline (222 features), "
                     "thresholds mean+4SD over the test corpus. Underlined text = human span; "
                     "dense docs are scored as ~all-forget coverage."),
      "rule_note": "Rathi & Radford 2601.21571 §5.1"}
out = {"sae": "test-v1-353", "model": "unsloth/gemma-3-27b-it", "layer": 40, "n_layers": 62,
       "width": W, "neuronpedia_model": NP_MODEL, "neuronpedia_source": NP_SOURCE,
       "axis_counts": {a["key"]: sum(1 for i2 in ids if axis_of_group(docs_meta[i2]["group"]) == a["key"]) for a in AXES},
       "axes_def": AXES, "docs": docs_out, "latents": latents, "tokenView": tv, "texts": texts}
json.dump(out, open(f"{ROOT}/out/corpus_view_test_v1.json", "w"), separators=(",", ":"))
print(f"test-v1 view: {len(docs_out)} docs, {len(latents)} latents, "
      f"{sum(len(d['gt']) for d in tv_docs)} human spans")
