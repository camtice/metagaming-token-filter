"""Assemble per-corpus view data for the v2 feature-firing artifact.

CPU-only: everything derives from the existing harvests
(out/harvest_oldprobe, out/harvest_dev, out/harvest_val) plus
examples.safetensors (raw autointerp evidence) and the label dumps
(captions, readonly). Two corpora:

  apollo   the original 73-doc probe set (Schoen CoT + SA transcript + controls)
  probe432 the metagaming-probe corpus, dev+val halves merged (432 docs)

Per corpus: docs (metadata + provenance strings + per-doc firing list over the
universe = v2 members + per-doc top-15 by max act), latent stats over that
corpus's axes, and a token view (offsets + sparse acts) for the 30 v2 members.

Usage: python build_corpus_views.py
Output: out/corpus_view_apollo.json, out/corpus_view_probe432.json
"""
import json
import glob
import gzip

import numpy as np

ROOT = "/home/a5k/cwtice.a5k/sae-exploration"
fin = json.load(open(f"{ROOT}/out/final_features_v4.json"))
V2 = fin["members"]
V2CLASS = {k: "+".join(v) for k, v in fin["class"].items()}
TOPK = 15
MAX_FIRES = 45
NP_MODEL, NP_SOURCE = "gemma-3-27b-it", "40-gemmascope-2-res-16k"

descs = {}
for fn in glob.glob(f"{ROOT}/out/np_expl_batch-*.jsonl.gz"):
    for line in gzip.open(fn, "rt"):
        d = json.loads(line)
        if d.get("layer") == NP_SOURCE:
            descs[int(d["index"])] = d["description"]

AXES_APOLLO = [
    {"key": "deception_metagaming", "name": "Deception · metagaming", "short": "dec-meta", "cssv": "--ax-dm",
     "desc": "fudging / sandbagging / misreporting aimed at a grader"},
    {"key": "metagaming_clean", "name": "Metagaming · clean", "short": "meta-clean", "cssv": "--ax-mc",
     "desc": "metagaming with no deception content"},
    {"key": "deception_generic", "name": "Deception · generic", "short": "dec-generic", "cssv": "--ax-dg",
     "desc": "deception without AI oversight — n=2, underpowered"},
    {"key": "control_other", "name": "Control · other", "short": "control", "cssv": "--ax-co",
     "desc": "same vocabulary, non-metagaming senses"},
]
AXES_P432 = [
    {"key": "positive", "name": "Positive · metagaming docs", "short": "positive", "cssv": "--ax-dm",
     "desc": "what the filter must catch — 100 synthetic + 32 Westover-real"},
    {"key": "near_miss", "name": "Control · near-miss", "short": "near-miss", "cssv": "--ax-dg",
     "desc": "human oversight vocabulary — grading, reviews, rewards, audits"},
    {"key": "ml_distractor", "name": "Control · ML distractor", "short": "ml-distr", "cssv": "--ax-mc",
     "desc": "benign ML — optimizers, diffusion, CV"},
    {"key": "far_negative", "name": "Control · far negative", "short": "far-neg", "cssv": "--ax-co",
     "desc": "everyday topics — cooking, travel, gardening"},
]


def load_harvest(prefix):
    z = np.load(prefix + ".npz")
    meta = json.load(open(prefix + ".meta.json"))
    av = z["act"].astype(np.float64)
    av[np.isinf(av)] = 65504.0
    return z, meta, av


def prov_apollo(doc, line):
    items = []
    src = str(doc.get("source"))
    if src.startswith("schoen_fig"):
        items.append(f"Verbatim CoT excerpt — Schoen et al. 2025 (arXiv:2509.15541), Appendix N, Figure {src[10:]}")
    elif src == "sa_oversight":
        items.append("Chunk of the OpenAI situational-awareness oversight transcript (context excluded from scoring)")
    else:
        items.append("Control written for the probe dataset")
    items.append(f"~/sae-exploration/data/probe_dataset.jsonl · line {line}")
    cmd = f"sed -n '{line}p' ~/sae-exploration/data/probe_dataset.jsonl | jq -r '.text'"
    return items, cmd


def prov_p432(doc, line, half):
    items = []
    if doc.get("origin") == "westover_filter":
        items.append("Real Alignment-Forum post labelled filter by awestover/filtering-for-misalignment")
    elif doc["group"] == "positive":
        items.append("Synthetic positive authored for the metagaming-probe dataset (PR #123)")
    else:
        items.append(f"Synthetic {doc['group']} control (PR #123, 3-tier control design)")
    f = f"~/sae-exploration/data/{half}_docs_pr123.jsonl"
    items.append(f"{f} · line {line} · {half} half")
    cmd = f"sed -n '{line}p' {f} | jq -r '.text'"
    return items, cmd


def corpus_view(harvests, axes, prov_fn, texts_files, corpus_label):
    """harvests: list of (prefix, half_tag). Merged into one corpus."""
    docs_out, tv_docs = [], []
    texts = {}
    for tf in texts_files:
        for l in open(tf):
            r = json.loads(l)
            texts[r["id"]] = {"text": r["text"], "context": r.get("context") or None}
    # accumulate doc-level stats
    all_mean, all_max, groups, metas, offsets_all = [], [], [], [], []
    for prefix, half in harvests:
        z, meta, av = load_harvest(prefix)
        nd = len(meta["docs"])
        W = meta["width"]
        ntok = z["doc_ntok"].astype(np.int64)
        mean_dl = np.zeros((nd, W))
        np.add.at(mean_dl, (z["doc_idx"], z["lat_idx"]), av)
        mean_dl /= ntok[:, None]
        max_dl = np.zeros((nd, W))
        np.maximum.at(max_dl, (z["doc_idx"], z["lat_idx"]), av)
        frac_dl = np.zeros((nd, W))
        np.add.at(frac_dl, (z["doc_idx"], z["lat_idx"]), 1.0)
        frac_dl /= ntok[:, None]
        offs = z["offsets"].reshape(-1, 2)
        tok_base = np.zeros(nd + 1, dtype=np.int64)
        np.cumsum(ntok, out=tok_base[1:])
        for i, d in enumerate(meta["docs"]):
            metas.append((d, half, i))
            offsets_all.append(offs[tok_base[i]:tok_base[i + 1]])
        all_mean.append(mean_dl)
        all_max.append(max_dl)
        groups.append((z, meta, av, tok_base, ntok, frac_dl))
    mean_dl = np.vstack(all_mean)
    max_dl = np.vstack(all_max)
    nd = mean_dl.shape[0]
    W = mean_dl.shape[1]
    axis_of = np.array([d.get("axis") or d.get("group") for d, _, _ in metas])
    axis_keys = [a["key"] for a in axes]
    axis_mean = {k: mean_dl[axis_of == k].mean(0) for k in axis_keys}

    # universe: v2 members + per-doc top-K by max
    universe = set(V2)
    for i in range(nd):
        order = np.argsort(-max_dl[i])[:TOPK]
        universe |= {int(l) for l in order if max_dl[i][l] > 0}
    universe = sorted(universe)

    # per-latent headline: AUC positive-axis vs the corpus's control axes
    pos_key = axis_keys[0]
    posm = axis_of == pos_key
    ctlm = ~posm
    def auc_vec(feats):
        out = {}
        for l in feats:
            x = np.concatenate([mean_dl[posm, l], mean_dl[ctlm, l]])
            n1 = int(posm.sum())
            order = x.argsort(kind="stable")
            r = np.arange(1, len(x) + 1, dtype=float)
            xv = x[order]
            _, first = np.unique(xv, return_index=True)
            bounds = np.append(first, len(x))
            for s, e in zip(bounds[:-1], bounds[1:]):
                r[s:e] = r[s:e].mean()
            rk = np.empty_like(r)
            rk[order] = r
            out[l] = float((rk[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(x) - n1)))
        return out
    aucs = auc_vec(universe)

    # raw autointerp examples for universe (evidence behind each caption)
    import sys
    sys.path.insert(0, f"{ROOT}/scripts")
    from sae_local import sae_dir
    from safetensors import safe_open
    from transformers import AutoTokenizer
    from sae_local import snapshot_dir
    tok = AutoTokenizer.from_pretrained(snapshot_dir("unsloth/gemma-3-27b-it"))
    raw = {}
    with safe_open(f"{sae_dir('gemma3-l40-16k')}/examples.safetensors", framework="np") as f:
        seq_t, pos_t, act_t = (f.get_slice("seq_ids"), f.get_slice("positions"), f.get_slice("activations"))
        toks_t, freq = f.get_slice("tokens"), f.get_slice("feature_frequencies")
        for lat in universe:
            sid_a, p_a, a_a = seq_t[lat], pos_t[lat], act_t[lat]
            ok = sid_a >= 0
            ex, seen = [], set()
            if ok.any():
                for i in np.flatnonzero(ok)[np.argsort(-a_a[ok])]:
                    sid = int(sid_a[i])
                    if sid in seen:
                        continue
                    seen.add(sid)
                    s = toks_t[sid]
                    p = int(p_a[i])
                    lo, hi = max(0, p - 28), min(len(s), p + 29)
                    ex.append({"act": round(float(a_a[i]), 1),
                               "text": tok.decode(s[lo:p]) + "«" + tok.decode(s[p:p + 1]) + "»" + tok.decode(s[p + 1:hi])})
                    if len(ex) >= 5:
                        break
            raw[lat] = {"rate": float(freq[lat]), "ex": ex}

    latents = {}
    for l in universe:
        latents[str(l)] = {
            "auc": round(aucs[l], 4),
            "dec_spec": None,
            "docs_firing": int((max_dl[:, l] > 0).sum()),
            "mean_pos": round(float(mean_dl[posm, l].mean()), 2),
            "mean_ctl": round(float(mean_dl[ctlm, l].mean()), 2),
            "axis_means": {k: round(float(axis_mean[k][l]), 2) for k in axis_keys},
            "fires_on_generic_deception": False,
            "in_sep_shortlist": l in set(V2),
            "in_dec_shortlist": l in set(V2) and "cot" in V2CLASS[str(l)],
            "description": descs.get(l),
            "explainer": None,
            "neuronpedia": f"https://www.neuronpedia.org/{NP_MODEL}/{NP_SOURCE}/{l}",
            "firing_rate": raw[l]["rate"],
            "raw_examples": raw[l]["ex"],
            "v2_class": V2CLASS.get(str(l)),
        }

    # docs with fires + provenance; token view for the 30 members
    tv = {"latents": V2, "stats": {}, "docs": [],
          "np_prefix": f"https://www.neuronpedia.org/{NP_MODEL}/{NP_SOURCE}/",
          "descs": {str(l): descs.get(l) for l in V2},
          "taglist": {"categories": {"v4": "v4 token-selected (dev only)", "cot": "independently matches the Apollo-66 shortlist"},
                      "tags": {str(l): [[c, "core"] for c in V2CLASS[str(l)].split("+")] for l in V2}},
          "stats_note": ("Per-feature mean/SD over this corpus's tokens (zeros included). "
                         "v2 classes: AI-oversight specific vs human-domain oversight & evaluation "
                         "(near-miss firing is descriptive, not a defect)."),
          "rule_note": "Rathi & Radford 2601.21571 §5.1 rule"}
    # corpus token stats for members
    doc_ptr = 0
    line_ctr = {}
    for (z, meta, av, tok_base, ntok, frac_dl), (prefix, half) in zip(groups, harvests):
        sel = np.isin(z["lat_idx"], V2)
        di, ti, li_, av_ = z["doc_idx"][sel], z["tok_idx"][sel], z["lat_idx"][sel], av[sel]
        n_tok = int(ntok.sum())
        for l in V2:
            m = li_ == l
            key = str(l)
            st = tv["stats"].setdefault(key, {"sum": 0.0, "sumsq": 0.0, "n": 0, "fire": 0})
            st["sum"] += float(av_[m].sum())
            st["sumsq"] += float((av_[m] ** 2).sum())
            st["n"] += n_tok
            st["fire"] += int(m.sum())
        for i, d in enumerate(meta["docs"]):
            gi = doc_ptr + i
            doc, half_tag, local_i = metas[gi]
            line = line_ctr.get(half, 0) + 1  # recompute below properly
        doc_ptr += len(meta["docs"])
    for key, st in tv["stats"].items():
        mu = st["sum"] / st["n"]
        sd = (max(st["sumsq"] / st["n"] - mu ** 2, 0)) ** 0.5
        tv["stats"][key] = {"mean": round(mu, 4), "sd": round(sd, 4),
                            "fire_rate": round(st["fire"] / st["n"], 5),
                            "ctl_mean": round(mu, 4), "ctl_sd": round(sd, 4),
                            "ctl_fire_rate": round(st["fire"] / st["n"], 5),
                            "max": 0}

    # per-doc fires + token-view docs
    line_in_file = {}
    for gi, (doc, half, local_i) in enumerate(metas):
        line_in_file[gi] = line_in_file.get(half, 0)
    # (line numbers = local index + 1 within each half's jsonl)
    for gi, (doc, half, local_i) in enumerate(metas):
        fires = []
        for l in universe:
            if max_dl[gi][l] <= 0:
                continue
            fires.append([int(l), round(float(mean_dl[gi][l]), 2), round(float(max_dl[gi][l]), 1), 0, 0])
        fires.sort(key=lambda x: -x[2])
        keep = [f for f in fires if f[0] in set(V2)] + [f for f in fires if f[0] not in set(V2)][:10]
        keep.sort(key=lambda x: -x[2])
        items, cmd = prov_fn(doc, local_i + 1, half) if prov_fn is prov_p432 else prov_fn(doc, local_i + 1)
        docs_out.append({
            "i": gi, "id": doc["id"], "line": local_i + 1,
            "source": doc.get("source"), "model": None,
            "env": doc.get("topic"), "category": doc.get("category") or doc.get("topic") or "",
            "axis": doc.get("axis") or doc.get("group"), "label": doc.get("label") or ("positive" if (doc.get("group") == "positive") else "control"),
            "n_tokens": len(offsets_all[gi]), "n_chars": len(texts.get(doc["id"], {}).get("text", "")),
            "has_context": bool(texts.get(doc["id"], {}).get("context")),
            "half": half,
            "fires": keep[:MAX_FIRES], "n_latents_firing": int((max_dl[gi] > 0).sum()),
            "prov_items": items, "prov_cmd": cmd,
        })
    # token-view docs: sparse acts for members from each harvest
    for (z, meta, av, tok_base, ntok, frac_dl), (prefix, half) in zip(groups, harvests):
        sel = np.isin(z["lat_idx"], V2)
        di, ti, li_, av_ = z["doc_idx"][sel], z["tok_idx"][sel], z["lat_idx"][sel], av[sel]
        offs = z["offsets"].reshape(-1, 2)
        for i, d in enumerate(meta["docs"]):
            m = di == i
            acts = {}
            for l in V2:
                mm = m & (li_ == l)
                if mm.any():
                    acts[str(l)] = [[int(t), round(float(a), 1)] for t, a in zip(ti[mm], av_[mm])]
            tv["docs"].append({"id": d["id"],
                               "toks": [[int(s), int(e)] for s, e in offs[tok_base[i]:tok_base[i + 1]]],
                               "acts": acts})
    return {"sae": corpus_label, "model": "unsloth/gemma-3-27b-it", "layer": 40, "n_layers": 62,
            "width": 16384, "neuronpedia_model": NP_MODEL, "neuronpedia_source": NP_SOURCE,
            "axis_counts": {k: int((axis_of == k).sum()) for k in axis_keys},
            "axes_def": axes, "docs": docs_out, "latents": latents, "tokenView": tv,
            "texts": texts}


apollo = corpus_view([(f"{ROOT}/out/harvest_oldprobe", "probe")], AXES_APOLLO, prov_apollo,
                     [f"{ROOT}/data/probe_dataset.jsonl"], "apollo-probe-73")
json.dump(apollo, open(f"{ROOT}/out/corpus_view_apollo.json", "w"), separators=(",", ":"))
print("apollo:", len(apollo["docs"]), "docs,", len(apollo["latents"]), "latents")

p432 = corpus_view([(f"{ROOT}/out/harvest_dev", "dev"), (f"{ROOT}/out/harvest_val", "val")],
                   AXES_P432, prov_p432,
                   [f"{ROOT}/data/dev_docs_pr123.jsonl", f"{ROOT}/data/val_docs_pr123.jsonl"],
                   "metagaming-probe-432")
json.dump(p432, open(f"{ROOT}/out/corpus_view_probe432.json", "w"), separators=(",", ":"))
print("probe432:", len(p432["docs"]), "docs,", len(p432["latents"]), "latents")
