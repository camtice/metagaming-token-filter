"""Run the metagaming probe dataset through an SAE and rank latents.

Two rankings, deliberately kept separate:

  activation  -- how hard a latent fires on metagaming text. Naive: high-frequency
                 latents ("the", punctuation, generic prose) score well on any text.
  contrast    -- mean activation on positives divided by mean on controls. This is
                 the ranking that actually isolates metagaming, because the controls
                 use the same surface vocabulary (test, grade, monitor, reward,
                 benchmark, oversight) in non-metagaming senses.

Latents are also reported with per-category breakdown and a Neuronpedia URL, so
the shortlist can be taken straight back to the dashboards for relevance checks.

Usage:
  python run_probe.py gemma3-l40-16k [--top 40] [--max-len 512]
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "/home/a5k/cwtice.a5k/sae-exploration/scripts")
from regex_v3 import COMPILED, char_mask  # noqa: E402
from sae_local import REGISTRY, Runner  # noqa: E402

ROOT = "/home/a5k/cwtice.a5k/sae-exploration"

ap = argparse.ArgumentParser()
ap.add_argument("sae")
ap.add_argument("--dataset", default=f"{ROOT}/data/probe_dataset.jsonl")
ap.add_argument("--top", type=int, default=40)
ap.add_argument("--max-len", type=int, default=512)
ap.add_argument("--out", default=None)
args = ap.parse_args()

cfg = REGISTRY[args.sae]
rows = [json.loads(l) for l in open(args.dataset)]
print(f"{args.sae}: {len(rows)} texts "
      f"({sum(r['label']=='positive' for r in rows)} positive / "
      f"{sum(r['label']=='control' for r in rows)} control)")

runner = Runner(args.sae)
tok = runner.tokenizer
W = cfg["width"]
bos_id = tok.bos_token_id

# accumulators
sum_pos = np.zeros(W, dtype=np.float64)      # summed activation over positive tokens
sum_ctl = np.zeros(W, dtype=np.float64)
n_pos_tok = n_ctl_tok = 0
max_pos = np.zeros(W, dtype=np.float32)
docs_pos = np.zeros(W, dtype=np.int32)       # positive docs where latent fires at all
cat_sum, cat_tok = {}, {}
top_ex = {}                                  # latent -> (act, doc_id, token, snippet)
# The SA transcript is 16 of 40 positive docs and far longer than any Schoen
# excerpt, so token-weighted means let its *content* dominate. Track per-document
# means and per-source firing so we can rank on concepts that recur across both
# independent sources rather than on one long document's subject matter.
docmean_pos = []                             # list of per-doc mean-activation vectors
ctlmean_pos = []                             # same, for controls (used for AUC)
src_fire = {}                                # source -> bool vector of latents firing
# v3 term-category profiling: summed activation on tokens that overlap a term of
# each category, so a latent can be told apart as level-1/2 oversight vs level-5
# SWE-test vs level-3 human-assessment.
CATS = list(COMPILED)
term_sum = {c: np.zeros(W, dtype=np.float64) for c in CATS}
term_tok = {c: 0 for c in CATS}


@torch.no_grad()
def encode(row):
    """Return (acts[n_scored, W], token ids, per-category token masks).

    Context tokens are excluded from scoring. Category masks come from character
    offsets so v3 term hits land on the right SAE token positions.
    """
    ctx_ids = []
    if row.get("context"):
        ctx_ids = tok(row["context"], add_special_tokens=False,
                      truncation=True, max_length=args.max_len // 2)["input_ids"]
    enc = tok(row["text"], add_special_tokens=False, truncation=True,
              max_length=args.max_len, return_offsets_mapping=True)
    body, offsets = enc["input_ids"], enc["offset_mapping"]
    ids = ([bos_id] if bos_id is not None else []) + ctx_ids + body
    ids = ids[-args.max_len:]                       # keep the body if we must clip
    n_body = min(len(body), len(ids))
    offsets = offsets[-n_body:]

    masks = {}
    for c in CATS:
        cm = char_mask(row["text"], categories=[c])
        masks[c] = np.array([any(cm[s:e]) if e > s else False for s, e in offsets])

    t = torch.tensor(ids)[None]
    acts = runner.sae.encode(runner.residual(t))[0]  # [seq, W]
    return acts[-n_body:].float().cpu().numpy(), ids[-n_body:], masks


for row in rows:
    acts, ids, masks = encode(row)
    for c in CATS:
        m = masks[c]
        if m.any():
            term_sum[c] += acts[m].sum(0)
            term_tok[c] += int(m.sum())
    s = acts.sum(0)
    if row["label"] == "positive":
        sum_pos += s
        n_pos_tok += acts.shape[0]
        max_pos = np.maximum(max_pos, acts.max(0))
        docs_pos += (acts.max(0) > 0).astype(np.int32)
        c = row["category"]
        cat_sum.setdefault(c, np.zeros(W)); cat_sum[c] += s
        cat_tok[c] = cat_tok.get(c, 0) + acts.shape[0]
        docmean_pos.append(s / acts.shape[0])
        grp = "sa_oversight" if row["source"] == "sa_oversight" else "schoen"
        src_fire.setdefault(grp, np.zeros(W, dtype=bool))
        src_fire[grp] |= acts.max(0) > 0
        # record the single strongest token per latent for eyeballing
        loc = acts.argmax(0)
        val = acts.max(0)
        for lat in np.flatnonzero(val > 0):
            if val[lat] > top_ex.get(lat, (0,))[0]:
                p = int(loc[lat])
                lo, hi = max(0, p - 12), min(len(ids), p + 13)
                top_ex[lat] = (float(val[lat]), row["id"],
                               tok.decode([ids[p]]),
                               tok.decode(ids[lo:p]) + "«" + tok.decode([ids[p]]) + "»"
                               + tok.decode(ids[p + 1:hi]))
    else:
        sum_ctl += s
        n_ctl_tok += acts.shape[0]
        ctlmean_pos.append(s / acts.shape[0])
    print(".", end="", flush=True)
print()

mean_pos = np.mean(np.stack(docmean_pos), axis=0)   # per-document, not per-token
mean_ctl = sum_ctl / max(n_ctl_tok, 1)
cross_source = src_fire.get("schoen", np.zeros(W, bool)) & \
    src_fire.get("sa_oversight", np.zeros(W, bool))
np_model, np_source = (cfg["neuronpedia"] or (None, None))

# Neuronpedia explanations, read from the local dump if it covers this SAE, so the
# shortlist is annotated without hitting the (rate-limited) API.
descs = {}
if np_source:
    import glob
    import gzip
    for fn in glob.glob(f"{ROOT}/out/np*batch-*.jsonl.gz"):
        for line in gzip.open(fn, "rt"):
            d = json.loads(line)
            if d.get("layer") == np_source:
                descs[int(d["index"])] = d["description"]
    print(f"loaded {len(descs)} local explanations for {np_source}")
# Additive smoothing on the control mean: without it, latents that are simply
# absent from the (small) control set get an infinite ratio and dominate.
eps = np.percentile(mean_pos[mean_pos > 0], 25) if (mean_pos > 0).any() else 1.0
contrast = mean_pos / (mean_ctl + eps)

def url(lat):
    if not np_model:
        return ""
    return f"https://www.neuronpedia.org/{np_model}/{np_source}/{lat}"


term_mean = {c: term_sum[c] / max(term_tok[c], 1) for c in CATS}
SHORT = {c: c.split("_")[0] for c in CATS}


def profile(lat):
    """Which v3 category this latent's activation concentrates on."""
    v = {c: term_mean[c][lat] for c in CATS}
    tot = sum(v.values())
    if tot <= 0:
        return "none", 0.0
    best = max(v, key=v.get)
    return SHORT[best], v[best] / tot


def table(idx, title, score_name, score):
    print(f"\n{'='*134}\n{title}\n{'='*134}")
    print(f"{'latent':>7} {score_name:>9} {'docs':>5} {'v3cat':>6} {'frac':>5}  "
          f"{'neuronpedia description':<34} {'top activating context':<56}")
    for lat in idx:
        lat = int(lat)
        snip = top_ex.get(lat, (0, "", "", ""))[3].replace("\n", "⏎")[:56]
        cat, frac = profile(lat)
        print(f"{lat:>7} {score[lat]:>9.1f} {docs_pos[lat]:>5} {cat:>6} {frac:>5.2f}  "
              f"{(descs.get(lat) or '—')[:34]:<34} {snip:<56}")


n_pos_docs = sum(r["label"] == "positive" for r in rows)
# Fire on a reasonable share of positive docs (not a one-off spike) AND on both
# independent sources -- otherwise a latent keyed to the SA transcript's subject
# matter (Hendra virus, "==START TEXT==") outranks genuine metagaming latents.
robust = (docs_pos >= max(3, int(0.15 * n_pos_docs))) & cross_source
cand = np.flatnonzero(robust & (mean_pos > 0))
print(f"\n{int(cross_source.sum())} latents fire on BOTH sources; "
      f"{len(cand)} also clear the >=15%-of-positive-docs bar")

by_contrast = cand[np.argsort(-contrast[cand])][:args.top]
by_act = cand[np.argsort(-mean_pos[cand])][:args.top]

table(by_contrast, "RANKED BY CONTRAST (metagaming vs controls), cross-source latents only",
      "contrast", contrast)
table(by_act, "RANKED BY RAW ACTIVATION — includes generic / high-frequency latents",
      "mean_act", mean_pos)

# per-category top latents
print(f"\n{'='*118}\ntop-5 contrast latents per category\n{'='*118}")
for c in sorted(cat_sum):
    m = cat_sum[c] / max(cat_tok[c], 1)
    sc = m / (mean_ctl + eps)
    sel = np.flatnonzero(robust)
    best = sel[np.argsort(-sc[sel])][:5]
    print(f"{c:<26} {', '.join(str(int(x)) for x in best)}")

# Latents whose activation concentrates on v3 categories 1+2 (the always-filter
# core) are the level-2 forget-set candidates; category-5 concentration marks a
# latent that would only be filtered at level 3.
print(f"\n{'='*134}\nv3 CATEGORY-SPECIFIC latents (top 10 per category, cross-source only)"
      f"\n{'='*134}")
for c in CATS:
    others = [x for x in CATS if x != c]
    spec = term_mean[c] / (np.max(np.stack([term_mean[o] for o in others]), axis=0) + eps)
    sel = np.flatnonzero(robust & (term_mean[c] > 0))
    best = sel[np.argsort(-spec[sel])][:10]
    print(f"\n{c}  ({term_tok[c]} matched tokens)")
    for lat in best:
        lat = int(lat)
        snip = top_ex.get(lat, (0, "", "", ""))[3].replace("\n", "⏎")[:58]
        print(f"   {lat:>6} spec={spec[lat]:>7.1f}  {(descs.get(lat) or '—')[:30]:<30} {snip}")

out_path = args.out or f"{ROOT}/out/probe_{args.sae}.json"
# --- separation: single-latent AUC over documents --------------------------
# Contrast rewards latents that spike hard on a few positives; AUC rewards
# latents that rank *every* metagaming doc above *every* control. The latter is
# what a forget-set latent actually needs to do, so it is the headline ranking.
doc_pos = np.stack(docmean_pos)
doc_ctl = np.stack(ctlmean_pos) if ctlmean_pos else np.zeros((1, W))
n_p, n_c = doc_pos.shape[0], doc_ctl.shape[0]

auc_all = np.zeros(W)
for lo in range(0, W, 4096):                      # chunked to bound memory
    hi = min(lo + 4096, W)
    a = doc_pos[:, lo:hi][:, None, :]             # [n_p, 1, chunk]
    b = doc_ctl[:, lo:hi][None, :, :]             # [1, n_c, chunk]
    auc_all[lo:hi] = ((a > b).sum((0, 1)) + 0.5 * (a == b).sum((0, 1))) / (n_p * n_c)

by_auc = cand[np.argsort(-(auc_all[cand] + 1e-6 * contrast[cand]))][:args.top]
table(by_auc, "RANKED BY SEPARATION (single-latent AUC, metagaming docs vs controls)",
      "AUC", auc_all)

aucs = [auc_all[int(l)] for l in by_contrast]
n_desc = sum(1 for l in by_contrast if descs.get(int(l)))
summary = {
    "width": W,
    "layer": cfg["layer"], "n_layers": cfg["n_layers"],
    "depth": round(cfg["layer"] / cfg["n_layers"], 3),
    "on_neuronpedia": bool(np_model),
    "explanations_available": len(descs),
    "explanation_coverage": round(len(descs) / W, 4) if descs else 0.0,
    "latents_firing_any_positive": int((docs_pos > 0).sum()),
    "frac_dict_firing": round(float((docs_pos > 0).sum()) / W, 4),
    "latents_cross_source": int(cross_source.sum()),
    "latents_passing_filters": int(len(cand)),
    "frac_dict_passing": round(len(cand) / W, 5),
    "shortlist_with_description": n_desc,
    "shortlist_size": len(by_contrast),
    "mean_auc_contrast_top": round(float(np.nanmean(aucs)), 4),
    "best_auc": round(float(auc_all[cand].max()), 4) if len(cand) else 0.0,
    "n_auc_perfect": int(((auc_all >= 0.999) & robust).sum()),
    "n_auc_over_95": int(((auc_all >= 0.95) & robust).sum()),
    "n_auc_over_90": int(((auc_all >= 0.90) & robust).sum()),
    "frac_dict_auc_over_95": round(float(((auc_all >= 0.95) & robust).sum()) / W, 5),
    "v3_category_counts": {c: int(((term_mean[c] > 0) & robust).sum()) for c in CATS},
}
print(f"\n{'='*80}\nSUMMARY  {args.sae}\n{'='*80}")
for k, v in summary.items():
    print(f"  {k:<30} {v}")

json.dump({
    "sae": args.sae, "neuronpedia": cfg["neuronpedia"], "summary": summary,
    "n_positive_docs": n_pos_docs, "n_pos_tokens": n_pos_tok, "n_ctl_tokens": n_ctl_tok,
    "ranked_by_contrast": [
        {"latent": int(l), "contrast": float(contrast[l]), "mean_pos": float(mean_pos[l]),
         "mean_ctl": float(mean_ctl[l]), "max_pos": float(max_pos[l]),
         "docs_firing": int(docs_pos[l]), "description": descs.get(int(l)),
         "v3_category": profile(int(l))[0], "v3_category_frac": profile(int(l))[1],
         "v3_by_category": {c: float(term_mean[c][l]) for c in CATS},
         "neuronpedia": url(l),
         "top_example": top_ex.get(int(l), (0, "", "", ""))[3]}
        for l in by_contrast],
    "ranked_by_activation": [
        {"latent": int(l), "mean_pos": float(mean_pos[l]), "contrast": float(contrast[l]),
         "neuronpedia": url(l)} for l in by_act],
    "ranked_by_auc": [
        {"latent": int(l), "auc": float(auc_all[l]), "contrast": float(contrast[l]),
         "mean_pos": float(mean_pos[l]), "mean_ctl": float(mean_ctl[l]),
         "docs_firing": int(docs_pos[l]), "description": descs.get(int(l)),
         "v3_category": profile(int(l))[0], "v3_category_frac": profile(int(l))[1],
         "neuronpedia": url(l),
         "top_example": top_ex.get(int(l), (0, "", "", ""))[3]}
        for l in by_auc],
}, open(out_path, "w"), indent=2)
print(f"\nwrote {out_path}")
