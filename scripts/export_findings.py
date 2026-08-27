"""Export everything needed to inspect metagaming latents, in one JSON.

For each shortlisted latent of an SAE this collects:
  * how it scores on the probe set  (AUC, contrast, docs firing, per-axis means)
  * its published Neuronpedia label + a link
  * the RAW ACTIVATING EXAMPLES the autointerp label was derived from -- for
    Gemma Scope 2 these ship in `examples.safetensors` (1000 per latent over a
    234435x512 token corpus), which is literally the evidence the explainer model
    saw. That lets a reader judge whether the published label is right.
  * the snippets from OUR probe set that made it fire, per deception axis

Deception axis (the thing to compare):
  deception_metagaming  an AI reasoning about fudging/sandbagging/misreporting to
                        a grader or overseer
  metagaming_clean      metagaming with no deception content
  deception_generic     control: spy fiction, VPNs -- deception, no AI oversight

A latent that fires on deception_metagaming but NOT deception_generic has learned
the oversight-specific sense; one that fires on both has only learned "deception".
That separation is what decides whether it belongs in a forget set.

Usage: python export_findings.py gemma3-l40-16k [--top 40]
Output: out/findings_<sae>.json
"""
import argparse
import glob
import gzip
import json
import sys

import numpy as np
import torch

sys.path.insert(0, "/home/a5k/cwtice.a5k/sae-exploration/scripts")
from regex_v3 import COMPILED, char_mask  # noqa: E402
from sae_local import REGISTRY, Runner, sae_dir  # noqa: E402

ROOT = "/home/a5k/cwtice.a5k/sae-exploration"
ap = argparse.ArgumentParser()
ap.add_argument("sae")
ap.add_argument("--top", type=int, default=40)
ap.add_argument("--max-len", type=int, default=512)
ap.add_argument("--raw-examples", type=int, default=6)
args = ap.parse_args()

cfg = REGISTRY[args.sae]
rows = [json.loads(l) for l in open(f"{ROOT}/data/probe_dataset.jsonl")]
runner = Runner(args.sae)
tok, W = runner.tokenizer, cfg["width"]
bos = tok.bos_token_id
np_model, np_source = cfg["neuronpedia"] or (None, None)
CATS = list(COMPILED)

# ---- published labels ------------------------------------------------------
descs = {}
if np_source:
    for fn in glob.glob(f"{ROOT}/out/np*batch-*.jsonl.gz"):
        for line in gzip.open(fn, "rt"):
            d = json.loads(line)
            if d.get("layer") == np_source:
                descs[int(d["index"])] = {
                    "description": d["description"],
                    "explainer": d.get("explanationModelName"),
                }

AXES = ["deception_metagaming", "metagaming_clean", "deception_generic", "control_other"]
axis_docs = {a: [] for a in AXES}
snippets = {}          # latent -> list of {axis, doc, act, text}
docs_fire = np.zeros(W, dtype=np.int32)
term_sum = {c: np.zeros(W) for c in CATS}
term_tok = {c: 0 for c in CATS}
src_fire = {}


@torch.no_grad()
def encode(row):
    ctx = []
    if row.get("context"):
        ctx = tok(row["context"], add_special_tokens=False, truncation=True,
                  max_length=args.max_len // 2)["input_ids"]
    enc = tok(row["text"], add_special_tokens=False, truncation=True,
              max_length=args.max_len, return_offsets_mapping=True)
    body, offs = enc["input_ids"], enc["offset_mapping"]
    ids = ([bos] if bos is not None else []) + ctx + body
    ids = ids[-args.max_len:]
    n = min(len(body), len(ids))
    acts = runner.sae.encode(runner.residual(torch.tensor(ids)[None]))[0]
    return acts[-n:].float().cpu().numpy(), ids[-n:], offs[-n:]


for row in rows:
    acts, ids, offs = encode(row)
    axis_docs[row["axis"]].append(acts.mean(0))
    docs_fire += (acts.max(0) > 0).astype(np.int32)
    grp = "sa" if row["source"] == "sa_oversight" else ("ctl" if row["label"] == "control"
                                                       else "schoen")
    src_fire.setdefault(grp, np.zeros(W, bool))
    src_fire[grp] |= acts.max(0) > 0
    for c in CATS:
        cm = char_mask(row["text"], categories=[c])
        m = np.array([any(cm[s:e]) if e > s else False for s, e in offs])
        if m.any():
            term_sum[c] += acts[m].sum(0)
            term_tok[c] += int(m.sum())
    # keep the strongest snippet per latent per document
    peak, val = acts.argmax(0), acts.max(0)
    for lat in np.flatnonzero(val > 0):
        p = int(peak[lat])
        lo, hi = max(0, p - 14), min(len(ids), p + 15)
        snippets.setdefault(int(lat), []).append({
            "axis": row["axis"], "doc": row["id"], "act": round(float(val[lat]), 1),
            "text": tok.decode(ids[lo:p]) + "«" + tok.decode([ids[p]]) + "»"
                    + tok.decode(ids[p + 1:hi])})
    print(".", end="", flush=True)
print()

M = {a: (np.stack(v) if v else np.zeros((0, W))) for a, v in axis_docs.items()}
pos = np.concatenate([M["deception_metagaming"], M["metagaming_clean"]])
ctl = np.concatenate([M["deception_generic"], M["control_other"]])
mean_pos, mean_ctl = pos.mean(0), ctl.mean(0)
eps = np.percentile(mean_pos[mean_pos > 0], 25)

auc = np.zeros(W)
for lo in range(0, W, 4096):
    hi = min(lo + 4096, W)
    a, b = pos[:, lo:hi][:, None, :], ctl[:, lo:hi][None, :, :]
    auc[lo:hi] = ((a > b).sum((0, 1)) + 0.5 * (a == b).sum((0, 1))) / (pos.shape[0] * ctl.shape[0])

axis_mean = {a: (M[a].mean(0) if len(M[a]) else np.zeros(W)) for a in AXES}
# the headline for this export: fires on AI-deception, not on generic deception
dec_spec = axis_mean["deception_metagaming"] / (axis_mean["deception_generic"] + eps)

robust = (docs_fire >= 6) & src_fire.get("schoen", np.zeros(W, bool))
cand = np.flatnonzero(robust)
sel = cand[np.argsort(-(auc[cand] + 1e-6 * (mean_pos[cand] / (mean_ctl[cand] + eps))))][:args.top]
dec_sel = cand[np.argsort(-dec_spec[cand])][:args.top]

# ---- raw autointerp source: Google's shipped activating examples ------------
raw = {}
if cfg["arch"] == "jumprelu_safetensors":
    from safetensors import safe_open
    want = set(int(x) for x in sel) | set(int(x) for x in dec_sel)
    with safe_open(f"{sae_dir(args.sae)}/examples.safetensors", framework="np") as f:
        seq_t, pos_t, act_t = (f.get_slice("seq_ids"), f.get_slice("positions"),
                               f.get_slice("activations"))
        toks_t, freq = f.get_slice("tokens"), f.get_slice("feature_frequencies")
        for lat in want:
            sid_a, p_a, a_a = seq_t[lat], pos_t[lat], act_t[lat]
            ok = sid_a >= 0
            if not ok.any():
                continue
            ex, seen = [], set()
            for i in np.flatnonzero(ok)[np.argsort(-a_a[ok])]:
                sid = int(sid_a[i])
                if sid in seen:
                    continue
                seen.add(sid)
                s = toks_t[sid]
                p = int(p_a[i])
                lo, hi = max(0, p - 28), min(len(s), p + 29)
                ex.append({"act": round(float(a_a[i]), 1),
                           "text": tok.decode(s[lo:p]) + "«" + tok.decode(s[p:p + 1])
                                   + "»" + tok.decode(s[p + 1:hi])})
                if len(ex) >= args.raw_examples:
                    break
            raw[lat] = {"firing_rate": float(freq[lat]), "examples": ex}


def pack(lat):
    lat = int(lat)
    sn = sorted(snippets.get(lat, []), key=lambda s: -s["act"])
    by_axis = {}
    for s in sn:
        by_axis.setdefault(s["axis"], []).append(s)
    d = descs.get(lat, {})
    tot = sum(term_sum[c][lat] / max(term_tok[c], 1) for c in CATS)
    v3 = ({c.split("_")[0]: round(float(term_sum[c][lat] / max(term_tok[c], 1) / tot), 3)
           for c in CATS} if tot > 0 else {})
    return {
        "latent": lat,
        "auc": round(float(auc[lat]), 4),
        "docs_firing": int(docs_fire[lat]),
        "mean_pos": round(float(mean_pos[lat]), 2),
        "mean_ctl": round(float(mean_ctl[lat]), 2),
        "axis_means": {a: round(float(axis_mean[a][lat]), 2) for a in AXES},
        "deception_specificity": round(float(dec_spec[lat]), 2),
        "fires_on_generic_deception": bool(axis_mean["deception_generic"][lat] > 0),
        "description": d.get("description"),
        "explainer_model": d.get("explainer"),
        "neuronpedia": f"https://www.neuronpedia.org/{np_model}/{np_source}/{lat}"
                       if np_model else None,
        "v3_profile": v3,
        "raw_autointerp_examples": raw.get(lat, {}).get("examples", []),
        "sae_firing_rate": raw.get(lat, {}).get("firing_rate"),
        "probe_snippets": {a: by_axis.get(a, [])[:4] for a in AXES if a in by_axis},
    }


out = {
    "sae": args.sae,
    "model": cfg["model"], "layer": cfg["layer"], "n_layers": cfg["n_layers"],
    "width": W, "neuronpedia_source": np_source, "neuronpedia_model": np_model,
    "dataset": {
        "n_docs": len(rows),
        "axis_counts": {a: len(v) for a, v in axis_docs.items()},
    },
    "raw_examples_note": ("raw_autointerp_examples are Google's shipped activating "
                          "examples from examples.safetensors -- the evidence the "
                          "autointerp explainer saw. Empty for non-gemmascope-2 SAEs."),
    "top_by_separation": [pack(l) for l in sel],
    "top_by_deception_specificity": [pack(l) for l in dec_sel],
}
p = f"{ROOT}/out/findings_{args.sae}.json"
json.dump(out, open(p, "w"), indent=1)
print(f"wrote {p}")
print(f"  axis counts: {out['dataset']['axis_counts']}")
print(f"  top separation AUC: {out['top_by_separation'][0]['auc']}")
print(f"  latents with raw examples: {sum(1 for r in out['top_by_separation'] if r['raw_autointerp_examples'])}"
      f"/{len(out['top_by_separation'])}")
