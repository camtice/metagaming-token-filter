"""Export a DOCUMENT-first view of the probe set: for each of the 73 probe
docs, which SAE latents fire on it and how hard.

This is the inverse of export_findings.py (latent-first). It exists so an
artifact can browse the dataset -- especially the deception_metagaming axis --
and drill from a document into the latents that fire on it.

PRIVACY: the probe texts are sensitive and are NEVER written to the output.
Documents are represented by their metadata (id, source, category, axis),
token counts, and a pointer to the line in data/probe_dataset.jsonl. The only
text exported is Google's shipped autointerp examples (examples.safetensors,
a generic public web corpus) -- the evidence behind each Neuronpedia label.

Per document we keep:
  * top latents by max activation (any latent) + every shortlist latent that
    fires, each as [latent, mean_act, max_act, frac_tokens_firing, peak_token]
    where peak_token is the body-token INDEX of the strongest activation
    (a position, not a token -- look it up in the jsonl yourself).
Per latent in the resulting universe we keep:
  * Neuronpedia description + explainer + link, global probe stats
    (AUC, deception specificity, axis means, docs firing), corpus firing rate,
    and the raw autointerp examples (Gemma Scope 2 only).

Scoring is identical to export_findings.py: BOS + context prepended, only
body tokens scored, doc score = mean over body tokens.

Usage: python export_doc_view.py gemma3-l40-16k [--topk 20]
Output: out/doc_view_<sae>.json
"""
import argparse
import glob
import gzip
import json
import sys

import numpy as np
import torch

sys.path.insert(0, "/home/a5k/cwtice.a5k/sae-exploration/scripts")
from sae_local import REGISTRY, Runner, sae_dir  # noqa: E402

ROOT = "/home/a5k/cwtice.a5k/sae-exploration"
NP_PREFIX = {  # which label dump belongs to which SAE (avoid scanning all)
    "gemma3-l40-16k": "np_expl_",
    "gemma3-l40-65k": "np65k_",
    "gemma2-l31-16k": "npg2_",
    "olmo3-l32-131k": "npolmo_",
}
AXES = ["deception_metagaming", "metagaming_clean", "deception_generic", "control_other"]

ap = argparse.ArgumentParser()
ap.add_argument("sae")
ap.add_argument("--topk", type=int, default=20, help="per-doc top latents by max act")
ap.add_argument("--topk-mean", type=int, default=10, help="per-doc top latents by mean act")
ap.add_argument("--shortlist", type=int, default=40, help="top-N by AUC and by dec-spec")
ap.add_argument("--raw-examples", type=int, default=5)
ap.add_argument("--max-len", type=int, default=512)
ap.add_argument("--max-fires", type=int, default=40,
                help="cap on stored NON-shortlist latents per doc (shortlist latents "
                     "that fire are always stored, so the doc x shortlist matrix is exact)")
args = ap.parse_args()

cfg = REGISTRY[args.sae]
W = cfg["width"]
rows = [json.loads(l) for l in open(f"{ROOT}/data/probe_dataset.jsonl")]
runner = Runner(args.sae)
tok = runner.tokenizer
bos = tok.bos_token_id
np_model, np_source = cfg["neuronpedia"] or (None, None)


@torch.no_grad()
def encode(row):
    ctx = []
    if row.get("context"):
        ctx = tok(row["context"], add_special_tokens=False, truncation=True,
                  max_length=args.max_len // 2)["input_ids"]
    body = tok(row["text"], add_special_tokens=False, truncation=True,
               max_length=args.max_len)["input_ids"]
    ids = ([bos] if bos is not None else []) + ctx + body
    ids = ids[-args.max_len:]
    n = min(len(body), len(ids))
    acts = runner.sae.encode(runner.residual(torch.tensor(ids)[None]))[0]
    return acts[-n:].float().cpu().numpy()


mean_m = np.zeros((len(rows), W), dtype=np.float32)
max_m = np.zeros((len(rows), W), dtype=np.float32)
frac_m = np.zeros((len(rows), W), dtype=np.float32)
peak_m = np.zeros((len(rows), W), dtype=np.int32)
n_tok = np.zeros(len(rows), dtype=np.int32)
src_fire = {}
for i, row in enumerate(rows):
    acts = encode(row)
    n_tok[i] = acts.shape[0]
    mean_m[i] = acts.mean(0)
    max_m[i] = acts.max(0)
    frac_m[i] = (acts > 0).mean(0)
    peak_m[i] = acts.argmax(0)
    grp = "sa" if row["source"] == "sa_oversight" else ("ctl" if row["label"] == "control"
                                                       else "schoen")
    src_fire.setdefault(grp, np.zeros(W, bool))
    src_fire[grp] |= max_m[i] > 0
    print(".", end="", flush=True)
print()

# ---- global per-latent stats (same definitions as export_findings.py) ------
axis_idx = {a: [i for i, r in enumerate(rows) if r["axis"] == a] for a in AXES}
pos = mean_m[axis_idx["deception_metagaming"] + axis_idx["metagaming_clean"]]
ctl = mean_m[axis_idx["deception_generic"] + axis_idx["control_other"]]
mean_pos, mean_ctl = pos.mean(0), ctl.mean(0)
eps = np.percentile(mean_pos[mean_pos > 0], 25)
docs_fire = (max_m > 0).sum(0)

auc = np.zeros(W)
for lo in range(0, W, 4096):
    hi = min(lo + 4096, W)
    a, b = pos[:, lo:hi][:, None, :], ctl[:, lo:hi][None, :, :]
    auc[lo:hi] = ((a > b).sum((0, 1)) + 0.5 * (a == b).sum((0, 1))) / (pos.shape[0] * ctl.shape[0])

axis_mean = {a: mean_m[axis_idx[a]].mean(0) for a in AXES}
dec_spec = axis_mean["deception_metagaming"] / (axis_mean["deception_generic"] + eps)

robust = (docs_fire >= 6) & src_fire.get("schoen", np.zeros(W, bool))
cand = np.flatnonzero(robust)
sel = cand[np.argsort(-(auc[cand] + 1e-6 * (mean_pos[cand] / (mean_ctl[cand] + eps))))][:args.shortlist]
dec_sel = cand[np.argsort(-dec_spec[cand])][:args.shortlist]
shortlist = set(int(x) for x in sel) | set(int(x) for x in dec_sel)

# ---- latent universe: shortlist + per-doc top-K ----------------------------
universe = set(shortlist)
for i in range(len(rows)):
    order = np.argsort(-max_m[i])[:args.topk]
    universe |= set(int(l) for l in order if max_m[i][l] > 0)
    order = np.argsort(-mean_m[i])[:args.topk_mean]
    universe |= set(int(l) for l in order if mean_m[i][l] > 0)
print(f"universe: {len(universe)} latents ({len(shortlist)} shortlist)")

# ---- Neuronpedia labels ----------------------------------------------------
descs = {}
if np_source:
    prefix = NP_PREFIX.get(args.sae, "np")
    for fn in glob.glob(f"{ROOT}/out/{prefix}batch-*.jsonl.gz"):
        for line in gzip.open(fn, "rt"):
            d = json.loads(line)
            if d.get("layer") == np_source and int(d["index"]) in universe:
                descs[int(d["index"])] = {
                    "description": d["description"],
                    "explainer": d.get("explanationModelName"),
                }
print(f"labels: {len(descs)}/{len(universe)}")

# ---- raw autointerp examples (Gemma Scope 2 ships them) --------------------
raw = {}
if cfg["arch"] == "jumprelu_safetensors":
    from safetensors import safe_open
    with safe_open(f"{sae_dir(args.sae)}/examples.safetensors", framework="np") as f:
        seq_t, pos_t, act_t = (f.get_slice("seq_ids"), f.get_slice("positions"),
                               f.get_slice("activations"))
        toks_t, freq = f.get_slice("tokens"), f.get_slice("feature_frequencies")
        for lat in sorted(universe):
            sid_a, p_a, a_a = seq_t[lat], pos_t[lat], act_t[lat]
            ok = sid_a >= 0
            if not ok.any():
                raw[lat] = {"firing_rate": float(freq[lat]), "examples": []}
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

# ---- emit ------------------------------------------------------------------
docs = []
for i, row in enumerate(rows):
    short_fires, other_fires = [], []
    for lat in universe:
        if max_m[i][lat] <= 0:
            continue
        entry = [int(lat), round(float(mean_m[i][lat]), 2),
                 round(float(max_m[i][lat]), 1),
                 round(float(frac_m[i][lat]), 3), int(peak_m[i][lat])]
        (short_fires if lat in shortlist else other_fires).append(entry)
    other_fires.sort(key=lambda x: -x[2])
    fires = short_fires + other_fires[:args.max_fires]
    fires.sort(key=lambda x: -x[2])
    docs.append({
        "i": i, "id": row["id"], "line": i + 1,
        "source": row["source"], "model": row.get("model"), "env": row.get("env"),
        "category": row["category"], "axis": row["axis"], "label": row["label"],
        "n_tokens": int(n_tok[i]), "n_chars": len(row["text"]),
        "has_context": bool(row.get("context")),
        "fires": fires,
        "n_latents_firing": int((max_m[i] > 0).sum()),
    })

latents = {}
for lat in sorted(universe):
    d = descs.get(lat, {})
    latents[str(lat)] = {
        "auc": round(float(auc[lat]), 4),
        "dec_spec": round(float(dec_spec[lat]), 2),
        "docs_firing": int(docs_fire[lat]),
        "mean_pos": round(float(mean_pos[lat]), 2),
        "mean_ctl": round(float(mean_ctl[lat]), 2),
        "axis_means": {a: round(float(axis_mean[a][lat]), 2) for a in AXES},
        "fires_on_generic_deception": bool(axis_mean["deception_generic"][lat] > 0),
        "in_sep_shortlist": bool(lat in set(int(x) for x in sel)),
        "in_dec_shortlist": bool(lat in set(int(x) for x in dec_sel)),
        "description": d.get("description"),
        "explainer": d.get("explainer"),
        "neuronpedia": (f"https://www.neuronpedia.org/{np_model}/{np_source}/{lat}"
                        if np_model else None),
        "firing_rate": raw.get(lat, {}).get("firing_rate"),
        "raw_examples": raw.get(lat, {}).get("examples", []),
    }

out = {
    "sae": args.sae, "model": cfg["model"], "layer": cfg["layer"],
    "n_layers": cfg["n_layers"], "width": W,
    "neuronpedia_model": np_model, "neuronpedia_source": np_source,
    "dataset_file": "data/probe_dataset.jsonl",
    "axis_counts": {a: len(axis_idx[a]) for a in AXES},
    "docs": docs,
    "latents": latents,
}
p = f"{ROOT}/out/doc_view_{args.sae}.json"
json.dump(out, open(p, "w"), indent=None)
print(f"wrote {p}")
print(f"  docs: {len(docs)}, universe latents: {len(latents)}, "
      f"with raw examples: {sum(1 for v in latents.values() if v['raw_examples'])}")
