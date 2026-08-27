"""Paulo et al. (2024)-style embedding scoring of feature captions (2026-08-15).

For each candidate latent: embed its Neuronpedia caption and 8 activating
contexts (from the SAE's shipped examples.safetensors) plus 8 non-activating
contexts (random positions from other sequences); score = AUC of
caption-context cosine similarity separating activating from non-activating.
Rathi & Radford discard latents scoring < 0.9; we record the full score so the
threshold can be swept. Uncaptioned latents get score null (auto-fail).

Approximations vs the original: MiniLM-L6-v2 embedder (theirs unspecified),
AUC in place of their accuracy variant, contexts of +-20 tokens.

Usage: python embed_score_captions.py 16k|65k
"""
import json
import sys
import glob
import gzip

import numpy as np
import torch

ROOT = "/home/a5k/cwtice.a5k/sae-exploration"
sys.path.insert(0, f"{ROOT}/scripts")
from sae_local import sae_dir, snapshot_dir  # noqa: E402
from safetensors import safe_open  # noqa: E402
from transformers import AutoTokenizer, AutoModel  # noqa: E402

width = sys.argv[1]
SAE = {"16k": "gemma3-l40-16k", "65k": "gemma3-l40-65k"}[width]
NP_SOURCE = {"16k": "40-gemmascope-2-res-16k", "65k": "40-gemmascope-2-res-65k"}[width]
CAP_GLOB = {"16k": f"{ROOT}/out/np_expl_batch-*.jsonl.gz", "65k": f"{ROOT}/out/np65k_batch-*.jsonl.gz"}[width]
POOLS = {"16k": ["fable_16k", "haiku_v6_16k", "haiku_v8_16k"],
         "65k": ["haiku_v6_65k", "haiku_v8_65k"]}[width]
N_EX, CTX, SEED = 8, 20, 20260815

cand = json.load(open(f"{ROOT}/data/candidate_sets/haiku_fable_forget_latents_v1.json"))
latents = sorted({int(l) for p in POOLS for l, _c, _f in cand["sets"][p]["latents"]})
print(f"{width}: {len(latents)} unique latents across {POOLS}")

caps = {}
for fn in glob.glob(CAP_GLOB):
    for line in gzip.open(fn, "rt"):
        d = json.loads(line)
        if d.get("layer") == NP_SOURCE or width == "65k":
            idx = int(d.get("index", d.get("latent", -1)))
            if idx >= 0:
                caps[idx] = (d.get("description") or d.get("explanation") or "").strip()
print(f"captions loaded: {len(caps)}; coverage of pool: "
      f"{sum(1 for l in latents if caps.get(l))}/{len(latents)}")

tok = AutoTokenizer.from_pretrained(snapshot_dir("unsloth/gemma-3-27b-it"))
rng = np.random.default_rng(SEED)
pos_ctx, neg_ctx, order = {}, {}, []
with safe_open(f"{sae_dir(SAE)}/examples.safetensors", framework="np") as f:
    seq_t, pos_t, act_t = f.get_slice("seq_ids"), f.get_slice("positions"), f.get_slice("activations")
    toks_t = f.get_slice("tokens")
    n_seq = toks_t.get_shape()[0]
    for k, lat in enumerate(latents):
        if not caps.get(lat):
            continue
        sid_a, p_a, a_a = seq_t[lat], pos_t[lat], act_t[lat]
        ok = sid_a >= 0
        exs, seen = [], set()
        if ok.any():
            for i in np.flatnonzero(ok)[np.argsort(-a_a[ok])]:
                sid = int(sid_a[i])
                if sid in seen:
                    continue
                seen.add(sid)
                s = toks_t[sid]; p = int(p_a[i])
                lo, hi = max(0, p - CTX), min(len(s), p + CTX + 1)
                exs.append(np.asarray(s[lo:hi]))
                if len(exs) >= N_EX:
                    break
        if len(exs) < 4:
            continue
        negs = []
        for _ in range(N_EX):
            sid = int(rng.integers(n_seq))
            while sid in seen:
                sid = int(rng.integers(n_seq))
            s = toks_t[sid]
            p = int(rng.integers(CTX, max(CTX + 1, len(s) - CTX)))
            negs.append(np.asarray(s[max(0, p - CTX):p + CTX + 1]))
        pos_ctx[lat], neg_ctx[lat] = exs, negs
        order.append(lat)
        if (k + 1) % 500 == 0:
            print(f"  contexts {k+1}/{len(latents)}", flush=True)
print(f"scoreable: {len(order)}")

texts, spans = [], {}
for lat in order:
    s0 = len(texts)
    texts.append(caps[lat])
    texts.extend(tok.decode(e) for e in pos_ctx[lat])
    texts.extend(tok.decode(e) for e in neg_ctx[lat])
    spans[lat] = (s0, len(pos_ctx[lat]), len(neg_ctx[lat]))
print(f"embedding {len(texts)} texts")

dev = "cuda" if torch.cuda.is_available() else "cpu"
etok = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
emb_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").to(dev).eval()
embs = []
with torch.no_grad():
    for i in range(0, len(texts), 512):
        b = etok(texts[i:i + 512], padding=True, truncation=True, max_length=128,
                 return_tensors="pt").to(dev)
        out = emb_model(**b).last_hidden_state
        mask = b["attention_mask"].unsqueeze(-1)
        v = (out * mask).sum(1) / mask.sum(1)
        embs.append(torch.nn.functional.normalize(v, dim=-1).cpu())
        if i % 20480 == 0:
            print(f"  emb {i}/{len(texts)}", flush=True)
E = torch.cat(embs).numpy()

scores = {}
for lat in order:
    s0, npos, nneg = spans[lat]
    cap_e = E[s0]
    sims = E[s0 + 1: s0 + 1 + npos + nneg] @ cap_e
    ps, ns = sims[:npos], sims[npos:]
    # AUC
    auc = float(np.mean([(p > n) + 0.5 * (p == n) for p in ps for n in ns]))
    scores[lat] = round(auc, 4)
for lat in latents:
    if lat not in scores:
        scores[lat] = None
kept = sum(1 for l in latents if scores[l] is not None and scores[l] >= 0.9)
print(f"score>=0.9: {kept}/{len(latents)} | uncaptioned/unscoreable: "
      f"{sum(1 for l in latents if scores[l] is None)}")
json.dump(scores, open(f"{ROOT}/out/embed_scores_{width}.json", "w"))
print(f"wrote out/embed_scores_{width}.json")
