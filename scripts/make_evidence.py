"""Build per-latent evidence packets for candidate screening.

For each candidate latent: description, firing rate, top promoted tokens, and
top-K distinct-sequence activating examples (±W tokens) from Google's shipped
examples.safetensors. Output: out/evidence.jsonl
"""
import glob
import json

import numpy as np
from safetensors import safe_open
from transformers import AutoTokenizer

OUT = "/home/a5k/cwtice.a5k/sae-exploration/out"
SAE_GLOB = ("/projects/a5k/public/data_cwtice.a5k/.cache/huggingface/hub/"
            "models--google--gemma-scope-2-27b-it/snapshots/*/resid_post/"
            "layer_40_width_16k_l0_medium")
N_EXAMPLES = 8
WINDOW = 32

candidates = [json.loads(l) for l in open(f"{OUT}/candidates.jsonl")]
tok = AutoTokenizer.from_pretrained("unsloth/gemma-3-27b-it")
sae_dir = glob.glob(SAE_GLOB)[0]

with safe_open(f"{sae_dir}/examples.safetensors", framework="np") as f, \
     open(f"{OUT}/evidence.jsonl", "w") as out:
    freqs = f.get_slice("feature_frequencies")
    seq_ids_t = f.get_slice("seq_ids")
    positions_t = f.get_slice("positions")
    acts_t = f.get_slice("activations")
    tokens_t = f.get_slice("tokens")
    top_tokens_t = f.get_slice("top_tokens")

    for cand in candidates:
        latent = cand["index"]
        seq_ids, positions, acts = seq_ids_t[latent], positions_t[latent], acts_t[latent]
        valid = seq_ids >= 0
        examples = []
        if valid.any():
            order = np.argsort(-acts[valid])
            seen = set()
            for i in np.flatnonzero(valid)[order]:
                sid, pos, act = int(seq_ids[i]), int(positions[i]), float(acts[i])
                if sid in seen:
                    continue
                seen.add(sid)
                seq = tokens_t[sid]
                lo, hi = max(0, pos - WINDOW), min(len(seq), pos + WINDOW + 1)
                text = (tok.decode(seq[lo:pos]) + "«" + tok.decode(seq[pos:pos + 1])
                        + "»" + tok.decode(seq[pos + 1:hi]))
                examples.append({"act": round(act, 1), "text": text})
                if len(examples) >= N_EXAMPLES:
                    break
        cand["firing_rate"] = float(freqs[latent])
        cand["top_promoted_tokens"] = [tok.decode([t]) for t in top_tokens_t[latent] if t >= 0]
        cand["examples"] = examples
        out.write(json.dumps(cand) + "\n")
        print(latent, end=" ", flush=True)
print("\ndone")
