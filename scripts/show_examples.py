"""Decode Google's shipped activating examples (examples.safetensors) for given latents.

Login-node friendly: numpy only, no torch/GPU. Reads per-latent top activating
(seq, pos, activation) triples and renders ±WINDOW-token windows with the
activating token highlighted.

Usage: python show_examples.py LATENT [LATENT ...] [--n 12] [--window 32]
"""
import argparse
import glob
import sys

import numpy as np
from safetensors import safe_open
from transformers import AutoTokenizer

SAE_GLOB = ("/projects/a5k/public/data_cwtice.a5k/.cache/huggingface/hub/"
            "models--google--gemma-scope-2-27b-it/snapshots/*/resid_post/"
            "layer_40_width_16k_l0_medium")
TOKENIZER = "unsloth/gemma-3-27b-it"

parser = argparse.ArgumentParser()
parser.add_argument("latents", type=int, nargs="+")
parser.add_argument("--n", type=int, default=12, help="examples per latent")
parser.add_argument("--window", type=int, default=32, help="tokens each side")
args = parser.parse_args()

sae_dir = glob.glob(SAE_GLOB)[0]
tok = AutoTokenizer.from_pretrained(TOKENIZER)

with safe_open(f"{sae_dir}/examples.safetensors", framework="np") as f:
    freqs = f.get_slice("feature_frequencies")
    seq_ids_t = f.get_slice("seq_ids")
    positions_t = f.get_slice("positions")
    acts_t = f.get_slice("activations")
    tokens_t = f.get_slice("tokens")
    top_logits_t = f.get_slice("top_tokens")

    for latent in args.latents:
        seq_ids = seq_ids_t[latent]
        positions = positions_t[latent]
        acts = acts_t[latent]
        freq = freqs[latent]
        top_toks = [tok.decode([t]) for t in top_logits_t[latent] if t >= 0]

        valid = seq_ids >= 0
        order = np.argsort(-acts[valid])
        print(f"\n{'='*100}")
        print(f"LATENT {latent}  fires on 1/{1/freq:.0f} tokens  "
              f"max_act={acts[valid].max():.1f}  n_examples={valid.sum()}")
        print(f"  top promoted tokens: {top_toks}")

        seen_seqs = set()
        shown = 0
        for i in np.flatnonzero(valid)[order]:
            sid, pos, act = int(seq_ids[i]), int(positions[i]), float(acts[i])
            if sid in seen_seqs:
                continue
            seen_seqs.add(sid)
            seq = tokens_t[sid]
            lo, hi = max(0, pos - args.window), min(len(seq), pos + args.window + 1)
            pre = tok.decode(seq[lo:pos], skip_special_tokens=False)
            mid = tok.decode(seq[pos:pos + 1], skip_special_tokens=False)
            post = tok.decode(seq[pos + 1:hi], skip_special_tokens=False)
            text = (pre + " «" + mid + "» " + post).replace("\n", "⏎")
            print(f"  [{act:6.1f}] …{text}…")
            shown += 1
            if shown >= args.n:
                break
