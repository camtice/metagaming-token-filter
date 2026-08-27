"""Is the gemma-3 +4.4% bias numeric precision, or a real wiring difference?

Compares bf16 (single GPU) against fp32 (sharded across this node's 4 GH200s)
on an identical set of reference points. If the bias is bf16 rounding it should
largely vanish in fp32. If it survives fp32, the residual capture point or the
mirror weights genuinely differ from what Google used.

fp16 is not tested: gemma-3's residual stream overflows its 65504 max (the first
attempt returned nan throughout).

One Gemma-specific precision trap this also quantifies: HF scales embeddings by
`torch.tensor(hidden_size**0.5, dtype=inputs_embeds.dtype)`, and sqrt(5376) =
73.3212 rounds to 73.5 in bf16 -- a +0.244% scale applied to the whole residual
stream before layer 0.
"""
import sys

import numpy as np
import torch
from safetensors import safe_open

sys.path.insert(0, "/home/a5k/cwtice.a5k/sae-exploration/scripts")
from sae_local import REGISTRY, Runner, sae_dir  # noqa: E402

NAME = "gemma3-l46-16k"
cfg = REGISTRY[NAME]

with safe_open(f"{sae_dir(NAME)}/examples.safetensors", framework="np") as f:
    freqs = f.get_slice("feature_frequencies")[:]
    seq_t, pos_t, act_t = (f.get_slice("seq_ids"), f.get_slice("positions"),
                           f.get_slice("activations"))
    toks_t = f.get_slice("tokens")
    order = np.argsort(-freqs)
    cases = []
    for latent in [int(x) for x in order[len(order)//20::len(order)//40][:5]]:
        seq_ids, positions, acts = seq_t[latent], pos_t[latent], act_t[latent]
        valid = seq_ids >= 0
        if not valid.any():
            continue
        sid = int(seq_ids[np.flatnonzero(valid)[np.argmax(acts[valid])]])
        same = [j for j in np.flatnonzero(valid) if int(seq_ids[j]) == sid]
        cases.append((latent, np.asarray(toks_t[sid], dtype=np.int64),
                      {int(positions[j]): float(acts[j]) for j in same}))

print(f"{len(cases)} latent/sequence cases\n")
print(f"{'config':>22} {'r':>8} {'med ratio':>10} {'med |rel err|':>14} {'top-q ratio':>12}")

for label, kwargs in [("bf16 1xGPU", dict(model_dtype=torch.bfloat16)),
                      ("fp32 4xGPU", dict(model_dtype=torch.float32, device_map="auto"))]:
    runner = None
    try:
        runner = Runner(NAME, **kwargs)
        exp_all, mine_all = [], []
        for latent, ids, ref in cases:
            t = torch.tensor(ids)[None]
            got = runner.sae.encode(runner.residual(t))[0, :, latent].float().cpu().numpy()
            for q, v in ref.items():
                if q < len(got):
                    exp_all.append(v)
                    mine_all.append(float(got[q]))
        exp, mine = np.array(exp_all), np.array(mine_all)
        ok = exp > 0
        topq = exp >= np.quantile(exp, 0.75)
        print(f"{label:>22} {np.corrcoef(mine, exp)[0,1]:>8.5f} "
              f"{np.median(mine[ok]/exp[ok]):>10.4f} "
              f"{np.median(np.abs(mine[ok]-exp[ok])/exp[ok]):>13.1%} "
              f"{np.median(mine[topq]/exp[topq]):>12.4f}")
    except Exception as e:
        print(f"{label:>22}  ERROR {type(e).__name__}: {str(e)[:70]}")
    finally:
        del runner
        torch.cuda.empty_cache()

emb_scale = cfg["d_model"] ** 0.5
print(f"\nembedding normalizer sqrt({cfg['d_model']}) = {emb_scale:.6f}")
for dn, dt in [("bfloat16", torch.bfloat16), ("float32", torch.float32)]:
    r = float(torch.tensor(emb_scale, dtype=dt))
    print(f"  as {dn}: {r:.6f} -> {(r/emb_scale - 1)*100:+.3f}% scale on the residual stream")
