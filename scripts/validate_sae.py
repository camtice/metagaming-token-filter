"""Validate local SAE inference against published ground-truth activations.

Two independent ground-truth sources:

  * Gemma Scope 2 (gemma-3) ships `examples.safetensors` with each SAE: 1000
    (seq_id, position, activation) triples per latent over a 234435x512 token
    corpus. Fully self-contained -- no network, exact token ids.
  * Gemma Scope 1 (gemma-2) and the Olmo SAE have no shipped examples, so we use
    Neuronpedia's stored activations, mapping its SentencePiece token strings
    back to ids with `convert_tokens_to_ids`.

Nothing about the wiring is assumed. Two conventions are *measured* by sweeping
a grid and reporting which setting reproduces the reference:

  layer offset : Gemma Scope names its hook `model.layers.<L>.output`, but
                 Rathi's code reads `hidden_states[L]` (= output of layer L-1).
                 Offsets -1/0/+1 relative to `layers[L]` output are all scored.
  bos          : Neuronpedia windows are mid-document chunks with no BOS shown;
                 Gemma SAEs are normally run with BOS prepended. Both are scored.

A configuration is judged by Pearson r and median relative error at positions
where the reference is nonzero. r > 0.99 and error < 2% means the SAE and hook
point are correct; a near-1.0 r under exactly one offset is what identifies the
right convention.

Usage:  python validate_sae.py gemma2-l31-16k [--n-latents 8]
"""
import argparse
import json
import sys

import numpy as np
import torch

sys.path.insert(0, "/home/a5k/cwtice.a5k/sae-exploration/scripts")
from sae_local import REGISTRY, Runner, sae_dir  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("sae", nargs="+", help="one or more SAEs; those sharing a base model reuse it")
p.add_argument("--n-latents", type=int, default=8)
p.add_argument("--n-seqs", type=int, default=2, help="sequences per latent")
p.add_argument("--offsets", type=int, nargs="+", default=[-1, 0, 1])
p.add_argument("--max-len", type=int, default=512)
args = p.parse_args()

def ground_truth_shipped(sae_name):
    """(latent, token_ids, {pos: expected_act}) from examples.safetensors."""
    from safetensors import safe_open
    path = f"{sae_dir(sae_name)}/examples.safetensors"
    out = []
    with safe_open(path, framework="np") as f:
        acts_t, seq_t, pos_t = (f.get_slice("activations"), f.get_slice("seq_ids"),
                                f.get_slice("positions"))
        toks_t = f.get_slice("tokens")
        freqs = f.get_slice("feature_frequencies")[:]
        # mid-frequency latents: avoid dead ones and avoid ultra-dense outliers
        order = np.argsort(-freqs)
        chosen = order[len(order) // 20::max(1, len(order) // (args.n_latents * 20))][:args.n_latents]
        for latent in chosen:
            latent = int(latent)
            seq_ids, positions, acts = seq_t[latent], pos_t[latent], acts_t[latent]
            valid = seq_ids >= 0
            if not valid.any():
                continue
            top = np.flatnonzero(valid)[np.argsort(-acts[valid])]
            seen = set()
            for i in top:
                sid = int(seq_ids[i])
                if sid in seen:
                    continue
                seen.add(sid)
                # every reference activation this latent has on this sequence
                same = [j for j in np.flatnonzero(valid) if int(seq_ids[j]) == sid]
                ref = {int(positions[j]): float(acts[j]) for j in same}
                out.append((latent, np.asarray(toks_t[sid], dtype=np.int64), ref))
                if len(seen) >= args.n_seqs:
                    break
    return out


def ground_truth_neuronpedia(tokenizer, cfg):
    """(latent, token_ids, {pos: expected_act}) from Neuronpedia's API."""
    import requests
    model_id, source = cfg["neuronpedia"]
    out = []
    latent = 0
    tried = 0
    while len([o for o in out]) < args.n_latents * args.n_seqs and tried < args.n_latents * 8:
        tried += 1
        latent += 137  # stride to sample across the dictionary
        r = requests.get(
            f"https://www.neuronpedia.org/api/feature/{model_id}/{source}/{latent}", timeout=60)
        if r.status_code != 200:
            print(f"  latent {latent}: HTTP {r.status_code}")
            continue
        acts = r.json().get("activations") or []
        acts = [a for a in acts if a.get("maxValue", 0) > 0][:args.n_seqs]
        for a in acts:
            ids = _tokens_to_ids(tokenizer, a["tokens"])
            if ids is None:
                continue
            ref = {i: float(v) for i, v in enumerate(a["values"]) if v > 0}
            if ref:
                out.append((latent, np.asarray(ids, dtype=np.int64), ref))
    return out


def _tokens_to_ids(tokenizer, tokens):
    """Neuronpedia stores raw vocab pieces for some models ('__or' for Gemma's
    SentencePiece) and decoded strings for others (' Room' for Olmo's BPE).
    Try the vocab lookup first, then fall back to encoding each piece alone."""
    ids = tokenizer.convert_tokens_to_ids(tokens)
    if not any(i is None or i == tokenizer.unk_token_id for i in ids):
        return ids
    out = []
    for tok in tokens:
        enc = tokenizer.encode(tok, add_special_tokens=False)
        if len(enc) != 1:
            return None
        out.append(enc[0])
    return out


def validate(runner, name):
    cfg = REGISTRY[name]
    print(f"\n=== validating {name} ===")
    print(f"model={cfg['model']} layer={cfg['layer']}/{cfg['n_layers']} "
          f"(depth {cfg['layer']/cfg['n_layers']:.3f}) width={cfg['width']} arch={cfg['arch']}")

    use_shipped = cfg["arch"] == "jumprelu_safetensors"
    cases = (ground_truth_shipped(name) if use_shipped
             else ground_truth_neuronpedia(runner.tokenizer, cfg))
    print(f"ground truth: {'shipped examples.safetensors' if use_shipped else 'neuronpedia API'}"
          f"  ({len(cases)} latent/sequence cases)")
    if not cases:
        print("no ground-truth cases retrieved")
        return

    bos_id = runner.tokenizer.bos_token_id
    # Olmo's tokenizer has no BOS, and gemma-3's shipped sequences already carry one.
    bos_modes = ["asis"] if (use_shipped or bos_id is None) else ["asis", "prepend"]
    results = {}

    for offset in args.offsets:
        layer = cfg["layer"] + offset
        if not (0 <= layer < cfg["n_layers"]):
            continue
        for bos in bos_modes:
            rs, errs = [], []
            for latent, ids, ref in cases:
                ids_run = ids[:args.max_len]
                shift = 0
                if bos == "prepend" and ids_run[0] != bos_id:
                    ids_run = np.concatenate([[bos_id], ids_run])
                    shift = 1
                t = torch.tensor(ids_run, device=runner.device)[None]
                h = runner.residual(t, layer=layer)
                got = runner.sae.encode(h)[0, :, latent].float().cpu().numpy()
                keep = [q for q in ref if q + shift < len(got)]
                if len(keep) < 2:
                    continue
                exp = np.array([ref[q] for q in keep])
                mine = got[np.array([q + shift for q in keep])]
                if exp.std() > 0 and mine.std() > 0:
                    rs.append(float(np.corrcoef(mine, exp)[0, 1]))
                nz = exp > 0
                if nz.any():
                    errs.append(float(np.median(np.abs(mine[nz] - exp[nz]) / exp[nz])))
            if rs:
                results[(offset, bos)] = (float(np.mean(rs)),
                                          float(np.median(errs)) if errs else float("nan"), len(rs))

    print(f"\n{'offset':>7} {'hidden_states':>14} {'bos':>8} {'pearson r':>10} "
          f"{'med rel err':>12} {'n':>4}")
    for (offset, bos), (r, e, n) in sorted(results.items(), key=lambda kv: -kv[1][0]):
        print(f"{offset:>7} {cfg['layer']+offset+1:>14} {bos:>8} {r:>10.4f} {e:>11.1%} {n:>4}")

    if results:
        best = max(results, key=lambda k: results[k][0])
        r, e, _ = results[best]
        verdict = "PASS" if (r > 0.99 and e < 0.02) else "FAIL"
        print(f"\n{verdict}: {name} best = layers[{cfg['layer']+best[0]}].output "
              f"(= hidden_states[{cfg['layer']+best[0]+1}]), bos={best[1]}, "
              f"r={r:.4f}, median relative error={e:.2%}")
        json.dump({"sae": name, "best_offset": best[0], "bos": best[1], "pearson_r": r,
                   "median_rel_err": e, "verdict": verdict,
                   "grid": {f"{k[0]}|{k[1]}": v for k, v in results.items()}},
                  open(f"/home/a5k/cwtice.a5k/sae-exploration/out/validate_{name}.json", "w"),
                  indent=2)


runner = None
for name in args.sae:
    if runner is None or REGISTRY[name]["model"] != runner.cfg["model"]:
        del runner
        torch.cuda.empty_cache()
        runner = Runner(name)          # loads base model + this SAE
    else:
        from sae_local import load_sae  # same base model: swap the SAE only
        del runner.sae
        torch.cuda.empty_cache()
        runner.sae = load_sae(name, device=runner.device)
        runner.cfg = REGISTRY[name]
    validate(runner, name)
