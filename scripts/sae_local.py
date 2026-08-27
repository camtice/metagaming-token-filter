"""Local SAE inference: load a pretrained SAE + its base model, encode text.

Four SAEs registered, chosen to bracket Rathi & Radford (arXiv 2601.21571),
who used gemma-2-9b layer 31 of 42 = 0.738 relative depth, average L0 114:

  gemma2-l31-16k    gemma-2-9b      L31/42  depth 0.738  16k   L0 114  <- the paper's SAE
  gemma3-l46-16k    gemma-3-27b-it  L46/62  depth 0.742  16k   L0 120  <- closest analogue
  gemma3-l40-65k    gemma-3-27b-it  L40/62  depth 0.645  65k   L0  60  <- 65k only exists at 16/31/40/53
  olmo3-l32-131k    Olmo-3-1125-32B L32/64  depth 0.500  131k  k=160   <- only layer published

HOOK CONVENTION (verified by validate_sae.py, not assumed):
HF `output_hidden_states=True` returns num_layers+1 tensors where
hidden_states[0] is the embedding output, so the output of `model.layers[i]`
is hidden_states[i+1]. Gemma Scope 2 configs name their hook point
`model.layers.<L>.output`, which therefore means hidden_states[L+1].
Note Rathi's `label-corpus.py` reads `hidden_states[31]` for `layer_31/...`,
i.e. the output of layer 30 — one layer early under this convention. Both
offsets are scored by the validator so the right one is measured, not guessed.
"""
import glob
import json
import os

import numpy as np
import torch

HUB = ("/projects/a5k/public/data_cwtice.a5k/.cache/huggingface/hub")

REGISTRY = {
    "gemma2-l31-16k": dict(
        arch="jumprelu_npz", model="unsloth/gemma-2-9b", sae_repo="google/gemma-scope-9b-pt-res",
        sae_folder="layer_31/width_16k/average_l0_114",
        layer=31, n_layers=42, width=16384, d_model=3584,
        neuronpedia=("gemma-2-9b", "31-gemmascope-res-16k"),
    ),
    "gemma3-l46-16k": dict(
        arch="jumprelu_safetensors", model="unsloth/gemma-3-27b-it",
        sae_repo="google/gemma-scope-2-27b-it",
        sae_folder="resid_post_all/layer_46_width_16k_l0_big",
        layer=46, n_layers=62, width=16384, d_model=5376,
        neuronpedia=None,  # not indexed on Neuronpedia; has shipped examples
    ),
    "gemma3-l40-65k": dict(
        arch="jumprelu_safetensors", model="unsloth/gemma-3-27b-it",
        sae_repo="google/gemma-scope-2-27b-it",
        sae_folder="resid_post/layer_40_width_65k_l0_medium",
        layer=40, n_layers=62, width=65536, d_model=5376,
        neuronpedia=("gemma-3-27b-it", "40-gemmascope-2-res-65k"),
    ),
    "gemma3-l40-16k": dict(  # the set already classified this session
        arch="jumprelu_safetensors", model="unsloth/gemma-3-27b-it",
        sae_repo="google/gemma-scope-2-27b-it",
        sae_folder="resid_post/layer_40_width_16k_l0_medium",
        layer=40, n_layers=62, width=16384, d_model=5376,
        neuronpedia=("gemma-3-27b-it", "40-gemmascope-2-res-16k"),
    ),
    # controlled comparison: same layer/width as gemma3-l40-16k, but from the
    # bulk-trained resid_post_all family rather than resid_post
    "gemma3-l40-16k-all": dict(
        arch="jumprelu_safetensors", model="unsloth/gemma-3-27b-it",
        sae_repo="google/gemma-scope-2-27b-it",
        sae_folder="resid_post_all/layer_40_width_16k_l0_big",
        layer=40, n_layers=62, width=16384, d_model=5376,
        neuronpedia=None,
    ),
    "olmo3-l32-131k": dict(
        arch="batchtopk", model="allenai/Olmo-3-1125-32B",
        sae_repo="bcywinski/Olmo-3-32B-Base-SAE",
        sae_folder="saes_allenai_Olmo-3-1125-32B_batch_top_k/resid_post_layer_32/trainer_0",
        layer=32, n_layers=64, width=131072, d_model=5120,
        neuronpedia=("olmo-3-1125-32b", "32-res-batchtopk-131k"),
    ),
}


def snapshot_dir(repo):
    pat = f"{HUB}/models--{repo.replace('/', '--')}/snapshots/*"
    hits = glob.glob(pat)
    if not hits:
        raise FileNotFoundError(f"not cached: {repo} ({pat})")
    return sorted(hits)[-1]


def sae_dir(name):
    c = REGISTRY[name]
    return os.path.join(snapshot_dir(c["sae_repo"]), c["sae_folder"])


class JumpReLUSAE(torch.nn.Module):
    """acts = pre * (pre > threshold), pre = x @ W_enc + b_enc."""

    def __init__(self, W_enc, b_enc, threshold, W_dec, b_dec):
        super().__init__()
        self.register_buffer("W_enc", W_enc)
        self.register_buffer("b_enc", b_enc)
        self.register_buffer("threshold", threshold)
        self.register_buffer("W_dec", W_dec)
        self.register_buffer("b_dec", b_dec)

    def encode(self, x):
        pre = x.to(self.W_enc.dtype) @ self.W_enc + self.b_enc
        return pre * (pre > self.threshold)

    def decode(self, acts):
        return acts @ self.W_dec + self.b_dec


class BatchTopKSAE(torch.nn.Module):
    """dictionary_learning BatchTopKSAE.

    At inference the trained per-latent `threshold` scalar is used rather than a
    hard top-k, so a single sequence is scored the same way it would be inside a
    full batch. Falls back to per-token top-k if no threshold was saved.
    """

    def __init__(self, W_enc, b_enc, W_dec, b_dec, k, threshold=None):
        super().__init__()
        self.register_buffer("W_enc", W_enc)
        self.register_buffer("b_enc", b_enc)
        self.register_buffer("W_dec", W_dec)
        self.register_buffer("b_dec", b_dec)
        self.k = int(k)
        self.threshold = None if threshold is None else float(threshold)

    def encode(self, x):
        pre = torch.relu((x.to(self.W_enc.dtype) - self.b_dec) @ self.W_enc + self.b_enc)
        if self.threshold is not None and self.threshold > 0:
            return pre * (pre > self.threshold)
        flat = pre.reshape(-1, pre.shape[-1])
        topk = torch.topk(flat, self.k, dim=-1)
        out = torch.zeros_like(flat).scatter_(-1, topk.indices, topk.values)
        return out.reshape(pre.shape)

    def decode(self, acts):
        return acts @ self.W_dec + self.b_dec


def load_sae(name, device="cuda", dtype=torch.float32):
    c = REGISTRY[name]
    d = sae_dir(name)
    t = lambda a: torch.as_tensor(np.array(a)).to(device=device, dtype=dtype)

    if c["arch"] == "jumprelu_npz":
        p = np.load(os.path.join(d, "params.npz"))
        sae = JumpReLUSAE(t(p["W_enc"]), t(p["b_enc"]), t(p["threshold"]),
                          t(p["W_dec"]), t(p["b_dec"]))
    elif c["arch"] == "jumprelu_safetensors":
        from safetensors.numpy import load_file
        p = load_file(os.path.join(d, "params.safetensors"))
        sae = JumpReLUSAE(t(p["w_enc"]), t(p["b_enc"]), t(p["threshold"]),
                          t(p["w_dec"]), t(p["b_dec"]))
    elif c["arch"] == "batchtopk":
        sd = torch.load(os.path.join(d, "ae.pt"), map_location="cpu", weights_only=True)
        cfg = json.load(open(os.path.join(d, "config.json")))["trainer"]
        # dictionary_learning stores encoder as [dict_size, act_dim]; we want [act_dim, dict]
        W_enc = sd["encoder.weight"].T.contiguous()
        W_dec = sd["decoder.weight"].T.contiguous()
        thr = sd.get("threshold")
        sae = BatchTopKSAE(t(W_enc), t(sd["encoder.bias"]), t(W_dec), t(sd["b_dec"]),
                           k=cfg["k"], threshold=None if thr is None else float(thr))
    else:
        raise ValueError(c["arch"])

    assert sae.W_enc.shape == (c["d_model"], c["width"]), \
        f"{name}: W_enc {tuple(sae.W_enc.shape)} != {(c['d_model'], c['width'])}"
    return sae.eval()


class _Stop(Exception):
    pass


class Runner:
    """Base model + SAE. Captures the residual stream and encodes it."""

    def __init__(self, name, device="cuda", model_dtype=torch.bfloat16, device_map=None):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.name, self.cfg = name, REGISTRY[name]
        self.device = device
        mdir = snapshot_dir(self.cfg["model"])
        self.tokenizer = AutoTokenizer.from_pretrained(mdir)
        self.model = AutoModelForCausalLM.from_pretrained(
            mdir, dtype=model_dtype, device_map=device_map or device,
            attn_implementation="eager").eval()
        self.layers = self._find_layers()
        assert len(self.layers) == self.cfg["n_layers"], \
            f"{len(self.layers)} layers found, expected {self.cfg['n_layers']}"
        self.sae = load_sae(name, device=device)

    def _find_layers(self):
        m = self.model
        for path in ("model.language_model.layers", "model.layers",
                     "model.model.language_model.layers", "model.model.layers"):
            obj = m
            try:
                for part in path.split("."):
                    obj = getattr(obj, part)
                return obj
            except AttributeError:
                continue
        raise RuntimeError("could not locate decoder layers")

    @torch.no_grad()
    def residual(self, input_ids, attention_mask=None, layer=None, early_exit=True):
        """Return the output of decoder layer `layer` (default: the SAE's layer).

        This is the tensor Gemma Scope calls `model.layers.<L>.output`, i.e.
        hidden_states[L+1] in HF's `output_hidden_states` indexing.
        """
        layer = self.cfg["layer"] if layer is None else layer
        captured = {}

        def hook(_mod, _inp, out):
            captured["h"] = (out[0] if isinstance(out, tuple) else out).detach()
            if early_exit:
                raise _Stop

        handle = self.layers[layer].register_forward_hook(hook)
        try:
            emb = self.model.get_input_embeddings()
            dev = next(emb.parameters()).device       # shards may live on other GPUs
            input_ids = input_ids.to(dev)
            if attention_mask is not None:
                attention_mask = attention_mask.to(dev)
            self.model(input_ids=input_ids, attention_mask=attention_mask)
        except _Stop:
            pass
        finally:
            handle.remove()
        return captured["h"].float().to(self.sae.W_enc.device)

    @torch.no_grad()
    def encode_text(self, text, max_length=512, add_special_tokens=True):
        enc = self.tokenizer(text, return_tensors="pt", truncation=True,
                             max_length=max_length, add_special_tokens=add_special_tokens)
        enc = {k: v.to(self.device) for k, v in enc.items()}
        h = self.residual(enc["input_ids"], enc.get("attention_mask"))
        return self.sae.encode(h), enc["input_ids"]

    @torch.no_grad()
    def encode_tokens(self, input_ids):
        if not torch.is_tensor(input_ids):
            input_ids = torch.tensor(input_ids)
        if input_ids.ndim == 1:
            input_ids = input_ids[None]
        input_ids = input_ids.to(self.device)
        return self.sae.encode(self.residual(input_ids)), input_ids
