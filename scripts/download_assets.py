import os
from huggingface_hub import snapshot_download

# Primary SAE: Neuronpedia gemma-3-27b-it/40-gemmascope-2-res-16k
snapshot_download("google/gemma-scope-2-27b-it",
                  allow_patterns=["resid_post/layer_40_width_16k_l0_medium/*"])
print("SAE l40 16k done")
# Validation SAE: Neuronpedia gemma-2-9b/31-gemmascope-res-16k
snapshot_download("google/gemma-scope-9b-pt-res",
                  allow_patterns=["layer_31/width_16k/average_l0_114/*"])
print("validation SAE done")
# Base model mirror (google repo is gated 403 for this account)
snapshot_download("unsloth/gemma-3-27b-it",
                  allow_patterns=["*.json", "*.safetensors", "tokenizer*"])
print("gemma-3-27b-it mirror done")
