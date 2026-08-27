from huggingface_hub import snapshot_download

jobs = [
    # 16k @ layer 46 = depth 0.742 (Rathi 0.738), l0_big L0=120 (Rathi 114)
    ("google/gemma-scope-2-27b-it", ["resid_post_all/layer_46_width_16k_l0_big/*"]),
    # 65k @ layer 40 = closest accessible depth for 65k; l0_medium matches Neuronpedia
    ("google/gemma-scope-2-27b-it", ["resid_post/layer_40_width_65k_l0_medium/*"]),
    # Olmo: Neuronpedia 32-res-batchtopk-131k
    ("bcywinski/Olmo-3-32B-Base-SAE",
     ["saes_allenai_Olmo-3-1125-32B_batch_top_k/resid_post_layer_32/trainer_0/*"]),
    # gemma-2-9b base model (google repo 403-gated for this account)
    ("unsloth/gemma-2-9b", ["*.json", "*.safetensors", "tokenizer*"]),
]
for repo, patterns in jobs:
    p = snapshot_download(repo, allow_patterns=patterns)
    print(f"DONE {repo} {patterns} -> {p}", flush=True)
