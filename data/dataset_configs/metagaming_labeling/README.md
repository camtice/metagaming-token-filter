# Control-set build configs (dataset-builder)

The six `ctl_*` control configs from the private
[GeodesicResearch/dataset-builder](https://github.com/GeodesicResearch/dataset-builder)
repo, copied verbatim so the control corpora are reproducible without access to that
repo. Each YAML is the complete build: source dataset + revision, transform chain,
purity gates, seeds, and published size (with the measured retention funnel in the
header comments). Build with:

    python -m dataset_builder configs/metagaming_labeling/<name>/<name>.yaml --build-and-push

| config | source | shape | purity gates |
|---|---|---|---|
| ctl_fineweb | HuggingFaceFW/fineweb (one pinned 2.1GB shard) | web prose, stratified over 8 CommonCrawl dumps | term blocklist only (VEA judge is CoT-calibrated, out of scope for prose) |
| ctl_wildchat | allenai/WildChat | real user chats | blocklist + VEA auditor judge |
| ctl_dolci_instruct | allenai/Dolci-Instruct-SFT | SFT answers | blocklist + VEA judge |
| ctl_dolci_think | allenai/Dolci-Think-SFT-32B | reasoning traces | blocklist + VEA judge |
| ctl_swe / ctl_swe_tests | allenai/Dolci-Instruct-SFT (Coding domain) | matched coding pair: disjoint halves of ONE stratified draw differing only in the category-5 (test-vocabulary) bound — the FP gap between them isolates test vocabulary | blocklist (cats 1–4) ± cat-5 bound + VEA judge |

`assets/` carries the two referenced files verbatim: `pools/blocklist_v3.json`
(the 338-term, 5-category screen) and `prompts/eval_awareness_auditor.jinja`
(the VEA judge prompt, itself provenance-pinned to geodesic-environments in its
header). Published data: private HF dataset `camgeodesic/metagaming-labeling`
at the revisions pinned in `data/splits/split_manifest_v5.json`.
