"""Eval protocol v1 — build the frozen test/validation split manifests.

Tiers (user decisions 2026-08-09):
  VALIDATION  all 57 frontier_metagame_rollouts (Apollo) + 12 annotated
              kimi_metagame_rollouts docs (seeded, stratified by family and
              span-presence). Locked: scored only for frozen feature sets,
              every access logged.
  TEST        all remaining annotated docs (spans + reviewed-clean) +
              all dense_c1..c5 docs (scored under the ~all-forget assumption).
  SELECTION   unannotated bulk (kimi remainder, kimi_agentic, westover_general,
              needham) — listed by config, not per-doc.
  EXCLUDED    claude_AI_forget / claude_AI_mix / claude_human_forget /
              claude_general_retain — not dense enough for token-level use.

Pinned to the dataset revision sha at build time. Output is sealed with its
own sha256; publish a v2 manifest for any change, never edit v1.

Usage: python build_split_manifests.py
Output: data/splits/split_manifest_v1.json, data/splits/validation_access_log.jsonl
"""
import hashlib
import json
import os
import random

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download

ROOT = "/home/a5k/cwtice.a5k/sae-exploration"
REPO = "camgeodesic/metagaming-labeling"
SEED = 20260809
N_CHAT_VAL = 12

api = HfApi()
rev = api.dataset_info(REPO).sha
print("pinned revision:", rev)

ann = pd.read_parquet(hf_hub_download(REPO, "annotations/eval-00000-of-00001.parquet",
                                      repo_type="dataset", revision=rev))
usable = ann[ann.status.isin(["done", "clean"])]
# the two docs whose word counts disagree with the annotation tool (dropped 08-01)
BAD = {"kimi:dolci_if_contradictory/4625", "kimi:doluschat/aviation_diversity_omission"}
usable = usable[~usable.id.isin(BAD)]

frontier = pd.read_parquet(hf_hub_download(REPO, "frontier_metagame_rollouts/eval-00000-of-00001.parquet",
                                           repo_type="dataset", revision=rev))

# ---- validation: all Apollo + seeded stratified chat slice ----
val = [{"config": "frontier_metagame_rollouts", "id": i, "role": "validation",
        "register": "cot"} for i in sorted(frontier.id)]
kimi_ann = usable[usable.doc_set == "chat_kimi"].copy()
kimi_ann["family"] = kimi_ann.id.str.split(":").str[-1].str.split("/").str[0]
kimi_ann["spanned"] = kimi_ann.n_spans > 0
rng = random.Random(SEED)
rows = sorted(kimi_ann.itertuples(), key=lambda r: r.id)
strata = {}
for r in rows:
    strata.setdefault((r.family, r.spanned), []).append(r.id)
picked = []
keys = sorted(strata)
while len(picked) < N_CHAT_VAL:
    for k in keys:
        if strata[k] and len(picked) < N_CHAT_VAL:
            picked.append(strata[k].pop(rng.randrange(len(strata[k]))))
picked = sorted(picked)
val += [{"config": "kimi_metagame_rollouts", "id": i, "role": "validation", "register": "chat"}
        for i in picked]

# ---- test: remaining annotated + dense ----
REG = {"cot_probe_hq": "cot", "chat_kimi": "chat", "metagaming_probe_docs": "prose",
       "westover_infocus": "prose", "control_far_negative": "control",
       "control_human_oversight": "control", "control_ml_distractor": "control"}
frontier_ids = set(frontier.id)
test = []
for r in usable.itertuples():
    if r.doc_set == "cot_probe_hq":
        did = r.id.split(":", 1)[-1]
        if did in frontier_ids:
            continue                   # Apollo docs -> validation
        # annotated ctl_* docs deleted from the frontier config: keep as
        # annotated hard-vocabulary CONTROLS in test (texts from the verified
        # 08-01 local assembly)
        test.append({"config": "cot_probe_hq", "id": r.id, "role": "test",
                     "register": "control",
                     "label_type": "spans" if r.n_spans > 0 else "clean"})
        continue
    if r.doc_set == "chat_kimi" and r.id in set(picked):
        continue
    test.append({"config": r.doc_set, "id": r.id, "role": "test", "register": REG[r.doc_set],
                 "label_type": "spans" if r.n_spans > 0 else "clean"})
for c in ["dense_c1_ai_evals", "dense_c2_ai_safety", "dense_c3_human_oversight",
          "dense_c4_training_mechanics", "dense_c5_swe_tests"]:
    df = pd.read_parquet(hf_hub_download(REPO, f"{c}/eval-00000-of-00001.parquet",
                                         repo_type="dataset", revision=rev))
    test += [{"config": c, "id": i, "role": "test", "register": "dense",
              "label_type": "dense_forget"} for i in sorted(df.id)]

manifest = {
    "version": 1, "created": "2026-08-09", "repo": REPO, "revision": rev, "seed": SEED,
    "rules": {
        "validation": "score frozen feature sets only; every access appended to validation_access_log.jsonl; baseline scored once as reference",
        "test": "scored freely during iteration",
        "selection_configs": ["kimi_metagame_rollouts (unannotated remainder)",
                               "kimi_agentic_rollouts", "westover_general_split",
                               "westover_metagame_split", "needham_eval_awareness",
                               "needham_eval_grading"],
        "excluded_configs": ["claude_AI_forget", "claude_AI_mix", "claude_human_forget",
                              "claude_general_retain"],
        "excluded_reason": "not dense enough for token-level scoring (user decision 2026-08-09)",
        "dropped_docs": sorted(BAD),
        "dense_assumption": "dense_c* docs scored as ~all-forget (uniform in-category); see protocol report spot-check",
        "no_migration": "a doc placed in test/validation never becomes selection data; changes require a v2 manifest",
    },
    "validation": val, "test": test,
}
blob = json.dumps(manifest, sort_keys=True)
manifest["manifest_sha256"] = hashlib.sha256(blob.encode()).hexdigest()
os.makedirs(f"{ROOT}/data/splits", exist_ok=True)
json.dump(manifest, open(f"{ROOT}/data/splits/split_manifest_v1.json", "w"), indent=1)
open(f"{ROOT}/data/splits/validation_access_log.jsonl", "a").close()
from collections import Counter
print("validation:", len(val), dict(Counter(v["register"] for v in val)))
print("test:", len(test), dict(Counter(t["register"] for t in test)))
print("chat val picks:", picked)
print("sealed:", manifest["manifest_sha256"][:16])
