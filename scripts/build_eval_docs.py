"""Eval protocol v1 — materialize test/val doc files from the sealed manifest.

Texts come from the pinned dataset revision where the config still exists
(frontier, kimi, dense_c*); the annotated control/prose docs whose source
configs were removed in the restructure come from the locally verified
08-01 assembly (data/labeled_eval_docs.jsonl). Span annotations are word-index
spans (end-exclusive) converted to char offsets, same convention verified on
2026-08-01. Where a doc exists in both sources, texts are asserted identical.

Usage: python build_eval_docs.py
Output: data/test_docs_v1.jsonl, data/val_docs_v1.jsonl
"""
import json

import pandas as pd
from huggingface_hub import hf_hub_download

ROOT = "/home/a5k/cwtice.a5k/sae-exploration"
man = json.load(open(f"{ROOT}/data/splits/split_manifest_v1.json"))
REPO, REV = man["repo"], man["revision"]

DL = lambda cfg: pd.read_parquet(hf_hub_download(REPO, f"{cfg}/eval-00000-of-00001.parquet",
                                                 repo_type="dataset", revision=REV))
ann = DL("annotations")
ann = ann[ann.status.isin(["done", "clean"])]
ann_by_id = {r.id: r for r in ann.itertuples()}

frontier = {r.id: r.text for r in DL("frontier_metagame_rollouts").itertuples()}
kimi = {r.id: r.text for r in DL("kimi_metagame_rollouts").itertuples()}
dense = {}
for c in ["dense_c1_ai_evals", "dense_c2_ai_safety", "dense_c3_human_oversight",
          "dense_c4_training_mechanics", "dense_c5_swe_tests"]:
    for r in DL(c).itertuples():
        dense[(c, r.id)] = r.text
local = {d["id"]: d for d in map(json.loads, open(f"{ROOT}/data/labeled_eval_docs.jsonl"))}


def spans_to_char(text, ann_row):
    words = text.split()
    assert len(words) == ann_row.tokenCount, f"word-count mismatch for {ann_row.id}"
    ranges, pos = [], 0
    for w in words:
        s = text.index(w, pos)
        ranges.append((s, s + len(w)))
        pos = s + len(w)
    out = []
    for s, e in [list(x) for x in ann_row.spans]:
        e_ex = min(int(e), len(words))
        if e_ex > int(s):
            out.append([ranges[int(s)][0], ranges[e_ex - 1][1]])
    return out


def find_ann(config, did):
    """Annotation row for a manifest doc, trying the id forms seen in the tool."""
    prefix = {"frontier_metagame_rollouts": "cotprobe", "kimi_metagame_rollouts": "kimi"}.get(config)
    cands = [did]
    if prefix:
        cands.append(f"{prefix}:{did}")
    if did.startswith(("kimi:", "cotprobe:", "pr123:")):
        cands.append(did)
    for c in cands:
        if c in ann_by_id:
            return ann_by_id[c]
    return None


def resolve_text(config, did):
    if config == "frontier_metagame_rollouts":
        return frontier.get(did)
    if config == "kimi_metagame_rollouts":
        for k in (did, did.split(":", 1)[-1]):
            if k in kimi:
                return kimi[k]
        return None
    if config.startswith("dense_"):
        return dense.get((config, did))
    # removed configs: use the verified local assembly (ids are the tool ids)
    if did in local:
        return local[did]["text"]
    return None


def build(entries, out_path):
    rows, stats = [], {"spans": 0, "clean": 0, "dense_forget": 0, "unlabeled": 0}
    for e in entries:
        text = resolve_text(e["config"], e["id"])
        assert text is not None, f"unresolved {e}"
        a = find_ann(e["config"], e["id"])
        if e.get("label_type") == "dense_forget" or e["register"] == "dense":
            lt, spans = "dense_forget", []
        elif a is not None:
            spans = spans_to_char(text, a)
            lt = "spans" if spans else "clean"
        else:
            lt, spans = "unlabeled", []
        # continuity check vs the 08-01 assembly where both exist
        lid = a.id if a is not None else None
        if lid in local:
            assert local[lid]["text"] == text, f"text drift vs 08-01 assembly: {lid}"
            assert local[lid]["char_spans"] == spans, f"span drift: {lid}"
        stats[lt] += 1
        rows.append({"id": e["id"], "config": e["config"], "role": e["role"],
                     "register": e["register"], "label_type": lt, "group": e["register"],
                     "text": text, "char_spans": spans})
    with open(out_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(out_path, len(rows), stats)
    return rows


build(man["validation"], f"{ROOT}/data/val_docs_v1.jsonl")
build(man["test"], f"{ROOT}/data/test_docs_v1.jsonl")
