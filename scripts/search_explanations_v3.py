"""Search Neuronpedia latent descriptions with the v3 category list.

Replaces the improvised 12-group keyword list previously in build_candidates.py.
Emits, per SAE, the latents whose autointerp description matches each v3
category -- so the candidate pool is partitioned by filtering level rather than
lumped together:

  categories 1+2  always filter  -> level-2 forget-set candidates
  category  5     SWE tests      -> only filtered at level 3
  categories 3+4  soft / general -> the main false-positive risk

Output: out/candidates_v3_<source>.jsonl and a per-category summary.
"""
import glob
import gzip
import json
import sys
from collections import Counter, defaultdict

sys.path.insert(0, "/home/a5k/cwtice.a5k/sae-exploration/scripts")
from regex_v3 import CORE, categorize  # noqa: E402

ROOT = "/home/a5k/cwtice.a5k/sae-exploration"
SOURCES = {
    "40-gemmascope-2-res-16k": (f"{ROOT}/out/np_expl_batch-*.jsonl.gz", 16384),
    "40-gemmascope-2-res-65k": (f"{ROOT}/out/np65k_batch-*.jsonl.gz", 65536),
}

for source, (pattern, width) in SOURCES.items():
    descs = {}
    for fn in glob.glob(pattern):
        for line in gzip.open(fn, "rt"):
            d = json.loads(line)
            if d.get("layer") == source:
                descs[int(d["index"])] = d["description"]
    if not descs:
        print(f"{source}: no local dump, skipping")
        continue

    hits, cat_counts = {}, Counter()
    for idx, desc in descs.items():
        c = categorize(desc)
        if not c:
            continue
        hits[idx] = {"index": idx, "description": desc,
                     "categories": {k: sorted(v) for k, v in c.items()}}
        for k in c:
            cat_counts[k] += 1

    core = {i for i, h in hits.items() if any(c in h["categories"] for c in CORE)}
    swe_only = {i for i, h in hits.items() if list(h["categories"]) == ["5_swe_tests"]}

    out = f"{ROOT}/out/candidates_v3_{source}.jsonl"
    with open(out, "w") as f:
        for i in sorted(hits):
            f.write(json.dumps(hits[i]) + "\n")

    print(f"\n=== {source} ===")
    print(f"  {len(descs)} descriptions available of {width} latents "
          f"({width - len(descs)} have none at all)")
    print(f"  {len(hits)} match any v3 category  ({100*len(hits)/width:.2f}% of dictionary)")
    print(f"  {len(core)} match categories 1+2 (always-filter core) "
          f"= {100*len(core)/width:.2f}% of dictionary")
    print(f"  {len(swe_only)} match ONLY category 5 (SWE) — these separate level 2 from level 3")
    for k, n in cat_counts.most_common():
        print(f"     {k:<32} {n:>5}")
    print(f"  wrote {out}")

    # which individual terms are doing the work
    term_counts = defaultdict(int)
    for h in hits.values():
        for cat, labels in h["categories"].items():
            for lb in labels:
                term_counts[(cat, lb)] += 1
    print("  top matching terms:")
    for (cat, lb), n in sorted(term_counts.items(), key=lambda kv: -kv[1])[:12]:
        print(f"     {n:>5}  {cat.split('_')[0]}  {lb}")
