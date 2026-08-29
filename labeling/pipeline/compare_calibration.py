"""Four-way model comparison on the calibration latents.

Usage: python3 relabel/compare_calibration.py haiku45=path.json sonnet5=path.json ...
Each file is a collect.py-style labels list. Prints agreement stats and the
latents where models disagree (with each model's category/confidence/rationale).
"""

import json
import pathlib
import sys
from itertools import combinations

ROOT = pathlib.Path(__file__).resolve().parent.parent


def main():
    models = {}
    for arg in sys.argv[1:]:
        name, path = arg.split("=", 1)
        models[name] = {x["index"]: x for x in json.load(open(path))}
    names = list(models)

    verdicts = {x["index"]: x for x in
                json.load(open(ROOT / "results" / "gemma3_L40_16k_verdicts.json"))}

    common = set.intersection(*(set(m) for m in models.values()))
    print(f"models: {names}; latents labeled by all: {len(common)} "
          f"(per-model: {', '.join(f'{n}={len(m)}' for n, m in models.items())})\n")

    print("forget counts on common set:")
    for n in names:
        f = sum(1 for i in common if models[n][i]["forget"])
        print(f"  {n}: {f}/{len(common)}")

    print("\npairwise forget/keep agreement:")
    for a, b in combinations(names, 2):
        agree = sum(1 for i in common if models[a][i]["forget"] == models[b][i]["forget"])
        print(f"  {a} vs {b}: {agree}/{len(common)} ({100 * agree / len(common):.1f}%)")

    split = [i for i in common
             if len({models[n][i]["forget"] for n in names}) > 1]
    print(f"\n{len(split)} latents with any disagreement:\n")
    for i in sorted(split):
        v = verdicts.get(i)
        prior = f" prior={v['verdict']}" if v else ""
        desc = (v or {}).get("desc") or models[names[0]][i].get("desc") or ""
        print(f"--- latent {i}{prior} desc: {desc}")
        for n in names:
            l = models[n][i]
            flag = "FORGET" if l["forget"] else "keep  "
            print(f"    {n:<9} {flag} {l['category']:<32} ({l['confidence']}) {l['rationale']}")
        print()


if __name__ == "__main__":
    main()
