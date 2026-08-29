"""Build full-16k Batch requests with a prompt_iter variant system prompt.

Usage: python3 relabel/build_16k_variant.py v6
Writes relabel/requests_16k_<variant>.jsonl. Targets = all live latents except
the 14 base few-shot examples (variant-specific example latents stay in as
targets; their labels are trivially biased — 6 latents, noted in outputs).
"""

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import build_prompts as bp
import prompt_iter

ROOT = pathlib.Path(__file__).resolve().parent.parent


def main():
    variant = sys.argv[1]
    descs, feats, acts = bp.load_data()
    system = prompt_iter.build_variant_system(variant, descs, feats, acts)
    print(f"system ({variant}): ~{len(system) // 4} tokens", flush=True)

    base_examples = {i for i, *_ in bp.FORGET_EXAMPLES} | {i for i, _ in bp.KEEP_EXAMPLES}
    live = sorted(i for i in acts
                  if feats.get(i, {}).get("max_act", 0) > 0 and i not in base_examples)
    out = ROOT / "relabel" / f"requests_16k_{variant}.jsonl"
    with open(out, "w") as fh:
        for idx in live:
            fh.write(json.dumps({
                "custom_id": f"latent-{idx}",
                "params": {
                    "model": bp.MODEL, "max_tokens": 150, "temperature": 0,
                    "system": [{"type": "text", "text": system,
                                "cache_control": {"type": "ephemeral"}}],
                    "messages": [{"role": "user",
                                  "content": bp.render_feature(idx, descs, feats, acts)}],
                },
            }) + "\n")
    print(f"wrote {len(live)} requests -> {out} ({out.stat().st_size // 2**20} MB)")


if __name__ == "__main__":
    main()
