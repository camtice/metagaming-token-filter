"""Assemble the v2 feature-firing artifact (Apollo + metagaming-probe corpora).

Usage: python3 scripts/build_v2_artifact.py
Output: out/feature_firing_v2.html
"""
import json

ROOT = "/home/a5k/cwtice.a5k/sae-exploration"
def encode_tv(corpus):
    for d in corpus["tokenView"]["docs"]:
        enc, prev_e = [], None
        for st, en in d["toks"]:
            enc.extend([st if prev_e is None else st - prev_e, en - st])
            prev_e = en
        d["toks_enc"] = enc
        aenc = {}
        for li, arr in d["acts"].items():
            # floor: drop sub-1.0 activations (negligible for the Rathi rule's
            # windowing; halves the payload for the 73-feature token views)
            arr = [(t, a) for t, a in arr if a >= 2.0]
            if not arr:
                continue
            fl, prev_t = [], None
            for t, a in arr:
                fl.extend([t if prev_t is None else t - prev_t, round(a)])
                prev_t = t
            aenc[li] = fl
        d["acts_enc"] = aenc
        del d["toks"], d["acts"]
    return corpus

data = {"saes": [encode_tv(json.load(open(f"{ROOT}/out/corpus_view_apollo.json"))),
                 encode_tv(json.load(open(f"{ROOT}/out/corpus_view_probe432.json")))]}
blob = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
tpl = open(f"{ROOT}/artifact/feature_firing_v2.template.html").read()
assert "/*__DATA__*/" in tpl
html = tpl.replace("/*__DATA__*/", blob, 1)
open(f"{ROOT}/out/feature_firing_v2.html", "w").write(html)
print(f"wrote out/feature_firing_v2.html ({len(html)/1e6:.2f} MB)")
rows = [json.loads(l) for l in open(f"{ROOT}/data/probe_dataset.jsonl")]
n = sum(1 for r in rows if json.dumps(r["text"])[1:-1].replace("</", "<\\/") in html)
print(f"apollo texts embedded: {n}/{len(rows)}")
