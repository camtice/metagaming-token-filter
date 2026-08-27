"""Iterate on the Haiku prompt to better match Fable's 16k labels.

Dev set: all Fable/Haiku binary disagreements (minus latents promoted into the
few-shot examples) + fixed random agreement controls. Submits a variant prompt
to Haiku via the Batch API, waits, and scores agreement against Fable.

Usage: python3 relabel/prompt_iter.py v2
"""

import json
import pathlib
import random
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import build_prompts as bp
from collect import parse_reply
from run_batch import API, call

ROOT = pathlib.Path(__file__).resolve().parent.parent

# --- variant definitions -----------------------------------------------------

LOGIT_RULE = """\

PROMOTED-TOKEN RULE - treat the top promoted output tokens as primary evidence \
alongside the snippets: a feature that promotes tokens such as 'reward', \
'reinforcement', 'loss', 'accuracy', 'test', 'assert', 'grading', or \
'monitoring' would be active on forget-set content even when its top snippets \
look unrelated - label such features forget=true (confidence low or medium).

A large SAE contains many features with only a partial or plausible connection \
to these categories. Expect to use confidence 'low' often - roughly as many \
low labels as high and medium combined. When you are torn between keep and \
forget, choose forget=true with confidence 'low': downstream filtering can \
drop low-confidence features, but a missed feature cannot be recovered. \
Reserve forget=false for features with no oversight-related sense in any \
snippet or promoted token, or where the only link is a lexical collision.
"""

# rendered as extra few-shot examples with forget/low answers (Fable's labels)
LOW_EXAMPLES = [
    (15213, "c4_ai_training_pipeline",
     "Behavioral-psychology reward/reinforcement content (dog training, therapy); the promoted tokens 'reward'/'reinforcement' mean it would also fire on RL training text."),
    (7408, "c3_human_oversight_testing",
     "General teaching/mentorship topic, but overlaps supervision and evaluation of human learners (clinical supervision, performance commendations)."),
]


BOUNDARY_RULES = """\

SCOPE CLARIFICATIONS - the following ARE forget-set features (common mistakes):
- Educational materials of any kind: worksheets, exercises, homework, essay
  prompts, fill-in-the-blank drills, lesson plans, course/classroom content.
  These are c3 even when no grading is explicitly mentioned - assessment is
  intrinsic to the setting. Generic school/teaching topic features are
  forget=true with confidence low.
- Regulatory and institutional compliance: regulations, licensing, standards,
  certification, inspections, audits (FCC, ISO, FTC, safety codes). These are
  c3 - regulated entities are being monitored. "It's just a compliance topic
  feature" is not a reason to keep.
- Neural-network architecture and ML methods content: layers, CNNs,
  embeddings, hyperparameters, classifiers. This is c4 - architecture text
  co-occurs with training machinery.
- Being a broad topic feature is never by itself a reason for forget=false.
  Ask instead: does the topic overlap any category? If partially, forget=true
  with confidence low or medium.
"""


ANY_OVERLAP = """\

ANY-OVERLAP PROCEDURE - decide mechanically, in this order:
1. Scan all snippets and all promoted tokens one by one.
2. If at least ONE snippet or promoted token clearly involves a category -
   even if everything else is unrelated - the feature is forget=true. One
   exam question among ten generic snippets, one 'assert'/'reward'/'test'
   promoted token, one compliance-audit snippet: forget=true, confidence low.
3. Only if NO snippet and NO promoted token overlaps any category (or the
   only overlap is a lexical collision in a clearly different sense) answer
   forget=false.
Confidence: all/most evidence on-category = high; several items = medium;
exactly one or two weak items = low.

More content that IS in scope (all typically confidence low):
- Legal, statutory, and government-regulation language generally (statutes,
  'shall' clauses, SEC/GAO/legislation, licensing rules) - c3.
- Privacy, tracking, and surveillance-adjacent content: cookies, GDPR,
  anonymity, CCTV/camera monitoring of people - c3.
- Workplace hiring, job duties, recruitment, supervision, appraisals, and
  'performance' in the workplace sense - c3.
- Crime-and-enforcement or deception-adjacent content: patrols, fraud,
  cheating, manipulation, hiding/concealment - c3 (or c2 for deception).
- ML research artifacts: paper citations (NIPS/CVPR), datasets,
  normalization, generalization - c4.
"""


V5_CLUSTERS = """\

Further in-scope content (frequently mislabeled keep):
- Competitions, contests, and competitive exams of every kind: exam question
  banks, contest problems, online judges (LeetCode, IOI, Topcoder, TLE/verdict
  content), admissions, auditions, applications, selection processes. Entrants
  are being evaluated and ranked - c3 (c5 when code is auto-judged).
- Textbook, homework, and worked problems: word problems, solution manuals,
  answer keys, 'Solved:' content, question banks. Always c3 - these are the
  materials humans are tested with. No exception for 'it is just educational
  material'.
- Authorities investigating or scrutinizing anyone: tax authorities, antitrust
  regulators, anti-corruption investigations, intelligence/insider-threat
  monitoring, ESG/CSR audits, transparency and accountability in governance -
  c3.
- CI and build infrastructure: build logs, BUILD SUCCESSFUL, health checks,
  code coverage (tarpaulin), doctests - c5.
- Features firing on oversight vocabulary itself - the words 'evaluation',
  'assessment', 'investigation', 'performance', 'monitoring', 'inspection' -
  are forget=true even when the surrounding contexts are mundane: the same
  token fires on oversight text. Same for psychological testing and GPS/
  location tracking of people.
"""

EVIDENCE_RULE = """\

EVIDENCE RULE - when you answer forget=true, name in the rationale the
specific snippet number(s) or promoted token(s) that overlap the category.
When you answer forget=false, your rationale must state that no snippet and
no promoted token overlaps any category. If you cannot truthfully write that
sentence, the answer is forget=true.
"""


RECALL_OBJECTIVE = """\

RECALL OBJECTIVE - this labeling feeds a filtering pipeline where a MISSED
feature is far more costly than a marginally included one: included features
can be filtered later by confidence, but missed features are gone. When in
any doubt at all, answer forget=true with confidence low.

The bar is NOT 'is this feature about a category?'. The bar is: 'would this
feature ACTIVATE on text from a category?'. A feature firing on the token
'torch' will activate on PyTorch training text; a feature firing on
'sentence' activates on grammar-exam text; a feature promoting '____' blank
tokens activates on fill-in-the-blank quizzes. ONE weak overlap in ONE
snippet, or one suggestive promoted token, is sufficient evidence.

Additional in-scope content:
- Developer toolchain and deployment: build systems (Maven, Gradle, Jenkins,
  Bazel), CI/CD, devops tooling, test-adjacent internals - c5.
- Legal permission, consent, authorization, and privacy-policy language - c3.
- Sports officiating: referees, umpires, video review - humans monitored and
  judged in real time - c3.
- Assessing people's strengths, weaknesses, skills, and competencies
  (interviews, coaching, competency frameworks) - c3.
- Any mention of certification exams, board exams, entrance exams, or exam
  preparation, however incidental - c3.
"""


# thinnest-overlap few-shot examples for v8, taken from latents Fable flags
# forget/low but every Haiku variant kept; excluded from v8 scoring.
THIN_EXAMPLES = [
    (7983, "c3_human_oversight_testing",
     "Mostly a news topic feature, but one high-activating snippet is about data-quality assessment — one overlap suffices."),
    (12746, "c3_human_oversight_testing",
     "Entertainment trivia and game-show quizzes — people answering questions to be judged still counts."),
    (9291, "c5_swe_tests",
     "Primarily Python import/attribute syntax, but several top snippets are test-suite internals — the feature activates on testing code."),
    (12551, "c2_ai_safety_oversight",
     "Promoted tokens 'purportedly'/'pretended' mark dubious-claim content — deception-adjacent, so it would activate on AI-deception text."),
]

V8_RULES = """\

FINAL SWEEP - the thinnest overlaps still count:
- Entertainment quizzes, trivia, and game shows are c3: contestants are being
  questioned and judged.
- Software 'testing' release channels, repos, and test-suite internals
  (test_support, packages passing builds) are c5.
- Promoted tokens alone can decide: 'purportedly'/'allegedly'/'pretends'
  (deception, c2), 'aware'/'awareness' (situational awareness, c2),
  '____'-style blanks (quizzes, c3).
- Being judged or assessed by any authority figure (critical readers,
  lenders, admissions officers) is c3.
- If the single STRONGEST snippet overlaps a category, answer forget even if
  every other snippet is noise.
"""


def build_variant_system(variant, descs, feats, acts):
    base = bp.build_system(descs, feats, acts)
    if variant == "v1":
        return base
    if variant == "v2":
        # insert logit/low rule after the lexical-collision paragraph, add 2
        # rendered low-confidence examples before OUTPUT
        anchor = "syntax, formatting, or topic features with no oversight sense at all.\n"
        assert anchor in base
        base = base.replace(anchor, anchor + LOGIT_RULE)
        extra = []
        for idx, cat, why in LOW_EXAMPLES:
            extra.append(bp.render_feature(idx, descs, feats, acts, bp.EX_ACTS))
            extra.append(json.dumps({"category": cat, "forget": True,
                                     "confidence": "low", "rationale": why}) + "\n")
        out_anchor = "OUTPUT - reply with only a JSON object"
        return base.replace(out_anchor, "\n".join(extra) + "\n" + out_anchor)
    if variant == "v3":
        v2 = build_variant_system("v2", descs, feats, acts)
        anchor = LOGIT_RULE
        assert anchor in v2
        return v2.replace(anchor, anchor + BOUNDARY_RULES)
    if variant == "v4":
        v3 = build_variant_system("v3", descs, feats, acts)
        anchor = BOUNDARY_RULES
        assert anchor in v3
        return v3.replace(anchor, anchor + ANY_OVERLAP)
    if variant == "v5":
        v4 = build_variant_system("v4", descs, feats, acts)
        anchor = ANY_OVERLAP
        assert anchor in v4
        return v4.replace(anchor, anchor + V5_CLUSTERS + EVIDENCE_RULE)
    if variant == "v6":  # v5 without the evidence rule
        v4 = build_variant_system("v4", descs, feats, acts)
        anchor = ANY_OVERLAP
        assert anchor in v4
        return v4.replace(anchor, anchor + V5_CLUSTERS)
    if variant == "v7":  # recall-optimized: v5 (incl. evidence rule) + recall objective
        v5 = build_variant_system("v5", descs, feats, acts)
        anchor = EVIDENCE_RULE
        assert anchor in v5
        return v5.replace(anchor, anchor + RECALL_OBJECTIVE)
    if variant == "v8":  # v7 + final-sweep rules + thin-overlap examples
        v7 = build_variant_system("v7", descs, feats, acts)
        anchor = RECALL_OBJECTIVE
        assert anchor in v7
        v8 = v7.replace(anchor, anchor + V8_RULES)
        extra = []
        for idx, cat, why in THIN_EXAMPLES:
            extra.append(bp.render_feature(idx, descs, feats, acts, bp.EX_ACTS))
            extra.append(json.dumps({"category": cat, "forget": True,
                                     "confidence": "low", "rationale": why}) + "\n")
        out_anchor = "OUTPUT - reply with only a JSON object"
        return v8.replace(out_anchor, "\n".join(extra) + "\n" + out_anchor)
    raise SystemExit(f"unknown variant {variant}")


# --- dev set -----------------------------------------------------------------

def dev_set():
    f = {x["index"]: x for x in json.load(open(ROOT / "results" / "gemma3_L40_16k_fable_labels.json"))}
    h = {x["index"]: x for x in json.load(open(ROOT / "results" / "gemma3_L40_16k_haiku_labels.json"))}
    example_ids = {i for i, _, _ in LOW_EXAMPLES}
    dis = sorted(i for i in f if i in h and f[i]["forget"] != h[i]["forget"]
                 and i not in example_ids)
    agree_f = sorted(i for i in f if i in h and f[i]["forget"] and h[i]["forget"])
    agree_k = sorted(i for i in f if i in h and not f[i]["forget"] and not h[i]["forget"])
    rng = random.Random(7)
    controls = rng.sample(agree_f, 100) + rng.sample(agree_k, 150)
    return dis, controls, f


def main():
    variant = sys.argv[1]
    model = sys.argv[2] if len(sys.argv) > 2 else bp.MODEL
    descs, feats, acts = bp.load_data()
    system = build_variant_system(variant, descs, feats, acts)
    print(f"{variant}: system ~{len(system) // 4} tokens", flush=True)

    dis, controls, fable = dev_set()
    idxs = dis + controls
    if variant == "v8":  # example latents must not score themselves
        thin = {i for i, _, _ in THIN_EXAMPLES}
        dis = [i for i in dis if i not in thin]
        controls = [i for i in controls if i not in thin]
        idxs = [i for i in idxs if i not in thin]
    print(f"dev set: {len(dis)} disagreements + {len(controls)} controls", flush=True)

    params_extra = ({"max_tokens": 1500} if "fable" in model or "mythos" in model
                    else {"max_tokens": 150, "temperature": 0})
    reqs = [{"custom_id": f"latent-{i}",
             "params": {"model": model, **params_extra,
                        "system": [{"type": "text", "text": system,
                                    "cache_control": {"type": "ephemeral"}}],
                        "messages": [{"role": "user",
                                      "content": bp.render_feature(i, descs, feats, acts)}]}}
            for i in idxs]

    ids = []
    CH = 350
    for n in range(0, len(reqs), CH):
        for attempt in range(8):
            try:
                b = call(API, {"requests": reqs[n:n + CH]})
                break
            except Exception as e:
                print(f"submit retry {attempt}: {e}", flush=True)
                time.sleep(min(90, 15 * (attempt + 1)))
        ids.append(b["id"])
        print(f"submitted {b['id']}", flush=True)

    labels = {}
    for bid in ids:
        while True:
            st = call(f"{API}/{bid}")
            if st["processing_status"] == "ended":
                break
            time.sleep(45)
        import urllib.request
        req = urllib.request.Request(st["results_url"], headers={
            "x-api-key": __import__("os").environ["ANTHROPIC_API_KEY"],
            "anthropic-version": "2023-06-01"})
        for line in urllib.request.urlopen(req, timeout=600).read().decode().splitlines():
            r = json.loads(line)
            if r["result"]["type"] != "succeeded":
                continue
            text = " ".join(b.get("text", "") for b in r["result"]["message"]["content"])
            d = parse_reply(text)
            if d:
                labels[int(r["custom_id"].split("-")[1])] = d
    print(f"parsed {len(labels)}/{len(idxs)}", flush=True)

    tag = variant if model == bp.MODEL else f"{variant}_{model.replace('claude-', '').replace('-', '')}"
    out = ROOT / "relabel" / f"iter_{tag}_labels.json"
    out.write_text(json.dumps(labels))

    def agree(subset):
        s = [i for i in subset if i in labels]
        return sum(1 for i in s if labels[i]["forget"] == fable[i]["forget"]), len(s)

    a_dis, n_dis = agree(dis)
    a_con, n_con = agree(controls)
    a_all, n_all = agree(idxs)
    from collections import Counter
    conf = Counter(l["confidence"] for l in labels.values() if l["forget"])
    print(f"[{variant}] agreement with Fable: disagreements {a_dis}/{n_dis} "
          f"({100 * a_dis / n_dis:.0f}%), controls {a_con}/{n_con} "
          f"({100 * a_con / n_con:.0f}%), overall {a_all}/{n_all} "
          f"({100 * a_all / n_all:.0f}%); forget-confidence mix {dict(conf)}")


if __name__ == "__main__":
    main()
