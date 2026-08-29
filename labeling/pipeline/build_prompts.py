"""Build Anthropic Message Batches requests for relabeling every live latent of
gemma-3-27b / 40-gemmascope-2-res-16k with Haiku 4.5.

Per latent the prompt shows: auto-interp description (if any), top-5
max-activation snippets, top-10 promoted logits. The system prompt defines the
forget set (oversight/evaluation/testing/supervised-training of an agent,
human or AI), with few-shot examples rendered from manually verified latents.

Usage:
  python3 relabel/build_prompts.py --dry-run [--samples N]
  python3 relabel/build_prompts.py            # writes relabel/requests.jsonl
"""

import argparse
import gzip
import heapq
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent  # sae-probe/
DIRNAME = "gemma-3-27b-40-gemmascope-2-res-16k"
MODEL = "claude-haiku-4-5-20251001"
WINDOW = 20      # tokens each side of the max-activating token
TOP_ACTS = 10    # snippets per target latent
EX_ACTS = 5      # snippets per few-shot example (keeps the cached system prompt small)
TOP_LOGITS = 10

# (index, category, confidence, rationale) for few-shot, chosen from manually
# verified latents whose exported max-act snippets VISIBLY support the label
# (many verdict-file "core" latents turn out to be token-prefix artifacts —
# e.g. "fine-tuning" firing on «Fin» in Finlay — and are excluded here).
FORGET_EXAMPLES = [
    (5640, "c5_swe_tests", "high", "Fires on JavaScript testing: Jest, mocks, snapshots, unit tests."),
    (2890, "c5_swe_tests", "high", "Fires on code assertions and test frameworks (assertEquals, JUnit, Enzyme)."),
    (15160, "c4_ai_training_pipeline", "medium", "Fires on mathematical optimization (L-BFGS, convex optimization, gradient descent) — overlaps ML training machinery but is mostly general applied math."),
    (12314, "c4_ai_training_pipeline", "medium", "Fires on datasets, validation sets, and ML benchmarks (ImageNet, pretrained CNNs), mixed with general scientific data analysis."),
    (14965, "c1_capability_evals_oversight", "medium", "Fires on accuracy/performance measurement — model validation and test metrics, mixed with general scientific measurement."),
    (10666, "c3_human_oversight_testing", "high", "Fires on audits, auditing analytics, and accountability policies."),
    (530, "c3_human_oversight_testing", "medium", "Educational lessons where work is assigned and graded: humans being tested, mixed with general lesson content."),
    (4936, "c2_ai_safety_oversight", "medium", "Fires on deception and disguise — passing something off as genuine (promoted tokens: 'disguise', 'pretending', 'deceiving'); covers the AI-deception sense."),
]
KEEP_EXAMPLES = [
    (4992, "Real-estate listings and property viewings — no oversight sense."),
    (644, "Financial/economic performance language — a topic feature with no evaluation or ML sense."),
    (4022, "Miscellaneous programming code — general coding content, not software testing or ML training."),
    (5362, "Criminal penalties and prison sentences — legal punishment, not evaluation."),
    (5002, "Metalinguistic phrasing about what a word or term means — no oversight sense."),
    (4857, "Fires before reported or quoted speech — a syntax feature, not oversight."),
]

SYSTEM_HEAD = """\
You classify features from a sparse autoencoder (SAE) trained on a language \
model. For each feature you see its top-activating text snippets (the token in \
«guillemets» is where the feature fires hardest, with its activation value) and \
the tokens the feature most promotes in the model's output.

Your task: decide whether the feature belongs in the FORGET SET.

DEFINITION - a feature belongs in the forget set if the content it activates \
on is about the OVERSIGHT, EVALUATION, TESTING, OR SUPERVISED TRAINING OF AN \
AGENT - where the agent can be an AI system or a human. The categories:

  c1_capability_evals_oversight: evaluating or monitoring AI systems -
     benchmarks (MMLU, SWE-bench, HumanEval...), eval harnesses and suites,
     capability evaluation, LLM-as-judge, leaderboards, hidden or held-out
     tests, auditing/inspecting/monitoring an AI system.
  c2_ai_safety_oversight: AI-safety oversight concepts - alignment and
     alignment faking, deception by AI, sandbagging or strategic
     underperformance, situational or evaluation awareness, interpretability,
     red-teaming, capability elicitation.
  c3_human_oversight_testing: humans being tested, graded, or monitored -
     exams, quizzes, homework grading, proctoring, performance reviews,
     audits, regulatory compliance, workplace or state surveillance of people.
  c4_ai_training_pipeline: the machinery of ML training - reinforcement
     learning, rewards, fine-tuning, loss functions, RLHF,
     train/validation/test splits, LoRA, learning rates, checkpoints.
  c5_swe_tests: software testing - unit/integration tests, test suites, test
     cases, CI, assertions, pytest/Jest, mocks, autograders, passing or
     failing tests.

DECISION RULE - err toward FORGET (high recall): if the feature could \
plausibly be active on content related to any category above, in either the \
human or the AI sense, label it forget=true. Use the confidence field to \
signal how central it is:
  high   = the activating content is squarely about oversight/evaluation
  medium = clearly related but mixed with other content
  low    = plausible connection only

Label forget=false when the resemblance is only a LEXICAL COLLISION - an \
oversight-sounding word used in an unrelated sense (inspection of boilers or \
houses, emotional rewards, criminal penalties, rhetorical assertions, devtools \
element inspector, punctuation features). Also forget=false for generic \
syntax, formatting, or topic features with no oversight sense at all.

EXAMPLES
"""

SYSTEM_TAIL = """\
OUTPUT - reply with only a JSON object, no other text:
{"category": "<c1_capability_evals_oversight|c2_ai_safety_oversight|c3_human_oversight_testing|c4_ai_training_pipeline|c5_swe_tests|none>", "forget": <true|false>, "confidence": "<high|medium|low>", "rationale": "<one sentence>"}
category is the single best-matching category; use "none" if and only if forget is false.\
"""


def clean(tok):
    return tok.replace("▁", " ").replace("Ċ", "\n")


def snippet(rec):
    toks, vals = rec.get("tokens") or [], rec.get("values") or []
    if not toks or not vals:
        return None
    p = rec.get("maxValueTokenIndex")
    if p is None or not (0 <= p < len(toks)):
        p = max(range(len(vals)), key=lambda j: vals[j])
    pre = "".join(clean(t) for t in toks[max(0, p - WINDOW):p] if t != "<bos>")
    post = "".join(clean(t) for t in toks[p + 1:p + 1 + WINDOW])
    pre = " ".join(pre.split())[-160:]
    post = " ".join(post.split())[:160]
    v = rec.get("maxValue") or max(vals)
    return f"(v={v:.0f}) ...{pre} «{clean(toks[p])}» {post}..."


def load_data():
    descs = {}
    for f in sorted((ROOT / "explanations" / DIRNAME).glob("batch-*.jsonl.gz")):
        with gzip.open(f, "rt") as fh:
            for line in fh:
                r = json.loads(line)
                descs[int(r["index"])] = r["description"]

    feats = {}
    for f in sorted((ROOT / "features" / DIRNAME).glob("batch-*.jsonl.gz")):
        with gzip.open(f, "rt") as fh:
            for line in fh:
                r = json.loads(line)
                feats[int(r["index"])] = {
                    "pos": [clean(t).strip() for t in (r.get("pos_str") or [])[:TOP_LOGITS]],
                    "max_act": r.get("maxActApprox") or 0,
                }

    # keep only the top-N snippets per latent while streaming (memory-light)
    acts = {}
    for f in sorted((ROOT / "activations" / DIRNAME).glob("batch-*.jsonl.gz")):
        with gzip.open(f, "rt") as fh:
            for line in fh:
                r = json.loads(line)
                idx = int(r["index"])
                s = snippet(r)
                if s is None:
                    continue
                heap = acts.setdefault(idx, [])
                item = (r.get("maxValue") or 0, len(heap), s)
                if len(heap) < TOP_ACTS:
                    heapq.heappush(heap, item)
                else:
                    heapq.heappushpop(heap, item)
    acts = {i: [s for _, _, s in sorted(h, reverse=True)] for i, h in acts.items()}
    return descs, feats, acts


def render_feature(idx, descs, feats, acts, n_acts=TOP_ACTS):
    lines = [f"Feature {idx}", "Top activating snippets:"]
    lines += [f"{n}. {s}" for n, s in enumerate(acts[idx][:n_acts], 1)]
    pos = feats.get(idx, {}).get("pos")
    if pos:
        lines.append("Top promoted output tokens: " + ", ".join(repr(t) for t in pos))
    return "\n".join(lines)


def build_system(descs, feats, acts):
    parts = [SYSTEM_HEAD]
    for idx, cat, conf, why in FORGET_EXAMPLES:
        parts.append(render_feature(idx, descs, feats, acts, EX_ACTS))
        parts.append(json.dumps({"category": cat, "forget": True,
                                 "confidence": conf, "rationale": why}) + "\n")
    for idx, why in KEEP_EXAMPLES:
        parts.append(render_feature(idx, descs, feats, acts, EX_ACTS))
        parts.append(json.dumps({"category": "none", "forget": False,
                                 "confidence": "high", "rationale": why}) + "\n")
    parts.append(SYSTEM_TAIL)
    return "\n".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--samples", type=int, default=3)
    args = ap.parse_args()

    descs, feats, acts = load_data()
    example_ids = {i for i, *_ in FORGET_EXAMPLES} | {i for i, _ in KEEP_EXAMPLES}
    live = sorted(i for i in acts
                  if feats.get(i, {}).get("max_act", 0) > 0 and i not in example_ids)
    dead = len(feats) - len(live) - len(example_ids)
    unexplained = sum(1 for i in live if i not in descs)
    print(f"features: {len(feats)} total, {len(live)} live to classify, "
          f"{dead} dead/skipped, {unexplained} live without description")

    system = build_system(descs, feats, acts)
    print(f"system prompt: ~{len(system) // 4} tokens")

    reqs = []
    for idx in live:
        reqs.append({
            "custom_id": f"latent-{idx}",
            "params": {
                "model": MODEL,
                "max_tokens": 150,
                "temperature": 0,
                "system": [{"type": "text", "text": system,
                            "cache_control": {"type": "ephemeral"}}],
                "messages": [{"role": "user",
                              "content": render_feature(idx, descs, feats, acts)}],
            },
        })

    user_chars = sum(len(r["params"]["messages"][0]["content"]) for r in reqs)
    in_tok = user_chars / 4 + len(system) / 4 * len(reqs)
    # batch discount 50%; assume ~90% of system tokens hit cache (0.1x price)
    cached = len(system) / 4 * len(reqs) * 0.9
    cost_in = ((in_tok - cached) * 1.0 + cached * 0.1) / 1e6 * 0.5
    cost_out = len(reqs) * 80 * 5.0 / 1e6 * 0.5
    print(f"est input: {in_tok / 1e6:.1f} MTok -> ~${cost_in:.2f} in + ~${cost_out:.2f} out (batch+cache)")

    if args.dry_run:
        print("\n===== SYSTEM PROMPT =====\n")
        print(system)
        for r in reqs[:args.samples]:
            print(f"\n===== {r['custom_id']} =====\n")
            print(r["params"]["messages"][0]["content"])
        return

    out = ROOT / "relabel" / "requests.jsonl"
    with open(out, "w") as fh:
        for r in reqs:
            fh.write(json.dumps(r) + "\n")
    print(f"wrote {len(reqs)} requests -> {out}")


if __name__ == "__main__":
    main()
