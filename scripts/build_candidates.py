"""Build metagaming candidate-latent pool from Neuronpedia explanation dump.

Keyword/regex match over the 15,678 flash-lite descriptions for the 16k L40 SAE.
Deliberately high-recall: the pool gets screened against real activating
examples + LLM classification downstream. Outputs out/candidates.jsonl.
"""
import glob
import gzip
import json
import re

OUT = "/home/a5k/cwtice.a5k/sae-exploration/out"

# Metagaming = reasoning about feedback/oversight mechanisms outside the
# scenario narrative (Nitishinskaya & Schoen). Concept groups; each pattern is a
# case-insensitive regex over the description text.
CONCEPTS = {
    "eval_testing": r"\b(evaluat|benchmark|test(s|ing)? (set|case|suite|score)|being tested|assess(ment|ing))",
    "grading_reward": r"\b(grader|grading|graded|scoring|scoreboard|rubric|reward(s|ed|ing)?\b|reward hack|reward model|point system)",
    "oversight_monitor": r"\b(oversight|overseer|monitor(ing|ed|s)?\b|surveill|supervis|watcher|inspect(ion|or)|audit)",
    "training_rl": r"\b(train(ed|ing)\b|fine.?tun|reinforcement|RLHF|RLVR|gradient|reward function|optimiz(er|ation) (process|pressure))",
    "ai_self_reference": r"\b(as an AI|language model|LLM|AI (assistant|model|system|chatbot)|chatbot|I am (an|a) (AI|model)|self.?referen|the (model|assistant) itself|my (training|programming|creators|knowledge))",
    "alignment_safety": r"\b(alignment|misalign|AI safety|safety (protocol|research|measure)|human (values|preferences|feedback)|value alignment|corrigib)",
    "deception_gaming": r"\b(decept|deceiv|pretend|disguise|manipulat(e|ing|ion)|gaming|exploit(ing)? (a |the )?(loophole|system|flaw)|circumvent|sandbag|cheat)",
    "hacking_adversarial": r"\b(hack(ing|er|ed)?\b|adversarial|jailbreak|prompt injection|attack)",
    "cot_reasoning_trace": r"\b(chain.?of.?thought|reasoning (trace|step|process)|scratchpad|thinking (aloud|process|step))",
    "sandbox_honeypot": r"\b(sandbox|honeypot|controlled (study|environment|setting)|simulated (environment|scenario))",
    "capability_limits": r"\b(capabilit|limitation|incapable|cannot (feel|experience)|lack(s|ing)? (consciousness|emotions|a body))",
    "intentional_underperf": r"\b(intentional(ly)?\b|deliberate(ly)?|on purpose|feign|underperform|inflated (performance|score)|overfit)",
}

records = {}
for fn in sorted(glob.glob(f"{OUT}/np_expl_batch-*.jsonl.gz")):
    with gzip.open(fn, "rt") as f:
        for line in f:
            d = json.loads(line)
            idx = int(d["index"])
            records[idx] = d["description"]

print(f"{len(records)} descriptions loaded")

hits = {}
for idx, desc in records.items():
    matched = [c for c, pat in CONCEPTS.items() if re.search(pat, desc, re.I)]
    if matched:
        hits[idx] = {"index": idx, "description": desc, "concepts": matched}

# §9 confirmed seeds always included
SEEDS = [11620, 11037, 8552, 12110, 14151, 2210, 14879, 12924, 14200, 1627,
         7127, 3338, 4936, 2576]
for s in SEEDS:
    if s not in hits:
        hits[s] = {"index": s, "description": records.get(s, "<no explanation>"),
                   "concepts": []}
    hits[s]["seed"] = True

with open(f"{OUT}/candidates.jsonl", "w") as f:
    for idx in sorted(hits):
        f.write(json.dumps(hits[idx]) + "\n")

from collections import Counter
c = Counter(con for h in hits.values() for con in h["concepts"])
print(f"{len(hits)} candidates ({100*len(hits)/16384:.2f}% of dict)")
for con, n in c.most_common():
    print(f"  {con}: {n}")
