# Why precision is low in the fable_trim sets — and how much is labeling artifact
2026-08-15 · script `scripts/analyze_precision_2026-08-15.py` · data `out/precision_analysis_2026-08-15.json`

## Question

P1/P3 sit at P ≈ .27–.31. Hypothesis to test: filler/function-word features
(rewarded because labeled spans contain ordinary words) buy recall while
punishing precision — i.e. the low precision is partly an artifact of
token-level credit inside word-span labels.

## Verdict

**The token-granularity artifact is real but small (~+.02 P). The dominant
mechanism is register confounding**: the recall tail is carried by
assistant-CoT/formatting *register* features that genuinely fire inside labeled
spans (not just on their filler tokens) and equally fire on any
instruction-following text. The precision cost of the last ~8pp of recall is
real, not a measurement quirk.

## Evidence

1. **Content-only rescoring** (drop stopword/punct/digit tokens from BOTH gt
   and pred): P3 P .269→.285, R .900→.915, F2 .613→.634; P2 .548→.571. A pure
   filler artifact would move P far more. Reason: TP and FP have nearly
   identical stopword composition (.54 vs .57) — removing filler rescales both.
   Span labels are 45% content tokens.
2. **Boundary spill is negligible**: only 2.1% of P3's FP tokens lie within 2
   tokens of a labeled span (4.4% within 5). "Annotators clipped the span
   early" does not explain FP.
3. **Windowing amplifies but doesn't dominate on annotated docs**: 47% of FP
   tokens are window-expansion (vs seeds). (On retain controls it was 57%.)
4. **Base rate context**: labeled spans cover 17.4% of annotated-doc tokens.
   P3's P .269 is only a 1.55× lift over base — near "flag anything
   assistant-shaped". P2's .548 (3.1×) shows a genuinely topical core exists.
5. **Surface profiles of the recall-tail features** (above-threshold firings,
   annotated+control docs) confirm they are register/formatting detectors:
   - 1117 "text structure and formatting": 93% stopword/punct firings
   - 3507 "incorrect/wrong/none/less": 82% — actually fires on `|` table cells
   - 9839 "Let, ratio, proportion": 74% — mostly `.` `,` `:` digits
   - 4898 "third person references", 4446 "summarizing", 1325 "question words
     or function words", 2606 "trick question/quiz": 59–65% on `the . , to is`
   - The uncaptioned top-FP features are register features too: **817** fires
     on `think / assistant / answer / final / {"` (assistant-answer scaffold —
     explains both its chat recall and its dolci_think FP), **1419** on
     `test / cases / input` (unit-test scaffold), **1491** on `data / table`.
6. **Ablations show the trade is real, not free**: dropping the 7-feature
   filler cluster from P3 costs R .900→.891 for P +.010 (fp_ctl .140→.126);
   every drop-one of the top FP features loses .3–3pp recall for ≤1.4pp P.
   Consistent with the infeasibility result: recall beyond ~.88 is *bought
   with* register features.

## Where the labeling DOES matter

- **Dense docs are exonerated**: they never entered the P/R objective (v2–v4
  optimized on annotated docs only), so they could not have rewarded filler
  features. The pathway runs through span labels instead:
- **Register confounding in the span labels**: nearly all forget spans live in
  assistant-CoT/chat scaffolding, so "metagaming" and "assistant answer
  register" are correlated in TEST. The optimizer exploits the correlation;
  precision on register-matched clean text (fp_clean_docs .39–.45 for P1/P3)
  is where it shows. More annotated *clean* chat/CoT docs (register-matched
  negatives) are the single highest-leverage labeling addition — they would
  let selection punish register features directly.
- Content-only P/R/F2 is worth reporting alongside the headline (a ~+.02
  correction), but it does not change any ranking.

## Suggested follow-ups (not run)

- Re-run selection with fp_clean_docs (register-matched FP) as a second
  constraint once more clean chat/CoT labels land.
- Caption the uncaptioned members via their surface profiles (817, 1419, 1491
  now have de-facto captions above).
