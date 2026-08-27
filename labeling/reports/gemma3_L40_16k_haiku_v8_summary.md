# Haiku 4.5 relabel summary

- classified: 16342, parse/API failures: 1
- forget: 921 (5.6%)

## Forget by category x confidence

| category | high | medium | low | total |
|---|---|---|---|---|
| c1_capability_evals_oversight | 0 | 10 | 30 | 40 |
| c2_ai_safety_oversight | 0 | 0 | 9 | 9 |
| c3_human_oversight_testing | 99 | 183 | 385 | 667 |
| c4_ai_training_pipeline | 3 | 20 | 27 | 50 |
| c5_swe_tests | 36 | 29 | 90 | 155 |

## Agreement with prior manual verdicts

| prior verdict | n seen | haiku forget | haiku keep |
|---|---|---|---|
| core | 37 | 12 | 25 |
| soft | 108 | 47 | 61 |
| fp | 63 | 6 | 57 |

### Disagreements (core->keep and fp->forget)

- 4161 [core] "expected output" -> forget=False none (high): Generic customer-service and contact-request language across business websites—no oversight, evaluation, testing, or training content.
- 5174 [core] "fine-tuning" -> forget=False none (high): Feature fires on the substring 'Fin' in various contexts (Finlay, Finchem, Finn, finishes, fineness, finagle) — a lexical/morphological pattern with no oversight sense.
- 5235 [core] "trainable LoRA layers" -> forget=False none (high): Feature fires on partial name matches ('Laur', 'Lara', 'Laurie', 'Lori') across unrelated contexts (sports, entertainment, legal proceedings); no snippet or promoted token overlaps any oversight category.
- 5322 [core] "visual regression, appeal, information, storytelling, inspection" -> forget=False none (high): Feature fires on 'Visual' as a generic adjective in contexts like Visual Arts, Visual Studio, visual design, and visual aids—no oversight, evaluation, testing, or training machinery sense.
- 5843 [core] "process of elimination, evaluation, goals" -> forget=False none (high): The feature fires on the word 'process' in generic operational, manufacturing, and administrative contexts (server processing, pharmaceutical PAT, student transition planning, aggregate crushing, legal process serving, library management) with no connection to oversight, evaluation, testing, or training of agents.
- 6044 [core] "fine-tuning" -> forget=False none (high): All snippets use 'fine' as an adjective describing quality, dining, art, or detail in everyday contexts (restaurants, galleries, minerals, cigars); no overlap with oversight, evaluation, testing, or training machinery.
- 7075 [core] "interpret and interpretability" -> forget=False none (high): Feature fires on the prefix 'Inter-' in various contexts (interlibrary loan, intermittent, interstate, interfax) — a lexical/morphological feature with no oversight sense.
- 8157 [core] "LoRA and supervised models" -> forget=False none (high): Miscellaneous programming documentation, blog posts, and product announcements with no oversight, evaluation, testing, or training machinery content.
- 1347 [fp] "Help improve, detect, prepare" -> forget=True c3_human_oversight_testing (low): Snippets 2, 3, 6, 7, 8 discuss GDPR, cookies, data collection, and privacy-policy language—regulatory compliance and surveillance-adjacent content that falls under c3 human oversight.
- 8464 [core] "hyperparameter values and code execution" -> forget=False none (high): News and political content about Papua New Guinea elections, government, and regional affairs—no oversight, evaluation, testing, or training machinery sense.
- 8552 [core] "AI model benchmark results" -> forget=False none (high): Feature fires on partial word fragments and name abbreviations (jucarave, Nacary, ABC, CMC, LGC, MKC, mercerized) across diverse contexts (gaming, shopping, education, home improvement, poetry, crafts) with no oversight, evaluation, testing, or training sense.
- 8903 [core] "handle errors and edge cases" -> forget=False none (high): All snippets use 'deal' in the generic sense of 'handle' or 'manage' (payroll, files, guilt, energy policy, permits, personal information) with no connection to oversight, evaluation, testing, or training of agents.
- 9098 [core] "Batch Size" -> forget=False none (high): Feature fires on the word 'bat' in various contexts (baseball, animals, comic characters, equipment) with no connection to oversight, evaluation, testing, or training.
- 10208 [fp] "unit tests" -> forget=True c5_swe_tests (low): Snippet 9 explicitly mentions 'Unit testing' as a software testing methodology; the feature activates on testing content despite most snippets being generic 'unit' references.
- 10714 [fp] "object detection segmentation bounding box" -> forget=True c1_capability_evals_oversight (medium): Snippet 1 fires on IoU (Intersection over Union) and validation dataset—standard ML evaluation metrics; promoted tokens include 'Segmentation', 'Detection', 'Annotation' which co-occur with model benchmarking and capability assessment.
- 11141 [fp] "index and benchmark returns" -> forget=True c1_capability_evals_oversight (low): Promoted token 'benchmark' appears prominently; while snippets are primarily about financial indices and ETF performance, the feature would activate on ML evaluation benchmarks and leaderboards that use similar language patterns.
- 11252 [fp] "question marks and formatting" -> forget=True c3_human_oversight_testing (medium): Fires on educational textbook and lesson-plan content: instructor-led discussions, lesson structures, writing assignments, guided reading levels, and student learning materials—all intrinsic to human assessment and instruction.
- 11383 [fp] ""You" or "The" followed by assertions" -> forget=True c3_human_oversight_testing (low): Snippets 2 and 4 reference essay writing and thesis papers; snippet 5 mentions job interviews; these are educational/assessment contexts where humans are being tested or evaluated.
- 11904 [core] "describing expected output" -> forget=False none (high): Feature fires on generic uses of 'output', 'outlook', and 'outcome' in financial reports, technical documentation, and scientific writing—no oversight, evaluation, or testing sense.
- 12110 [core] "about being trained" -> forget=False none (high): Feature fires on past-tense 'was' constructions in biographical, historical, and descriptive contexts with no oversight, evaluation, or testing sense; promoted tokens ('formerly', 'born', 'originally') are temporal markers, not oversight-related.
- 12725 [core] "security integrity sandbox CSP" -> forget=False none (high): Fires on web-performance optimization (minification, compression, CSS/JS asset reduction) and security headers (SameSite, CSRF) — general web development, not oversight or evaluation.
- 12924 [core] "AI alignment" -> forget=False none (high): Fires on philosophical and ideological discourse (humanism, identity politics, narcissism, libertarianism) with no connection to oversight, evaluation, testing, or training of agents.
- 13804 [core] "training data" -> forget=False none (high): No snippet or promoted token overlaps any oversight, evaluation, testing, or training category; content spans market research, hardware specs, product shopping, and technical documentation with no evaluation or assessment sense.
- 13838 [core] "continuous integration and improvement" -> forget=False none (high): The feature fires on the word 'continuing' and its variants in generic contexts (continuing education, continuing professional development, continuing a business activity, continuing to tour) — no oversight, evaluation, testing, or training machinery sense.
- 14442 [core] "content moderation and safety guidelines" -> forget=False none (high): Fires on content moderation, censorship, and offensive language in general contexts (audio editing, social media, game design, religious sensitivity) — no oversight, evaluation, testing, or training machinery sense.
- 14995 [core] "reward" -> forget=False none (high): Feature fires on 'award-winning' as a generic marketing/descriptive phrase across design firms, entertainment, and products; no snippet or promoted token overlaps oversight, evaluation, testing, or training categories.
- 15173 [core] "training data" -> forget=False none (high): All snippets are about general professional, occupational, or skill training (dog handling, sailing, legal education, business school, job fairs, therapy workshops, driving courses) with no evaluation, testing, oversight, or ML training machinery sense.
- 2107 [core] "code output or execution logs" -> forget=False none (high): Fires on debugging output, error messages, and code logging—generic programming diagnostics with no oversight, evaluation, or testing sense.
- 2281 [core] "zero or 0-shot evaluations" -> forget=False none (high): Fires on zero-sum games, product listings, phone numbers, and geographic coordinates—no oversight, evaluation, testing, or training machinery content.
- 16229 [core] "bypass and exemptions" -> forget=False none (high): Fires on legal exceptions, exemptions, pardons, and discretionary rulings in criminal/civil law—no oversight of AI systems, human testing, ML training, or software testing.
- 2576 [core] "data trained on" -> forget=False none (high): Fires on political agreements, court rulings, and legislative/diplomatic documents—no oversight, evaluation, testing, or training machinery sense.
