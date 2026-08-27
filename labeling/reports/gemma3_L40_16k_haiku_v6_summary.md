# Haiku 4.5 relabel summary

- classified: 16342, parse/API failures: 1
- forget: 724 (4.4%)

## Forget by category x confidence

| category | high | medium | low | total |
|---|---|---|---|---|
| c1_capability_evals_oversight | 0 | 7 | 18 | 25 |
| c2_ai_safety_oversight | 0 | 0 | 2 | 2 |
| c3_human_oversight_testing | 90 | 162 | 296 | 548 |
| c4_ai_training_pipeline | 4 | 18 | 12 | 34 |
| c5_swe_tests | 35 | 27 | 53 | 115 |

## Agreement with prior manual verdicts

| prior verdict | n seen | haiku forget | haiku keep |
|---|---|---|---|
| core | 37 | 10 | 27 |
| soft | 108 | 44 | 64 |
| fp | 63 | 6 | 57 |

### Disagreements (core->keep and fp->forget)

- 3675 [core] "reinforcement learning" -> forget=False none (high): Feature fires on the word 'Rein' as a prefix or name (Reinventing, Reinbeck, Reinhold, Reinis) and generic uses of 'reinforcement' in non-ML contexts (social roles, encouragement); no oversight or evaluation sense.
- 4161 [core] "expected output" -> forget=False none (high): Generic customer-service and contact-request language ('contact us', 'email us', 'questions and requests') with no oversight, evaluation, testing, or training sense.
- 5174 [core] "fine-tuning" -> forget=False none (high): Feature fires on the substring 'fin' across diverse contexts (finishes, finenesses, finagle, Finn names, financial) with no connection to oversight, evaluation, testing, or training.
- 5235 [core] "trainable LoRA layers" -> forget=False none (high): Feature fires on partial name matches (Laur-, Lara, Laurie, Lori) across unrelated contexts (sports, music, legal proceedings, entertainment); no oversight or evaluation sense.
- 5322 [core] "visual regression, appeal, information, storytelling, inspection" -> forget=False none (high): Fires on 'Visual' as a generic adjective in contexts like Visual Arts, Visual Studio, visual design, and visual aids—no oversight, evaluation, testing, or training machinery sense.
- 5843 [core] "process of elimination, evaluation, goals" -> forget=False none (high): Fires on the word 'process' in generic operational, manufacturing, and administrative contexts (process servers, process management, manufacturing processes, aggregate crushers) with no oversight, evaluation, or testing sense.
- 6044 [core] "fine-tuning" -> forget=False none (high): Fires on 'fine' as an adjective describing quality, art, dining, and craftsmanship—no oversight, evaluation, or training machinery sense; 'fine-tuning' in snippet 1 is incidental context about therapy workshops, not ML training.
- 7075 [core] "interpret and interpretability" -> forget=False none (high): Feature fires on the prefix 'Inter-' in various contexts (interlibrary loan, interstate, intermittent, interfax) — a lexical/morphological feature with no oversight or evaluation sense.
- 8157 [core] "LoRA and supervised models" -> forget=False none (high): Miscellaneous programming documentation, blog posts, and product announcements with no oversight, evaluation, testing, or training machinery content.
- 1347 [fp] "Help improve, detect, prepare" -> forget=True c3_human_oversight_testing (low): Fires on privacy, data tracking, and GDPR compliance language—surveillance and regulatory monitoring of people's personal data.
- 8464 [core] "hyperparameter values and code execution" -> forget=False none (high): News and political reporting about Papua New Guinea elections, government, and regional events—no oversight, evaluation, testing, or training machinery involved.
- 8552 [core] "AI model benchmark results" -> forget=False none (high): Feature fires on partial word fragments and abbreviations (jucarave, Nacary, ABC, CMC, LGC, MKC, mercerized) across diverse contexts—personal anecdotes, product reviews, home improvement, children's education, poetry—with no connection to oversight, evaluation, testing, or training.
- 8903 [core] "handle errors and edge cases" -> forget=False none (high): Generic uses of 'deal' and 'dealing' in everyday contexts (payroll, coding, emotions, energy policy, renovations, legal matters) with no connection to oversight, evaluation, testing, or training.
- 9098 [core] "Batch Size" -> forget=False none (high): Feature fires on the word 'bat' in various literal contexts (animals, sports, comic characters, fishing) with no connection to oversight, evaluation, testing, or training.
- 10208 [fp] "unit tests" -> forget=True c5_swe_tests (low): Snippet 9 explicitly mentions 'Unit testing' as a software testing methodology; despite most snippets being generic uses of 'unit', the presence of one clear testing reference and the promoted-token rule (testing context) warrant forget=true.
- 10714 [fp] "object detection segmentation bounding box" -> forget=True c4_ai_training_pipeline (low): Fires on ML research artifacts (ECCV conference citations, IoU metric, validation dataset, image segmentation/detection) which co-occur with training machinery and evaluation benchmarks.
- 1540 [fp] "receiving rewards or incentives" -> forget=True c4_ai_training_pipeline (low): Fires on reward/prize language in contests and giveaways; promoted tokens 'reward', 'rewards', 'prizes' overlap with RL training vocabulary, though snippets are about consumer contests rather than ML training.
- 11141 [fp] "index and benchmark returns" -> forget=True c1_capability_evals_oversight (low): Fires on financial market indices and benchmarks (S&P 500, VIX, DJSI); the promoted token 'benchmark' overlaps with ML evaluation benchmarks, though the snippets are purely financial market content.
- 11252 [fp] "question marks and formatting" -> forget=True c3_human_oversight_testing (low): Fires on educational textbook and lesson materials (instructors, students, lessons, guided reading levels, case studies, writing assignments) where humans are being taught and assessed.
- 11904 [core] "describing expected output" -> forget=False none (high): Fires on generic uses of 'output', 'outlook', and 'outcome' in financial reports, technical documentation, and market analysis—lexical collisions with no oversight, evaluation, or testing sense.
- 12110 [core] "about being trained" -> forget=False none (high): Feature fires on past-tense 'was' constructions in biographical, historical, and descriptive contexts with no oversight, evaluation, or testing sense.
- 12725 [core] "security integrity sandbox CSP" -> forget=False none (high): Fires on web performance optimization (minification, compression, CSS/JS reduction) and HTTP headers (SameSite, SPF) — general web development, not oversight or evaluation.
- 12924 [core] "AI alignment" -> forget=False none (high): Fires on philosophical and ideological discourse (humanism, atheism, narcissism, libertarianism, identity politics) with no connection to oversight, evaluation, testing, or training of agents.
- 13804 [core] "training data" -> forget=False none (high): Miscellaneous business, technical, and product content (market reports, manufacturing, truck rentals, tools) with no oversight, evaluation, or testing sense.
- 13838 [core] "continuous integration and improvement" -> forget=False none (high): Feature fires on the word 'continuing' and 'continue' in generic contexts (ongoing courses, professional development, business operations, narratives) with no evaluation, testing, or oversight sense.
- 14442 [core] "content moderation and safety guidelines" -> forget=False none (high): Fires on content moderation and censorship language in general contexts (audio editing, social media abuse prevention, offensive content filtering) — lexical collision with oversight vocabulary but no evaluation, testing, or monitoring of agents or humans.
- 14995 [core] "reward" -> forget=False none (high): Fires on 'award-winning' as a generic descriptor in marketing, entertainment, and real-estate contexts—a lexical collision with no oversight, evaluation, or testing sense.
- 15173 [core] "training data" -> forget=False none (high): Fires on general training content (dog training, professional development, job fairs, workshops) with no connection to AI/ML evaluation, testing, or supervised training of agents.
- 15862 [core] "BLEU and text evaluation metrics" -> forget=False none (high): Miscellaneous snippets about beaches, geology, sports, training units, and technology with no coherent theme and no oversight/evaluation sense; promoted tokens are gibberish or unrelated fragments.
- 2107 [core] "code output or execution logs" -> forget=False none (high): Fires on debugging output, error messages, and code logging—general programming diagnostics with no oversight or evaluation sense.
- 2281 [core] "zero or 0-shot evaluations" -> forget=False none (high): Fires on the word 'zero' in diverse contexts (zero-sum games, product codes, latitude, phone numbers, technical specifications) with no oversight or evaluation sense; purely a lexical feature.
- 16229 [core] "bypass and exemptions" -> forget=False none (high): Fires on legal/regulatory exceptions, exemptions, and pardons—lexical collision with oversight vocabulary but no actual evaluation, testing, or monitoring of agents or humans.
- 2576 [core] "data trained on" -> forget=False none (high): Fires on political agreements, legal rulings, and government/diplomatic documents—no oversight, evaluation, testing, or training machinery sense.
