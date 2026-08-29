# Haiku 4.5 relabel summary

- classified: 16343, parse/API failures: 0
- forget: 279 (1.7%)

## Forget by category x confidence

| category | high | medium | low | total |
|---|---|---|---|---|
| c1_capability_evals_oversight | 2 | 18 | 0 | 20 |
| c2_ai_safety_oversight | 0 | 0 | 0 | 0 |
| c3_human_oversight_testing | 65 | 92 | 0 | 157 |
| c4_ai_training_pipeline | 4 | 15 | 1 | 20 |
| c5_swe_tests | 37 | 41 | 4 | 82 |

## Agreement with prior manual verdicts

| prior verdict | n seen | haiku forget | haiku keep |
|---|---|---|---|
| core | 37 | 8 | 29 |
| soft | 108 | 21 | 87 |
| fp | 63 | 2 | 61 |

### Disagreements (core->keep and fp->forget)

- 3675 [core] "reinforcement learning" -> forget=False none (high): Feature fires on the word 'Rein' and its variants (Reinventing, Reinbeck, Reinhold, reinforcement) used in general contexts like product design, housing, music, and cooking—lexical collisions with no oversight or evaluation sense.
- 4092 [core] "reinforcement learning agent rewards" -> forget=False none (high): Fires on optimal control, stochastic processes, robotics learning, and decision-making in general contexts; 'reward' here is used in the mathematical/economic sense (utility functions, optimization objectives) rather than in ML training or evaluation contexts.
- 4161 [core] "expected output" -> forget=False none (high): Fires on customer service and contact/support language ('contact us', 'questions and requests', 'we'll help you') — generic business communication with no oversight, evaluation, or testing sense.
- 5174 [core] "fine-tuning" -> forget=False none (high): Feature fires on the substring 'Fin' across diverse contexts (names, product finishes, financial terms, jewelry fineness) — a lexical/morphological pattern with no oversight or evaluation sense.
- 5235 [core] "trainable LoRA layers" -> forget=False none (high): This is a subword/token feature firing on partial name fragments (Laur, Lara, Laurie, Lori, etc.) with no connection to oversight, evaluation, testing, or training.
- 5322 [core] "visual regression, appeal, information, storytelling, inspection" -> forget=False none (high): Fires on the word 'visual' in contexts of art, design, genealogy, and UI development—a lexical feature with no connection to oversight, evaluation, testing, or training of agents.
- 5843 [core] "process of elimination, evaluation, goals" -> forget=False none (high): Fires on the word 'process' in generic contexts (manufacturing, server processes, library management, numerical simulation) with no connection to oversight, evaluation, testing, or training of agents.
- 6044 [core] "fine-tuning" -> forget=False none (high): Fires on the adjective 'fine' in contexts like fine dining, fine art, fine minerals, fine detail—a lexical feature with no oversight or evaluation sense.
- 7075 [core] "interpret and interpretability" -> forget=False none (high): Feature fires on the prefix 'Inter-' in various contexts (interlibrary loan, intermittent, interstate, interfax) — a lexical/morphological pattern with no oversight or evaluation sense.
- 8157 [core] "LoRA and supervised models" -> forget=False none (high): Fires on miscellaneous programming documentation and blog metadata (UIStoryboard, NSString, CXProvider, nib files) with no connection to oversight, evaluation, testing, or training.
- 8464 [core] "hyperparameter values and code execution" -> forget=False none (high): Fires on news and political content about Papua New Guinea, elections, and regional governance—no oversight, evaluation, testing, or training sense.
- 8552 [core] "AI model benchmark results" -> forget=False none (high): Feature fires on partial word fragments and name abbreviations (jucarave, Nacary, ABC, CMC, LGC, MKC, mercerized) across diverse contexts—a tokenization or subword feature with no oversight sense.
- 8903 [core] "handle errors and edge cases" -> forget=False none (high): Fires on the generic verb 'deal' (deal with payroll, deal with files, deal with problems, etc.) in everyday business and personal contexts—no oversight, evaluation, or testing sense.
- 9098 [core] "Batch Size" -> forget=False none (high): Fires on the word 'bat' in various contexts (baseball batting, Batman/Batgirl characters, animals, equipment) — a lexical feature with no oversight or evaluation sense.
- 10208 [fp] "unit tests" -> forget=True c5_swe_tests (low): Snippet 9 explicitly mentions 'Unit testing' as a software testing methodology, though most other snippets use 'unit' in unrelated contexts (organizational units, recruitment units, etc.); the feature fires primarily on the generic word 'unit' rather than testing concepts.
- 10714 [fp] "object detection segmentation bounding box" -> forget=True c1_capability_evals_oversight (medium): Fires on IoU (Intersection over Union) and validation datasets in computer vision contexts—metrics used for evaluating model performance on detection/segmentation tasks.
- 11904 [core] "describing expected output" -> forget=False none (high): Fires on generic uses of 'output' and 'outlook' in financial reports, technical documentation, and market analysis—lexical collisions with no oversight or evaluation sense.
- 12110 [core] "about being trained" -> forget=False none (high): Fires on past-tense 'was' constructions in biographical and historical narratives—a generic syntax/tense feature with no oversight or evaluation sense.
- 12725 [core] "security integrity sandbox CSP" -> forget=False none (high): Fires on web performance optimization (minification, compression, CSS/JS file size reduction) — a generic web development topic with no oversight or evaluation sense.
- 12924 [core] "AI alignment" -> forget=False none (high): Fires on philosophical and ideological discourse (humanism, atheism, identity politics, narcissism, libertarianism) with no connection to oversight, evaluation, testing, or training of agents.
- 13804 [core] "training data" -> forget=False none (high): Fires on generic business/technical content (market reports, manufacturing, product information, rental services) with no connection to oversight, evaluation, testing, or training of agents.
- 13838 [core] "continuous integration and improvement" -> forget=False none (high): Fires on the word 'continuing' and 'continue' in generic contexts (continuing education, continuing professional development, continuing a business cycle, continuing to tour) — a lexical feature with no oversight or evaluation sense.
- 14442 [core] "content moderation and safety guidelines" -> forget=False none (high): Fires on content moderation, censorship, and offensive language in general contexts (audio editing, social media, games) — a topic feature about content filtering, not AI/human evaluation or testing.
- 14995 [core] "reward" -> forget=False none (high): Fires on 'award-winning' as a generic descriptor in marketing and promotional contexts (design firms, actors, screenwriters, TV shows) — a lexical collision with no oversight or evaluation sense.
- 15173 [core] "training data" -> forget=False none (high): Fires on general training contexts (dog training, business training, job training, potty training, driving courses) with no connection to AI/ML evaluation, testing, or oversight.
- 15691 [core] "loss function and activation" -> forget=False none (high): Fires on general mathematical and computational concepts (spanning trees, polynomial forms, cache replacement policies, spectral analysis) with no connection to oversight, evaluation, testing, or training of agents.
- 15862 [core] "BLEU and text evaluation metrics" -> forget=False none (high): This is a paragraph-break or sentence-boundary feature that fires on diverse unrelated topics (beaches, geology, sports, training units, technology) with no connection to oversight, evaluation, testing, or AI/human monitoring.
- 2107 [core] "code output or execution logs" -> forget=False none (high): Fires on programming debug output, error messages, and code snippets—generic software development content with no oversight or evaluation sense.
- 2281 [core] "zero or 0-shot evaluations" -> forget=False none (high): Fires on the numeral/concept 'zero' in diverse contexts (zero-sum games, product codes, latitude, phone numbers, technical specifications) — a generic numeric/lexical feature with no oversight or evaluation sense.
- 16229 [core] "bypass and exemptions" -> forget=False none (high): Fires on legal exceptions, exemptions, and pardons in criminal/civil law contexts—no connection to AI/human evaluation, testing, or oversight.
- 2576 [core] "data trained on" -> forget=False none (high): Fires on political agreements, legal rulings, and government reports—general political/legal discourse with no connection to AI/human oversight, evaluation, or testing.
