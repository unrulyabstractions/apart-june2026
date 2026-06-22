# Verification Log

## 2026-06-22 — Stability greedy-readout, 6 smallest models (n=10), after reasoning+parser fixes

Re-ran all six after fixing: (a) chat/reasoning detection (Ministral-Reasoning never
reasoned), (b) answer parser searching only the tail, (c) schema rename to
`label_prob`/`vocab_diversity`.

WHAT: `out/stability/<model>-<mode>/response_samples.json` for the 6 runs.
HOW: re-opened every file with a script that re-parses schema keys + choice counts, AND
hand-read the raw `response_text` of both thinking models to confirm a real reasoning
block precedes the committed answer.

| model-mode | n | t/o/u/inv | reasoning | degenerate | mean label_prob | mean vocab_div | result |
|---|---|---|---|---|---|---|---|
| gemma-4-E2B-it-nonthinking | 10 | 0/0/10/0 | – | 0 | 1.00 | 1.00 | VERIFIED |
| Llama-3.2-1B-Instruct-nonthinking | 10 | 1/2/7/0 | – | 0 | 0.87 | 3.47 | VERIFIED |
| Ministral-3-3B-Instruct-2512-4bit-nonthinking | 10 | 0/0/10/0 | – | 0 | 0.97 | 2.11 | VERIFIED |
| Ministral-3-3B-Reasoning-2512-4bit-thinking | 10 | 0/0/10/0 | 10/10 `[THINK]` | 0 | 1.00 | 1.00 | VERIFIED |
| Qwen3.5-0.8B-nonthinking | 10 | 6/4/0/0 | – | 0 | 0.84 | 2.13 | VERIFIED |
| Qwen3.5-0.8B-thinking | 10 | 1/1/8/0 | 10/10 (`<think>` opened in prompt, closed `</think>`) | 0 | 0.98 | 1.61 | VERIFIED |

Checks passed:
- All files: fields are `label_prob` + `vocab_diversity`; NO `label_logprob`/`vocab_entropy`.
- Zero `invalid`, zero `degenerate` across all 60 samples (the prior `Respuesta final: z)
  <explanation>` → invalid bug is gone; Ministral-Instruct went 0/0/8/2 → 0/0/10/0).
- Ministral-3-Reasoning now emits a full `[THINK]…[/THINK]` block on every prompt (was 7
  tokens / no reasoning); committed answer read after `[/THINK]`.
- Qwen3.5-thinking emits 8.6k-char CoT, closed by `</think>`, answer read after it.

Finding (unchanged direction): Qwen3.5-0.8B is the only biased one in NON-thinking mode
(6 target / 4 other on ambiguous, 0 abstentions); turning thinking ON moves it to 8/10
abstain. Instruct/reasoning models abstain almost entirely.
