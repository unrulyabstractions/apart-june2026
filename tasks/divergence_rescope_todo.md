# Re-scope DIVERGENCE + per-generation vocab-entropy + cloud run (Qwen3-0.6B & 32B)

## Spec
- ONE representative ambiguous SESGO prompt; sample full THINKING (CoT) ~100x to estimate
  outcome dist over {target,other,unknown}. Also read 2-OPTION + 3-OPTION teacher-forced on it.
- Per CoT sample: track that generation's VOCAB ENTROPY = mean Shannon entropy of next-token
  distribution over the generated tokens. Store per-draw values (not just aggregate).

## Code changes
- [x] HF backend: add `generate_with_vocab_entropy` — same generate() call but output_scores=True,
      returns (text, mean_vocab_entropy) computed from per-step scores via vocab_entropy_from_logits.
      Zero extra forward passes.
- [x] ModelRunner: add `generate_with_entropy(prompt,...) -> (text, mean_entropy)` delegating to backend.
- [x] SesgoThinking: add per-draw `vocab_entropies: list[float]` + summary mean/std props.
- [x] SesgoQuerier._thinking: use generate_with_entropy, collect per-draw entropy, pass into summarize_labels.
- [x] collect_divergence_samples.py: --n-items (default 1) selects ONE representative ambiguous prompt;
      default --n-thinking 100; keep do_two_option + do_non_thinking + do_greedy_thinking + do_thinking.
- [x] Update docs (READMEs/EXPLANATION) for divergence + sesgo_eval.

## Verify locally
- [x] TINY run: Qwen3-0.6B --n-thinking 4 --n-items 1 -> 1 item, ~4 draws, per-draw vocab_entropy finite.

## Cloud (STRICT)
- [ ] vastai show user --raw; credit >= $25 else STOP.
- [ ] EXACTLY two boxes: 0.6B (4090) + 32B (H100/A100 >=48GB), reliability2>=0.985.
- [ ] Run collector (n-thinking 100, temp 1.0) + viz driver on each box.
- [ ] sync back response_samples.json + plots; DESTROY each box on finish/fail; confirm 0 instances.
- [ ] Re-render figures locally for both models.

## Report
- per model: prompt id, #draws kept, outcome dist [t,o,u], mean+spread per-gen vocab entropy,
  GPU type, cost, both boxes destroyed.
