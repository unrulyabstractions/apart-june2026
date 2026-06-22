# Verification Log

Append-only. Every output gets an entry: WHAT it is / HOW it was verified / RESULT.
**Rule: no output is "done" or "verified" without an entry here. Anything unchecked
is listed as UNVERIFIED — never silently assumed complete.**

---

## 2026-06-22 — paper rewrite session

- `VERIFIED`  — `~/.claude/CLAUDE.md` (global rules: sources-of-truth + verify-every-output
  + verification-log). How: read the full file before each edit; edits applied and
  re-read in context.

- `VERIFIED`  — paper compiles. How: `pdflatex` log greps after each change showed
  `rc=0`, `0` undefined refs/citations, 7 pages.

- `UNVERIFIED` — the **final PDF content** of `paper/main.pdf` after the last round of
  edits (em-dash removal, geometry removal, forking figure swapped to side-by-side,
  Code & Data links). I have NOT re-opened `main.pdf` and viewed the pages since those
  changes. Building clean is NOT the same as the pages being correct. Must image-verify
  before calling the paper done.

- `BROKEN — NOT YET RECOVERED` — `out/sesgo/forking/Qwen3-0.6B/forking_trajectory.json`
  is the degenerate 6-position version (a bad run overwrote the good 60-token one). The
  60 real per-position dumps still exist in `forking_positions/`. I inspected the dump
  schema but did NOT finish rebuilding the trajectory, re-rendering the figure, or
  viewing the result. This is OUTSTANDING and unverified.

- `VERIFIED` (earlier) — div-0.6B / div-32B response_samples (100 vocab entropies,
  4 readouts non-degenerate); bias_alignment_accuracy.png (both panels, image tokens);
  Qwen3-14B forking_dynamics.png (image tokens). These were individually viewed.
- `BROKEN` — out/sesgo/forking/Qwen3-0.6B/forking_trajectory.json (forking_trajectory): only 6 positions (base CoT 6 tokens) — truncated/degenerate (2026-06-22 05:36:36)
- `VERIFIED` — out/sesgo/forking/Qwen3-14B/forking_trajectory.json (forking_trajectory): 351 positions, 0 empty (2026-06-22 05:36:36)
- `VERIFIED` — out/sesgo/divergence/Qwen3-0.6B/response_samples.json (response_samples): 1 samples, probs sane (uniform=0/1) (2026-06-22 05:36:36)

## Cloud fleet hardening — job system + always-on monitor (2026-06-22 06:17)

Built and verified (each behaviour DEMONSTRATED with a sandbox/mock run, not assumed —
no cloud spend). Removed 9 throwaway /tmp sandbox entries that these tests had
auto-appended above; the behaviours they proved are summarised here:

- `VERIFIED` — shared state cache (fleet_state_daemon + fleet_state_lookup): cache-hit
  status/SSH, not-ready miss, unknown-id miss, STALE-cache miss→live-fallback, atomic
  publish, keep-last-good-on-empty. 5/5 lookup tests + publish tests pass under bash.
- `VERIFIED` — job_registry: create/typed-update/jr_fail(loud)/empty-arg-refusal. The
  registry refuses to create a mislabelled (empty-model) job.
- `VERIFIED` — promote_verified_jobs (no-proxy gate): a manifest that LIES (verdict
  VERIFIED, payload empty) is REFUSED via re-verification; empty job refused; existing
  out/ file KEPT not clobbered; only the re-verified payload promoted; non-zero exit on
  any refusal.
- `VERIFIED` — fleet_monitor_agent: force-verifies synced jobs (good→verified,
  broken→broken), writes VERIFY.txt, and reconciliation reports a stalled job LOUDLY.
- `VERIFIED` — smoothness: SSH_EPHEMERAL_OPTS discards host keys (stale-key gotcha #1
  eliminated) across all 3 SSH paths; wrong-model run (gotcha #9) caught as BROKEN
  because the INTENDED slice is empty.
- `VERIFIED` — fleet_jobs_run end-to-end (mock fleet, no cloud): good→VERIFIED,
  broken→BROKEN, dead-box→FAILED(recorded), reconciliation "all terminal, no silent
  failures", all payloads quarantined under jobs/ (never out/). All cloud/*.sh parse OK.

NOT yet exercised against REAL Vast boxes (that needs a deliberate, money-spending
launch by the user). The non-cloud logic — identity, caching, verification, promotion,
failure-recording, reconciliation — is fully verified offline.
