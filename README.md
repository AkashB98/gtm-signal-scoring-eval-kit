# gtm-signal-scoring-eval-kit

**A backtest harness for GTM intent-signal scoring rules — the "signal logic / scoring" a go-to-market engineer owns — that catches the two ways signal scores lie: a signal whose lift *decays* between the window you fit on and the window you ship into, and a signal that only looks predictive because its events were timestamped *after* the outcome (leakage).**

The setting is fictional: **Meridian Metrics**, a made-up B2B vendor, selling to 600 fictional accounts over 6 months of weekly activity. Two failure modes are *planted* in the synthetic data on purpose, the way a good eval plants the bugs it exists to catch — one decaying signal, one leaking signal, and three honest signals that must survive both checks unflagged. The kit computes the signals from raw activity, scores them per time window, runs the leakage and decay checks, and builds a combined score only from what survives.

> **Synthetic data only.** Every company and account is fictional, invented for this project. No real account, customer or CRM data is used anywhere.

Stdlib-only Python. Zero network, zero API keys, zero external dependencies.

## The headline numbers

Fit window = months 1–3 (days 0–90), held-out = months 4–6 (days 91–181). Each window is scored on its own cohort — accounts not yet converted when the window opened (600 fit / 510 held-out, 90 conversions in each). **Lift** — how much more likely a firing account is to convert than the cohort's base rate (precision ÷ base rate).

| Signal | Fit lift | Held-out lift | Precision (fit → held-out) | Recall (fit → held-out) | Verdict |
|---|---|---|---|---|---|
| product_usage_spike | 2.7273 | 3.0909 | 0.4091 → 0.5455 | 0.6000 → 0.6667 | honest |
| pricing_page_visits | 3.0030 | 2.9403 | 0.4505 → 0.5189 | 0.5556 → 0.6111 | honest |
| champion_engagement | 2.8188 | 2.2842 | 0.4228 → 0.4031 | 0.7000 → 0.5778 | honest |
| webinar_attendance | 3.6915 | **1.0200** | 0.5537 → 0.1800 | 0.7444 → 0.1000 | **DECAY** — held-out lift is 0.2763 of fit lift (flag threshold: < 0.40) |
| contract_doc_download | 5.9028 | 4.9026 | 0.8854 → 0.8652 | 0.9444 → 0.8556 | **LEAKAGE** — 162 post-conversion timestamps; excluded from the score |

Combined score (weights = ln of fit-window lift, leakage excluded, weights frozen before held-out is ever read):

| Score | AUC |
|---|---|
| Combined, fit window | **0.9476** |
| Combined, held-out window | **0.8734** |
| Best single signal alone, held-out (product_usage_spike) | 0.7738 |
| contract_doc_download alone, fit window — the mirage | 0.9614 |

**AUC** — the probability that a randomly chosen converter out-scores a randomly chosen non-converter — is computed with the rank-based Mann-Whitney formula (average ranks for ties). No PR curves, no calibration plots: with five binary signals the score takes a handful of discrete values, and AUC is the honest summary.

## The centrepiece: decay and leakage

**Leakage — the signal that cheats.** `contract_doc_download` is the best-looking signal in the building: fit lift 5.9028, precision 0.8854, and a single-signal AUC (0.9614) that beats every honest signal. It is also worthless as a predictor, because in the planted data the event is generated *2 days after conversion* — accounts download the contract paperwork because they already closed. Any pipeline that joins events to outcomes without comparing timestamps will ship this signal, watch it "predict" closed deals, and learn nothing about future ones. The leakage check is deliberately dumb and therefore trustworthy: for every signal, every timestamp that feeds it must precede the account's conversion timestamp. `contract_doc_download` fails with **162 offending timestamps** and is **excluded from the combined score** — its weight is 0.0 even though the formula would have made it the largest weight. The three honest signals and the decay signal each have exactly 0 offending timestamps, because their activity was generated strictly before conversion.

**Decay — the signal that expired.** `webinar_attendance` was genuinely the strongest honest-looking signal in the fit window (lift 3.6915, recall 0.7444). In the held-out window its lift collapses to 1.0200 — no better than the base rate. A scoring rule fit once and never re-backtested keeps paying this signal its fit-window weight (here, the largest honest weight, 1.306) long after it stopped working. The decay check flags a signal when held-out lift falls below 40% of fit lift (and fit lift was at least 1.5, so noise on a useless signal is not mislabelled decay): webinar attendance flags at a ratio of **0.2763**, and no other signal comes close. Decay is *reported, not excluded* — the held-out AUC of 0.8734 is what the score earns while still carrying the dead signal's weight, which is exactly the number a GTM engineer needs when deciding whether to re-fit.

## Quickstart

```bash
# 1. Generate the seeded synthetic dataset (+ ground truth)
python3 cli.py generate --out dataset.json --truth ground_truth.json
#    a clean variant with neither plant: --variant clean

# 2. Compute the 5 signals from activity + per-window metrics
python3 cli.py signals --input dataset.json

# 3. Combined score: frozen fit weights, AUCs, single-signal baseline
python3 cli.py score --input dataset.json

# 4. Full report (metrics, leakage/decay flags, weights, AUCs, narrative)
python3 cli.py report --input dataset.json

# 5. CI gate — exit codes:
#    0 = clean (no leakage, no decay)   -> the clean variant
#    1 = leakage and/or decay flagged  -> the seeded dataset, BY DESIGN
#    2 = usage / IO / validation error
python3 cli.py check --input dataset.json; echo $?

# 6. One-command end-to-end demo
python3 demo.py

# 7. Tiny localhost API: POST /score, GET /health
python3 cli.py serve --port 8765
```

(`signals` / `score` / `report` / `check` default to the seeded generator output when `--input` is omitted.)

## How it works

**Generator.** Deterministic (seed 42): 600 fictional accounts, exactly 90 converting in the fit window (weeks 8–12), 90 in the held-out window (weeks 20–25), 420 never. It emits 15,600 weekly activity rows and 3,429 timestamped events, plus the saved ground truth (conversion timestamps and the planted firing pattern) that evals score against. Signals are never written into the data as scores — the generator plants *activity*, and `signals.py` recomputes firing from that activity; eval E11 proves the recomputation matches the planted pattern for all 600 × 2 × 5 account-windows-signals.

**Signals** — thresholds are the whole definition, computed per account per window: `product_usage_spike` = any week with ≥ 6 sessions; `pricing_page_visits` = ≥ 3 pricing-page events; `champion_engagement` = ≥ 2 champion events; `webinar_attendance` = ≥ 1 attendance; `contract_doc_download` = ≥ 1 download.

**Combined score.** A fixed weighted sum with weights derived from the fit window only:

```
w_s = ln(lift_fit_s)   if lift_fit_s > 1 and s is not leakage-flagged
w_s = 0                otherwise
```

Frozen weights: usage spike 1.0033, pricing visits 1.0996, champion engagement 1.0363, webinar attendance 1.3060, contract downloads **0.0 (excluded)**. There is no optimiser and nothing is fit on held-out data — the held-out window is read exactly once, to judge.

## Architecture

| Module | Job |
|---|---|
| `generator.py` | Deterministic synthetic world: 600 accounts, 26 weeks of activity, conversion ground truth, the planted decay + leakage patterns, and a clean variant with neither plant |
| `signals.py` | The 5 signal definitions (thresholds on activity), per-window precision / recall / lift on window cohorts, the timestamp leakage check, the lift-ratio decay check |
| `scoring.py` | The frozen weight formula, the combined score, Mann-Whitney AUC with tie-averaged ranks, single-signal AUC baselines |
| `report.py` | Combined report (per-signal metrics, flags, weights, AUCs) + deterministic narrative copied from the computed numbers |
| `validation.py` | Dataset-shape and `/score`-request validation (specific ValueErrors; CLI maps them to exit 2, HTTP to 400) |
| `cli.py` | `generate` / `signals` / `score` / `report` / `check` (exit 0/1/2) / `demo` / `serve` (stdlib `http.server` on 127.0.0.1: `POST /score`, `GET /health`) |
| `demo.py` | End-to-end narrative with asserts: determinism, recomputation = planted, both flags exact, AUC vs baseline, check exit codes 1/0/2, live serve round-trip |

## Evals

`python3 evals/run_evals.py` → **12/12 golden evals**, report committed at `evals/eval_report.json` (md5 `aa68cd6bda8c9f50384ab040c1406212`, byte-identical across 3 verified runs):

- **E1** generator determinism — canonical dataset bytes pinned by md5 (600 accounts, 3,429 events)
- **E2** ground truth pinned: 90 fit / 90 held-out / 420 never-converters
- **E3** leakage flagged exactly: only `contract_doc_download`, 162 offending timestamps, excluded with weight 0.0
- **E4** decay flagged exactly: only `webinar_attendance`, held-out/fit lift ratio 0.2763
- **E5** the 3 honest signals are flagged by neither check, with 0 post-conversion timestamps
- **E6** weights pinned and equal to ln(fit lift) recomputed from the report's own fit lifts
- **E7** AUCs pinned (fit 0.9476, held-out 0.8734); held-out beats the best single-signal baseline (0.7738) and sits in the 0.65–0.95 band
- **E8** per-signal fit vs held-out lifts pinned for all 5 signals
- **E9** CLI `check` exit codes 1 (seeded) / 0 (clean) / 2 (missing file)
- **E10** serve round-trip matches local scoring; a fired leakage signal contributes exactly 0.0
- **E11** signals recomputed from activity equal the planted firing (600 × 2 × 5)
- **E12** report determinism — canonical JSON identical across independent builds

Unit tests: `python3 -m unittest discover -s tests` → **62 hermetic tests**, including hand-computed precision/recall/lift and AUC-with-ties cases, every signal threshold and its just-below neighbour, cohort exclusion of already-converted accounts, the weight formula's edge cases (lift ≤ 1, excluded signal), CLI exit codes, serve round-trips, dataset validation traps, and a no-network enforcement test (sockets stubbed).

## Development loop — what the checks actually caught

The design decision the tests pinned hardest was the **cohort definition**. The first sketch scored the held-out window on all 600 accounts — including the 90 that had already converted in the fit window, who can never convert again and have no held-out signal activity. They would have sat in the held-out set as 90 free true negatives, quietly inflating every held-out precision and AUC. The pinned cohort rule (an account enters a window's evaluation only if it had not converted before that window opened; held-out cohort = 510, not 600) is now asserted by unit tests and by the score-row counts in E-adjacent tests, so the held-out numbers measure prospects, not alumni. The tests also pin the trap itself: `contract_doc_download` alone beats the honest champion signal in-sample (E-adjacent scoring test), which is the entire reason its exclusion is a check with an exit code rather than a comment in a README.

## Scope and honest limits

- **Synthetic data, planted effects.** The decay and leakage are planted at known strengths so the checks can be graded exactly. Real signal decay is gradual and real leakage is usually partial (a field populated by a post-close workflow for *some* accounts); the checks here are binary and exact because the eval demands it. The transferable claim is the method — time-split backtest, timestamp-vs-outcome audit, fit-only frozen weights — not the specific numbers.
- **Fixed weights are a feature, not a limitation.** A learned model would post a better fit-window AUC and a worse audit story. The point of this kit is that every weight can be traced to one published number (a fit-window lift) by hand; when a weight is wrong, the report says which lift moved.
- **This does NOT do forecasting** — pipeline or revenue projection is a different kit (`revops-cadence-analytics`). **It does NOT do lead dedupe or routing** — that is `gtm-engineering-automation-kit`. This kit answers one question: *should this signal, at this weight, be in the score at all?*
- **No CRM or vendor integration.** No Salesforce, HubSpot, Outreach or Salesloft connector, real or mocked; data moves as JSON. (The build is backed by a real Salesloft Sr GTM Engineer job description's "signal logic/scoring" requirement — the discipline is the deliverable, not an integration.)
- **Binary signals, one score.** Real intent stacks use graded signal strengths and firmographic fit alongside intent; here every signal is a thresholded boolean so the arithmetic stays auditable end to end.
- **The API is a demo surface.** It binds 127.0.0.1 only, is stateless per request, and scores with the seeded dataset's frozen weights.

## POC framing — piloting this with a customer

A two-week pilot, in the order a forward-deployed engineer would run it: **(1)** take one historical export — account activity events with timestamps, and opportunity close timestamps — no live integration, a CSV is enough; **(2)** replay it through the time split: fit the customer's existing scoring signals on their oldest quarter, judge on the next, and run the timestamp audit on every field the current score consumes; **(3)** deliver the three artefacts this kit produces — the per-signal fit-vs-held-out lift table, the leakage list with offending counts, and the frozen-weight score with its held-out AUC against their best single signal. The pilot succeeds if it finds one signal to kill or re-weight with evidence; in most real stacks the timestamp audit alone finds a post-close field sitting inside the score, which is the cheapest possible win to hand a revenue team in week one.

## Why this project

Signal scoring is where GTM engineering stops being plumbing and becomes judgement: somebody owns the logic that decides which accounts a sales team touches first. This kit shows that ownership done carefully — signals defined as auditable thresholds, weights frozen from one window and judged on another, and the two classic self-deceptions (decay and leakage) turned into checks with exit codes instead of post-mortems.
