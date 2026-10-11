# Judge review: Invariant Sentinel

A mock viva by three judges: an ML faculty member (is the method correct and honestly evaluated?),
an industry engineer (would it work for real?) and an external examiner (do the students understand
what they built?). Everything below was checked against the code in this branch on 2026-10-12.

## 1. Test results

`pytest -q`: **21 passed** (about 35 s normally; 2 min on a busy machine). 16 tests cover the method
on synthetic data with known relationships; 5 cover the affected-rows report and incident merging.

### Evaluation (artifacts/*/evaluation.json, 30 simulated batches per row)

**Diamonds** (6 features, 200 stones per day)

| Situation | Sentinel | Standard KS check |
|---|---|---|
| Ordinary days: false alarms | 0/30 | 0/30 |
| Harmless shift, strength 0.5 / 1 / 1.5: false alarms | 0 / 0 / 6 of 30 | 30 / 30 / 30 of 30 |
| Small faults caught (6 features) | 177/180 (98%) | 78/180 (43%) |
| Right feature named | 117/180 (65%) | 77/180 (43%) |
| Wrong feature blamed | **0** | (always names the most drifted feature) |
| Unit change called a quality issue | 30/30 | 0/30 (alarms, but calls it drift) |

**Abalone** (7 features, 100 shellfish per catch)

| Situation | Sentinel | Standard KS check |
|---|---|---|
| Ordinary catches: false alarms | 0/30 | 2/30 |
| Harmless shift 0.5 / 1 / 1.5: false alarms | 0 / 1 / 5 of 30 | 30 / 30 / 30 of 30 |
| Small faults caught | 210/210 (100%) | 165/210 (79%) |
| Right feature named | 76/210 (36%) | **165/210 (79%)** |
| Wrong feature blamed | **0** | |

Headline: Sentinel almost never raises a false alarm on harmless shifts (the baseline always does),
catches more small faults, and never blames the wrong feature. **But** on abalone, and on Depth and
Depth % in diamonds, it answers "unclear" where the KS check happens to name the right feature. That
is the most dangerous number in the project (question 1 below).

### Status of the known weak spots (checked in the code)

| Weak spot | Status in this branch |
|---|---|
| No live data ingestion | **True.** Only replay (`/api/runs`) and CSV upload (`/api/uploads`). The engine itself, `analyse(profile, dataframe)`, does accept any batch. |
| No report of which rows were affected | **Built in this branch, not yet merged:** `sentinel/rows.py`, `GET /api/runs/{id}/batches/{i}/rows` (JSON or CSV), an "Affected rows" panel, and incidents. |
| Relationship map labels overlap | **Fixed in this branch, not yet merged** (`drawMap` in `web/js/charts.js`). |
| First run is slow | **Mostly fixed in this branch, not yet merged.** The cause was a full rebuild (13–15 s per dataset) on a scikit-learn version mismatch, plus a deleted evaluation cache. Now only the model is retrained (0.5–3 s). Plotly is gzipped (4.6 MB down to 1.3 MB). Python imports still take about 3 s. |
| Wine quality: few usable rules, identical fingerprints | **True.** 4 of 11 rules are usable (Residual sugar, Density, Alcohol, Total SO₂). Eight feature pairs have fingerprint similarity ≥ 0.98, e.g. Fixed acidity / Density 1.00, Free / Total SO₂ 1.00. |
| Monitor screen is dense and full of jargon | **True.** "Shift score", "Scatter", "KS", "fingerprint" and "usual errors" appear with only short hints. |
| *Found during this review:* the heatmap shows Depth % / Table % = 1.00, and the docs never explained it | **Fixed (A1).** Checked in code: for a Depth % fault the runner-up is Depth (explains 81–85%, which is about 0.91²), so the docs' reason is correct. Table % is never a candidate: it would need an error of about 10× its normal spread, and `locate.py` drops anything above 5×. HOW_IT_WORKS now explains both. |

## 2. Likely questions, ranked from most to least dangerous

**1. (ML faculty) "Your own table shows the plain KS check names the right feature far more often than Sentinel on abalone: Length 28/30 against your 0/30. So does fingerprinting actually help localise faults?"**
- *Why dangerous:* it is in `evaluation.json` and on the Evaluation page.
- *Honest answer:* On abalone, Length and Diameter leave identical fingerprints (similarity 1.00), and the four weights are close (0.90–0.92). So Sentinel says "unclear between Length and Diameter" instead of guessing, and we count that as not right, which is a strict rule. It never blamed a wrong feature. The KS check names the feature that drifted most, but it stays quiet on most small diamond faults (1/30 for Length) and alarms on 100% of harmless shifts, so in practice its blame cannot be trusted. We also know in advance which features cannot be separated: the Relationships page shows it before monitoring starts.
- *Small code change helps:* **yes.** Report a third number, "the true feature is among the named candidates" (item A3).

**2. (ML faculty) "Your harmless shift is made by re-sampling real rows, so every relationship holds by construction. Isn't the 0% false-alarm rate guaranteed by your simulator?"**
- *Why dangerous:* `scenarios.make_batch` only reweights real rows.
- *Honest answer:* Partly, yes. What it really tests is whether rules learned on typical rows still hold in rarer regions, and at strength 1.5 they start to fail (6/30 false alarms on diamonds, 5/30 on abalone). That shows the result is not automatic. A real shift that changes how features relate (a new product line) would be flagged, and arguably should be.
- *Small code change helps:* no (it belongs in the limitations list).

**3. (ML faculty) "0.7 explained, a 0.2 margin, a scatter floor of 1.5, 0.25 for material impact: where do these come from?"**
- *Why dangerous:* the threshold registry labels them "Heuristic".
- *Honest answer:* They are disclosed fixed choices. Everything that decides *whether an alarm fires* is calibrated on healthy batches: the shift-score limit, the KS limits, the unfamiliar share and the row limit. The heuristics only decide whether we name one feature or say "unclear", and how we word the impact.
- *Small code change helps:* **yes.** A report-only sensitivity table: re-run the evaluation with 0.6/0.8 and 0.1/0.3 and show the results barely move (item B4).

**4. (External examiner) "Your heatmap says Depth % and Table % are 1.00, identical. So why do you say Depth % is confused with Depth, and why is Table % never blamed?"**
- *Why dangerous:* a direct contradiction between `docs/HOW_IT_WORKS.md` and the Relationships page.
- *Honest answer:* The heatmap compares directions only. Table % has no usable rule, so its fingerprint lies on the Depth % rule alone, the same direction as Depth %'s. But explaining a Depth % fault would need Table % to be off by about 10× its normal spread, and candidates needing more than 5× are dropped as implausible. The real runner-up is Depth: cosine 0.91 means it explains about 0.91² ≈ 83% of the pattern, which is within the 20-point margin, so the verdict is "unclear between Depth % and Depth".
- *Small code change helps:* **yes** (doc fix, item A1).

**5. (ML faculty) "30 batches per situation, and abalone has only about 1,250 replay rows, so your 30 batches overlap. Likewise, 800 calibration windows come from about 835 calibration rows. How much is 0% worth?"**
- *Why dangerous:* `evaluate.run_suite` draws k from 0 to 4 × n_batches, so rows are reused; `profile._windows` cuts overlapping windows.
- *Honest answer:* The rates come with 95% intervals (0/30 means 0–11%). The batches are not independent, so the true uncertainty is larger than shown. Overlapping windows are the usual way to estimate a percentile from limited data; with more healthy data we would use separate windows.
- *Small code change helps:* **yes.** State it next to the results (item A3).

**6. (Industry engineer) "How would this get data in production? I see no ingestion, only replays."**
- *Why dangerous:* there is no endpoint that takes a new batch.
- *Honest answer:* Correct, it is not a streaming system. The engine is one function, `analyse(profile, dataframe)`, that takes any batch. Today it is fed by the replay or by a CSV upload. Connecting it to a scheduler or database is plumbing we did not build.
- *Small code change helps:* **yes.** A "check one batch" endpoint plus a 20-line example script (item B1). We would call it on-demand checking, not real-time.

**7. (Industry engineer) "What if the features are not related? Try wine quality."**
- *Why dangerous:* the claim depends on relationships existing.
- *Honest answer:* On white wine only 4 of 11 rules are good enough to use, and several fingerprints are identical (acidity / density / alcohol, free / total SO₂). The system says this before monitoring (Relationships page: usable rules, blind spots, inseparable pairs). For features without a rule it behaves like a normal drift check and says "no relationship can confirm this change is harmless". Its value is limited to datasets with related features.
- *Small code change helps:* **yes,** a little: show these limits on screen (item B2).

**8. (External examiner) "What did you build yourselves and what came from libraries?"**
- *Why dangerous:* a common viva trap.
- *Honest answer:* scikit-learn fits the straight-line rules (`LinearRegression`) and the price model being protected (`HistGradientBoostingRegressor`). pandas and NumPy hold and compute the data. We wrote:
  - the KS statistic (`checks.ks_stat`, about 10 lines);
  - greedy rule selection;
  - the signature matrix and fingerprint matching (`locate.py`);
  - calibration on healthy windows;
  - the quality gate, the impact estimate and the row report;
  - the evaluation, the API and the whole interface.
- *Small code change helps:* no.

**9. (ML faculty) "Why linear rules? Real relationships are curved."**
- *Why dangerous:* it sounds simplistic.
- *Honest answer:* On a log scale, physical laws like weight ∝ length³ become straight lines (R² ≥ 0.99 on diamonds). More importantly, a linear rule gives every feature an exact, signed fingerprint: S = 1/σ for a rule's target and −coefficient/σ for its inputs. That is what makes "which feature?" a matching problem. A tree model has no such closed-form fingerprint. Genuinely curved relationships are approximated, and that is a stated limitation.
- *Small code change helps:* no.

**10. (Industry engineer) "Why not just use Evidently or NannyML?"**
- *Why dangerous:* it invites a "you reinvented a wheel" judgement.
- *Honest answer:* They provide per-feature drift tests, which is exactly our baseline, and NannyML also estimates performance without labels. Neither separates a harmless shift from a broken input using relationships between features, or names the broken feature. Our checks could run alongside them. We did not benchmark them directly; our baseline is the same kind of test, with fairer, calibrated limits.
- *Small code change helps:* no.

**11. (ML faculty) "What does 'impact' really measure? You replace the input with a guess from a rule, which isn't the truth."**
- *Why dangerous:* the word "impact" sounds like "accuracy loss".
- *Honest answer:* It measures how much the model's predictions change if the suspect is replaced with the value its neighbours imply, in units of the model's usual error. It is discounted against what the same repair does on healthy batches (their 95th percentile). It is an estimate that needs no labels. When true prices arrive, the app shows the real comparison (main demo: error $635 with the recorded length against $575 repaired).
- *Small code change helps:* no.

**12. (Industry engineer) "How many false alarms would I get per month?"**
- *Why dangerous:* alarm fatigue is a real problem in production.
- *Honest answer:* By design about 1 in 100 healthy batches: the shift-score limit is the 99.5th percentile of the largest score across all rules, and the KS budget of 1% is split across features. Measured: 0/30 on both datasets. For a daily batch that is roughly one false alarm every three months. In the row report, about 1 healthy row in 100 is flagged, and the report always says so.
- *Small code change helps:* no.

**13. (Industry engineer) "How much work is a new model or dataset?"**
- *Why dangerous:* it tests whether the system generalises.
- *Honest answer:* A small JSON settings file, or uploading a CSV and choosing columns. Learning takes 10–15 s. One honest gap: we train our own stand-in model (`TargetModel`). In a real deployment the monitor would call the team's existing model, and only the impact step needs it. After a genuine permanent change the profile must be rebuilt.
- *Small code change helps:* no.

**14. (External examiner) "How do you know the ground truth? And what can't it detect?"**
- *Why dangerous:* it tests understanding of the evaluation and of the limits.
- *Honest answer:* Faults are injected into real replayed rows with exact formulas (×1.03, +1 mm, noise, unit change) and logged, and the Reveal button shows them. Real faults are messier, and we have no dataset of real incidents. It cannot detect:
  - a fault that shifts every feature together (relationships stay intact);
  - faults on features without a usable rule;
  - faults smaller than a rule's normal error.

  Two faults at once usually give "unclear".
- *Small code change helps:* **yes.** Show the limitations in the app, not only in the docs (item B2).

**15. (External examiner) "Explain 'shift score', 'scatter', 'KS' and 'fingerprint' to me without jargon."**
- *Why dangerous:* the Monitor screen uses these words heavily.
- *Honest answer:*
  - **Shift score:** how far a rule's errors moved together today, scaled so the calibrated limit is the healthy maximum (6.1 on diamonds, 2.9 on abalone).
  - **Scatter:** how much noisier those errors got (1.0 = as usual).
  - **KS:** how different today's values of one feature look from the reference (0 = the same, 1 = no overlap).
  - **Fingerprint:** the pattern of rule breaks that a fault on one feature would cause.
- *Small code change helps:* **yes.** Plain names and hover explanations on screen (item A2).

## 3. Improvement plan

No change below alters detection logic, thresholds or calibration. Items that touch `evaluate.py`
only *add* report rows; existing evaluation numbers must stay byte-identical, and we will show that.

### A. Fixes

| # | Change | Answers | Files | Size | How we check it worked |
|---|---|---|---|---|---|
| A0 | **Merge the work already built on this branch** (affected-rows report, map fix, first-run speed, legend fix). It has been tested and reviewed, but it is not on `main` yet. | Q6, the speed and overlap weak spots | already changed: `sentinel/rows.py`, `profile.py`, `server/app.py`, `web/*`, docs | S | `pytest` passes; demo runs; the PR diff is reviewed |
| A1 | **Explain the Depth / Depth % / Table % results** in HOW_IT_WORKS (sections 5 and 7). Depth and Depth % are unclear because each explains about 0.91² ≈ 83% of the other's fault. Depth % / Table % show 1.00 on the heatmap, but Table % is dropped as implausible (it would need about 10× its normal error). **Done.** | Q4 | `docs/HOW_IT_WORKS.md` | S | Matches the candidates `analyse()` returns for Depth, Depth % and Table % faults (checked on 30 batches each) |
| A2 | **Plain words on the Monitor screen.** Column headers become "How far the errors moved (shift score)", "Extra noise (scatter)" and "Distribution change (KS)", each with a one-line hover explanation, plus a short "What these words mean" note under the investigation title. Wording only. | Q15 | `web/index.html`, `web/js/monitor.js` | S | Every jargon word on the Monitor page has an explanation next to it or on hover; tests unchanged |
| A3 | **Honest evaluation extras (report only).** (a) A new row per feature: "True feature among the named candidates". (b) A note on the Evaluation page that batches overlap on small datasets, so intervals are optimistic. | Q1, Q5 | `sentinel/evaluate.py` (new rows only), `web/js/pages.js`, a new test | M | Old rows are identical before and after (script diff of `evaluation.json` minus the new rows); new test: on the synthetic c/d case, the true feature is always among the candidates |

### B. Small new features that answer an attack

| # | Change | Answers | Files | Size | How we check it worked |
|---|---|---|---|---|---|
| B1 | **"Check one batch" endpoint**: `POST /api/datasets/{name}/check` takes a CSV file or JSON rows, runs the same `analyse()` and `flag_rows()`, and returns the verdict, headline and affected rows. Plus `examples/check_batch.py`, about 20 lines using only the standard library. Documented as on-demand checking, **not** real-time streaming. | Q6, Q13 | `server/app.py`, `examples/check_batch.py`, `README.md`, a test | M | Test: posting a replay batch gives the same verdict as `analyse()` on that batch; posting the centimetres fault gives "quality" with every row flagged |
| B2 | **Limitations box in the app**: a short "What this system cannot do" panel on the Relationships page. It takes the general limits from the docs (all features shifting together, unrelated features, two faults at once, simulated faults) and adds the dataset's own known limits, which the app already computes: blind spots and inseparable pairs. | Q7, Q14, Q2 | `web/index.html`, `web/js/pages.js` | S | Visible for diamonds, abalone and wine; for wine it lists the 7 non-usable rules and the identical pairs |
| B3 | **Run report download**: one CSV per run with day, verdict, headline sentence, suspect, rows flagged and expected, and the standard check's alarm. It uses the existing headline, which is already the plain-English one-liner for each day. | Q6, Q12 (how an operations team would read it) | `server/app.py` (`GET /api/runs/{id}/report.csv`), `web/js/monitor.js` (a link) | S | Downloaded CSV has one line per analysed day, matching the timeline and incidents |
| B4 | **Threshold sensitivity table (report only)**: re-run the evaluation with explained-share 0.6 / 0.7 / 0.8 and margin 0.1 / 0.2 / 0.3; show caught, right and wrong rates. The thresholds in use do **not** change. | Q3 | `sentinel/evaluate.py` (a separate function), a small table on the Evaluation page | M | The table shows the current setting matches today's numbers exactly; runs in about 1 minute and is cached like the evaluation |

Dropped as too big or not explainable for this level: streaming ingestion (Kafka or a scheduler),
non-linear rules, multi-fault localisation, and benchmarking against Evidently.

## 4. Suggested order of work

1. **A0** Merge the existing branch work. Everything else builds on it.
2. **A1** Explain the Depth % / Table % heatmap value. **Done.**
3. **A3** Candidate-list metric and overlap note. This defuses question 1, the most dangerous one.
4. **B2** Limitations box.
5. **A2** Plain words on the Monitor screen.
6. **B1** "Check one batch" endpoint and example script.
7. **B3** Run report CSV.
8. **B4** Sensitivity table (only if time allows).

Run `pytest` after each item. For A3 and B4, also show that the existing evaluation rows are
unchanged.
