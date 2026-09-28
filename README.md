# Invariant Sentinel

**Data drift and quality monitoring that can tell a harmless change from a broken input.**

A machine-learning model is only as good as the data fed to it. When that data changes, a
standard drift monitor raises an alarm, but it cannot say whether:

- the **world changed** (more expensive stones sold in the holiday season: harmless), or
- an **input broke** (the length gauge reads 3% high: the model is now fed wrong numbers).

Invariant Sentinel learns how healthy features relate to each other ("a diamond's weight grows like
its length cubed"). When values move but the relationships still hold, the change is harmless.
When a relationship stops holding, something is wrong, and the pattern of broken relationships
points to the faulty feature, like a fingerprint.

For every batch of data it answers five questions: **what happened, where, why this feature,
how strong is the evidence, and what to check.**

## Run it

Requires Python 3.10 or newer.

```bash
pip install -r requirements.txt
python run.py
```

The browser opens at <http://127.0.0.1:8000>. Everything runs locally; no internet connection or
GPU is needed.

The two example datasets come with their learned profiles. If your scikit-learn version differs
from the one used to build them, the app relearns them automatically the first time
(about 15 seconds).

Run the tests with `pytest -q`.

## The five-minute demo

1. **Monitor** tab, scenario *Holiday season, then the length gauge goes wrong (main demo)*.
   Press **Start**, then **Play**.
2. Days 1–5 are grey (normal). From day 6, customers buy bigger stones: the standard drift check
   (the thin lane) alarms every day, while Sentinel shows **blue: harmless shift**. Open a blue day:
   in *Does it still agree with its neighbours?* the points slid **along** the dashed line.
3. From day 14 the length gauge reads 3% high. Sentinel turns **red** and names Length. The
   points are pushed **off** the line; *Why this feature* shows the observed pattern matching
   Length's fingerprint; *Does the model care?* shows the effect on predicted prices.
4. Press **Reveal what was injected**: the ground truth lines up with the verdicts.
5. Optional: open *Change something while it runs* and inject your own fault, or switch the
   width to centimetres to see a purple **data quality** verdict.

## Pages

| Page | What it shows |
|---|---|
| Monitor | Run a scenario batch by batch. Timeline of verdicts next to a standard drift check; full investigation of any batch |
| Relationships | The rules learned from healthy data, which faults can and cannot be told apart, and every threshold with its source |
| Evaluation | Sentinel vs. the standard check on 30 simulated batches per situation, with 95% confidence intervals |
| Your data | Switch datasets, or upload any CSV and choose which columns to monitor |

Charts: drag to zoom; double-click, the toolbar's home icon or the **Reset zoom** button returns to
the original view.

## Datasets

| Dataset | Rows | Predicts | Monitors |
|---|---|---|---|
| Diamonds (ggplot2 / seaborn example data) | 53,940 | price | weight, length, width, depth, depth %, table % |
| Abalone (UCI Machine Learning Repository) | 4,177 | age (shell rings) | length, diameter, height, four weights |

**Your own CSV** works if it has a numeric column to predict and at least three numeric columns
that are related to each other (at least 500 rows). With unrelated columns the relationships are
weak and the system falls back to behaving like a standard drift check, and says so.

## Project layout

```
sentinel/            the method (no web code)
  config.py          dataset settings: which columns, units, log scale, batch size
  data.py            loading, splitting (learn 50% / calibrate 20% / replay 30%), cleaning
  relationships.py   learning the rules, fingerprints, which faults can be told apart
  checks.py          quality checks, drift check (KS), rule statistics
  locate.py          matching fingerprints to find the faulty feature
  impact.py          repairing a suspect feature and re-running the model
  model.py           the model being protected (gradient-boosted trees)
  profile.py         builds everything from healthy data and calibrates thresholds
  diagnose.py        one batch: checks -> verdict -> plain-language explanation
  scenarios.py       simulated batches, harmless shifts and injected faults
  evaluate.py        repeatable comparison with the standard drift check
server/app.py        FastAPI: JSON API + serves the web page
web/                 plain HTML, CSS and JavaScript with Plotly (all files local)
datasets/            example CSVs and their settings
artifacts/           learned profiles and cached evaluation results
tests/               pytest
docs/HOW_IT_WORKS.md plain-language explanation of the method and likely viva questions
```

API documentation is served at <http://127.0.0.1:8000/api/docs> while the app runs.
