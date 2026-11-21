# Model card — EV battery failure classifier

*Format follows [Mitchell et al., "Model Cards for Model Reporting" (2019)](https://arxiv.org/abs/1810.03993).
Values reflect the model committed on `main`; CI regenerates them on every run
([`models/`](../models), [`reports/`](../reports)).*

## Model details

| | |
|---|---|
| **Task** | Binary classification: will this battery pack fail? |
| **Model** | L2-regularised logistic regression (`C=0.5`, `class_weight="balanced"`), selected automatically from 5 candidates |
| **Inputs** | 67 raw features (59 numeric, 8 categorical) → 157 after one-hot encoding + standard scaling |
| **Output** | Failure score in [0, 1], a binary *inspect / don't inspect* decision at the cost-optimal threshold (0.555), and a low/medium/high risk band |
| **Training** | 14,000 vehicles (70 % stratified split); selection by 5-fold CV PR-AUC; threshold tuned on a separate 3,000-vehicle validation fold |
| **Versioning** | Each training run is an MLflow run + Model Registry version; the version passing the gates holds the `@champion` alias. The serving image tag = git SHA |
| **License** | MIT |

## Intended use

* **Primary use:** rank and flag vehicles in a fleet for **preventive battery inspection**, as decision
  support for maintenance planners.
* **Not intended for:** automated safety decisions (for example, remotely disabling a vehicle), warranty
  adjudication, insurance pricing, or any decision about an individual driver without human review.
* **Users:** fleet maintenance teams and reliability engineers.

## Metrics

Test fold: 3,000 vehicles, 208 failures (6.9 %), untouched until final evaluation.

| Metric | Value | Why it matters |
|---|---:|---|
| ROC-AUC | 0.985 | Ranking quality across all thresholds |
| PR-AUC | 0.785 | Ranking quality on the rare positive class (no-skill baseline = 0.069) |
| Recall @ 0.555 | 0.952 | 198 of 208 failures caught |
| Precision @ 0.555 | 0.571 | 57 % of flagged vehicles really fail |
| Flag rate | 11.6 % | Inspection workload created |
| Expected cost / vehicle | 0.083 | vs. 0.693 if nothing is inspected (−88 %) |
| Brier score | 0.043 | Probability calibration (see caveats) |

**Decision threshold.** Chosen on the validation fold to minimise `10 × FN + 1 × FP`. The 10:1 ratio is
an assumption standing in for "roadside failure + pack replacement" vs. "one diagnostic check". It lives
in `params.yaml` and should be set with the business.

**Quality gates.** The pipeline fails if test ROC-AUC < 0.90 or PR-AUC < 0.55. A new model is also
rejected if its PR-AUC is more than 0.02 below the model on `main`.

## Evaluation data and slices

The held-out test fold comes from the same distribution as the training data (stratified random split).
Performance on a **future** period, or on a new market, is not measured by this split. The drift monitor
exists to detect exactly that situation.

## Explainability

Global SHAP importance (see [`reports/figures/shap_importance.png`](../reports/figures/shap_importance.png))
is led by previous faults, BMS warning count, thermal-runaway risk, abnormal voltage events, capacity
loss and state of health. All of these are physically plausible degradation signals. Per-prediction
explanations are served by `POST /explain` (exact linear SHAP in log-odds, with one-hot columns
aggregated back to raw fields).

## Leakage checks

* No single feature exceeds AUC 0.85 on its own (the flag threshold is 0.95).
* Removing all eight derived scores of unknown provenance changes ROC-AUC from 0.986 to 0.982
  (PR-AUC 0.837 → 0.799, LightGBM probe). The model does not hinge on them.
  Details: [`reports/leakage_audit.json`](../reports/leakage_audit.json).

## Caveats and limitations

* **Synthetic data.** The dataset shows signs of being simulated (see the [dataset card](DATASET_CARD.md)).
  The metrics describe this dataset, not real-world fleets.
* **Uncalibrated scores.** Class weighting inflates predicted probabilities. Use the decision and
  risk band, not the raw probability, until a calibration step (isotonic / Platt) is added.
* **Snapshot features, no time axis.** The model sees one aggregated snapshot per vehicle. It cannot
  say *when* a failure will happen.
* **Missing inputs.** Callers may omit fields; they are median/mode-imputed. Predictions made from very
  sparse inputs drift towards the population average. The `InputMissingnessSpike` alert watches for this.

## Ethical considerations

* **False negatives carry safety risk.** The threshold deliberately trades precision for recall (95 %).
  Some failures will still be missed, so the system supplements scheduled maintenance and BMS safety
  interlocks; it does not replace them.
* **Fairness across fleet segments.** Brand, model and fleet/private status are inputs. If inspections
  carry a cost to owners, operators should review flag rates per segment so one group is not
  systematically over-inspected because of correlated, non-causal features.
* **Privacy.** Inputs are vehicle telemetry, but driving-behaviour features (speed, braking, trip
  distance) can be personal data in some jurisdictions when linked to a driver.
