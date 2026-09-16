# Design and Performance Evaluation of a Differentially Private Federated Learning Framework for Privacy-Preserving Smart City IoT Analytics

Dissertation practical work. The project simulates a smart-city traffic analytics
network from twelve independent road-sensor streams and compares three ways of
training a traffic-condition classifier over them:

1. **Centralised machine learning** — all client data pooled on one server.
2. **Standard federated learning** — weighted FedAvg, only model updates leave the client.
3. **Differentially private federated learning** — FedAvg with per-client update
   clipping and Gaussian noise, evaluated across five privacy budgets.

The comparison covers predictive quality (accuracy, macro precision, macro recall,
macro F1), training time, communication overhead, and per-client performance under
heterogeneous data.

---

## Contents

```
Vamshi_Dissertation.ipynb     Full pipeline: data prep -> baseline -> FL -> DP-FL -> evaluation
Results/                      Exported CSV outputs from a completed run
  raw_data_summary.csv
  client_quality_and_target_summary.csv
  model_comparison.csv
  dp_privacy_utility_results.csv
  client_level_evaluation.csv
  communication_overhead.csv
  standard_fl_history.csv
  standard_fl_client_updates.csv
  best_dp_fl_history.csv
  best_dp_client_updates.csv
```

---

## Data

Twelve CSV files from the CityPulse Aarhus road-traffic collection, each one
Bluetooth travel-time observations from a single road segment and treated here as
one IoT client:

```
trafficData158324  trafficData158386  trafficData158536  trafficData158655
trafficData178548  trafficData178821  trafficData178929  trafficData179038
trafficData179148  trafficData179202  trafficData180601  trafficData180872
```

Each file has 9 columns and spans **2014-02-13 11:30 to 2014-06-09 05:35** at
five-minute resolution, with 27,501–32,075 rows per client. There are no missing
values; a small number of exact duplicate rows (0–16 per file) are dropped during
preprocessing. The fields used are `TIMESTAMP`, `avgSpeed`, `avgMeasuredTime`,
`medianMeasuredTime` and `vehicleCount`.

The files are **not included in this repository** — download them from the CityPulse
dataset collection and place them where the notebook expects them (see *Running it*).

### Per-client reference speeds

Each road segment has its own free-flow reference speed (`NDT`), hard-coded in the
notebook as `CLIENT_NDT` and ranging from 27 to 110. Using a local reference rather
than a global threshold is what makes the label definition consistent across
segments of very different character, and it is also what produces the class
imbalance that varies from client to client.

---

## Method

### Label construction

The target is a three-class traffic condition derived locally on each client:

```
speed_ratio = avgSpeed / NDT

speed_ratio < 0.70  ->  0  Congested
speed_ratio < 0.90  ->  1  Moderate
otherwise           ->  2  Free flow
```

### Features

Ten predictors, all computed from the past so the task stays a genuine forecasting
problem:

| Group | Features |
|---|---|
| Lag-1 | `avgSpeed_lag1`, `avgMeasuredTime_lag1`, `medianMeasuredTime_lag1`, `vehicleCount_lag1` |
| Lag-2 | `avgSpeed_lag2`, `vehicleCount_lag2` |
| Time of day | `hour_sin`, `hour_cos` |
| Day of week | `weekday_sin`, `weekday_cos` |

Lags are only valid across a real time gap — lag-1 requires the previous reading
within 600 s and lag-2 within 900 s — so rows sitting after a sensor outage are
dropped rather than lagged against a stale value. This leaves 26,685–31,782 usable
rows per client.

### Splitting and scaling

Each client is split **chronologically** 70 / 15 / 15 into train, validation and test.
A `StandardScaler` is fitted on the client's own training partition only and applied
to its validation and test partitions, so no scaling statistics cross client
boundaries and no future information leaks into training. Balanced class weights are
likewise computed per client from its local training data.

### Model

A multinomial logistic regression (softmax) with **33 trainable parameters**
(10 features × 3 classes + 3 biases). It is implemented twice: via scikit-learn's
`LogisticRegression` for the centralised baseline, and as an explicit NumPy
forward/backward pass for the federated runs. The manual implementation is what makes
DP possible — clipping and noising require direct access to the parameter update
vector.

### Federated protocol

`run_federated()` implements the loop for both standard and DP training:

- **5 communication rounds**, **1 local epoch** per round, batch size 256, learning rate 0.05.
- Each client trains locally from the current global model, then reports the
  *difference* `local_params - global_params`.
- Updates are clipped to an L2 norm of **C = 1.0**.
- In the DP variant, Gaussian noise is added to each clipped update before it leaves
  the client (client-level / local DP, not a trusted-aggregator model).
- The server aggregates by **weighted FedAvg**, weighting each client by its training
  set size.

### Privacy accounting

A total budget ε is split evenly across the five rounds (ε_round = ε / 5) and the
per-round noise scale comes from the classical Gaussian mechanism:

```
sigma = C * sqrt(2 * ln(1.25 / delta)) / eps_round      with delta = 1e-5, C = 1.0
```

Budgets tested: **ε ∈ {0.5, 1, 2, 4, 8}**. The final DP model is selected by macro F1
on the pooled test set.

---

## Results from the included run

### Model comparison (pooled test set)

| Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | Train time (s) |
|---|---|---|---|---|---|
| Centralised ML | 0.6079 | 0.5734 | 0.6120 | 0.5818 | 1.70 |
| Standard FL | 0.5832 | 0.5603 | 0.5723 | 0.5641 | 1.47 |
| DP-FL ε=0.5 | 0.3250 | 0.3012 | 0.3033 | 0.2831 | 1.44 |
| DP-FL ε=1 | 0.3291 | 0.3095 | 0.3087 | 0.2896 | 1.48 |
| DP-FL ε=2 | 0.3409 | 0.3296 | 0.3230 | 0.3070 | 1.49 |
| DP-FL ε=4 | 0.3735 | 0.3749 | 0.3620 | 0.3517 | 1.91 |
| **DP-FL ε=8 (selected)** | **0.4422** | **0.4467** | **0.4420** | **0.4326** | 2.46 |

Federation costs about **2.5 accuracy points and 1.8 macro-F1 points** against the
centralised baseline — a small price for keeping raw observations on the client.
Differential privacy is far more expensive at these settings: even the loosest budget
tested gives up roughly 14 accuracy points against standard FL, and the tightest
collapses to near chance level on a three-class problem.

### Privacy–utility trade-off

| ε | Noise σ per round | Macro F1 |
|---|---|---|
| 0.5 | 48.45 | 0.2831 |
| 1 | 24.22 | 0.2896 |
| 2 | 12.11 | 0.3070 |
| 4 | 6.06 | 0.3517 |
| 8 | 3.03 | 0.4326 |

Utility rises monotonically with ε, as expected. The scale of the problem is visible
in the noise column: raw local update norms are roughly 0.5–3.4, so even at ε=8 the
per-coordinate noise standard deviation (3.03) is of the same order as the entire
clipped update, and at ε=0.5 it swamps the signal completely.

### Convergence

Standard FL converges quickly and cleanly — macro F1 goes 0.5401 → 0.5529 → 0.5605 →
0.5612 → 0.5641 over the five rounds, with validation F1 tracking test F1 closely.

The DP run at ε=8 is non-monotonic (0.332 → 0.296 → 0.308 → 0.328 → 0.433), which is
the signature of noise dominating early rounds before the accumulated signal wins out.

### Clipping behaviour

In round 1 only 2 of 12 clients exceed the clipping threshold. By round 5, under DP,
every client's raw update norm sits between 2.6 and 3.4, so clip scales fall to
roughly 0.30–0.39 — clipping is binding on every client. Under standard FL the update
norms shrink across rounds instead (most fall below 0.8 by round 5), with only client
178821 consistently clipped. The divergence is itself a finding: injected noise keeps
pushing local models away from the global model, so updates never settle.

### Communication overhead

| Approach | Estimated upload |
|---|---|
| Centralised (raw training records) | 20.900 MB |
| Standard FL (model updates) | 0.0151 MB |
| DP-FL (model updates) | 0.0151 MB |

Roughly a **1,380× reduction**, because 12 clients × 5 rounds × 33 parameters × 8
bytes is trivially small next to shipping every training row. Note this favours FL
partly because the model is tiny — the ratio would narrow considerably for a deep
network.

### Per-client performance

Macro F1 for the selected DP model varies from **0.2535** (client 178821) to **0.5222**
(client 178548) — a spread of nearly 0.27 across clients from one global model. The
weakest clients are the most skewed ones: 178821 is 86% free-flow and 158655 is 77%
free-flow, and both fall below 0.32. The strongest, 178548 and 178929, are the most
evenly balanced across the three classes. Statistical heterogeneity, not sensor
quality, is what drives the gap.

---

## Running it

The notebook was written for **Google Colab** and reads data from `/content`. On the
first data cell it checks which of the twelve CSVs are present and prompts an upload
for any that are missing via `google.colab.files.upload()`.

To run locally instead, change two lines:

```python
DATA_DIR = Path("./data")                 # cell 2  — where the CSVs live
OUTPUT_DIR = Path("./fl_results")         # cell 37 — where CSVs are written
```

and delete the `google.colab` import block. Then run all cells top to bottom; the
whole pipeline takes under a minute on CPU.

**Requirements:** Python 3.9+, `numpy`, `pandas`, `matplotlib`, `scikit-learn`. No
GPU and no federated learning framework — the FL protocol is implemented directly in
NumPy so every step is inspectable.

**Reproducibility:** `SEED = 42` throughout, with per-client and per-round seeds
derived deterministically (`seed + round*100 + client_index` for local training,
`seed + round*1000 + client_index` for noise), so repeated runs give identical
numbers. Timing columns will vary with hardware.

---

## Output files

| File | Contents |
|---|---|
| `raw_data_summary.csv` | Row/column counts, duplicates, missing values and time span per client |
| `client_quality_and_target_summary.csv` | Usable rows after preprocessing and class counts per client |
| `model_comparison.csv` | The headline seven-row comparison table |
| `dp_privacy_utility_results.csv` | Metrics, noise scale and training time for each ε, with the selected row flagged |
| `client_level_evaluation.csv` | Selected DP model evaluated on each client's own test partition |
| `communication_overhead.csv` | Estimated upload volume by approach |
| `standard_fl_history.csv` | Per-round metrics for standard FL |
| `best_dp_fl_history.csv` | Per-round metrics and cumulative ε for the selected DP run |
| `standard_fl_client_updates.csv` | Per-round, per-client update norms, clip scales and local training times |
| `best_dp_client_updates.csv` | Same, for the selected DP run, including noise σ |

The notebook writes these to `/content/fl_results`; the bundled copies are in `Results/`.

---

## Limitations and known quirks

Worth being explicit about these, since several affect how the numbers should be read.

- **Privacy accounting is naive.** The budget is divided evenly across rounds under
  basic sequential composition, and per-round σ comes from the classical Gaussian
  mechanism bound. A moments accountant or RDP accountant would give a tighter ε for
  the same noise, so the DP results here are pessimistic relative to what a
  production DP-FL system would achieve. It also means the reported ε is not directly
  comparable to figures from Opacus or TensorFlow Privacy.
- **Local DP, not central DP.** Noise is added per client rather than once by a
  trusted aggregator, which is the stronger threat model but also the reason the
  utility cost is so steep. Secure aggregation would allow far less noise for the
  same guarantee.
- **The privacy unit is the whole client update**, not an individual record. The
  guarantee is therefore client-level for one round of participation, and the notebook
  does not apply subsampling amplification.
- **Model selection uses the test set.** `best_epsilon` is chosen by macro F1 on
  `X_central_test`, and per-round DP metrics are also computed on the test set.
  Validation F1 is recorded but not used for selection. The comparison is still
  internally consistent, but the reported DP figure is mildly optimistic.
- **`target_rate` is mislabelled.** In `client_quality_and_target_summary.csv` this
  column is the *mean of the three-class label*, not a rate, so values exceed 1. Read
  it as an ordinal skew indicator — higher means more free-flow — or use the raw
  `class_0` / `class_1` / `class_2` counts instead.
- **Gradient scaling in `local_train`** divides the weighted error by `weights.sum()`
  rather than by the batch size, so the effective learning rate depends on the class
  composition of each batch. It works consistently across all runs, but it is not
  standard weighted cross-entropy.
- **Five rounds is short.** Standard FL had essentially converged, but the DP run was
  still improving at round 5, so the DP numbers are a floor rather than a ceiling.
- **`multi_class="multinomial"`** is deprecated in scikit-learn 1.5+ and removed in
  1.7. On a current install the argument can simply be dropped — multinomial is the
  default for multi-class problems with `lbfgs`.
- **The communication estimate is analytical**, computed from parameter counts rather
  than measured on a wire. It excludes protocol overhead, encryption and any secure
  aggregation cost.

---

## Suggested extensions

- Swap the fixed budget split for an RDP accountant and re-run the ε sweep.
- Adaptive clipping, or a clipping norm tuned per round, given that C = 1.0 becomes
  binding on every client by round 5.
- More communication rounds with a smaller per-round budget, to test whether DP-FL
  closes the gap given time.
- Personalisation or fine-tuning on top of the global model, targeted at the skewed
  clients that currently score worst.
- A non-linear model (MLP or gradient boosting) to establish how much of the headroom
  above 0.61 accuracy is model capacity rather than federation cost.
