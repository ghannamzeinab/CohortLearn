# CohortLearn

Build matched exposed and unexposed cohorts from electronic health records, under
the target trial emulation framework.

Two problems get in the way of causal analysis on EHR data. Cohorts assembled by
hand pick up design bias: follow-up that starts before eligibility is confirmed,
eligibility decided using records that postdate time zero, or controls drawn from a
different calendar period. Separately, realistic studies need many confounders at
once, and exact matching on all of them leaves the strata empty.

CohortLearn addresses both. Time zero is aligned across arms, and controls are drawn
by risk-set sampling using only pre-baseline records. The confounder set is then
collapsed to a single propensity score and matched.

Nothing about the clinical question is fixed in the code. You supply the exposure,
the outcome and the confounders as ICD code prefixes.

## Install

From the repository root:

```bash
pip install -e .
```

## Quick start

```bash
python scripts/generate_synthetic_data.py --n 60000 --out-dir data
python scripts/run_pipeline_example.py --data-dir data --out-dir outputs
```

The first command writes three CSV files to `data/`. The second builds the cohorts,
matches, reports the diagnostics, fits the outcome model, and writes figures and the
matched cohort to `outputs/`.

To sweep several washout windows:

```bash
python scripts/run_pipeline_example.py --data-dir data --sweep 12 24 36 60 120
```

## Reproduce with Docker

The Docker image fixes the operating system, the Python version and every package
version, so the worked example gives the same numbers on any machine. You need
[Docker](https://docs.docker.com/get-docker/) installed and running.

### Run the published image

Windows (PowerShell):

```powershell
docker run --rm -v "${PWD}\outputs:/app/outputs" ghcr.io/ghannamzeinab/cohortlearn:v0.2.0
```

macOS or Linux:

```bash
docker run --rm -v "$(pwd)/outputs:/app/outputs" ghcr.io/ghannamzeinab/cohortlearn:v0.2.0
```

To pin the exact image, use its digest instead of the tag:

```
ghcr.io/ghannamzeinab/cohortlearn@sha256:18744072d20d7201bd5e0589177f6a77e68edf3ceb19dea2e873fee5aebd819d
```

### Or build the image from this repository

```bash
docker build -t cohortlearn:v0.2.0 .
docker run --rm -v "$(pwd)/outputs:/app/outputs" cohortlearn:v0.2.0
```

On Windows, use `"${PWD}\outputs:/app/outputs"` for the mount.

### What it runs

The container generates the synthetic dataset (seed 42), builds the depression and
no-depression cohorts, matches them, fits the Cox model and computes the E-values.
The results are written to `outputs/`:

| File                    | Content                                   |
| ----------------------- | ----------------------------------------- |
| `results_summary.txt`   | cohort sizes, balance, hazard ratio, E-values |
| `love_plot.pdf`         | standardised mean differences             |
| `ps_overlap.pdf`        | propensity-score overlap                  |
| `matched_cohort_12m.csv`| the matched cohort                        |

### Expected results

| Metric                        | Value              |
| ----------------------------- | ------------------ |
| Cohort after matching         | 9,712 depression / 28,292 no depression |
| Max \|SMD\| after matching    | 0.027              |
| Events                        | 430                |
| Hazard ratio (95% CI)         | 1.49 (1.23–1.81)   |
| E-value (estimate / CI limit) | 2.35 / 1.76        |

The console also prints two validation checks on separate synthetic datasets with a
known hazard ratio. With a true HR of 1.65, the pipeline returns 1.68 (1.45–1.96).
With a true HR of 1.00, it returns 1.10 (0.93–1.30). Both intervals contain the
true value.

## Use as a library

```python
from cohortlearn import CohortBuilder, PSMCalculator, SurvivalAnalyser
from cohortlearn.survival import full_rank_covariates

CONF = {
    "exposure_codes": ["F32", "F33"],
    "outcome_codes":  ["F00", "G30"],
    "confounder_codes": {
        "hypertension": ["I10", "I11"],
        "anxiety":      ["F40", "F41"],
    },
    "demographic_confounders": {"age": True, "sex": True, "bmi": True,
                                "education": True, "risk_allele": False},
    "age_range": (18, 110),
}

builder = CohortBuilder(washout_months=12, require_observation=True, **CONF)
builder.attach_dataframes(icd10_long=icd10_long_df,
                          demographics=demographics_df,
                          bmi=bmi_df)

cases    = builder.build_case_cohort()
controls = builder.build_control_cohort(cases)
master   = builder.build_master_df(cases, controls)

psm = PSMCalculator(master, exact_match_cols=["Sex"], ratio=3,
                    consort=builder.consort)
psm.fit()
psm.match()

psm.balance_table()
psm.plot_smd()
psm.plot_overlap()
psm.strata_table()

probe = SurvivalAnalyser(psm.matched_cohort, study_end="2024-12-31")
probe.prepare()
keep, dropped = full_rank_covariates(probe)

sa = SurvivalAnalyser(psm.matched_cohort, study_end="2024-12-31", covariates=keep)
sa.prepare()
sa.fit(adjusted=True, cluster=True)
sa.summary(exposure_only=True)
sa.e_value()
```

## Input format

Flat tables. No common data model required.

| File                | Rows                    | Columns                                          |
| ------------------- | ----------------------- | ------------------------------------------------ |
| diagnoses           | one per coded diagnosis | `id`, `code`, `date`                             |
| demographics        | one per participant     | `id`, `Sex`, `YOB`, `Education`, `Date of Death` |
| bmi (optional)      | one per participant     | `id`, `BMI`                                      |
| genotype (optional) | one per participant     | `id`, plus one genotype column                   |

The diagnosis table also defines each participant's observation span, taken as their
first and last coded dates. Identifier fields named `eid` or `person_id` are renamed
to `id` automatically.

## Repository layout

```
src/cohortlearn/     the library
  cohort.py          CohortBuilder, CONSORTTracker, rename_id
  matching.py        PSMCalculator
  survival.py        SurvivalAnalyser, full_rank_covariates
scripts/
  generate_synthetic_data.py
  run_pipeline_example.py
tests/               smoke tests
data/                generated, not tracked
outputs/             generated, not tracked
```

## What the library does

**Cohort construction.** Time zero for an exposed participant is their first exposure
code. Time zero for an unexposed participant is drawn from the observed exposed time
zeros, accepted only if that person is alive, under observation, outcome-free and
unexposed on that date. Participants who are unexposed early and exposed later
contribute their unexposed time and are censored at exposure onset. Prevalent
outcomes are excluded, and a configurable washout window removes outcomes that were
already developing at time zero.

**Matching.** Propensity scores come from regularised logistic regression. Matching
is greedy nearest neighbour on the logit score, without replacement, 1:k, inside
exact strata for sex and a two-year index-period bucket, with a caliper of 0.2
standard deviations.

**Validation.** Standardised mean differences before and after matching, a Love plot,
propensity overlap plots, and a stratum composition table.

**Outcome stage.** Cox regression with target-trial censoring, standard errors
clustered on participant, a proportional hazards check, and E-values for unmeasured
confounding.

## Synthetic dataset

`scripts/generate_synthetic_data.py` writes a synthetic dataset in the input format
above. Exposure is generated from a logistic function of the covariates, so there is
real confounding to remove. The outcome depends on exposure with a known
coefficient, so the recovered estimate can be checked against the truth. Diagnosis
dates are set from a drawn age at diagnosis rather than from a calendar window, so
timing depends on the person. The generator is seeded.

## Tests

```bash
pip install -e ".[dev]"
pytest -q
```

## Scope and limitations

- Time-varying confounding is not supported.
- Diagnoses are assumed ICD-coded. Other vocabularies need mapping first.
- One matching algorithm PSM.

## License

MIT. See `LICENSE`.
