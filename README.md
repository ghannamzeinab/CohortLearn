# CohortLearn

Build matched exposed and unexposed cohorts from electronic health records, under
the target trial emulation framework.

Two problems get in the way of causal analysis on EHR data. Cohort construction is prone to design bias: follow-up that starts before eligibility is confirmed,
eligibility decided using records that postdate time zero, or controls drawn from a
different calendar period. Separately, many studies need to adjust for several confounders at once, and exact matching on all of them can leave strata empty.

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
matches, reports the diagnostics, fits the outcome model, and writes figures, the
matched cohort and a results summary to `outputs/`.

To sweep several washout windows:

```bash
python scripts/run_pipeline_example.py --data-dir data --sweep 12 24 36 60 120
```

## Reproduce with Docker

The Docker image fixes the operating system, the Python version and every package
version. It also forces one generic CPU maths code path and a single thread, so the
worked example gives the same numbers on different machines. You need
[Docker](https://docs.docker.com/get-docker/) installed and running.

### Run the published image

Windows (PowerShell):

```powershell
docker run --rm -v "${PWD}\outputs:/app/outputs" ghcr.io/ghannamzeinab/cohortlearn:v0.3.0
```

Linux:

```bash
docker run --rm -v "$(pwd)/outputs:/app/outputs" ghcr.io/ghannamzeinab/cohortlearn:v0.3.0
```

macOS:

```bash
docker run --rm --platform linux/amd64 -v "$(pwd)/outputs:/app/outputs" ghcr.io/ghannamzeinab/cohortlearn:v0.3.0
```


### Or build the image from this repository

```bash
docker build -t cohortlearn:v0.3.0 .
docker run --rm -v "$(pwd)/outputs:/app/outputs" cohortlearn:v0.3.0
```

On Windows, use `"${PWD}\outputs:/app/outputs"` for the mount.

### What it runs

The container generates the synthetic dataset (seed 42), builds the depression and
no-depression cohorts, matches them, fits the Cox model and computes the E-values.
The results are written to `outputs/`:

| File                     | Content                                        |
| ------------------------ | ---------------------------------------------- |
| `results_summary.txt`    | cohort sizes, balance, hazard ratio, E-values  |
| `love_plot.pdf`          | standardised mean differences                  |
| `ps_overlap.pdf`         | propensity-score overlap                       |
| `matched_cohort_12m.csv` | the matched cohort                             |

You should get a hazard ratio of 1.52 (95% CI 1.25–1.84), with 432 events.

The console also prints two validation checks on separate synthetic datasets with a
known hazard ratio. With a true HR of 1.65, the pipeline returns 1.68 (1.45–1.96).
With a true HR of 1.00, it returns 1.12 (0.94–1.32). Both intervals contain the
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
run_all.py           worked example and validation checks, run by Docker
Dockerfile           reproducible environment
requirements.lock    pinned package versions
data/                generated, not tracked
outputs/             generated, not tracked
```

## What the library does

**Cohort construction.** Time zero for an exposed participant is their first exposure
code. Time zero for an unexposed participant is drawn from the observed exposed time
zeros, accepted only if that person is alive, under observation and unexposed on
that date. Controls with the outcome on or before that date are then removed as
prevalent cases. Participants who are unexposed early and exposed later contribute
their unexposed time and are censored at exposure onset. A configurable washout
window removes outcomes that were already developing at time zero.

**Matching.** Propensity scores come from regularised logistic regression, fitted to
tight convergence. Matching is greedy nearest neighbour on the logit score, without
replacement, 1:k, inside exact strata for sex and a two-year index-period bucket,
with a caliper of 0.2 standard deviations. Distances are compared on an exact
integer grid with stable tie-breaking, so matches do not depend on floating-point
rounding.

**Validation.** Standardised mean differences before and after matching, a Love plot,
propensity overlap plots, and a stratum composition table.

**Outcome stage.** Cox regression with target-trial censoring, optional standard
errors clustered on participant, a proportional hazards check, and E-values for
unmeasured confounding.

## Synthetic dataset

`scripts/generate_synthetic_data.py` writes a synthetic dataset in the input format
above. Exposure is generated from a logistic function of the covariates, so there is
real confounding to remove. Diagnosis dates are set from a drawn age at diagnosis
rather than from a calendar window, so timing depends on the person. The generator
is seeded.

Two outcome models are available through `--outcome-model`:

- `logistic` (default) sets the exposure effect on the odds of ever having the
  outcome. No true hazard ratio is defined, so the Cox estimate can be checked for
  direction only. The worked example uses this model.
- `hazard` draws outcome times from a proportional hazards model with time-varying
  exposure, so the true hazard ratio is known and the Cox estimate can be checked
  against it. The validation checks use this model.

The size of the effect is set with `--true-effect`, on the log scale. Each run also
writes `generation_info.json`, which records the true effect and its scale.

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
