"""Run with: pytest -q"""

import numpy as np
import pandas as pd
import pytest

from cohortlearn import CohortBuilder, PSMCalculator, SurvivalAnalyser
from cohortlearn.survival import full_rank_covariates

CONF = {
    "exposure_codes": ["F32"],
    "outcome_codes": ["G30"],
    "confounder_codes": {"hypertension": ["I10"], "anxiety": ["F41"]},
    "demographic_confounders": {"age": True, "sex": True, "bmi": True,
                                "education": False, "risk_allele": False},
    "age_range": (18, 110),
}


@pytest.fixture(scope="module")
def inputs():
    rng = np.random.default_rng(0)
    n = 4000
    ids = np.arange(1_000_000, 1_000_000 + n)
    start = pd.Timestamp("2000-01-01")
    demo = pd.DataFrame({"id": ids, "Sex": rng.integers(0, 2, n),
                         "YOB": rng.integers(1940, 1980, n),
                         "Education": "[2]",
                         "Date of Death": pd.NaT})
    bmi = pd.DataFrame({"id": ids, "BMI": rng.normal(27, 4, n).round(1)})
    rows = [(i, "Z00", start + pd.Timedelta(days=int(rng.integers(0, 1000)))) for i in ids]
    for i in ids:
        if rng.random() < 0.3:
            rows.append((i, "I10", start + pd.Timedelta(days=int(rng.integers(0, 1500)))))
        if rng.random() < 0.25:
            rows.append((i, "F32", start + pd.Timedelta(days=int(rng.integers(3000, 7000)))))
        if rng.random() < 0.08:
            rows.append((i, "G30", start + pd.Timedelta(days=int(rng.integers(4000, 9000)))))
    icd = pd.DataFrame(rows, columns=["id", "code", "date"])
    icd["date"] = pd.to_datetime(icd["date"])
    return icd, demo, bmi


@pytest.fixture(scope="module")
def matched(inputs):
    icd, demo, bmi = inputs
    b = CohortBuilder(washout_months=12, require_observation=True, **CONF)
    b.attach_dataframes(icd10_long=icd, demographics=demo, bmi=bmi)
    cases = b.build_case_cohort()
    controls = b.build_control_cohort(cases)
    master = b.build_master_df(cases, controls)
    psm = PSMCalculator(master, exact_match_cols=["Sex"], ratio=3, consort=b.consort)
    psm.fit()
    psm.match()
    return b, psm, cases, controls


def test_cohorts_non_empty(matched):
    _, psm, cases, controls = matched
    assert len(cases) > 0
    assert len(controls) > 0
    assert len(psm.matched_cohort) > 0


def test_time_zero_uses_only_past(matched):
    """A control's own exposure onset must fall after its time zero."""
    _, _, _, controls = matched
    later = controls.dropna(subset=["exposure_onset_date"])
    assert (later["exposure_onset_date"] > later["index_date"]).all()


def test_exposed_have_no_exposure_censoring(matched):
    _, _, cases, _ = matched
    assert cases["exposure_onset_date"].isna().all()


def test_matched_sets_share_sex(matched):
    _, psm, _, _ = matched
    g = psm.matched_cohort.groupby("match_id")["Sex"].nunique()
    assert (g == 1).all()


def test_balance_improves(matched):
    _, psm, _, _ = matched
    tbl = psm.balance_table(verbose=False)
    assert tbl["SMD_post"].abs().max() <= 0.25


def test_survival_stage(matched):
    _, psm, _, _ = matched
    sa = SurvivalAnalyser(psm.matched_cohort, study_end="2024-12-31")
    sa.prepare()
    keep, _ = full_rank_covariates(sa, verbose=False)
    sa = SurvivalAnalyser(psm.matched_cohort, study_end="2024-12-31", covariates=keep)
    sa.prepare()
    sa.fit(adjusted=True, cluster=False)
    eff = sa.exposure_effect()
    assert eff["HR"] > 0
    assert eff["CI_low"] < eff["HR"] < eff["CI_high"]


def test_followup_positive(matched):
    _, psm, _, _ = matched
    sa = SurvivalAnalyser(psm.matched_cohort, study_end="2024-12-31")
    d = sa.prepare()
    assert (d["followup_years"] > 0).all()


def test_e_value_at_least_one(matched):
    _, psm, _, _ = matched
    sa = SurvivalAnalyser(psm.matched_cohort, study_end="2024-12-31")
    sa.prepare()
    sa.fit(adjusted=False, cluster=False)
    ev = sa.e_value(verbose=False)
    assert ev["E_value_estimate"] >= 1.0
