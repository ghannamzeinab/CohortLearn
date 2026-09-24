"""Generate a synthetic electronic health record dataset.

Version 2. The change from version 1 is how the exposure date is drawn.

Version 1 drew the exposure date uniformly from a fixed calendar window, with no
reference to the person. Control index dates were then taken from that same pool
and clipped by each control's eligible window. The two arms therefore differed in
age at index for a purely mechanical reason, which no propensity score can remove,
because it is not confounding.

Version 2 draws an age at diagnosis first, then converts it to a calendar date
through the person's date of birth. Diagnosis timing is now a function of the
person, as it is in real records, and the mechanical age gap disappears.

Exposure still depends on the covariates, so real confounding remains. The outcome
still depends on exposure with a known coefficient, so the true effect is known.

Two outcome models are available.

logistic (default)
    Exposure changes the log-odds of ever having the outcome. The age at
    outcome is drawn independently of exposure. The known effect is an odds
    ratio. No true hazard ratio is defined, so a Cox estimate can only be
    checked for direction.

hazard
    Outcome ages are drawn from a Gompertz proportional hazards model on the
    age scale, with exposure as a time-varying covariate that switches on at
    the exposure date (Bender et al. 2005; Austin 2012). The known effect is a
    hazard ratio, so a Cox estimate can be checked against it directly.

Only the outcome dates differ between the two models. Demographics, BMI,
comorbidities, exposure and background codes are identical for the same seed.
"""

import json

import numpy as np
import pandas as pd

SEED = 42
FIRST_ID = 1_000_000

STUDY_START = pd.Timestamp("2000-01-01")
STUDY_END = pd.Timestamp("2024-12-31")
STUDY_DAYS = (STUDY_END - STUDY_START).days

CONFOUNDER_PREV = {"I10": 0.30, "E11": 0.14, "E10": 0.04, "N18": 0.05,
                   "M80": 0.05, "F41": 0.20, "G47": 0.15, "K58": 0.03}
BACKGROUND = ("Z00", "Z01", "M54", "J06", "R51")

COMORB_WINDOW = (0, int(STUDY_DAYS * 0.25))

AGE_AT_EXPOSURE_MEAN = 58.0
AGE_AT_EXPOSURE_SD = 11.0
AGE_AT_EXPOSURE_RANGE = (18.0, 90.0)

AGE_AT_OUTCOME_MEAN = 76.0
AGE_AT_OUTCOME_SD = 7.0

# Gompertz baseline hazard for the hazard model, on the age scale from age 50.
# The hazard doubles about every 5 years of age, as dementia incidence does.
GOMPERTZ_START_AGE = 50.0
GOMPERTZ_LAMBDA = 1.0e-4
GOMPERTZ_GAMMA = np.log(2) / 5
HAZARD_I10_LOG_HR = 0.20


def make_demographics(n, seed=SEED, death_rate=0.12, first_id=FIRST_ID):
    """One row per participant: id, Sex, YOB, Education, Date of Death."""
    rng = np.random.default_rng(seed)
    ids = np.arange(first_id, first_id + n)

    sex = rng.integers(0, 2, n)
    yob = rng.integers(1935, 1985, n)

    levels = np.array([-1, 0, 1, 2, 3])
    probs = np.array([0.02, 0.10, 0.20, 0.29, 0.39])
    n_vals = rng.choice([1, 2], n, p=[0.85, 0.15])
    education = [str(sorted(rng.choice(levels, size=k, replace=False, p=probs).tolist()))
                 for k in n_vals]

    dod = pd.Series(pd.NaT, index=range(n), dtype="datetime64[ns]")
    died = rng.random(n) < death_rate
    dod[died] = STUDY_START + pd.to_timedelta(rng.integers(0, STUDY_DAYS, died.sum()),
                                              unit="D")

    return pd.DataFrame({"id": ids, "Sex": sex, "YOB": yob,
                         "Education": education, "Date of Death": dod.values})


def make_bmi(ids, seed=SEED, missing_rate=0.03):
    """One row per participant: id, BMI."""
    ids = np.asarray(ids)
    rng = np.random.default_rng(seed + 1)
    n = len(ids)
    bmi = pd.Series(np.round(np.exp(rng.normal(np.log(27.5), 0.22, n)), 1).clip(14, 60))
    bmi[rng.random(n) < missing_rate] = np.nan
    return pd.DataFrame({"id": ids, "BMI": bmi.values})


def _truncated_normal(rng, mean, sd, lo, hi):
    """Normal draw restricted to [lo, hi], element-wise, by inverse transform."""
    from scipy.stats import norm
    a = norm.cdf((lo - mean) / sd)
    b = norm.cdf((hi - mean) / sd)
    u = a + rng.random(len(lo)) * (b - a)
    u = np.clip(u, 1e-9, 1 - 1e-9)
    return mean + sd * norm.ppf(u)


def _date_from_age(dob, age_years):
    """Calendar date at which a person born on dob reaches the given age."""
    return dob + pd.to_timedelta(np.asarray(age_years) * 365.25, unit="D")


def _gompertz_inverse(y, lam, gam):
    """Age at which the Gompertz cumulative baseline hazard reaches y."""
    return GOMPERTZ_START_AGE + np.log1p(y * gam / lam) / gam


def _gompertz_cumhaz(age, lam, gam):
    """Cumulative Gompertz baseline hazard from GOMPERTZ_START_AGE to age."""
    a = np.maximum(age - GOMPERTZ_START_AGE, 0.0)
    return (lam / gam) * np.expm1(gam * a)


def _hazard_outcome_ages(rng, age_exp, exposed, i10, log_hr,
                         lam=GOMPERTZ_LAMBDA, gam=GOMPERTZ_GAMMA):
    """Outcome ages from a Cox model with time-varying exposure.

    Hazard at age a:  h0(a) * exp(b_i10 * I10 + log_hr * 1[a >= exposure age]).
    Drawn by inverting the piecewise cumulative hazard (Austin 2012).
    """
    n = len(exposed)
    c = np.exp(HAZARD_I10_LOG_HR * i10)
    e = np.where(exposed, np.maximum(age_exp, GOMPERTZ_START_AGE), np.inf)
    target = -np.log(rng.random(n))

    h_e = np.where(np.isfinite(e), c * _gompertz_cumhaz(np.where(np.isfinite(e), e, 0.0),
                                                        lam, gam), np.inf)
    before = target < h_e
    ages = np.empty(n)
    ages[before] = _gompertz_inverse(target[before] / c[before], lam, gam)
    after = ~before
    g_e = _gompertz_cumhaz(e[after], lam, gam)
    ages[after] = _gompertz_inverse(
        g_e + (target[after] - h_e[after]) / (c[after] * np.exp(log_hr)), lam, gam)
    return ages


def make_icd10_long(demographics_df, BMI_df, n_dx=12, seed=SEED,
                    exposure_codes=("F32", "F33"),
                    outcome_codes=("G30", "F00"),
                    true_effect=0.5,
                    age_coef=-0.02,
                    prevalent_frac=0.08,
                    outcome_model="logistic"):
    """Long diagnosis table: id, code, date.

    Exposure probability is a logistic function of the participant's own
    comorbidities, body mass index, age and education. The exposure date is then
    set from a drawn age at diagnosis rather than from a calendar window, so
    diagnosis timing depends on the person.

    Set age_coef to 0.0 to remove age from the exposure model. Age imbalance
    before matching is then sampling noise only.

    true_effect is on the log scale of the chosen outcome model: log-odds for
    outcome_model="logistic", log-hazard for outcome_model="hazard".
    """
    if outcome_model not in ("logistic", "hazard"):
        raise ValueError("outcome_model must be 'logistic' or 'hazard'")
    rng = np.random.default_rng(seed + 2)
    demo = demographics_df.set_index("id")
    bmi_s = BMI_df.set_index("id")["BMI"]
    ids = demo.index.to_numpy()
    n = len(ids)

    dob = pd.to_datetime(
        pd.DataFrame({"year": demo["YOB"].to_numpy(), "month": 7, "day": 1}))
    dob = pd.Series(dob.to_numpy(), index=range(n))

    flags = {c: (rng.random(n) < p).astype(int) for c, p in CONFOUNDER_PREV.items()}

    bmi_v = bmi_s.reindex(ids).to_numpy()
    bmi_f = np.where(np.isnan(bmi_v), 27.5, bmi_v)
    edu_hi = demo["Education"].astype(str).str.contains("3").to_numpy().astype(int)

    # ---- age at exposure, drawn per person -------------------------------
    # The draw is truncated to the ages at which this person falls inside the
    # study window. Without truncation, a large share of drawn dates would land
    # outside the window and those participants would silently lose their
    # exposure, which distorts both the prevalence and the age distribution.
    age_lo_win = (STUDY_START - dob).dt.days.to_numpy() / 365.25
    age_hi_win = (STUDY_END - dob).dt.days.to_numpy() / 365.25
    lo = np.maximum(age_lo_win, AGE_AT_EXPOSURE_RANGE[0])
    hi = np.minimum(age_hi_win, AGE_AT_EXPOSURE_RANGE[1])
    hi = np.maximum(hi, lo + 0.5)

    age_exp = _truncated_normal(rng, AGE_AT_EXPOSURE_MEAN, AGE_AT_EXPOSURE_SD, lo, hi)

    # ---- exposure model: confounding by the covariates -------------------
    logit = (-1.6
             + 0.90 * flags["F41"]
             + 0.50 * flags["G47"]
             + 0.30 * flags["I10"]
             + 0.03 * (bmi_f - 27)
             + age_coef * (age_exp - 58)
             - 0.35 * edu_hi)
    exposed = rng.random(n) < 1 / (1 + np.exp(-logit))

    exp_date = pd.Series(pd.NaT, index=range(n), dtype="datetime64[ns]")
    exp_date[exposed] = _date_from_age(dob[exposed], age_exp[exposed])

    # keep only exposures that fall inside the study window
    inside = exposed & exp_date.between(STUDY_START, STUDY_END).to_numpy()
    exp_date[~inside] = pd.NaT
    exposed = inside

    # ---- outcome model: depends on exposure and age ----------------------
    lo_o = np.maximum(age_lo_win, 50.0)
    hi_o = np.maximum(np.minimum(age_hi_win, 100.0), lo_o + 0.5)
    age_out = _truncated_normal(rng, AGE_AT_OUTCOME_MEAN, AGE_AT_OUTCOME_SD, lo_o, hi_o)
    o_logit = -3.6 + true_effect * exposed + 0.05 * (age_out - 76) + 0.20 * flags["I10"]
    has_out = rng.random(n) < 1 / (1 + np.exp(-o_logit))

    out_date = pd.Series(pd.NaT, index=range(n), dtype="datetime64[ns]")
    cand = _date_from_age(dob, age_out)
    out_date[has_out] = cand[has_out]

    # an outcome must not precede exposure, except for the prevalent fraction
    both = has_out & exposed
    bad = both & (out_date <= exp_date).to_numpy()
    keep_prevalent = bad & (rng.random(n) < prevalent_frac)
    out_date[bad & ~keep_prevalent] = pd.NaT

    inside_out = out_date.between(STUDY_START, STUDY_END).to_numpy()
    out_date[~inside_out] = pd.NaT

    if outcome_model == "hazard":
        # Separate random stream, so every other table matches the logistic run.
        rng_h = np.random.default_rng(seed + 3)
        age_h = _hazard_outcome_ages(rng_h, age_exp, exposed, flags["I10"], true_effect)
        out_date = pd.Series(pd.NaT, index=range(n), dtype="datetime64[ns]")
        ok = age_h <= 100.0
        out_date[ok] = _date_from_age(dob[ok], age_h[ok])
        dod = pd.to_datetime(demo["Date of Death"]).reset_index(drop=True)
        after_death = dod.notna() & (out_date > dod)
        out_date[after_death] = pd.NaT
        inside_h = out_date.between(STUDY_START, STUDY_END).to_numpy()
        out_date[~inside_h] = pd.NaT

    # ---- assemble the long table -----------------------------------------
    frames = []

    k_bg = 1 + rng.integers(0, n_dx, n)
    frames.append(pd.DataFrame({
        "id": np.repeat(ids, k_bg),
        "code": rng.choice(np.array(BACKGROUND), int(k_bg.sum())),
        "date": STUDY_START + pd.to_timedelta(
            rng.integers(0, STUDY_DAYS, int(k_bg.sum())), unit="D"),
    }))

    for code, f in flags.items():
        sel = ids[f == 1]
        frames.append(pd.DataFrame({
            "id": sel, "code": code,
            "date": STUDY_START + pd.to_timedelta(
                rng.integers(*COMORB_WINDOW, len(sel)), unit="D")}))

    em = exp_date.notna().to_numpy()
    frames.append(pd.DataFrame({"id": ids[em],
                                "code": rng.choice(exposure_codes, em.sum()),
                                "date": exp_date[em].values}))

    om = out_date.notna().to_numpy()
    frames.append(pd.DataFrame({"id": ids[om],
                                "code": rng.choice(outcome_codes, om.sum()),
                                "date": out_date[om].values}))

    rows = pd.concat(frames, ignore_index=True)
    rows["code"] = rows["code"].astype(str).str.upper()
    rows["date"] = pd.to_datetime(rows["date"])
    rows = (rows.dropna(subset=["date"])
                .drop_duplicates(["id", "code", "date"])
                .sort_values(["id", "date"], kind="mergesort")
                .reset_index(drop=True))

    print(f"exposure prevalence {em.mean():.1%} | outcome prevalence {om.mean():.1%}")
    return rows


if __name__ == "__main__":
    import argparse
    import os

    ap = argparse.ArgumentParser(description="Generate a synthetic EHR dataset.")
    ap.add_argument("--n", type=int, default=60_000, help="number of participants")
    ap.add_argument("--out-dir", default="data", help="output directory")
    ap.add_argument("--n-dx", type=int, default=12,
                    help="background codes per participant")
    ap.add_argument("--age-coef", type=float, default=-0.02,
                    help="age coefficient in the exposure model, 0 to remove age")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--outcome-model", choices=["logistic", "hazard"],
                    default="logistic",
                    help="logistic: effect on odds of ever having the outcome; "
                         "hazard: effect on the hazard (true HR known)")
    ap.add_argument("--true-effect", type=float, default=0.5,
                    help="exposure effect on the log scale of the outcome model")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    demo = make_demographics(args.n, seed=args.seed)
    ids = demo["id"].to_numpy()
    bmi = make_bmi(ids, seed=args.seed)
    icd = make_icd10_long(demo, bmi, n_dx=args.n_dx, seed=args.seed,
                          age_coef=args.age_coef, true_effect=args.true_effect,
                          outcome_model=args.outcome_model)

    demo.to_csv(os.path.join(args.out_dir, "demographics.csv"), index=False)
    bmi.to_csv(os.path.join(args.out_dir, "bmi.csv"), index=False)
    icd.to_csv(os.path.join(args.out_dir, "icd10_long.csv"), index=False)

    info = {
        "outcome_model": args.outcome_model,
        "true_effect_log": args.true_effect,
        "true_effect": float(np.exp(args.true_effect)),
        "effect_scale": "odds ratio" if args.outcome_model == "logistic" else "hazard ratio",
        "n": args.n,
        "seed": args.seed,
    }
    with open(os.path.join(args.out_dir, "generation_info.json"), "w") as f:
        json.dump(info, f, indent=2)

    print(f"wrote {args.out_dir}/: {args.n:,} participants, {len(icd):,} diagnosis rows")
