"""Run the full CohortLearn pipeline on a set of input tables.

Edit CONF and the paths, then:

    python scripts/run_pipeline.py
"""

import argparse
import os

import pandas as pd

from cohortlearn import CohortBuilder, PSMCalculator, SurvivalAnalyser, rename_id
from cohortlearn.survival import full_rank_covariates


CONF = {
    "exposure_codes": ["F32", "F33"],
    "outcome_codes": ["F00", "F001", "F002", "F009", "G30", "G301", "G308", "G309"],
    "confounder_codes": {
        "hypertension": ["I10", "I11", "I12", "I13"],
        "diabetes_type1": ["E10"],
        "diabetes_type2": ["E11"],
        "ckd": ["N18"],
        "osteoporosis": ["M80", "M81"],
        "anxiety": ["F40", "F41"],
        "sleep_disorders": ["G47"],
        "ibs": ["K58"],
    },
    "demographic_confounders": {
        "age": True,
        "sex": True,
        "bmi": True,
        "education": True,
        "risk_allele": False,
    },
    "age_range": (18, 110),
}


def load_inputs(data_dir):
    """Read the three flat input tables and harmonise the identifier."""
    icd = pd.read_csv(os.path.join(data_dir, "icd10_long.csv"), parse_dates=["date"])
    demo = pd.read_csv(os.path.join(data_dir, "demographics.csv"),
                       parse_dates=["Date of Death"])
    bmi = pd.read_csv(os.path.join(data_dir, "bmi.csv"))

    icd = rename_id(icd)
    demo = rename_id(demo)
    bmi = rename_id(bmi)

    valid = set(demo["id"])
    icd = icd[icd["id"].isin(valid)].copy()
    bmi = bmi[bmi["id"].isin(valid)].copy()

    print(f"icd10 : {len(icd):,} rows ({icd['id'].nunique():,} ids)")
    print(f"demo  : {len(demo):,} rows")
    print(f"bmi   : {len(bmi):,} rows")
    return icd, demo, bmi


def build_and_match(icd, demo, bmi, washout_months, ratio=3, caliper_sd=0.2, seed=42):
    """Build both arms for one washout window, then match."""
    builder = CohortBuilder(washout_months=washout_months,
                            require_observation=True, **CONF)
    builder.attach_dataframes(icd10_long=icd, demographics=demo, bmi=bmi)

    cases = builder.build_case_cohort()
    controls = builder.build_control_cohort(cases, seed=seed)
    master = builder.build_master_df(cases, controls)
    # Stratify age into bands for matching, this gives a more representative distribution of controls across the age range, rather than just matching the mean
    master = master.assign(
        age_band=pd.cut(master["age_at_index"], [0, 40, 50, 60, 70, 80, 200],
                        right=False).astype(str))

    psm = PSMCalculator(master, treatment_col="is_exposed",
                        exact_match_cols=["Sex", "age_band"], ratio=ratio,
                        caliper_sd=caliper_sd, calendar_bucket_years=2,
                        calendar_exact=True, consort=builder.consort,
                        random_state=seed)
    psm.fit()
    psm.match()
    psm.summary()
    return builder, psm


def main():
    parser = argparse.ArgumentParser(description="Run the CohortLearn pipeline.")
    parser.add_argument("--data-dir", default="data",
                        help="directory holding the three input CSV files")
    parser.add_argument("--out-dir", default="outputs",
                        help="directory for matched cohorts and figures")
    parser.add_argument("--washout-months", type=int, default=12)
    parser.add_argument("--ratio", type=int, default=3)
    parser.add_argument("--study-end", default="2024-12-31")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sweep", nargs="*", type=int,
                        default=None, help="washout windows in months")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    icd, demo, bmi = load_inputs(args.data_dir)

    print("\n=== cohort construction and matching ===")
    builder, psm = build_and_match(icd, demo, bmi, args.washout_months,
                                   ratio=args.ratio, seed=args.seed)
    matched = psm.matched_cohort

    print("\n=== participant flow ===")
    builder.consort.summary()

    print("\n=== look-back symmetry ===")
    builder.lookback_report(builder.cases, builder.controls)
    print("\n=== balance ===")
    psm.balance_table()
    psm.plot_overlap(save_path=os.path.join(args.out_dir, "ps_overlap.pdf"))
    psm.plot_smd(save_path=os.path.join(args.out_dir, "love_plot.pdf"))
    print()
    psm.strata_table()

    print("\n=== outcome model ===")
    probe = SurvivalAnalyser(matched, study_end=args.study_end)
    probe.prepare()
    keep, dropped = full_rank_covariates(probe)

    sa = SurvivalAnalyser(matched, study_end=args.study_end, covariates=keep)
    sa.prepare()
    sa.fit(adjusted=True, cluster=False)
    print()
    sa.summary(exposure_only=True)

    print("\n=== sensitivity to unmeasured confounding ===")
    sa.e_value()

    path = os.path.join(args.out_dir, f"matched_cohort_{args.washout_months}m.csv")
    matched.to_csv(path, index=False)
    print(f"\nsaved {path}")

    if args.sweep:
        print("\n=== washout sweep ===")
        rows = []
        for w in args.sweep:
            b_w, psm_w = build_and_match(icd, demo, bmi, w,
                                         ratio=args.ratio, seed=args.seed)
            s = SurvivalAnalyser(psm_w.matched_cohort, study_end=args.study_end,
                                 covariates=keep)
            s.prepare()
            s.fit(adjusted=True, cluster=False)
            e = s.exposure_effect()
            n_ev = int(s.data["event"].sum())
            rows.append({"washout_years": w // 12, "events": n_ev,
                         "HR": e["HR"], "CI_low": e["CI_low"],
                         "CI_high": e["CI_high"], "p": e["p"]})
        sweep = pd.DataFrame(rows)
        sweep["reportable"] = sweep["events"] >= 20
        print(sweep.to_string(index=False))
        sweep.to_csv(os.path.join(args.out_dir, "washout_sweep.csv"), index=False)


if __name__ == "__main__":
    main()
