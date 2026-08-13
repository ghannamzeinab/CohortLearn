"""Cox outcome stage on the matched cohort."""

import numpy as np
import pandas as pd


class SurvivalAnalyser:

    RENAME = {
        "index_date":          "time_zero",
        "exposure_onset_date": "censor_at_exposure_date",   # controls only
        "first_outcome_date":  "outcome_date",
        "Date of Death":       "death_date",
    }

    def __init__(self, matched_cohort, study_end, treatment_col="is_exposed",
                 id_col="id", covariates=None, strata=None):
        """
        matched_cohort : output of PSMCalculator.match()
        study_end      : administrative end of follow-up, e.g. '2024-12-31'
        covariates     : adjustment set for the Cox model. None -> auto-detect the
                         same pre-baseline covariates used in the propensity model,
                         which gives the doubly robust design (matching + regression).
        strata         : optional list of columns for a stratified baseline hazard
                         (e.g. ['age_band']) when proportional hazards is doubtful.
        """
        self.raw = matched_cohort.copy()
        self.study_end = pd.Timestamp(study_end)
        self.treatment_col = treatment_col
        self.id_col = id_col
        self.covariates = covariates
        self.strata = strata

        self.data = None          # survival-ready frame
        self._fit_df = None
        self.model = None
        self.results_ = None

    # ---------- data preparation ----------

    def prepare(self, min_followup_days=1):
        """Build the follow-up clock, the event flag, and a censoring-reason column."""
        df = self.raw.rename(columns={k: v for k, v in self.RENAME.items()
                                      if k in self.raw.columns}).copy()

        for c in ["time_zero", "censor_at_exposure_date", "outcome_date", "death_date"]:
            df[c] = pd.to_datetime(df[c], errors="coerce") if c in df.columns else pd.NaT

        BIG = pd.Timestamp.max.normalize()
        censor = pd.concat([df["censor_at_exposure_date"], df["death_date"],
                            pd.Series(self.study_end, index=df.index)], axis=1)
        censor = censor.apply(lambda s: s.fillna(BIG)).min(axis=1)

        ev = df["outcome_date"]
        had_event = ev.notna() & (ev > df["time_zero"]) & (ev <= censor)

        df["followup_end_date"] = ev.where(had_event, censor)
        df["event"] = had_event.astype(int)
        df["followup_years"] = (df["followup_end_date"] - df["time_zero"]).dt.days / 365.25

        reason = pd.Series("administrative end", index=df.index)
        reason[df["death_date"].notna() & (df["death_date"] == censor)] = "death"
        reason[df["censor_at_exposure_date"].notna() &
               (df["censor_at_exposure_date"] == censor)] = "became exposed"
        reason[had_event] = "outcome"
        df["followup_end_reason"] = reason

        n0 = len(df)
        df = df[df["followup_years"] >= min_followup_days / 365.25].copy()
        if n0 - len(df):
            print(f"dropped {n0 - len(df):,} rows with non-positive follow-up")

        self.data = df
        self._report_followup()
        return df

    def _report_followup(self):
        d = self.data
        for grp, name in [(1, "Exposed"), (0, "Control")]:
            s = d[d[self.treatment_col] == grp]
            ev, py = int(s["event"].sum()), s["followup_years"].sum()
            rate = 1000 * ev / py if py else np.nan
            flag = "  [<20 -- suppress]" if 0 < ev < 20 else ""
            print(f"  {name:<8} n = {len(s):>8,} | events = {ev:>6,} | "
                  f"person-years = {py:>10,.0f} | rate/1000py = {rate:>5.2f}{flag}")
        print("\ncensoring reasons:")
        print(d["followup_end_reason"].value_counts().to_string())

    # ---------- model ----------

    def _auto_covariates(self):
        cols = [c for c in ["age_at_index", "BMI", "Sex"] if c in self.data.columns]
        cols += sorted(c for c in self.data.columns if c.startswith("pre_"))
        cols += sorted(c for c in self.data.columns if c.startswith("edu_"))
        return cols

    def fit(self, cluster=True, adjusted=True, penalizer=0.0):
        """Fit the Cox model. adjusted=False gives the matching-only estimate."""
        from lifelines import CoxPHFitter
        if self.data is None:
            self.prepare()

        covs = (self.covariates if self.covariates is not None else self._auto_covariates()) \
            if adjusted else []
        covs = [c for c in covs if c in self.data.columns]
        use = [self.treatment_col, "followup_years", "event"] + covs
        if self.strata:
            use += [s for s in self.strata if s in self.data.columns]

        fit_df = self.data[list(dict.fromkeys(use + [self.id_col]))].copy()
        for c in [self.treatment_col] + covs:
            fit_df[c] = pd.to_numeric(fit_df[c], errors="coerce")
        fit_df = fit_df.dropna()

        self.model = CoxPHFitter(penalizer=penalizer)
        kwargs = dict(duration_col="followup_years", event_col="event")
        if cluster:
            kwargs["cluster_col"] = self.id_col          # robust SEs, one person may recur
        else:
            fit_df = fit_df.drop(columns=[self.id_col])
        if self.strata:
            kwargs["strata"] = [s for s in self.strata if s in fit_df.columns]

        self._fit_df = fit_df
        self.model.fit(fit_df, **kwargs)
        self.results_ = self._tidy()
        return self

    def _tidy(self):
        s = self.model.summary
        out = pd.DataFrame({
            "term": s.index,
            "HR": np.exp(s["coef"]),
            "CI_low": np.exp(s["coef lower 95%"]),
            "CI_high": np.exp(s["coef upper 95%"]),
            "p": s["p"],
        }).reset_index(drop=True)
        out["HR (95% CI)"] = out.apply(
            lambda r: f"{r['HR']:.2f} ({r['CI_low']:.2f}-{r['CI_high']:.2f})", axis=1)
        return out

    def summary(self, exposure_only=False):
        """Reportable table: HR, 95% CI, p-value."""
        if self.results_ is None:
            raise RuntimeError("Call fit() first.")
        t = self.results_
        if exposure_only:
            t = t[t["term"] == self.treatment_col]
        cols = ["term", "HR", "CI_low", "CI_high", "HR (95% CI)", "p"]
        with pd.option_context("display.float_format", lambda v: f"{v:.4f}"):
            print(t[cols].to_string(index=False))
        print(f"\nconcordance {self.model.concordance_index_:.3f} | "
              f"n = {self.model._n_examples:,} | events = {int(self.model.event_observed.sum()):,}")
        print("Covariate coefficients are adjustment terms, not causal effects (Table 2 fallacy).")
        return t

    def exposure_effect(self):
        """Just the primary estimate, as a dict."""
        r = self.results_[self.results_["term"] == self.treatment_col].iloc[0]
        return {"HR": float(r["HR"]), "CI_low": float(r["CI_low"]),
                "CI_high": float(r["CI_high"]), "p": float(r["p"])}

    def check_proportional_hazards(self, p_threshold=0.05):
        """Schoenfeld residual test -- reviewers expect this for any Cox model."""
        if self.model is None:
            raise RuntimeError("Call fit() first.")
        print("Proportional-hazards check (Schoenfeld residuals):")
        return self.model.check_assumptions(self._fit_df,
                                            p_value_threshold=p_threshold,
                                            show_plots=False)

    # ---------- sensitivity to unmeasured confounding ----------

    @staticmethod
    def _e(rr):
        """E-value for a single risk-ratio-scale estimate."""
        if rr <= 0 or not np.isfinite(rr):
            return np.nan
        if rr < 1:
            rr = 1 / rr
        return rr + np.sqrt(rr * (rr - 1))

    def e_value(self, rare_outcome=True, verbose=True):
        """E-value for the exposure effect and for the CI limit closest to the null.

        The E-value is the minimum association an *unmeasured* confounder would need
        with both exposure and outcome, above the measured covariates, to explain the
        estimate away (VanderWeele & Ding 2017). It says nothing about any specific
        named variable -- a measurable confounder should simply be added and the
        analysis rerun.
        """
        if self.results_ is None:
            raise RuntimeError("Call fit() first.")
        est = self.exposure_effect()
        hr, lo, hi = est["HR"], est["CI_low"], est["CI_high"]

        def to_rr(h):
            # rare outcome -> HR approximates RR; otherwise use the VanderWeele-Ding
            # square-root transformation
            if rare_outcome:
                return h
            return (1 - 0.5 ** np.sqrt(h)) / (1 - 0.5 ** np.sqrt(1 / h))

        rr = to_rr(hr)
        e_point = self._e(rr)

        # CI limit nearest the null; if the interval crosses 1, no confounding is needed
        if lo > 1:
            e_ci = self._e(to_rr(lo))
        elif hi < 1:
            e_ci = self._e(to_rr(hi))
        else:
            e_ci = 1.0

        out = {"HR": hr, "CI_low": lo, "CI_high": hi,
               "E_value_estimate": e_point, "E_value_CI": e_ci,
               "rare_outcome_assumed": rare_outcome}
        if verbose:
            print(f"HR {hr:.2f} ({lo:.2f}-{hi:.2f})")
            print(f"E-value, point estimate : {e_point:.2f}")
            print(f"E-value, CI limit       : {e_ci:.2f}"
                  + ("   [CI includes 1]" if e_ci == 1.0 else ""))
            print("\nAn unmeasured confounder would need associations of at least this "
                  "size\nwith both exposure and outcome, beyond the measured covariates, "
                  "to explain\nthe result away. Interpret alongside its published critiques.")
        self.e_value_ = out
        return out


import numpy as np
import pandas as pd


def full_rank_covariates(sa, verbose=True):
    """Return the largest adjustment set that leaves the Cox design matrix full rank.

    Columns are added one at a time and kept only if they increase the rank of the
    design. The exposure is always kept. This resolves constant columns, complete
    dummy sets that sum to one, and duplicated covariates in a single pass.
    """
    covs = sa.covariates if sa.covariates is not None else sa._auto_covariates()
    covs = [c for c in covs if c in sa.data.columns]

    X = sa.data[[sa.treatment_col] + covs].apply(pd.to_numeric, errors="coerce").dropna()
    # Cox has no intercept, so a column combination that is constant is also
    # unidentifiable -- append a constant column to catch that case
    base = np.ones((len(X), 1))

    keep, dropped = [], []
    M = base
    for c in [sa.treatment_col] + covs:
        v = X[[c]].to_numpy(dtype=float)
        trial = np.hstack([M, v])
        if np.linalg.matrix_rank(trial) > np.linalg.matrix_rank(M):
            keep.append(c)
            M = trial
        else:
            dropped.append(c)

    if verbose:
        print(f"kept {len(keep)} of {len(covs) + 1} columns")
        if dropped:
            print("dropped as linearly dependent:", dropped)
        else:
            print("no dependency found -- singularity may come from elsewhere")
    keep_covs = [c for c in keep if c != sa.treatment_col]
    return keep_covs, dropped
