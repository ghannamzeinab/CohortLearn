"""Propensity score estimation, matching and balance diagnostics."""

from typing import Optional, List
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


class PSMCalculator:
    PRE_PREFIX = "pre_"
    EDU_PREFIX = "edu_"
    DEFAULT_CONTINUOUS = ["age_at_index", "BMI"]
    DEFAULT_EXACT = ["Sex"]
    STRATUM_SEP = "||"

    # WHO adult BMI bands.
    BMI_BINS = [0, 18.5, 25, 30, 35, 40, np.inf]
    BMI_LABELS = ["Underweight (<18.5)", "Normal (18.5-24.9)", "Overweight (25-29.9)",
                  "Obese I (30-34.9)", "Obese II (35-39.9)", "Obese III (>=40)"]
    AGE_BINS = [18, 40, 50, 60, 70, 80, np.inf]
    AGE_LABELS = ["18-39", "40-49", "50-59", "60-69", "70-79", "80+"]

    def __init__(self, master, treatment_col="is_exposed",
                 numeric_cols=None, categorical_cols=None, exact_match_cols=None,
                 ratio=3, caliper_sd=0.2, calendar_col="index_year",
                 calendar_bucket_years=2, calendar_exact=True,
                 random_state=42, consort=None):
        """
        ratio : controls per case (1:k).
        caliper_sd : caliper width as a multiple of the SD of the logit PS (Austin's 0.2).
        calendar_col / calendar_bucket_years / calendar_exact : force cases and their
            controls into the same time window; set calendar_exact=False to switch off.
        """
        if not (isinstance(ratio, (int, np.integer)) and ratio >= 1):
            raise ValueError(f"ratio must be an integer >= 1, got {ratio!r}.")
        if not (np.isscalar(caliper_sd) and caliper_sd > 0):
            raise ValueError(f"caliper_sd must be > 0, got {caliper_sd!r}.")

        self.master = master.reset_index(drop=True)
        self.treatment_col = treatment_col
        self.ratio = ratio
        self.caliper_sd = caliper_sd
        self.calendar_col = calendar_col
        self.calendar_bucket_years = calendar_bucket_years
        self.calendar_exact = calendar_exact
        self.random_state = random_state
        self.consort = consort

        groups = self.suggest_column_groups(self.master, treatment_col)
        self.numeric_cols = numeric_cols if numeric_cols is not None else groups["numeric"]
        self.categorical_cols = categorical_cols if categorical_cols is not None else groups["categorical"]
        self.exact_match_cols = exact_match_cols if exact_match_cols is not None else groups["exact"]

        self.ps_model = None
        self.matched_cohort = None
        self._caliper = None

    # ---------- column discovery ----------

    @classmethod
    def suggest_column_groups(cls, df, treatment_col="is_exposed"):
        numeric = [c for c in cls.DEFAULT_CONTINUOUS if c in df.columns]
        numeric += sorted(c for c in df.columns if c.startswith(cls.PRE_PREFIX))
        numeric += sorted(c for c in df.columns if c.startswith(cls.EDU_PREFIX))
        categorical = [c for c in ["APOE"] if c in df.columns]
        exact = [c for c in cls.DEFAULT_EXACT if c in df.columns]
        return {"numeric": numeric, "categorical": categorical, "exact": exact}

    # ---------- propensity model ----------

    def _design_matrix(self, df):
        X = pd.DataFrame(index=df.index)
        for c in self.numeric_cols:
            if c in df.columns:
                X[c] = pd.to_numeric(df[c], errors="coerce")
        if self.categorical_cols:
            present = [c for c in self.categorical_cols if c in df.columns]
            if present:
                X = pd.concat([X, pd.get_dummies(df[present].astype("object"),
                                                 prefix=present, dummy_na=True)], axis=1)
        return X.fillna(X.median(numeric_only=True))

    def fit(self):
        """Fit the propensity model and store the score on every row."""
        df = self.master
        X = self._design_matrix(df)
        y = df[self.treatment_col].astype(int).values
        self.ps_model = LogisticRegression(max_iter=2000, solver="lbfgs")
        self.ps_model.fit(X.values, y)
        p = np.clip(self.ps_model.predict_proba(X.values)[:, 1], 1e-6, 1 - 1e-6)
        self.master["ps"] = p
        self.master["ps_logit"] = np.log(p / (1 - p))
        self._caliper = self.caliper_sd * self.master["ps_logit"].std(ddof=1)
        print(f"Propensity model fitted on {X.shape[1]} features. Caliper = {self._caliper:.4f} (logit).")
        return self

    # ---------- matching ----------

    def _stratum_keys(self, df):
        cols = list(self.exact_match_cols)
        if self.calendar_exact and self.calendar_col in df.columns:
            b = (pd.to_numeric(df[self.calendar_col], errors="coerce")
                 // self.calendar_bucket_years * self.calendar_bucket_years)
            df = df.assign(_cal_bucket=b.astype("Int64").astype(str))
            cols = cols + ["_cal_bucket"]
        if not cols:
            return pd.Series(["_all"] * len(df), index=df.index), df
        key = df[cols].astype(str).agg(self.STRATUM_SEP.join, axis=1)
        return key, df

    def match(self):
        """Greedy 1:k nearest-neighbour matching on the logit PS within strata."""
        if self.ps_model is None:
            self.fit()
        df = self.master
        key, df = self._stratum_keys(df)
        df = df.assign(_stratum=key)

        rng = np.random.default_rng(self.random_state)
        matched_index, match_ids = [], []
        n_cases_in = n_cases_matched = n_controls = 0

        for _, g in df.groupby("_stratum", sort=False):
            cases = g[g[self.treatment_col] == 1]
            controls = g[g[self.treatment_col] == 0]
            n_cases_in += len(cases)
            if cases.empty or controls.empty:
                continue

            ctrl_ps = controls["ps_logit"].to_numpy()
            ctrl_used = np.zeros(len(controls), dtype=bool)
            ctrl_ids = controls["id"].to_numpy()

            order = rng.permutation(len(cases))
            for pos in order:
                case = cases.iloc[pos]
                diff = np.abs(ctrl_ps - case["ps_logit"])
                cand = np.where((diff <= self._caliper) & ~ctrl_used & (ctrl_ids != case["id"]))[0] # Explicitly exclude self-match
                if cand.size == 0:
                    continue
                pick = cand[np.argsort(diff[cand])[: self.ratio]]
                ctrl_used[pick] = True
                n_cases_matched += 1
                n_controls += len(pick)

                matched_index.append(cases.index[pos])
                match_ids.append(case["id"])
                matched_index.extend(controls.index[pick])
                match_ids.extend([case["id"]] * len(pick))

        if not matched_index:
            warnings.warn("No matches formed. Loosen the caliper or widen the calendar bucket.")
            self.matched_cohort = df.iloc[0:0].copy()
            return self.matched_cohort

        # Select rows by index in one pass. This keeps the original dtypes, which
        # row-by-row concatenation would not.
        matched = df.loc[matched_index].copy()
        matched["match_id"] = match_ids
        matched["matched_case_id"] = match_ids
        matched = matched.reset_index(drop=True)
        matched = matched.drop(columns=["_stratum", "_cal_bucket"], errors="ignore")
        ratio_achieved = n_controls / n_cases_matched if n_cases_matched else 0.0

        for k, v in [("cases_entering_psm", n_cases_in), ("cases_matched", n_cases_matched),
                     ("cases_lost_in_psm", n_cases_in - n_cases_matched),
                     ("controls_matched", n_controls)]:
            if self.consort:
                self.consort.log(k, v)
        if self.consort:
            self.consort.log("matching_ratio_achieved", round(ratio_achieved, 3))

        self.matched_cohort = matched
        print(f"Matched {n_cases_matched:,}/{n_cases_in:,} cases to {n_controls:,} controls "
              f"(1:{ratio_achieved:.2f}).")
        return self.matched_cohort

    def run(self):
        self.fit()
        self.match()
        return self

    def summary(self):
        if self.matched_cohort is None or self.matched_cohort.empty:
            print("No matched cohort yet.")
            return
        n_e = int((self.matched_cohort[self.treatment_col] == 1).sum())
        n_c = int((self.matched_cohort[self.treatment_col] == 0).sum())
        print(f"Matched cohort: {len(self.matched_cohort):,} rows ({n_e:,} exposed / {n_c:,} controls).")

    # ================= diagnostics =================

    @staticmethod
    def _smd_cont(t, c):
        d = np.sqrt((t.var(ddof=1) + c.var(ddof=1)) / 2)
        return 0.0 if d == 0 else (t.mean() - c.mean()) / d

    @staticmethod
    def _smd_bin(t, c):
        pt, pc = t.mean(), c.mean()
        d = np.sqrt((pt * (1 - pt) + pc * (1 - pc)) / 2)
        return 0.0 if d == 0 else (pt - pc) / d

    def _one_balance(self, df):
        t = df[df[self.treatment_col] == 1]
        c = df[df[self.treatment_col] == 0]
        rows = []
        for col in ["age_at_index", "BMI"]:
            if col in df.columns:
                tv, cv = pd.to_numeric(t[col], errors="coerce"), pd.to_numeric(c[col], errors="coerce")
                vr = cv.var(ddof=1) / tv.var(ddof=1) if tv.var(ddof=1) else np.nan
                rows.append([col, f"{tv.mean():.2f}\u00b1{tv.std():.2f}",
                             f"{cv.mean():.2f}\u00b1{cv.std():.2f}",
                             self._smd_cont(tv, cv), vr])
        bins = (sorted(x for x in df.columns if x.startswith("pre_"))
                + sorted(x for x in df.columns if x.startswith("edu_")))
        for col in bins:
            tv, cv = pd.to_numeric(t[col], errors="coerce"), pd.to_numeric(c[col], errors="coerce")
            rows.append([col, f"{tv.mean()*100:.1f}%", f"{cv.mean()*100:.1f}%",
                         self._smd_bin(tv, cv), np.nan])
        # Sex and APOE are compared level by level, so any coding works: 0/1, 1/2
        # or 'M'/'F'. Treating a 1/2 column as an indicator gives a mean of 1.5,
        # a negative variance and a NaN SMD.
        for col in ["Sex", "APOE"]:
            if col in df.columns:
                for lvl in sorted(df[col].dropna().unique()):
                    ti = (t[col] == lvl).astype(float)
                    ci = (c[col] == lvl).astype(float)
                    rows.append([f"{col} {lvl}", f"{ti.mean()*100:.1f}%",
                                 f"{ci.mean()*100:.1f}%", self._smd_bin(ti, ci), np.nan])
        return pd.DataFrame(rows, columns=["covariate", "exposed", "control",
                                           "SMD", "variance_ratio"])

    def balance_table(self, verbose=True, vr_bounds=(0.8, 1.25)):
        """SMD before and after matching, plus the variance ratio.

        |SMD| <= 0.1 is the target. An SMD compares first moments only, so a
        variance ratio outside vr_bounds means the distributions still differ
        even when every SMD passes.
        """
        if self.matched_cohort is None:
            raise RuntimeError("Run match() first.")
        post = self._one_balance(self.matched_cohort)
        pre = self._one_balance(self.master)[["covariate", "SMD"]].rename(columns={"SMD": "SMD_pre"})
        tbl = (post.merge(pre, on="covariate", how="left")
                   [["covariate", "exposed", "control", "SMD_pre", "SMD", "variance_ratio"]]
                   .rename(columns={"SMD": "SMD_post"}))
        lo, hi = vr_bounds
        vr = tbl["variance_ratio"]
        tbl["imbalanced"] = ((tbl["SMD_post"].abs() > 0.1)
                             | tbl["SMD_post"].isna()
                             | (vr.notna() & ((vr < lo) | (vr > hi))))
        if verbose:
            with pd.option_context("display.float_format", lambda v: f"{v:.3f}"):
                print(tbl.to_string(index=False))
            print(f"\nmax |SMD| post-match : {tbl['SMD_post'].abs().max():.3f}")
            print(f"covariates flagged   : {int(tbl['imbalanced'].sum())} of {len(tbl)}")
            if tbl["SMD_post"].isna().any():
                print(f"  !! {int(tbl['SMD_post'].isna().sum())} covariate(s) could not "
                      f"be assessed and are counted as imbalanced")
            out_of_band = tbl.loc[vr.notna() & ((vr < lo) | (vr > hi)), "covariate"].tolist()
            if out_of_band:
                print(f"  !! variance ratio outside {vr_bounds} for: "
                      f"{', '.join(out_of_band)} -- means match but spreads do not")
        self.balance_ = tbl
        return tbl

    def plot_overlap(self, bins=40, save_path=None):
        """Propensity overlap for exposed vs control, before and after matching."""
        import matplotlib.pyplot as plt
        import matplotlib as mpl
        if self.matched_cohort is None:
            raise RuntimeError("Run match() first.")
        mpl.rcParams.update({"font.family": "sans-serif",
                             "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                             "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
        c_exp, c_ctrl = "#C44E52", "#4C72B0"

        fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharex=True, sharey=True)
        for ax, data, title in [(axes[0], self.master, "Before matching"),
                                 (axes[1], self.matched_cohort, "After matching")]:
            e = data.loc[data[self.treatment_col] == 1, "ps"].astype(float)
            c = data.loc[data[self.treatment_col] == 0, "ps"].astype(float)
            ax.hist(e, bins=bins, range=(0, 1), density=True, alpha=0.5, color=c_exp, label="Exposed")
            ax.hist(c, bins=bins, range=(0, 1), density=True, alpha=0.5, color=c_ctrl, label="Control")
            ax.set_title(title, fontsize=10)
            ax.set_xlabel("Propensity score")
        axes[0].set_ylabel("Density")
        axes[1].legend(frameon=False, fontsize=8)
        fig.suptitle("Propensity-score overlap", fontsize=11)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, bbox_inches="tight")
            print("saved", save_path)
        return fig

    def _band(self, df):
        out = df.copy()
        if "age_at_index" in out.columns:
            out["_age_band"] = pd.cut(pd.to_numeric(out["age_at_index"], errors="coerce"),
                                      bins=self.AGE_BINS, labels=self.AGE_LABELS, right=False)
        if "BMI" in out.columns:
            out["_bmi_band"] = pd.cut(pd.to_numeric(out["BMI"], errors="coerce"),
                                      bins=self.BMI_BINS, labels=self.BMI_LABELS, right=False)
        return out

    def strata_table(self, verbose=True):
        """Table-1 style breakdown: % of each arm in each stratum after matching."""
        if self.matched_cohort is None:
            raise RuntimeError("Run match() first.")
        df = self._band(self.matched_cohort)
        e = df[df[self.treatment_col] == 1]
        c = df[df[self.treatment_col] == 0]
        ne, nc = max(len(e), 1), max(len(c), 1)
        rows = []

        def pct(sub, mask_e, mask_c, var, level):
            rows.append([var, str(level),
                         f"{mask_e.sum()/ne*100:.1f}%", f"{mask_c.sum()/nc*100:.1f}%"])

        if "Sex" in df.columns:
            for lvl in sorted(df["Sex"].dropna().unique()):
                pct(df, e["Sex"] == lvl, c["Sex"] == lvl, "Sex", lvl)
        if "_age_band" in df.columns:
            for lvl in self.AGE_LABELS:
                pct(df, e["_age_band"] == lvl, c["_age_band"] == lvl, "Age group", lvl)
        if "_bmi_band" in df.columns:
            for lvl in self.BMI_LABELS:
                pct(df, e["_bmi_band"] == lvl, c["_bmi_band"] == lvl, "BMI (WHO)", lvl)
        for col in sorted(x for x in df.columns if x.startswith("edu_")):
            pct(df, e[col] == 1, c[col] == 1, "Education", col.replace("edu_", "level "))
        for col in sorted(x for x in df.columns if x.startswith("pre_")):
            pct(df, e[col] == 1, c[col] == 1, "Comorbidity", col.replace("pre_", ""))
        if "APOE" in df.columns:
            for lvl in sorted(df["APOE"].dropna().unique()):
                pct(df, e["APOE"] == lvl, c["APOE"] == lvl, "APOE", lvl)

        tbl = pd.DataFrame(rows, columns=["variable", "level", "exposed_%", "control_%"])
        if verbose:
            print(f"Strata composition  (exposed n={len(e):,}, control n={len(c):,})\n")
            print(tbl.to_string(index=False))
        self.strata_ = tbl
        return tbl

    # --------------------------------------
#    plot the SMD

    def plot_smd(self, threshold=0.1, save_path=None):
        """Love plot: |SMD| per covariate, before vs after matching."""
        import matplotlib.pyplot as plt
        import matplotlib as mpl
        if self.matched_cohort is None:
            raise RuntimeError("Run match() first.")
        tbl = self.balance_table(verbose=False)

        mpl.rcParams.update({"font.family": "sans-serif",
                             "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                             "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
        c_pre, c_post = "#B0B0B0", "#C44E52"

        d = tbl.copy()
        if d["SMD_pre"].isna().any() or d["SMD_post"].isna().any():
            warnings.warn(f"{int(d['SMD_post'].isna().sum())} covariate(s) have no "
                          f"computable SMD and are plotted at zero with a marker.")
        d["abs_pre"] = d["SMD_pre"].abs().fillna(0.0)
        d["abs_post"] = d["SMD_post"].abs().fillna(0.0)
        d = d.sort_values("abs_pre").reset_index(drop=True)
        y = np.arange(len(d))

        fig, ax = plt.subplots(figsize=(6.2, 0.30 * len(d) + 1.4))
        ax.axvline(threshold, color="0.35", ls="--", lw=0.9)
        for x0, x1, yy in zip(d["abs_pre"], d["abs_post"], y):
            ax.plot([x0, x1], [yy, yy], color="0.80", lw=1.0, zorder=1)
        ax.scatter(d["abs_pre"], y, s=34, color=c_pre, zorder=2, label="Before matching")
        ax.scatter(d["abs_post"], y, s=34, color=c_post, zorder=3, label="After matching")

        unknown = d["SMD_post"].isna().to_numpy()
        if unknown.any():
            ax.scatter(d.loc[unknown, "abs_post"], y[unknown], s=90, facecolors="none",
                       edgecolors="0.2", linewidths=1.1, zorder=4, label="not assessable")
        ax.set_yticks(y)
        ax.set_yticklabels([f"{n} (n/a)" if u else n
                            for n, u in zip(d["covariate"], unknown)])
        ax.set_xlabel("Absolute standardized mean difference")
        ax.set_title("Covariate balance before and after matching", fontsize=10, pad=8)
        ax.set_xlim(left=0)
        ax.tick_params(length=0)
        ax.legend(frameon=False, fontsize=8, loc="lower right")
        ax.text(threshold, -0.8, f" {threshold}", fontsize=7, color="0.35", va="top")
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, bbox_inches="tight")
            print("saved", save_path)
        return fig
