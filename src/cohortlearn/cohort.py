"""Cohort construction under target trial emulation.

Time zero is the first exposure code for the exposed arm, and a risk-set
sampled date for the unexposed arm. Everything used to decide eligibility,
arm and time zero comes from records dated at or before that instant.
"""

import ast
import warnings
from typing import Optional, List, Dict, Tuple

import numpy as np
import pandas as pd

# Column names that different datasets use for the person key.
ID_ALIASES = ("id", "eid", "person_id", "PersonId", "IID", "Id", "ID")


def rename_id(df: pd.DataFrame) -> pd.DataFrame:
    """Rename whatever person-key the source uses to a single 'id'."""
    for alias in ID_ALIASES:
        if alias in df.columns:
            return df.rename(columns={alias: "id"}) if alias != "id" else df
    raise KeyError(f"No id-like column found. Looked for {ID_ALIASES}.")


def to_datetime_safe(values, field):
    """Parse a date column without pandas' epoch-nanoseconds fallback.

    pd.to_datetime(20100101) silently yields 1970-01-01, so an integer YYYYMMDD
    column would corrupt every date. Numeric input is read as YYYYMMDD instead,
    timezone-aware input is flattened, and anything unreadable raises.
    """
    s = pd.Series(values).copy()
    if pd.api.types.is_numeric_dtype(s):
        digits = s.dropna().astype("int64").astype(str)
        out = pd.to_datetime(digits.reindex(s.index), format="%Y%m%d", errors="coerce")
    else:
        out = pd.to_datetime(s, errors="coerce")
    if getattr(out.dtype, "tz", None) is not None:
        out = out.dt.tz_localize(None)
    unreadable = out.isna() & s.notna()
    if unreadable.any():
        raise ValueError(
            f"{int(unreadable.sum()):,} unparseable value(s) in '{field}', "
            f"e.g. {s[unreadable].iloc[0]!r}. Supply ISO dates or YYYYMMDD.")
    return out


class CONSORTTracker:
    """Keeps the participant-flow counts so the CONSORT diagram writes itself."""

    def __init__(self):
        self._data = {}
        self._order = []

    def log(self, key, value):
        if key not in self._data:
            self._order.append(key)
        self._data[key] = value

    def get(self, key, default=0):
        return self._data.get(key, default)

    def summary(self):
        for key in self._order:
            label = key.replace("_", " ").capitalize()
            print(f"  {label:<58} {self._data[key]:>10,}")


class CohortBuilder:
    """Build exposed and unexposed cohorts from harmonised biobank data.

    The unexposed arm is drawn by risk-set (incidence-density) sampling:
    a person is eligible to be a control at a given case's time zero only
    if, on that date, they are born, alive, in the observation window,
    free of the outcome, and not yet exposed. People who become exposed
    later still contribute this earlier unexposed time and are flagged so
    the survival stage can censor them at exposure onset.
    """

    EDUCATION_CODES = [-1, 0, 1, 2, 3]  # fallback if auto-discovery finds nothing

    def __init__(self,
                 exposure_codes: List[str],
                 icd10_path: Optional[str] = None,
                 demographics_path: Optional[str] = None,
                 bmi_path: Optional[str] = None,
                 risk_allele_path: Optional[str] = None,
                 risk_allele_col: Optional[str] = None,
                 outcome_codes: Optional[List[str]] = None,
                 confounder_codes: Optional[Dict[str, List[str]]] = None,
                 demographic_confounders: Optional[Dict[str, bool]] = None,
                 education_codes: Optional[List[int]] = None,
                 age_range: Optional[Tuple[float, float]] = None,
                 washout_months: int = 12,
                 require_observation: bool = True):
        """
        exposure_codes / outcome_codes : ICD-10 prefixes, e.g. ['F32','F33'] / ['G30'].
        confounder_codes : {name: [prefixes]} -> one binary pre_<name> per group.
        demographic_confounders : which of age, sex, bmi, risk_allele, education to use.
            'age' and 'sex' are always required.
        washout_months : drop anyone whose outcome lands in (time_zero, time_zero+N months].
        require_observation : only place a pseudo-index inside a control's own
            record span (first..last coded date) so time zero has data coverage.
        """
        self.icd10_path = icd10_path
        self.demographics_path = demographics_path
        self.bmi_path = bmi_path
        self.risk_allele_path = risk_allele_path
        self.risk_allele_col = risk_allele_col

        self.exposure_codes = exposure_codes
        self.outcome_codes = outcome_codes or []
        self.confounder_codes = confounder_codes or {}
        self.demographic_confounders = demographic_confounders or {}
        self.education_codes = education_codes

        # Age and sex are needed for the balance table and eligibility; enforce them.
        if not self.demographic_confounders.get("sex", True):
            raise ValueError("'sex' cannot be disabled.")
        if not self.demographic_confounders.get("age", True):
            raise ValueError("'age' cannot be disabled.")
        self.demographic_confounders.setdefault("sex", True)
        self.demographic_confounders.setdefault("age", True)

        if age_range is None:
            warnings.warn("No age_range set; using (0,120). Adults: (18,110).", UserWarning, stacklevel=2)
        self.age_range = age_range or (0, 120)
        self.washout_months = washout_months
        self.require_observation = require_observation

        self.icd10 = self.demographics = self.bmi = self.risk_allele = None
        self.icd10_long = None
        self.consort = CONSORTTracker()
        self.cases = self.controls = self.master = None

    # ---------- IO ----------

    def _read_any(self, path):
        return pd.read_parquet(path) if str(path).endswith(".parquet") else pd.read_csv(path, low_memory=False)

    def _harmonise_long_icd(self, df):
        """Normalise the long table once: id, upper-cased code, real datetimes."""
        out = rename_id(df.copy())
        out = out[["id", "code", "date"]].copy()
        out["code"] = out["code"].astype("string").str.strip().str.upper()
        out["date"] = to_datetime_safe(out["date"], "date")
        n0 = len(out)
        out = out.dropna(subset=["id", "code", "date"])
        out = out[out["code"] != ""]
        if n0 - len(out):
            warnings.warn(f"dropped {n0 - len(out):,} diagnosis rows with a missing "
                            f"id, code or date.", UserWarning, stacklevel=3)
        return out.drop_duplicates(["id", "code", "date"]).reset_index(drop=True)

    def _harmonise_demographics(self):
        """Parse the one date column demographics may carry."""
        if "Date of Death" in self.demographics.columns:
            self.demographics["Date of Death"] = to_datetime_safe(
                self.demographics["Date of Death"], "Date of Death")

    def _prepare_long_icd(self, df):
        """Turn a wide UKB table (diagnosis_* / datediagnosis_*) into long [id, code, date]."""
        df = rename_id(df)
        diag_cols = [c for c in df.columns
                     if "diagnosis" in c.lower() and c != "id" and not c.lower().startswith("date")]
        pairs, unpaired = {}, []
        for d in diag_cols:
            t = "date" + d.replace("diagnosis", "", 1)
            (pairs.__setitem__(d, t) if t in df.columns else unpaired.append(d))
        if unpaired:
            warnings.warn(f"{len(unpaired)} diagnosis cols had no date partner and were skipped.",
                          UserWarning, stacklevel=2)
        if not pairs:
            return pd.DataFrame(columns=["id", "code", "date"])
        chunks = []
        for d, t in pairs.items():
            sub = df[["id", d, t]].dropna(subset=[d]).rename(columns={d: "code", t: "date"})
            chunks.append(sub)
        out = pd.concat(chunks, ignore_index=True)
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        out["code"] = out["code"].astype(str).str.strip().str.upper()
        out = out.dropna(subset=["date", "code"])
        out = out[out["code"] != "NAN"].drop_duplicates(["id", "code", "date"])
        return out.sort_values(["id", "date"]).reset_index(drop=True)

    def load_data(self):
        """Read every file from disk, harmonise, and validate."""
        print("Loading data...")
        icd10 = rename_id(self._read_any(self.icd10_path))
        self.demographics = rename_id(self._read_any(self.demographics_path))
        self.bmi = rename_id(self._read_any(self.bmi_path))

        if "Year of Birth" in self.demographics.columns and "YOB" not in self.demographics.columns:
            self.demographics = self.demographics.rename(columns={"Year of Birth": "YOB"})

        self.education_codes = self._resolve_education_codes()
        print(f"Education codes: {self.education_codes}")

        if self.risk_allele_path is not None:
            self.risk_allele = rename_id(self._read_any(self.risk_allele_path))
            self._check_risk_allele_col()

        self._harmonise_demographics()
        if {"code", "date"}.issubset(icd10.columns):
            self.icd10_long = self._harmonise_long_icd(icd10)
        else:
            self.icd10_long = self._prepare_long_icd(icd10)
        print(f"Long ICD rows: {len(self.icd10_long):,} | ids: {self.icd10_long['id'].nunique():,}")
        self.validate_data()

    def attach_dataframes(self, icd10_long, demographics, bmi, risk_allele=None):
        """Inject already-loaded frames instead of reading from disk."""
        self.icd10_long = self._harmonise_long_icd(icd10_long) # Renaming of aliases handled in _harmonise_long_icd alongside other harmonisation steps
        self.demographics = rename_id(demographics.copy())
        self.bmi = rename_id(bmi.copy())
        if "Year of Birth" in self.demographics.columns and "YOB" not in self.demographics.columns:
            self.demographics = self.demographics.rename(columns={"Year of Birth": "YOB"})
        self._harmonise_demographics()
        if risk_allele is not None:
            self.risk_allele = rename_id(risk_allele.copy())
            self._check_risk_allele_col()
        self.education_codes = self._resolve_education_codes()
        print(f"Education codes: {self.education_codes}")
        self.validate_data()

    def _check_risk_allele_col(self):
        if self.risk_allele_col is None:
            raise ValueError("risk_allele_col must be given when a risk-allele frame is provided.")
        if self.risk_allele_col not in self.risk_allele.columns:
            raise ValueError(f"'{self.risk_allele_col}' not in risk-allele frame.")

    def _resolve_education_codes(self):
        if self.education_codes is not None:
            return self.education_codes
        if "Education" in self.demographics.columns:
            return self._discover_education_codes()
        return list(self.EDUCATION_CODES)

    def validate_data(self):
        assert {"id", "code", "date"}.issubset(self.icd10_long.columns), "ICD10 needs id, code, date"
        assert len(self.icd10_long) > 0, "ICD10 is empty"
        for col in ("id", "Sex", "YOB"):
            assert col in self.demographics.columns, f"Demographics missing {col}"
        if self.demographics["id"].duplicated().any():
            raise ValueError("Demographics has duplicate ids.")
        if self.bmi["id"].duplicated().any():
            raise ValueError("BMI has duplicate ids; dedupe first.")
        if self.risk_allele is not None and self.risk_allele["id"].duplicated().any():
            raise ValueError("Risk allele has duplicate ids.")
        print("Validation passed.\n")

    # ---------- helpers ----------

    def search_long_icd(self, target_codes):
        """Prefix match against the long ICD table."""
        if not target_codes:
            return pd.DataFrame(columns=["id", "code", "date"])
        codes = tuple(c.upper() for c in target_codes)
        return self.icd10_long[self.icd10_long["code"].str.startswith(codes)].copy()

    def _first_date_per_id(self, codes, colname):
        hit = self.search_long_icd(codes)
        if hit.empty:
            return pd.DataFrame(columns=["id", colname])
        return (hit.sort_values(["id", "date"]).drop_duplicates("id")
                   .rename(columns={"date": colname})[["id", colname]])

    def _parse_edu_codes(self, val):
        if isinstance(val, (list, tuple)):
            return {int(v) for v in val}
        if pd.isna(val):
            return set()
        try:
            parsed = ast.literal_eval(str(val))
        except (ValueError, SyntaxError):
            return set()
        if isinstance(parsed, (list, tuple)):
            return {int(v) for v in parsed}
        try:
            return {int(parsed)}
        except (TypeError, ValueError):
            return set()

    def _discover_education_codes(self):
        codes = set()
        for val in self.demographics["Education"].dropna().unique():
            codes |= self._parse_edu_codes(val)
        return sorted(codes) if codes else list(self.EDUCATION_CODES)

    def _has_edu_code(self, val, code):
        return int(code in self._parse_edu_codes(val))

    def _dob_from_yob(self, series):
        """Birthdate proxy: 1 July of the birth year."""
        yob = pd.to_numeric(series, errors="coerce")
        return pd.to_datetime(pd.DataFrame({"year": yob, "month": 7, "day": 1}), errors="coerce")

    def compute_pre_features(self, id_anchor_df, confounder_codes):
        """Binary pre_<name> flags for comorbidities coded strictly before time zero."""
        anchor = id_anchor_df[["id", "index_date"]].copy()
        anchor["index_date"] = pd.to_datetime(anchor["index_date"])
        feat = anchor[["id"]].copy()
        if not confounder_codes:
            return feat
        all_codes = [c for codes in confounder_codes.values() for c in codes]
        rel = self.search_long_icd(all_codes)
        if rel.empty:
            for g in confounder_codes:
                feat[f"pre_{g}"] = 0
            return feat
        merged = rel.merge(anchor, on="id", how="inner")
        merged = merged[merged["date"] < merged["index_date"]]
        for g, codes in confounder_codes.items():
            m = merged["code"].str.startswith(tuple(c.upper() for c in codes))
            has = (merged[m].groupby("id")["code"].count().gt(0).astype(int)
                          .reset_index().rename(columns={"code": f"pre_{g}"}))
            feat = feat.merge(has, on="id", how="left")
            feat[f"pre_{g}"] = feat[f"pre_{g}"].fillna(0).astype(int)
        return feat

    def _add_demographic_features(self, cohort, preserve_cols=None):
        """Attach age-at-index, sex, BMI, APOE and one-hot education; keep survival fields."""
        keep = ["id", "index_date"] + list(preserve_cols or [])

        if "date_of_birth" not in cohort.columns:
            cohort["date_of_birth"] = self._dob_from_yob(cohort["YOB"])
        cohort["age_at_index"] = (cohort["index_date"] - cohort["date_of_birth"]).dt.days / 365.25
        keep += ["age_at_index", "Sex", "YOB", "date_of_birth"]

        if self.demographic_confounders.get("bmi", False):
            assert not self.bmi["id"].duplicated().any(), "BMI has duplicate ids."
            cohort = cohort.merge(self.bmi[["id", "BMI"]], on="id", how="left")
            keep.append("BMI")

        if self.demographic_confounders.get("risk_allele", False) and self.risk_allele is not None:
            ra = self.risk_allele[["id", self.risk_allele_col]].rename(columns={self.risk_allele_col: "APOE"})
            cohort = cohort.merge(ra, on="id", how="left")
            keep.append("APOE")

        if self.demographic_confounders.get("education", False) and "Education" in cohort.columns:
            for code in self.education_codes:
                col = f"edu_{code}"
                cohort[col] = cohort["Education"].apply(lambda v: self._has_edu_code(v, code))
                keep.append(col)

        if "Date of Death" in cohort.columns:
            keep.append("Date of Death")

        keep = [c for c in dict.fromkeys(keep) if c in cohort.columns]
        return cohort[keep]

    def _drop_missing_features(self, cohort, prefix=""):
        """Complete-case drop on the covariates that feed the propensity model."""
        req = ["age_at_index", "Sex"]
        if self.demographic_confounders.get("bmi", False):
            req.append("BMI")
        if self.demographic_confounders.get("risk_allele", False):
            req.append("APOE")
        counts = {}
        for col in req:
            if col in cohort.columns:
                n = cohort[col].isna().sum()
                if n:
                    counts[col] = int(n)
                    self.consort.log(f"{prefix}excluded_missing_{col}", int(n))
                    cohort = cohort[cohort[col].notna()]
        return cohort, counts

    # ---------- cohorts ----------

    def build_case_cohort(self):
        """Exposed arm: first exposure = time zero; drop prevalent/washout outcomes."""
        exposed = self.search_long_icd(self.exposure_codes)
        if exposed.empty:
            print("No exposed individuals.")
            return pd.DataFrame()

        idx = (exposed.sort_values(["id", "date"]).drop_duplicates("id")
                      .rename(columns={"date": "index_date", "code": "exposure_code"})
                      [["id", "index_date", "exposure_code"]])
        self.consort.log("exposed_individuals", len(idx))
        idx["first_outcome_date"] = pd.NaT

        if self.outcome_codes:
            first_out = self._first_date_per_id(self.outcome_codes, "outcome_cand")
            if not first_out.empty:
                idx = idx.merge(first_out, on="id", how="left")
                prevalent = idx["outcome_cand"].notna() & (idx["outcome_cand"] <= idx["index_date"])
                self.consort.log("excluded_prevalent_outcome", int(prevalent.sum()))
                idx = idx[~prevalent].copy()
                if self.washout_months > 0:
                    w_end = idx["index_date"] + pd.DateOffset(months=self.washout_months)
                    wash = (idx["outcome_cand"].notna() &
                            (idx["outcome_cand"] > idx["index_date"]) & (idx["outcome_cand"] <= w_end))
                    self.consort.log("excluded_washout_violation", int(wash.sum()))
                    idx = idx[~wash].copy()
                idx["first_outcome_date"] = idx["outcome_cand"]
                idx = idx.drop(columns=["outcome_cand"])

        self.consort.log("after_outcome_exclusions", len(idx))
        cohort = idx.merge(self.demographics, on="id", how="left")
        cohort = self._add_demographic_features(cohort, preserve_cols=["exposure_code", "first_outcome_date"])
        
        if "Date of Death" in cohort.columns:
            death = pd.to_datetime(cohort["Date of Death"])
            after_death = death.notna() & (death < cohort["index_date"])
            if after_death.any():
                self.consort.log("excluded_exposure_after_death", int(after_death.sum()))
                cohort = cohort[~after_death].copy()

        if "age_at_index" in cohort.columns:
            n0 = len(cohort)
            cohort = cohort[cohort["age_at_index"].between(*self.age_range)]
            if n0 - len(cohort):
                self.consort.log("excluded_age_out_of_range", n0 - len(cohort))

        if self.confounder_codes:
            pre = self.compute_pre_features(cohort[["id", "index_date"]], self.confounder_codes)
            cohort = cohort.merge(pre, on="id", how="left")
            pcols = [c for c in cohort.columns if c.startswith("pre_")]
            cohort[pcols] = cohort[pcols].fillna(0).astype(int)

        cohort, _ = self._drop_missing_features(cohort, prefix="")
        cohort["is_exposed"] = 1
        cohort["exposure_onset_date"] = pd.NaT          # exposed from time zero -> no exposure censoring
        cohort["index_year"] = cohort["index_date"].dt.year
        cohort["matched_case_id"] = np.nan

        self.consort.log("final_case_cohort", len(cohort))
        print(f"Case cohort: {len(cohort):,}")
        self.cases = cohort.reset_index(drop=True)
        return self.cases

    def build_control_cohort(self, case_cohort, seed=42):
        """Unexposed arm by risk-set sampling. Nobody is dropped for a future exposure;
        their pre-exposure time is used and their exposure onset is recorded for censoring."""
        universe = set(self.demographics["id"]) & set(self.icd10_long["id"])
        self.consort.log("controls_removed_no_icd10_records",
                         len(set(self.demographics["id"])) - len(universe))
        self.consort.log("control_universe", len(universe))

        controls = (self.demographics[self.demographics["id"].isin(universe)]
                    .drop_duplicates("id").copy())

        controls = self._assign_pseudo_dates(controls, case_cohort, seed=seed)
        controls = controls.dropna(subset=["index_date"])
        self.consort.log("after_pseudo_date_assignment", len(controls))
        if controls.empty:
            print("No controls after pseudo-index assignment.")
            return pd.DataFrame()

        if self.outcome_codes:
            first_out = self._first_date_per_id(self.outcome_codes, "first_outcome_date")
            if not first_out.empty:
                controls = controls.merge(first_out, on="id", how="left")
                prevalent = controls["first_outcome_date"].notna() & (controls["first_outcome_date"] <= controls["index_date"])
                self.consort.log("controls_removed_prevalent_outcome", int(prevalent.sum()))
                controls = controls[~prevalent].copy()
                if self.washout_months > 0:
                    w_end = controls["index_date"] + pd.DateOffset(months=self.washout_months)
                    wash = (controls["first_outcome_date"].notna() &
                            (controls["first_outcome_date"] > controls["index_date"]) &
                            (controls["first_outcome_date"] <= w_end))
                    self.consort.log("controls_removed_washout_violation", int(wash.sum()))
                    controls = controls[~wash].copy()
        self.consort.log("controls_after_outcome_exclusions", len(controls))
        if controls.empty:
            print("No controls after outcome exclusions.")
            return pd.DataFrame()
        if "first_outcome_date" not in controls.columns:
            controls["first_outcome_date"] = pd.NaT

        controls = self._add_demographic_features(
            controls, preserve_cols=["first_outcome_date", "exposure_onset_date"])

        if "age_at_index" in controls.columns:
            n0 = len(controls)
            controls = controls[controls["age_at_index"].between(*self.age_range)]
            if n0 - len(controls):
                self.consort.log("controls_excluded_age_out_of_range", n0 - len(controls))

        if self.confounder_codes:
            pre = self.compute_pre_features(controls[["id", "index_date"]], self.confounder_codes)
            controls = controls.merge(pre, on="id", how="left")
            pcols = [c for c in controls.columns if c.startswith("pre_")]
            controls[pcols] = controls[pcols].fillna(0).astype(int)

        controls, miss = self._drop_missing_features(controls, prefix="controls_")
        controls["is_exposed"] = 0
        controls["index_year"] = controls["index_date"].dt.year
        controls["matched_case_id"] = np.nan
        self.consort.log("final_control_pool_size", len(controls))

        print(f"Control pool: {len(controls):,}")
        if miss:
            print("  dropped for missingness:", miss)
        self.controls = controls.reset_index(drop=True)
        return self.controls

    def _assign_pseudo_dates(self, controls, case_cohort, seed=42):
        """Risk-set (incidence-density) pseudo-index.

        For each candidate we work out the window in which they are a legitimate
        control -- born, alive, in their record span and not yet exposed -- then
        draw a real case time zero that falls inside it. The window does not
        end at the outcome date, because that would use future information
        and shorten the eligible time of controls who later have the outcome.
        Controls with the outcome on or before the drawn date are removed
        afterwards as prevalent cases. Age and
        calendar era therefore come straight from the sampled date; no reference
        year is assumed anywhere.
        """
        need = {"Sex", "index_date"}
        if need - set(case_cohort.columns):
            raise ValueError(f"case_cohort missing {sorted(need - set(case_cohort.columns))}")
        if {"id", "Sex", "YOB"} - set(controls.columns):
            raise ValueError("controls need id, Sex, YOB.")

        rng = np.random.default_rng(seed)
        ctrl = controls.reset_index(drop=True).copy()
        ctrl["date_of_birth"] = self._dob_from_yob(ctrl["YOB"])

        dep = self._first_date_per_id(self.exposure_codes, "dep_onset")
        ctrl = ctrl.merge(dep, on="id", how="left")
        dem = self._first_date_per_id(self.outcome_codes, "dem_onset") if self.outcome_codes \
            else pd.DataFrame(columns=["id", "dem_onset"])
        ctrl = ctrl.merge(dem, on="id", how="left")

        if self.require_observation:
            span = (self.icd10_long.groupby("id")["date"].agg(["min", "max"])
                        .rename(columns={"min": "obs_start", "max": "obs_end"}).reset_index())
            ctrl = ctrl.merge(span, on="id", how="left")
        else:
            ctrl["obs_start"], ctrl["obs_end"] = pd.NaT, pd.NaT

        BIG = pd.Timestamp.max.normalize()
        one_day = pd.Timedelta(days=1)

        lo = ctrl["date_of_birth"]
        if self.require_observation:
            lo = pd.concat([lo, ctrl["obs_start"]], axis=1).max(axis=1)

        hi_parts = pd.DataFrame({
            "death": ctrl["Date of Death"] if "Date of Death" in ctrl.columns else pd.Series(pd.NaT, index=ctrl.index),
            "pre_dep": ctrl["dep_onset"] - one_day,          # unexposed strictly before onset
            "obs_end": ctrl["obs_end"] if self.require_observation else pd.Series(pd.NaT, index=ctrl.index),
        }).apply(lambda s: pd.to_datetime(s).fillna(BIG))
        hi = hi_parts.min(axis=1)

        exposure_onset = ctrl["dep_onset"]                    # NaT unless they become exposed later
        valid = lo.notna() & (hi >= lo)

        cases = case_cohort[["Sex", "index_date"]].dropna().copy()
        all_dates = np.sort(pd.to_datetime(cases["index_date"]).values.astype("datetime64[ns]")).view("i8")
        by_sex = {s: np.sort(pd.to_datetime(g["index_date"]).values.astype("datetime64[ns]")).view("i8")
                  for s, g in cases.groupby("Sex")}

        lo_i = pd.to_datetime(lo).values.astype("datetime64[ns]").view("i8")
        hi_i = pd.to_datetime(hi).values.astype("datetime64[ns]").view("i8")
        chosen = np.zeros(len(ctrl), dtype="i8")
        assigned = np.zeros(len(ctrl), dtype=bool)
        sex_vals = ctrl["Sex"].values

        def _draw(arr, rows):
            """Pick one random case date inside each row's [lo, hi]; returns which rows succeeded."""
            if arr is None or len(arr) == 0 or len(rows) == 0:
                return np.array([], dtype=int)
            left = np.searchsorted(arr, lo_i[rows], side="left")
            right = np.searchsorted(arr, hi_i[rows], side="right")
            cnt = right - left
            ok = cnt > 0
            if ok.any():
                offset = (rng.random(int(ok.sum())) * cnt[ok]).astype(np.int64)
                pos = left[ok] + offset
                chosen[rows[ok]] = arr[pos]
                assigned[rows[ok]] = True
            return rows[~ok]

        for s in pd.unique(sex_vals):
            rows = np.where((sex_vals == s) & valid.values & ~assigned)[0]
            leftover = _draw(by_sex.get(s), rows)          # same-sex first
            _draw(all_dates, leftover)                     # fall back to all cases
        
        self.consort.log("controls_no_eligible_window", int((~valid.values).sum()))
        self.consort.log("controls_no_valid_riskset_date", int((valid.values & ~assigned).sum()))

        ctrl["index_date"] = pd.NaT
        ctrl.loc[assigned, "index_date"] = chosen[assigned].view("datetime64[ns]")
        ctrl["exposure_onset_date"] = pd.to_datetime(exposure_onset)
        ctrl = ctrl.drop(columns=["dep_onset", "dem_onset", "obs_start", "obs_end"], errors="ignore")
        return ctrl

    def lookback_report(self, case_cohort, control_pool, tolerance_years=1.0):
        """Compare how much record history each arm has before time zero."""
        span = (self.icd10_long.groupby("id")["date"].min()
                    .rename("obs_start").reset_index())
        means = {}
        for name, arm in [("exposed", case_cohort), ("unexposed", control_pool)]:
            merged = arm[["id", "index_date"]].merge(span, on="id", how="left")
            years = (merged["index_date"] - merged["obs_start"]).dt.days / 365.25
            means[name] = years.mean()
            print(f"  {name:<10} look-back years  mean {years.mean():6.2f}  "
                  f"median {years.median():6.2f}  10th pct {years.quantile(.10):6.2f}")
        gap = means["exposed"] - means["unexposed"]
        print(f"  difference in mean look-back: {gap:+.2f} years")
        if abs(gap) > tolerance_years:
            warnings.warn(
                f"Mean look-back differs by {gap:+.2f} years between arms. Every "
                f"pre_<name> confounder is measured over a different window, so "
                f"the arms are differentially ascertained and matching cannot "
                f"correct it. Consider requiring a common minimum look-back.",
                UserWarning, stacklevel=2)
        self.lookback_ = means
        return means
    
    def build_master_df(self, case_cohort, control_pool):
        """Stack the two arms into one model-ready frame."""
        if case_cohort is None or case_cohort.empty:
            raise ValueError("Empty case cohort.")
        if control_pool is None or control_pool.empty:
            raise ValueError("Empty control pool.")

        master = pd.concat([case_cohort, control_pool], ignore_index=True)
        ind = [c for c in master.columns if c.startswith("pre_") or c.startswith("edu_")]
        if ind:
            master[ind] = master[ind].fillna(0).astype(int)

        self.consort.log("master_df_total", len(master))
        self.consort.log("master_df_exposed", int((master["is_exposed"] == 1).sum()))
        self.consort.log("master_df_unexposed", int((master["is_exposed"] == 0).sum()))
        print(f"Master: {len(master):,} "
              f"({int((master['is_exposed']==1).sum()):,} exposed / "
              f"{int((master['is_exposed']==0).sum()):,} unexposed)")
        self.master = master.reset_index(drop=True)
        return self.master

    def add_followup_columns(self, df, study_end=None):
        """Add survival fields with the correct target-trial censoring.

        A control who becomes exposed is censored at exposure onset, so only their
        genuinely-unexposed time counts. Death, the study end, and (for controls)
        exposure onset all compete with the outcome; the earliest wins.
        """
        out = df.copy()
        out["index_date"] = pd.to_datetime(out["index_date"])
        out["first_outcome_date"] = pd.to_datetime(out.get("first_outcome_date"))
        if "exposure_onset_date" not in out.columns:
            out["exposure_onset_date"] = pd.NaT
        out["exposure_onset_date"] = pd.to_datetime(out["exposure_onset_date"])
        death = pd.to_datetime(out["Date of Death"]) if "Date of Death" in out.columns \
            else pd.Series(pd.NaT, index=out.index)
        end = pd.to_datetime(study_end) if study_end is not None \
            else pd.to_datetime(out["first_outcome_date"]).max()

        BIG = pd.Timestamp.max.normalize()
        censor = pd.concat([out["exposure_onset_date"], death,
                            pd.Series(end, index=out.index)], axis=1).apply(
            lambda s: s.fillna(BIG)).min(axis=1)

        ev = out["first_outcome_date"]
        is_event = ev.notna() & (ev > out["index_date"]) & (ev <= censor)
        end_date = ev.where(is_event, censor)
        out["event"] = is_event.astype(int)
        out["time_years"] = (end_date - out["index_date"]).dt.days / 365.25
        out = out[out["time_years"] > 0]
        return out
