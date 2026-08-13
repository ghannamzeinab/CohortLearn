"""CohortLearn: matched cohort construction from electronic health records.

Builds exposed and unexposed cohorts under the target trial emulation framework,
balances user-specified confounders by propensity score matching, and fits the
outcome model.

    from cohortlearn import CohortBuilder, PSMCalculator, SurvivalAnalyser
"""

from .cohort import CohortBuilder, CONSORTTracker, rename_id
from .matching import PSMCalculator
from .survival import SurvivalAnalyser, full_rank_covariates

__version__ = "0.1.0"

__all__ = [
    "CohortBuilder",
    "CONSORTTracker",
    "rename_id",
    "PSMCalculator",
    "SurvivalAnalyser",
    "full_rank_covariates",
]
