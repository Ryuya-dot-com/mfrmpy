"""Compare eRm to the saved CML fits and verify finite-fit holds on the original data."""
import sys

import pandas as pd

from mfrm_app.cmle import prepare_cmle_design
from mfrm_app.cmle_existence import audit_cmle_finite_mle
from validation.m2_sparse_intervals import DESIGNS, REPLICATES, TRUTH, rate, simulated_ratings


def qualify(native_path, reference_path):
    raw = pd.read_csv(native_path)
    reference = pd.read_csv(reference_path)
    reference = reference.loc[reference.engine.eq("mfrmr CML") & reference.design.isin(DESIGNS)]
    keys = ["replicate", "design"]
    result = raw.merge(reference[keys+["estimate", "lower", "upper"]], on=keys,
                       validate="one_to_one", suffixes=("_erm", "_mfrmr"))
    assert len(result) == len(raw) == len(reference) == len(DESIGNS)*REPLICATES
    result["native_interval_available"] = result.lower_erm.notna() & result.upper_erm.notna()
    result["qualified_interval"] = result.native_interval_available & result.lower_mfrmr.notna()
    result["conditional_hold"] = ""
    for row in result.loc[~result.qualified_interval].itertuples():
        data = dict(simulated_ratings(row.replicate, DESIGNS))[row.design]
        design = prepare_cmle_design(data, person_col="Person", facet_cols=["Item", "Rater"],
                                     score_col="Score", rating_min=0, rating_max=3, model="RSM")
        audit = audit_cmle_finite_mle(design)
        summary = audit["summary"].iloc[0]
        result.loc[row.Index, "conditional_hold"] = summary.Status
        result.loc[row.Index, "conditional_rank"] = summary.PrefitRank
        if summary.Status == "boundary_no_finite_cmle":
            assert summary.EnumerationComplete and summary.LPComplete and summary.DirectionalObjectiveNonincreasing
            trace = audit["directional_trace"]["Objective"]
            result.loc[row.Index, "nll_drop_along_ray_at_32"] = trace.iloc[0] - trace.iloc[-1]
    return result


if __name__ == "__main__":
    results = qualify(sys.argv[1], sys.argv[2])
    results.to_csv(sys.argv[3], index=False)
    for design, group in results.groupby("design"):
        eligible = group.loc[group.qualified_interval]
        covered = eligible.lower_erm.le(TRUTH) & eligible.upper_erm.ge(TRUTH)
        reference_covered = eligible.lower_mfrmr.le(TRUTH) & eligible.upper_mfrmr.ge(TRUTH)
        print(design, "returned", len(eligible), "covered", rate(covered.sum(), len(eligible))[0],
              "paired coverage changes", (covered != reference_covered).sum())
