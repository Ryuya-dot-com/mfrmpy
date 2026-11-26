"""Audit mfrmr's independent-anchor intervals using analytic Python information."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from mfrm_app.engine import mfrm_estimate
from mfrm_app.mml_joint import GroupedMmlLikelihood, normal_rule


LEVELS = {"Item": ["I1", "I2", "I3", "I4"], "Rater": ["R1", "R2", "R3"]}


def audit(rows, result):
    likelihood = GroupedMmlLikelihood(rows, dict(
        model="RSM", facet_names=list(LEVELS), facet_levels=LEVELS, noncenter_facet="Rater",
        facet_signs={"Item": -1, "Rater": -1}, n_cat=4, rating_min=0,
    ))
    # Absorb the receiving population mean into the rater locations in the
    # fixed-N(0,1) kernel, retaining the two anchor values as input variables.
    point = np.r_[result.I1, result.I2, result.I3,
                  np.array([result.a1, result.a2, result.r3]) - result.mean,
                  result.step1, result.step2, 0.]
    unit = np.eye(len(point))
    free = np.c_[unit[:, [0, 1, 2, 5, 6, 7]], -unit[:, 3:6].sum(axis=1)]
    anchored = unit[:, 3:5]
    score = likelihood.evaluate(point, *normal_rule(31), information=True, moments=False)
    information = score["information"]
    h = free.T @ information @ free
    weight = np.eye(free.shape[1])[3]
    projected = np.linalg.solve(h, weight)
    jac = np.array([-1., 0.]) - projected @ (free.T @ information @ anchored)
    v = np.array([[result.v11, result.v12], [result.v12, result.v22]])
    conditional = np.sqrt(weight @ projected)
    anchor = np.sqrt(jac @ v @ jac)
    propagated = np.hypot(conditional, anchor)
    np.testing.assert_allclose(jac, [result.j1, result.j2], atol=3e-6, rtol=0)
    np.testing.assert_allclose([conditional, anchor, propagated],
                               [result.conditional_se, result.anchor_se, result.propagated_se], atol=3e-6, rtol=0)
    np.testing.assert_allclose(jac.sum(), 0, atol=1e-12, rtol=0)
    assert result.zero_covariance_checked
    return dict(analytic_j1=jac[0], analytic_j2=jac[1], analytic_conditional_se=conditional,
                analytic_anchor_se=anchor, analytic_propagated_se=propagated,
                analytic_diagonal_se=np.hypot(conditional, np.sqrt(jac @ np.diag(np.diag(v)) @ jac)),
                analytic_score_max=np.max(np.abs(free.T @ score["gradient"])))


def engine_estimate(rows, result):
    anchors = [dict(Facet="Rater", Level=level, Anchor=value)
               for level, value in zip(LEVELS["Rater"][:2], (result.a1, result.a2), strict=True)]
    fit = mfrm_estimate(rows, person_col="Person", facet_cols=list(LEVELS), score_col="Score", model="RSM",
                        method="MML", rating_min=0, rating_max=3, quad_points=31, reltol=1e-12,
                        mml_engine="direct", noncenter_facet="Rater", anchor_df=pd.DataFrame(anchors),
                        population_formula="~ 1", population_prior_sd=1., anchor_policy="silent")
    values = dict(zip(fit["config"]["facet_levels"]["Rater"], fit["params"]["facets"]["Rater"], strict=True))
    assert fit["summary"]["Converged"].iat[0]
    np.testing.assert_allclose([values["R1"], values["R2"]], [result.a1, result.a2], atol=1e-12, rtol=0)
    value = values["R3"] - values["R1"]
    np.testing.assert_allclose(value, result.estimate, atol=5e-5, rtol=0)
    return value


def run(ratings, paths):
    results = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    assert not results.duplicated(["replicate", "design"]).any()
    data = pd.read_csv(ratings).groupby(["replicate", "design"])
    critical = norm.ppf(.975)
    for row in results.itertuples():
        if not np.isfinite(row.propagated_se):
            assert isinstance(row.reason, str) and row.reason
            continue
        ratings_at = data.get_group((row.replicate, row.design))
        details = audit(ratings_at, row)
        se = details["analytic_propagated_se"]
        np.testing.assert_allclose(row.estimate + np.array([-1, 1]) * critical * se,
                                   [row.propagated_lower, row.propagated_upper], atol=6e-6, rtol=0)
        if row.replicate <= 6:
            details["python_engine_estimate"] = engine_estimate(ratings_at, row)
        for name, value in details.items():
            results.loc[row.Index, name] = value
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ratings", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("intervals", nargs="+", type=Path)
    args = parser.parse_args()
    run(args.ratings, args.intervals).to_csv(args.out, index=False,
        compression={"method": "gzip", "mtime": 0} if args.out.suffix == ".gz" else None)
