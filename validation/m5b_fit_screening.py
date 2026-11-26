"""Check whether rater fit flags catch false annual contrasts in existing scenarios."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import log_softmax, softmax

import mfrmpy
from mfrm_app.mml_joint import normal_rule
from validation.m5b_annual_drift import NEW_COHORT_MEAN, RATERS, SEVERITY, STEPS, rating_draws
from validation.m5b_category_drift import NODES, cases_at


CONDITIONS = ("common", "four")
N_REPLICATES = 500
STATISTICS = ("Infit", "Outfit", "DF_Infit", "DF_Outfit", "InfitZSTD", "OutfitZSTD")
FIELDS = ("replicate", "condition", "severity_estimate", "FitStatus", *STATISTICS,
          *(f"known_{name}" for name in STATISTICS))


def fit_statistics(squared_residual, variance, fourth):
    # Unit-weight formulas, with variances and fourth moments supplied per rating.
    infit = squared_residual.sum()/variance.sum()
    outfit = np.mean(squared_residual/variance)
    df_in = 2*variance.sum()**2/np.sum(fourth-variance**2)
    df_out = 2*len(variance)**2/np.sum(fourth/variance**2-1)
    z = []
    for value, df in ((infit, df_in), (outfit, df_out)):
        standardized = (value**(1/3)-1+2/(9*df))/np.sqrt(2/(9*df)) if np.isfinite(df) and df >= 1 else np.nan
        z.append(np.clip(standardized, -9, 9))
    return [infit, outfit, df_in, df_out, *z]


def known_statistics(observed, location):
    categories = np.arange(5)
    probabilities = softmax(location[:, None]*categories - STEPS, axis=1)
    expected = probabilities @ categories
    deviation = categories - expected[:, None]
    variance = (probabilities*deviation**2).sum(axis=1)
    fourth = (probabilities*deviation**4).sum(axis=1)
    return fit_statistics((observed-expected)**2, variance, fourth)


def known_fit(rows, replicate):
    ability = rating_draws(replicate)[1]
    work_year = rows.WorkYear.eq("B").to_numpy(dtype=int)
    work_index = rows.Recording.str[1:].to_numpy(dtype=int)
    location = ability[work_year, work_index] + NEW_COHORT_MEAN*work_year - SEVERITY[RATERS.index("R3")]
    return dict(zip((f"known_{name}" for name in STATISTICS),
                    known_statistics(rows.Score.to_numpy()-1, location), strict=True))


def known_calibration_eap(rows):
    """Hold the generating RSM parameters fixed and estimate only work locations."""
    nodes, weights = normal_rule(len(NODES))
    mean = NEW_COHORT_MEAN*rows.WorkYear.eq("B").to_numpy()
    severity = SEVERITY[[RATERS.index(name) for name in rows.Rater]]
    location = nodes[None,:] + (mean-severity)[:,None]
    logp = log_softmax(location[:,:,None]*np.arange(5)-STEPS, axis=2)
    observed = rows.Score.to_numpy()-1
    person, work = pd.factorize(rows.Recording)
    log_posterior = np.tile(np.log(weights), (len(work), 1))
    np.add.at(log_posterior, person, logp[np.arange(len(rows)), :, observed])
    theta = (softmax(log_posterior, axis=1) @ nodes)[person] + mean
    target = rows.RaterYear.eq("B-R3").to_numpy()
    return known_statistics(observed[target], theta[target]-severity[target])


def evaluate(replicate):
    out = []
    for condition, rows in cases_at(replicate):
        if condition not in CONDITIONS:
            continue
        fit = mfrmpy.analyze(rows, person="Recording", facets=["WorkYear", "RaterYear"], score="Score",
            model="RSM", method="MML", noncenter_facet="RaterYear", positive_facets=["WorkYear"],
            rating_min=1, rating_max=5, quad_points=len(NODES), reltol=1e-12)
        measures = fit["diagnostics"]["measures"]
        r3 = measures.loc[measures.Facet.eq("RaterYear")].set_index("Level").loc[["A-R3", "B-R3"]]
        target = r3.loc["B-R3"]
        obs = fit["diagnostics"]["obs"]
        observed = obs.loc[obs.RaterYear.eq("B-R3")]
        assert len(observed) == 40 and fit["diagnostics"]["fit_df_method"] == "facets"
        manual = fit_statistics(observed.Residual.to_numpy()**2, observed.Var.to_numpy(),
                                observed.FourthCentralMoment.to_numpy())
        values = pd.to_numeric(target[list(STATISTICS)], errors="coerce")
        np.testing.assert_allclose(values, manual, atol=1e-12, rtol=1e-12)
        out.append(dict(replicate=replicate, condition=condition, FitStatus=target.FitStatus,
                        severity_estimate=r3.Estimate.astype(float).diff().iloc[-1],
                        **values.to_dict(),
                        **(known_fit(rows.loc[rows.RaterYear.eq("B-R3")], replicate) if condition == "common" else {})))
    return out


def run(path, end, workers):
    done = set()
    if path.exists():
        saved = pd.read_csv(path)
        assert tuple(saved.columns) == FIELDS and saved.condition.isin(CONDITIONS).all()
        assert not saved.duplicated(["replicate", "condition"]).any()
        assert saved.groupby("replicate").size().eq(len(CONDITIONS)).all()
        done.update(saved.replicate)
    pending = sorted(set(range(1, end+1))-done)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for count, rows in enumerate(pool.map(evaluate, pending), 1):
            pd.DataFrame(rows).reindex(columns=FIELDS).to_csv(path, mode="a", header=not path.exists(), index=False)
            if count % 25 == 0 or count == len(pending):
                print(f"{count}/{len(pending)} replicate sets", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--end", type=int, default=N_REPLICATES)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if not 1 <= args.end <= N_REPLICATES:
        parser.error(f"replicate count must be between 1 and {N_REPLICATES}")
    run(args.path, args.end, args.workers)
