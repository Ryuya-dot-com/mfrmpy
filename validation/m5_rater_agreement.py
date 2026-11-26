"""Compare rater agreement, severity separation and person accuracy in paired examples."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import softmax

import mfrmpy
from mfrm_app.mml_joint import normal_rule
from mfrm_app.reliability import calculate_reliability_series
from validation.m5b_annual_drift import RATERS, SEVERITY, STEPS, scores


REPLICATES = 100
N_WORKS = 60
SHARED_PROPORTION = .85
CONDITIONS = ("different_severity", "same_severity", "shared_judgment")
WORK_IDS = [f"W{k:03d}" for k in range(N_WORKS)]


def cases_at(replicate):
    rng = np.random.default_rng(np.random.SeedSequence([2026112119, replicate]))
    ability = rng.normal(size=N_WORKS)
    uniforms = rng.random((N_WORKS, len(RATERS)))
    common = rng.random((N_WORKS, 1))
    is_shared = rng.random((N_WORKS, 1)) < SHARED_PROPORTION
    equal = np.zeros(len(RATERS))
    for condition, severity, draws, share in (
        (CONDITIONS[0], SEVERITY, uniforms, 0.),
        (CONDITIONS[1], equal, uniforms, 0.),
        (CONDITIONS[2], equal, np.where(is_shared, common, uniforms), SHARED_PROPORTION),
    ):
        yield condition, ability, severity, share, scores(ability, draws, severity)


def generating_posterior(observed, severity, share):
    """Reference with the generating calibration and the joint response law."""
    nodes, weights = normal_rule(61)
    categories = np.arange(5)
    probabilities = softmax((nodes[:, None, None] - severity[None, :, None])*categories-STEPS, axis=-1)
    response = probabilities[:, np.arange(len(RATERS)), observed-1]
    independent = response.prod(axis=2)
    equal = np.all(observed == observed[:, :1], axis=1)
    likelihood = (1-share)*independent + share*response[:, :, 0]*equal
    mass = weights[:, None]*likelihood
    posterior = mass/mass.sum(axis=0)
    mean = nodes @ posterior
    variance = nodes**2 @ posterior - mean**2
    return mean, variance


def fit_panel(observed):
    rows = pd.DataFrame(dict(Recording=np.repeat(WORK_IDS, len(RATERS)), Rater=np.tile(RATERS, N_WORKS),
                             Score=observed.ravel()))
    return mfrmpy.analyze(rows, person="Recording", score="Score", facets=["Rater"],
                          model="RSM", method="MML", rating_min=1, rating_max=5,
                          quad_points=61, reltol=1e-12)


def evaluate(replicate):
    output = []
    for condition, ability, severity, share, observed in cases_at(replicate):
        analysis = fit_panel(observed)
        diagnostics = analysis["diagnostics"]
        measures = diagnostics["measures"]
        reliability = calculate_reliability_series(measures, analysis["result"], obs_df=diagnostics["obs"],
                                                    saved_reliability=diagnostics["reliability"])
        model_sample = reliability.loc[reliability.SEMode.eq("model") & reliability.DistributionBasis.eq("sample")].set_index("Facet")
        rater = model_sample.loc["Rater"]
        person = model_sample.loc["Person"]
        assert person.Use == "screening_only", person.StatusReason
        assert rater.Use == "screening_only" or rater.StatusReason == "stat.reliability.zero_observed_variance"
        person_measures = measures.loc[measures.Facet.eq("Person")].set_index("Level").loc[WORK_IDS]
        rats = measures.loc[measures.Facet.eq("Rater")]
        agree, = diagnostics["interrater_agreement"]["summary"].itertuples(index=False)
        exact = np.mean(observed[:, [0, 0, 1]] == observed[:, [1, 2, 2]])
        assert agree.TotalPairs == N_WORKS*3 and agree.ExactAgreement == exact
        assert np.isclose(person.RMSE, np.sqrt(np.mean(person_measures.SE.astype(float)**2)))
        for facet in (rater, person):
            if facet.Use == "screening_only":
                assert np.isclose(facet.Strata, (4*facet.Separation+1)/3)
        correct_eap, correct_var = generating_posterior(observed, severity, share)
        output.append(dict(
            replicate=replicate, condition=condition, exact_agreement=exact,
            expected_agreement=agree.ExpectedExactAgreement, mean_rater_infit=float(rats.Infit.mean()),
            rater_sd=rater.SD, rater_rms_se=rater.RMSE, rater_separation=rater.Separation,
            rater_strata=rater.Strata, rater_reliability=rater.Reliability, rater_status=rater.StatusReason,
            person_separation=person.Separation, person_strata=person.Strata,
            person_reliability=person.Reliability, person_mean_posterior_variance=person.RMSE**2,
            person_mean_squared_error=float(np.mean((person_measures.Estimate.astype(float).to_numpy()-ability)**2)),
            reference_mean_posterior_variance=float(correct_var.mean()),
            reference_mean_squared_error=float(np.mean((correct_eap-ability)**2)),
        ))
    return output


def run(path, end, workers):
    done = set()
    if path.exists():
        saved = pd.read_csv(path)
        assert not saved.duplicated(["replicate", "condition"]).any()
        assert saved.condition.isin(CONDITIONS).all() and saved.groupby("replicate").size().eq(len(CONDITIONS)).all()
        done.update(saved.replicate)
    pending = sorted(set(range(1, end+1))-done)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for count, rows in enumerate(pool.map(evaluate, pending), 1):
            pd.DataFrame(rows).to_csv(path, mode="a", header=not path.exists(), index=False)
            if count % 20 == 0 or count == len(pending):
                print(f"{count}/{len(pending)} replicate sets", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--end", type=int, default=REPLICATES)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if not 1 <= args.end <= REPLICATES:
        parser.error(f"replicate count must be between 1 and {REPLICATES}")
    run(args.path, args.end, args.workers)
