"""Vary generating facet spreads in M3's sparse rater-contrast design."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import cho_solve
from scipy.stats import norm

from mfrm_app.mml_joint import GroupedMmlLikelihood
from validation.m2_sparse_intervals import (
    ITEM_DIFFICULTY, RATER_SEVERITY, REPLICATES, SEED, potential_scores, simulated_ratings, support,
)
from validation.m3_shared_anchors import ANCHORS, CONFIG, CONTRAST, NODES, SOURCE, TARGET, TARGETS, WEIGHTS, fit


ITEM_LOW = ITEM_DIFFICULTY * (.25 / ITEM_DIFFICULTY.std())
ITEM_HIGH = ITEM_DIFFICULTY * (1.25 / ITEM_DIFFICULTY.std())
RATER_STRICT = np.array([-1.2, 1.2, 0.])
RATER_LENIENT = np.array([0., -1.2, 1.2])
PROFILES = {
    "base": (1., ITEM_DIFFICULTY, RATER_SEVERITY),
    "person_low": (.5, ITEM_DIFFICULTY, RATER_SEVERITY),
    "person_high": (1.5, ITEM_DIFFICULTY, RATER_SEVERITY),
    "item_low": (1., ITEM_LOW, RATER_SEVERITY),
    "item_high": (1., ITEM_HIGH, RATER_SEVERITY),
    "rater_narrow": (1., ITEM_DIFFICULTY, RATER_SEVERITY*.2),
    "rater_R2_strict": (1., ITEM_DIFFICULTY, RATER_STRICT),
    "rater_R2_lenient": (1., ITEM_DIFFICULTY, RATER_LENIENT),
    "combined_high": (1.5, ITEM_HIGH, RATER_STRICT),
}
MODELS = ("free", "known", "shifted_R1", "estimated")
FREE_CONTRAST = SOURCE[5] - SOURCE[3]
ITEM_MAP = np.vstack([SOURCE[:3], -SOURCE[:3].sum(axis=0)])
RATER_MAP = SOURCE[3:6]
CRITICAL = norm.ppf(.975)
KEYS = ("replicate", "profile", "design", "model", "assumed_person_sd")
FIELDS = (*KEYS, "generating_person_sd", "item_sd", "rater_sd", "truth", "support",
          "count_0", "count_1", "count_2", "count_3", "estimate", "se", "lower", "upper",
          "propagated_se", "anchor_R1", "anchor_R2", "sensitivity_R1", "person_mse", "person_mean_eap_sd_sq",
          "person_reliability", "task_separation", "rater_separation", "reason")


def assumptions(person_sd):
    yield 1., MODELS
    if person_sd != 1.:
        yield person_sd, ("free", "estimated")


def validation_designs(replicate):
    ability_rng = np.random.default_rng(np.random.SeedSequence([SEED, replicate]).spawn(1)[0])
    ability = ability_rng.normal(size=60)
    for name, (person_sd, items, raters) in PROFILES.items():
        for design, data in simulated_ratings(replicate, TARGETS, ability_sd=person_sd, items=items, raters=raters):
            yield name, design, person_sd, items, raters, ability*person_sd, data


def calibration_ratings(replicate, items, raters):
    rng = np.random.default_rng(np.random.SeedSequence([2026112251, replicate]))
    ratings = pd.MultiIndex.from_product([range(60), CONFIG["facet_levels"]["Item"], CONFIG["facet_levels"]["Rater"]],
                                          names=["Person", "Item", "Rater"]).to_frame(index=False)
    ratings["Score"] = potential_scores(rng.normal(size=60), rng, items=items, raters=raters).ravel()
    return ratings


def calibration(replicate, items, raters):
    ratings = calibration_ratings(replicate, items, raters)
    if ratings.Score.nunique() != 4:
        raise ValueError("Calibration has an unused category")
    point, curvature, _, _ = fit(GroupedMmlLikelihood(ratings, CONFIG), SOURCE, np.zeros(9))
    return point[3:5], cho_solve(curvature, np.eye(len(point)))[3:5, 3:5]


def facet_separation(loading, point, curvature):
    spread = np.var(loading @ point, ddof=1)
    mse = np.einsum("ij,ji->", loading, cho_solve(curvature, loading.T))/len(loading)
    return float(np.sqrt(max(spread/mse - 1., 0.)))


def unanchored(target, offset, ability):
    point, curvature, _, _ = fit(target, SOURCE, offset)
    fitted = target.evaluate(offset + SOURCE @ point, NODES, WEIGHTS)
    errors = fitted["eap"] - ability[np.asarray(target.person_labels, dtype=int)]
    return dict(estimate=float(FREE_CONTRAST@point), se=float(np.sqrt(FREE_CONTRAST@cho_solve(curvature, FREE_CONTRAST))),
                person_mse=float(np.mean(errors**2)), person_mean_eap_sd_sq=float(np.mean(fitted["sd"]**2)),
                person_reliability=float(np.var(fitted["eap"], ddof=1) /
                    (np.var(fitted["eap"], ddof=1) + np.mean(fitted["sd"]**2))),
                task_separation=facet_separation(ITEM_MAP, point, curvature),
                rater_separation=facet_separation(RATER_MAP, point, curvature))


def anchored(target, offset, fixed, covariance):
    point, curvature, _, information = fit(target, TARGET, offset + ANCHORS @ fixed)
    target_direction = cho_solve(curvature, CONTRAST)
    jacobian = np.array([-1., 0.]) - target_direction @ (TARGET.T @ information @ ANCHORS)
    np.testing.assert_allclose(jacobian.sum(), 0., rtol=0, atol=1e-12)
    conditional = float(np.sqrt(CONTRAST @ target_direction))
    propagated = np.hypot(conditional, np.sqrt(max(float(jacobian @ covariance @ jacobian), 0.)))
    return dict(estimate=float(point[3]-fixed[0]), se=float(conditional), propagated_se=float(propagated),
                anchor_R1=float(fixed[0]), anchor_R2=float(fixed[1]), sensitivity_R1=float(jacobian[0]))


def run_replicate(replicate):
    calibrations, assignments = {}, {}
    rows = []
    for profile, design, sd, items, raters, ability, data in validation_designs(replicate):
        key = tuple(items), tuple(raters)
        if key not in calibrations:
            try:
                calibrations[key] = calibration(replicate, items, raters)
            except (ValueError, np.linalg.LinAlgError) as exc:
                calibrations[key] = str(exc)
        source = calibrations[key]
        if design not in assignments:
            assignments[design] = support(data)
        assert assignments[design] == "observed"
        target = GroupedMmlLikelihood(data, CONFIG)
        score_counts = np.bincount(data.Score, minlength=4)
        for assumed_sd, models in assumptions(sd):
            offset = np.zeros(9)
            offset[-1] = np.log(assumed_sd)
            for model in models:
                row = dict(replicate=replicate, profile=profile, design=design, model=model,
                           generating_person_sd=sd, assumed_person_sd=assumed_sd, item_sd=items.std(),
                           rater_sd=raters.std(), truth=raters[2]-raters[0], support=assignments[design],
                           **{f"count_{i}": int(n) for i, n in enumerate(score_counts)})
                try:
                    if 0 in score_counts:
                        raise ValueError("Receiving ratings have an unused category")
                    if model == "free":
                        output = unanchored(target, offset, ability)
                    elif model == "estimated":
                        if isinstance(source, str):
                            raise ValueError(f"Calibration: {source}")
                        output = anchored(target, offset, *source)
                    else:
                        fixed = raters[:2] + (np.array([.4, 0.]) if model == "shifted_R1" else 0.)
                        output = anchored(target, offset, fixed, np.zeros((2, 2)))
                    row.update(output, reason="")
                    row["lower"] = row["estimate"] - CRITICAL*row["se"]
                    row["upper"] = row["estimate"] + CRITICAL*row["se"]
                except (ValueError, np.linalg.LinAlgError) as exc:
                    row["reason"] = str(exc)
                rows.append(row)
    return rows


def run(out, workers, end):
    per_rep = len(TARGETS)*sum(len(models) for sd, _, _ in PROFILES.values() for _, models in assumptions(sd))
    complete = set()
    if out.exists():
        saved = pd.read_csv(out)
        assert tuple(saved.columns) == FIELDS and not saved.duplicated(list(KEYS)).any()
        assert saved.groupby("replicate").size().eq(per_rep).all()
        complete = set(saved.replicate)
    pending = [replicate for replicate in range(1, end+1) if replicate not in complete]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for count, rows in enumerate(pool.map(run_replicate, pending), 1):
            pd.DataFrame(rows, columns=FIELDS).to_csv(out, index=False, mode="a", header=not out.exists())
            if count % 25 == 0 or count == len(pending):
                print(f"{count}/{len(pending)} replicate sets", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--end", type=int, default=REPLICATES)
    args = parser.parse_args()
    if not 1 <= args.end <= REPLICATES:
        parser.error("Choose a replicate count between 1 and 500")
    run(args.out, args.workers, args.end)
