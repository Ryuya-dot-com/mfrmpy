"""Compare random and high-score double-rating audits on the same works."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from validation.m5_agreement_coefficients import RATERS, diagnostics
from validation.m5_rater_agreement import N_WORKS, SHARED_PROPORTION
from validation.m5b_annual_drift import STEPS, scores


REPLICATES = 500
STEP_PROFILES = {
    "standard": STEPS,
    "concentrated": np.cumsum([0., -4., -3., 3., 4.]),
}
DEPENDENCE = {"independent": 0., "shared": SHARED_PROPORTION}
AUDITS = {"all60": (60, False), "random6": (6, False), "high6": (6, True),
          "random24": (24, False), "high24": (24, True)}


def cases_at(replicate):
    rng = np.random.default_rng(np.random.SeedSequence([2026112241, replicate]))
    ability = rng.normal(size=N_WORKS)
    independent = rng.random((N_WORKS, len(RATERS)))
    shared = rng.random((N_WORKS, 1))
    shared_indicator = rng.random((N_WORKS, 1))
    assignment = rng.permutation(N_WORKS).reshape(2, N_WORKS//2)
    for profile, steps in STEP_PROFILES.items():
        for dependence, share in DEPENDENCE.items():
            draws = np.where(shared_indicator < share, shared, independent)
            response = scores(ability, draws, np.zeros(len(RATERS)), steps)
            # Stable sorting breaks primary-score ties in random assignment order.
            high_order = np.take_along_axis(
                assignment, np.argsort(-response[assignment, 0], axis=1, kind="stable"), axis=1
            )
            for audit, (n_double, by_high_score) in AUDITS.items():
                chosen = (high_order if by_high_score else assignment)[:, :n_double//2]
                observed = np.full(response.shape, np.nan)
                observed[:, 0] = response[:, 0]
                for partner, works in enumerate(chosen, start=1):
                    observed[works, partner] = response[works, partner]
                yield profile, dependence, audit, observed


def evaluate(replicate):
    for profile, dependence, audit, observed in cases_at(replicate):
        condition = f"{profile}_{dependence}_{audit}"
        rows = diagnostics(condition, replicate, observed)
        assert [row["scope"] for row in rows] == ["panel", "R1–R2", "R1–R3"]
        n_double = AUDITS[audit][0]
        assert [row["pairable_units"] for row in rows] == [n_double, n_double//2, n_double//2]
        available = np.isfinite(observed[:, 1:])
        for row, selected in zip(rows, [available.any(axis=1), *available.T], strict=True):
            sample = observed[selected]
            ratings = sample[np.isfinite(sample)].astype(int)
            count = np.bincount(ratings, minlength=6)
            assert len(ratings) == row["ratings_used"]
            row.update(profile=profile, dependence=dependence, audit=audit)
            row.update({f"score_{score}_count": count[score] for score in range(1, 6)})
            row["mean_primary_score"] = sample[:, 0].mean()
        yield rows, observed


def run(output, reference, replicates):
    result, inputs = [], []
    for replicate in range(1, replicates+1):
        for rows, observed in evaluate(replicate):
            result.extend(rows)
            if reference is not None and replicate == 1:
                inputs.append(pd.DataFrame(observed, columns=RATERS).assign(
                    condition=rows[0]["condition"], replicate=replicate
                ))
        if replicate % 100 == 0 or replicate == replicates:
            print(f"{replicate}/{replicates} replicate sets", flush=True)
    pd.DataFrame(result).to_csv(output, index=False, compression={"method": "gzip", "mtime": 0})
    if reference is not None:
        pd.concat(inputs, ignore_index=True).to_csv(reference, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--reference-ratings", type=Path)
    parser.add_argument("--replicates", type=int, default=REPLICATES)
    args = parser.parse_args()
    if not 1 <= args.replicates <= REPLICATES:
        parser.error(f"replicate count must be between 1 and {REPLICATES}")
    run(args.output, args.reference_ratings, args.replicates)
