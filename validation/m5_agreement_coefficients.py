"""Compare observed agreement, Krippendorff alpha and paired Cohen kappa."""
import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from mfrm_app.engine import KRIPPENDORFF_ALPHA_METRICS, _krippendorff_alpha_from_wide
from validation.m5_rater_agreement import REPLICATES, WORK_IDS, cases_at


SCORES = np.arange(1, 6)
RATERS = ("R1", "R2", "R3")


def cases():
    for replicate in range(1, REPLICATES+1):
        for condition, _, _, _, scores in cases_at(replicate):
            yield condition, replicate, scores

    shared = list(cases_at(1))[-1][-1]
    yield "perfect_varied", 1, np.repeat(shared[:, :1], 3, axis=1)
    yield "perfect_constant", 1, np.full_like(shared, 3)

    concentrated = np.full((len(WORK_IDS), 2), 3)
    concentrated[0, 0] = concentrated[1, 1] = 4
    yield "different_rare_scores", 1, concentrated

    sparse = np.full(shared.shape, np.nan)
    for i, pair in enumerate(combinations(range(3), 2)):
        first = slice(i*10, (i+1)*10)
        sparse[first, pair] = shared[first, pair]
    for i in range(3):
        first = slice(30+i*10, 40+i*10)
        sparse[first, i] = shared[first, i]
    yield "shared_judgment_sparse", 1, sparse


def kappa(paired):
    cross = np.zeros((len(SCORES), len(SCORES)), dtype=int)
    np.add.at(cross, tuple(paired.astype(int).T-1), 1)
    levels = np.abs(np.subtract.outer(SCORES, SCORES))
    observed = cross/cross.sum()
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0))
    result = {}
    for name, distance in (("nominal", levels > 0), ("linear", levels), ("quadratic", levels**2)):
        de = np.sum(expected*distance)
        result[f"kappa_{name}"] = float(1 - np.sum(observed*distance)/de) if de > 0 else np.nan
    return result


def diagnostics(condition, replicate, scores):
    wide = pd.DataFrame(scores, index=WORK_IDS, columns=RATERS[:scores.shape[1]])
    pairable = wide.notna().sum(axis=1).ge(2)
    panel = dict(condition=condition, replicate=replicate, scope="panel",
                 units=len(wide), pairable_units=int(pairable.sum()), ratings_used=int(wide.loc[pairable].count().sum()))
    for method in KRIPPENDORFF_ALPHA_METRICS:
        alpha, = _krippendorff_alpha_from_wide(wide, {}, "Rater", metric=method,
                                              include_coverage=False)["summary"].itertuples(index=False)
        assert alpha.PairableUnits == panel["pairable_units"] and alpha.RatingsUsed == panel["ratings_used"]
        panel[f"alpha_{method}"] = alpha.KrippendorffAlpha
    pairs = []
    matched = 0
    n_pairs = 0
    for first, second in combinations(wide.columns, 2):
        paired = wide[[first, second]].dropna().to_numpy()
        count = len(paired)
        if count == 0:
            continue
        exact_count = int(np.sum(paired[:, 0] == paired[:, 1]))
        matched += exact_count
        n_pairs += count
        pairs.append(dict(condition=condition, replicate=replicate, scope=f"{first}–{second}",
                          units=len(wide), pairable_units=count, compared_pairs=count, ratings_used=2*count,
                          exact_agreement=exact_count/count, **kappa(paired)))
    return [dict(panel, exact_agreement=matched/n_pairs if n_pairs else np.nan,
                 compared_pairs=n_pairs), *pairs]


def main(output, reference):
    rows, inputs = [], []
    for condition, replicate, scores in cases():
        rows.extend(diagnostics(condition, replicate, scores))
        if reference is not None and replicate == 1:
            inputs.append(pd.DataFrame(scores, columns=RATERS[:scores.shape[1]]).assign(
                condition=condition, replicate=replicate))
    pd.DataFrame(rows).to_csv(output, index=False, compression={"method": "gzip", "mtime": 0})
    if reference is not None:
        pd.concat(inputs, ignore_index=True).to_csv(reference, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--reference-ratings", type=Path)
    args = parser.parse_args()
    main(args.output, args.reference_ratings)
