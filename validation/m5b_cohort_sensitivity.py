"""Compare high-score replays with new entries when cohort means are equal."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

import mfrmpy
from validation.m5b_annual_drift import AUDIT, EFFECTS
from validation.m5b_category_drift import MODELS, fit_case, unused_categories
from validation.m5b_rating_allocation import COUNTS, INTERVAL, REPLICATES, cases_at


CONFIG = MODELS["PCM_shared_except_B_R3"]
ALLOCATIONS = ("replay_high", "new")
TRUTH = EFFECTS["r3_drifts"][-1]
FIELDS = ["replicate", "allocation", "support", *COUNTS, "unused_step_categories", "reason", *INTERVAL]


def study_cases(replicate, new_cohort_mean=0.):
    for selection, target in zip(("high", "random"), ALLOCATIONS, strict=True):
        yield next((allocation, rows)
                   for effect, allocation, rows in cases_at(
                       replicate, replay_selection=selection, new_cohort_mean=new_cohort_mean)
                   if effect == "r3_drifts" and allocation == target)


def evaluate(replicate):
    out = []
    settings = AUDIT | dict(facets=["WorkYear", "RaterYear"], noncenter_facet="RaterYear",
                           positive_facets=["WorkYear"])
    for allocation, rows in study_cases(replicate):
        audit = mfrmpy.check_rater_contrast(rows, **settings)
        counts = rows.loc[rows.RaterYear.eq("B-R3"), "Score"].value_counts()
        assert len(rows) == 264 and counts.sum() == 52
        unused = unused_categories(rows, CONFIG["step_facet"])
        fit = fit_case(rows, CONFIG, audit["observation_estimable"])
        assert not unused or fit["reason"] == "Design rank or boundary: no_finite_mle"
        out.append(dict(replicate=replicate, allocation=allocation, support=audit["status"],
                        **{name: counts.get(k, 0) for k, name in enumerate(COUNTS, 1)},
                        unused_step_categories=unused, **fit))
    return out


def run(path, end, workers):
    completed = set()
    if path.exists():
        previous = pd.read_csv(path)
        assert list(previous.columns) == FIELDS and previous.allocation.isin(ALLOCATIONS).all()
        assert not previous.duplicated(["replicate", "allocation"]).any()
        assert previous.groupby("replicate").size().eq(len(ALLOCATIONS)).all()
        completed.update(previous.replicate)
    pending = sorted(set(range(1, end+1)) - completed)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for count, rows in enumerate(pool.map(evaluate, pending), 1):
            pd.DataFrame(rows).reindex(columns=FIELDS).to_csv(path, mode="a", header=not path.exists(), index=False)
            if count % 100 == 0 or count == len(pending):
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
