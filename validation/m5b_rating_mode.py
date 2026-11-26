"""Audit and compare alternative scoring-mode links for an annual contrast."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

import mfrmpy
from validation.m5b_annual_drift import (
    AUDIT, DESIGNS, NODES, PAIR, RATERS, REPLICATES, SEVERITY, rating_draws, ratings_at, rows_for_year, scores,
)


TRUTHS = {"mode_only": 0., "drift_only": .5}
BASE = {"repeat24": DESIGNS["repeat24"]}
FIELDS = (
    "replicate", "case", "allocation", "model", "truth", "support", "reason", "estimate", "se", "lower", "upper",
)


def cases_at(replicate):
    base = next(rows for _, effect, rows in ratings_at(replicate, BASE) if effect == "r3_drifts")
    base = base.assign(Mode=base.RatingYear.map({"A": "live", "B": "video"}))
    old = base.loc[base.RatingYear.eq("A")]
    pair_names = {tuple(np.array(RATERS)[members]): k for k, members in enumerate(PAIR)}
    recordings = old.groupby("Recording").agg(
        Pair=("Rater", lambda labels: pair_names[tuple(sorted(labels))]), Band=("Band", "first"))
    old_ability = rating_draws(replicate)[1][0]
    rng = np.random.default_rng(np.random.SeedSequence([2026112114, replicate]))
    repeat_uniforms = rng.random((len(old_ability), len(RATERS)))
    target_pairs = np.flatnonzero((PAIR == RATERS.index("R3")).any(axis=1))
    has_target = recordings.Pair.isin(target_pairs)
    for allocation, eligible in (("other_raters", ~has_target), ("includes_R3", has_target)):
        selected = recordings.loc[rng.choice(recordings.index[eligible], 12, replace=False)]
        indices = np.array([int(label[1:]) for label in selected.index])
        for case, drift in TRUTHS.items():
            severity = SEVERITY + np.array([0., 0., .5 - drift])
            ratings = scores(old_ability, repeat_uniforms, severity)
            added = rows_for_year("A", ["A"]*len(selected), indices, selected.Pair.to_numpy(),
                                  ratings[indices], selected.Band.to_numpy())
            rows = pd.concat([base, added], ignore_index=True)
            rows["RaterYearMode"] = rows.RaterYear + "-" + rows.Mode
            assert len(rows) == 264 and not rows.duplicated(["Recording", "RaterYearMode"]).any()
            yield case, allocation, rows


def assess(rows, model):
    if model == "common_mode":
        rater = "RaterYear"
        facets = ["WorkYear", rater, "Mode"]
        difference = ("B-R3", "A-R3")
    else:
        rater = "RaterYearMode"
        facets = ["WorkYear", rater]
        difference = ("B-R3-video", "A-R3-video")
        if difference[1] not in set(rows[rater]):
            return dict(support="not_identified", reason="No R3 video score in year A")
    settings = AUDIT | dict(rater=rater, facets=facets, difference=difference,
                           noncenter_facet=rater, positive_facets=["WorkYear"])
    audit = mfrmpy.check_rater_contrast(rows, **settings)
    assert audit["status"] == "observed"
    fit = mfrmpy.analyze(
        rows, person="Recording", facets=facets, score="Score", model="RSM", method="MML",
        noncenter_facet=rater, positive_facets=["WorkYear"], rating_min=1, rating_max=5,
        quad_points=len(NODES), reltol=1e-12, interval_requests=[dict(facet=rater, difference=difference)],
    )
    entry, = fit["interval_analyses"]
    if entry["status"] != "computed":
        return dict(support=audit["status"], reason=entry["reason"])
    result = next(record for record in entry["record"]["table"] if record["Kind"] == "difference")
    return dict(support=audit["status"], reason=result["Reason"], estimate=result["Estimate"],
                se=result["SE"], lower=result["Lower"], upper=result["Upper"])


def evaluate(replicate):
    out = []
    other_raters = None
    for case, allocation, rows in cases_at(replicate):
        for model in ("common_mode", "separate_cells"):
            if allocation == "other_raters" and case == "drift_only" and model == "common_mode":
                pd.testing.assert_frame_equal(rows, other_raters[0])
                result = other_raters[1]
            else:
                result = assess(rows, model)
                if allocation == "other_raters" and model == "common_mode":
                    other_raters = (rows, result)
            out.append(dict(replicate=replicate, case=case, allocation=allocation, model=model,
                            truth=TRUTHS[case], **result))
    return out


def run(path, end, workers):
    done = set()
    if path.exists():
        saved = pd.read_csv(path)
        key = ["replicate", "case", "allocation", "model"]
        assert list(saved.columns) == list(FIELDS) and not saved.duplicated(key).any()
        assert saved.groupby("replicate").size().eq(8).all()
        done = set(saved.replicate)
    pending = sorted(set(range(1, end+1)) - done)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for count, rows in enumerate(pool.map(evaluate, pending), 1):
            pd.DataFrame(rows).reindex(columns=FIELDS).to_csv(path, mode="a", header=not path.exists(), index=False)
            if count % 25 == 0 or count == len(pending):
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
