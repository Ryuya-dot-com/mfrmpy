"""Compare two uses of the same extra judging budget for annual rater drift."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from itertools import islice
from pathlib import Path

import numpy as np
import pandas as pd

import mfrmpy
from validation.m2_sparse_intervals import rate
from validation.m5b_annual_drift import (
    AUDIT, EFFECTS, NEW_COHORT_MEAN, PAIR, RATERS, SEVERITY,
    current_step_scores, rating_draws, ratings_at, rows_for_year, scores,
)
from validation.m5b_category_drift import DESIGN, MODELS, SHIFTS, fit_case, unused_categories


REPLICATES = 2000
EXTRA_RECORDINGS = 12
ALLOCATIONS = ("baseline", "replay", "new")
SCORE_SELECTIONS = ("high", "tails")
SELECTED_ALLOCATIONS = tuple(f"replay_{label}" for label in SCORE_SELECTIONS)
CONDITIONS = ("cohort_only", "r3_drifts")
MODEL_NAMES = ("RSM", "PCM_shared_except_B_R3")
KEY = ["replicate", "effect", "allocation", "model"]
COUNTS = [f"score_{k}" for k in range(1, 6)]
INTERVAL = ["severity_estimate", "severity_se", "severity_lower", "severity_upper"]
FIELDS = [*KEY, "truth", "r3_ratings", "support", "unused_step_categories", "reason", *COUNTS, *INTERVAL]
EVIDENCE = Path(__file__).with_name("evidence")
BASELINE = [EVIDENCE / "m5b_category_drift.csv.gz", EVIDENCE / "m5b_category_drift_501_2000.csv.gz"]
RATER_PAIRS = {tuple(np.array(RATERS)[members]): code for code, members in enumerate(PAIR)}


def additional_pairs(base, replicate, selection="random"):
    old = base.loc[base.RatingYear.eq("A")]
    current = base.loc[base.RatingYear.eq("B")]
    eligible = sorted(set(old.loc[old.Rater.eq("R3"), "Recording"]) - set(current.Recording))
    rng = np.random.default_rng(np.random.SeedSequence([2026112108, replicate]))
    chosen = rng.choice(eligible, EXTRA_RECORDINGS, replace=False)
    records = old.loc[old.Recording.isin(eligible)].groupby("Recording").agg(
        Pair=("Rater", lambda ratings: RATER_PAIRS[tuple(sorted(ratings))]),
        Band=("Band", "first"), Total=("Score", "sum"))
    pair = records.loc[chosen, "Pair"].to_numpy()
    if selection != "random":
        for code in sorted(set(pair)):
            positions = pair == code
            n = int(positions.sum())
            pool = records.loc[records.Pair.eq(code)]
            ranked = pool.loc[rng.permutation(pool.index)].sort_values("Total", kind="stable")
            bottom = n//2 if selection == "tails" else 0
            labels = list(ranked.head(bottom).index) + list(ranked.tail(n-bottom).index)
            assert len(labels) == len(set(labels)) == n
            chosen[positions] = labels
    recordings = np.array([int(label[1:]) for label in chosen])
    band = records.loc[chosen, "Band"].to_numpy()
    assert np.array_equal(pair, records.loc[chosen, "Pair"].to_numpy())
    return recordings, pair, band


def cases_at(replicate, *, replay_selection="random", new_cohort_mean=NEW_COHORT_MEAN):
    if replay_selection not in ("random", *SCORE_SELECTIONS):
        raise ValueError("Choose random, high or tails for replay selection")
    _, (old_ability, new_ability), (_, u_replay, u_new) = rating_draws(replicate)
    steps = current_step_scores(SHIFTS["four"])
    original = None
    base_cases = ratings_at(replicate, DESIGN, r3_step_shift=SHIFTS["four"], new_cohort_mean=new_cohort_mean)
    for _, effect, base in islice(base_cases, 2):
        assert len(base) == 240 and base.RaterYear.eq("B-R3").sum() == 40
        if original is None:
            original = base
            selected, pair, band = additional_pairs(base, replicate, replay_selection)
        else:
            assert base.drop(columns="Score").equals(original.drop(columns="Score"))
            unchanged = base.RaterYear.ne("B-R3")
            assert base.loc[unchanged, "Score"].equals(original.loc[unchanged, "Score"])
        severity = SEVERITY + np.array(EFFECTS[effect])
        replay = rows_for_year("B", ["A"]*EXTRA_RECORDINGS, selected, pair,
                                scores(old_ability, u_replay, severity, steps)[selected], band)
        augmented = pd.concat([base, replay], ignore_index=True)
        if replay_selection == "random":
            new_ids = np.arange(36, 36+EXTRA_RECORDINGS)
            new = rows_for_year("B", ["B"]*EXTRA_RECORDINGS, new_ids, pair,
                                scores(new_ability+new_cohort_mean, u_new, severity, steps)[new_ids],
                                np.full(EXTRA_RECORDINGS, -1))
            pd.testing.assert_series_equal(replay.RaterYear.value_counts(), new.RaterYear.value_counts())
            allocations = (("baseline", base), ("replay", augmented), ("new", pd.concat([base, new], ignore_index=True)))
        else:
            allocations = ((f"replay_{replay_selection}", augmented),)
        for allocation, rows in allocations:
            expected = 240 if allocation == "baseline" else 240 + 2*EXTRA_RECORDINGS
            assert len(rows) == expected and not rows.duplicated(["Recording", "RaterYear"]).any()
            yield effect, allocation, rows.assign(StepProfile=np.where(rows.RaterYear.eq("B-R3"), "separate", "common"))


def study_cases(replicate, score_selection=False):
    for selection in SCORE_SELECTIONS if score_selection else ("random",):
        yield from cases_at(replicate, replay_selection=selection)


def evaluate(replicate, *, score_selection=False):
    out = []
    settings = AUDIT | dict(facets=["WorkYear", "RaterYear"], noncenter_facet="RaterYear",
                           positive_facets=["WorkYear"])
    for effect, allocation, rows in study_cases(replicate, score_selection):
        if (effect, allocation) == ("cohort_only", "baseline"):
            continue
        audit = mfrmpy.check_rater_contrast(rows, **settings)
        reviews = rows.loc[rows.RaterYear.eq("B-R3"), "Score"]
        count = reviews.value_counts()
        assert len(reviews) == (40 if allocation == "baseline" else 40+EXTRA_RECORDINGS)
        for model in MODEL_NAMES:
            cfg = MODELS[model]
            fit = fit_case(rows, cfg, audit["observation_estimable"])
            out.append(dict(replicate=replicate, effect=effect, allocation=allocation, model=model,
                            truth=EFFECTS[effect][-1], r3_ratings=len(reviews), support=audit["status"],
                            unused_step_categories=unused_categories(rows, cfg.get("step_facet")),
                            **{k: count.get(i, 0) for i, k in enumerate(COUNTS, 1)}, **fit))
    return out


def run(path, end, workers, score_selection=False):
    done = set()
    if path.exists():
        previous = pd.read_csv(path)
        assert list(previous.columns) == FIELDS and not previous.duplicated(KEY).any()
        allowed = SELECTED_ALLOCATIONS if score_selection else ALLOCATIONS
        assert previous.groupby("replicate").size().eq(8 if score_selection else 10).all()
        assert previous.allocation.isin(allowed).all() and previous.effect.isin(CONDITIONS).all()
        assert previous.model.isin(MODEL_NAMES).all()
        done = set(previous.replicate)
    pending = sorted(set(range(1, end+1)) - done)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, rows in enumerate(pool.map(partial(evaluate, score_selection=score_selection), pending), 1):
            pd.DataFrame(rows).reindex(columns=FIELDS).to_csv(path, mode="a", header=not path.exists(), index=False)
            if i % 25 == 0 or i == len(pending):
                print(f"{i}/{len(pending)} replicate sets", flush=True)


def combined(path, end, selection_results=None):
    new = pd.read_csv(path)
    assert len(new) == end*10 and not (new.effect.eq("cohort_only") & new.allocation.eq("baseline")).any()
    baseline = pd.concat([pd.read_csv(source) for source in BASELINE])
    baseline = baseline.loc[baseline.condition.eq("four") & baseline.model.isin(MODEL_NAMES) & baseline.replicate.le(end)]
    baseline = baseline.assign(effect="cohort_only", allocation="baseline", truth=0., r3_ratings=40)
    tables = [new, baseline[FIELDS]]
    allocations = ALLOCATIONS
    if selection_results is not None:
        tables.append(pd.read_csv(selection_results))
        allocations += SELECTED_ALLOCATIONS
    data = pd.concat(tables, ignore_index=True).set_index(KEY).sort_index()
    expected = pd.MultiIndex.from_product([range(1, end+1), CONDITIONS, allocations, MODEL_NAMES], names=KEY)
    assert data.index.is_unique and len(data) == len(expected) and expected.isin(data.index).all()
    assert data[COUNTS].sum(axis=1).eq(data.r3_ratings).all()
    return data


def summarize(path, end, selection_results=None):
    data = combined(path, end, selection_results)
    for (effect, allocation, model), group in data.groupby(level=["effect", "allocation", "model"]):
        returned = group.severity_lower.notna()
        assert returned.equals(group.reason.isna()) and returned.equals(group.severity_upper.notna())
        fit = group.loc[returned]
        assert np.isfinite(fit[INTERVAL]).all().all() and fit.severity_se.gt(0).all()
        truth = EFFECTS[effect][-1]
        assert group.truth.eq(truth).all()
        covered = fit.severity_lower.le(truth) & fit.severity_upper.ge(truth)
        _, return_ci = rate(len(fit), end)
        _, coverage_ci = rate(covered.sum(), len(fit))
        meets = (return_ci.low >= .80 and coverage_ci is not None
                 and coverage_ci.low >= .92 and coverage_ci.high <= .98)
        print(effect, allocation, model, f"returned={len(fit)}, cover={covered.sum()}, qualified={meets}",
              return_ci, coverage_ci, f"negative/positive={fit.severity_upper.lt(0).sum()}/{fit.severity_lower.gt(0).sum()}",
              f"bias={fit.severity_estimate.mean()-truth:+.4f}",
              f"median width={(fit.severity_upper-fit.severity_lower).median():.4f}",
              f"score 1 absent={group.score_1.eq(0).sum()}",
              f"score 5 absent={group.score_5.eq(0).sum()}, mean score 5={group.score_5.mean():.4f}",
              group.reason.value_counts().to_dict(), group.support.value_counts().to_dict())

    pcm = data.xs("PCM_shared_except_B_R3", level="model")
    pairs = [("replay", "new")]
    if selection_results is not None:
        pairs.extend((a, b) for a in SELECTED_ALLOCATIONS for b in ("replay", "new"))
    for effect, (left, right) in ((e, pair) for e in CONDITIONS for pair in pairs):
        left_fit = pcm.xs((effect, left), level=("effect","allocation"))
        right_fit = pcm.xs((effect, right), level=("effect","allocation"))
        both = left_fit.severity_lower.notna() & right_fit.severity_lower.notna()
        for name, lhs, rhs in (
            ("return percentage points", 100*left_fit.severity_lower.notna().astype(int),
             100*right_fit.severity_lower.notna().astype(int)),
            ("positive detection percentage points", 100*left_fit.severity_lower.gt(0).astype(int),
             100*right_fit.severity_lower.gt(0).astype(int)),
            ("width logit", (left_fit.severity_upper-left_fit.severity_lower)[both],
             (right_fit.severity_upper-right_fit.severity_lower)[both]),
        ):
            delta = lhs-rhs
            print(effect, f"{left} minus {right}", name, f"paired={len(delta)}, {delta.mean():+.4f} ± {1.96*delta.sem():.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--end", type=int, default=REPLICATES)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--score-selection", action="store_true", help="run the two extra score-based replay conditions")
    parser.add_argument("--selection-results", type=Path, help="include the extra score-based replay results in the summary")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--summarize", action="store_true")
    mode.add_argument("--pilot-ratings", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.end <= REPLICATES:
        parser.error("replicate count must be between 1 and 2000")
    if args.selection_results is not None and not args.summarize:
        parser.error("--selection-results requires --summarize")
    if args.score_selection and args.summarize:
        parser.error("pass score-based results with --selection-results")
    if args.summarize:
        summarize(args.path, args.end, args.selection_results)
    elif args.pilot_ratings:
        pd.concat(rows.assign(replicate=rep, condition=f"{effect}/{allocation}") for rep in range(1,args.end+1)
                  for effect, allocation, rows in study_cases(rep, args.score_selection)).to_csv(args.path, index=False)
    else:
        run(args.path, args.end, args.workers, args.score_selection)
