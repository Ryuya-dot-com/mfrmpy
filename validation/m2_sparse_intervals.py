"""Reproduce the paired, fixed-budget simulation in m2_sparse_intervals.md."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import softmax
from scipy.stats import binomtest

import mfrmpy


REPLICATES = 500
SEED = 2026112002
TRUTH = 1.2
DESIGNS = dict(balanced6=(2, 2, 2), balanced24=(8, 8, 8), chain24=(12, 0, 12), balanced60=(20, 20, 20))
SETTINGS = dict(default=1e-6, tight=1e-10)
FIT = dict(person="Person", facets=["Item", "Rater"], score="Score", rating_min=0, rating_max=3,
           model="RSM", method="MML", population_prior_sd=1., quad_points=31,
           interval_requests=[dict(facet="Rater", difference=("R3", "R1"), level=.95)])
PAIR = np.array([[0, 1], [0, 2], [1, 2]])
ITEM_DIFFICULTY = np.array([-.75, -.25, .25, .75])
RATER_SEVERITY = np.array([-.6, 0., .6])
FIELDS = ("replicate", "design", "setting", "support", "estimate", "lower", "upper", "reason",
          "scaled_gradient_31", "scaled_gradient_61", "quadrature_score_shift")


def potential_scores(ability, score_rng, eta_shift=0., *, items=ITEM_DIFFICULTY, raters=RATER_SEVERITY):
    eta = ability[:, None, None] - items[None, :, None]
    eta = eta - raters[None, None, :]
    eta += eta_shift
    probs = softmax(eta[..., None] * np.arange(4) - np.cumsum([0., -.9, 0., .9]), axis=-1)
    return (score_rng.random(eta.shape)[..., None] > probs.cumsum(axis=-1)[..., :-1]).sum(axis=-1)


def simulated_ratings(replicate, designs, *, ability_sd=1., items=ITEM_DIFFICULTY, raters=RATER_SEVERITY):
    streams = np.random.SeedSequence([SEED, replicate]).spawn(2 + len(designs))
    score_rng, person_rng = (np.random.default_rng(x) for x in streams[:2])
    scores = potential_scores(ability_sd*score_rng.normal(size=60), score_rng, items=items, raters=raters)
    persons = person_rng.permutation(60)
    double_items = person_rng.random((60, 4)).argsort(axis=1)[:, :2]

    for (design, double_counts), stream in zip(designs.items(), streams[2:], strict=True):
        assignment_rng = np.random.default_rng(stream)
        pairs = assignment_rng.permutation(np.repeat(PAIR, double_counts, axis=0))
        shared = len(pairs)
        singles = (80 - 2 * np.bincount(pairs.ravel(), minlength=3)) // 4
        single_raters = assignment_rng.permutation(np.repeat(np.arange(3), singles))
        rows = [(p, i, r) for p, raters in zip(persons[:shared], pairs, strict=True)
                for i in double_items[p] for r in raters]
        rows.extend((p, i, r) for p, r in zip(persons[shared:], single_raters, strict=True) for i in range(4))
        p, i, r = np.array(rows).T
        assert len(rows) == len(set(rows)) == 240 and np.bincount(r).tolist() == [80, 80, 80]
        yield design, pd.DataFrame(dict(Person=p, Item=np.array(["I1", "I2", "I3", "I4"])[i],
                                        Rater=np.array(["R1", "R2", "R3"])[r], Score=scores[p, i, r]))


def support(data):
    return mfrmpy.check_rater_contrast(data, person="Person", facets=["Item", "Rater"], score="Score",
                                     rater="Rater", difference=("R3", "R1"))["status"]


def fit_result(data, reltol):
    analysis = mfrmpy.analyze(data, **FIT, reltol=reltol)
    entry, = analysis["interval_analyses"]
    if entry["status"] != "computed":
        return dict(reason=entry["reason"])
    row, = (r for r in entry["record"]["table"] if r["Kind"] == "difference")
    checks = {c["check"]: c["value"] for c in entry["record"]["checks"]}
    return dict(estimate=row["Estimate"], lower=row["Lower"], upper=row["Upper"], reason=row["Reason"],
                scaled_gradient_31=checks.get("GH31:scaled_gradient"), scaled_gradient_61=checks.get("GH61:scaled_gradient"),
                quadrature_score_shift=checks.get("quadrature_score_shift"))


def run_replicate(job):
    rep, remaining = job
    rows = []
    for design, data in simulated_ratings(rep, DESIGNS):
        if not any((design, setting) in remaining for setting in SETTINGS):
            continue
        observed = support(data)
        for setting, tol in SETTINGS.items():
            if (design, setting) in remaining:
                rows.append(dict(replicate=rep, design=design, setting=setting, support=observed,
                                 **fit_result(data, tol)))
    return rows


def run(path, workers):
    completed = set()
    if path.exists():
        with path.open(newline="") as saved:
            reader = csv.DictReader(saved)
            if tuple(reader.fieldnames or ()) != FIELDS:
                raise ValueError("Result columns differ from this protocol")
            for row in reader:
                key = (int(row["replicate"]), row["design"], row["setting"])
                if key in completed or None in row.values():
                    raise ValueError(f"Duplicate or incomplete saved result: {key}")
                completed.add(key)
    jobs = []
    for rep in range(1, REPLICATES + 1):
        remaining = {(design, setting) for design in DESIGNS for setting in SETTINGS
                     if (rep, design, setting) not in completed}
        if remaining:
            jobs.append((rep, remaining))
    if not jobs:
        return
    with path.open("a", newline="") as destination, ProcessPoolExecutor(max_workers=workers) as pool:
        writer = csv.DictWriter(destination, FIELDS)
        if not completed and destination.tell() == 0:
            writer.writeheader()
        for i, rows in enumerate(pool.map(run_replicate, jobs), 1):
            writer.writerows(rows)
            destination.flush()
            if i % 25 == 0 or i == len(jobs):
                print(f"{i}/{len(jobs)} replicate sets processed", flush=True)


def rate(numerator, denominator):
    if not denominator:
        return "NA", None
    ci = binomtest(numerator, denominator).proportion_ci(method="wilson")
    return f"{numerator}/{denominator} ({100*ci.low:.1f}–{100*ci.high:.1f}%)", ci


def summarize(path):
    data = pd.read_csv(path)
    expected = {(r, d, s) for r in range(1, REPLICATES+1) for d in DESIGNS for s in SETTINGS}
    assert len(data) == len(expected)
    assert set(data[["replicate", "design", "setting"]].itertuples(index=False, name=None)) == expected
    assert data.lower.notna().equals(data.upper.notna())
    assert data.loc[data.lower.notna(), "reason"].isna().all()
    assert data.loc[data.lower.isna(), "reason"].notna().all()
    assert (data.loc[data.lower.notna(), "lower"] <= data.loc[data.lower.notna(), "upper"]).all()
    print("| 設計 | EM | 返却（95% Wilson） | 被覆（95% Wilson） | 幅の中央値（logit） | 事前基準 |")
    print("|---|---|---:|---:|---:|---|")
    for design in DESIGNS:
        for setting in SETTINGS:
            group = data.loc[(data.design == design) & (data.setting == setting)]
            eligible = group.dropna(subset=["lower", "upper"])
            covered = ((eligible.lower <= TRUTH) & (TRUTH <= eligible.upper)).sum()
            returned, return_ci = rate(len(eligible), len(group))
            coverage, coverage_ci = rate(covered, len(eligible))
            width = f"{(eligible.upper - eligible.lower).median():.3f}" if len(eligible) else "NA"
            passed = (return_ci.low >= .80 and coverage_ci is not None
                      and coverage_ci.low >= .92 and coverage_ci.high <= .98)
            print(f"| {design} | {setting} | {returned} | {coverage} | {width} | {'適合' if passed else '条件未達'} |")
    held = data.loc[data.lower.isna()]
    print(held.groupby(["design", "setting", "reason"]).size().to_string())
    print("Design support:", data.groupby(["design", "setting", "support"]).size().to_string())


if __name__ == "__main__":
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("action", choices=["run", "summarize", "control"])
    command.add_argument("path", nargs="?", type=Path)
    command.add_argument("--workers", type=int, default=4)
    args = command.parse_args()
    if args.action == "control":
        _, control = next(simulated_ratings(1, {"isolated": (12, 0, 0)}))
        observed = support(control)
        for setting, tol in SETTINGS.items():
            print(setting, observed, fit_result(control, tol))
    elif args.path is None:
        command.error("run and summarize require a CSV path")
    elif args.action == "run":
        run(args.path, args.workers)
    else:
        summarize(args.path)
