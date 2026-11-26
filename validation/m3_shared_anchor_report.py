"""Check the public shared-calibration report against the retained two-stage benchmark."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

import mfrmpy
from validation import m3_shared_anchors as benchmark
from validation.m3_anchor_interval_report import SETTINGS


REFERENCE = Path(__file__).with_name("evidence") / "m3_shared_anchors.csv.gz"
TARGET = ("R3", "R1")


def event_ids(data):
    return data[["Person", "Item", "Rater"]].agg("/".join, axis=1)


def source_scenario(rep, condition, order):
    data = benchmark.calibration(rep, condition)
    data["RatingID"] = event_ids(data)
    if condition == "fresh60":
        data["RatingID"] = "separate/" + data["RatingID"]
    analysis = mfrmpy.analyze(data, **(SETTINGS | dict(quad_points=order)))
    fit = analysis["result"]
    values = dict(zip(fit["config"]["facet_levels"]["Rater"], fit["params"]["facets"]["Rater"], strict=True))
    anchors = [dict(Facet="Rater", Level=label, Anchor=float(values[label]), Source=condition) for label in ("R1", "R2")]
    return dict(anchors=anchors, calibration=dict(kind="shared", analysis=analysis, rating_id="RatingID"))


def check_perturbation(source, receiving, shared):
    src = benchmark.GroupedMmlLikelihood(source, benchmark.CONFIG)
    dst = benchmark.GroupedMmlLikelihood(receiving, benchmark.CONFIG)
    src_point, *_ = benchmark.fit(src, benchmark.SOURCE, np.zeros(9))
    dst_point, *_ = benchmark.fit(dst, benchmark.TARGET, benchmark.ANCHORS @ src_point[3:5])
    u, v = pd.Series(shared["calibration"]), pd.Series(shared["receiving"])
    # Perturb a shared person and, where available, a person unique to either stage.
    ids = [group[0] for group in (u.index.intersection(v.index), u.index.difference(v.index),
                                  v.index.difference(u.index)) if len(group)]
    return max(benchmark.perturbation(src, dst, src_point, dst_point, who, u.get(who, 0.) + v.get(who, 0.))
               for who in ids)


def run(rep, expected, example=None):
    order = 61
    scenarios = {c: source_scenario(rep, c, order) for c in benchmark.CONDITIONS}
    rows = []
    for design, data in benchmark.simulated_ratings(rep, benchmark.TARGETS):
        data = data.assign(Person=data.Person.map("P{:03}".format))
        data["RatingID"] = event_ids(data)
        base = mfrmpy.analyze(data, **(SETTINGS | dict(quad_points=order)))
        analysis = mfrmpy.with_anchor_sensitivity(base, rater="Rater", difference=TARGET, intervals=True, scenarios=scenarios)
        table = mfrmpy.tables(analysis, "Anchor sensitivity intervals")[0][1].set_index(["Scenario", "Basis"])
        for scenario in analysis["anchor_sensitivity"]["scenarios"][1:]:
            condition = scenario["name"]
            row = table.loc[condition, "shared_calibration"]
            ref = expected.loc[(rep, design, condition)]
            assert row.ReusedRatings == ref.reused_ratings
            assert len(scenario["calibration"]["overlap"]) == ref.shared_people
            status = dict(replicate=rep, design=design, condition=condition, reason=row.Reason)
            if row.Status == "withheld":
                rows.append(status)
                continue
            conditional = table.loc[condition, "conditional"]
            observed = [row.Estimate, conditional.SE, row.CalibrationVariance, row.ReceivingVariance, row.SharedCrossTerm, row.SE]
            reference = ref[["estimate", "model_fixed_se", "score_anchor_variance", "score_target_variance",
                             "score_cross_term", "score_shared_se"]].to_numpy(dtype=float)
            if rep <= 2:
                np.testing.assert_allclose(observed, reference, atol=5e-5, rtol=0)
            source = scenarios[condition]["calibration"]["analysis"]["input_data"]
            error = check_perturbation(source, data, scenario["interval"]["shared"]) if rep <= 2 else None
            rows.append(dict(status, estimate=row.Estimate, shared_se=row.SE, covered=row.Lower <= 1.2 <= row.Upper, max_difference=max(abs(observed-reference)),
                             perturbation_error=error))
        if example and (rep, design) == (1, "balanced24"):
            mfrmpy.save_project(example, analysis)
            stored = mfrmpy.load_project(example)
            pd.testing.assert_frame_equal(mfrmpy.tables(stored, "Anchor sensitivity intervals")[0][1],
                                          mfrmpy.tables(analysis, "Anchor sensitivity intervals")[0][1])
    return rows


def run_job(job):
    rep, expected, example = job
    return run(rep, expected, example)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--example", type=Path)
    parser.add_argument("--out", type=Path, help="Checkpoint CSV for the six rows of each completed replicate")
    parser.add_argument("--end", type=int, default=2, choices=range(1, 501), metavar="1..500")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    expected = pd.read_csv(REFERENCE).set_index(["replicate", "design", "condition"])
    columns = ["replicate", "design", "condition", "reason", "estimate", "shared_se", "covered",
               "max_difference", "perturbation_error"]
    complete = set()
    if args.out and args.out.exists():
        saved = pd.read_csv(args.out)
        if (list(saved) != columns or saved.duplicated(columns[:3]).any()
                or not saved.groupby("replicate").size().eq(6).all()):
            raise ValueError("Incomplete or incompatible checkpoint")
        complete = set(saved.replicate)
    jobs = [(i, expected.loc[[i]], args.example if i == 1 else None)
            for i in range(1, args.end+1) if i not in complete]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for rows in pool.map(run_job, jobs):
            frame = pd.DataFrame(rows, columns=columns)
            if args.out:
                frame.to_csv(args.out, index=False, mode="a", header=not args.out.exists())
            else:
                print(frame.to_string(index=False), flush=True)
