"""Compare the annual public API with retained, independently computed M5b results."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import mfrmpy
from mfrm_app import engine
from validation.m5b_annual_drift import DESIGNS, ratings_at
from validation.m5b_anchor_report import SETTINGS, TARGET


def run(output):
    output.mkdir(parents=True, exist_ok=True)
    evidence = Path(__file__).parent / "evidence"
    expected = pd.read_csv(evidence / "m5b_estimated_anchors.csv.gz").set_index(["replicate", "design", "effect"])
    unanchored = pd.read_csv(evidence / "m5b_annual_drift.csv.gz")
    unanchored = unanchored.loc[unanchored.model.eq("recordings") & unanchored.target.eq("R3")].set_index(["replicate", "design", "effect"])
    records = []
    for replicate in (1, 2, 44):
        source = None
        for design, effect, data in ratings_at(replicate, {key: DESIGNS[key] for key in ("repeat6", "repeat24")}):
            if effect == "panel_drifts":
                continue
            data = data.assign(Event=["rating-" + str(i) for i in range(len(data))])
            if source is None:
                source = mfrmpy.analyze(data.loc[data.RatingYear.eq("A")], **SETTINGS,
                                         facets=["Rater"], noncenter_facet="Rater")
                values = dict(zip(source["result"]["config"]["facet_levels"]["Rater"], source["result"]["params"]["facets"]["Rater"]))
                fixed = [dict(Facet="RaterYear", Level=f"{year}-{rater}", Anchor=float(values[rater]), Source="Previous-year calibration")
                         for year in ("A", "B") for rater in ("R1", "R2")]
                calibration = dict(kind="shared", analysis=source, facet="Rater", rating_id="Event",
                    level_map={f"{year}-{rater}": rater for year in ("A", "B") for rater in ("R1", "R2", "R3")})
            base = mfrmpy.analyze(data, **SETTINGS, facets=["WorkYear", "RaterYear"],
                                   noncenter_facet="RaterYear", positive_facets=["WorkYear"])
            linked = mfrmpy.with_anchor_sensitivity(base, rater="RaterYear", difference=TARGET, intervals=True,
                occasion="RatingYear", important_difference=.5,
                scenarios={"Previous values": dict(anchors=fixed, calibration=calibration)})
            frame = mfrmpy.tables(linked, "Anchor sensitivity intervals")[0][1].set_index("Basis")
            old = expected.loc[replicate, design, effect]
            reference = unanchored.loc[replicate, design, effect]
            assert frame.Status.eq("model_approximation").all(), (replicate, design, effect, frame.Reason.tolist())
            actual = frame.loc["shared_calibration"]
            comparisons = dict(estimate=(actual.Estimate, old.estimate), conditional_se=(frame.loc["conditional", "SE"], old.fixed_se),
                               shared_se=(actual.SE, old.score_shared_se), calibration_variance=(actual.CalibrationVariance, old.score_anchor_variance),
                               receiving_variance=(actual.ReceivingVariance, old.score_target_variance), cross_term=(actual.SharedCrossTerm, old.score_cross_term),
                               unanchored_estimate=(frame.loc["no_anchors", "Estimate"], reference.estimate),
                               unanchored_se=(frame.loc["no_anchors", "SE"], reference.se))
            errors = {key: abs(a-b) for key, (a, b) in comparisons.items()}
            assert all(error < (5e-5 if "estimate" in key else 2e-6) for key, error in errors.items()), errors
            assert actual.CalibrationWorks == actual.OverlapWorks == 60 and actual.ReusedRatings == 120
            overlap = mfrmpy.tables(linked, "Anchor sensitivity occasion overlap")[0][1].SharedWorks.item()
            assert overlap == DESIGNS[design][0]
            records.append(dict(replicate=replicate, design=design, effect=effect, errors=errors,
                                estimate=float(actual.Estimate), shared_se=float(actual.SE), occasion_overlap=overlap))
            if (replicate, design, effect) == (44, "repeat6", "anchors_drift"):
                mfrmpy.save_project(output / "annual_drift.mfrm", linked)
                changed = data.assign(Mode=data.RatingYear.map({"A": "live", "B": "video"}))
                control = mfrmpy.analyze(changed, **SETTINGS, facets=["WorkYear", "RaterYear", "Mode"],
                                         noncenter_facet="RaterYear", positive_facets=["WorkYear"])
                control = mfrmpy.with_anchor_sensitivity(control, rater="RaterYear", difference=TARGET,
                    intervals=True, occasion="RatingYear", scenarios={"Previous-year anchors": dict(
                        anchors=[item for item in fixed if item["Level"].startswith("A-")])})
                assert mfrmpy.tables(control, "Anchor sensitivity")[0][1].AssignmentSupport.eq("unsupported").all()
                mfrmpy.save_project(output / "annual_mode_confounded.mfrm", control)
        print(f"Replicate {replicate}: six cases matched", flush=True)
    def forbidden(*args, **kwargs):
        raise AssertionError("Reporting must not refit or diagnose")
    engine.mfrm_estimate = engine.mfrm_diagnostics = forbidden
    for name in ("annual_drift", "annual_mode_confounded"):
        linked = mfrmpy.load_project(output / (name + ".mfrm"))
        mfrmpy.export_out(linked, output / (name + ".out"), kind="anchor_sensitivity")
        mfrmpy.export_bundle(linked, output / (name + ".zip"), kind="anchor_sensitivity", plots=["anchor_sensitivity"])
        (output / (name + ".html")).write_text(mfrmpy.report_html(linked), encoding="utf-8")
    record = dict(version=mfrmpy.__version__, cases=records,
                  maximum_errors={key: max(row["errors"][key] for row in records) for key in records[0]["errors"]},
                  source_sha256={name: hashlib.sha256((evidence / name).read_bytes()).hexdigest()
                                 for name in ("m5b_estimated_anchors.csv.gz", "m5b_annual_drift.csv.gz")},
                  scope="18 numerical API comparisons, not a new coverage experiment. Previous 500-replicate conditions remain distinct from API availability.")
    (output / "comparison.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record["maximum_errors"], indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    run(parser.parse_args().output)
