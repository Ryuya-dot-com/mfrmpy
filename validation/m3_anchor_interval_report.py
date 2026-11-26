"""Reproduce the anchor-interval output against previously simulated results."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import mfrmpy
from validation.m3_facet_sd import PROFILES, calibration, validation_designs


PROFILE = "person_high"
EVIDENCE = Path(__file__).with_name("evidence") / "m3_facet_sd.csv.gz"
SETTINGS = dict(person="Person", facets=["Item", "Rater"], score="Score", method="MML",
                rating_min=0, rating_max=3, quad_points=61, reltol=1e-12)


def checked_report(design):
    data = next(data for profile, assignment, *_, data in validation_designs(1)
                if (profile, assignment) == (PROFILE, design))
    values, covariance = calibration(1, *PROFILES[PROFILE][1:])
    labels = ["R1", "R2"]
    fixed = [dict(Facet="Rater", Level=level, Anchor=float(value), Source="Independent simulated calibration")
             for level, value in zip(labels, values, strict=True)]
    reference = mfrmpy.analyze(data, **SETTINGS)
    independent = dict(anchors=fixed, calibration=dict(
        kind="independent", work_ids=[f"C{i:03d}" for i in range(60)],
        covariance=pd.DataFrame(covariance, index=labels, columns=labels),
    ))
    analysis = mfrmpy.with_anchor_sensitivity(
        reference, rater="Rater", difference=("R3", "R1"), intervals=True, important_difference=2.,
        scenarios={
            "Independent, SD 1": independent,
            "Independent, SD 1.5": dict(independent, population_prior_sd=1.5),
            "No anchors, SD 1.5": dict(population_prior_sd=1.5),
        },
    )
    expected = pd.read_csv(EVIDENCE).query(
        'replicate==1 and profile==@PROFILE and design==@design'
    ).set_index(["model", "assumed_person_sd"])
    _, table, _ = mfrmpy.tables(analysis, "Anchor sensitivity intervals")[0]
    for row in table.itertuples(index=False):
        model = "free" if row.Basis == "no_anchors" else "estimated"
        comparison = expected.loc[(model, row.PersonSD)]
        se = comparison.propagated_se if row.Basis == "calibration_propagated" else comparison.se
        np.testing.assert_allclose([row.Estimate, row.SE], [comparison.estimate, se], rtol=0, atol=5e-5)
    return analysis


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project", type=Path, help="Output .mfrm for the 24-person double-scoring example")
    parser.add_argument("--read", action="store_true")
    args = parser.parse_args()
    if not args.read:
        checked_report("balanced6")
        mfrmpy.save_project(args.project, checked_report("balanced24"))
    restored = mfrmpy.load_project(args.project)
    _, table, _ = mfrmpy.tables(restored, "Anchor sensitivity intervals")[0]
    print(table[["Scenario", "Basis", "Estimate", "SE", "Lower", "Upper",
                 "IncludesImportantDifference", "Status"]].round(6).to_string(index=False))


if __name__ == "__main__":
    main()
