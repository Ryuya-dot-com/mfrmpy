"""Save and report one prespecified group DFF contrast."""
import argparse
from pathlib import Path

import pandas as pd

import mfrmpy
from validation.m5a_direct_overlap import DESIGNS, ratings_at
from validation.m5a_group_dff import AUDIT


DFF = {key: AUDIT[key] for key in ("rater", "group", "difference", "group_difference")}
DFF["cell_facet"] = "SchoolRater"


def fit_example(ratings):
    prepared = mfrmpy.prepare_rater_dff_data(ratings, rater="Rater", group="School", cell_facet="SchoolRater")
    return mfrmpy.analyze(
        prepared,
        person="Person", facets=["Item", "SchoolRater"], score="Score",
        rating_min=1, rating_max=5, method="MML", model="RSM", noncenter_facet="SchoolRater",
        population_prior_sd=1., quad_points=61, reltol=1e-12,
        interval_requests=[dict(facet="SchoolRater")],
        bias_requests=[dict(facet_a="SchoolRater", facet_b="Item")],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path)
    args = parser.parse_args()
    ratings = next(rows for design, effect, rows in ratings_at(1, DESIGNS)
                   if (design, effect) == ("direct20", "r3_stricter"))
    audit = mfrmpy.check_rater_contrast(ratings, **AUDIT)
    if audit["status"] != "observed":
        raise SystemExit(f"Assignment support: {audit['status']}")
    analysis = mfrmpy.with_rater_dff(fit_example(ratings), **DFF)
    tables = mfrmpy.tables(analysis, "rater_dff")
    for title, frame, note in tables:
        print(title)
        print(frame.to_string(index=False))
        print(note)
    if args.project is not None:
        mfrmpy.save_project(args.project, analysis)
        restored = mfrmpy.load_project(args.project)
        for (_, before, _), (_, after, _) in zip(tables, mfrmpy.tables(restored, "rater_dff"), strict=True):
            pd.testing.assert_frame_equal(before, after)


if __name__ == "__main__":
    main()
