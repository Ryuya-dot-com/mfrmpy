"""Build or reopen two group DFF examples and trace their shared ratings."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import mfrmpy
from validation.m5a_group_dff import AUDIT
from validation.m5a_public_example import DFF, fit_example
from validation.m5a_school_variance import SCHOOL_SD, data_at


REPLICATE = 13
EFFECT = "r3_stricter"
EVIDENCE = Path(__file__).with_name("evidence")
SUMMARY = "Prespecified group DFF contrast"
SHARED = "Group DFF shared works"
ROSTER = ["School", "Person", "Item", "Rater"]
ESTIMATES = ["Estimate", "SE", "Lower", "Upper", "WaldP"]
SAVED_ESTIMATES = ["estimate", "se", "lower", "upper", "p"]


def saved_row(file):
    rows = pd.read_csv(EVIDENCE / file)
    matching = rows.loc[rows.replicate.eq(REPLICATE) & rows.sd_B.eq(SCHOOL_SD[1]) & rows.effect.eq(EFFECT)]
    assert len(matching) == 1 and matching.reason.isna().all()
    return matching.iloc[0]


def examples():
    ratings = data_at(REPLICATE, SCHOOL_SD, effect=EFFECT)
    b = ratings.loc[ratings.School.eq("B")]
    first_rater = b.groupby("Person").Rater.transform("min")
    missing = ratings.drop(b.index[b.Rater.ne(first_rater)]).reset_index(drop=True)
    assert len(ratings) == 480 and len(missing) == 432
    return ratings, missing


def shared_scores(ratings, overlap):
    work_ids = [(link.Group, person) for link in overlap.itertuples() for person in json.loads(link.WorkIDs)]
    works = pd.DataFrame(work_ids, columns=["School", "Person"]).drop_duplicates()
    target_ratings = ratings.loc[ratings.Rater.isin(DFF["difference"])]
    matched = target_ratings.merge(works, on=["School", "Person"], validate="many_to_one")
    return matched.pivot(index=["School", "Person", "Item"], columns="Rater", values="Score").dropna().reset_index()


def compare_before_after(before, after):
    pd.testing.assert_frame_equal(before["input_data"], after["input_data"])
    for old, new in zip(mfrmpy.tables(before, "rater_dff"), mfrmpy.tables(after, "rater_dff"), strict=True):
        assert (old[0], old[2]) == (new[0], new[2])
        pd.testing.assert_frame_equal(old[1], new[1])


def run(directory, read, exports):
    ratings, missing = examples()
    planned = ratings[ROSTER]
    if not read:
        directory.mkdir(parents=True, exist_ok=True)
    cml = saved_row("m5a_school_variance_nonzero_cml.csv.gz")
    assert cml.lower > 0

    for name, raw, expected, overlaps in (
        ("m5a_dff_sd_difference", ratings, "observed", [20, 20]),
        ("m5a_dff_missing_shared_works", missing, "common_location_only", [0, 20]),
    ):
        path = directory / f"{name}.mfrm"
        if not read:
            audit = mfrmpy.check_rater_contrast(raw, **AUDIT, planned_assignments=planned)
            assert audit["status"] == expected
            analysis = mfrmpy.with_rater_dff(fit_example(raw), **DFF, planned_assignments=planned)
            mfrmpy.save_project(path, analysis)
        restored = mfrmpy.load_project(path)
        if not read:
            compare_before_after(analysis, restored)
        original = restored["input_data"][list(raw.columns)]
        pd.testing.assert_frame_equal(raw, original)
        tables = {title: frame for title, frame, _ in mfrmpy.tables(restored, "rater_dff")}
        result = tables[SUMMARY].iloc[0]
        assert result.AssignmentSupport == expected
        assert tables[SHARED].SharedWorks.tolist() == overlaps
        assert not result.CoverageQualified and not result.SupportsFormalInference
        if expected == "observed":
            reference = saved_row("m5a_school_variance_nonzero.csv.gz")
            np.testing.assert_allclose(result[ESTIMATES].to_numpy(dtype=float),
                                       reference[SAVED_ESTIMATES].to_numpy(dtype=float),
                                       atol=3e-6, rtol=0)
            assert result.Status == "model_approximation" and result.Lower < 0 < result.Upper
        else:
            assert result.Status == "withheld" and result.Reason and result[ESTIMATES].isna().all()
            assert tables["Group DFF assignment discrepancies"].RestoresContrastOnItsOwn.sum() == 40
        scores = shared_scores(original, tables[SHARED])
        assert len(scores) == 2 * sum(overlaps)

        print(name)
        print(tables[SUMMARY][["AssignmentSupport", *ESTIMATES, "Status", "Reason"]].to_string(index=False))
        print(scores.head(4).to_string(index=False))
        if exports is not None:
            out = exports / name
            out.mkdir(parents=True, exist_ok=True)
            mfrmpy.export_csv(restored, out / "tables")
            mfrmpy.plot(restored, kind="rater_dff").savefig(out / "dff.png", dpi=140)
            (out / "report.html").write_text(mfrmpy.report_html(restored), encoding="utf-8")
            mfrmpy.export_out(restored, out / "report.out", kind="rater_dff")
            mfrmpy.export_bundle(restored, out / "report.zip", kind="rater_dff", plots=["rater_dff"])
            scores.to_csv(out / "shared-scores.csv", index=False)

    print(f"mfrmr CML, saved result on the same 480 ratings: {cml.estimate:.6f} "
          f"(95% interval {cml.lower:.6f}, {cml.upper:.6f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project_dir", type=Path)
    parser.add_argument("--read", action="store_true", help="Reopen saved projects without fitting")
    parser.add_argument("--exports", type=Path, help="Write the figure, HTML and CSV files for each project")
    args = parser.parse_args()
    run(args.project_dir, args.read, args.exports)
