"""Annual contrasts with repeated calibration estimates and retained provenance."""
from copy import deepcopy
from pathlib import Path
import json
import zipfile

import numpy as np
import pandas as pd
import pytest

import mfrmpy
from mfrm_app import project


@pytest.fixture(scope="module")
def annual():
    # M5b repeat6 / anchors_drift / replicate 44; synthetic retained design.
    data = pd.read_csv(Path(__file__).with_name("annual_drift.csv"))
    settings = dict(person="Recording", score="Score", rating_min=1, rating_max=5,
                    method="MML", quad_points=61, reltol=1e-12)
    base = mfrmpy.analyze(data, **settings, facets=["WorkYear", "RaterYear"],
                         noncenter_facet="RaterYear", positive_facets=["WorkYear"])
    source = mfrmpy.analyze(data.loc[data.RatingYear.eq("A")], **settings, facets=["Rater"], noncenter_facet="Rater")
    values = dict(zip(source["result"]["config"]["facet_levels"]["Rater"], source["result"]["params"]["facets"]["Rater"]))
    fixed = [dict(Facet="RaterYear", Level=f"{year}-{rater}", Anchor=float(values[rater]), Source="Previous year")
             for year in ("A", "B") for rater in ("R1", "R2")]
    calibration = dict(kind="shared", analysis=source, facet="Rater", rating_id="Event",
                       level_map={f"{year}-{rater}": rater for year in ("A", "B") for rater in ("R1", "R2", "R3")})
    scenario = dict(anchors=fixed, calibration=calibration)
    args = dict(rater="RaterYear", difference=("B-R3", "A-R3"), intervals=True,
                occasion="RatingYear", important_difference=.5)
    linked = mfrmpy.with_anchor_sensitivity(base, **args, scenarios={"Previous values": scenario})
    return base, source, scenario, args, linked, settings


def test_annual_contrast_matches_independent_saved_calculation(annual, tmp_path, monkeypatch):
    base, source, scenario, args, linked, _ = annual
    _, table, _ = mfrmpy.tables(linked, "Anchor sensitivity intervals")[0]
    rows = table.set_index("Basis")
    np.testing.assert_allclose(rows.Estimate, [.203173, -.7029019585632107, -.7029019585632107], atol=5e-6, rtol=0)
    np.testing.assert_allclose(rows.SE, [.492023, .333616556491211, .31439649784347057], atol=2e-6, rtol=0)
    shared = rows.loc["shared_calibration"]
    np.testing.assert_allclose([shared.CalibrationVariance, shared.ReceivingVariance, shared.SharedCrossTerm],
                              [.00012019444991346627, .0986685965084029, .000056366897923034966], atol=2e-7, rtol=0)
    assert shared.CalibrationWorks == shared.OverlapWorks == 60 and shared.ReusedRatings == 120
    assert shared.SE**2 == pytest.approx(shared.CalibrationVariance + shared.ReceivingVariance + shared.SharedCrossTerm)
    assert json.loads(shared.CalibrationLevelMap)["A-R1"] == json.loads(shared.CalibrationLevelMap)["B-R1"] == "R1"
    assert table.Status.eq("model_approximation").all() and not table.CoverageQualified.any()
    overlap = mfrmpy.tables(linked, "Anchor sensitivity occasion overlap")[0][1].iloc[0]
    assert overlap.SharedWorks == 6 and overlap.FirstOccasion == "B" and overlap.SecondOccasion == "A"
    assert len(json.loads(overlap.WorkIDs)) == 6
    inverted = mfrmpy.with_anchor_sensitivity(base, **(args | dict(difference=("A-R3", "B-R3"))), scenarios={"Previous values": scenario})
    reverse = mfrmpy.tables(inverted, "Anchor sensitivity intervals")[0][1]
    np.testing.assert_allclose(reverse.Estimate, -table.Estimate, atol=1e-10)
    np.testing.assert_allclose(reverse.SE, table.SE, atol=1e-10)
    from mfrm_app import engine
    monkeypatch.setattr(engine, "mfrm_estimate", lambda *a, **k: pytest.fail("Refit on export"))
    path = tmp_path / "annual.mfrm"
    mfrmpy.save_project(path, linked)
    loaded = mfrmpy.load_project(path)
    pd.testing.assert_frame_equal(table, mfrmpy.tables(loaded, "Anchor sensitivity intervals")[0][1])
    text = mfrmpy.export_out(loaded, tmp_path / "annual.out", kind="anchor_sensitivity").read_text()
    assert "shared_calibration" in text and "occasion overlap" in text and "CalibrationLevelMap" in text
    mfrmpy.export_bundle(loaded, tmp_path / "annual.zip", kind="anchor_sensitivity", plots=[])
    assert loaded["content_fingerprint"] == linked["content_fingerprint"]


def test_annual_mapping_and_event_validation_precede_fitting(annual, monkeypatch):
    base, source, scenario, args, _, _ = annual
    from mfrm_app import engine
    monkeypatch.setattr(engine, "mfrm_estimate", lambda *a, **k: pytest.fail("Validation should precede fitting"))
    for mapping in ({"A-R1": "R1"}, {key: "absent" for key in scenario["calibration"]["level_map"]}):
        with pytest.raises(ValueError, match="level_map"):
            mfrmpy.with_anchor_sensitivity(base, **args, scenarios={"Invalid": dict(scenario, calibration=scenario["calibration"] | dict(level_map=mapping))})
    # Even identical works, rater and score cannot make a new occasion the old rating event.
    wrong_year = deepcopy(source)
    wrong_year["input_data"]["RatingYear"] = "B"
    wrong_year["content_fingerprint"] = project.content_fingerprint(wrong_year)
    with pytest.raises(ValueError, match="same rating ID"):
        mfrmpy.with_anchor_sensitivity(base, **args, scenarios={"Wrong events": dict(scenario, calibration=scenario["calibration"] | dict(analysis=wrong_year))})
    with pytest.raises(ValueError, match="occasion"):
        mfrmpy.with_anchor_sensitivity(base, **(args | dict(occasion="absent")), scenarios={"Previous values": scenario})
    with pytest.raises(ValueError, match="distinct occasions"):
        mfrmpy.with_anchor_sensitivity(base, **(args | dict(difference=("A-R3", "A-R1"))), scenarios={"Previous values": scenario})


def test_annual_retained_mapping_and_overlap_cannot_disagree(annual):
    linked = annual[4]
    for field in ("mapping", "overlap", "cross_term"):
        changed = deepcopy(linked)
        entry = changed["anchor_sensitivity"]
        if field == "mapping":
            entry["scenarios"][1]["calibration"]["source"]["level_map"]["A-R1"] = "R2"
        elif field == "overlap":
            entry["occasion"]["shared_works"].pop()
        else:
            entry["scenarios"][1]["interval"]["shared"]["cross_term"] += .01
        changed["content_fingerprint"] = project.content_fingerprint(changed)
        with pytest.raises(mfrmpy.ProjectError):
            project.validate_analysis(changed)


def test_annual_mode_confounding_is_withheld(annual, tmp_path):
    base, _, scenario, args, _, settings = annual
    data = base["input_data"].assign(Mode=lambda d: d.RatingYear.map({"A": "live", "B": "video"}))
    fit = mfrmpy.analyze(data, **settings, facets=["WorkYear", "RaterYear", "Mode"],
                        noncenter_facet="RaterYear", positive_facets=["WorkYear"])
    previous_only = [item for item in scenario["anchors"] if item["Level"].startswith("A-")]
    held = mfrmpy.with_anchor_sensitivity(fit, **args, scenarios={"Previous year only": dict(anchors=previous_only)})
    table = mfrmpy.tables(held, "Anchor sensitivity")[0][1]
    assert table.AssignmentSupport.eq("unsupported").all() and table.Estimate.isna().all()
    path = tmp_path / "confounded.mfrm"
    mfrmpy.save_project(path, held)
    assert "unidentified" in mfrmpy.report_text(mfrmpy.load_project(path), kind="anchor_sensitivity")


@pytest.mark.figures
def test_annual_figure_and_bundle_use_retained_intervals(annual, tmp_path, monkeypatch):
    linked = annual[4]
    from mfrm_app import engine
    monkeypatch.setattr(engine, "mfrm_estimate", lambda *a, **k: pytest.fail("Refit on drawing"))
    fig = mfrmpy.plot(linked, "anchor_sensitivity")
    table = mfrmpy.tables(linked, "Anchor sensitivity intervals")[0][1]
    bars = [item for item in fig.axes[0].containers if type(item).__name__ == "ErrorbarContainer"]
    assert len(bars) == len(table) == 3
    for artist, row in zip(bars, table.itertuples(), strict=True):
        assert artist.lines[0].get_xdata()[0] == pytest.approx(row.Estimate)
        np.testing.assert_allclose(artist.lines[2][0].get_segments()[0][:, 0], [row.Lower, row.Upper])
    assert "CoverageQualified=False" in " ".join(fig._mfrm_caption["notes"])
    assert "important difference = 0.5 logits" in " ".join(fig._mfrm_caption["notes"])
    fig.clear()
    path = mfrmpy.export_bundle(linked, tmp_path / "annual.zip", kind="anchor_sensitivity", plots=["anchor_sensitivity"])
    with zipfile.ZipFile(path) as archive:
        assert archive.read("plots/01_anchor_sensitivity.png").startswith(b"\x89PNG")
        assert b"shared_calibration" in archive.read("report.out")
    html = mfrmpy.report_html(linked)
    assert "Anchor sensitivity occasion overlap" in html and "shared_calibration" in html
