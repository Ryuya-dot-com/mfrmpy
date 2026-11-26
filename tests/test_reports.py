"""Retained predictions and portable reports, without changing the fitted model."""
import hashlib
import io
import json
import subprocess
import sys
import zipfile

import numpy as np
import pandas as pd
import pytest

import mfrmpy


@pytest.fixture(scope="module", params=[("RSM", "JMLE"), ("PCM", "JMLE"), ("RSM", "MML")])
def analysis(request):
    rng = np.random.default_rng(110)
    data = pd.DataFrame([
        (f"受験者{p:03d}", f"R{r}", f"T{t}", int(rng.choice([1, 2, 3], p=[.2, .3, .5])))
        for p in range(35) for r in range(3) for t in range(3)
    ], columns=["Person", "Rater", "Task", "Score"])
    return mfrmpy.analyze(data, person="Person", facets=["Rater", "Task"], score="Score",
                         rating_min=1, rating_max=3, model=request.param[0], method=request.param[1],
                         step_facet="Task" if request.param[0] == "PCM" else None,
                         positive_facets=["Rater"], maxit=100)


def test_fair_average_matches_independent_category_probabilities(analysis):
    result = analysis["result"]
    config, params = result["config"], result["params"]
    frame = mfrmpy.tables(analysis, "fair_average")[0][1]
    assert not frame.SupportsFormalInference.any()
    assert frame.FairZ.notna().all()
    for row in frame.itertuples():
        location = (result["facets"]["person"].set_index("Person").loc[row.Level, "Estimate"]
                    if row.Facet == "Person" else params["facets"][row.Facet][config["facet_levels"][row.Facet].index(row.Level)])
        other_mean = sum(config["facet_signs"][facet] * np.mean(values)
                         for facet, values in params["facets"].items() if facet != row.Facet)
        if row.Facet != "Person":
            location *= config["facet_signs"][row.Facet]
            other_mean += result["facets"]["person"].Estimate.mean()
        if config["model"] == "RSM":
            steps = params["steps"]
        elif row.Facet == "Task":
            steps = params["steps_mat"][config["facet_levels"]["Task"].index(row.Level)]
        else:
            steps = np.mean(params["steps_mat"], axis=0)
        for metric, eta in (("FairM", location + other_mean), ("FairZ", location)):
            weights = np.array([1., np.exp(eta - steps[0]), np.exp(2 * eta - sum(steps))])
            expected = np.dot([1., 2., 3.], weights / weights.sum())
            assert getattr(row, metric) == pytest.approx(expected, abs=1e-12)
    assert (frame.loc[frame.Facet.eq("Rater"), "FairM"] != frame.loc[frame.Facet.eq("Rater"), "FairZ"]).any()


def test_reports_roundtrip_without_refitting_or_optional_dependencies(analysis, tmp_path, monkeypatch):
    from mfrmpy._core import engine
    path = tmp_path / "saved.mfrm"
    mfrmpy.save_project(path, analysis)
    def forbidden(*args, **kwargs):
        pytest.fail("Export must reuse retained results")
    monkeypatch.setattr(engine, "mfrm_estimate", forbidden)
    monkeypatch.setattr(engine, "mfrm_diagnostics", forbidden)
    restored = mfrmpy.load_project(path)
    text = mfrmpy.report_text(restored, kind="fair_average")
    assert "受験者034" in text and "FairZStatus" in text and "No fair-score SE" in " ".join(text.split())
    output = mfrmpy.export_out(restored, tmp_path / "結果.out.txt", kind="fair_average")
    assert output.read_text(encoding="utf-8") == text
    bundle = mfrmpy.export_bundle(restored, tmp_path / "結果.zip", kind="fair_average", plots=[])
    with zipfile.ZipFile(bundle) as archive:
        assert archive.read("report.out").decode("utf-8") == text
        manifest = json.loads(archive.read("manifest.json"))
        assert manifest["AnalysisID"] == analysis["analysis_id"]
        assert {r["File"] for r in manifest["Files"]} == set(archive.namelist()) - {"manifest.json"}
        for record in manifest["Files"]:
            assert hashlib.sha256(archive.read(record["File"])).hexdigest() == record["SHA256"]
        csv_manifest = pd.read_csv(io.BytesIO(archive.read("tables/00_manifest.csv")))
        for row in csv_manifest.itertuples():
            assert hashlib.sha256(archive.read("tables/" + row.File)).hexdigest() == row.SHA256
    script = """
import importlib.abc, sys
class NoPlots(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'matplotlib', 'streamlit', 'tkinter'}:
            raise AssertionError(fullname)
sys.meta_path.insert(0, NoPlots())
import mfrmpy
analysis = mfrmpy.load_project(sys.argv[1])
mfrmpy.export_out(analysis, sys.argv[2])
mfrmpy.export_bundle(analysis, sys.argv[3], plots=[])
"""
    subprocess.run([sys.executable, "-I", "-c", script, str(path), str(output), str(bundle)], check=True)


def test_fair_average_calibration_hold_and_long_text(analysis, tmp_path, monkeypatch):
    held = mfrmpy.analyze(analysis["input_data"], **(analysis["fit_settings"] | {"maxit": 1}))
    frame = mfrmpy.tables(held, "fair_average")[0][1]
    assert frame.FairM.isna().all() and frame.FairZ.isna().all()
    assert frame.FairMStatus.eq("withheld_calibration").all()
    assert "stat.estimation.optimizer_termination_unverified" in mfrmpy.report_text(held)
    from mfrmpy._reports import render_text
    label = "日本語" * 100
    table = pd.DataFrame({"ID": [label, "last\nrow"], **{f"col{i}": [i, i] for i in range(30)}})
    with pd.option_context("display.max_rows", 1, "display.max_columns", 1, "display.max_colwidth", 1):
        text = render_text(analysis, [("test", table, label)])
    assert label in text and "col29" in text and "last\\nrow" in text
    target = tmp_path / "retained.out"
    target.write_text("previous")
    with pytest.raises(ValueError):
        mfrmpy.export_out(analysis, target, kind="no such table")
    assert target.read_text() == "previous"
    target = tmp_path / "retained.zip"
    target.write_bytes(b"previous")
    def fail(*args, **kwargs):
        raise RuntimeError("plot failure")
    monkeypatch.setattr(mfrmpy, "plot", fail)
    with pytest.raises(RuntimeError, match="plot failure"):
        mfrmpy.export_bundle(analysis, target, plots=["fair_average"])
    assert target.read_bytes() == b"previous"
    assert not list(tmp_path.glob(".mfrmpy-report-*"))


def test_pending_diagnostics_and_gd_report(tmp_path):
    data = pd.DataFrame([(p, r, (p + r) % 3) for p in range(20) for r in range(4)], columns=["P", "R", "Y"])
    fitted = mfrmpy.fit(data, person="P", facets=["R"], score="Y", rating_min=0, rating_max=2)
    text = mfrmpy.report_text(fitted)
    assert "Computed outputs: none" in text and "Fair Average Scores" not in text
    assert not mfrmpy.tables(fitted, "fair_average")
    gd = mfrmpy.analyze(data, workflow="gd", gd_settings=dict(object_facet="P", random_facets=["R"], score_col="Y"))
    assert "G/D: observed_coefficients" in mfrmpy.report_text(gd)
    assert mfrmpy.export_bundle(gd, tmp_path / "gd.zip", plots=[]).is_file()


@pytest.mark.figures
def test_zip_figures_have_explicit_scope_and_match_fair_table(analysis, tmp_path):
    options = [{"kind": "fair_average", "facet": "Rater", "statistic": "FairZ"}]
    if analysis["result"]["config"]["model"] == "PCM":
        options.append({"kind": "category", "level": "T1"})
    path = mfrmpy.export_bundle(analysis, tmp_path / "plots.zip", plots=options)
    with zipfile.ZipFile(path) as archive:
        assert archive.read("plots/01_fair_average.png").startswith(b"\x89PNG\r\n\x1a\n")
        assert b"<svg" in archive.read("plots/01_fair_average.svg")
        manifest = json.loads(archive.read("manifest.json"))
        figures = [entry for entry in manifest["Files"] if entry["Type"] == "figure"]
        assert len(figures) == 2 * len(options)
        assert figures[0]["Options"] == options[0] and figures[0]["Caption"]
    fig = mfrmpy.plot(analysis, **options[0])
    frame = mfrmpy.tables(analysis, "fair_average")[0][1].query("Facet == 'Rater'")
    np.testing.assert_allclose(fig._mfrm_hover_points[0][0].get_offsets()[:, 1], frame.FairZ)
    fig.clear()
    if analysis["result"]["config"]["model"] == "PCM":
        from mfrmpy._core.desktop_outputs import figure_choices
        mfrmpy.export_bundle(analysis, path, kind="fair_average")
        with zipfile.ZipFile(path) as archive:
            manifest = json.loads(archive.read("manifest.json"))
            pngs = [entry for entry in manifest["Files"] if entry["File"].endswith(".png")]
            assert [entry["Options"] for entry in pngs] == [options for _, options in figure_choices(analysis)]
