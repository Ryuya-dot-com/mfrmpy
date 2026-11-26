"""Public Python interface to the shared MFRM application calculations.

Use :func:`check_design` and :func:`check_rater_contrast` for optimization-free preflight checks, :func:`analyze`
for RSM/PCM or observed-score G/D studies, :func:`summary`, :func:`tables` and
:func:`export_csv` for qualified results, and :func:`save_project` /
:func:`load_project` for portable .mfrm archives. Raw retained results do not
override output limits.
"""
from contextlib import contextmanager
import hashlib
from importlib.resources import files
import os
import tempfile
import warnings
from pathlib import Path

import pandas as pd

from mfrmpy._core import anchor_sensitivity, basic_analysis, desktop_analysis, exports, project, rater_dff
from mfrmpy._core.anchor_sensitivity import with_anchor_sensitivity
from mfrmpy._core.desktop_analysis import check_design, run_analysis as analyze
from mfrmpy._core.project import ProjectError, load_project, save_project
from mfrmpy._core.rater_contrast import check_rater_contrast
from mfrmpy._core.rater_dff import prepare_rater_dff_data, with_rater_dff

from ._category_design import category_design

__version__ = "0.2.3.dev5"

__all__ = [
    "__version__", "analyze", "fit", "diagnose", "check_design", "check_rater_contrast", "category_design", "summary", "tables", "export_csv",
    "load_example", "save_project", "load_project", "ProjectError", "plot", "report_html",
    "prepare_rater_dff_data", "with_rater_dff", "with_anchor_sensitivity",
    "report_text", "export_out", "export_bundle",
]


def fit(data, **settings):
    """Fit RSM/PCM and return a saveable result before observation diagnostics.

    Accepts the MFRM fit settings of analyze(), including column aliases.
    Use diagnose() to add selected outputs without refitting.
    """
    return desktop_analysis.run_analysis(data, **settings, outputs=[])


def diagnose(analysis, *, outputs=("measures", "agreement", "scorefile"),
             interval_requests=None, bias_requests=None, joint_settings=None,
             gd_settings=None, progress=None):
    """Return an analysis with selected outputs added; keep the source intact.

    outputs: measures (fit, reliability and fair averages), agreement,
    scorefile, or pca. Already computed sections are reused.
    """
    completed = basic_analysis.complete_analysis(analysis, outputs=outputs, progress=progress)
    return desktop_analysis.complete_requests(
        completed, interval_requests=interval_requests, bias_requests=bias_requests,
        joint_settings=joint_settings, gd_settings=gd_settings, progress=progress)

_TABLE_ALIASES = {
    "estimates": "measures",
    "fit": "measures",
    "category": "categories",
    "step": "steps",
    "thresholds": "steps",
    "linking": "connectivity",
    "anchors": "connectivity",
    "separation": "reliability",
    "interval": "intervals",
    "joint_interaction": "joint",
}

_VALID_TABLE_KINDS = frozenset({
    "design", "measures", "categories", "steps", "connectivity", "reliability",
    "agreement", "agreement_summary", "pairwise_agreement", "rescaled",
    "bias", "intervals", "joint", "rater_dff", "anchor_sensitivity", "gd", "gd_observed", "gd_selected",
    "gd_variance", "gd_d_study", "gd_metadata", "pca",
    "gd_covariance", "gd_component_diagnostics", "fair_average",
    "gd_estimation", "gd_component_support", "gd_kernel_gram", "gd_excluded_rows",
    *_TABLE_ALIASES,
})


def _canonical_table_kind(title: str) -> str:
    if title in {"G/D: " + s for s in ("estimation", "component_support", "kernel_gram", "excluded_rows")}:
        return "gd_" + title[5:]
    if title == "Analysis Design and Retained Ratings":
        return "design"
    if title == "Measures and Fit":
        return "measures"
    if title == "Fair Average Scores":
        return "fair_average"
    if title == "Observed Category Use":
        return "categories"
    if title == "Step Thresholds":
        return "steps"
    if title == "Connectivity and Scale Linking":
        return "connectivity"
    if title == "Separation and Reliability by Facet":
        return "reliability"
    if title == "Inter-rater Agreement and Krippendorff's Alpha":
        return "agreement_summary"
    if title.startswith("Pairwise Agreement:"):
        return "pairwise_agreement"
    if title == "Rescaled Measures (Umean / Uscale)":
        return "rescaled"
    if title.startswith("Bias:"):
        return "bias"
    if title.startswith("Specified intervals:"):
        return "intervals"
    if title.startswith("Joint interaction:") or title == "Joint model: main effects and steps":
        return "joint"
    if title in (rater_dff.TITLE, rater_dff.OVERLAP, rater_dff.SCORES, rater_dff.PLAN, rater_dff.GAPS):
        return "rater_dff"
    if title in (anchor_sensitivity.TITLE, anchor_sensitivity.CONSTRAINTS, anchor_sensitivity.WORKS,
                 anchor_sensitivity.INTERVALS, anchor_sensitivity.OCCASIONS, anchor_sensitivity.LEVELS):
        return "anchor_sensitivity"
    if title == "G/D: observed_coefficients":
        return "gd_observed"
    if title == "G/D: selected_d_study":
        return "gd_selected"
    if title == "G/D: variance_components":
        return "gd_variance"
    if title == "G/D: covariance_components":
        return "gd_covariance"
    if title == "G/D: component_diagnostics":
        return "gd_component_diagnostics"
    if title == "G/D: d_study":
        return "gd_d_study"
    if title == "G/D: metadata":
        return "gd_metadata"
    if title.startswith("Residual PCA:"):
        return "pca"
    return "other"


def _available_plot_kinds(analysis) -> list[str]:
    kinds: list[str] = []
    if "result" in analysis:
        kinds.extend([
            "wright", "pathway", "category", "facet", "rater_severity", "step",
            "bubble", "expected", "information", "fair_average", "unexpected", "residual_qq",
        ])
    if analysis.get("bias_analyses"):
        kinds.extend(["bias", "bias_ranked"])
    if analysis.get("interval_analyses"):
        kinds.append("interval")
    if "gd_analysis" in analysis:
        kinds.append("gd")
    if "joint_analysis" in analysis:
        kinds.append("joint")
    if "rater_dff" in analysis:
        kinds.append("rater_dff")
    if "anchor_sensitivity" in analysis:
        kinds.append("anchor_sensitivity")
    if (analysis.get("diagnostics") or {}).get("pca_enabled"):
        kinds.extend(["pca", "pca_loadings"])
    if "computed_outputs" in analysis:
        if "measures" not in analysis["computed_outputs"]:
            kinds = [k for k in kinds if k not in {"pathway", "facet", "rater_severity", "bubble", "fair_average"}]
        if "obs" not in analysis.get("diagnostics", {}):
            kinds = [k for k in kinds if k not in {"unexpected", "residual_qq"}]
    return kinds


def tables(analysis, kind=None):
    """Return (title, DataFrame, interpretation note) tuples without refitting.

    Pass ``kind`` (e.g. ``"measures"``, ``"steps"``, ``"reliability"``,
    ``"agreement"``, ``"categories"``, ``"design"``, ``"bias"``, ``"intervals"``,
    ``"joint"``, ``"rater_dff"``, ``"anchor_sensitivity"``, ``"gd"``, ``"pca"``) or an iterable of kinds to filter specific tables.
    """
    if kind is None:
        return desktop_analysis.current_tables(analysis)
    requested = [kind] if isinstance(kind, str) else list(kind)
    if not requested or any(not isinstance(k, str) or not k.strip() for k in requested):
        raise ValueError("kind must be a nonblank table kind string or iterable of strings")
    target_kinds: set[str] = set()
    target_titles: set[str] = set()
    for raw_k in requested:
        norm_k = raw_k.strip().lower()
        mapped = _TABLE_ALIASES.get(norm_k, norm_k)
        if mapped == "agreement":
            target_kinds.update({"agreement_summary", "pairwise_agreement"})
        elif mapped == "gd":
            target_kinds.update({"gd_observed", "gd_selected", "gd_variance", "gd_d_study", "gd_metadata",
                                 "gd_covariance", "gd_component_diagnostics", "gd_estimation", "gd_component_support",
                                 "gd_kernel_gram", "gd_excluded_rows"})
        elif mapped in _VALID_TABLE_KINDS:
            target_kinds.add(mapped)
        else:
            target_titles.add(norm_k)
    selected = desktop_analysis.current_tables(
        analysis, include=lambda title: _canonical_table_kind(title) in target_kinds or title.lower() in target_titles)
    unknown = target_titles - {title.lower() for title, _, _ in selected}
    if unknown:
        raise ValueError(f"Unknown table kind: {sorted(unknown)[0]!r}")
    return selected


def summary(analysis) -> pd.DataFrame:
    """Return a concise one-screen DataFrame of design, qualification, and available outputs without refitting."""
    if reason := project.input_refit_reason(analysis):
        project.validate_analysis(analysis)
        return pd.DataFrame([("AnalysisID", analysis["analysis_id"]), ("RunQualificationState", "refit_required"),
                             ("Reason", reason), ("SupportsFormalInference", "False")], columns=["Field", "Value"])
    if "computed_outputs" in analysis:
        project.validate_analysis(analysis)
        available = ["design", "categories", "steps"]
        done = analysis["computed_outputs"]
        if "measures" in done:
            available.extend(["measures", "reliability", "fair_average"])
        if "agreement" in done:
            available.append("agreement")
        if "pca" in done:
            available.append("pca")
        settings = analysis["fit_settings"]
        connectivity = analysis["result"].get("identifiability", {}).get("connectivity")
        if (settings.get("anchors") or settings.get("group_anchors")
                or isinstance(connectivity, pd.DataFrame) and not connectivity.empty and connectivity.Components.gt(1).any()):
            available.append("connectivity")
        if "measures" in done and (settings.get("umean", 0.), settings.get("uscale", 1.)) != (0., 1.):
            available.append("rescaled")
        for key, name in (("bias_analyses", "bias"), ("interval_analyses", "intervals"), ("gd_analysis", "gd"),
                          ("joint_analysis", "joint"), ("rater_dff", "rater_dff"), ("anchor_sensitivity", "anchor_sensitivity")):
            if analysis.get(key):
                available.append(name)
        avail_tables = ", ".join(available)
    else:
        all_tables = desktop_analysis.current_tables(analysis)
        avail_tables = ", ".join(dict.fromkeys(
            ("gd" if k.startswith("gd_") else ("agreement" if k in {"agreement_summary", "pairwise_agreement"} else k))
            for k in (_canonical_table_kind(t) for t, _, _ in all_tables)
        ))
    order = ("design", "measures", "fair_average", "categories", "steps", "connectivity", "reliability",
             "agreement", "rescaled", "bias", "intervals", "joint", "rater_dff", "anchor_sensitivity", "gd", "pca")
    names = set(avail_tables.split(", ")) - {""}
    avail_tables = ", ".join([k for k in order if k in names] + sorted(names-set(order)))
    avail_plots = ", ".join(_available_plot_kinds(analysis))

    if analysis.get("kind") == "gd-analysis-v2":
        gd_entry = analysis["gd_analysis"]
        gd_cfg = gd_entry["settings"]
        gd_res = gd_entry["result"]
        avail = bool(gd_res.get("available"))
        obs_c = gd_res.get("observed_coefficients") or {}
        ci = gd_res.get("coefficient_intervals", {})
        rows = [
            ("Workflow", "gd"),
            ("AnalysisID", str(analysis.get("analysis_id", ""))),
            ("Available", str(avail)),
            ("ReasonCode", str(gd_res.get("reason_code") or "none")),
            ("ObjectFacet", str(gd_cfg.get("object_facet", ""))),
            ("RandomFacets", ", ".join(map(str, gd_cfg.get("random_facets") or ()))),
            ("Observations", str((gd_res.get("design") or {}).get("n_observations", 0))),
            ("ObservedG", f"{float(obs_c['G']):.3f}" if avail and pd.notna(obs_c.get("G")) else ""),
            ("ObservedPhi", f"{float(obs_c['Phi']):.3f}" if avail and pd.notna(obs_c.get("Phi")) else ""),
            ("SupportsFormalInference", "False"),
            ("CoverageQualified", "False"),
            ("CIEligible", str(any(ci.get("observed", {}).get(f"{k}_interval_status")
                                   == "approximate_normal_delta" for k in ("G", "Phi")))),
            ("IntervalMethod", ci.get("method", "not_requested")),
            ("IntervalAssumption", ci.get("assumption", "none")),
            ("IntervalScope", "pointwise_normal_random_effects_approximation" if ci else "none"),
            ("GIntervalStatus", ci.get("observed", {}).get("G_interval_status", "withheld")),
            ("PhiIntervalStatus", ci.get("observed", {}).get("Phi_interval_status", "withheld")),
            ("AvailableTables", avail_tables),
            ("AvailablePlots", avail_plots),
        ]
        if "score_cols" in gd_cfg:
            diagnostics = gd_res.get("multivariate", {}).get("component_diagnostics")
            rows.extend([("ScoreColumns", ", ".join(gd_cfg["score_cols"])),
                         ("ObservedCoefficientTarget", "Composite"),
                         ("CompositeWeights", str(gd_cfg["composite_weights"])),
                         ("ComponentTreatment", "clip_scalar_after_projection"),
                         ("NonPSDComponents", ", ".join(diagnostics.loc[~diagnostics.PositiveSemidefinite, "Source"])
                          if isinstance(diagnostics, pd.DataFrame) else "not_computed"),
                         ("IntervalHoldReason", "Multivariate G/Phi intervals are not implemented or qualified.")])
        if gd_cfg.get("method") == "minque0":
            design, audit = gd_res["design"], gd_res.get("estimation", {})
            rows = [row for row in rows if row[0] != "IntervalHoldReason"]
            rows.extend([("Method", "minque0"), ("MissingPolicy", design["missing"]),
                         ("OmittedRows", str(design["n_omitted_rows"])),
                         ("Nesting", str(design["nesting"])), ("ComponentRank", str(audit.get("rank", "not_checked"))),
                         ("PlanningStatus", design.get("planning_status", "unavailable")),
                         ("ObservedCoefficientHoldReason", design.get("observed_coefficient_reason", "")),
                         ("IntervalHoldReason", gd_res["bootstrap_ci"]["inference_reason"])])
        return pd.DataFrame(rows, columns=["Field", "Value"])

    result = analysis["result"]
    cfg = result.get("config") or {}
    prep = result.get("prep") or {}
    snap = basic_analysis.qualification_snapshot(result)
    fnames = list(cfg.get("facet_names") or ())
    flevels = cfg.get("facet_levels") or {}
    fsigns = cfg.get("facet_signs") or {}
    pos_facets = [f for f in fnames if int(fsigns.get(f, -1)) == 1]
    unused = prep.get("unused_score_categories") or []
    rows = [
        ("Workflow", "mfrm"),
        ("AnalysisID", str(analysis.get("analysis_id", ""))),
        ("Model", str(cfg.get("model", ""))),
        ("Method", str(cfg.get("method", ""))),
        ("RetainedRatings", str(len(prep.get("data", ())))),
        ("Persons", str(cfg.get("n_person", ""))),
        ("Facets", ", ".join(f"{f} ({len(flevels.get(f, ()))})" for f in fnames)),
        ("NoncenterFacet", str(cfg.get("noncenter_facet") or "Person")),
        ("PositiveFacets", ", ".join(pos_facets) if pos_facets else "none"),
        ("ScoreRange", f"{prep.get('rating_min')}–{prep.get('rating_max')} (unused: {', '.join(map(str, unused)) if unused else 'none'})"),
        ("OptimizerTerminated", str(bool(snap.get("optimizer_terminated")))),
        ("RunQualificationState", str(snap.get("run_state", ""))),
        ("IdentifiabilityState", str(snap.get("identifiability_state") or "n/a")),
        ("FormalHoldReason", str(snap.get("calibration_hold_reason") or "none")),
        ("ScreeningBlockReason", str(snap.get("screening_block_reason") or "none")),
        ("SupportsFormalInference", "False"),
        ("AvailableTables", avail_tables),
        ("AvailablePlots", avail_plots),
    ]
    if "computed_outputs" in analysis:
        rows.extend([
            ("ComputedOutputs", ", ".join(analysis["computed_outputs"]) or "none"),
            ("PendingOutputs", ", ".join(sorted({"measures", "agreement", "scorefile", "pca"} - set(analysis["computed_outputs"])))),
        ])
    return pd.DataFrame(rows, columns=["Field", "Value"])


@contextmanager
def _manifest_lock(directory):
    # OS locks are released if a writer exits, including on Windows.
    with (directory / ".mfrmpy-export.lock").open("a+b") as lock:
        if os.name == "nt":
            import msvcrt
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl
            fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def _previous_export(manifest):
    if not manifest.exists():
        return []
    previous = pd.read_csv(manifest, dtype=str, keep_default_na=False)
    required = {"AnalysisID", "TableIndex", "File", "Title", "Rows", "RunQualificationState"}
    if not required.issubset(previous):
        raise ValueError(f"{manifest} already exists and is not an mfrmpy export manifest")
    managed = []
    for record in previous.to_dict("records"):
        relative = Path(record["File"])
        index = int(record["TableIndex"])
        slug = exports.safe_export_name(record["Title"], default=f"table_{index}", max_len=64)
        if relative.name != f"{index:02d}_{slug}.csv" or relative.is_absolute():
            continue
        if len(relative.parts) > 1 and (len(relative.parts) != 2 or not relative.parts[0].startswith("mfrmpy-tables-")):
            continue
        path = manifest.parent / relative
        if path.is_symlink() or path.parent.is_symlink() or not path.is_file():
            continue
        if "SHA256" in record:
            if hashlib.sha256(path.read_bytes()).hexdigest() != record["SHA256"]:
                continue
        else:
            # 0.2.0 wrote flat tables without digests. Verify their provenance.
            try:
                table = pd.read_csv(path, dtype=str, usecols=["AnalysisID", "Table"], keep_default_na=False)
            except (ValueError, UnicodeError):
                continue
            if (len(table) != int(record["Rows"]) or not table.AnalysisID.eq(record["AnalysisID"]).all()
                    or not table.Table.eq(record["Title"]).all()):
                continue
        managed.append(path)
    return managed


def export_csv(analysis, directory, *, kind=None) -> list[Path]:
    """Write qualified CSV tables and a provenance manifest without refitting.

    Applies the same shared qualification rules as :func:`tables`, :func:`plot`
    and :func:`report_html`, while recording both current and save-time
    qualification reasons in ``00_manifest.csv``.
    """
    qualified_tables = tables(analysis, kind=kind)
    return _export_csv(analysis, directory, qualified_tables)


def _export_csv(analysis, directory, qualified_tables):
    out_dir = Path(directory)
    out_dir.mkdir(parents=True, exist_ok=True)

    analysis_id = str(analysis.get("analysis_id", ""))
    has_mfrm = "result" in analysis
    current_snap = basic_analysis.qualification_snapshot(analysis["result"]) if has_mfrm else {}
    saved_snap = analysis.get("qualification_at_save") if isinstance(analysis.get("qualification_at_save"), dict) else {}

    generation = Path(tempfile.mkdtemp(prefix="mfrmpy-tables-", dir=out_dir))
    try:
        manifest_rows: list[dict[str, object]] = []
        written_paths: list[Path] = [out_dir / "00_manifest.csv"]
        for idx, (title, frame, note) in enumerate(qualified_tables, start=1):
            slug = exports.safe_export_name(title, default=f"table_{idx}", max_len=64)
            filename = f"{idx:02d}_{slug}.csv"
            target = generation / filename
            out_frame = frame.copy()
            out_frame["AnalysisID"] = analysis_id
            out_frame["ContentFingerprint"] = analysis["content_fingerprint"]
            out_frame["Table"] = title
            out_frame["Note"] = note
            payload = out_frame.to_csv(index=False).encode("utf-8")
            project.write_bytes_atomic(target, payload)
            written_paths.append(target)
            manifest_rows.append({
                "AnalysisID": analysis_id,
                "TableIndex": idx,
                "File": target.relative_to(out_dir).as_posix(),
                "SHA256": hashlib.sha256(payload).hexdigest(),
                "ContentFingerprint": analysis["content_fingerprint"],
                "PackageVersion": __version__,
                "Title": title,
                "Rows": int(len(frame)),
                "SupportsFormalInference": False,
                "CoverageQualified": False,
                "CIEligible": bool(title.startswith("G/D: ") and any(
                    frame[column].eq("approximate_normal_delta").any()
                    for column in ("G_interval_status", "Phi_interval_status") if column in frame)),
                "IntervalBasis": (analysis.get("gd_analysis", {}).get("result", {}).get(
                    "coefficient_intervals", {}).get("method", "not_requested") if title.startswith("G/D: ") else ""),
                "RunQualificationState": str(current_snap.get("run_state", "gd_observed_scores" if not has_mfrm else "")),
                "FormalHoldReason": str(current_snap.get("calibration_hold_reason", "")),
                "ScreeningBlockReason": str(current_snap.get("screening_block_reason", "")),
                "SavedHoldReason": str(saved_snap.get("calibration_hold_reason", "")),
                "SavedHoldMessage": str(saved_snap.get("calibration_hold_message", "")),
                "Note": note,
            })

        manifest_df = pd.DataFrame(manifest_rows)
        with _manifest_lock(out_dir):
            previous = _previous_export(written_paths[0])
            project.write_bytes_atomic(written_paths[0], manifest_df.to_csv(index=False).encode("utf-8"))
    except Exception:
        for path in generation.iterdir():
            path.unlink()
        generation.rmdir()
        raise

    for path in previous:
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:
            warnings.warn(f"Previous export table retained: {exc}", stacklevel=2)
    for directory in {p.parent for p in previous if p.parent != out_dir}:
        try:
            directory.rmdir()
        except OSError:
            pass  # Retain any user files added to that directory.
    return written_paths


def load_example():
    """Return a fresh copy of the synthetic 0–5 writing ratings as a DataFrame."""
    resource = files(__package__).joinpath("data", "writing.csv")
    if not resource.is_file():
        resource = Path(__file__).resolve().parents[1] / "examples" / "desktop_writing.csv"
    with resource.open("r", encoding="utf-8") as handle:
        return pd.read_csv(handle)


def plot(analysis, kind="wright", *, level=None, index=0, facet=None,
         statistic="Infit", presentation="explained", gd_swap=False, top_n=40,
         view="levels", style="forest", levels=None):
    """Return a Matplotlib Figure; install the ``figures`` extra first.

    Supported ``kind`` values include ``"wright"``, ``"pathway"`` (alias
    ``"fit_pathway"``; ``statistic="Infit"|"Outfit"|"InfitZSTD"|"OutfitZSTD"``),
    ``"category"`` (alias ``"ccc"``), ``"facet"``, ``"rater_severity"``,
    ``"step"`` (alias ``"thresholds"``), ``"bubble"``, ``"expected"`` (alias
    ``"tcc"``), ``"information"``, ``"fair_average"``, ``"unexpected"``,
    ``"residual_qq"``, ``"bias"``, ``"bias_ranked"``, ``"interval"``,
    ``"joint"``, ``"rater_dff"``, ``"anchor_sensitivity"``, ``"gd"``, ``"pca"`` (alias ``"residual_pca"``), and
    ``"pca_loadings"``. PCM Wright maps and category curves require ``level``.
    ``anchor_sensitivity`` defaults to a colored forest of the selected levels.
    Choose ``view="difference"`` for the retained contrast, ``style="halfeye"``
    for a labelled normal approximation, or ``levels=[...]`` for other levels.
    No estimator is rerun to draw results.
    """
    from mfrmpy._core.desktop_outputs import analysis_figure
    return analysis_figure(analysis, kind, level=level, index=index, facet=facet,
                           statistic=statistic, presentation=presentation,
                           gd_swap=gd_swap, top_n=top_n, view=view, style=style, levels=levels)


def report_html(analysis, *, pathway_facet=None, pathway_statistic="Infit",
                facet=None, statistic=None, presentation=None, language=None, gd_swap=None):
    """Return an offline HTML report; requires the ``figures`` extra.

    ``facet`` and ``statistic`` are accepted as aliases for ``pathway_facet``
    and ``pathway_statistic`` to match :func:`plot`. Reports and .mfrm archives
    can contain the original ratings and identifiers; handle them as research data.
    """
    from mfrmpy._core.desktop_outputs import report_html as render
    return render(analysis, pathway_facet=pathway_facet,
                  pathway_statistic=pathway_statistic, facet=facet, statistic=statistic,
                  presentation=presentation, language=language, gd_swap=gd_swap)


def report_text(analysis, *, kind=None):
    """Return a UTF-8-compatible plain-text report of selected qualified tables.

    Includes every selected row, column and interpretation note; no refitting,
    automatic diagnostics, or optional plotting dependency. Not a FACETS input file.
    """
    from ._reports import render_text
    return render_text(analysis, tables(analysis, kind))


def export_out(analysis, path, *, kind=None):
    """Atomically write a plain-text .out (or .out.txt) report; return its Path."""
    path = Path(path)
    project.write_bytes_atomic(path, report_text(analysis, kind=kind).encode("utf-8"))
    return path


def export_bundle(analysis, path, *, kind=None, plots=None):
    """Atomically write report.out, CSV tables, PNG/SVG plots and manifest.json.

    plots=None selects available report figures; plots=[] needs no plotting extra.
    Otherwise pass plot kinds or dictionaries of plot() options (e.g. PCM level).
    kind filters tables before construction. Figures retain their explicit scope.
    The ZIP contains reporting outputs, not a reopenable .mfrm project.
    """
    from ._reports import write_bundle
    return write_bundle(analysis, path, kind=kind, plots=plots)
