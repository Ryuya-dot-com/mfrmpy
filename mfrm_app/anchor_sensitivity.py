"""Retain one contrast under alternate anchors and fixed population SDs."""
import json

import numpy as np
import pandas as pd
from scipy.stats import norm

from . import anchor_intervals, basic_analysis, engine, output_qualification, project, shared_calibration
from .rater_contrast import check_rater_contrast


TITLE = "Anchor sensitivity"
CONSTRAINTS = "Anchor sensitivity constraints"
WORKS = "Anchor sensitivity shared works"
INTERVALS = "Anchor sensitivity intervals"
OCCASIONS = "Anchor sensitivity occasion overlap"
REFERENCE = "No anchors"
_SUPPORTED = {"observed", "anchor_dependent"}
_HOLD = {
    "fixed_by_anchors": "The constraints fix this contrast before estimation.",
    "common_location_only": "The observed assignments need a common work location to identify this contrast.",
    "unsupported": "The contrast is unidentified under these assignments and constraints.",
    "audit_size_limit": "The assignment review exceeded its calculation size limit.",
}


def _without_sources(records):
    if not isinstance(records, (list, tuple)) or any(
        not isinstance(row, dict) or not isinstance(row.get("Source"), str) or not row["Source"].strip()
        for row in records
    ):
        raise ValueError("Supply anchor records with a nonblank Source on each record")
    return [{key: value for key, value in row.items() if key != "Source"} for row in records]


def _attach_sources(records, normalized, person):
    sources = {(("Person" if row["Facet"].strip() in {"Person", person} else row["Facet"].strip()),
                row["Level"].strip()): row["Source"].strip() for row in records}
    return [dict(row, Source=sources[row["Facet"], row["Level"]]) for row in normalized]


def _check_group_sources(records):
    sources = {}
    for row in records:
        key = row["Facet"], row["Group"]
        if sources.setdefault(key, row["Source"]) != row["Source"]:
            raise ValueError("All members of a group-mean constraint must have the same Source")


def _point(result, audit):
    reason = output_qualification.calibration_screening_block_reason(result)
    if reason:
        return None, reason
    facet = audit["rater"]
    levels = result["config"]["facet_levels"][facet]
    values = dict(zip(levels, result["params"]["facets"][facet], strict=True))
    point = sum(t["weight"] * values[t["rater_level"]] for t in audit["contrast_weights"])
    return (float(point), "") if np.isfinite(point) else (None, "Nonfinite contrast estimate")


def _finite(value, label):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating))
            or not np.isfinite(value)):
        raise ValueError(f"{label} must be a finite number")
    return float(value)


def _occasion_info(analysis, rater, difference, column):
    if not isinstance(column, str) or column not in analysis["input_data"]:
        raise ValueError("occasion must name an input column")
    prep = analysis["result"]["prep"]
    raw = analysis["input_data"][column].iloc[prep["row_audit"].IncludedInLikelihood.to_numpy(dtype=bool)]
    if not raw.map(lambda value: isinstance(value, str) and bool(value.strip())).all():
        raise ValueError("Fitted occasion labels must be nonblank strings")
    data = prep["data"]
    occasions = raw.str.strip().to_numpy()
    selected = []
    for level in difference:
        labels = np.unique(occasions[data[rater].eq(level).to_numpy()])
        if len(labels) != 1:
            raise ValueError("Each contrast level must identify exactly one occasion")
        selected.append(labels[0])
    if selected[0] == selected[1]:
        raise ValueError("An occasion contrast requires two distinct occasions")
    work_sets = [set(data.loc[occasions == label, "Person"]) for label in selected]
    return dict(column=column, first=selected[0], second=selected[1],
                first_works=sorted(work_sets[0]), second_works=sorted(work_sets[1]),
                shared_works=sorted(work_sets[0] & work_sets[1]))


def with_anchor_sensitivity(analysis, *, rater, difference, scenarios, intervals=False, important_difference=None,
                            occasion=None):
    """Compare a severity contrast with explicit alternate anchor constraints.

    Start with an unanchored fixed-SD MML RSM fit. Each named scenario uses the
    same ratings and may change `population_prior_sd` or use `anchors`,
    `group_anchors` and `noncenter_facet`. Add a `Source` string to each
    anchor record. With `intervals=True`, a scenario's `calibration` can
    request intervals with individually fixed rater anchors and a fitted
    population mean. For shared sampling units, pass the fitted calibration
    analysis and optionally a rating-event ID column as described in the guide.
    A calibration facet and level_map can map the same source estimate to
    multiple receiving levels. occasion labels the selected levels' occasions
    and records overlap across all works, separately from calibration overlap.
    Supported alternatives are fitted once. Displays reuse the retained results.
    """
    project.validate_analysis(analysis)
    fit = analysis.get("fit_settings", {})
    if ((fit.get("model"), fit.get("method")) != ("RSM", "MML")
            or fit.get("anchors") or fit.get("group_anchors")):
        raise ValueError("Anchor sensitivity requires an unanchored fixed-SD MML RSM as the reference")
    project.require_reportable_input(analysis)
    if not isinstance(scenarios, dict) or not scenarios:
        raise ValueError("Supply a mapping of scenario names to alternate anchor settings")
    if not isinstance(intervals, bool) or (important_difference is not None and not intervals):
        raise ValueError("Use intervals=True to check an important signed difference")
    if important_difference is not None:
        important_difference = _finite(important_difference, "important_difference")

    data = analysis["input_data"]
    origin = fit.get("noncenter_facet", "Person")
    reference_sd = fit["population_prior_sd"]
    args = dict(person=fit["person_col"], facets=fit["facet_cols"], score=fit["score_col"],
                rater=rater, difference=difference, positive_facets=fit.get("positive_facets"))
    reference = check_rater_contrast(data, **args, noncenter_facet=origin)
    occasion_info = _occasion_info(analysis, rater, difference, occasion) if occasion is not None else None
    receiving = set(analysis["result"]["prep"]["levels"]["Person"])
    names = {REFERENCE}
    prepared = [(REFERENCE, reference, [], [], reference_sd, None)]
    sources = {}
    for name, scenario in scenarios.items():
        if (not isinstance(name, str) or not name.strip() or name.strip() in names
                or not isinstance(scenario, dict)
                or scenario.keys() - {"anchors", "group_anchors", "noncenter_facet", "population_prior_sd", "calibration"}):
            raise ValueError("Use distinct nonblank scenario names and anchor settings")
        name = name.strip()
        names.add(name)
        fixed, means = scenario.get("anchors", []), scenario.get("group_anchors", [])
        sd = _finite(scenario.get("population_prior_sd", reference_sd), "population_prior_sd")
        if sd <= 0 or (not (fixed or means) and sd == reference_sd):
            raise ValueError(f"Scenario {name!r} requires positive SD and an alternate SD or anchor constraint")
        calibration = scenario.get("calibration")
        if calibration is not None and (not intervals or scenario.get("noncenter_facet", rater) != rater):
            raise ValueError("Calibration intervals require intervals=True and the rater as noncenter_facet")
        scenario_origin = rater if calibration is not None else scenario.get("noncenter_facet", origin)
        audit = check_rater_contrast(
            data, **args, noncenter_facet=scenario_origin,
            anchors=_without_sources(fixed), group_anchors=_without_sources(means),
        )
        sourced_fixed = _attach_sources(fixed, audit["anchors"], args["person"])
        sourced_means = _attach_sources(means, audit["group_anchors"], args["person"])
        _check_group_sources(sourced_means)
        if calibration is not None:
            source_record = None
            if isinstance(calibration, dict) and "analysis" in calibration:
                if calibration.get("kind") != "shared" or calibration.keys() - {"kind", "analysis", "rating_id", "facet", "level_map"}:
                    raise ValueError("Use analysis, optional rating_id, and explicit facet/level_map with shared calibration")
                source = calibration["analysis"]
                source_record = shared_calibration.source_settings(
                    source, analysis, sourced_fixed, rater, calibration.get("rating_id"),
                    facet=calibration.get("facet"), level_map=calibration.get("level_map"), occasion=occasion)
                sources[name] = source
                calibration = dict(kind="shared", work_ids=list(source["result"]["prep"]["levels"]["Person"]))
            calibration = anchor_intervals.calibration_settings(calibration, sourced_fixed, sourced_means, rater, receiving)
            if source_record is not None:
                calibration["source"] = source_record
        prepared.append((name, audit, sourced_fixed, sourced_means, sd, calibration))

    rows = []
    for name, audit, fixed, means, sd, calibration in prepared:
        status = audit["status"]
        row = dict(name=name, anchors=fixed, group_anchors=means, noncenter_facet=audit["noncenter_facet"],
                   support=status, observation_estimable=audit["observation_estimable"],
                   population_prior_sd=sd, calibration=calibration,
                   estimate=None, reason=_HOLD.get(status, ""), fit_id=None)
        if status in _SUPPORTED:
            try:
                if name == REFERENCE:
                    result, fit_id = analysis["result"], analysis["analysis_id"]
                elif calibration is not None:
                    result = anchor_intervals.fit_with_anchor_origin(data, fit | dict(population_prior_sd=sd), audit)
                    fit_id = engine.build_result_analysis_identity(result).analysis_id
                else:
                    candidate = basic_analysis.run_basic_analysis(data, **(fit | dict(
                        anchors=audit["anchors"], group_anchors=audit["group_anchors"],
                        noncenter_facet=audit["noncenter_facet"], population_prior_sd=sd, compute_pca=False,
                    )))
                    result, fit_id = candidate["result"], candidate["analysis_id"]
                row["estimate"], row["reason"] = _point(result, audit)
                row["fit_id"] = fit_id
            except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
                row["reason"] = f"Calibration failed: {exc}"
        if intervals:
            try:
                if row["estimate"] is None:
                    raise ValueError(row["reason"])
                row["interval"] = dict(anchor_intervals.intervals(result, audit, calibration, sources.get(name)), reason="")
            except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
                row["interval"] = dict(se=None, propagated_se=None, sensitivity=None, propagation_reason="", reason=str(exc))
        rows.append(row)

    out = dict(analysis, anchor_sensitivity=dict(
        version=1, analysis_id=analysis["analysis_id"], rater=rater,
        difference=[t["rater_level"] for t in reference["contrast_weights"]],
        overlap=reference["overlap"][0], scenarios=rows,
        fit_source=basic_analysis._source_versions(), important_difference=important_difference,
    ))
    if occasion_info is not None:
        out["anchor_sensitivity"]["occasion"] = occasion_info
    out["content_fingerprint"] = project.content_fingerprint(out)
    project.validate_analysis(out)
    return out


def validate_entry(analysis, entry):
    fit = analysis["fit_settings"]
    if (not isinstance(entry, dict) or entry.get("version") != 1
            or entry.get("analysis_id") != analysis["analysis_id"]
            or entry.get("rater") not in fit["facet_cols"]
            or (fit["model"], fit["method"]) != ("RSM", "MML")
            or fit.get("anchors") or fit.get("group_anchors")
            or not isinstance(entry.get("fit_source"), dict)
            or not isinstance(entry.get("scenarios"), list) or len(entry["scenarios"]) < 2):
        raise project.ProjectError("Anchor sensitivity does not match the reference calibration")
    difference = entry.get("difference")
    known = analysis["result"]["config"]["facet_levels"][entry["rater"]]
    overlap = entry.get("overlap")
    if (not isinstance(difference, list) or len(difference) != 2
            or any(not isinstance(level, str) or level not in known for level in difference)
            or difference[0] == difference[1] or not isinstance(overlap, dict)
            or overlap.get("kind") not in ("direct", "indirect", "none")
            or not isinstance(overlap.get("path"), list)
            or any(not isinstance(link, dict) or not isinstance(link.get("persons"), list)
                   or not all(isinstance(work, str) for work in link["persons"])
                   or any(link.get(key) not in known for key in ("from_rater", "to_rater"))
                   for link in overlap["path"])):
        raise project.ProjectError("Anchor sensitivity requires a retained contrast and shared-work evidence")
    if "occasion" in entry:
        info = entry["occasion"]
        if not isinstance(info, dict) or "column" not in info:
            raise project.ProjectError("Invalid occasion overlap")
        try:
            expected = _occasion_info(analysis, entry["rater"], difference, info["column"])
        except ValueError as exc:
            raise project.ProjectError("Invalid occasion overlap: " + str(exc)) from exc
        if info != expected:
            raise project.ProjectError("Occasion overlap disagrees with the fitted ratings")
    names = set()
    roles = dict(person_col=fit["person_col"], facet_cols=fit["facet_cols"])
    for number, row in enumerate(entry["scenarios"]):
        if (not isinstance(row, dict) or not isinstance(row.get("name"), str) or not row["name"].strip()
                or row["name"] in names or (row["name"] == REFERENCE) != (number == 0)
                or row.get("support") not in (*_SUPPORTED, *_HOLD)
                or row.get("noncenter_facet") not in ["Person", *fit["facet_cols"]]
                or not isinstance(row.get("reason"), str) or "fit_id" not in row or "estimate" not in row
                or "observation_estimable" not in row or row["observation_estimable"] not in (True, False, None)):
            raise project.ProjectError("Invalid anchor sensitivity scenario")
        names.add(row["name"])
        fixed = _without_sources(row.get("anchors"))
        means = _without_sources(row.get("group_anchors"))
        baseline_sd = fit["population_prior_sd"]
        sd = _finite(row.get("population_prior_sd", baseline_sd), "population_prior_sd")
        if (sd <= 0 or (bool(fixed or means or sd != baseline_sd) != (number > 0))
                or (number == 0 and row["noncenter_facet"] != fit.get("noncenter_facet", "Person"))
                or basic_analysis.normalize_anchor_records(fixed, **roles) != fixed
                or basic_analysis.normalize_group_anchor_records(means, fixed_records=fixed, **roles) != means):
            raise project.ProjectError("Inconsistent anchor sensitivity constraints")
        _check_group_sources(row["group_anchors"])
        value, calibration_id = row.get("estimate"), row.get("fit_id")
        if (value is None and not row["reason"]
                or value is not None and (type(value) is not float or not np.isfinite(value) or row["reason"]
                                          or row["support"] not in _SUPPORTED or not calibration_id)
                or calibration_id is not None and not isinstance(calibration_id, str)
                or number == 0 and calibration_id not in (None, analysis["analysis_id"])):
            raise project.ProjectError("An anchor scenario must retain a supported estimate or a reason")
        _validate_intervals(analysis, entry, row)


def _validate_intervals(analysis, entry, row):
    threshold = entry.get("important_difference")
    interval, calibration = row.get("interval"), row.get("calibration")
    if interval is None:
        if calibration is not None or threshold is not None:
            raise project.ProjectError("Anchor interval evidence is missing")
        return
    if threshold is not None:
        _finite(threshold, "important_difference")
    if not isinstance(interval, dict) or set(interval) - {"shared"} != {"se", "propagated_se", "sensitivity", "reason", "propagation_reason"}:
        raise project.ProjectError("Invalid anchor interval evidence")
    shared = interval.get("shared")
    if "shared" in interval and (not isinstance(shared, dict) or not calibration or not calibration.get("source")):
        raise project.ProjectError("Shared score evidence needs its calibration source")
    if not all(isinstance(interval[key], str) for key in ("reason", "propagation_reason")):
        raise project.ProjectError("Anchor interval holds need explicit reasons")
    se, propagated = interval["se"], interval["propagated_se"]
    if (se is None) != bool(interval["reason"]) or (se is not None and (
        row["estimate"] is None or _finite(se, "conditional SE") <= 0
    )):
        raise project.ProjectError("Invalid conditional anchor interval")
    if se is None and (propagated is not None or interval["sensitivity"] is not None):
        raise project.ProjectError("A held conditional interval has no propagation evidence")
    if calibration is None:
        if propagated is not None or interval["sensitivity"] is not None:
            raise project.ProjectError("Anchor propagation needs calibration evidence")
        return
    kind = calibration["kind"]
    spec = dict(kind=kind)
    if kind != "known":
        spec["work_ids"] = calibration["work_ids"]
    if kind == "independent" or (kind == "shared" and calibration["covariance"] is not None):
        labels = calibration["levels"]
        spec["covariance"] = pd.DataFrame(calibration["covariance"], index=labels, columns=labels)
    receiving = set(analysis["result"]["prep"]["levels"]["Person"])
    source = calibration.get("source")
    if "source" in calibration and (kind != "shared" or not isinstance(source, dict)
            or set(source) not in ({"analysis_id", "fingerprint", "rating_id", "reused_ratings",
                                    "population_sd", "quadrature_points"},
                                   {"analysis_id", "fingerprint", "rating_id", "reused_ratings",
                                    "population_sd", "quadrature_points", "facet", "level_map", "estimates"})
            or not all(isinstance(source[key], str) and source[key] for key in ("analysis_id", "fingerprint"))
            or _finite(source["population_sd"], "calibration SD") <= 0
            or type(source["quadrature_points"]) is not int or source["quadrature_points"] < 3
            or not ((source["rating_id"] is None and source["reused_ratings"] is None)
                    or (isinstance(source["rating_id"], str) and source["rating_id"].strip()
                        and type(source["reused_ratings"]) is int
                        and 0 <= source["reused_ratings"] <= len(analysis["result"]["prep"]["data"])
                        and (not source["reused_ratings"] or calibration["overlap"])))):
        raise project.ProjectError("Invalid shared calibration source or rating reuse evidence")
    if source and "level_map" in source:
        mapping, estimates = source["level_map"], source["estimates"]
        levels = analysis["result"]["config"]["facet_levels"][entry["rater"]]
        if (not isinstance(source["facet"], str) or not source["facet"].strip()
                or not isinstance(mapping, dict) or set(mapping) != set(levels)
                or not isinstance(estimates, dict) or not estimates
                or any(not isinstance(k, str) or not k.strip() for k in estimates)
                or any(not isinstance(v, str) or v not in estimates for v in mapping.values())):
            raise project.ProjectError("Invalid source-to-receiving anchor mapping")
        for value in estimates.values():
            _finite(value, "calibration source estimate")
        if any(not np.isclose(item["Anchor"], estimates[mapping[item["Level"]]], atol=1e-12, rtol=0)
               for item in row["anchors"]):
            raise project.ProjectError("Mapped source estimates disagree with fixed anchor values")
    settings = {key: value for key, value in calibration.items() if key != "source"}
    if (row["noncenter_facet"] != entry["rater"] or settings != anchor_intervals.calibration_settings(
        spec, row["anchors"], row["group_anchors"], entry["rater"], receiving
    )):
        raise project.ProjectError("Anchor calibration evidence disagrees with the fixed values or work IDs")
    sensitivity = interval["sensitivity"]
    if se is not None:
        if not isinstance(sensitivity, list) or len(sensitivity) != len(calibration["levels"]):
            raise project.ProjectError("Missing anchor sensitivity")
        for value in sensitivity:
            _finite(value, "anchor sensitivity")
        if abs(sum(sensitivity)) > 1e-10:
            raise project.ProjectError("Common shifts in the anchors must leave the contrast unchanged")
    independent = kind == "independent" and not calibration["overlap"]
    if independent and se is not None and not interval["propagation_reason"]:
        v = np.asarray(sensitivity) @ np.asarray(calibration["covariance"]) @ np.asarray(sensitivity)
        expected = np.hypot(se, np.sqrt(max(float(v), 0.)))
        if propagated is None or not np.isclose(_finite(propagated, "propagated SE"), expected, rtol=1e-12, atol=0.):
            raise project.ProjectError("Propagated SE disagrees with the retained anchor covariance")
    elif shared is not None:
        if (se is None or interval["propagation_reason"] or set(shared) != {
            "calibration", "receiving", "calibration_variance", "receiving_variance", "cross_term", "se"
        }):
            raise project.ProjectError("Invalid shared score interval")
        for key, ids in (("calibration", calibration["work_ids"]), ("receiving", receiving)):
            if not isinstance(shared[key], dict) or set(shared[key]) != set(ids):
                raise project.ProjectError("Shared score IDs disagree with fitted sampling units")
            for value in shared[key].values():
                _finite(value, "person contribution")
        calculated = shared_calibration.variance_components(pd.Series(shared["calibration"]), pd.Series(shared["receiving"]))
        if any(not np.isclose(_finite(shared[k], k), v, atol=1e-14, rtol=1e-12) for k, v in calculated.items()
               ) or propagated != shared["se"]:
            raise project.ProjectError("Shared SE or variance terms disagree with person contributions")
    elif propagated is not None:
        raise project.ProjectError("Anchor propagation needs eligible calibration evidence")
    elif se is not None and kind in ("independent", "shared") and not interval["propagation_reason"]:
        raise project.ProjectError("An unavailable propagated interval requires a reason")


def tables(analysis):
    entry = analysis["anchor_sensitivity"]
    reference = entry["scenarios"][0]["estimate"]
    left, right = entry["difference"]
    signs = {"Person": 1, **analysis["result"]["config"]["facet_signs"]}
    rows, constraints, intervals = [], [], []
    for row in entry["scenarios"]:
        estimate = row["estimate"]
        rows.append(dict(Scenario=row["name"], RaterFacet=entry["rater"], FirstLevel=left, SecondLevel=right,
                         PersonSD=row.get("population_prior_sd", analysis["fit_settings"]["population_prior_sd"]),
                         PersonMean="estimated" if row.get("calibration") is not None else "fixed_zero",
                         NoncenterFacet=row["noncenter_facet"], AssignmentSupport=row["support"],
                         ObservationEstimable=row["observation_estimable"], Estimate=estimate,
                         ChangeFromNoAnchors=(estimate-reference if estimate is not None and reference is not None else None),
                         FitID=row["fit_id"], Reason=row["reason"], SupportsFormalInference=False))
        if "interval" in row:
            intervals.extend(_interval_rows(row, rows[-1]["PersonSD"], entry.get("important_difference")))
        for item in row["anchors"]:
            constraints.append(dict(Scenario=row["name"], Kind="fixed", Facet=item["Facet"], Group="",
                                    FacetSign=signs[item["Facet"]],
                                    Levels=json.dumps([item["Level"]], ensure_ascii=False),
                                    Value=item["Anchor"], Source=item["Source"]))
        by_group = {}
        for item in row["group_anchors"]:
            key = item["Facet"], item["Group"]
            by_group.setdefault(key, []).append(item)
        for (facet, group), items in by_group.items():
            constraints.append(dict(Scenario=row["name"], Kind="group_mean", Facet=facet, Group=group,
                                    FacetSign=signs[facet],
                                    Levels=json.dumps([item["Level"] for item in items], ensure_ascii=False),
                                    Value=items[0]["GroupValue"], Source=items[0]["Source"]))
    overlap = entry["overlap"]
    works = [dict(WorkColumn=analysis["fit_settings"]["person_col"], Path=overlap["kind"],
                  FromRater=link["from_rater"], ToRater=link["to_rater"], SharedWorks=len(link["persons"]),
                  WorkIDs=json.dumps(link["persons"], ensure_ascii=False))
             for link in (overlap["path"] or [dict(from_rater=left, to_rater=right, persons=[])])]
    tables = [
        (TITLE, pd.DataFrame(rows), "Point estimates in severity logits for the first level minus the second. "
         "The ratings are common to all scenarios. A change in NoncenterFacet changes which facet carries the "
         "free location. Review the scenario and anchor values together."),
        (CONSTRAINTS, pd.DataFrame(constraints, columns=["Scenario", "Kind", "Facet", "Group", "FacetSign",
                                                        "Levels", "Value", "Source"]),
         "Anchor values in native model logits. With FacetSign = −1 "
         "the anchor coefficient lowers the predicted score. Each group mean has one row. "
         "Source is supplied by the analyst."),
        (WORKS, pd.DataFrame(works), "One observed path between the compared levels. Shared-work links show "
         "who scored the same work. The assignment status also checks the full facet design."),
    ]
    if intervals:
        tables.append((INTERVALS, pd.DataFrame(intervals),
            "Pointwise 95% Wald intervals for sensitivity checks under the listed fixed population SD. "
            "Conditional holds the anchor values fixed. Independent calibration adds their full covariance "
            "without a cross-sample term. Shared calibration combines both stages' influence for each work. "
            "Interpret these diagnostic intervals using coverage evidence for the sampling model and assignment."))
    if info := entry.get("occasion"):
        tables.append((OCCASIONS, pd.DataFrame([dict(
            OccasionColumn=info["column"], FirstOccasion=info["first"], SecondOccasion=info["second"],
            FirstOccasionWorks=len(info["first_works"]), SecondOccasionWorks=len(info["second_works"]),
            SharedWorks=len(info["shared_works"]), WorkIDs=json.dumps(info["shared_works"], ensure_ascii=False),
        )]), "All fitted works shared by the selected occasions, regardless of rater. This is distinct from the "
            "selected raters' shared-work path and from CalibrationWorks/OverlapWorks in the interval table. "
            "The contrast is FirstLevel minus SecondLevel; shared works alone do not remove occasion/mode confounding."))
    return tables


def _interval_rows(row, sd, threshold):
    info = row["interval"]
    calibration = row.get("calibration") or {}
    kind = calibration.get("kind")
    estimate = row["estimate"]
    bases = [("conditional" if row["anchors"] or row["group_anchors"] else "no_anchors", info["se"], info["reason"])]
    if kind in ("independent", "shared"):
        basis = "shared_calibration" if calibration.get("source") else "calibration_propagated"
        bases.append((basis, info["propagated_se"], info["propagation_reason"] or info["reason"]))
    critical = norm.ppf(.975)
    rows = []
    for basis, se, reason in bases:
        width = float(2*critical*se) if se is not None else None
        low = estimate-width/2 if width is not None else None
        high = estimate+width/2 if width is not None else None
        uses_covariance = basis == "calibration_propagated" and calibration["covariance"] is not None
        source = calibration.get("source") or {}
        shared = info.get("shared", {}) if basis == "shared_calibration" else {}
        rows.append(dict(Scenario=row["name"], PersonSD=sd, Basis=basis, CalibrationKind=kind,
                         CalibrationWorks=len(calibration["work_ids"]) if kind in ("independent", "shared") else None,
                         OverlapWorks=len(calibration["overlap"]) if kind in ("independent", "shared") else None,
                         OverlapWorkIDs=json.dumps(calibration["overlap"], ensure_ascii=False) if kind in ("independent", "shared") else None,
                         CalibrationAnalysisID=source.get("analysis_id"), CalibrationPersonSD=source.get("population_sd"),
                         CalibrationQuadraturePoints=source.get("quadrature_points"), RatingIDColumn=source.get("rating_id"),
                         CalibrationFacet=source.get("facet"),
                         CalibrationLevelMap=json.dumps(source["level_map"], ensure_ascii=False) if "level_map" in source else None,
                         CalibrationEstimates=json.dumps(source["estimates"], ensure_ascii=False) if "estimates" in source else None,
                         ReusedRatings=source.get("reused_ratings"), CalibrationVariance=shared.get("calibration_variance"),
                         ReceivingVariance=shared.get("receiving_variance"), SharedCrossTerm=shared.get("cross_term"),
                         AnchorLevels=json.dumps(calibration["levels"], ensure_ascii=False) if uses_covariance or shared else None,
                         AnchorCovariance=json.dumps(calibration["covariance"]) if uses_covariance else None,
                         AnchorSensitivity=json.dumps(info["sensitivity"]) if (uses_covariance or shared) and info["sensitivity"] is not None else None,
                         Estimate=estimate, SE=se, Lower=low, Upper=high, Width=width, CILevel=.95,
                         IncludesZero=(low <= 0 <= high) if low is not None else None,
                         ImportantDifference=threshold,
                         IncludesImportantDifference=(low <= threshold <= high) if low is not None and threshold is not None else None,
                         Status="model_approximation" if se is not None else "withheld", Reason=reason,
                         CoverageQualified=False, SupportsFormalInference=False))
    return rows
