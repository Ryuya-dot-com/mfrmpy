"""One retained, observation-supported group contrast from a school–rater RSM."""
import json

import numpy as np
import pandas as pd
from scipy.stats import norm

from . import engine, output_qualification, project, structural_intervals
from .rater_contrast import check_rater_contrast


TITLE = "Prespecified group DFF contrast"
OVERLAP = "Group DFF shared works"
SCORES = "Group DFF category use"
PLAN = "Group DFF assignment coverage"
GAPS = "Group DFF assignment discrepancies"
NOTE = ("Severity logits for one prespecified contrast. Positive means the first rater is relatively "
        "stricter in the first group. When shown, the interval and Wald test are fixed-SD model approximations. "
        "An interval spanning zero can include differences large enough to matter. Coverage evidence is specific "
        "to the simulated models and assignments. Inspect shared works, tasks and rubric use. "
        "Conditional bias diagnostics are exploratory.")


def cell_label(rater, group):
    return json.dumps([rater, group], ensure_ascii=False)


def prepare_rater_dff_data(data, *, rater, group, cell_facet):
    """Add validated composite levels and preserve missing rater or group IDs."""
    if (not isinstance(data, pd.DataFrame) or not data.columns.is_unique or rater == group
            or any(col not in data for col in (rater, group))
            or not isinstance(cell_facet, str) or not cell_facet.strip() or cell_facet in data
            or cell_facet in {"Person", "Score", "Weight"}):
        raise ValueError("Select distinct rater and group columns and a new composite facet name")
    keys = data[[rater, group]]
    complete = ~keys.apply(engine._blank_or_na_mask).any(axis=1).to_numpy()
    labels = np.full(len(data), None, dtype=object)
    normalized = keys.iloc[complete].apply(engine.normalized_rating_ids)
    labels[complete] = [cell_label(*row) for row in normalized.itertuples(index=False, name=None)]
    return data.assign(**{cell_facet: pd.array(labels, dtype="string")})


def with_rater_dff(analysis, *, rater, group, difference, group_difference, cell_facet, planned_assignments=None):
    """Retain one ordered severity contrast on a fitted composite-facet analysis.

    The composite facet uses JSON arrays [rater, group] as level labels. The
    original columns must be retained. An existing interval request on the
    composite facet supplies the information for a model-based interval.
    An optional planned roster is retained as assignment diagnostics.
    Creating or displaying this record never refits the model.
    """
    project.validate_analysis(analysis)
    fit = analysis.get("fit_settings", {})
    facets = fit.get("facet_cols", [])
    raw = analysis["input_data"]
    if cell_facet not in facets or rater == group or any(
        col not in raw or col in [*facets, fit["person_col"], fit["score_col"]] for col in (rater, group)
    ):
        raise ValueError("Select a fitted composite facet and the original, distinct rater and group columns")
    audit = check_rater_contrast(
        raw, person=fit["person_col"], score=fit["score_col"], rating_min=fit["rating_min"],
        rating_max=fit["rating_max"], facets=[f for f in facets if f != cell_facet] + [rater, group],
        rater=rater, group=group, difference=difference, group_difference=group_difference,
        planned_assignments=planned_assignments,
    )
    settings = dict(rater=rater, group=group, cell_facet=cell_facet,
                    difference=[x.strip() for x in difference],
                    group_difference=[x.strip() for x in group_difference])
    out = dict(analysis, rater_dff=dict(version=1, analysis_id=analysis["analysis_id"], settings=settings, audit=audit))
    out["content_fingerprint"] = project.content_fingerprint(out)
    project.validate_analysis(out)
    return out


def validate_entry(analysis, entry):
    if not isinstance(entry, dict) or not isinstance(entry.get("settings"), dict) or not isinstance(entry.get("audit"), dict):
        raise project.ProjectError("Group DFF requires a retained request and assignment evidence")
    settings = entry.get("settings", {})
    audit = entry.get("audit", {})
    fit = analysis.get("fit_settings", {})
    if (entry.get("version") != 1 or entry.get("analysis_id") != analysis["analysis_id"]
            or set(settings) != {"rater", "group", "difference", "group_difference", "cell_facet"}
            or settings["cell_facet"] not in fit.get("facet_cols", [])
            or any(settings[key] != audit.get(key) for key in ("rater", "group"))
            or audit.get("person") != fit.get("person_col")):
        raise project.ProjectError("Group DFF request or assignment evidence does not match the calibration")
    raters, groups = settings["difference"], settings["group_difference"]
    if (any(not isinstance(pair, list) or len(pair) != 2 or any(not isinstance(x, str) for x in pair)
            or pair[0] == pair[1] for pair in (raters, groups))
            or [g["group_level"] for g in audit.get("overlap", [])] != groups
            or [(t["rater_level"], t["group_level"], t["weight"]) for t in audit.get("contrast_weights", [])]
            != [(r, g, w) for g, weights in zip(groups, [(-1, 1), (1, -1)], strict=True)
                for r, w in zip(raters, weights, strict=True)]):
        raise project.ProjectError("Group DFF order or weights disagree with the assignment evidence")
    plan = audit.get("plan")
    if "plan" in audit and (not isinstance(plan, dict)
            or any(not isinstance(plan.get(k), list) for k in ("by_group", "missing", "unplanned"))
            or [row["group_level"] for row in plan["by_group"]] != groups):
        raise project.ProjectError("Planned assignment evidence does not match the requested groups")
    support = audit.get("score_support")
    if "score_support" in audit and (not isinstance(support, dict)
            or any(not isinstance(support.get(k), list) for k in ("works_at_minimum", "works_at_maximum", "by_rater"))
            or (support.get("rating_min"), support.get("rating_max")) != (fit.get("rating_min"), fit.get("rating_max"))):
        raise project.ProjectError("Group DFF category evidence does not match the rubric")


def _model_point(analysis, cell, settings, audit, weights):
    result = analysis["result"]
    cfg = result["config"]
    if (cfg["model"] != "RSM" or cfg["method"] != "MML" or cfg["population_prior_sd"] != 1.
            or (cfg["rating_min"], cfg["n_cat"]) != (1, 5) or len(cfg["facet_names"]) != 2
            or cfg["noncenter_facet"] != cell or cfg["facet_signs"][cell] != -1
            or analysis["fit_settings"].get("anchors") or analysis["fit_settings"].get("group_anchors")):
        raise ValueError("Group DFF requires an unanchored five-category fixed-SD RSM with one shared task facet")
    prep = result["prep"]
    fitted = prep["data"]
    row_audit = prep.get("row_audit")
    if not isinstance(row_audit, pd.DataFrame) or "IncludedInLikelihood" not in row_audit:
        raise ValueError("No retained row audit matches the fitted ratings to the assignments")
    used = row_audit["IncludedInLikelihood"].to_numpy(dtype=bool)
    if len(fitted) != audit["included_rows"] or len(used) != len(analysis["input_data"]) or used.sum() != len(fitted):
        raise ValueError("Fitted ratings and assignment evidence are different")
    raw = analysis["input_data"].iloc[used]
    labels = [engine.normalized_rating_ids(raw[k]).tolist() for k in (settings["rater"], settings["group"])]
    if (any(not label for values in labels for label in values)
            or [cell_label(*pair) for pair in zip(*labels, strict=True)] != fitted[cell].astype(str).tolist()):
        raise ValueError("Composite levels disagree with the original rater and group labels")
    owners = pd.DataFrame(dict(person=fitted["Person"], group=labels[1]))
    if set(labels[1]) != set(settings["group_difference"]) or owners.groupby("person", observed=True).group.nunique().gt(1).any():
        raise ValueError("Group DFF requires two groups with each work belonging to one group")
    support = output_qualification.audit_boundary_and_support(result)
    if not support["active"] or support["hypergraph_components"] != 1 or support["disconnected_facets"] - {cell}:
        raise ValueError("Connectivity review failed outside the composite facet")
    reason = project.input_refit_reason(analysis) or output_qualification.calibration_screening_block_reason(result)
    if reason:
        raise ValueError(reason)
    cells = cfg["facet_levels"][cell]
    if not weights.keys() <= set(cells):
        raise ValueError("The fitted levels do not include all four contrast cells")
    point = np.array([weights.get(level, 0) for level in cells]) @ result["params"]["facets"][cell]
    if not np.isfinite(point):
        raise ValueError("Nonfinite group DFF estimate")
    return float(point)


def _interval(analysis, cell, weights, estimate):
    entry = next((e for e in analysis.get("interval_analyses", [])
                  if e["settings"]["facet"] == cell and e["settings"]["difference"] is None), None)
    if entry is None:
        raise ValueError("No retained information review for the composite facet")
    if entry["status"] != "computed":
        raise ValueError(entry["reason"])
    record = entry["record"]
    structural_intervals.read_structural_intervals(analysis["result"], record)
    checks, covariance = structural_intervals._assess(record)
    # A group location shift cancels for this observation-supported contrast.
    failed = [c["check"] for c in checks if not c["passed"] and c["check"] != "person_facet_connectivity"]
    if failed:
        raise ValueError(", ".join(failed))
    cells = [t["Level"] for t in record["targets"]]
    if cells != analysis["result"]["config"]["facet_levels"][cell]:
        raise ValueError("Retained information levels disagree with the fitted composite facet")
    linear = np.array([weights.get(level, 0) for level in cells]) @ np.asarray(record["target_jacobian"])
    variance = float(linear @ covariance @ linear)
    if not np.isfinite(variance) or variance <= 0 or not np.isclose(
        linear @ record["coordinates"], estimate, rtol=0, atol=1e-12
    ):
        raise ValueError("Invalid contrast variance or inconsistent retained estimate")
    se = float(np.sqrt(variance))
    level = record["level"]
    half = float(norm.ppf(.5 + level/2) * se)
    return dict(SE=se, Lower=estimate-half, Upper=estimate+half, CI_Level=level,
                WaldP=float(2*norm.sf(abs(estimate)/se)), Status="model_approximation", Reason="")


def report(analysis):
    entry = analysis["rater_dff"]
    s, audit = entry["settings"], entry["audit"]
    r1, r2 = s["difference"]
    g1, g2 = s["group_difference"]
    target = f"({r1} − {r2}) in {g1} minus {g2}"
    row = dict(Target=target, RaterFacet=s["rater"], GroupFacet=s["group"], WorkColumn=audit["person"],
               Rater1=r1, Rater2=r2, Group1=g1, Group2=g2, Unit="severity logits",
               AssignmentSupport=audit["status"], Estimate=None, SE=None, Lower=None, Upper=None,
               CI_Level=None, WaldP=None, Status="withheld", Reason=audit["status"],
               CoverageQualified=False, SupportsFormalInference=False)
    if audit["status"] == "observed":
        weights = {cell_label(t["rater_level"], t["group_level"]): -t["weight"] for t in audit["contrast_weights"]}
        try:
            row.update(Estimate=_model_point(analysis, s["cell_facet"], s, audit, weights), Status="point_only")
            row.update(_interval(analysis, s["cell_facet"], weights, row["Estimate"]))
        except ValueError as exc:
            row["Reason"] = str(exc)
    else:
        row["Reason"] = {
            "common_location_only": "Observed assignments do not support the contrast with separate work locations.",
            "unsupported": "Observed assignments do not identify the requested group contrast.",
            "audit_size_limit": "Assignment review exceeds the supported calculation size.",
        }.get(audit["status"], audit["status"])
    scores = audit.get("score_support")
    if scores is not None:
        row["AssignmentSupportWithoutEndpointWorks"] = scores["observation_estimable_without_endpoint_works"]
        endpoint_works = set(scores["works_at_minimum"]).union(scores["works_at_maximum"])
    overlap = []
    for group in audit["overlap"]:
        for link in group["path"] or [dict(from_rater=r1, to_rater=r2, persons=[])]:
            overlap.append(dict(Group=group["group_level"], Path=group["kind"], FromRater=link["from_rater"],
                                ToRater=link["to_rater"], SharedWorks=len(link["persons"]),
                                WorkIDs=json.dumps(link["persons"], ensure_ascii=False)))
            if scores is not None:
                overlap[-1]["EndpointWorks"] = len(endpoint_works.intersection(link["persons"]))
    return row, overlap


def tables(analysis):
    row, overlap = report(analysis)
    audit = analysis["rater_dff"]["audit"]
    scores = audit.get("score_support")
    overlap_note = "Shared work IDs on each observed link of the specified comparison."
    if scores is not None:
        overlap_note += " EndpointWorks counts shared works whose full ratings are all at one rubric endpoint."
        overlap_note += " The contrast table checks assignment support after dropping endpoint works for CML review."
    tables = [(TITLE, pd.DataFrame([row]), NOTE),
              (OVERLAP, pd.DataFrame(overlap), overlap_note)]
    if scores is not None:
        rows = [dict(Group=item["group_level"], Rater=item["rater_level"], Score=item["score"], Ratings=item["count"])
                for item in scores["by_rater"]]
        tables.append((SCORES, pd.DataFrame(rows, columns=["Group", "Rater", "Score", "Ratings"]),
                       f"Observed category counts by group and rater. The rubric runs from "
                       f"{scores['rating_min']} to {scores['rating_max']}. Unlisted categories have zero ratings."))
    plan = audit.get("plan")
    if plan is not None:
        summary = [dict(Group=g["group_level"], PlannedCells=g["planned_cells"], ObservedCells=g["observed_cells"],
                        MissingCells=g["missing_cells"], UnplannedCells=g["unplanned_cells"],
                        PlannedSharedWorks=len(g["planned_shared_works"]),
                        ObservedSharedWorks=len(g["observed_shared_works"]),
                        MissingSharedWorkIDs=json.dumps(g["missing_shared_works"], ensure_ascii=False),
                        PlannedSupport=plan["planned_observation_estimable"],
                        SupportIfAllMissingCompleted=plan["completed_observation_estimable"])
                   for g in plan["by_group"]]
        gaps = [dict(Kind="planned_missing", Assignment=json.dumps(e["assignment"], ensure_ascii=False),
                     RestoresContrastOnItsOwn=e["restores_contrast"]) for e in plan["missing"]]
        gaps.extend(dict(Kind="observed_unplanned", Assignment=json.dumps(e, ensure_ascii=False),
                         RestoresContrastOnItsOwn=None) for e in plan["unplanned"])
        tables.extend([
            (PLAN, pd.DataFrame(summary), "Counts of unique work-by-facet cells and directly shared works. "
             "Plan and completion support apply to the whole contrast without anchors."),
            (GAPS, pd.DataFrame(gaps, columns=["Kind", "Assignment", "RestoresContrastOnItsOwn"]),
             "Restoration checks structural support for the contrast if this cell alone were scored. "
             "This check makes no claim about precision or what score would be obtained."),
        ])
    return tables
