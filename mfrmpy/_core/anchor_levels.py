"""Native rater levels and joint uncertainty retained for sensitivity displays."""
import json

import numpy as np
import pandas as pd
from scipy.stats import norm

from . import project

TITLE = "Anchor sensitivity levels"
NOTE = (
    "Native coefficient logits: FacetSign=-1 lowers the predicted score as the coefficient increases. "
    "Compare levels within a scenario; origins may differ between scenarios. Conditional intervals hold "
    "anchors fixed; propagated intervals include calibration uncertainty. Pointwise 95% normal approximations, "
    "not simultaneous intervals. CoverageQualified=False: contrast coverage does not establish level coverage. "
    "Interval overlap is not a test of the difference; use the retained joint covariance or contrast table. "
    "JointCovariance uses LevelOrder on both axes. Legacy files without level evidence are not refitted."
)


def _array(value, shape):
    try:
        out = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise project.ProjectError("Invalid level uncertainty array") from exc
    if out.shape != shape or not np.isfinite(out).all():
        raise project.ProjectError("Invalid level uncertainty dimensions or values")
    return out


def validate(analysis, entry, row):
    info = row.get("level_estimates")
    if info is None:  # Older archives retained only the contrast.
        return
    labels = analysis["result"]["config"]["facet_levels"][entry["rater"]]
    n = len(labels)
    keys = {"labels", "estimates", "covariance", "propagated_covariance", "sensitivity", "reason", "propagation_reason"}
    if (not isinstance(info, dict) or set(info) - {"shared"} != keys or info["labels"] != labels
            or not all(isinstance(info[k], str) for k in ("reason", "propagation_reason"))):
        raise project.ProjectError("Invalid retained level estimates")
    weights = np.zeros(n)
    weights[labels.index(entry["difference"][0])] = 1
    weights[labels.index(entry["difference"][1])] = -1
    values = info["estimates"]
    if values is None:
        if row["estimate"] is not None or not info["reason"]:
            raise project.ProjectError("Missing level estimates or hold reason")
    else:
        values = _array(values, (n,))
        if row["estimate"] is None or not np.isclose(weights @ values, row["estimate"], atol=1e-10, rtol=1e-10):
            raise project.ProjectError("Level estimates disagree with the contrast")
        if row is entry["scenarios"][0] and not np.allclose(
                values, analysis["result"]["params"]["facets"][entry["rater"]], atol=1e-12, rtol=0):
            raise project.ProjectError("Reference levels disagree with the retained fit")
        for anchor in row["anchors"]:
            if anchor["Facet"] == entry["rater"] and not np.isclose(
                    values[labels.index(anchor["Level"])], anchor["Anchor"], atol=1e-10, rtol=0):
                raise project.ProjectError("Level estimate disagrees with a fixed anchor")
    interval = row.get("interval", {})
    matrices = {}
    for key, se_key in (("covariance", "se"), ("propagated_covariance", "propagated_se")):
        raw, se = info[key], interval.get(se_key)
        if (raw is None) != (se is None):
            raise project.ProjectError("Level and contrast uncertainty availability disagree")
        if raw is not None:
            matrix = _array(raw, (n, n))
            tol = 1e-10 * max(1., float(np.max(np.abs(matrix))))
            if (values is None or not np.allclose(matrix, matrix.T, atol=tol, rtol=0)
                    or np.linalg.eigvalsh(matrix)[0] < -tol
                    or not np.isclose(weights @ matrix @ weights, se**2, atol=1e-12, rtol=1e-9)):
                raise project.ProjectError("Level covariance disagrees with the contrast or is invalid")
            matrices[key] = matrix
    if (info["covariance"] is None) != bool(info["reason"]):
        raise project.ProjectError("Level interval holds need explicit reasons")
    if info["propagation_reason"] != interval.get("propagation_reason", ""):
        raise project.ProjectError("Level propagation reason disagrees with the contrast")
    calibration = row.get("calibration") or {}
    sensitivity = info["sensitivity"]
    if (sensitivity is None) != (interval.get("sensitivity") is None):
        raise project.ProjectError("Level sensitivity availability disagrees with the contrast")
    if sensitivity is not None:
        sensitivity = _array(sensitivity, (n, len(calibration["levels"])))
        if (not np.allclose(sensitivity.sum(axis=1), 1., atol=1e-10, rtol=0)
                or not np.allclose(weights @ sensitivity, interval["sensitivity"], atol=1e-10, rtol=1e-10)):
            raise project.ProjectError("Invalid native level anchor sensitivity")
        for i, label in enumerate(calibration["levels"]):
            j = labels.index(label)
            if (not np.allclose(matrices["covariance"][j], 0., atol=1e-12, rtol=0)
                    or not np.allclose(sensitivity[j], np.eye(len(calibration["levels"]))[i], atol=1e-10, rtol=0)):
                raise project.ProjectError("Fixed anchor uncertainty is not conditional")
    shared = info.get("shared")
    if (shared is None) != (interval.get("shared") is None):
        raise project.ProjectError("Missing joint shared score evidence")
    if shared is not None:
        if not isinstance(shared, dict) or set(shared) != {"calibration", "receiving"}:
            raise project.ProjectError("Invalid joint shared score evidence")
        frames = []
        for key in ("calibration", "receiving"):
            scores = shared[key]
            if not isinstance(scores, dict) or set(scores) != set(interval["shared"][key]):
                raise project.ProjectError("Joint shared score IDs disagree with the contrast")
            array = _array(list(scores.values()), (len(scores), n))
            if not np.allclose(array @ weights, [interval["shared"][key][k] for k in scores], atol=1e-12, rtol=1e-9):
                raise project.ProjectError("Joint shared scores disagree with the contrast")
            frames.append(pd.DataFrame(array, index=list(scores)))
        joint = frames[0].add(frames[1], fill_value=0.)
        expected = (joint.T @ joint).to_numpy()
    elif "propagated_covariance" in matrices:
        if calibration.get("kind") != "independent" or calibration["overlap"]:
            raise project.ProjectError("Level propagation lacks calibration evidence")
        expected = matrices["covariance"] + sensitivity @ np.asarray(calibration["covariance"]) @ sensitivity.T
    else:
        return
    if not np.allclose(matrices["propagated_covariance"], expected, atol=1e-12, rtol=1e-9):
        raise project.ProjectError("Propagated level covariance disagrees with the retained evidence")


def table(analysis):
    entry, result = analysis["anchor_sensitivity"], analysis["result"]
    labels = result["config"]["facet_levels"][entry["rater"]]
    critical, rows = norm.ppf(.975), []
    for number, scenario in enumerate(entry["scenarios"]):
        calibration = scenario.get("calibration") or {}
        info = scenario.get("level_estimates")
        if info is None:
            info = dict(estimates=result["params"]["facets"][entry["rater"]].tolist()
                        if number == 0 and scenario["estimate"] is not None else None,
                        reason=scenario["reason"] or "Individual level uncertainty not retained in this file")
        bases = [("conditional" if scenario["anchors"] or scenario["group_anchors"] else "no_anchors", "covariance", "reason")]
        if calibration.get("kind") in ("independent", "shared"):
            bases.append(("shared_calibration" if calibration.get("source") else "calibration_propagated",
                          "propagated_covariance", "propagation_reason"))
        origin = ("Anchor values; estimated person mean" if calibration else
                  f"Fixed person mean = 0; noncenter facet = {scenario['noncenter_facet']}")
        for basis, key, reason_key in bases:
            covariance = info.get(key)
            for i, label in enumerate(labels):
                estimate = info["estimates"][i] if info["estimates"] is not None else None
                se = float(np.sqrt(max(0., covariance[i][i]))) if covariance is not None else None
                reason = (info.get(reason_key) or info.get("reason", "")) if se is None else ""
                fixed = (key == "covariance" and estimate is not None and any(
                    a["Facet"] == entry["rater"] and a["Level"] == label for a in scenario["anchors"]))
                status = "fixed" if fixed else ("model_approximation" if se is not None else
                                               "point_only" if estimate is not None else "withheld")
                rows.append(dict(Scenario=scenario["name"], Basis=basis, RaterFacet=entry["rater"], Level=label,
                                 Estimate=estimate, SE=se, Lower=estimate-critical*se if se is not None else None,
                                 Upper=estimate+critical*se if se is not None else None,
                                 Width=2*critical*se if se is not None else None, CILevel=.95,
                                 Origin=origin, FacetSign=result["config"]["facet_signs"][entry["rater"]],
                                 Status=status, Reason=reason, CoverageQualified=False, SupportsFormalInference=False,
                                 LevelOrder=json.dumps(labels, ensure_ascii=False) if covariance is not None else None,
                                 JointCovariance=json.dumps(covariance) if covariance is not None else None))
    return pd.DataFrame(rows)
