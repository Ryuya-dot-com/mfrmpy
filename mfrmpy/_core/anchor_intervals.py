"""Observed-information contrast intervals for fixed rater anchors and a fitted mean."""
from copy import copy

import numpy as np
import pandas as pd

from . import engine, mml_joint, output_qualification, shared_calibration, structural_intervals


def calibration_settings(spec, fixed, means, rater, receiving):
    """Validate labelled anchor covariance and document the sampling relationship."""
    if not isinstance(spec, dict) or spec.get("kind") not in ("known", "independent", "shared"):
        raise ValueError("Calibration kind must be known, independent or shared")
    if not fixed or means or any(row["Facet"] != rater for row in fixed):
        raise ValueError("Calibration intervals require individually fixed levels of the selected rater facet")
    kind = spec["kind"]
    keys = {"known": {"kind"}, "shared": {"kind", "work_ids"},
            "independent": {"kind", "work_ids", "covariance"}}[kind]
    if set(spec) != keys and not (kind == "shared" and set(spec) == keys | {"covariance"}):
        raise ValueError("Supply work_ids for shared or independent calibration and a labelled covariance for independent calibration")
    labels = [row["Level"] for row in fixed]
    if kind == "known":
        return dict(kind=kind, levels=labels, work_ids=[], overlap=[], covariance=None)
    raw = spec["work_ids"]
    if not isinstance(raw, (list, tuple)) or not raw or any(not isinstance(x, str) or not x.strip() for x in raw):
        raise ValueError("Supply nonblank calibration work IDs in the same namespace as receiving IDs")
    work_ids = sorted({x.strip() for x in raw})
    overlap = sorted(set(work_ids).intersection(receiving))
    covariance = None
    if "covariance" in spec:
        frame = spec["covariance"]
        if (not isinstance(frame, pd.DataFrame) or not frame.index.is_unique or not frame.columns.is_unique
                or set(frame.index) != set(labels) or set(frame.columns) != set(labels)):
            raise ValueError("Calibration covariance must be a DataFrame labelled by the fixed rater levels on both axes")
        covariance = frame.loc[labels, labels].to_numpy(dtype=float)
        if not np.isfinite(covariance).all():
            raise ValueError("Calibration covariance must be finite")
        tolerance = 1e-12 * np.max(np.abs(covariance))
        if not np.allclose(covariance, covariance.T, rtol=0., atol=tolerance):
            raise ValueError("Calibration covariance must be symmetric")
        covariance = covariance/2 + covariance.T/2
        if np.linalg.eigvalsh(covariance)[0] < -tolerance:
            raise ValueError("Calibration covariance must be positive semidefinite")
        covariance = covariance.tolist()
    return dict(kind=kind, levels=labels, work_ids=work_ids, overlap=overlap, covariance=covariance)


def fit_with_anchor_origin(data, settings, audit):
    """Use the existing engine with the anchors setting the population origin."""
    options = {k: v for k, v in settings.items() if k not in {
        "anchors", "group_anchors", "umean", "uscale", "compute_pca", "mml_engine", "population_mean",
    }}
    return engine.mfrm_estimate(data, **(options | dict(
        noncenter_facet=audit["rater"], population_formula="~ 1", mml_engine="direct",
        anchor_df=pd.DataFrame(audit["anchors"]), anchor_policy="silent",
    )))


def anchored_information(result, audit, *, include_levels=False):
    """Project the observed information onto free and fixed-value coordinates."""
    cfg, prep, params = (result[k] for k in ("config", "prep", "params"))
    rater = audit["rater"]
    sigma = engine.get_population_prior_sd(cfg)
    q = cfg["quad_points"]
    facets = cfg["facet_names"]
    dimension = sum(len(cfg["facet_levels"][f]) - (f != rater) for f in facets) + cfg["n_cat"] - 2
    cells = len(prep["data"][facets].drop_duplicates())
    if (not 3 <= q <= 121 or not .05 <= sigma <= 10 or dimension < 1
            or (2*q-1)*cells*cfg["n_cat"]*(dimension+1) > 8_000_000
            or cfg["n_person"]*cells*cfg["n_cat"] > 2_000_000
            or len(prep["data"])*cfg["n_cat"]*(dimension+1) > 8_000_000
            or cfg["n_person"]*(2*q-1)*(dimension+1) > 8_000_000
            or (dimension+1)**2 > 8_000_000):
        raise ValueError("Anchored information exceeds the fixed-SD numerical scope")
    fitted_raters = cfg["facet_levels"][rater]
    # The intercept can be absorbed into every rater coefficient without changing
    # any contrast. Holding the fitted anchors fixes their relative locations.
    population, = params["population"]
    values = [params["facets"][f] + population/cfg["facet_signs"][rater] if f == rater
              else params["facets"][f][:-1] for f in facets]
    point = np.r_[*values, params["steps"][:-1], np.log(sigma)]
    start = sum(len(block) for block in values[:facets.index(rater)])
    index = {level: start+i for i, level in enumerate(fitted_raters)}
    fixed = [index[row["Level"]] for row in audit["anchors"]]
    eye = np.eye(dimension)
    free = np.c_[np.delete(eye, fixed, axis=1), eye[:, list(index.values())].sum(axis=1)]
    anchors = eye[:, fixed]
    contrast = np.zeros(dimension)
    for term in audit["contrast_weights"]:
        contrast[index[term["rater_level"]]] = term["weight"]
    if not np.any(contrast @ free):
        raise ValueError("The constraints fix the requested contrast")

    # The unrestricted coordinates give the full mixed derivatives. The
    # recession check uses only the integer design of feasible directions.
    config = {key: cfg[key] for key in ("model", "facet_names", "facet_levels", "facet_signs", "n_cat", "rating_min")}
    problem = mml_joint.GroupedMmlLikelihood(prep["data"], dict(config, noncenter_facet=rater))
    constrained = copy(problem)
    constrained.design = problem.design @ free
    constrained.linear_d = free.shape[1]
    boundary = constrained.structural_recession_audit()
    checks, at_fit = [], None
    for order in (q, 2*q-1):
        nodes, weights = mml_joint.normal_rule(order)
        evaluated = problem.evaluate(point, nodes, weights, information=True, moments=False)
        information = evaluated["information"][:-1, :-1]
        checks.append(dict(points=order, gradient=free.T @ evaluated["gradient"][:-1],
                           information=free.T @ information @ free))
        if at_fit is None:
            at_fit = information
            if not np.isclose(evaluated["nll"], result["opt"].fun, rtol=0, atol=1e-8):
                raise ValueError("Anchored likelihood disagrees with the fitted objective")
    review, covariance = structural_intervals._assess(dict(
        boundary=boundary, calculations=checks, coordinates=np.zeros(free.shape[1]),
        optimizer_success=output_qualification.optimizer_termination_established(result),
    ))
    failed = [row["check"] for row in review if not row["passed"]]
    if failed:
        raise ValueError("; ".join(failed))
    direction = contrast @ free @ covariance
    sensitivity = contrast @ anchors - direction @ (free.T @ at_fit @ anchors)
    variance = float(direction @ (contrast @ free))
    if (not np.isfinite(variance) or variance <= 0 or not np.isfinite(sensitivity).all()
            or not np.isclose(sensitivity.sum(), 0., rtol=0, atol=1e-10)):
        raise ValueError("Anchored contrast variance or sensitivity is invalid")
    output = (float(np.sqrt(variance)), sensitivity, (problem, point, q, free @ direction))
    if not include_levels:
        return output
    # Native coefficients exclude the fitted population shift absorbed into q.
    selected = eye[list(index.values())]
    level_jacobian = selected @ free
    level_jacobian[:, -1] -= 1.
    level_direction = level_jacobian @ covariance
    level_sensitivity = selected @ anchors - level_direction @ (free.T @ at_fit @ anchors)
    return (*output, dict(covariance=(level_direction @ level_jacobian.T).tolist(),
                         sensitivity=level_sensitivity.tolist(),
                         receiving=(problem, point, q, free @ level_direction.T)))


def intervals(result, audit, calibration, source=None):
    if calibration is None:
        if audit["anchors"] or audit["group_anchors"]:
            raise ValueError("Specify an anchor calibration basis to request an anchored interval")
        record = structural_intervals.compute_structural_intervals(
            result, facet=audit["rater"], difference=[t["rater_level"] for t in audit["contrast_weights"]])
        row, = (row for row in record["table"] if row["Kind"] == "difference")
        if not row["CIEligible"]:
            raise ValueError(row["Reason"])
        count = len(result["config"]["facet_levels"][audit["rater"]])
        return dict(se=row["SE"], propagated_se=None, sensitivity=None, propagation_reason="",
                    levels=dict(covariance=np.asarray(record["target_covariance"])[:count, :count].tolist(),
                                propagated_covariance=None, sensitivity=None, reason="", propagation_reason=""))

    se, sensitivity, receiving, levels = anchored_information(result, audit, include_levels=True)
    level_receiving = levels.pop("receiving")
    levels.update(propagated_covariance=None, reason="", propagation_reason="")
    reason = ""
    propagated = None
    evidence = {}
    if source is not None:
        try:
            problem, point, q, directions = level_receiving
            target = shared_calibration.person_influence(problem, point, q, directions)
            calibrated = shared_calibration.calibration_scores(source, audit["rater"], calibration["levels"],
                         np.asarray(levels["sensitivity"]), mapping=calibration.get("source"))
            fine_calibrated, fine_target = calibrated.attrs["fine"], target.attrs["fine"]
            fine_joint = fine_calibrated.add(fine_target, fill_value=0.)
            joint = calibrated.add(target, fill_value=0.)
            covariance = (joint.T @ joint).to_numpy()
            fine_covariance = (fine_joint.T @ fine_joint).to_numpy()
            common = calibrated.index.intersection(target.index)
            cross = (calibrated.loc[common].T @ target.loc[common]).to_numpy()
            fine_cross = (fine_calibrated.loc[common].T @ fine_target.loc[common]).to_numpy()
            scale = max(np.linalg.norm(calibrated)**2 + np.linalg.norm(target)**2, np.finfo(float).tiny)
            change = float(np.linalg.norm(fine_covariance-covariance, 2)/scale)
            cross_change = float(np.linalg.norm(fine_cross-cross, 2)/scale)
            if not np.isfinite([change, cross_change]).all() or max(change, cross_change) > .01:
                raise ValueError("Shared calibration covariance or cross term changes with quadrature")
            evidence["quadrature"] = dict(calibration_points=calibrated.attrs["points"],
                receiving_points=target.attrs["points"], covariance_change=change, cross_change=cross_change,
                calibration_influence_change=calibrated.attrs["relative_change"],
                receiving_influence_change=target.attrs["relative_change"], continuous_reference_checked=False)
            levels["propagated_covariance"] = (joint.T @ joint).to_numpy().tolist()
            levels["shared"] = {key: dict(zip(frame.index, frame.to_numpy().tolist(), strict=True))
                                for key, frame in (("calibration", calibrated), ("receiving", target))}
            labels = result["config"]["facet_levels"][audit["rater"]]
            weights = np.zeros(len(labels))
            for term in audit["contrast_weights"]:
                weights[labels.index(term["rater_level"])] = term["weight"]
            a, b = calibrated @ weights, target @ weights
            shared = dict(calibration=a.to_dict(), receiving=b.to_dict(),
                          **shared_calibration.variance_components(a, b))
            evidence["shared"] = shared
            propagated = shared["se"]
        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
            reason = str(exc)
    elif calibration["kind"] == "shared":
        reason = "Shared sampling units require calibration analysis scores for the cross-sample covariance"
    elif calibration["overlap"]:
        reason = "Calibration and receiving samples share work or sampling units"
    elif calibration["kind"] == "independent":
        v = float(sensitivity @ np.asarray(calibration["covariance"]) @ sensitivity)
        propagated = float(np.hypot(se, np.sqrt(max(v, 0.))))
        s = np.asarray(levels["sensitivity"])
        levels["propagated_covariance"] = (np.asarray(levels["covariance"]) + s @ np.asarray(calibration["covariance"]) @ s.T).tolist()
        if not np.isfinite(propagated):
            reason, propagated = "Nonfinite propagated contrast standard error", None
    if reason:
        levels["propagated_covariance"] = None
        levels.pop("shared", None)
    levels["propagation_reason"] = reason
    return dict(se=se, propagated_se=propagated, sensitivity=sensitivity.tolist(), propagation_reason=reason, levels=levels, **evidence)
