"""Person-cluster sandwich for a contrast with individually calibrated rater anchors."""
import numpy as np
import pandas as pd
from scipy.special import softmax

from . import engine, mml_joint, project, structural_intervals


def retained_quadrature_hold(scenario):
    """Old shared archives keep their points, not unreviewed propagated SEs."""
    interval = scenario.get("interval") or {}
    if ((scenario.get("calibration") or {}).get("source")
            and interval.get("propagated_se") is not None and "quadrature" not in interval):
        return "Shared calibration influence and cross-term quadrature evidence was not saved."
    return ""


def _rating_rows(analysis, column, facets):
    raw = analysis["input_data"]
    prep = analysis["result"]["prep"]
    audit = prep.get("row_audit")
    if column not in raw or audit is None or len(raw) != len(audit):
        raise ValueError("Linking reused ratings requires the rating ID column and fitted row audit in both analyses")
    ids = raw.loc[:, column].iloc[audit["IncludedInLikelihood"].to_numpy(dtype=bool)]
    if not ids.map(lambda value: isinstance(value, str) and bool(value.strip())).all():
        raise ValueError("Each fitted rating needs a nonblank string rating ID")
    ids = pd.Index(ids.str.strip())
    if ids.has_duplicates or len(ids) != len(prep["data"]):
        raise ValueError("Rating IDs must uniquely identify the fitted rows within each sample")
    rows = prep["data"][["Person", *facets, "Score"]].astype(object).copy()
    rows.index = ids
    return rows


def source_settings(source, target, fixed, rater, rating_id, *, facet=None, level_map=None, occasion=None):
    """Validate a shared source before fitting; retain its identity without copying the project."""
    project.validate_analysis(source)
    project.require_reportable_input(source)
    if "result" not in source:
        raise ValueError("Shared calibration requires a fitted MFRM analysis")
    src, dst = (a["result"]["config"] for a in (source, target))
    mapped = facet is not None or level_map is not None
    source_facet = facet if mapped else rater
    if (source_facet not in src["facet_names"] or
            (src["method"], src["model"]) != ("MML", "RSM")
            or source["fit_settings"].get("anchors") or source["fit_settings"].get("group_anchors")
            or (src["rating_min"], src["n_cat"]) != (dst["rating_min"], dst["n_cat"])):
        raise ValueError("Use unanchored fixed-SD MML RSM calibration with a fitted source facet and matching rating scale")
    if mapped:
        other = [name for name in src["facet_names"] if name != source_facet]
        if (not isinstance(level_map, dict) or set(level_map) != set(dst["facet_levels"][rater])
                or any(not isinstance(v, str) or v not in src["facet_levels"][source_facet] for v in level_map.values())):
            raise ValueError("level_map must map every receiving rater level to an observed source facet level")
        if ((source_facet != rater and source_facet in dst["facet_names"])
                or src["facet_signs"][source_facet] != dst["facet_signs"][rater]
                or any(name == rater or name not in dst["facet_names"]
                       or src["facet_signs"][name] != dst["facet_signs"][name] for name in other)):
            raise ValueError("Mapped rater orientation and remaining source facet names/signs must match")
    else:
        if src["noncenter_facet"] != "Person" or src["facet_signs"] != dst["facet_signs"]:
            raise ValueError("Use centered calibration with matching facet signs, or specify facet and level_map explicitly")
        level_map = {level: level for level in dst["facet_levels"][rater]}
    values = dict(zip(src["facet_levels"][source_facet], source["result"]["params"]["facets"][source_facet], strict=True))
    for row in fixed:
        if level_map[row["Level"]] not in values or not np.isclose(row["Anchor"], values[level_map[row["Level"]]], atol=1e-12, rtol=0):
            raise ValueError("Fixed anchor levels and native logit values must match the calibration analysis")
    reused = None
    if rating_id is not None:
        if not isinstance(rating_id, str) or not rating_id.strip():
            raise ValueError("rating_id must name a rating-event ID column")
        source_rows = _rating_rows(source, rating_id, src["facet_names"])
        target_rows = _rating_rows(target, rating_id, dst["facet_names"])
        target_rows[rater] = target_rows[rater].map(level_map)
        target_rows = target_rows.rename(columns={rater: source_facet}).reindex(columns=source_rows.columns).astype(object)
        common = source_rows.index.intersection(target_rows.index)
        if mapped and occasion is not None:
            contexts = []
            for analysis, rows in ((source, source_rows), (target, target_rows)):
                if occasion not in analysis["input_data"]:
                    raise ValueError("Reused annual rating IDs require the occasion column in both inputs")
                included = analysis["result"]["prep"]["row_audit"].IncludedInLikelihood.to_numpy(dtype=bool)
                labels = analysis["input_data"][occasion].iloc[included]
                if not labels.map(lambda value: isinstance(value, str) and bool(value.strip())).all():
                    raise ValueError("Reused annual rating IDs require nonblank occasion labels")
                contexts.append(pd.Series(labels.str.strip().to_numpy(dtype=object), index=rows.index).loc[common])
            if not contexts[0].equals(contexts[1]):
                raise ValueError("The same rating ID refers to different occasions across samples")
        if not source_rows.loc[common].equals(target_rows.loc[common]):
            raise ValueError("The same rating ID has different work, facet levels or score across samples")
        reused = len(common)
    record = dict(analysis_id=source["analysis_id"], fingerprint=source["content_fingerprint"],
                population_sd=engine.get_population_prior_sd(src), quadrature_points=int(src["quad_points"]),
                rating_id=rating_id, reused_ratings=reused)
    if mapped:
        record.update(facet=source_facet, level_map=dict(level_map), estimates={k: float(v) for k, v in values.items()})
    return record


def person_influence(problem, point, order, direction, *, check=True):
    """Marginal scores by person for one direction or a matrix of columns."""
    direction = np.asarray(direction)
    scalar = direction.ndim == 1
    if scalar:
        direction = direction[:, None]
    nodes, weights = mml_joint.normal_rule(order)
    evaluated = problem.evaluate(point, nodes, weights, moments=False)
    predictor = problem.design @ point[:-1]
    probability = softmax(predictor + np.exp(point[-1])*nodes[:, None, None]*problem.k, axis=-1)
    derivative = problem.design @ direction
    observed = np.einsum("nik,ikd->nd", problem.counts, derivative)
    expected = np.einsum("qik,ikd->qid", probability, derivative)
    score = observed - np.einsum("nq,ni,qid->nd", evaluated["posterior"], problem.cell_totals, expected, optimize=True)
    if not np.isfinite(score).all() or not np.allclose(score.sum(axis=0), -evaluated["gradient"][:-1] @ direction,
                                                       atol=1e-9, rtol=1e-9):
        raise ValueError("Person contributions disagree with the marginal likelihood gradient")
    output = (pd.Series(score[:, 0], index=problem.person_labels) if scalar else
              pd.DataFrame(score, index=problem.person_labels))
    if check:
        fine = person_influence(problem, point, 2*order-1, direction, check=False)
        scale = np.maximum(np.linalg.norm(score, axis=0), np.finfo(float).tiny)
        fine_values = fine.to_numpy().reshape(score.shape)
        change = float(np.max(np.linalg.norm(fine_values-score, axis=0)/scale))
        if not np.isfinite(change) or change > .01:
            raise ValueError("Shared calibration person contributions change with quadrature")
        output.attrs.update(fine=fine, points=[order, 2*order-1], relative_change=change)
    return output


def calibration_scores(source, rater, levels, sensitivity, mapping=None):
    """Joint calibration influence for native coefficients or contrasts."""
    result = source["result"]
    cfg = result["config"]
    if mapping and "level_map" in mapping:
        rater = mapping["facet"]
        levels = [mapping["level_map"][level] for level in levels]
    record = structural_intervals.compute_structural_intervals(result, facet=rater)
    failed = [check["check"] for check in record["checks"] if not check["passed"]]
    if failed:
        raise ValueError("Shared calibration source: " + ", ".join(failed))
    ix = {row["Level"]: i for i, row in enumerate(record["targets"])}
    # Repeated source rows represent the same estimate, with perfect covariance.
    jacobian = np.asarray(record["target_jacobian"])[[ix[level] for level in levels]]
    directions = np.linalg.solve(record["calculations"][0]["information"], (sensitivity @ jacobian).T)
    problem = mml_joint.GroupedMmlLikelihood(result["prep"]["data"], cfg, allow_all_centered=True)
    point = np.r_[record["coordinates"], np.log(engine.get_population_prior_sd(cfg))]
    return person_influence(problem, point, cfg["quad_points"], directions)


def variance_components(calibration, receiving):
    shared = calibration.index.intersection(receiving.index)
    joint = calibration.add(receiving, fill_value=0.)
    variance = float(joint @ joint)
    if not np.isfinite(variance) or variance <= 0:
        raise ValueError("Shared calibration contrast variance is nonpositive or nonfinite")
    return dict(calibration_variance=float(calibration @ calibration),
                receiving_variance=float(receiving @ receiving),
                cross_term=float(2 * (calibration.loc[shared] @ receiving.loc[shared])), se=float(np.sqrt(variance)))
