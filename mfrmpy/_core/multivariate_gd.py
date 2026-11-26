"""Balanced crossed covariance components and explicitly weighted composites."""
import itertools
import json
import math

import numpy as np
import pandas as pd

from mfrmpy._core import generalizability as gd
from mfrmpy._core.engine import dataframe_fingerprint


VERSION = "balanced_multivariate_anova_v1"
CI_REASON = "Multivariate G/Phi intervals are not implemented or qualified."
CAUTION = (
    "Complete balanced crossed observed scores on the same cells. Raw covariance components "
    "are retained, including negative entries and non-PSD matrices; inspect component diagnostics. "
    "For each score/composite, project each component as w' Sigma w, then clip negative scalar "
    "variances at zero for G/Phi. This is not an entrywise or PSD matrix correction. "
    "Highest-order interaction is confounded with residual error. "
    "Weights are used as supplied, without normalization or score standardization. "
    "SEMs are measurement-error SDs, not sampling SEs of G/Phi. " + CI_REASON
)


def validate_settings(settings):
    reserved = {"Target", "TargetType", "Weights", "GStatus", "PhiStatus", "UniverseVariance",
                "RelativeErrorVariance", "AbsoluteErrorVariance", "RelativeSEM", "AbsoluteSEM",
                "ComponentPSD", "ComponentTreatment", "ClippedComponents", "IsSelectedPlan"}
    if set(settings.get("random_facets", [])) & reserved:
        raise ValueError("Random facet names must not use multivariate output column names")
    scores = settings.get("score_cols")
    if (not isinstance(scores, (list, tuple)) or len(scores) < 2
            or any(not isinstance(s, str) or not s.strip() for s in scores)
            or len(set(scores)) != len(scores)
            or set(scores) & {settings.get("object_facet"), *settings.get("random_facets", []), "Weight"}):
        raise ValueError("score_cols must name at least two distinct score columns, separate from facets and Weight")
    weights = settings.get("composite_weights")
    if (not isinstance(weights, dict) or set(weights) != set(scores)
            or any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, float, np.integer, np.floating))
                   or not np.isfinite(v) or v < 0 for v in weights.values())
            or not any(v > 0 for v in weights.values())):
        raise ValueError("composite_weights must name every score with finite nonnegative weights and at least one positive weight")
    if settings.get("coefficient_intervals") is not None:
        raise ValueError(CI_REASON)
    fc = settings.get("fixed_condition")
    if fc is not None and fc.get("facet") in scores:
        raise ValueError("The fixed conditioning facet must be separate from all score columns")
    return list(scores), {s: float(weights[s]) for s in scores}


def prepare(data, settings):
    """Require the same validated cells for every score; never delete pairwise rows."""
    scores, weights = validate_settings(settings)
    obj, facets = settings["object_facet"], settings["random_facets"]
    work = None
    design = dict(score_basis="original_observed_scores", score_columns=scores,
                  composite_weights=weights, object_facet=obj, random_facets=facets,
                  n_input_rows=len(data) if isinstance(data, pd.DataFrame) else 0,
                  n_observations=0, target_population=settings.get("target_population", ""))
    if not isinstance(data, pd.DataFrame) or data.empty:
        if settings.get("fixed_condition") and isinstance(data, pd.DataFrame):
            design.update(raw_data_fingerprint=dataframe_fingerprint(data, length=64),
                          fixed_condition=settings["fixed_condition"])
        return None, design, "missing_data"
    for score in scores:
        prepared, reason, fixed = gd.prepare_gd_input_data(data, dict(settings, score_col=score))
        if not reason:
            current, checked, reason = gd._prepare_g_d_data(prepared, obj, facets, score)
            design.update(checked)
        if reason:
            design["failed_score_column"] = score
            break
        if work is None:
            work = current.set_index([obj, *facets])
        else:
            work[score] = current.set_index([obj, *facets])[score].reindex(work.index)
    design.update(score_basis="original_observed_scores", score_columns=scores, composite_weights=weights)
    design.pop("score_column", None)
    design.pop("grand_mean", None)
    if settings.get("fixed_condition"):
        design.update(raw_data_fingerprint=dataframe_fingerprint(data, length=64),
                      fixed_condition=settings["fixed_condition"], fixed_condition_applied=fixed,
                      generalization_basis="fixed_facet_conditioned_observed_scores")
    else:
        design["data_fingerprint"] = dataframe_fingerprint(data, length=64)
        design["generalization_basis"] = "all_non_object_facets_random"
    return None if reason else work.reset_index(), design, reason


def covariance_components(work, design):
    """Orthogonal tensor projections; no observation-by-observation dense matrix."""
    facets = [design["object_facet"], *design["random_facets"]]
    scores = design["score_columns"]
    cats = [pd.Categorical(work[f]) for f in facets]
    shape = tuple(len(c.categories) for c in cats)
    values = work[scores].to_numpy(dtype=float)
    # Normalize only for arithmetic; all returned components use original score units.
    values = values - values[0]
    scale = np.max(np.abs(values), axis=0)
    scale[scale == 0] = 1.
    normalized = values / scale
    normalized -= normalized.mean(axis=0)
    normalized -= normalized.mean(axis=0)
    tensor = np.empty((*shape, len(scores)))
    tensor[tuple(c.codes for c in cats)] = normalized
    subsets = [s for size in range(1, len(shape)+1) for s in itertools.combinations(range(len(shape)), size)]
    sources = [":".join(facets[i] for i in s) for s in subsets]
    sources[-1] = "Residual"
    dfs = np.array([math.prod(shape[i]-1 for i in s) for s in subsets])
    mean_products = []
    for s, df in zip(subsets, dfs):
        outside = tuple(i for i in range(len(shape)) if i not in s)
        effect = tensor.mean(axis=outside, keepdims=True) if outside else tensor.copy()
        for _ in range(2):
            for i in s:
                effect = effect - effect.mean(axis=i, keepdims=True)
        rows = effect.reshape(-1, len(scores))
        mean_products.append((rows.T @ rows) * math.prod(shape[i] for i in outside) / df)
    expected = np.array([[math.prod(shape[i] for i in range(len(shape)) if i not in t)
                          if set(s) <= set(t) else 0 for t in subsets] for s in subsets], dtype=float)
    raw_scaled = np.linalg.solve(expected, np.asarray(mean_products).reshape(len(subsets), -1)).reshape(
        len(subsets), len(scores), len(scores))
    raw = (raw_scaled * scale[None, :, None]) * scale[None, None, :]
    if not np.isfinite(raw).all():
        raise ValueError("Covariance components exceed numeric range; rescale scores")
    matrices = {s: pd.DataFrame(m, index=scores, columns=scores) for s, m in zip(sources, raw)}
    score_sd = normalized.std(axis=0, ddof=1)
    score_sd[score_sd == 0] = 1.
    diagnostics = []
    for source, matrix in zip(sources, raw_scaled):
        standardized = matrix / score_sd[:, None] / score_sd[None, :]
        eigenvalues = np.linalg.eigvalsh(standardized)
        tolerance = np.sqrt(np.finfo(float).eps) * max(1., np.max(np.abs(eigenvalues)))
        diagnostics.append(dict(Source=source, MinimumScaledEigenvalue=float(eigenvalues.min()),
                                Tolerance=tolerance, PositiveSemidefinite=bool(eigenvalues.min() >= -tolerance),
                                Rank=int(np.sum(eigenvalues > tolerance)), Dimension=len(scores)))
    return matrices, pd.DataFrame(diagnostics), dfs


def scalar_components(matrices, dfs, weights):
    rows = []
    for (source, matrix), df in zip(matrices.items(), dfs):
        with np.errstate(over="ignore", invalid="ignore"):
            raw = float(weights @ matrix.to_numpy() @ weights)
        if not np.isfinite(raw):
            raise ValueError("Composite variance exceeds numeric range; rescale weights or scores")
        weight_scale = float(np.max(np.abs(weights)))
        if raw == 0 and weight_scale > 0:
            unit_weights = weights / weight_scale
            with np.errstate(over="ignore", invalid="ignore"):
                unit_raw = float(unit_weights @ matrix.to_numpy() @ unit_weights)
            if (np.isfinite(unit_raw) and unit_raw != 0 and math.log(abs(unit_raw)) + 2*math.log(weight_scale)
                    < math.log(np.nextafter(0., 1.))):
                raise ValueError("Composite variance underflows numeric range; rescale weights or scores")
        rows.append(dict(Source=source, Term="residual" if source == "Residual" else
                         ("two-way" if ":" in source else "main"), df=int(df) if np.isfinite(df) else np.nan,
                         RawVariance=raw, Variance=max(0., raw)))
    frame = pd.DataFrame(rows)
    with np.errstate(over="ignore", invalid="ignore"):
        total = frame.Variance.sum()
    if not np.isfinite(total):
        raise ValueError("Total projected variance exceeds numeric range; rescale weights or scores")
    frame["ProportionVariance"] = frame.Variance / total if total > 0 else np.nan
    return frame


def study(data, settings):
    work, design, reason = prepare(data, settings)
    result = dict(available=False, reason=gd._GD_INPUT_REASONS.get(reason, ""), reason_code=reason,
                  calculation_version=VERSION, design=design, caveat=CAUTION,
                  reference="Brennan (2001), multivariate generalizability theory.",
                  variance_components=pd.DataFrame(), observed_coefficients={"G": np.nan, "Phi": np.nan},
                  d_study=pd.DataFrame(), bootstrap_ci=dict(requested=False, inference_allowed=False,
                  inference_reason=CI_REASON), multivariate=dict(score_columns=settings["score_cols"],
                  weights=settings["composite_weights"]))
    if reason:
        return result
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        try:
            matrices, diagnostics, dfs = covariance_components(work, design)
        except (ValueError, np.linalg.LinAlgError) as exc:
            result.update(reason_code="invalid_components", reason=str(exc))
            return result
    return project_components(result, settings, matrices, diagnostics, dfs)


def project_components(result, settings, matrices, diagnostics, dfs, coefficient_fn=None):
    """Retain score/composite projections with explicit observed and future counts."""
    design = result["design"]
    scores = design["score_columns"]
    multivariate = "score_cols" in settings
    record_key = "multivariate" if multivariate else "design_study"
    vectors = [("score", s, np.eye(len(scores))[i]) for i, s in enumerate(scores)]
    if multivariate:
        vectors.append(("composite", "Composite", np.array([settings["composite_weights"][s] for s in scores])))
    if coefficient_fn is None:
        def coefficient_fn(components, counts):
            return gd._g_phi_from_variance_components(components, object_facet=settings["object_facet"],
                                                       facet_observations=counts)
    facets = settings["random_facets"]
    observed = design.get("observed_n", {})
    selected = settings["selected_plan"] or observed
    points = gd._planned_facet_grid(settings["n_max_per_facet"])
    plans = set(itertools.product(points, repeat=len(facets))) if selected else set()
    for counts in (observed, selected):
        if counts:
            plans.add(tuple(counts[f] for f in facets))
    coefficient_rows, component_rows, observed_rows = [], [], []
    for kind, target, weights in vectors:
        try:
            components = scalar_components(matrices, dfs, weights)
        except ValueError as exc:
            result.update(reason_code="invalid_components", reason=str(exc))
            return result
        label = dict(TargetType=kind, Target=target,
                     Weights=json.dumps(dict(zip(scores, weights.tolist())), ensure_ascii=False, sort_keys=True))
        component_rows.append(components.assign(**label))
        treatment = dict(ComponentTreatment="clip_scalar_after_projection",
                         ClippedComponents="; ".join(components.loc[components.RawVariance < 0, "Source"]),
                         ComponentPSD=bool(diagnostics.PositiveSemidefinite.all()))
        for counts in sorted(plans):
            counts = dict(zip(facets, counts))
            coefficients = coefficient_fn(components, counts)
            row = dict(**label, **counts, **gd._g_d_error_row(coefficients),
                       UniverseVariance=coefficients["details"]["sigma2_object"],
                       IsObservedDesign=counts == observed, IsSelectedPlan=counts == selected, **treatment)
            row.update({k+"Status": "point_estimate" if np.isfinite(row[k]) else "undefined_zero_denominator"
                        for k in ("G", "Phi")})
            coefficient_rows.append(row)
            if counts == observed:
                observed_rows.append(row)
        if not observed:
            observed_rows.append(dict(**label, **{f: np.nan for f in facets}, **treatment,
                G=np.nan, Phi=np.nan, GStatus="no_balanced_observed_design", PhiStatus="no_balanced_observed_design",
                Reason=design.get("observed_coefficient_reason", "Observed replication counts are not defined."),
                IsObservedDesign=True, IsSelectedPlan=False))
        if kind == "composite" or not multivariate:
            result["variance_components"] = components
            result["observed_coefficients"] = (coefficient_fn(components, observed) if observed else
                dict(G=np.nan, Phi=np.nan, status="no_balanced_observed_design"))
    columns = ["TargetType", "Target", "Weights", *facets, "G", "Phi", "IsObservedDesign", "IsSelectedPlan"]
    all_plans = pd.DataFrame(coefficient_rows) if coefficient_rows else pd.DataFrame(columns=columns)
    primary = all_plans.loc[all_plans.TargetType.eq("composite" if multivariate else "score")].reset_index(drop=True)
    chosen = primary.loc[primary.IsSelectedPlan.eq(True)]
    result.update(available=True, reason="", reason_code="", d_study=primary,
                  selected_plan=chosen.iloc[0].to_dict() if not chosen.empty else {})
    result[record_key].update(covariance_components=matrices, component_diagnostics=diagnostics,
        variance_components=pd.concat(component_rows, ignore_index=True), d_study=all_plans,
        observed=pd.DataFrame(observed_rows), selected=all_plans.loc[all_plans.IsSelectedPlan.eq(True)].reset_index(drop=True))
    return result


def export_frame(bundle):
    base = dict(bundle)
    multi = base.pop("multivariate", None)
    if multi is None:
        multi = base.pop("design_study")
    base["available"] = False  # Obtain metadata only; coefficient tables are retained below.
    frame = gd.g_d_study_export_frame(base)
    frame.loc[frame.Field.eq("available"), "Value"] = str(bundle["available"])
    metadata = frame._section.eq("metadata")
    for field, value in dict(variance_policy="Raw covariance matrices; scalar negative variances clipped after projection",
                             interval_hold_reason=CI_REASON, interval_scope=CI_REASON).items():
        frame.loc[metadata & frame.Field.eq(field), "Value"] = value
    if not bundle["available"]:
        return frame
    frame = frame.loc[metadata]
    frames = [frame]
    for key, section in (("observed", "observed_coefficients"), ("selected", "selected_d_study"),
                         ("variance_components", "variance_components"), ("d_study", "d_study"),
                         ("component_diagnostics", "component_diagnostics")):
        frames.append(multi[key].assign(_section=section))
    entries = [dict(Source=source, ScoreA=a, ScoreB=b, Covariance=matrix.loc[a, b], _section="covariance_components")
               for source, matrix in multi["covariance_components"].items() for a in matrix.index for b in matrix.columns]
    frames.append(pd.DataFrame(entries))
    return pd.concat(frames, ignore_index=True)
