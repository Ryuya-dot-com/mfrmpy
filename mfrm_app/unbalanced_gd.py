"""MINQUE(0) for observed crossed cells or one random facet nested in another."""
import itertools
import json
import math

import numpy as np
import pandas as pd

from mfrm_app import generalizability as gd, multivariate_gd as mv
from mfrm_app.engine import dataframe_fingerprint

VERSION = "minque0_gd_v1"
CI_REASON = "MINQUE(0) G/Phi intervals are not implemented or qualified."
PLAN_REASON = "The observed design is unbalanced; specify selected_plan for a future balanced mean. Pooled level counts are not replications."
CAUTION = (
    "MINQUE(0) assumes independent random effects and a common covariance component across levels, "
    "conditional on the retained assignment. Explicit listwise omission does not correct informative missingness. "
    "D-study projects a balanced future mean; nested child counts are per parent. "
    "Raw covariance components are retained; negative scalar projections are clipped at zero for G/Phi. "
    "This is not a PSD matrix correction. Highest-order interaction is confounded with residual error. "
    "Weights are not normalized. SEMs are measurement-error SDs, not sampling SEs of G/Phi. " + CI_REASON
)
REASONS = dict(gd._GD_INPUT_REASONS, missing_values="Missing selected scores or labels require missing='omit'.",
               unidentified_components="The centered covariance kernels are numerically rank deficient at the recorded tolerance; this method cannot separate all requested components.")


def validate_settings(settings):
    method = settings.get("method", "anova")
    if not isinstance(method, str) or method not in ("anova", "minque0"):
        raise ValueError("G/D method must be anova or minque0")
    if method != "minque0":
        if "missing" in settings or "nesting" in settings:
            raise ValueError("missing and nesting require method='minque0'")
        return
    if settings.get("missing", "error") not in ("error", "omit"):
        raise ValueError("missing must be error or omit")
    if settings.get("coefficient_intervals") is not None:
        raise ValueError(CI_REASON)
    if settings.get("fixed_condition") is not None:
        raise ValueError("MINQUE(0) does not support fixed_condition aggregation")
    obj, facets = settings["object_facet"], settings["random_facets"]
    scores = settings.get("score_cols", [settings.get("score_col", "Score")])
    reserved = gd._RESERVED_GD_NAMES | {"Target", "TargetType", "Weights", "GStatus", "PhiStatus",
        "UniverseVariance", "RelativeErrorVariance", "AbsoluteErrorVariance", "RelativeSEM", "AbsoluteSEM",
        "ComponentPSD", "ComponentTreatment", "ClippedComponents", "IsSelectedPlan", "Reason", "Weight"}
    if (len(facets) not in (1, 2) or len(set([obj, *facets])) != len(facets)+1
            or any(not f.strip() or any(c in f for c in ":()") or f in reserved for f in [obj, *facets])
            or any(not isinstance(s, str) or not s.strip() for s in scores)
            or set(scores) & {obj, *facets, "Weight"}):
        raise ValueError("MINQUE(0) requires one or two distinct random facets, separate score columns and unreserved names")
    nesting = settings.get("nesting")
    if nesting is not None and (not isinstance(nesting, dict) or len(nesting) != 1 or len(facets) != 2
            or any(not isinstance(v, str) for v in nesting.values())
            or set(nesting) | set(nesting.values()) != set(facets)):
        raise ValueError("nesting must map one random child facet to the other random parent facet")


def prepare(data, settings):
    validate_settings(settings)
    obj, facets = settings["object_facet"], settings["random_facets"]
    scores = settings.get("score_cols", [settings.get("score_col", "Score")])
    nesting = settings.get("nesting") or {}
    design = dict(score_basis="original_observed_scores", score_columns=scores, object_facet=obj,
                  random_facets=facets, nesting=nesting, missing=settings.get("missing", "error"),
                  method="minque0", n_input_rows=len(data) if isinstance(data, pd.DataFrame) else 0,
                  n_observations=0, n_omitted_rows=0, observed_n={}, target_population=settings["target_population"],
                  weight_basis="unit_observation_weights", generalization_basis="random_effects_conditional_on_assignment")
    if "score_cols" in settings:
        design["composite_weights"] = settings["composite_weights"]
    excluded = pd.DataFrame(columns=["RowPosition", "InputIndex", "MissingColumns", "Action"])
    if not isinstance(data, pd.DataFrame) or data.empty:
        return None, design, excluded, "missing_data"
    design["data_fingerprint"] = dataframe_fingerprint(data, length=64)
    columns = [obj, *facets, *scores]
    if not data.columns.is_unique or any(c not in data.columns for c in columns):
        return None, design, excluded, "invalid_columns"
    work = data[columns].copy()
    if "Weight" in data:
        weights = pd.to_numeric(data.Weight, errors="coerce").to_numpy(dtype=float)
        if not (np.isfinite(weights).all() and (weights == 1).all()):
            return None, design, excluded, "unsupported_weights"
    for score in scores:
        numeric = pd.to_numeric(work[score], errors="coerce")
        invalid = work[score].notna() & ~np.isfinite(numeric.to_numpy(dtype=float))
        if invalid.any():
            return None, design, excluded, "invalid_scores"
        work[score] = numeric
    for facet in [obj, *facets]:
        labels = work[facet].dropna()
        if (labels.astype(str).str.strip().eq("").any() or labels.nunique() != labels.astype(str).nunique()
                or any(isinstance(v, (float, np.floating)) and not np.isfinite(v) for v in labels)):
            return None, design, excluded, "invalid_labels"
    identifiers = work[[obj, *facets]].dropna().astype(str)
    if identifiers.duplicated().any():
        return None, design, excluded, "duplicate_cells"
    missing = work.isna()
    positions = np.flatnonzero(missing.any(axis=1).to_numpy())
    excluded = pd.DataFrame([dict(RowPosition=int(i), InputIndex=str(data.index[i]),
        MissingColumns=json.dumps(list(missing.columns[missing.iloc[i]]), ensure_ascii=False),
        Action="omitted" if design["missing"] == "omit" else "rejected_missing") for i in positions],
        columns=excluded.columns)
    design["n_rows_with_missing"] = len(positions)
    if len(positions) and design["missing"] == "error":
        return None, design, excluded, "missing_values"
    work = work.iloc[np.flatnonzero(~missing.any(axis=1).to_numpy())].copy()
    design.update(n_omitted_rows=len(positions), n_observations=len(work))
    for f in [obj, *facets]:
        work[f] = work[f].astype(str)
    levels = {f: int(work[f].nunique()) for f in [obj, *facets]}
    design["levels"] = levels
    if work.empty or levels[obj] < 2 or any(levels[f] < 2 for f in facets if f not in nesting):
        return None, design, excluded, "insufficient_levels"
    observed = {f: levels[f] for f in facets}
    if nesting:
        child, parent = next(iter(nesting.items()))
        child_counts = work[[parent, child]].drop_duplicates().groupby(parent, sort=True).size()
        expected = levels[obj] * int(child_counts.sum())
        balanced = len(work) == expected and child_counts.nunique() == 1
        observed[child] = int(child_counts.iloc[0])
        design.update(child_counts={str(k): int(v) for k, v in child_counts.items()},
                      nested_child_units=int(child_counts.sum()),
                      count_convention=f"{child} counts are per {parent}; {parent} counts are distinct parent levels")
        terms = [(obj, [obj]), (parent, [parent]), (f"{child}({parent})", [parent, child]),
                 (f"{obj}:{parent}", [obj, parent]), ("Residual", [obj, parent, child])]
    else:
        expected = math.prod(levels.values())
        balanced = len(work) == expected
        factors = [obj, *facets]
        terms = [(":".join(s), list(s)) for n in range(1, len(factors)+1)
                 for s in itertools.combinations(factors, n)]
        terms[-1] = ("Residual", factors)
        design["count_convention"] = "Counts are common random facet levels in a balanced crossed future mean"
    design.update(n_expected_cells=expected, balanced=balanced, complete=len(work) == expected,
                  component_factors=dict(terms), observed_n=observed if balanced else {},
                  observed_coefficient_reason="" if balanced else PLAN_REASON,
                  planning_status="balanced_mean_projection" if balanced or settings["selected_plan"] else "selected_plan_required")
    return work, design, excluded, ""


def kernel_audit(work, design):
    """tr(P K_i P K_j) from group counts; memory grows with rows, not rows squared."""
    groups = [pd.factorize(pd.MultiIndex.from_frame(work[cols]), sort=True)[0]
              for cols in design["component_factors"].values()]
    sizes = [np.bincount(g).astype(float) for g in groups]
    row_sizes = [counts[g] for counts, g in zip(sizes, groups)]
    totals = [r.sum() for r in row_sizes]
    n, k = len(work), len(groups)
    gram = np.empty((k, k))
    for i in range(k):
        for j in range(i+1):
            joint = pd.factorize(pd.MultiIndex.from_arrays([groups[i], groups[j]]), sort=False)[0]
            counts = np.bincount(joint).astype(float)
            value = counts @ counts - 2 * (row_sizes[i] @ row_sizes[j]) / n + totals[i]*totals[j]/n**2
            gram[i, j] = gram[j, i] = value
    scale = np.sqrt(np.maximum(np.diag(gram), 0.))
    normalized = np.divide(gram, np.outer(scale, scale), out=np.zeros_like(gram), where=np.outer(scale, scale)>0)
    eigenvalues = np.linalg.eigvalsh(normalized)
    tolerance = np.sqrt(np.finfo(float).eps) * max(float(eigenvalues.max()), 0.)
    rank = int(np.sum(eigenvalues > tolerance))
    names = list(design["component_factors"])
    audit = dict(rank=rank, n_components=k, identified=rank == k and bool(np.all(scale > 0)),
                 tolerance=tolerance, relative_tolerance=float(np.sqrt(np.finfo(float).eps)), scaled_eigenvalues=eigenvalues.tolist(),
                 condition_number=float(eigenvalues[-1]/eigenvalues[0]) if eigenvalues[0] > tolerance else np.inf,
                 kernel_gram=pd.DataFrame(gram, index=names, columns=names),
                 component_support=pd.DataFrame([dict(Source=s, Groups=len(counts), MinimumRows=int(counts.min()),
                     MaximumRows=int(counts.max())) for s, counts in zip(names, sizes)]))
    return audit, groups, normalized, scale


def covariance_components(work, design, groups, normalized, scale):
    scores = design["score_columns"]
    values = work[scores].to_numpy(dtype=float)
    values = values - values[0]
    units = np.max(np.abs(values), axis=0)
    units[units == 0] = 1.
    y = values / units
    y -= y.mean(axis=0)
    y -= y.mean(axis=0)
    products = []
    for g in groups:
        sums = np.zeros((int(g.max())+1, len(scores)))
        np.add.at(sums, g, y)
        products.append(sums.T @ sums)
    rhs = np.asarray(products).reshape(len(groups), -1)
    scaled = (np.linalg.solve(normalized, rhs / scale[:, None]) / scale[:, None]).reshape(-1, len(scores), len(scores))
    scaled = (scaled + scaled.transpose(0, 2, 1))/2
    raw = scaled * units[None, :, None] * units[None, None, :]
    if not np.isfinite(raw).all() or np.any((raw == 0) & (scaled != 0)):
        raise ValueError("Covariance components exceed numeric range; rescale scores")
    matrices = {s: pd.DataFrame(m, index=scores, columns=scores) for s, m in zip(design["component_factors"], raw)}
    sd = y.std(axis=0, ddof=1)
    sd[sd == 0] = 1.
    diagnostics = []
    for source, matrix in zip(matrices, scaled):
        eigenvalues = np.linalg.eigvalsh(matrix / sd[:, None] / sd[None, :])
        tolerance = np.sqrt(np.finfo(float).eps)*max(1., np.max(np.abs(eigenvalues)))
        diagnostics.append(dict(Source=source, MinimumScaledEigenvalue=float(eigenvalues.min()), Tolerance=tolerance,
            PositiveSemidefinite=bool(eigenvalues.min() >= -tolerance), Rank=int(np.sum(eigenvalues > tolerance)), Dimension=len(scores)))
    return matrices, pd.DataFrame(diagnostics)


def coefficients(components, counts, design):
    obj = design["object_facet"]
    values = dict(zip(components.Source, components.Variance))
    universe = values[obj]
    relative, absolute = 0., 0.
    for source, factors in design["component_factors"].items():
        if source == obj:
            continue
        contribution = values[source] / math.prod(counts[f] for f in factors if f != obj)
        absolute += contribution
        if obj in factors:
            relative += contribution
    return dict(G=universe/(universe+relative) if universe+relative > 0 else np.nan,
                Phi=universe/(universe+absolute) if universe+absolute > 0 else np.nan,
                details=dict(sigma2_object=universe, sigma2_relative_error=relative, sigma2_absolute_error=absolute))


def study(data, settings):
    work, design, excluded, reason = prepare(data, settings)
    key = "multivariate" if "score_cols" in settings else "design_study"
    result = dict(available=False, reason=REASONS.get(reason, ""), reason_code=reason, calculation_version=VERSION,
                  design=design, caveat=CAUTION, reference="MINQUE(0); numerical reference: mfrmr 0.3.2",
                  variance_components=pd.DataFrame(), observed_coefficients=dict(G=np.nan, Phi=np.nan),
                  d_study=pd.DataFrame(), selected_plan={}, excluded_rows=excluded,
                  bootstrap_ci=dict(requested=False, inference_allowed=False, inference_reason=CI_REASON))
    result[key] = dict(score_columns=design["score_columns"], weights=settings.get("composite_weights", {}))
    if reason:
        return result
    audit, groups, normalized, scale = kernel_audit(work, design)
    result["estimation"] = audit
    if not audit["identified"]:
        result.update(reason_code="unidentified_components", reason=REASONS["unidentified_components"])
        return result
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        try:
            matrices, diagnostics = covariance_components(work, design, groups, normalized, scale)
        except (ValueError, np.linalg.LinAlgError) as exc:
            result.update(reason_code="invalid_components", reason=str(exc))
            return result
    return mv.project_components(result, settings, matrices, diagnostics, np.full(len(matrices), np.nan),
                                 coefficient_fn=lambda components, counts: coefficients(components, counts, design))


def export_frame(bundle):
    frame = mv.export_frame(bundle)
    overrides = dict(supported_design="Crossed or one non-object facet nested in the other; one observation per cell; unit weights",
                     interval_hold_reason=CI_REASON, interval_scope=CI_REASON, interval_method="not_available")
    for field, value in overrides.items():
        frame.loc[frame._section.eq("metadata") & frame.Field.eq(field), "Value"] = value
    frames = [frame, bundle["excluded_rows"].assign(_section="excluded_rows")]
    audit = bundle.get("estimation")
    if audit:
        fields = {k: v for k, v in audit.items() if k not in ("kernel_gram", "component_support")}
        gram = audit["kernel_gram"]
        frames += [pd.DataFrame([dict(Field=k, Value=str(v), _section="estimation") for k, v in fields.items()]),
                   audit["component_support"].assign(_section="component_support"),
                   pd.DataFrame([dict(SourceA=a, SourceB=b, Trace=gram.loc[a, b], _section="kernel_gram")
                                 for a in gram.index for b in gram.columns])]
    return pd.concat(frames, ignore_index=True)
