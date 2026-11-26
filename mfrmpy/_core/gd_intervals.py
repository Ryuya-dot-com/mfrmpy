"""Pointwise normal-model intervals for balanced crossed G/D coefficients."""
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import norm


CAUTION = (
    "Pointwise approximate logit-delta intervals under independent normal random effects "
    "for objects, declared random facets and their interactions. Component covariance uses "
    "a Gaussian plug-in model, including uncertainty in every random facet. "
    "These are not simultaneous bands, prediction intervals or distribution-robust intervals. "
    "Relevant nonpositive raw components and boundary coefficients withhold intervals; "
    "small facet samples can still give poor coverage. Fixed-facet conditioning remains conditional."
)
FIELDS = [f"{k}_{s}" for k in ("G", "Phi") for s in ("SE", "lower", "upper", "width", "interval_status")]
FIELDS += ["IntervalMethod", "IntervalConfidence", "IntervalAssumption"]


def validate_request(request):
    if request is None:
        return None
    if (not isinstance(request, dict) or set(request) - {"method", "assumption", "confidence"}
            or request.get("method") != "normal_delta" or request.get("assumption") != "normal"):
        raise ValueError("coefficient_intervals requires method='normal_delta' and assumption='normal'")
    level = request.get("confidence", .95)
    if isinstance(level, (bool, np.bool_)) or not isinstance(level, (int, float, np.integer, np.floating)) or not np.isfinite(level) or not 0 < level < 1:
        raise ValueError("Interval confidence must be a finite number strictly between zero and one")
    return dict(method="normal_delta", assumption="normal", confidence=float(level))


def component_system(components, design):
    """E[MS] = E v; covariance of A MS is A diag(2 E[MS]^2/df) A'."""
    levels = design["levels"]
    terms = [set(levels) if s == "Residual" else set(s.split(":")) for s in components.Source]
    expected = np.array([
        [np.prod([n for f, n in levels.items() if f not in t]) if s <= t else 0.
         for t in terms] for s in terms], dtype=float)
    inverse = np.linalg.solve(expected, np.eye(len(terms)))
    means = expected @ components.Variance.to_numpy(dtype=float)
    with np.errstate(over="ignore", invalid="ignore"):
        covariance = (inverse * (2 * means**2 / components.df.to_numpy(dtype=float))) @ inverse.T
    return terms, covariance


def coefficient_row(components, design, terms, covariance, counts, confidence):
    obj = design["object_facet"]
    sources = list(components.Source)
    object_index = sources.index(obj)
    raw = components.RawVariance.to_numpy(dtype=float)
    values = components.Variance.to_numpy(dtype=float)
    universe = values[object_index]
    absolute = np.array([1 / float(np.prod([counts[f] for f in t if f != obj])) for t in terms])
    absolute[object_index] = 0.
    relative = absolute * np.array([obj in t for t in terms])
    out = {}
    for name, weights in (("G", relative), ("Phi", absolute)):
        status = "approximate_normal_delta"
        lower = upper = se = np.nan
        error = float(weights @ values)
        relevant = weights > 0
        relevant[object_index] = True
        if not np.isfinite(covariance).all() or not np.isfinite(error + universe):
            status = "unavailable_numerical_range"
        elif universe <= 0 or error <= 0:
            status = "boundary_coefficient"
        elif (raw[relevant] <= 0).any():
            # The clipped estimator is not differentiable at the component boundary.
            status = "nonpositive_component"
        else:
            gradient = -weights / error
            gradient[object_index] = 1 / universe
            variance = float(gradient @ covariance @ gradient)
            if not np.isfinite(variance) or variance <= 0:
                status = "unavailable_numerical_range"
            else:
                logit = np.log(universe) - np.log(error)
                spread = norm.ppf((1 + confidence) / 2) * np.sqrt(variance)
                lower, upper = expit([logit - spread, logit + spread])
                point = universe / (universe + error)
                se = point * (1 - point) * np.sqrt(variance)
                if not np.isfinite([lower, upper, se]).all():
                    lower = upper = se = np.nan
                    status = "unavailable_numerical_range"
        out.update({f"{name}_SE": se, f"{name}_lower": lower, f"{name}_upper": upper, f"{name}_width": upper-lower,
                    f"{name}_interval_status": status})
    return out


def add_intervals(bundle, request):
    """Enrich a newly computed bundle; never reinterpret legacy bootstrap bounds."""
    request = validate_request(request)
    if request is None:
        return
    record = dict(request, scope=CAUTION, status="design_unavailable", observed={},
                  covariance_basis="normal_expected_mean_squares_nonnegative_plugin")
    bundle["coefficient_intervals"] = record
    if not bundle["available"]:
        return
    components, design = bundle["variance_components"], bundle["design"]
    terms, covariance = component_system(components, design)
    record.update(status="computed", covariance=pd.DataFrame(
        covariance, index=components.Source.tolist(), columns=components.Source.tolist()))
    def interval(counts):
        return dict(coefficient_row(components, design, terms, covariance, counts, request["confidence"]),
                    IntervalMethod=request["method"], IntervalConfidence=request["confidence"],
                    IntervalAssumption=request["assumption"])
    record["observed"] = interval(design["observed_n"])
    if bundle.get("selected_plan"):
        bundle["selected_plan"].update(interval(bundle["selected_plan"]))
    grid = bundle["d_study"]
    rows = [interval(row) for row in grid.to_dict("records")]
    bundle["d_study"] = pd.concat([grid.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    bundle["caveat"] = bundle["caveat"].replace(
        "Intervals remain unqualified for the random-facet targets.", CAUTION)
