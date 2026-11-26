"""Retained numerical evidence for ordinary fixed-SD MML uncertainty outputs.

This is a same-point sensitivity check, not an integration error bound or a
coverage certificate. Displaying a saved result never runs it implicitly.
"""
import numpy as np

from . import engine, mml_fixed_sd_review as review, structural_intervals


REASON = "stat.mml.general_numerical_review_unavailable"


def _identity(result):
    return engine.stable_json_fingerprint(dict(
        config=result["config"], data=result["prep"]["data"],
        coordinates=result["opt"].x, objective=result["opt"].fun,
        params=result["params"], success=result["opt"].success), 64)


def compute(result):
    """Check the returned fit without changing its model or point estimates."""
    record = dict(version=1, identity=_identity(result), passed=False,
                  covariance=None, information=None, checks=[],
                  continuous_reference_checked=False, coverage_qualified=False)
    try:
        prepared = review.prepare_fixed_sd_check(result)
        q = prepared["config"]["quad_points"]
        evaluated = review.evaluate_fixed_sd_quadrature_check(prepared, comparison_points=2*q-1)
        rows = evaluated["comparison"]["rows"]
        checks, covariance = structural_intervals._assess(dict(
            boundary=evaluated["structural_boundary"], calculations=rows,
            coordinates=prepared["coordinates"], optimizer_success=bool(result["opt"].success)))
        # Posterior scores feed ordinary fit, agreement and person precision.
        # Their stability is not implied by stability of marginal information.
        sd = np.asarray(rows[0]["posterior_sd"])
        fine_sd = np.asarray(rows[1]["posterior_sd"])
        eap_shift = np.abs(np.asarray(rows[1]["eap"])-rows[0]["eap"])
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            mean_change = eap_shift / np.maximum(sd, np.finfo(float).tiny)
            sd_change = np.abs(fine_sd/sd-1)
        for name, values in (("posterior_mean_change", mean_change), ("posterior_sd_change", sd_change)):
            value = float(np.max(values))
            checks.append(dict(check=name, passed=bool(np.isfinite(value) and value <= .01),
                               value=value, limit=.01))
        passed = all(row["passed"] for row in checks)
        record.update(passed=passed, checks=checks,
                      covariance=covariance.tolist() if passed else None,
                      information=rows[0]["information"] if passed else None,
                      points=[q, 2*q-1],
                      reason="; ".join(row["check"] for row in checks if not row["passed"]))
    except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
        record["reason"] = str(exc)
    record["record_hash"] = engine.stable_json_fingerprint(record, 64)
    return record


def valid_record(result):
    record = result.get("numerical_review")
    if not isinstance(record, dict) or record.get("version") != 1:
        return None
    body = {k: v for k, v in record.items() if k != "record_hash"}
    if record.get("record_hash") != engine.stable_json_fingerprint(body, 64):
        return None
    if record.get("identity") != _identity(result):
        return None
    return record


def block_reason(result):
    if result.get("config", {}).get("method") != "MML":
        return ""
    record = valid_record(result)
    return "" if record and record.get("passed") is True else REASON
