"""Propagate old-calibration uncertainty through the shared annual data."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import cho_solve

from mfrm_app import mml_joint, structural_intervals
from validation.m2_sparse_intervals import REPLICATES
from validation.m3_shared_anchors import NODES, fit as fit_stationary
from validation.m5b_annual_drift import ANCHORED, CONFIG, DESIGNS, DRIFT_COLUMNS, EFFECTS, RATERS, UNIT, ratings_at


SELECTED = {name: DESIGNS[name] for name in ("repeat6", "repeat24")}
SOURCE = dict(CONFIG, facet_names=["Rater"], facet_levels={"Rater": RATERS},
              noncenter_facet="Rater", facet_signs={"Rater": -1})
SOURCE_BASIS = np.eye(7)[:, :-1]
SOURCE_OFFSET = np.zeros(7)
SOURCE_CONTRAST = np.array([-1., 1., 0., 0., 0., 0.])
OFFSET_DIRECTION = UNIT[:, 2]+UNIT[:, 5]
TARGET_CONTRAST = DRIFT_COLUMNS[2] @ ANCHORED
assert DRIFT_COLUMNS[2] @ OFFSET_DIRECTION == 0
FIELDS = ("replicate", "design", "effect", "truth", "anchor_R1", "anchor_R2", "anchor_se", "estimate", "sensitivity",
          "fixed_se", "model_independent_se", "score_independent_se", "score_shared_se", "score_target_variance",
          "score_anchor_variance", "score_cross_term", "shared_recordings", "perturbation_error", "reason")


def reviewed_fit(problem, basis, offset):
    boundary = problem.structural_recession_audit()
    if boundary["status"] != "no_structural_ray_detected" or boundary["structural_design_rank"] != problem.linear_d:
        raise ValueError(f"Structural review: {boundary['status']}")
    point, curvature, scores, information = fit_stationary(problem, basis, offset)
    calculations = []
    for q in (len(NODES), 2*len(NODES)-1):
        at = problem.evaluate(offset+basis@point, *mml_joint.normal_rule(q), information=True, moments=False)
        calculations.append(dict(points=q, information=basis.T@at["information"]@basis, gradient=basis.T@at["gradient"]))
    checks, _ = structural_intervals._assess(dict(coordinates=point, optimizer_success=True,
        boundary=dict(boundary, structural_design_rank=basis.shape[1]), calculations=calculations))
    if failed := [c["check"] for c in checks if not c["passed"]]:
        raise ValueError(" | ".join(failed))
    return point, curvature, scores, information


def perturbation(source, target, src, dst, recording, expected):
    estimates = []
    for change in (-.01, .01):
        ws = 1+change*(np.asarray(source.person_labels)==recording)
        wt = 1+change*(np.asarray(target.person_labels)==recording)
        u = fit_stationary(source, SOURCE_BASIS, SOURCE_OFFSET, start=src, weights=ws)
        v = fit_stationary(target, ANCHORED, (SOURCE_CONTRAST @ u)*OFFSET_DIRECTION, start=dst, weights=wt)
        estimates.append(TARGET_CONTRAST @ v)
    error = abs((estimates[1]-estimates[0])/.02-expected)
    assert error < 2e-5, (recording, error)
    return error


def compare(source, source_fit, target, check_perturbation):
    src, h_source, source_scores, _ = source_fit
    a = SOURCE_CONTRAST @ src
    source_direction = cho_solve(h_source, SOURCE_CONTRAST)
    variance_a = SOURCE_CONTRAST @ source_direction
    dst, h_target, target_scores, information = reviewed_fit(target, ANCHORED, a*OFFSET_DIRECTION)
    target_direction = cho_solve(h_target, TARGET_CONTRAST)
    sensitivity = -target_direction @ (ANCHORED.T @ information @ OFFSET_DIRECTION)
    fixed_variance = TARGET_CONTRAST @ target_direction
    u = pd.Series(sensitivity*(source_scores @ source_direction), index=source.person_labels)
    v = pd.Series(target_scores @ target_direction, index=target.person_labels)
    shared = v.index.intersection(u.index)
    assert len(shared)==len(u)==60
    joint = u.add(v, fill_value=0.)
    cross = 2*(u[shared] @ v[shared])
    np.testing.assert_allclose(joint@joint, (u@u)+(v@v)+cross, atol=1e-12, rtol=0)
    audit_error = perturbation(source, target, src, dst, shared[0], joint[shared[0]]) if check_perturbation else np.nan
    return dict(anchor_R1=src[0], anchor_R2=src[1], anchor_se=np.sqrt(variance_a), estimate=TARGET_CONTRAST @ dst,
        sensitivity=sensitivity, fixed_se=np.sqrt(fixed_variance),
        model_independent_se=np.sqrt(fixed_variance+sensitivity**2*variance_a),
        score_independent_se=np.sqrt((u@u)+(v@v)), score_shared_se=np.sqrt(joint@joint),
        score_target_variance=v@v, score_anchor_variance=u@u, score_cross_term=cross,
        shared_recordings=len(shared), perturbation_error=audit_error, reason="")


def run_replicate(rep):
    cases = [(design, effect, data.rename(columns={"Recording": "Person"}))
             for design, effect, data in ratings_at(rep, SELECTED) if effect != "panel_drifts"]
    first = cases[0][2]
    old = first[first.RatingYear.eq("A")].reset_index(drop=True)
    source = mml_joint.GroupedMmlLikelihood(old, SOURCE)
    try:
        source_fit = reviewed_fit(source, SOURCE_BASIS, SOURCE_OFFSET)
        source_reason = ""
    except (ValueError, np.linalg.LinAlgError) as exc:
        source_reason = f"Calibration: {exc}"
    rows = []
    for design, effect, data in cases:
        assert data[data.RatingYear.eq("A")].reset_index(drop=True).equals(old)
        if source_reason:
            result = dict(reason=source_reason)
        else:
            try:
                target = mml_joint.GroupedMmlLikelihood(data, CONFIG)
                result = compare(source, source_fit, target, rep <= 2)
            except (ValueError, np.linalg.LinAlgError) as exc:
                result = dict(reason=f"Annual fit: {exc}")
        rows.append(dict(replicate=rep, design=design, effect=effect, truth=EFFECTS[effect][2], **result))
    return rows


def run(path, workers, end):
    complete = set()
    if path.exists():
        saved = pd.read_csv(path)
        assert tuple(saved.columns) == FIELDS and saved.groupby("replicate").size().eq(6).all()
        complete.update(saved.replicate)
    pending = [rep for rep in range(1, end+1) if rep not in complete]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, rows in enumerate(pool.map(run_replicate, pending), 1):
            pd.DataFrame(rows).reindex(columns=FIELDS).to_csv(path, mode="a", header=not path.exists(), index=False)
            if i % 25 == 0 or i == len(pending):
                print(f"{i}/{len(pending)} replicate sets", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--end", type=int, default=REPLICATES)
    args = parser.parse_args()
    run(args.out, args.workers, args.end)
