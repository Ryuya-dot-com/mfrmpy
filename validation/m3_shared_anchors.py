"""Evaluate two-stage uncertainty when calibration and receiving ratings overlap."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from scipy.special import log_softmax, logsumexp

from mfrm_app.mml_joint import GroupedMmlLikelihood, normal_rule
from validation.m2_sparse_intervals import DESIGNS, REPLICATES, SEED, potential_scores, simulated_ratings
from validation.m3_independent_anchors import LEVELS


CONDITIONS = {"fresh60": 60, "reuse30": 30, "reuse60": 60}
TARGETS = {name: DESIGNS[name] for name in ("balanced6", "balanced24")}
UNIT = np.eye(9)
SOURCE = np.c_[UNIT[:, :3], UNIT[:, 3:5] - UNIT[:, [5]], UNIT[:, 6:8]]
TARGET = np.c_[UNIT[:, [0, 1, 2, 5, 6, 7]], -UNIT[:, 3:6].sum(axis=1)]
ANCHORS = UNIT[:, 3:5]
CONTRAST = np.eye(TARGET.shape[1])[3]
NODES, WEIGHTS = normal_rule(61)
CONFIG = dict(model="RSM", facet_names=list(LEVELS), facet_levels=LEVELS, noncenter_facet="Rater",
              facet_signs={"Item": -1, "Rater": -1}, n_cat=4, rating_min=0)
FIELDS = ("replicate", "condition", "design", "shared_people", "reused_ratings", "estimate", "a1", "a2",
          "j1", "j2", "model_fixed_se", "model_independent_se", "score_independent_se", "score_shared_se",
          "score_anchor_variance", "score_target_variance", "score_cross_term", "perturbation_error", "reason")


def calibration(replicate, condition):
    m2_rng = np.random.default_rng(np.random.SeedSequence([SEED, replicate]).spawn(1)[0])
    m2_ability = m2_rng.normal(size=60)
    membership, source_ability, source_scores = (
        np.random.default_rng(s) for s in np.random.SeedSequence([2026112203, replicate]).spawn(3))
    shared = membership.permutation(60)[:CONDITIONS[condition]]
    people = np.array([f"S{i:03}" for i in range(60)])
    people[shared] = [f"P{i:03}" for i in shared]
    ability = source_ability.normal(size=60)
    ability[shared] = m2_ability[shared]
    scores = potential_scores(ability, source_scores)
    if condition.startswith("reuse"):
        scores[shared] = potential_scores(m2_ability, m2_rng)[shared]
    rows = pd.MultiIndex.from_product([people, LEVELS["Item"], LEVELS["Rater"]],
        names=["Person", "Item", "Rater"]).to_frame(index=False)
    rows["Score"] = scores.ravel()
    return rows


def person_contributions(likelihood, point):
    design = likelihood.design
    theta = np.exp(point[-1])*NODES
    log_prob = log_softmax(design @ point[:-1] + theta[:, None, None]*likelihood.k, axis=-1)
    counts = likelihood.counts.reshape(len(likelihood.person_labels), -1)
    log_joint = counts @ log_prob.reshape(len(NODES), -1).T + np.log(WEIGHTS)
    loglik = logsumexp(log_joint, axis=1)
    posterior = np.exp(log_joint - loglik[:, None])
    expected = np.einsum("qik,ikd->qid", np.exp(log_prob), design)
    observed = counts @ design.reshape(-1, likelihood.linear_d)
    score = observed - np.einsum("nq,ni,qid->nd", posterior, likelihood.cell_totals, expected, optimize=True)
    return loglik, np.c_[score, np.zeros(len(loglik))]


def fit(likelihood, basis, offset, start=None, weights=None):
    def objective(coordinates):
        point = offset + basis @ coordinates
        if weights is None:
            result = likelihood.evaluate(point, NODES, WEIGHTS, moments=False)
            return result["nll"], basis.T @ result["gradient"]
        loglik, scores = person_contributions(likelihood, point)
        return -weights @ loglik, -weights @ scores @ basis

    start = np.zeros(basis.shape[1]) if start is None else start
    opt = minimize(objective, start, jac=True, method="BFGS", options=dict(gtol=1e-8, maxiter=180))
    coordinates = opt.x
    if weights is not None:
        if not np.isfinite(opt.fun) or np.max(np.abs(opt.jac)) > 1e-5:
            raise ValueError("Perturbed score did not converge")
        return coordinates

    # One Newton step resolves BFGS line searches that stop at objective roundoff.
    at = likelihood.evaluate(offset + basis @ coordinates, NODES, WEIGHTS, information=True, moments=False)
    curvature = cho_factor(basis.T @ at["information"] @ basis)
    gradient = basis.T @ at["gradient"]
    coordinates = coordinates - cho_solve(curvature, gradient)
    point = offset + basis @ coordinates
    at = likelihood.evaluate(point, NODES, WEIGHTS, information=True, moments=False)
    curvature = cho_factor(basis.T @ at["information"] @ basis)
    gradient = basis.T @ at["gradient"]
    if np.sqrt(gradient @ cho_solve(curvature, gradient)) > 1e-5:
        raise ValueError("Marginal score did not converge")
    loglik, scores = person_contributions(likelihood, point)
    np.testing.assert_allclose(-loglik.sum(), at["nll"], atol=1e-10, rtol=0)
    np.testing.assert_allclose(-scores.sum(axis=0)[:-1], at["gradient"][:-1], atol=1e-9, rtol=0)
    return coordinates, curvature, scores @ basis, at["information"]


def perturbation(source, target, source_point, target_point, person, expected):
    estimates = []
    for change in (-.01, .01):
        ws = 1 + change*(np.array(source.person_labels) == person)
        wt = 1 + change*(np.array(target.person_labels) == person)
        source_fit = fit(source, SOURCE, np.zeros(9), start=source_point, weights=ws)
        a = source_fit[3:5]
        target_fit = fit(target, TARGET, ANCHORS @ a, start=target_point, weights=wt)
        estimates.append(target_fit[3] - a[0])
    difference = abs((estimates[1] - estimates[0])/.02 - expected)
    assert difference < 2e-5, (person, difference)
    return difference


def compare(source, source_fit, target, overlap, audit):
    src, src_h, src_score, _ = source_fit
    a = src[3:5]
    dst, dst_h, dst_score, information = fit(target, TARGET, ANCHORS @ a)
    target_direction = cho_solve(dst_h, CONTRAST)
    j = np.array([-1., 0.]) - target_direction @ (TARGET.T @ information @ ANCHORS)
    np.testing.assert_allclose(j.sum(), 0, atol=1e-12, rtol=0)
    source_weight = np.zeros(len(src))
    source_weight[3:5] = j
    source_direction = cho_solve(src_h, source_weight)
    model_t = CONTRAST @ target_direction
    model_s = source_weight @ source_direction
    u = pd.Series(src_score @ source_direction, index=source.person_labels)
    v = pd.Series(dst_score @ target_direction, index=target.person_labels)
    shared = u.index.intersection(v.index)
    assert len(shared) == overlap
    joint = u.add(v, fill_value=0.)
    ss, tt = u @ u, v @ v
    cross = 2*(u[shared] @ v[shared])
    np.testing.assert_allclose(joint @ joint, ss + tt + cross, atol=1e-12, rtol=0)
    if audit:
        errors = [perturbation(source, target, src, dst, shared[0], joint[shared[0]])]
        if overlap < 60:
            alone = u.index.difference(v.index)[0]
            errors.append(perturbation(source, target, src, dst, alone, joint[alone]))
        audit_error = max(errors)
    else:
        audit_error = np.nan
    return dict(estimate=dst[3] - a[0], a1=a[0], a2=a[1], j1=j[0], j2=j[1],
                model_fixed_se=np.sqrt(model_t), model_independent_se=np.sqrt(model_t+model_s),
                score_independent_se=np.sqrt(ss+tt), score_shared_se=np.sqrt(joint@joint),
                score_anchor_variance=ss, score_target_variance=tt, score_cross_term=cross,
                perturbation_error=audit_error, reason="")


def run_replicate(replicate):
    targets = {}
    for name, data in simulated_ratings(replicate, TARGETS):
        ratings = data.assign(Person=data.Person.map("P{:03d}".format))
        targets[name] = ratings, GroupedMmlLikelihood(ratings, CONFIG)
    rows = []
    for condition, overlap in CONDITIONS.items():
        data = calibration(replicate, condition)
        source = GroupedMmlLikelihood(data, CONFIG)
        try:
            source_fit = fit(source, SOURCE, np.zeros(9))
            status = ""
        except (ValueError, np.linalg.LinAlgError) as exc:
            status = f"Calibration: {exc}"
        for name, (ratings, target) in targets.items():
            common = ratings.merge(data, on=["Person", "Item", "Rater"], validate="one_to_one")
            assert len(common) == 4*overlap
            reused = len(common) if condition.startswith("reuse") else 0
            if reused:
                assert common.Score_x.eq(common.Score_y).all()
            row = dict(replicate=replicate, condition=condition, design=name,
                       shared_people=overlap, reused_ratings=reused)
            try:
                out = dict(reason=status) if status else compare(source, source_fit, target, overlap, replicate <= 2)
            except (ValueError, np.linalg.LinAlgError) as exc:
                out = dict(reason=f"Receiving: {exc}")
            rows.append(row | out)
    return rows


def run(path, workers, start, end):
    complete = set()
    if path.exists():
        saved = pd.read_csv(path)
        assert tuple(saved.columns) == FIELDS and saved.groupby("replicate").size().eq(6).all()
        complete = set(saved.replicate)
    pending = [rep for rep in range(start, end+1) if rep not in complete]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, rows in enumerate(pool.map(run_replicate, pending), 1):
            pd.DataFrame(rows).reindex(columns=FIELDS).to_csv(path, mode="a", header=not path.exists(), index=False)
            if i % 50 == 0 or i == len(pending):
                print(f"{i}/{len(pending)} replicate sets", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=REPLICATES)
    args = parser.parse_args()
    run(args.out, args.workers, args.start, args.end)
