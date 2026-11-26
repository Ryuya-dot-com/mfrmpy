"""Simulate annual rater drift with shared recordings and old-calibration anchors."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import softmax
from scipy.stats import norm

import mfrmpy
from mfrm_app import mml_joint, structural_intervals
from validation.m2_sparse_intervals import PAIR, REPLICATES
from validation.m3_shared_anchors import NODES, fit as fit_stationary


RATERS = ["R1", "R2", "R3"]
LABELS = [f"{year}-{rater}" for year in ("A", "B") for rater in RATERS]
CONFIG = dict(model="RSM", facet_names=["WorkYear", "RaterYear"],
    facet_levels={"WorkYear": ["A", "B"], "RaterYear": LABELS}, noncenter_facet="RaterYear",
    facet_signs={"WorkYear": 1, "RaterYear": -1}, n_cat=5, rating_min=1)
DESIGNS = {"repeat6": (6, False), "repeat24": (24, False), "rotate24": (24, True)}
EFFECTS = {"cohort_only": (0., 0., 0.), "r3_drifts": (0., 0., .5), "anchors_drift": (.5, .5, 0.), "panel_drifts": (.5, .5, .5)}
SEVERITY = np.array([-.6, 0., .6])
NEW_COHORT_MEAN = .75
STEPS = np.cumsum([0., -1.2, -.4, .4, 1.2])
UNIT = np.eye(11)
# Cohort locations are centered. A free common translation maps the two fixed
# old severities (-.6 and 0) to this location convention in both years.
ANCHORED = np.c_[UNIT[:, 0], UNIT[:, [1, 2, 4, 5]].sum(axis=1), UNIT[:, [3, 6, 7, 8, 9]]]
FIXED = .6*(UNIT[:, 2] + UNIT[:, 5])
MODELS = {"recordings": (UNIT[:, :-1], np.zeros(11)), "old_anchors": (ANCHORED, FIXED)}
TARGETS = {"R3": np.array([0., 0., 1.]), "panel_mean": np.ones(3)/3, "R3_relative": np.array([-.5, -.5, 1.])}
DRIFT_COLUMNS = UNIT[4:7] - UNIT[1:4]
FIELDS = ("replicate", "design", "effect", "model", "target", "truth", "estimate", "se", "lower", "upper", "p", "reason")
AUDIT = dict(person="Recording", facets=["RaterYear"], score="Score", rater="RaterYear", difference=("B-R3", "A-R3"),
             rating_min=1, rating_max=5)


def scores(ability, uniforms, severity, cumulative_steps=STEPS):
    logits = (ability[:, None, None]-severity[None, :, None])*np.arange(5)-cumulative_steps
    return 1+(uniforms[..., None] > softmax(logits, axis=-1).cumsum(axis=-1)[..., :-1]).sum(axis=-1)


def rows_for_year(rating_year, work_year, video, pairing, response, band):
    rater = PAIR[pairing].ravel()
    label = np.array(RATERS)[rater]
    ids = np.repeat(video, 2)
    return pd.DataFrame(dict(Recording=np.repeat([f"{year}{i:03}" for year, i in zip(work_year, video, strict=True)], 2),
        Speaker=[f"S{i:03}" if year=="A" or i<12 else f"T{i:03}" for year, i in zip(np.repeat(work_year, 2), ids, strict=True)],
        WorkYear=np.repeat(work_year, 2), RatingYear=rating_year, Rater=label, RaterYear=[f"{rating_year}-{r}" for r in label],
        Mode="video", Band=np.repeat(band, 2), Score=response[np.repeat(np.arange(len(video)), 2), rater]))


def rating_draws(replicate):
    rng = np.random.default_rng(np.random.SeedSequence([2026112206, replicate]))
    abilities = rng.normal(size=(2, 60))
    uniforms = rng.random((3, 60, 3))
    return rng, abilities, uniforms


def current_step_scores(r3_step_shift):
    if r3_step_shift is None:
        return STEPS
    shift = np.asarray(r3_step_shift, dtype=float)
    if shift.shape != (4,) or not np.isfinite(shift).all() or not np.isclose(shift.sum(), 0.):
        raise ValueError("r3_step_shift requires four finite, centered step changes")
    steps = np.tile(STEPS, (len(RATERS), 1))
    steps[2] += np.r_[0., shift.cumsum()]
    return steps


def ratings_at(replicate, designs=DESIGNS, *, score_band="spread", r3_step_shift=None,
               new_cohort_mean=NEW_COHORT_MEAN):
    if score_band not in {"spread", "low", "high"}:
        raise ValueError("score_band must be spread, low or high")
    rng, (old_ability, new_ability), (u_first, u_replay, u_new) = rating_draws(replicate)
    pairs = rng.permutation(np.repeat(np.arange(3), 20))
    first = scores(old_ability, u_first, SEVERITY)
    totals = first[np.arange(60)[:, None], PAIR[pairs]].sum(axis=1)
    ranked = np.lexsort((rng.random(60), totals)).reshape(3, 20)
    tiers = np.empty(60, dtype=int)
    for band, ids in enumerate(ranked):
        tiers[ids] = band
    within_tier = np.array([rng.permutation(indices) for indices in ranked])
    current_priority = rng.random(60)
    old_rows = rows_for_year("A", np.repeat("A", 60), np.arange(60), pairs, first, tiers)
    current_steps = current_step_scores(r3_step_shift)
    for name, (shared_count, rotate) in designs.items():
        shared = within_tier[:, :shared_count//3].ravel()
        if score_band != "spread":
            # Match the number of replays per original rater pair, including ties.
            by_pair = np.bincount(pairs[shared], minlength=3)
            order = ranked.ravel()[::-1] if score_band == "high" else ranked.ravel()
            shared = np.concatenate([order[pairs[order] == pair][:count] for pair, count in enumerate(by_pair)])
        second_pairs = (pairs[shared] + int(rotate)) % 3
        remaining_pairs = np.repeat(np.arange(3), 20 - np.bincount(second_pairs, minlength=3))
        remaining_pairs = remaining_pairs[current_priority[:len(remaining_pairs)].argsort()]
        new = np.arange(len(remaining_pairs))
        layout = np.r_[second_pairs, remaining_pairs]
        assert np.bincount(layout, minlength=3).tolist()==[20]*3
        if len(shared):
            count = sum(len(set(PAIR[a]) & set(PAIR[b])) for a, b in zip(pairs[shared], second_pairs, strict=True))
            assert count == (1 if rotate else 2)*shared_count
        for effect, delta in EFFECTS.items():
            severity = SEVERITY+np.asarray(delta)
            changed = np.r_[scores(old_ability, u_replay, severity, current_steps)[shared],
                            scores(new_ability+new_cohort_mean, u_new, severity, current_steps)[new]]
            now = rows_for_year("B", np.repeat(["A", "B"], [len(shared), len(new)]), np.r_[shared, new], layout,
                                changed, np.r_[tiers[shared], np.repeat(-1, len(new))])
            yield name, effect, pd.concat([old_rows, now], ignore_index=True)


def targets_at(rows, support, effect):
    truth = np.asarray(EFFECTS[effect])
    targets = {name: (float(w @ truth), w @ DRIFT_COLUMNS) for name, w in TARGETS.items()}
    reason = ""
    if support["status"] != "observed":
        reason = f"Assignment audit: {support['status']}"
    else:
        problem = mml_joint.GroupedMmlLikelihood(rows.rename(columns={"Recording": "Person"}), CONFIG)
        boundary = problem.structural_recession_audit()
        if boundary["status"] != "no_structural_ray_detected" or boundary["structural_design_rank"] != problem.linear_d:
            reason = f"Joint design rank or boundary check: {boundary['status']}"
    for model, (basis, offset) in MODELS.items():
        rows_out = {name: dict(model=model, target=name, truth=value, reason=reason) for name, (value, _) in targets.items()}
        if reason:
            yield from rows_out.values()
            continue
        try:
            point, _, _, _ = fit_stationary(problem, basis, offset)
        except (ValueError, np.linalg.LinAlgError) as exc:
            yield from (r | dict(reason=f"Joint fit: {exc}") for r in rows_out.values())
            continue
        native = offset + basis @ point
        calculations = []
        for q in (len(NODES), 2*len(NODES)-1):
            at = problem.evaluate(native, *mml_joint.normal_rule(q), information=True, moments=False)
            calculations.append(dict(points=q, information=basis.T @ at["information"] @ basis, gradient=basis.T @ at["gradient"]))
        # Full raw rank and absence of a raw recession ray also hold on this injective subspace.
        checks, covariance = structural_intervals._assess(dict(coordinates=point, optimizer_success=True,
            boundary=dict(boundary, structural_design_rank=basis.shape[1]), calculations=calculations))
        failed = [c["check"] for c in checks if not c["passed"]]
        for name, (_, weight) in targets.items():
            output = rows_out[name] | dict(estimate=float(weight @ native))
            if failed:
                yield output | dict(reason=" | ".join(failed))
                continue
            a = weight @ basis
            se = np.sqrt(a @ covariance @ a)
            assert se > 0 and np.isfinite(se)
            center = output["estimate"]
            radius = norm.ppf(.975)*se
            yield output | dict(se=se, lower=center-radius, upper=center+radius, p=2*norm.sf(abs(center)/se))


def run_replicate(rep):
    assignments = {}
    out = []
    for design, effect, ratings in ratings_at(rep):
        if design not in assignments:
            audit = mfrmpy.check_rater_contrast(ratings, **AUDIT)
            assert audit["status"] == "observed" and ratings.groupby(["RatingYear", "Rater"]).size().eq(40).all()
            assignments[design] = audit
        out.extend(dict(replicate=rep, design=design, effect=effect, **r) for r in targets_at(ratings, assignments[design], effect))
    return out


def controls():
    design = {"repeat24": DESIGNS["repeat24"], "unlinked": (0, False)}
    rows = {name: data for name, effect, data in ratings_at(1, design) if effect=="cohort_only"}
    cases = []
    for name, ratings, has_new_panel, mixed_mode in (
        ("shared_same_panel", rows["repeat24"], False, False),
        ("shared_replaced_panel", rows["repeat24"], True, False),
        ("no_shared_replaced_panel", rows["unlinked"], True, False),
        ("same_recording_mode_changes", rows["repeat24"], False, True),
    ):
        current = ratings.RatingYear.eq("B")
        if has_new_panel:
            ratings = ratings.assign(Rater=ratings.Rater.where(~current, ratings.Rater.str.replace("R", "S")))
            ratings = ratings.assign(RaterYear=ratings.RatingYear+"-"+ratings.Rater)
        if mixed_mode:
            ratings = ratings.assign(Mode=ratings.RatingYear.map({"A": "live", "B": "video"}))
        settings = AUDIT | dict(difference=("B-S3" if has_new_panel else "B-R3", "A-R3"))
        if mixed_mode:
            settings["facets"] = ["RaterYear", "Mode"]
        status = mfrmpy.check_rater_contrast(ratings, **settings)["status"]
        assert (status == "observed") == name.startswith("shared")
        cases.append(dict(case=name, ratings=len(ratings), group_origin_link=status,
                          r3_observed_both_years=not has_new_panel, individual_supported=status=="observed" and not has_new_panel,
                          panel_mean_supported=status=="observed"))
    return pd.DataFrame(cases)


def public_check():
    table = []
    for design, effect, data in ratings_at(1):
        support = mfrmpy.check_rater_contrast(data, **AUDIT)
        assert support["status"] == "observed"
        reference = next(targets_at(data, support, effect))
        assert reference["target"] == "R3" and reference["model"] == "recordings"
        fit = mfrmpy.analyze(data, person="Recording", facets=["WorkYear", "RaterYear"], score="Score",
            model="RSM", method="MML", positive_facets=["WorkYear"], noncenter_facet="RaterYear",
            rating_min=1, rating_max=5, quad_points=len(NODES), reltol=1e-12,
            interval_requests=[dict(facet="RaterYear", difference=("B-R3", "A-R3"))])
        entry, = fit["interval_analyses"]
        assert entry["status"] == "computed", entry["reason"]
        target = next(row for row in entry["record"]["table"] if row["Kind"] == "difference")
        assert target["CIEligible"], target["Reason"]
        assert abs(target["Estimate"]-reference["estimate"]) < 5e-5 and abs(target["SE"]-reference["se"]) < 2e-6
        table.append(dict(design=design, effect=effect, public_estimate=target["Estimate"],
            research_estimate=reference["estimate"], public_se=target["SE"], research_se=reference["se"]))
    return pd.DataFrame(table)


def run(path, workers, end):
    complete = set()
    if path.exists():
        saved = pd.read_csv(path)
        assert tuple(saved.columns)==FIELDS and saved.groupby("replicate").size().eq(72).all()
        complete.update(saved.replicate)
    pending = [rep for rep in range(1, end+1) if rep not in complete]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, rows in enumerate(pool.map(run_replicate, pending), 1):
            pd.DataFrame(rows).reindex(columns=FIELDS).to_csv(path, mode="a", header=not path.exists(), index=False)
            if i % 25 == 0 or i == len(pending):
                print(f"{i}/{len(pending)} replicate sets", flush=True)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--end", type=int, default=REPLICATES)
    checks = parser.add_mutually_exclusive_group()
    checks.add_argument("--controls", action="store_true")
    checks.add_argument("--public-check", action="store_true")
    args = parser.parse_args()
    if args.controls:
        controls().to_csv(args.out, index=False)
    elif args.public_check:
        public_check().to_csv(args.out, index=False)
    else:
        run(args.out, args.workers, args.end)
