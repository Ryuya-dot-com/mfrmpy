"""Validate a specified rater-by-school severity contrast through public fits."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import softmax
from scipy.stats import norm

import mfrmpy
from mfrm_app import engine, joint_interaction, mml_fixed_sd_review, mml_joint, output_qualification, structural_intervals
from validation.m2_sparse_intervals import DESIGNS, REPLICATES, simulated_ratings
from validation.m3_shared_anchors import NODES, fit as fit_stationary


FACETS = ["Item", "Rater", "School"]
TARGETS = {name: DESIGNS[name] for name in ("balanced6", "balanced24")}
ASSIGNMENT = dict(person="Person", facets=FACETS, score="Score", rating_min=1, rating_max=5,
                  noncenter_facet="School", positive_facets=["School"])
AUDIT = dict(**ASSIGNMENT, rater="Rater", difference=("R3", "R1"), group="School", group_difference=("B", "A"))
FIT = dict(**ASSIGNMENT, method="MML", model="RSM", quad_points=len(NODES), reltol=1e-10)
JOINT = dict(facet_a="Rater", facet_b="School")
EFFECTS = {"ability_only": 0., "r3_stricter": .5}
COLUMNS = ("replicate", "design", "effect", "truth", "support", "overlap_A", "overlap_B",
           "estimate", "se", "lower", "upper", "p", "school_A", "school_B", "reason")


def ratings_at(replicate, designs=TARGETS, *, school_sd=(1., 1.)):
    sets = {(name, effect): [] for name in designs for effect in EFFECTS}
    for g, (school, mean) in enumerate((("A", -.5), ("B", .5))):
        rng = np.random.default_rng(np.random.SeedSequence([2026112205, replicate, g]))
        ability = mean + school_sd[g]*rng.normal(size=60)
        uniforms = rng.random((60, 4, 3))
        eta = ability[:, None, None] - np.array([-.75, -.25, .25, .75])[None, :, None] - np.array([-.6, 0., .6])[None, None, :]
        alternatives = {}
        for name, amount in EFFECTS.items():
            shift = np.array([0., 0., amount*g])
            probability = softmax((eta-shift)[..., None]*np.arange(5) - np.cumsum([0., -1.2, -.4, .4, 1.2]), axis=-1)
            alternatives[name] = 1 + (uniforms[..., None] > probability.cumsum(axis=-1)[..., :-1]).sum(axis=-1)
        for design, source in simulated_ratings(replicate + g*1_000_000, designs):
            ix = (source.Person.to_numpy(), source.Item.str[1:].astype(int)-1, source.Rater.str[1:].astype(int)-1)
            people = source.Person.map(lambda person: f"{school}{person:03d}")
            for effect, scores in alternatives.items():
                sets[design, effect].append(source.assign(Person=people, School=school, Score=scores[ix]))
    for (design, effect), rows in sets.items():
        yield design, effect, pd.concat(rows, ignore_index=True)


def review(rows, audit):
    if not audit["observation_estimable"]:
        return dict(reason=f"Assignment audit: {audit['status']}")
    baseline = mfrmpy.analyze(rows, **FIT)["result"]
    support = output_qualification.audit_boundary_and_support(baseline)
    if (support["unsupported_internal"] or support["unobserved_global_boundary"]
            or support["unbounded_facet_levels"] or support["hypergraph_components"] > 1):
        return dict(reason="Baseline category, boundary or global connectivity check")
    prepared = mml_fixed_sd_review.prepare_fixed_sd_check(baseline)
    request = joint_interaction.settings(JOINT, FACETS, "MML")
    design = joint_interaction._design(prepared, request)
    problem = design["problem"]
    boundary = problem.structural_recession_audit()
    if boundary["structural_design_rank"] != problem.linear_d or boundary["status"] != "no_structural_ray_detected":
        return dict(reason=f"Joint design rank or boundary check: {boundary['status']}")
    start = np.r_[prepared["coordinates"], np.zeros(problem.linear_d-design["base_dimension"])]
    try:
        point, _, _, _ = fit_stationary(problem, np.eye(problem.dimension)[:, :-1], np.zeros(problem.dimension), start=start)
    except (ValueError, np.linalg.LinAlgError) as exc:
        return dict(reason=f"Joint fit: {exc}")
    weights_by_cell = {(t["rater_level"], t["group_level"]): t["weight"] for t in audit["contrast_weights"]}
    weights = np.array([weights_by_cell.get(tuple(cell), 0.) for cell in design["cells"]])
    assert np.count_nonzero(weights) == 4 and weights.sum() == 0
    calculations = []
    for q in (FIT["quad_points"], 2*FIT["quad_points"]-1):
        at = problem.evaluate(np.r_[point, 0.], *mml_joint.normal_rule(q), information=True, moments=False)
        np.testing.assert_allclose(at["nll"], joint_interaction._row_nll(point, prepared, design, q), atol=1e-8, rtol=0)
        calculations.append(dict(points=q, information=at["information"][:-1, :-1], gradient=at["gradient"][:-1]))
    results, _, _ = structural_intervals._table(dict(boundary=boundary, optimizer_success=True, coordinates=point,
        calculations=calculations, level=.95, targets=[dict(Target="(R3-R1)_B - (R3-R1)_A")],
        target_jacobian=np.atleast_2d(weights @ design["target_jacobian"])))
    result, = results
    estimate, se = result["Estimate"], result["SE"]
    cfg = prepared["config"]
    group = engine.expand_params(point[:design["base_dimension"]], prepared["sizes"], cfg)["facets"]["School"]
    school = dict(zip(cfg["facet_levels"]["School"], group, strict=True))
    return dict(estimate=estimate, school_A=school["A"], school_B=school["B"], se=se, lower=result["Lower"],
                upper=result["Upper"], p=2*norm.sf(abs(estimate)/se) if se else None, reason=result["Reason"])


def run_replicate(rep):
    outputs = []
    for design, effect, rows in ratings_at(rep):
        assert rows.groupby(["School", "Rater"]).size().eq(80).all()
        audit = mfrmpy.check_rater_contrast(rows, **AUDIT)
        overlap = {r["group_level"]: r["kind"] for r in audit["overlap"]}
        outputs.append(dict(replicate=rep, design=design, effect=effect, truth=EFFECTS[effect],
                            support=audit["status"], overlap_A=overlap["A"], overlap_B=overlap["B"], **review(rows, audit)))
    return outputs


def controls():
    design = {**TARGETS, "single": (0, 0, 0)}
    rows = {name: data for name, effect, data in ratings_at(1, design) if effect == "ability_only"}
    base = rows["balanced24"]
    public = mfrmpy.analyze(base, **FIT, joint_settings=JOINT)["joint_analysis"]
    assert public["status"] == "unavailable" and "population-assumption-only" in public["reason"]
    one_school = pd.concat([base[base.School.eq("A")], rows["single"].loc[lambda d: d.School.eq("B")]], ignore_index=True)
    separate = base.assign(Rater=base.Rater.where(base.School.eq("A"), "B-"+base.Rater))
    output = []
    for label, data, expected in (("both_schools_linked", base, "observed"),
                                  ("only_A_shared_works", one_school, "common_location_only"),
                                  ("disjoint_panels", separate, "unsupported")):
        audit = mfrmpy.check_rater_contrast(data, **AUDIT)
        assert audit["status"] == expected, audit
        result = review(data, audit)
        output.append(dict(case=label, ratings=len(data), status=expected, public_reason=public["reason"] if label=="both_schools_linked" else "",
                           **{f"overlap_{x['group_level']}": x['kind'] for x in audit['overlap']}, **result))
    return pd.DataFrame(output)


def run(path, workers, end):
    complete = set()
    if path.exists():
        saved = pd.read_csv(path)
        assert tuple(saved.columns) == COLUMNS and saved.groupby("replicate").size().eq(4).all()
        complete.update(saved.replicate)
    pending = [rep for rep in range(1, end+1) if rep not in complete]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, rows in enumerate(pool.map(run_replicate, pending), 1):
            pd.DataFrame(rows).reindex(columns=COLUMNS).to_csv(path, mode="a", header=not path.exists(), index=False)
            if i % 25 == 0 or i == len(pending):
                print(f"{i}/{len(pending)} replicate sets", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("--end", type=int, default=REPLICATES)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--controls", action="store_true")
    args = parser.parse_args()
    if args.controls:
        controls().to_csv(args.out, index=False)
    else:
        run(args.out, args.workers, args.end)
