"""Compare annual severity and category changes with matched recordings."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

import mfrmpy
from mfrm_app import mml_joint, structural_intervals
from validation.m2_sparse_intervals import REPLICATES, rate
from validation.m5b_annual_drift import (
    AUDIT, CONFIG, DESIGNS, DRIFT_COLUMNS, LABELS, NODES, fit_stationary, ratings_at,
)


SHIFTS = {"common": (0., 0., 0., 0.), "middle": (-1., -1., 1., 1.), "four": (0., 0., -2., 2.)}
MODELS = {
    "RSM": CONFIG,
    "PCM_shared_except_B_R3": dict(CONFIG, model="PCM", step_facet="StepProfile",
                                  facet_levels=CONFIG["facet_levels"] | {"StepProfile": ["common", "separate"]}),
    "PCM_RaterYear": dict(CONFIG, model="PCM", step_facet="RaterYear"),
}
BASELINE = Path(__file__).with_name("evidence") / "m5b_annual_drift.csv.gz"
KEY = ["replicate", "condition", "model"]
TARGETS = ["severity", "width3", "width4"]
FIELDS = [*KEY, "support", "unused_step_categories", "reason",
          *(f"score_{k}" for k in range(1, 6)),
          *(f"{target}_{field}" for target in TARGETS for field in ("estimate", "se", "lower", "upper"))]
WIDTHS = {"width3": np.array([0., -1., 1.]), "width4": np.array([-1., -1., -2.])}
DESIGN = {"repeat24": DESIGNS["repeat24"]}


def cases_at(rep):
    _, effect, control = next(ratings_at(rep, DESIGN))
    assert effect == "cohort_only" and len(control) == 240
    changed = control.RaterYear.eq("B-R3")
    assert changed.sum() == 40
    for name, shift in SHIFTS.items():
        rows = control if name == "common" else next(ratings_at(rep, DESIGN, r3_step_shift=shift))[2]
        assert rows.drop(columns="Score").equals(control.drop(columns="Score"))
        assert rows.loc[~changed, "Score"].equals(control.loc[~changed, "Score"])
        yield name, rows.assign(StepProfile=np.where(changed, "separate", "common"))


def unused_categories(rows, step_owner):
    groups = [("all", rows)] if step_owner is None else rows.groupby(step_owner, observed=True)
    return ",".join(f"{label}:{k}" for label, group in groups
                    for k in sorted(set(range(1, 6)).difference(group.Score)))


def contrasts(config, dimension):
    weights = {"severity": np.r_[DRIFT_COLUMNS[2, :7], np.zeros(dimension-7)]}
    if config["model"] == "PCM":
        owners = config["facet_levels"][config["step_facet"]]
        a, b = (0, 1) if config["step_facet"] == "StepProfile" else (LABELS.index("A-R3"), LABELS.index("B-R3"))
        for name, step_weight in WIDTHS.items():
            weight = np.zeros(dimension)
            weight[7+3*a:10+3*a], weight[7+3*b:10+3*b] = -step_weight, step_weight
            weights[name] = weight
    return weights


def fit_case(rows, cfg, support):
    if not support:
        return dict(reason="Assignment support unavailable")
    likelihood = mml_joint.GroupedMmlLikelihood(rows.rename(columns={"Recording": "Person"}), cfg)
    boundary = likelihood.structural_recession_audit()
    d = likelihood.linear_d
    if boundary["status"] != "no_structural_ray_detected" or boundary["structural_design_rank"] != d:
        return dict(reason=f"Design rank or boundary: {boundary['status']}")
    coordinates = np.eye(d+1)[:, :d]
    try:
        point, _, _, _ = fit_stationary(likelihood, coordinates, np.zeros(d+1))
    except (ValueError, np.linalg.LinAlgError) as exc:
        return dict(reason=f"Marginal fit: {exc}")
    calculations = []
    for q in (len(NODES), 2*len(NODES)-1):
        at = likelihood.evaluate(np.r_[point, 0.], *mml_joint.normal_rule(q), information=True, moments=False)
        calculations.append(dict(points=q, information=at["information"][:-1, :-1], gradient=at["gradient"][:-1]))
    checks, covariance = structural_intervals._assess(dict(coordinates=point, optimizer_success=True,
                                                           boundary=boundary, calculations=calculations))
    out = dict(reason=" | ".join(check["check"] for check in checks if not check["passed"]))
    for name, weight in contrasts(cfg, d).items():
        center = float(weight @ point)
        out[f"{name}_estimate"] = center
        if not out["reason"]:
            se = np.sqrt(weight @ covariance @ weight)
            radius = norm.ppf(.975)*se
            out.update({f"{name}_se": se, f"{name}_lower": center-radius, f"{name}_upper": center+radius})
    return out


def evaluate(rep):
    results = []
    for condition, rows in cases_at(rep):
        audit = mfrmpy.check_rater_contrast(rows, **(AUDIT | dict(facets=["WorkYear","RaterYear"])))
        counts = rows.loc[rows.RaterYear.eq("B-R3"), "Score"].value_counts()
        categories = {f"score_{k}": counts.get(k, 0) for k in range(1, 6)}
        for model, cfg in MODELS.items():
            gaps = unused_categories(rows, cfg.get("step_facet"))
            reuse_baseline = (condition, model) == ("common", "RSM") and rep <= REPLICATES
            fit = {} if reuse_baseline else fit_case(rows, cfg, audit["observation_estimable"])
            results.append(dict(replicate=rep, condition=condition, model=model, support=audit["status"],
                                unused_step_categories=gaps, **categories, **fit))
    return results


def run(path, start, end, workers):
    completed = set()
    if path.exists():
        saved = pd.read_csv(path)
        assert list(saved.columns) == FIELDS and not saved.duplicated(KEY).any()
        assert saved.groupby("replicate").size().eq(len(SHIFTS)*len(MODELS)).all()
        assert saved.condition.isin(SHIFTS).all() and saved.model.isin(MODELS).all()
        completed = set(saved.replicate)
    pending = sorted(set(range(start, end+1)) - completed)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for count, rows in enumerate(pool.map(evaluate, pending), 1):
            pd.DataFrame(rows).reindex(columns=FIELDS).to_csv(path, index=False, mode="a", header=not path.exists())
            if count % 25 == 0 or count == len(pending):
                print(f"{count}/{len(pending)} replicate sets", flush=True)


def summarize(paths, end):
    data = pd.concat([pd.read_csv(path) for path in paths]).set_index(KEY).sort_index()
    expected = pd.MultiIndex.from_product([range(1, end+1), SHIFTS, MODELS], names=KEY)
    assert data.index.is_unique and len(data) == len(expected) and expected.isin(data.index).all()
    baseline = pd.read_csv(BASELINE).query(
        "design == 'repeat24' and effect == 'cohort_only' and model == 'recordings' and target == 'R3' and replicate <= @end"
    ).assign(condition="common", model="RSM").set_index(KEY)
    cols = ["estimate", "se", "lower", "upper"]
    assert len(baseline) == min(end, REPLICATES) and baseline.index.is_unique and baseline.reason.isna().all()
    data.loc[baseline.index, [f"severity_{col}" for col in cols]] = baseline[cols].to_numpy(dtype=float)
    assert data.severity_lower.notna().equals(data.reason.isna())
    for (condition, model), group in data.groupby(level=["condition","model"]):
        fit = group.loc[group.severity_lower.notna()]
        _, return_ci = rate(len(fit), end)
        print(condition, model, "returned", len(fit), "return Wilson", return_ci,
              "gap owners", group.unused_step_categories.notna().sum(), "holds", group.reason.value_counts().to_dict())
        for name in (["severity"] if model=="RSM" else TARGETS):
            result = fit.rename(columns=lambda key: key.removeprefix(name+"_"))
            assert np.isfinite(result[cols]).all().all() and result.se.gt(0).all()
            truth = 0. if name == "severity" else WIDTHS[name] @ np.array(SHIFTS[condition][:3])
            covered = result.lower.le(truth) & result.upper.ge(truth)
            _, coverage_ci = rate(covered.sum(), len(fit))
            qualified = (return_ci.low >= .80 and coverage_ci is not None
                         and coverage_ci.low >= .92 and coverage_ci.high <= .98)
            error = result.estimate - truth
            print(name, f"truth={truth:+.1f}", f"bias={error.mean():+.4f} ± {1.96*error.sem():.4f}",
                  f"covered={covered.sum()} ({covered.mean():.1%})", coverage_ci, f"qualified={qualified}",
                  f"negative/positive={result.upper.lt(0).sum()}/{result.lower.gt(0).sum()}",
                  f"median width={(result.upper-result.lower).median():.4f}")
    count_columns = [f"score_{k}" for k in range(1, 6)]
    print(data.xs("RSM", level="model").groupby(level="condition")[count_columns].mean())
    four = data.xs("four", level="condition")
    samples = {"all": four}
    if end > REPLICATES:
        samples["additional"] = four.loc[four.index.get_level_values("replicate") > REPLICATES]
    for label, sample in samples.items():
        rsm = sample.xs("RSM", level="model")
        pcm = sample.xs("PCM_shared_except_B_R3", level="model")
        available = rsm.severity_lower.notna() & pcm.severity_lower.notna()
        delta = (pcm.severity_estimate - rsm.severity_estimate).loc[available]
        _, return_ci = rate(pcm.severity_lower.notna().sum(), len(pcm))
        excluded = rsm.severity_upper.lt(0) | rsm.severity_lower.gt(0)
        _, error_ci = rate(excluded.sum(), rsm.severity_lower.notna().sum())
        print(label, "four", f"paired={len(delta)}",
              f"PCM minus RSM={delta.mean():+.4f} ± {1.96*delta.sem():.4f}",
              f"RSM false positives={excluded.sum()}/{rsm.severity_lower.notna().sum()}", error_ci,
              f"targeted PCM return={pcm.severity_lower.notna().sum()}/{len(pcm)}", return_ci)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=REPLICATES)
    parser.add_argument("--previous", type=Path, help="earlier replicates to include when summarizing")
    parser.add_argument("--workers", type=int, default=1)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--summarize", action="store_true")
    mode.add_argument("--pilot-ratings", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.start <= args.end:
        parser.error("require 1 <= start <= end")
    if args.previous is not None and not args.summarize:
        parser.error("--previous requires --summarize")
    if args.summarize:
        paths = [args.path] if args.previous is None else [args.previous, args.path]
        summarize(paths, args.end)
    elif args.pilot_ratings:
        pd.concat(rows.assign(replicate=rep, condition=label) for rep in range(args.start, args.end+1)
                  for label, rows in cases_at(rep)).to_csv(args.path, index=False)
    else:
        run(args.path, args.start, args.end, args.workers)
