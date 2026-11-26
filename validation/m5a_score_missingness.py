"""Compare equal numbers of score-independent and low-score omissions in M5a."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

import mfrmpy
from validation.m2_sparse_intervals import REPLICATES, rate
from validation.m5a_group_dff import AUDIT, review
from validation.m5a_school_variance import CONTROL, DESIGN, EFFECT, data_at
from validation.m5a_school_variance_cml import CML_NULL


SCENARIOS = ("random", "lowest")
N_MISSING = 12
KEY = ["replicate", "scenario"]
ESTIMATES = ["estimate", "se", "lower", "upper", "p", "reason"]
FIELDS = [*KEY, "ratings", "persons", "omitted_mean", "omitted_minimum", "support",
          "without_endpoint_works", "shared_A", "shared_B", "school_A", "school_B", *ESTIMATES]


def incomplete_ratings(rep):
    full = data_at(rep)
    eligible = full.loc[full.School.eq("B") & full.Rater.eq("R3")]
    assert len(full) == 480 and len(eligible) == 80
    rng = np.random.default_rng(np.random.SeedSequence([2026112107, rep]))
    shuffled = eligible.iloc[rng.permutation(len(eligible))]
    missing = (shuffled.head(N_MISSING), shuffled.sort_values("Score", kind="stable").head(N_MISSING))
    for scenario, absent in zip(SCENARIOS, missing, strict=True):
        retained = full.drop(index=absent.index)
        assert len(absent) == N_MISSING and len(retained) == 468
        yield scenario, retained, absent.Score


def export_ratings(path, end):
    with path.open("w") as output:
        for rep in range(1, end + 1):
            for scenario, ratings, _ in incomplete_ratings(rep):
                ratings.assign(replicate=rep, scenario=scenario).to_csv(
                    output, index=False, header=rep == 1 and scenario == SCENARIOS[0]
                )


def evaluate(rep):
    results = []
    for scenario, ratings, absent in incomplete_ratings(rep):
        audit = mfrmpy.check_rater_contrast(ratings, **AUDIT)
        overlaps = {group["group_level"]: group for group in audit["overlap"]}
        shared = {school: len(group["path"][0]["persons"]) if group["kind"] == "direct" else 0
                  for school, group in overlaps.items()}
        results.append(dict(
            replicate=rep, scenario=scenario, ratings=len(ratings), persons=ratings.Person.nunique(),
            omitted_mean=absent.mean(), omitted_minimum=absent.eq(1).sum(),
            support=audit["status"], without_endpoint_works=audit["score_support"]["observation_estimable_without_endpoint_works"],
            shared_A=shared["A"], shared_B=shared["B"], **review(ratings, audit)
        ))
    return results


def run(path, end, workers):
    done = set()
    if path.exists():
        saved = pd.read_csv(path)
        assert list(saved.columns) == FIELDS and not saved.duplicated(KEY).any()
        assert saved.scenario.isin(SCENARIOS).all() and saved.groupby("replicate").size().eq(len(SCENARIOS)).all()
        done = set(saved.replicate)
    pending = sorted(set(range(1, end + 1)) - done)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for count, results in enumerate(pool.map(evaluate, pending), 1):
            # Each checkpoint includes both scenarios for the same replicated allocation.
            frame = pd.DataFrame(results).reindex(columns=FIELDS)
            frame.to_csv(path, index=False, mode="a", header=not path.exists())
            if count % 25 == 0 or count == len(pending):
                print(f"{count}/{len(pending)} paired replicates", flush=True)


def results(path):
    index = pd.MultiIndex.from_product([range(1, REPLICATES + 1), SCENARIOS], names=KEY)
    retained = pd.read_csv(path).set_index(KEY)
    assert retained.index.is_unique and len(retained) == len(index) and index.isin(retained.index).all()
    return retained.reindex(index)


def summarize(mml_path, cml_path):
    mml, cml = results(mml_path), results(cml_path)
    mml_full = pd.read_csv(CONTROL).query("design == @DESIGN and effect == @EFFECT").set_index("replicate")
    cml_full = pd.read_csv(CML_NULL).query("sd_B == 1").set_index("replicate")
    assert mml_full.index.is_unique and cml_full.index.is_unique
    for method, complete, incomplete in (("MML", mml_full, mml), ("CML", cml_full, cml)):
        scenarios = {name: incomplete.xs(name, level="scenario") for name in SCENARIOS}
        for scenario, fit in {"complete": complete, **scenarios}.items():
            eligible = fit.lower.notna()
            assert len(fit) == REPLICATES and eligible.equals(fit.upper.notna()) and eligible.equals(fit.reason.isna())
            retained = fit.loc[eligible]
            assert np.isfinite(retained[ESTIMATES[:-1]]).all().all() and retained.se.gt(0).all()
            coverage = retained.lower.le(0) & retained.upper.ge(0)
            assert coverage.eq(retained.p.ge(.05)).all()
            _, return_ci = rate(len(retained), REPLICATES)
            _, coverage_ci = rate(coverage.sum(), len(retained))
            qualified = (return_ci.low >= .80 and coverage_ci is not None
                         and coverage_ci.low >= .92 and coverage_ci.high <= .98)
            limits = f"({coverage_ci.low:.4f}, {coverage_ci.high:.4f})" if coverage_ci is not None else "NA"
            print(method, scenario, len(retained), f"return={len(retained)/REPLICATES:.1%}",
                  f"cover={coverage.mean():.1%}",
                  f"Wilson={limits}", f"qualified={qualified}",
                  f"positive={retained.lower.gt(0).sum()}", f"negative={retained.upper.lt(0).sum()}",
                  f"bias={retained.estimate.mean():+.4f}", f"width={(retained.upper-retained.lower).median():.4f}",
                  fit.loc[~eligible, "reason"].value_counts().to_dict())
        finite = scenarios["lowest"].lower.notna() & scenarios["random"].lower.notna()
        delta = (scenarios["lowest"].estimate - scenarios["random"].estimate)[finite]
        half = 1.96 * delta.sem()
        print(method, f"paired n={len(delta)}, shift={delta.mean():+.4f}, "
              f"95% MC interval=({delta.mean()-half:+.4f}, {delta.mean()+half:+.4f})")
        rejection = {name: fit.lower.gt(0) | fit.upper.lt(0) for name, fit in scenarios.items()}
        low_only = (rejection["lowest"] & ~rejection["random"]).sum()
        random_only = (rejection["random"] & ~rejection["lowest"]).sum()
        print(method, f"rejections only under low-score omissions={low_only}, "
              f"only under random omissions={random_only}")
    for scenario in SCENARIOS:
        subset = mml.xs(scenario, level="scenario")
        print(scenario, f"omitted mean={subset.omitted_mean.mean():.4f}", f"min persons={subset.persons.min()}",
              f"B shared min/median={subset.shared_B.min()}/{subset.shared_B.median():g}",
              subset.support.value_counts().to_dict(), subset.without_endpoint_works.value_counts().to_dict())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("export", "mml", "summarize"))
    parser.add_argument("path", type=Path)
    parser.add_argument("--cml", type=Path)
    parser.add_argument("--end", type=int, default=REPLICATES)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.action == "export":
        export_ratings(args.path, args.end)
    elif args.action == "mml":
        run(args.path, args.end, args.workers)
    else:
        if args.cml is None:
            parser.error("summarize requires --cml")
        summarize(args.path, args.cml)
