"""Compare annual severity contrasts under matched replay score bands."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

import mfrmpy
from validation.m2_sparse_intervals import REPLICATES, rate
from validation.m5b_annual_drift import AUDIT, DESIGNS, EFFECTS, ratings_at, targets_at


BANDS = ("spread", "low", "high")
CONDITIONS = ("cohort_only", "r3_drifts")
KEY = ["replicate", "design", "effect"]
FIELDS = [*KEY, "truth", "support", "without_endpoint_works", "old_sum",
          "old_floor", "old_ceiling", "extreme_works", "r3_raw_drop",
          "estimate", "se", "lower", "upper", "p", "reason"]
BASELINE = Path(__file__).with_name("evidence") / "m5b_annual_drift.csv.gz"


def cases_at(rep):
    reference = {}
    for band in BANDS:
        for _, effect, rows in ratings_at(rep, {"repeat24": DESIGNS["repeat24"]}, score_band=band):
            if effect not in CONDITIONS:
                continue
            replay = rows.RatingYear.eq("B") & rows.WorkYear.eq("A")
            pairs = rows.loc[replay].groupby("Recording").Rater.agg(tuple).value_counts().sort_index()
            common = rows.loc[~replay].reset_index(drop=True)
            assert len(rows) == 240 and len(pairs) <= 3 and pairs.sum() == 24
            assert rows.groupby(["RatingYear", "Rater"]).size().eq(40).all()
            if band == "spread":
                reference[effect] = (common, pairs)
            else:
                expected, expected_pairs = reference[effect]
                assert common.equals(expected) and pairs.equals(expected_pairs)
            yield band, effect, rows


def design_evidence(rows):
    replay_ids = rows.loc[rows.RatingYear.eq("B") & rows.WorkYear.eq("A"), "Recording"]
    replay = rows.loc[rows.Recording.isin(replay_ids)]
    wide = replay.pivot(index=["Recording", "Rater"], columns="RatingYear", values="Score")
    old_sum = wide.A.groupby(level="Recording").sum()
    by_work = replay.groupby("Recording").Score.agg(["min", "max"])
    r3 = wide.xs("R3", level="Rater")
    assert len(replay) == 96 and wide.notna().all().all()
    return dict(old_sum=old_sum.mean(), old_floor=old_sum.eq(2).sum(), old_ceiling=old_sum.eq(10).sum(),
                extreme_works=(by_work["min"].eq(by_work["max"]) & by_work["min"].isin([1, 5])).sum(),
                r3_raw_drop=(r3.A - r3.B).mean())


def evaluate(rep):
    result = []
    for band, effect, rows in cases_at(rep):
        settings = AUDIT | dict(facets=["WorkYear", "RaterYear"], positive_facets=["WorkYear"],
                                noncenter_facet="RaterYear")
        support = mfrmpy.check_rater_contrast(rows, **settings)
        details = design_evidence(rows)
        if band == "spread":
            fit = dict(truth=EFFECTS[effect][-1])
        else:
            fit = next(targets_at(rows, support, effect))
            assert fit["target"] == "R3" and fit["model"] == "recordings"
        result.append(dict(replicate=rep, design=band, effect=effect, support=support["status"],
                           without_endpoint_works=support["score_support"]["observation_estimable_without_endpoint_works"],
                           **details, **fit))
    return result


def run(path, end, workers):
    done = set()
    if path.exists():
        saved = pd.read_csv(path)
        assert list(saved.columns) == FIELDS and not saved.duplicated(KEY).any()
        assert saved.groupby("replicate").size().eq(len(BANDS)*len(CONDITIONS)).all()
        assert saved.design.isin(BANDS).all() and saved.effect.isin(CONDITIONS).all()
        done = set(saved.replicate)
    pending = sorted(set(range(1, end+1)) - done)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for count, rows in enumerate(pool.map(evaluate, pending), 1):
            pd.DataFrame(rows).reindex(columns=FIELDS).to_csv(path, index=False, mode="a", header=not path.exists())
            if count % 25 == 0 or count == len(pending):
                print(f"{count}/{len(pending)} replicate sets", flush=True)


def export(path, end):
    with path.open("w") as output:
        for rep in range(1, end+1):
            for band, effect, rows in cases_at(rep):
                rows.assign(replicate=rep, design=band, effect=effect).to_csv(
                    output, index=False, header=rep == 1 and band == BANDS[0] and effect == CONDITIONS[0]
                )


def checked(path):
    data = pd.read_csv(path).set_index(KEY).sort_index()
    expected = pd.MultiIndex.from_product([range(1, REPLICATES+1), BANDS, CONDITIONS], names=KEY)
    assert data.index.is_unique and len(data) == len(expected) and expected.isin(data.index).all()
    return data


def summarize(mml_path, cml_path):
    mml, cml = checked(mml_path), checked(cml_path)
    cols = ["estimate", "se", "lower", "upper", "p", "reason"]
    base = pd.read_csv(BASELINE).query("design == 'repeat24' and model == 'recordings' and target == 'R3'")
    base = base.loc[base.effect.isin(CONDITIONS)].assign(design="spread").set_index(KEY)
    assert base.index.is_unique and len(base) == REPLICATES*len(CONDITIONS)
    assert mml.loc[base.index, "truth"].equals(base["truth"])
    mml.loc[base.index, cols] = base[cols]
    for name, estimates in (("MML", mml), ("CML", cml)):
        for effect in CONDITIONS:
            truth = EFFECTS[effect][-1]
            found = estimates.xs(effect, level="effect")
            spread = found.xs("spread", level="design").dropna(subset=["lower"]).estimate
            for band in BANDS:
                group = found.xs(band, level="design")
                returned = group.lower.notna()
                assert returned.equals(group.upper.notna()) and returned.equals(group.reason.isna())
                fits = group.loc[returned]
                assert np.isfinite(fits[cols[:-1]]).all().all() and fits.se.gt(0).all()
                covered = fits.lower.le(truth) & fits.upper.ge(truth)
                positive, negative = fits.lower.gt(0).sum(), fits.upper.lt(0).sum()
                _, return_ci = rate(len(fits), REPLICATES)
                _, coverage_ci = rate(covered.sum(), len(fits))
                limits = (coverage_ci.low, coverage_ci.high) if coverage_ci is not None else (np.nan, np.nan)
                qualified = return_ci.low >= .80 and limits[0] >= .92 and limits[1] <= .98
                paired = (fits.estimate - spread).dropna()
                half = 1.96 * paired.sem()
                print(name, effect, band, f"returned={len(fits)}, covered={covered.sum()} ({covered.mean():.1%}), "
                      f"Wilson=({limits[0]:.4f},{limits[1]:.4f}), qualified={qualified}", f"signs=({positive},{negative}), "
                      f"bias={(fits.estimate-truth).mean():+.4f}, median width={(fits.upper-fits.lower).median():.4f}, "
                      f"paired change={paired.mean():+.4f} ± {half:.4f}", group.loc[~returned, "reason"].value_counts().to_dict())
    for (effect, band), data in mml.groupby(level=["effect","design"]):
        print(effect, band, data[["old_sum","old_floor","old_ceiling","extreme_works","r3_raw_drop"]].mean().round(4).to_dict(),
              data.support.value_counts().to_dict(), data.without_endpoint_works.value_counts().to_dict())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("export", "mml", "summarize"))
    parser.add_argument("path", type=Path)
    parser.add_argument("--cml", type=Path)
    parser.add_argument("--end", type=int, default=REPLICATES)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.action == "export":
        export(args.path, args.end)
    elif args.action == "mml":
        run(args.path, args.end, args.workers)
    else:
        if args.cml is None:
            parser.error("summarize requires --cml")
        summarize(args.path, args.cml)
