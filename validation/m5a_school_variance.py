"""Check the fixed-SD DFF interval when only school B's generating SD changes."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

import mfrmpy
from validation.m2_sparse_intervals import REPLICATES, rate
from validation.m5a_direct_overlap import DESIGNS
from validation.m5a_group_dff import AUDIT, COLUMNS, EFFECTS, ratings_at, review


SCHOOL_SD = (1., 1.5)
DESIGN = "direct20"
EFFECT = "ability_only"
FIELDS = (*COLUMNS, "sd_A", "sd_B")
CONTROL = Path(__file__).with_name("evidence") / "m5a_direct_overlap.csv.gz"


def data_at(rep, sd=(1., 1.), *, effect=EFFECT):
    # Keep the original order: each assignment design has its own seed stream.
    return next(rows for design, condition, rows in ratings_at(rep, DESIGNS, school_sd=sd)
                if (design, condition) == (DESIGN, effect))


def matched_data_at(rep, effect=EFFECT):
    control, changed = data_at(rep, effect=effect), data_at(rep, SCHOOL_SD, effect=effect)
    pd.testing.assert_frame_equal(control.drop(columns="Score"), changed.drop(columns="Score"))
    pd.testing.assert_frame_equal(control.loc[control.School.eq("A")], changed.loc[changed.School.eq("A")])
    return control, changed


def run_replicate(rep, effect=EFFECT):
    _, changed = matched_data_at(rep, effect)
    assert changed.groupby(["School", "Rater"]).size().eq(80).all()
    audit = mfrmpy.check_rater_contrast(changed, **AUDIT)
    overlap = {r["group_level"]: r["kind"] for r in audit["overlap"]}
    assert audit["status"] == "observed" and overlap == {"A": "direct", "B": "direct"}
    assert all(len(g["path"][0]["persons"]) == 20 for g in audit["overlap"])
    return dict(replicate=rep, design=DESIGN, effect=effect, truth=EFFECTS[effect],
                support=audit["status"], overlap_A=overlap["A"], overlap_B=overlap["B"],
                sd_A=SCHOOL_SD[0], sd_B=SCHOOL_SD[1], **review(changed, audit))


def run(path, workers, end, effect=EFFECT):
    complete = set()
    if path.exists():
        saved = pd.read_csv(path)
        assert tuple(saved.columns) == FIELDS and saved.replicate.is_unique
        assert saved[["sd_A", "sd_B"]].eq(list(SCHOOL_SD)).all().all()
        assert saved.effect.eq(effect).all() and saved.truth.eq(EFFECTS[effect]).all()
        complete.update(saved.replicate)
    pending = [rep for rep in range(1, end+1) if rep not in complete]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, row in enumerate(pool.map(partial(run_replicate, effect=effect), pending), 1):
            pd.DataFrame([row]).reindex(columns=FIELDS).to_csv(path, mode="a", header=not path.exists(), index=False)
            if i % 25 == 0 or i == len(pending):
                print(f"{i}/{len(pending)} replicates", flush=True)


def check_results(rows, effect=EFFECT):
    assert list(rows.index) == list(range(1, REPLICATES+1))
    assert rows.design.eq(DESIGN).all() and rows.effect.eq(effect).all() and rows.truth.eq(EFFECTS[effect]).all()
    assert rows.support.eq("observed").all()
    assert rows[["overlap_A", "overlap_B"]].eq("direct").all().all()
    returned = rows.lower.notna()
    assert returned.equals(rows.upper.notna()) and returned.equals(rows.reason.isna())
    finite = rows.loc[returned, ["estimate", "se", "lower", "upper", "p"]]
    assert np.isfinite(finite).all().all() and finite.se.gt(0).all()
    assert finite.lower.le(finite.estimate).all() and finite.estimate.le(finite.upper).all()
    assert finite.p.between(0, 1).all()
    covered = finite.lower.le(0) & finite.upper.ge(0)
    assert covered.eq(finite.p.ge(.05)).all()
    return returned


def mml_results(path, effect=EFFECT):
    control = pd.read_csv(CONTROL).query("design == @DESIGN and effect == @effect").set_index("replicate").sort_index()
    varied = pd.read_csv(path).set_index("replicate").sort_index()
    assert varied[["sd_A", "sd_B"]].eq(list(SCHOOL_SD)).all().all()
    return control, varied


def summarize_pair(control, varied, effect=EFFECT):
    def percentage(count, total, ci):
        return f"{count}/{total} ({count/total:.1%}, 95% Wilson {ci.low:.2%}–{ci.high:.2%})" if total else "NA"

    truth = EFFECTS[effect]
    metric = "正方向の検出" if truth > 0 else "偽陽性"
    summaries, fits = [], []
    for label, rows in (("1 / 1", control), ("1 / 1.5", varied)):
        returned = check_results(rows, effect)
        finite = rows.loc[returned]
        hits = (finite.lower.gt(0) if truth > 0 else finite.p.lt(.05)).sum()
        covered = (finite.lower.le(truth) & finite.upper.ge(truth)).sum()
        n, k = len(rows), len(finite)
        hit_total = n if truth > 0 else k
        _, return_ci = rate(k, n)
        _, coverage_ci = rate(covered, k)
        _, hit_ci = rate(hits, hit_total)
        passed = (return_ci.low >= .80 and coverage_ci is not None
                  and coverage_ci.low >= .92 and coverage_ci.high <= .98)
        summaries.append(f"| {label} | {percentage(k, n, return_ci)} | {percentage(covered, k, coverage_ci)} | "
                         f"{percentage(hits, hit_total, hit_ci)} | {'適合' if passed else '条件未達'} |")
        error = rows.estimate - truth
        bias = error.mean()
        rmse = np.sqrt(error.pow(2).mean())
        rms_se = np.sqrt(finite.se.pow(2).mean())
        width = (finite.upper - finite.lower).median()
        fits.append(f"| {label} | {bias:.4f} | {rmse:.4f} | {rows.estimate.std():.4f} | {rms_se:.4f} | {width:.4f} |")
        print(f"{label}: 負側に外れた区間 {finite.upper.lt(0).sum()}件、正側に外れた区間 {finite.lower.gt(0).sum()}件")
        if not returned.all():
            print(label, rows.loc[~returned, "reason"].value_counts().to_dict())
    print(f"| 生成SD A / B | 返却 | 被覆 | {metric} | M2の基準 |")
    print("|---|---:|---:|---:|---|")
    print(*summaries, sep="\n")
    print("\n| 生成SD A / B | 偏り | RMSE | 推定値SD | SEの二乗平均平方根 | 区間幅の中央値 |")
    print("|---|---:|---:|---:|---:|---:|")
    print(*fits, sep="\n")
    paired_estimate = (varied.estimate - control.estimate).dropna()
    if truth > 0:
        paired_hits = varied.lower.gt(0).astype(int) - control.lower.gt(0).astype(int)
    else:
        both = control.lower.notna() & varied.lower.notna()
        paired_hits = varied.loc[both, "p"].lt(.05).astype(int) - control.loc[both, "p"].lt(.05).astype(int)
    print(f"\nSD 1.5のみ{metric} {paired_hits.eq(1).sum()}件、SD 1のみ{metric} {paired_hits.eq(-1).sum()}件")
    for label, change in (("推定差（logit）", paired_estimate), (f"{metric}率の差（ポイント）", 100 * paired_hits)):
        mean, half_width = change.mean(), 1.96 * change.sem()
        print(f"{label} {mean:+.4f}（{len(change)}対、95% MC区間 {mean-half_width:+.4f} ～ {mean+half_width:+.4f}）")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--end", type=int, default=REPLICATES)
    parser.add_argument("--effect", choices=EFFECTS, default=EFFECT)
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    if args.summarize:
        summarize_pair(*mml_results(args.out, args.effect), effect=args.effect)
    else:
        run(args.out, args.workers, args.end, args.effect)
