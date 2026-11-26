"""Build or reopen the annual anchor-sensitivity example from retained M5b evidence."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

import mfrmpy
from mfrm_app import output_qualification, project, structural_intervals
from validation.m5b_annual_drift import AUDIT, DESIGNS, ratings_at


EVIDENCE = Path(__file__).parent / "evidence"
KEY = "annual_anchor_sensitivity_example"
CASE = dict(design="repeat6", effect="anchors_drift", replicate=44)
TARGET = ("B-R3", "A-R3")
SETTINGS = dict(person="Recording", score="Score", rating_min=1, rating_max=5, method="MML",
                model="RSM", population_prior_sd=1., quad_points=61, reltol=1e-12)
DIFFERENCE = dict(facet="RaterYear", difference=TARGET)


def evidence_row(file, **selection):
    rows = pd.read_csv(EVIDENCE / file)
    for key, value in (CASE | selection).items():
        rows = rows.loc[rows[key].eq(value)]
    row, = rows.to_dict("records")
    if pd.notna(row.get("reason")):
        raise ValueError(f"Saved result was withheld: {row['reason']}")
    return row


def mode_control(ratings, anchors):
    modified = ratings.assign(Mode=ratings.RatingYear.map({"A": "live", "B": "video"}))
    fit = mfrmpy.analyze(modified, **SETTINGS, facets=["WorkYear", "RaterYear", "Mode"],
                         noncenter_facet="RaterYear", positive_facets=["WorkYear"])
    analysis = mfrmpy.with_anchor_sensitivity(fit, rater="RaterYear", difference=TARGET,
        scenarios={"Previous year only": dict(anchors=[a for a in anchors if a["Level"].startswith("A-")])})
    points = mfrmpy.tables(analysis, "anchor_sensitivity")[0][1]
    if not (points.AssignmentSupport.eq("unsupported").all() and points.Estimate.isna().all()):
        raise ValueError("Expected confounding of mode and year with and without previous-year anchors")
    return "unsupported"


def levels(analysis, facet):
    result = analysis["result"]
    reason = output_qualification.calibration_screening_block_reason(result)
    if reason:
        raise ValueError(reason)
    return dict(zip(result["config"]["facet_levels"][facet], result["params"]["facets"][facet], strict=True))


def fixed_scenario(values, retained, source):
    anchors = [dict(Facet="RaterYear", Level=f"{year}-{rater}", Anchor=values[rater], Source=source)
               for rater in retained for year in ("A", "B")]
    return dict(anchors=anchors, noncenter_facet="WorkYear")


def build_example():
    ratings = next(rows for _, effect, rows in ratings_at(CASE["replicate"], {CASE["design"]: DESIGNS[CASE["design"]]})
                   if effect == CASE["effect"])
    audit = mfrmpy.check_rater_contrast(ratings, **(AUDIT | dict(
        facets=["WorkYear", "RaterYear"], noncenter_facet="RaterYear", positive_facets=["WorkYear"],
    )))
    if audit["status"] != "observed":
        raise ValueError(f"Recordings-only link: {audit['status']}")
    baseline = mfrmpy.analyze(ratings, **SETTINGS, facets=["WorkYear", "RaterYear"],
        noncenter_facet="RaterYear", positive_facets=["WorkYear"], interval_requests=[DIFFERENCE])
    entry, = baseline["interval_analyses"]
    if entry["status"] != "computed":
        raise ValueError(entry["reason"])
    table = structural_intervals.read_structural_intervals(baseline["result"], entry["record"])
    origin = table.loc[table.Kind.eq("difference")].iloc[0]
    if not origin.CIEligible:
        raise ValueError(origin.Reason)
    old = evidence_row("m5b_annual_drift.csv.gz", model="recordings", target="R3")
    np.testing.assert_allclose(origin.Estimate, old["estimate"], rtol=0, atol=5e-5)
    np.testing.assert_allclose(origin.SE, old["se"], rtol=0, atol=2e-6)

    shared = evidence_row("m5b_estimated_anchors.csv.gz")
    prior = ratings.loc[ratings.RatingYear.eq("A")].copy()
    calibration = mfrmpy.analyze(prior, **SETTINGS, facets=["Rater"], noncenter_facet="Rater")
    values = {r: float(shared[f"anchor_{r}"]) for r in ("R1", "R2")}
    estimated = levels(calibration, "Rater")
    np.testing.assert_allclose([estimated[r] for r in values], list(values.values()), rtol=0, atol=5e-5)
    source = f"Year A, 60 recordings and 120 shared ratings ({calibration['analysis_id']})"
    scenarios = {name: fixed_scenario(values, retained, source) for name, retained in (
        ("R1・R2を前年値に固定", ("R1", "R2")), ("R1を外す", ("R2",)), ("R2を外す", ("R1",)),
    )}
    scenarios["各年のR1・R2の平均を前年値に固定"] = dict(noncenter_facet="WorkYear", group_anchors=[
        dict(Facet="RaterYear", Level=f"{year}-{rater}", Group=year,
             GroupValue=np.mean(list(values.values())), Source=source)
        for year in ("A", "B") for rater in values
    ])
    baseline = mfrmpy.with_anchor_sensitivity(baseline, rater="RaterYear", difference=TARGET, scenarios=scenarios)
    points = mfrmpy.tables(baseline, "anchor_sensitivity")[0][1]
    fits = [dict(name=row.Scenario, support=row.AssignmentSupport, estimate=row.Estimate,
                 reason=row.Reason, change_from_recordings=row.ChangeFromNoAnchors)
            for row in points.iloc[1:].itertuples()]
    np.testing.assert_allclose(points.Estimate.iloc[:2], [origin.Estimate, shared["estimate"]], rtol=0, atol=5e-5)

    ids_a = set(prior.Recording)
    ids_b = set(ratings.loc[ratings.RatingYear.eq("B"), "Recording"])
    if shared["shared_recordings"] != len(ids_a) or len(ids_a & ids_b) != 6:
        raise ValueError("Retained covariance evidence does not match the recording reuse")
    baseline[KEY] = dict(
        version=1, case=CASE, analysis_id=baseline["analysis_id"], target=TARGET, support=audit["status"],
        recording_overlap=sorted(ids_a & ids_b), path=audit["overlap"][0]["path"],
        mode_control=mode_control(ratings, scenarios["R1・R2を前年値に固定"]["anchors"]),
        source=dict(kind="shared_ratings", year="A", recording_ids=sorted(ids_a), ratings=len(prior),
                    calibration_id=calibration["analysis_id"], anchors=values,
                    evidence="m5b_estimated_anchors.csv.gz"),
        recordings=dict(estimate=float(origin.Estimate), se=float(origin.SE),
                        lower=float(origin.Lower), upper=float(origin.Upper)),
        anchored_interval={k: float(shared[k]) for k in
            ("estimate", "fixed_se", "score_shared_se", "score_cross_term")},
        scenarios=fits,
    )
    baseline["content_fingerprint"] = project.content_fingerprint(baseline)
    return baseline


def render(analysis):
    project.validate_analysis(analysis)
    record = analysis.get(KEY)
    if not record or record.get("version") != 1 or record.get("analysis_id") != analysis["analysis_id"]:
        raise ValueError("This project has no matching annual anchor-sensitivity example")
    case, source = record["case"], record["source"]
    ref, both = record["recordings"], record["anchored_interval"]
    z = norm.ppf(.975)
    print("# 年度をつなぐ基準で結論はどう変わるか\n")
    print(f"固定SDの5段階RSMを用いた合成例。各年120評定で、共通録画は6本。"
          f"既存の500反復から、符号と区間の判断が変わる反復{case['replicate']}を例示用に選んだ。"
          "真の変化はR3が0、R1・R2が各+0.5 logit。差の向きはR3の今年−前年で、正は厳しくなる方向。\n")
    print("## 指定したR3の年度差\n")
    print("| 条件 | 割付の判定 | 点推定 | 録画のみとの差 | SE | 95%区間または保留 |")
    print("|---|---|---:|---:|---:|---|")
    print(f"| 共通録画のみ | {record['support']} | {ref['estimate']:+.3f} | 0 | {ref['se']:.3f} | "
          f"[{ref['lower']:+.3f}, {ref['upper']:+.3f}] |")
    fixed_fit = record["scenarios"][0]
    center = fixed_fit["estimate"]
    for name, key in (("R1・R2を固定、校正値を既知とする条件付き計算", "fixed_se"),
                      ("R1・R2を固定、共有校正誤差を含む計算", "score_shared_se")):
        se = both[key]
        print(f"| {name} | {fixed_fit['support']} | {center:+.3f} | {center-ref['estimate']:+.3f} | {se:.3f} | "
              f"[{center-z*se:+.3f}, {center+z*se:+.3f}] |")
    for row in record["scenarios"][1:]:
        name = f"{row['name']} | {row['support']}"
        if row["reason"]:
            print(f"| {name} | — | — | — | {row['reason'].replace('|', '/')} |")
        else:
            print(f"| {name} | {row['estimate']:+.3f} | {row['change_from_recordings']:+.3f} | "
                  "— | 感度確認用の点推定 |")
    print("\nこの例で前年の2人を固定すると、R3が甘くなったように見える。録画のみでつないだ区間は"
          "ゼロを含み、符号も反転する。アンカーを一人ずつ外した結果にも負の差が残る。"
          "共有校正を考慮してもこの例の見かけのドリフトは残る。前年値からの変化を審査員一人に帰属させるには、"
          "どの審査員を安定と仮定したかを確認する。\n")
    print("## 比較を支える録画と前年値の出所\n")
    print(f"- 両年に評定がある録画。{', '.join(record['recording_overlap'])}。")
    for link in record["path"]:
        print(f"- 観測上の経路。{link['from_rater']} → {link['to_rater']}（{', '.join(link['persons'])}）。")
    fixed = "、".join(f"{r} {value:+.6f} logit" for r, value in source["anchors"].items())
    print(f"- 固定値。前年{source['year']}の{len(source['recording_ids'])}本・{source['ratings']}評定で校正した"
          f"{fixed}を、それぞれ両年の水準に固定。前年評定は年度差の推定にも再利用した。")
    print("- 校正値が既知という扱いは表の条件付き計算の仮定。独立校正として扱えるのは、"
          "校正標本と本番標本の独立性を確認できる場合。共有評定を使った本例では、"
          "共有共分散を含む区間を併記した。")
    print("- 一人ずつ外した条件は点推定の感度確認。区間の被覆は既存の反復実験の範囲で評価する。\n")
    print(f"**保留の対照例**　同じ6本を使っても、前年は会場・今年は録画とすると、方式が採点年度と完全に重なる。"
          f"この割付ではR3の年度差を識別できない（`{record['mode_control']}`）。\n")
    print("## プロジェクトを再読込する\n")
    print("リポジトリのルートから次のコマンドを実行すると、`.mfrm` に保持した比較表と根拠を再適合せず表示できる。\n")
    print("```bash\npython -m validation.m5b_anchor_report validation/evidence/m5b_anchor_sensitivity_example.mfrm --read"
          "\n```\n")
    if "anchor_sensitivity" in analysis:
        print("点推定の比較と根拠は、通常のプロジェクトを開いて `tables(analysis, \"anchor_sensitivity\")` からも読める。"
              "CSV・HTMLにも同じ表が入り、共有校正を含む区間は本検証スクリプトで確認できる。\n")
    else:
        print("この保存例の付帯情報は検証スクリプトから読める。\n")
    print("[年度差の検証](m5b_annual_drift.md)・[共有校正誤差の検証](m5b_estimated_anchors.md)"
          "・[独立校正の検証](m3_independent_anchors.md)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project", type=Path)
    parser.add_argument("--read", action="store_true", help="Render the retained example without fitting")
    args = parser.parse_args()
    if not args.read:
        mfrmpy.save_project(args.project, build_example())
    render(mfrmpy.load_project(args.project))
