"""Export, summarize and independently verify the matched M5a CML comparison."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from mfrm_app.cmle import fit_cmle, prepare_cmle_design
from mfrm_app.cmle_existence import audit_cmle_finite_mle_oracle
from validation.m5a_school_variance import (
    EFFECT, EFFECTS, REPLICATES, SCHOOL_SD, check_results, data_at, matched_data_at, mml_results, summarize_pair,
)


MML_VARIANCE = Path(__file__).with_name("evidence") / "m5a_school_variance.csv.gz"
CML_NULL = MML_VARIANCE.with_name("m5a_school_variance_cml.csv.gz")
ESTIMATES = ["estimate", "se", "lower", "upper", "p", "reason"]


def export_ratings(path, effect=EFFECT):
    with path.open("w") as output:
        for rep in range(1, REPLICATES + 1):
            control, varied = matched_data_at(rep, effect)
            paired = pd.concat([control.assign(sd_B=1.), varied.assign(sd_B=SCHOOL_SD[1])])
            paired.assign(replicate=rep, effect=effect).to_csv(output, index=False, header=rep == 1)


def read_results(path, effect):
    raw = pd.read_csv(path).set_index(["sd_B", "replicate"]).sort_index()
    saved_effects = set(raw["effect"]) if "effect" in raw else {EFFECT}
    assert saved_effects == {effect}
    assert raw.index.is_unique and set(raw.index.get_level_values("sd_B")) == {1., SCHOOL_SD[1]}
    return raw


def summarize(path, effect=EFFECT):
    raw = read_results(path, effect)
    mml_path = MML_VARIANCE if effect == EFFECT else MML_VARIANCE.with_name("m5a_school_variance_nonzero.csv.gz")
    mml = mml_results(mml_path, effect)
    cml = []
    for sd, control in zip((1., SCHOOL_SD[1]), mml, strict=True):
        fit = raw.loc[sd]
        assert fit.index.equals(control.index) and np.isfinite(fit.estimate.dropna()).all()
        valid = fit.lower.notna()
        assert fit.loc[valid, "rank"].eq(10).all() and fit.loc[valid, "dimension"].eq(10).all()
        assert fit.loc[valid, "boundary"].eq("no_recession_detected").all()
        assert fit.loc[valid, "numerical_state"].eq("ready").all()
        assert fit.informative_persons.between(0, 120).all()
        cml.append(control.drop(columns=ESTIMATES).join(fit[ESTIMATES], validate="one_to_one"))
        check_results(control, effect)
        print(f"B校のSD {sd:g}、CMLの有情報受験者数の中央値 {fit.informative_persons.median():g}/120")
    summarize_pair(*cml, effect=effect)

    shifts = (cml[1].estimate - cml[0].estimate) - (mml[1].estimate - mml[0].estimate)
    comparisons = [(f"B校のSD {sd:g}での CML−MML（logit）", c.estimate - m.estimate)
                   for sd, c, m in zip((1., SCHOOL_SD[1]), cml, mml, strict=True)]
    comparisons.append(("SD変更による移動量の差 CML−MML（logit）", shifts))
    if EFFECTS[effect] > 0:
        for sd, c, m in zip((1., SCHOOL_SD[1]), cml, mml, strict=True):
            gain = c.lower.gt(0).astype(int) - m.lower.gt(0).astype(int)
            print(f"B校のSD {sd:g}、CMLのみ正方向検出 {gain.eq(1).sum()}件、MMLのみ {gain.eq(-1).sum()}件")
            comparisons.append((f"B校のSD {sd:g}での正方向検出率の差 CML−MML（ポイント）", 100 * gain))
        mml_null = mml_results(MML_VARIANCE)
        cml_null = read_results(CML_NULL, EFFECT)
        null_mml_shift = mml_null[1].estimate - mml_null[0].estimate
        null_cml_shift = cml_null.loc[SCHOOL_SD[1]].estimate - cml_null.loc[1.].estimate
        comparisons.extend([
            ("SD変更による移動量の差 真値0.5−真値0 MML（logit）",
             mml[1].estimate - mml[0].estimate - null_mml_shift),
            ("SD変更による移動量の差 真値0.5−真値0 CML（logit）",
             cml[1].estimate - cml[0].estimate - null_cml_shift),
        ])
    for label, difference in comparisons:
        finite = difference.dropna()
        center, half_width = finite.mean(), 1.96 * finite.sem()
        print(f"{label} {center:+.4f}（{len(finite)}対、95% MC区間 "
              f"{center-half_width:+.4f} ～ {center+half_width:+.4f}）")


def native_settings(reference):
    # Within-school location shifts are invisible to the conditional likelihood.
    anchors = [dict(ParameterType="Facet", Facet="SchoolRater", Level=f"{school}_{reference}", Value=0.)
               for school in ("A", "B")]
    return dict(person_col="Person", facet_cols=["Item", "SchoolRater"], score_col="Score",
                rating_min=1, rating_max=5, hard_anchors=anchors)


def verify(path, effect=EFFECT):
    saved = read_results(path, effect)
    critical = norm.ppf(.975)
    weights = {"A_R1": 1, "A_R3": -1, "B_R1": -1, "B_R3": 1}
    free_weights = {f"facet:SchoolRater:free:{key}": value for key, value in weights.items()}
    discrepancies = []
    for sd in (1., SCHOOL_SD[1]):
        for rep in (1, REPLICATES // 2, REPLICATES):
            if pd.isna(saved.loc[(sd, rep), "lower"]):
                continue
            data = data_at(rep, (1., sd), effect=effect)
            data["SchoolRater"] = data.School + "_" + data.Rater
            result = saved.loc[(sd, rep)]
            for reference in (("R1", "R2") if rep == 1 else ("R1",)):
                fit = fit_cmle(data, **native_settings(reference))
                diagnostic = fit["summary"].iloc[0]
                assert diagnostic.InferenceReady and diagnostic.InformationRank == 10
                assert diagnostic.PersonsInformative == result.informative_persons
                coefficient = fit["coefficients"]
                contrast = coefficient.Parameter.map(free_weights).fillna(0).to_numpy()
                assert np.count_nonzero(contrast) == (2 if reference == "R1" else 4)
                estimate = contrast @ coefficient.Estimate.to_numpy()
                se = np.sqrt(contrast @ fit["covariance"] @ contrast)
                py = np.array([estimate, se, estimate - critical * se, estimate + critical * se])
                reference_values = result[ESTIMATES[:4]].to_numpy(dtype=float)
                np.testing.assert_allclose(py, reference_values, atol=1e-6, rtol=0)
                discrepancies.append(abs(reference_values - py))
    print(f"{len(discrepancies)} independent fits, max absolute differences: "
          f"{dict(zip(ESTIMATES[:4], np.max(discrepancies, axis=0).tolist(), strict=True))}")

    for (sd, rep), _ in saved.loc[saved.lower.isna()].iterrows():
        data = data_at(rep, (1., sd), effect=effect)
        data["SchoolRater"] = data.School + "_" + data.Rater
        design = prepare_cmle_design(data, **native_settings("R1"))
        audit = audit_cmle_finite_mle_oracle(design)
        diagnostic = audit["summary"].iloc[0]
        assert diagnostic.Status == "boundary_no_finite_cmle" and diagnostic.PrefitRank == 10
        assert diagnostic.OracleComplete and diagnostic.LPComplete and diagnostic.DirectionalObjectiveNonincreasing
        direction = pd.Series(audit["direction"], index=design.parameter_names)
        print(f"SD {sd:g}, replicate {rep}: {diagnostic.Status}, "
              f"direction {direction[direction.abs().gt(1e-10)].to_dict()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    operations = dict(export=export_ratings, summarize=summarize, verify=verify)
    parser.add_argument("action", choices=operations)
    parser.add_argument("path", type=Path)
    parser.add_argument("--effect", choices=EFFECTS, default=EFFECT)
    args = parser.parse_args()
    operations[args.action](args.path, args.effect)
