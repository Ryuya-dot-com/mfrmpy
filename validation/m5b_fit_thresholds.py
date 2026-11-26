"""Compare fit-screening rules on the saved annual-drift experiments."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from mfrm_app.fit_threshold_sensitivity import (
    CANONICAL_FIT_THRESHOLDS,
    classify_fit_mnsq_array,
)


EVIDENCE = Path(__file__).with_name("evidence")
MNSQ_BANDS = (CANONICAL_FIT_THRESHOLDS[:2], (.6, 1.4), (.7, 1.3))
ZSTD_LIMITS = (1.5, 2., 2.5, 3.)


def screening_data():
    data = pd.read_csv(EVIDENCE / "m5b_fit_screening.csv.gz")
    common = data.loc[data.condition.eq("common")].set_index("replicate").sort_index()
    changed = data.loc[data.condition.eq("four")].set_index("replicate").sort_index()
    prior = pd.read_csv(EVIDENCE / "m5b_category_drift.csv.gz").query(
        "condition == 'four' and model == 'RSM'").set_index("replicate").sort_index()
    assert common.index.equals(changed.index) and changed.index.equals(prior.index)
    assert common.index.tolist() == list(range(1, 501))
    assert data.FitStatus.eq("available_descriptive").all()
    assert np.isfinite(data[["Infit", "Outfit", "InfitZSTD", "OutfitZSTD"]]).all().all()
    assert prior[["severity_lower", "severity_upper"]].notna().all().all()
    np.testing.assert_allclose(changed.severity_estimate, prior.severity_estimate, rtol=0, atol=1e-5)
    false_drift = prior.severity_lower.gt(0) | prior.severity_upper.lt(0)
    assert false_drift.sum() == 151
    return common, changed, false_drift


def mnsq_flags(frame, lower, upper):
    values = frame[["Infit", "Outfit"]].to_numpy()
    labels = classify_fit_mnsq_array(values.ravel(), (lower, upper, CANONICAL_FIT_THRESHOLDS[2])).reshape(-1, 2)
    return np.any(labels == "overfit", axis=1), np.isin(labels, ("noisy", "distorting")).any(axis=1)


def zstd_flags(frame, limit):
    values = frame[["InfitZSTD", "OutfitZSTD"]]
    return values.le(-limit).any(axis=1).to_numpy(), values.ge(limit).any(axis=1).to_numpy()


def summarize_rule(kind, label, common, changed, false_drift):
    base_lower, base_upper = common
    changed_lower, changed_upper = changed
    base_flag = base_lower | base_upper
    changed_flag = changed_lower | changed_upper
    return dict(metric=kind, threshold=label, common_n=len(base_flag), changed_n=len(changed_flag),
                common_review=int(base_flag.sum()), changed_review=int(changed_flag.sum()),
                common_lower=int(base_lower.sum()), common_upper=int(base_upper.sum()),
                changed_lower=int(changed_lower.sum()), changed_upper=int(changed_upper.sum()),
                wrong_contrasts=int(false_drift.sum()), wrong_reviewed=int((false_drift & changed_flag).sum()),
                wrong_not_reviewed=int((false_drift & ~changed_flag).sum()),
                changed_not_reviewed=int((~changed_flag).sum()))


def draw(table, path):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    fig, ax = plt.subplots(figsize=(7.5, 5.0), layout="constrained")
    specs = {"MnSq": ("#23648a", "o", (-8, 9)), "ZSTD": ("#a84724", "s", (10, -13))}
    for metric, (color, marker, offset) in specs.items():
        rows = table.loc[table.metric.eq(metric)].sort_values("common_review")
        x = rows.common_review/rows.common_n
        y = rows.wrong_reviewed/rows.wrong_contrasts
        ax.plot(x, y, marker=marker, color=color, label="Absolute ZSTD" if metric == "ZSTD" else metric)
        for a, b, label in zip(x, y, rows.threshold, strict=True):
            ax.annotate(label, (a, b), xytext=offset, textcoords="offset points", ha="center",
                        fontsize=9, color=color)
    ax.set(xlim=(0, 1), ylim=(0, 1), title="Sensitivity of fit-screening rules",
           xlabel="Cases flagged when the RSM was correctly specified (500 cases)",
           ylabel="False annual contrasts flagged (151 cases)")
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(alpha=.22)
    ax.legend(loc="lower right", frameon=False)
    fig.savefig(path, dpi=165, facecolor="white")
    plt.close(fig)


def main(directory):
    common, changed, false_drift = screening_data()
    rows = [summarize_rule("MnSq", f"{lo:.2f}–{hi:.2f}", mnsq_flags(common, lo, hi),
                           mnsq_flags(changed, lo, hi), false_drift) for lo, hi in MNSQ_BANDS]
    rows.extend(summarize_rule("ZSTD", f"{limit:.1f}", zstd_flags(common, limit),
                               zstd_flags(changed, limit), false_drift) for limit in ZSTD_LIMITS)
    table = pd.DataFrame(rows)
    assert table.iloc[0][["common_review", "changed_review", "wrong_reviewed"]].tolist() == [56, 208, 68]
    assert table.loc[table.metric.eq("ZSTD") & table.threshold.eq("2.0"),
                     ["common_review", "changed_review", "wrong_reviewed"]].iloc[0].tolist() == [246, 382, 124]
    table.to_csv(directory / "m5b_fit_thresholds.csv", index=False)
    draw(table, directory / "m5b_fit_thresholds.png")
    print(table.to_string(index=False))

    print("Raw vs rounded disagreements for the 0.50–1.50 MnSq band:")
    lo, hi = MNSQ_BANDS[0]
    for decimals in (2, 3):
        for label, frame in (("common", common), ("four", changed)):
            raw = np.logical_or.reduce(mnsq_flags(frame, lo, hi))
            rounded = np.logical_or.reduce(mnsq_flags(frame.round(decimals), lo, hi))
            print(decimals, label, int(np.sum(raw != rounded)))

    print("Calibrate the review threshold on common replicates 1–250, then apply to 251–500:")
    base_score = common[["InfitZSTD", "OutfitZSTD"]].abs().max(axis=1)
    changed_score = changed[["InfitZSTD", "OutfitZSTD"]].abs().max(axis=1)
    cutoff = base_score.iloc[:250].quantile(.95, interpolation="higher")
    reviewed = changed_score.iloc[250:].gt(cutoff)
    wrong = false_drift.iloc[250:]
    print(cutoff, int(base_score.iloc[250:].gt(cutoff).sum()), int(reviewed.sum()),
          int((reviewed & wrong).sum()), int(wrong.sum()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    main(parser.parse_args().directory)
