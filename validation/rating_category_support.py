"""Probe per-person score patterns, actual omissions, and empty step categories."""
import argparse
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from scipy.special import softmax


PEOPLE = {"A333": [3, 3, 3, 3], "B124": [1, 2, 4, 5], "CMIS": [3, 3, 3, None]}
ITEMS = ["I1", "I2"]
RATERS = ["R1", "R2", "R3", "R4"]


def generate(path):
    rng = np.random.default_rng(2026112204)
    eta = (rng.normal(size=120)[:, None, None] - np.array([-.35, .35])[None, :, None]
           - np.array([-.45, -.15, .15, .45])[None, None, :])
    probabilities = softmax(eta[..., None]*np.arange(5) - np.cumsum([0, -1.2, -.4, .4, 1.2]), axis=-1)
    scores = 1 + (rng.random(eta.shape)[..., None] > probabilities.cumsum(axis=-1)[..., :-1]).sum(axis=-1)
    data = pd.MultiIndex.from_product([[f"P{i:03d}" for i in range(120)], ITEMS, RATERS],
        names=["Person", "Item", "Rater"]).to_frame(index=False).assign(Score=scores.ravel())
    assert data.groupby(["Item", "Rater"]).Score.nunique().eq(5).all()
    samples = pd.DataFrame([(person, "I1", rater, score) for person, pattern in PEOPLE.items()
                            for rater, score in zip(RATERS, pattern, strict=True)], columns=data.columns)
    data = pd.concat([data, samples], ignore_index=True)
    other = np.where(data.Rater.isin(["R1", "R3"]) ^ (np.arange(len(data)) % 3 == 0), 2, 4)
    sets = {"base": data, "rater_gap": data.assign(Score=data.Score.mask(data.Rater.eq("R4") & data.Score.eq(3), other)),
            "global_gap": data.assign(Score=data.Score.mask(data.Score.eq(3), other))}
    path.mkdir(exist_ok=True, parents=True)
    for label, rows in sets.items():
        rows.to_csv(path / f"{label}.csv", index=False)


def fit_mfrmpy(path):
    import mfrmpy
    from mfrm_app.output_qualification import audit_boundary_and_support

    summaries = []
    cases = (("base", None, 31), ("base", "Rater", 31), ("rater_gap", None, 31), ("rater_gap", "Item", 31),
             ("rater_gap", "Rater", 31), ("global_gap", None, 31), ("rater_gap", None, 61), ("rater_gap", "Item", 61))
    for label, owner, quadrature in cases:
        data = pd.read_csv(path / f"{label}.csv")
        fit = mfrmpy.analyze(data, person="Person", facets=["Item", "Rater"], score="Score", model="PCM" if owner else "RSM",
            step_facet=owner, method="MML", rating_min=1, rating_max=5, reltol=1e-10, quad_points=quadrature,
            interval_requests=[dict(facet="Rater", difference=("R4", "R1"))])
        result = fit["result"]
        entry, = fit["interval_analyses"]
        audit = audit_boundary_and_support(result)
        count = len(result["prep"]["data"])
        assert count == int(data.Score.notna().sum())
        interval = next((x for x in entry["record"]["table"] if x["Kind"] == "difference"), None) if entry["record"] else None
        summaries.append(dict(engine="mfrmpy", variant=label, model=f"PCM:{owner}" if owner else "RSM", ratings=count,
            quadrature=quadrature,
            interval=interval["CIEligible"] if interval else False, reason=interval["Reason"] if interval else entry["reason"],
            unsupported_step=audit["unsupported_internal"]))
        if label == "base" and not owner:
            persons = result["facets"]["person"].set_index("Person").loc[list(PEOPLE), ["Estimate", "SD"]]
            assert np.isfinite(persons.Estimate).all() and abs(persons.Estimate.iloc[0]-persons.Estimate.iloc[1]) < 1e-12
            d = fit["diagnostics"]["measures"]
            fit_columns = [k for k in ("Infit", "Outfit") if k in d]
            if fit_columns:
                persons = persons.join(d.loc[d.Facet.eq("Person")].set_index("Level")[fit_columns])
            persons.to_csv(path / "mfrmpy_people.csv")
    pd.DataFrame(summaries).to_csv(path / "mfrmpy_cases.csv", index=False)


def fit_raschpy(path):
    from importlib.metadata import version
    from raschpy import MFRM

    assert version("raschpy") == "1.4.0"
    summaries = []
    for label in ("base", "rater_gap", "global_gap"):
        data = pd.read_csv(path / f"{label}.csv")
        wide = data.pivot(index=["Rater", "Person"], columns="Item", values="Score") - 1
        with warnings.catch_warnings(record=True) as notes:
            warnings.simplefilter("always")
            fit = MFRM(wide, max_score=4)
            fit.calibrate(model="global", estimation="cp-full-rcl")
            if label == "base":
                people = fit.person(list(PEOPLE), warm_corr=False)
                assert np.isfinite(people).all() and abs(people["A333"] - people["B124"]) < 1e-12
                pd.Series(people, name="Estimate").to_csv(path / "raschpy_people.csv")
            valid = fit.responses
            assert all(p in valid.index.get_level_values(1) for p in PEOPLE)
            assert int(valid.notna().sum().sum()) == int(data.Score.notna().sum())
        status = fit.convergence_global
        summaries.append(dict(engine="RaschPy", variant=label, model=fit.estimation_global, ratings=int(valid.notna().sum().sum()),
            converged=status["converged"], steps=len(fit.thresholds_global), warning=" | ".join(str(w.message) for w in notes)))
    pd.DataFrame(summaries).to_csv(path / "raschpy_cases.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("mode", choices=["generate", "mfrmpy", "raschpy"])
    args = parser.parse_args()
    {"generate": generate, "mfrmpy": fit_mfrmpy, "raschpy": fit_raschpy}[args.mode](args.path)
