"""Compare the two public interaction workflows under null and planted effects."""
import argparse
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

import mfrmpy
from validation.m2_sparse_intervals import DESIGNS, REPLICATES, SEED, potential_scores, simulated_ratings


TARGETS = {name: DESIGNS[name] for name in ("balanced6", "balanced24")}
LEVELS = [(rater, item) for item in ("I1", "I2", "I3", "I4") for rater in ("R1", "R2", "R3")]
SHIFT = .5 * np.outer([1, -1, 0, 0], [-1, 0, 1])
FOCAL = ("R3", "I1"), ("R1", "I1")
FIT = dict(person="Person", facets=["Item", "Rater"], score="Score", model="RSM", method="MML",
           rating_min=0, rating_max=3, quad_points=31, reltol=1e-10,
           bias_requests=[dict(facet_a="Rater", facet_b="Item")],
           joint_settings=dict(facet_a="Rater", facet_b="Item", difference=FOCAL))
COLUMNS = ("replicate", "effect", "design", "kind", "rater", "item", "truth", "ratings", "screen_estimate",
           "screen_se", "screen_lr_p", "screen_lr", "screen_use", "screen_reason", "joint_estimate", "joint_se",
           "joint_lower", "joint_upper", "joint_p", "joint_holm_p", "joint_reason")


def ratings_at(replicate):
    stream = np.random.SeedSequence([SEED, replicate]).spawn(1)[0]
    rng = np.random.default_rng(stream)
    planted_scores = potential_scores(rng.normal(size=60), rng, SHIFT)
    for name, rows in simulated_ratings(replicate, TARGETS):
        y = planted_scores[rows.Person, rows.Item.str[1:].astype(int)-1, rows.Rater.str[1:].astype(int)-1]
        data = rows.assign(Person=rows.Person.map("P{:03d}".format))
        yield name, "zero", data
        yield name, "planted", data.assign(Score=y)


def extract(analysis, effect, rows):
    info, = analysis["bias_analyses"]
    raw = info["result"].get("table", pd.DataFrame())
    screening = {}
    if not raw.empty:
        _, display, _ = next(entry for entry in mfrmpy.tables(analysis, kind="bias") if entry[0].startswith("Bias:"))
        statuses = display.set_index(["Level A", "Level B"])
        for r in raw.to_dict("records"):
            key = (r["FacetA_Level"], r["FacetB_Level"])
            published = statuses.loc[key]
            assert not bool(published.SupportsFormalInference)
            screening[key] = dict(screen_estimate=r["Bias Size"], screen_se=r["S.E."], screen_lr_p=r["LR Prob."],
                                  screen_lr=r["LR ChiSq"], screen_use=published.Use,
                                  screen_reason=published.StatusReason)
    joint = analysis["joint_analysis"]
    cells = joint["record"]["table"] if joint["status"] == "computed" else []
    if cells:
        assert len(cells) == 13 and all(r["FamilySize"] == 13 for r in cells)
    lookup = {(r["LevelA"], r["LevelB"]) if r["Kind"] == "cell" else None: r for r in cells}
    for cell in [*LEVELS, None]:
        if cell is None:
            truth, rater, item, n = 1. if effect == "planted" else 0., "", "", None
            one, two = [screening.get(label) for label in FOCAL]
            if one and two and one["screen_use"] == two["screen_use"] == "screening_only":
                conditional = dict(screen_estimate=one["screen_estimate"] - two["screen_estimate"],
                                   screen_use="diagnostic_difference")
            else:
                conditional = dict(screen_use="withheld", screen_reason="At least one diagnostic cell unavailable")
        else:
            rater, item = cell
            n = int((rows.Rater.eq(rater) & rows.Item.eq(item)).sum())
            truth = SHIFT[int(item[1])-1, int(rater[1])-1] if effect == "planted" else 0.
            conditional = screening.get(cell, dict(screen_use="withheld",
                screen_reason=info["result"].get("_skip_reason", "Cell absent")))
        r = lookup.get(cell)
        if r is None:
            fitted = dict(joint_reason=joint["reason"])
        else:
            fitted = dict(joint_estimate=r["Estimate"], joint_se=r["SE"], joint_lower=r["Lower"],
                          joint_upper=r["Upper"], joint_p=r["WaldP"], joint_holm_p=r["HolmP"], joint_reason=r["Reason"])
        yield dict(kind="contrast" if cell is None else "cell", rater=rater, item=item, truth=truth,
                   ratings=n, **conditional, **fitted)


def run_replicate(rep):
    out = []
    for design, effect, rows in ratings_at(rep):
        analysis = mfrmpy.analyze(rows, **FIT)
        out.extend(dict(replicate=rep, effect=effect, design=design, **row) for row in extract(analysis, effect, rows))
    return out


def run(path, workers, end):
    complete = set()
    if path.exists():
        saved = pd.read_csv(path)
        assert tuple(saved.columns) == COLUMNS and saved.groupby("replicate").size().eq(52).all()
        complete.update(saved.replicate)
    pending = [rep for rep in range(1, end+1) if rep not in complete]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, rows in enumerate(pool.map(run_replicate, pending), 1):
            pd.DataFrame(rows).reindex(columns=COLUMNS).to_csv(path, mode="a", header=not path.exists(), index=False)
            if i % 25 == 0 or i == len(pending):
                print(f"{i}/{len(pending)} replicate sets", flush=True)


if __name__ == "__main__":
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--end", type=int, default=REPLICATES)
    args = parser.parse_args()
    run(args.out, args.workers, args.end)
