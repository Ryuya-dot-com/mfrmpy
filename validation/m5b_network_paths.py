"""Trace an annual rater contrast and verify its sensitivity to missing observations."""
import argparse
from pathlib import Path

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.csgraph import breadth_first_order, maximum_flow

import mfrmpy
from validation.m5b_annual_drift import AUDIT
from validation.m5b_rating_mode import assess, cases_at


RECORDING = "Recording"
RATER = "RaterYearMode"
AFTER, BEFORE = "B-R3-video", "A-R3-video"
FACETS = ["WorkYear", RATER]


def minimum_cut(rows, *, by_work):
    pairs = list(rows[[RECORDING, RATER]].drop_duplicates().itertuples(index=False, name=None))
    raters = {name: i for i, name in enumerate(sorted(rows[RATER].unique()))}
    works = {name: i for i, name in enumerate(sorted(rows[RECORDING].unique()), len(raters))}
    split = len(works) if by_work else 0
    size = len(raters) + len(works) + split
    capacity = lil_matrix((size, size), dtype=np.int64)
    # Split work vertices so removing a recording costs one even if it has several ratings.
    for work, rater in pairs:
        vertex = works[work]
        weight = len(pairs) + 1 if by_work else 1
        capacity[raters[rater], vertex] = weight
        capacity[vertex + split, raters[rater]] = weight
        if by_work:
            capacity[vertex, vertex + split] = 1
    capacity = capacity.tocsr()
    flow = maximum_flow(capacity, raters[BEFORE], raters[AFTER])
    reachable = set(breadth_first_order(
        (capacity - flow.flow) > 0, raters[BEFORE], return_predecessors=False))
    if by_work:
        cut = [label for label, i in works.items() if i in reachable and i + split not in reachable]
        remaining = rows.loc[~rows[RECORDING].isin(cut)]
    else:
        cut = [(w, r) for w, r in pairs if (works[w] in reachable) != (raters[r] in reachable)]
        remaining = rows.set_index([RECORDING, RATER]).drop(cut).reset_index()
    assert len(cut) == flow.flow_value
    return cut, remaining


def draw(path, views):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(11.8, 2.7), layout="constrained")
    fig.suptitle("R3's annual severity contrast under video scoring", fontsize=16, fontweight="bold")
    for ax, (title, status, overlap) in zip(axes, views, strict=True):
        ax.set(xlim=(0, 1), ylim=(0, 1))
        ax.set_axis_off()
        ax.set_title(title, fontsize=11, pad=8, fontweight="bold")
        links = overlap["path"]
        nodes = [links[0]["from_rater"], *(edge["to_rater"] for edge in links)] if links else [AFTER, BEFORE]
        x = np.linspace(.16, .84, len(nodes))
        for i, edge in enumerate(links):
            ax.plot(x[i:i+2], [.55, .55], color="#496a95", linewidth=2, zorder=1)
            count = len(edge["persons"])
            label = f"1 shared work\n{edge['persons'][0]}" if count == 1 else f"{count} shared works"
            ax.text((x[i]+x[i+1])/2, .62, label, ha="center", va="bottom", color="#263b56", fontsize=8)
        for position, node in zip(x, nodes, strict=True):
            hidden = not links and node == BEFORE
            edgecolor = "#758397" if hidden else "#294867"
            fill = "#ffffff" if hidden else ("#dbe4ee" if node not in {AFTER, BEFORE} else "#22577a")
            ax.scatter(position, .55, s=128, facecolor=fill, edgecolor=edgecolor, zorder=2)
            year, rater, mode = node.split("-")
            ax.text(position, .44, f"{rater}\nYear {year}\n{mode}", ha="center", va="top",
                    fontsize=9, color="#18283b")
        ax.text(.5, .10, status, ha="center", color="#294867", fontsize=9)
    fig.savefig(path, dpi=170, facecolor="white")
    plt.close(fig)


def main(path):
    rows = next(rows for case, allocation, rows in cases_at(1)
                if case == "mode_only" and allocation == "includes_R3")
    settings = AUDIT | dict(facets=FACETS, rater=RATER, difference=(AFTER, BEFORE), noncenter_facet=RATER,
                           positive_facets=["WorkYear"], planned_assignments=rows[[RECORDING, *FACETS]])
    full = mfrmpy.check_rater_contrast(rows, **settings)
    direct = full["overlap"][0]
    assert full["status"] == "observed" and direct["kind"] == "direct"
    direct_work, = direct["path"][0]["persons"]

    no_direct = rows.loc[rows[RECORDING].ne(direct_work)]
    alternate = mfrmpy.check_rater_contrast(no_direct, **settings)
    indirect = alternate["overlap"][0]
    assert alternate["status"] == "observed" and indirect["kind"] == "indirect"

    no_baseline = rows.loc[rows[RATER].ne(BEFORE)]
    missing = mfrmpy.check_rater_contrast(no_baseline, **settings)
    assert missing["status"] == "unsupported" and missing["plan"]["planned_observation_estimable"]
    restorations = [entry["assignment"] for entry in missing["plan"]["missing"]
                    if entry["restores_contrast"]]
    assert len(restorations) == len(rows) - len(no_baseline) == 12

    assert rows.groupby(RECORDING)["WorkYear"].nunique().eq(1).all()
    for label, data, expected_cut in (("All observations", rows, 12), (f"Without {direct_work}", no_direct, 11)):
        print(label, len(data), "ratings and", data[RECORDING].nunique(), "works")
        for by_work in (True, False):
            cut, remaining = minimum_cut(data, by_work=by_work)
            assert len(cut) == expected_cut
            check = mfrmpy.check_rater_contrast(remaining, **settings)
            assert check["observation_estimable"] is False
            print("  Minimum", "work" if by_work else "rating", "cut:", len(cut), sorted(cut))
        fit = assess(data, "separate_cells")
        assert not fit["reason"]
        print("  Same-mode interval:", fit["lower"], fit["upper"])
    print("Alternate route:", indirect["path"])
    print("Restoring planned ratings:", [item[RECORDING] for item in restorations])
    draw(path, [
        ("All observations", "Direct comparison supported", direct),
        (f"Remove work {direct_work}", "Indirect comparison supported", indirect),
        ("Remove all 12 Year A R3 video ratings", "Year A target has no observed ratings", missing["overlap"][0]),
    ])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("figure", type=Path)
    main(parser.parse_args().figure)
