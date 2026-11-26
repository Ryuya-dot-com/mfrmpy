"""Remove each fixed rater anchor or retain only its group mean in the 0.2.1 pilot."""
from pathlib import Path
import sys
from zipfile import ZipFile

import numpy as np
import pandas as pd

import mfrmpy


def run(archive):
    settings = dict(person="Person", facets=["Item", "Rater"], score="Score")
    r1 = dict(Facet="Rater", Level="R1", Anchor=-.6)
    r2 = dict(Facet="Rater", Level="R2", Anchor=0.)
    group = [dict(Facet="Rater", Level=r, Group="R1_R2", GroupValue=-.3) for r in ("R1", "R2")]
    rows = []
    with ZipFile(archive) as evidence:
        with evidence.open("anchors.csv") as file:
            previous = pd.read_csv(file)
        for rep in previous.rep.unique():
            with evidence.open(f"data/shared6/{rep}/ratings.csv") as file:
                data = pd.read_csv(file)
            reference = previous.loc[previous.rep.eq(rep) & previous.fixed_R1.eq(-.6), "R3_minus_R1"].item()
            for label, anchors, groups in [("drop_R1", [r2], []), ("drop_R2", [r1], []), ("mean_R1_R2", [], group)]:
                audit = mfrmpy.check_rater_contrast(data, **settings, rater="Rater", difference=("R3", "R1"),
                                                   anchors=anchors, group_anchors=groups)
                result = mfrmpy.analyze(data, **settings, rating_min=0, rating_max=3, method="MML",
                                        anchors=anchors, group_anchors=groups)["result"]
                expected = result["config"]["facet_levels"]["Rater"]
                raters = dict(zip(expected, result["params"]["facets"]["Rater"], strict=True))
                for anchor in anchors:
                    assert np.isclose(raters[anchor["Level"]], anchor["Anchor"], atol=1e-12)
                if groups:
                    assert np.isclose((raters["R1"] + raters["R2"]) / 2, -.3, atol=1e-12)
                contrast = raters["R3"] - raters["R1"]
                rows.append(dict(rep=rep, condition=label, support=audit["status"],
                                 with_anchors=audit["estimable_with_anchors"],
                                 terminated=bool(result["summary"]["Converged"].iat[0]),
                                 estimate=contrast, both_fixed=reference, change=contrast-reference))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    evidence = Path(__file__).parent / "evidence" / "mfrmpy-0.2.1-evidence.zip"
    run(evidence).to_csv(sys.argv[1], index=False)
