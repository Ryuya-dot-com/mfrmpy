"""Compare balanced and focal rater assignments at a fixed rating budget."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

import mfrmpy
from validation.m2_sparse_intervals import PAIR, REPLICATES
from validation.m5a_group_dff import AUDIT, COLUMNS, EFFECTS, TARGETS, ratings_at, review


FOCAL = {"direct16": (4, 16, 4), "direct20": (2, 20, 2)}
DESIGNS = {**TARGETS, **FOCAL}


def run_replicate(rep):
    outputs = []
    for design, effect, data in ratings_at(rep, DESIGNS):
        if design not in FOCAL:
            continue
        assert data.groupby(["School", "Rater"]).size().eq(80).all()
        per_work = data.groupby(["School", "Person"]).Rater.agg(lambda x: tuple(sorted(set(x))))
        counts = per_work.groupby(level="School").value_counts()
        for school in ("A", "B"):
            pairs = [tuple(f"R{i+1}" for i in pair) for pair in PAIR]
            assert tuple(counts.loc[school][pair] for pair in pairs) == FOCAL[design]
        audit = mfrmpy.check_rater_contrast(data, **AUDIT)
        overlap = {r["group_level"]: r["kind"] for r in audit["overlap"]}
        assert audit["status"] == "observed" and overlap == {"A": "direct", "B": "direct"}
        outputs.append(dict(replicate=rep, design=design, effect=effect, truth=EFFECTS[effect],
            support=audit["status"], overlap_A=overlap["A"], overlap_B=overlap["B"], **review(data, audit)))
    return outputs


def run(path, workers, end):
    complete = set()
    if path.exists():
        saved = pd.read_csv(path)
        assert tuple(saved.columns) == COLUMNS and saved.groupby("replicate").size().eq(4).all()
        complete.update(saved.replicate)
    pending = [rep for rep in range(1, end+1) if rep not in complete]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, rows in enumerate(pool.map(run_replicate, pending), 1):
            pd.DataFrame(rows).reindex(columns=COLUMNS).to_csv(path, mode="a", header=not path.exists(), index=False)
            if i % 25 == 0 or i == len(pending):
                print(f"{i}/{len(pending)} replicate sets", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--end", type=int, default=REPLICATES)
    args = parser.parse_args()
    run(args.out, args.workers, args.end)
