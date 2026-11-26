"""Regenerate the M2 ratings and run the local ConQuest 5.47.5 command-line comparator."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from scipy.stats import norm

from validation.m2_sparse_intervals import DESIGNS, REPLICATES, TRUTH, rate, simulated_ratings


LABELS = ["item I1", "item I2", "item I3", "rater R1", "rater R2", "category 1", "category 2"]
CONTRAST = np.array([0, 0, 0, -2, -1, 0, 0])
MODEL = """title M2 common-constraint calibration;
set lconstraints=items,seed=2,nodefilter=0,exit_on_error=yes,addextension=no;
datafile ratings.csv ! filetype=csv,responses=Score,pid=Person,keeps=Item Rater,width=1,pidwidth=4,keepswidth=2,header=yes;
codes 0,1,2,3;
model Item + Rater + step;
import anchor_covariance << covariance.txt;
import anchor_beta << mean.txt;
estimate ! method=quadrature,nodes=161,minnode=-8,maxnode=8,fit=no,stderr=empirical,matrixout=checked,convergence=0.000000001,deviancechange=0.00000000001,iterations=2000;
export parameters ! filetype=csv >> parameters.csv;
export history >> history.txt;
print checked_estimatecovariances ! filetype=csv >> errors.csv;
quit;
"""


def generate(destination):
    with destination.open("w") as stream:
        for rep in range(1, REPLICATES + 1):
            for design, data in simulated_ratings(rep, DESIGNS):
                data = data.assign(Person=data.Person.map("P{:03d}".format)).sort_values(["Person", "Item", "Rater"])
                data.assign(replicate=rep, design=design).to_csv(stream, index=False,
                                                                header=rep == 1 and design == next(iter(DESIGNS)))


def estimate(binary, root, key, data):
    rep, design = key
    row = dict(replicate=rep, design=design)
    with TemporaryDirectory(dir=root, prefix=f"{design}-{rep}-") as directory:
        folder = Path(directory)
        data[["Person", "Item", "Rater", "Score"]].to_csv(folder / "ratings.csv", index=False)
        (folder / "covariance.txt").write_text("1 1 1.0\n")
        # This build uses index 1 for the intercept in its anchor import.
        (folder / "mean.txt").write_text("1 1 0.0\n")
        result = subprocess.run([binary], cwd=folder, input=MODEL, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        log = result.stdout
        row["step_warning"] = "Items have different number of steps" in log
        try:
            if result.returncode or "End of Program" not in log or "less than convergence criterion" not in log:
                raise ValueError(f"native calibration status {result.returncode}")
            if "No. of records in file=240" not in log or "No. of cases retained=60" not in log:
                raise ValueError("unexpected retained cases or ratings")
            estimates = pd.read_csv(folder / "parameters.csv")
            errors = pd.read_csv(folder / "errors.csv", index_col=0)
            parameters = estimates.Estimate.to_numpy()
            covariance = errors.to_numpy()
            history = pd.read_csv(folder / "history.txt", sep="\t", header=None).iloc[-1]
            if (estimates.Label.str.strip().tolist() != LABELS or errors.columns.str.strip().tolist() != LABELS
                    or covariance.shape != (7, 7) or not np.isfinite(covariance).all()
                    or not np.allclose(covariance, covariance.T, atol=1e-6, rtol=0)
                    or not np.isfinite(parameters).all() or np.linalg.eigvalsh(covariance).min() <= 0
                    or (history.iloc[3], history.iloc[4]) != (0., 1.)):
                raise ValueError("unexpected parameter/covariance layout or population")
        except (ValueError, OSError) as exc:
            (root / f"failed-{design}-{rep}.log").write_text(f"{exc}\n{log}")
            return row | dict(reason=str(exc))
        center = CONTRAST @ parameters
        se = np.sqrt(CONTRAST @ covariance @ CONTRAST)
        radius = norm.ppf(.975) * se
        return row | dict(estimate=center, se=se, lower=center-radius, upper=center+radius,
                          deviance=history.iloc[2], reason="") | dict(zip(LABELS, parameters, strict=True))


def run(binary, ratings, output, workers):
    data = pd.read_csv(ratings)
    completed = set()
    if output.exists():
        complete = pd.read_csv(output)
        completed = set(complete[["replicate", "design"]].itertuples(index=False, name=None))
    jobs = [(key, sub) for key, sub in data.groupby(["replicate", "design"], sort=False) if key not in completed]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, row in enumerate(pool.map(lambda job: estimate(binary, output.parent, *job), jobs), 1):
            columns = ["replicate", "design", "estimate", "se", "lower", "upper", "deviance", "reason", "step_warning", *LABELS]
            pd.DataFrame([row]).reindex(columns=columns).to_csv(output, mode="a", header=not output.exists(), index=False)
            if i % 100 == 0 or i == len(jobs):
                print(f"{i}/{len(jobs)} ConQuest calibrations", flush=True)


def summarize(external_path, python_path):
    external = pd.read_csv(external_path)
    python = pd.read_csv(python_path).loc[lambda x: x.setting.eq("tight")]
    keys = ["replicate", "design"]
    fields = ["estimate", "lower", "upper"]
    assert not python.duplicated(keys).any() and len(python) == len(DESIGNS) * REPLICATES
    assert not external.duplicated(["engine", *keys]).any()
    print("| 設計 | エンジン | 再構成数（95% Wilson） | 被覆（95% Wilson） | 幅の中央値 |")
    print("|---|---|---:|---:|---:|")
    contrasts = []
    for engine, rows in external.groupby("engine"):
        assert len(rows) == len(python)
        joined = rows.merge(python[keys + fields], on=keys, validate="one_to_one", suffixes=("", "_python"))
        assert len(joined) == len(rows)
        for design in DESIGNS:
            requested = joined.loc[joined.design.eq(design)]
            available = requested.dropna(subset=fields)
            covered = available.lower.le(TRUTH) & available.upper.ge(TRUTH)
            covered_python = available.lower_python.le(TRUTH) & available.upper_python.ge(TRUTH)
            returned, _ = rate(len(available), len(requested))
            coverage, _ = rate(covered.sum(), len(available))
            width = (available.upper - available.lower).median()
            print(f"| {design} | {engine} | {returned} | {coverage} | {width:.3f} |")
            contrasts.append(dict(engine=engine, design=design, verdict_changes=(covered != covered_python).sum(),
                                  **{f"max_abs_{field}": (available[field] - available[field + "_python"]).abs().max()
                                     for field in fields}))
    print(pd.DataFrame(contrasts).to_string(index=False))


def combine(tam_path, conquest_path, destination, retry_path=None):
    conquest = pd.read_csv(conquest_path).assign(first_attempt_reason=None)
    if retry_path:
        keys = ["replicate", "design"]
        retry = pd.read_csv(retry_path)
        failed = conquest.loc[conquest.reason.notna(), keys+["reason"]].rename(columns={"reason": "first_attempt_reason"})
        retry = retry.merge(failed, on=keys, validate="one_to_one")
        assert len(retry) and retry.reason.isna().all()
        recovered = set(retry[keys].itertuples(index=False, name=None))
        keep = [key not in recovered for key in conquest[keys].itertuples(index=False, name=None)]
        conquest = pd.concat([conquest.loc[keep], retry], ignore_index=True)
    tam = pd.read_csv(tam_path)
    combined = pd.concat([tam.assign(engine="TAM"), conquest.assign(engine="ConQuest")], ignore_index=True)
    combined.sort_values(["replicate", "design", "engine"]).to_csv(destination, index=False)


if __name__ == "__main__":
    if sys.argv[1] == "generate":
        generate(Path(sys.argv[2]))
    elif sys.argv[1] == "summarize":
        summarize(Path(sys.argv[2]), Path(sys.argv[3]))
    elif sys.argv[1] == "combine":
        combine(*(Path(arg) for arg in sys.argv[2:]))
    else:
        run(str(Path(sys.argv[1]).resolve()), Path(sys.argv[2]), Path(sys.argv[3]), int(sys.argv[4]))
