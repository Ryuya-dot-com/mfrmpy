"""Compute M4 contrasts from ConQuest's empirical covariance on the common scale."""
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from validation.m2_external_conquest import LABELS, MODEL
from validation.m4_rater_task import FOCAL, LEVELS


PARAMETERS = [*LABELS, *(f"item I{i} rater R{r}" for r in (1, 2) for i in (1, 2, 3))]
COMMAND = MODEL.replace("model Item + Rater + step;", "model Item + Rater + step + Item*Rater;")
ITEM_BASIS = np.r_[np.eye(3), -np.ones((1, 3))]
RATER_BASIS = np.r_[np.eye(2), -np.ones((1, 2))]


def fit(binary, parent, ratings):
    with TemporaryDirectory(dir=parent) as tmp:
        folder = Path(tmp)
        ratings[["Person", "Item", "Rater", "Score"]].to_csv(folder / "ratings.csv", index=False)
        (folder / "mean.txt").write_text("1 1 0.0\n")
        (folder / "covariance.txt").write_text("1 1 1.0\n")
        result = subprocess.run([binary], cwd=folder, input=COMMAND, text=True, stderr=subprocess.STDOUT,
                                stdout=subprocess.PIPE)
        assert result.returncode == 0 and "End of Program" in result.stdout, (result.returncode, result.stdout)
        for marker in ("less than convergence criterion", "No. of records in file=240", "No. of cases retained=60"):
            assert marker in result.stdout, result.stdout
        point = pd.read_csv(folder / "parameters.csv")
        errors = pd.read_csv(folder / "errors.csv", index_col=0)
        v = errors.to_numpy()
        h = pd.read_csv(folder / "history.txt", sep="\t", header=None).iloc[-1]
        assert (h.iloc[3], h.iloc[4]) == (0., 1.)
        assert point.Label.str.strip().tolist() == errors.columns.str.strip().tolist() == PARAMETERS
        assert np.isfinite(v).all() and np.linalg.eigvalsh(v).min() > 0

        # ConQuest's interaction increases difficulty. Python's increases eta.
        L = -np.array([np.kron(RATER_BASIS[int(r[1])-1], ITEM_BASIS[int(i[1])-1]) for r, i in LEVELS])
        L = np.r_[L, [L[LEVELS.index(FOCAL[0])] - L[LEVELS.index(FOCAL[1])]]]
        estimates = L @ point.Estimate.to_numpy()[7:]
        errors = np.sqrt(np.diag(L @ v[7:, 7:] @ L.T))
        return pd.DataFrame(dict(engine="ConQuest", rater=[r for r, _ in LEVELS]+[""],
            item=[i for _, i in LEVELS]+[""], kind=["cell"]*12+["contrast"], estimate=estimates, se=errors,
            nll=h.iloc[2]/2))


if __name__ == "__main__":
    executable, ratings, destination = sys.argv[1:]
    path = Path(destination)
    assert not path.exists()
    fields = ["replicate", "design", "effect"]
    for values, data in pd.read_csv(ratings, keep_default_na=False).groupby(fields):
        fit(executable, path.parent, data).assign(**dict(zip(fields, values, strict=True))).to_csv(
            path, index=False, mode="a", header=not path.exists())
