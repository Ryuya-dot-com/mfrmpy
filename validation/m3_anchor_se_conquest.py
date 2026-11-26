"""Use ConQuest's covariance and anchored refits to check both M3 SE components."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from validation.m2_external_conquest import LABELS, MODEL


TARGET_LABELS = [*LABELS[:3], "item I4", "flag2 0", "flag3 0", *LABELS[-2:]]
TARGET_MODEL = (MODEL.replace("lconstraints=items", "lconstraints=cases")
                .replace("keeps=Item Rater", "keeps=Item Flag2 Flag3")
                .replace("model Item + Rater + step;", "model Item + Flag2 + Flag3 + step;")
                .replace("import anchor_beta << mean.txt;", "import anchor_parameters << anchor.txt;"))


def fit(binary, directory, ratings, difference=None):
    with TemporaryDirectory(dir=directory) as tmp:
        folder = Path(tmp)
        columns = ["Person", "Item", "Score", "Rater"]
        if difference is not None:
            ratings = ratings.assign(Flag2=ratings.Rater.eq("R2").astype(int),
                                     Flag3=ratings.Rater.eq("R3").astype(int))
            columns += ["Flag2", "Flag3"]
            (folder / "anchor.txt").write_text(f"5 {-difference / 2:.17g}\n")
        ratings[columns].to_csv(folder / "ratings.csv", index=False)
        (folder / "covariance.txt").write_text("1 1 1.0\n")
        (folder / "mean.txt").write_text("1 1 0.0\n")
        result = subprocess.run([binary], cwd=folder, input=MODEL if difference is None else TARGET_MODEL,
                                text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        log = result.stdout
        if (result.returncode or "End of Program" not in log or "less than convergence criterion" not in log
                or f"No. of records in file={len(ratings)}" not in log or "No. of cases retained=60" not in log):
            raise ValueError(f"ConQuest status {result.returncode}: {log[-1800:]}")
        estimates = pd.read_csv(folder / "parameters.csv")
        values = estimates.Estimate.to_numpy()
        errors = pd.read_csv(folder / "errors.csv", index_col=0)
        cov = errors.to_numpy()
        expected = LABELS if difference is None else TARGET_LABELS
        free = [label for label in expected if label != "flag2 0"]
        history = pd.read_csv(folder / "history.txt", sep="\t", header=None).iloc[-1]
        if (estimates.Label.str.strip().tolist() != expected or errors.columns.str.strip().tolist() != free
                or cov.shape != (7, 7) or not np.isfinite(values).all() or not np.isfinite(cov).all()
                or not np.allclose(cov, cov.T, atol=1e-6, rtol=0) or np.linalg.eigvalsh(cov).min() <= 0
                or (history.iloc[3], history.iloc[4]) != (0., 1.)
                or (difference is not None and abs(values[4] + difference/2) > 5.1e-7)):
            raise ValueError("Unexpected parameter/covariance layout, constraint, or population")
        return values, cov


def run(binary, reference, calibration, ratings, output, workers):
    reference = pd.read_csv(reference)
    source = list(pd.read_csv(calibration).groupby("replicate"))
    target = pd.read_csv(ratings).groupby(["replicate", "design"])
    if output.exists():
        raise FileExistsError(output)

    def receiving(rep, ref, values, v):
        anchor_diff_se = np.sqrt(v[0, 0] - 2*v[0, 1] + v[1, 1])
        data = target.get_group((rep, ref.design))
        delta = ref.a2 - ref.a1
        point, info = fit(binary, output.parent, data, delta)
        d = []
        for step in (.04, .02):
            minus, _ = fit(binary, output.parent, data, delta - step)
            plus, _ = fit(binary, output.parent, data, delta + step)
            d.append(-(plus[5] - minus[5]) / step)
        jac = (4*d[1] - d[0]) / 3
        if abs(d[1] - d[0]) > 5e-4:
            raise ValueError(f"Anchor perturbation sensitivities disagree: {d}")
        conditional = 2*np.sqrt(info[4, 4])
        anchor = abs(jac)*anchor_diff_se
        return dict(a1=values[3], a2=values[4], v11=v[0, 0], v12=v[0, 1], v22=v[1, 1],
                    estimate=-2*point[5], conditional_se=conditional, j2=jac, anchor_se=anchor,
                    propagated_se=np.hypot(conditional, anchor), perturbation_gap=abs(d[1] - d[0]), reason="")

    def job(entry):
        rep, d_source = entry
        status = ""
        try:
            values, covariance = fit(binary, output.parent, d_source)
        except (ValueError, OSError) as exc:
            status = f"Source: {exc}"
        for ref in reference[reference.replicate.eq(rep)].itertuples():
            row = dict(engine="ConQuest", replicate=rep, design=ref.design)
            try:
                result = dict(reason=status) if status else receiving(rep, ref, values, covariance[3:5, 3:5])
            except (ValueError, OSError) as exc:
                result = dict(reason=str(exc))
            yield row | result

    fields = ["engine", "replicate", "design", "a1", "a2", "v11", "v12", "v22", "estimate", "conditional_se",
              "j2", "anchor_se", "propagated_se", "perturbation_gap", "reason"]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for rows in pool.map(lambda entry: list(job(entry)), source):
            pd.DataFrame(rows).reindex(columns=fields).to_csv(output, mode="a", header=not output.exists(), index=False)


if __name__ == "__main__":
    binary, reference, calibration, ratings, output, workers = sys.argv[1:]
    run(binary, reference, calibration, ratings, Path(output), int(workers))
