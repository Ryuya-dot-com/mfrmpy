"""Balanced normal-model verification, using exact independent ANOVA MS draws.

Run with PYTHONPATH=. Optional --r-library points to mfrmr 0.3.2.
This is not a simulation of ordinal scores, missingness or misspecification.
"""
import argparse
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

import mfrmpy
from mfrm_app import gd_intervals


def ratings(shape):
    rng = np.random.default_rng(15)
    y = np.zeros(shape)
    for mask in itertools.product([False, True], repeat=len(shape)):
        if any(mask):
            y += rng.normal(size=tuple(n if v else 1 for n, v in zip(shape, mask)))
    return pd.DataFrame(list(np.ndindex(shape)), columns=["Person", "Rater", "Task"][:len(shape)]).assign(Score=y.ravel())


def wilson(success, total):
    if not total:
        return [None, None]
    z = 1.959963984540054
    p = success / total
    center = (p + z*z/(2*total))/(1+z*z/total)
    half = z*np.sqrt(p*(1-p)/total + z*z/(4*total*total))/(1+z*z/total)
    return [center-half, center+half]


def simulation(shape, repetitions, seed):
    data = ratings(shape)
    facets = list(data.columns[1:-1])
    result = mfrmpy.analyze(data, workflow="gd", gd_settings=dict(
        object_facet="Person", random_facets=facets, n_max_per_facet=1))["gd_analysis"]["result"]
    vc, design = result["variance_components"].copy(), result["design"]
    terms, _ = gd_intervals.component_system(vc, design)
    levels = design["levels"]
    expected = np.array([[np.prod([n for f,n in levels.items() if f not in t]) if s <= t else 0.
                          for t in terms] for s in terms])
    inverse = np.linalg.inv(expected)
    # Every true variance component is one, including the highest-order residual.
    df = vc.df.to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    mean_squares = rng.chisquare(df, size=(repetitions,len(df))) / df * expected.sum(axis=1)
    estimates = mean_squares @ inverse.T
    records = []
    for i, raw in enumerate(estimates):
        vc["RawVariance"] = raw
        vc["Variance"] = np.maximum(0, raw)
        _, covariance = gd_intervals.component_system(vc, design)
        for plan, multiplier in (("observed",1), ("double_facets",2)):
            counts = {f: n*multiplier for f,n in design["observed_n"].items()}
            values = gd_intervals.coefficient_row(vc, design, terms, covariance, counts, .95)
            weights = np.array([1/np.prod([counts[f] for f in t if f != "Person"]) for t in terms])
            weights[0] = 0
            for name, error in (("G",weights @ np.array(["Person" in t for t in terms])), ("Phi", weights.sum())):
                truth = 1/(1+error)
                lo, hi = values[name+"_lower"], values[name+"_upper"]
                returned = bool(np.isfinite([lo,hi]).all())
                records.append(dict(shape="x".join(map(str,shape)), replicate=i, plan=plan, coefficient=name,
                    truth=truth, returned=returned, covered=bool(returned and lo<=truth<=hi),
                    lower=lo, upper=hi, status=values[name+"_interval_status"]))
    frame = pd.DataFrame(records)
    summaries = []
    for (plan,name), rows in frame.groupby(["plan","coefficient"]):
        returned, covered = int(rows.returned.sum()), int(rows.covered.sum())
        coverage_interval = wilson(covered, returned)
        return_interval = wilson(returned, repetitions)
        summaries.append(dict(shape=list(shape), plan=plan, coefficient=name, attempted=repetitions,
            returned=returned, covered=covered, return_wilson=return_interval, coverage_wilson=coverage_interval,
            conditional_coverage=covered/returned if returned else None,
            median_width=float((rows.upper-rows.lower).median()), statuses=dict(Counter(rows.status)),
            criterion_met=bool(return_interval[0]>=.8 and coverage_interval[0]>=.92 and coverage_interval[1]<=.98)))
    return frame, summaries


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--repetitions",type=int,default=3000)
    parser.add_argument("--r-library")
    args=parser.parse_args()
    output=Path("validation/evidence")
    output.mkdir(exist_ok=True)
    reference=[]
    if args.r_library:
        python_results={}
        for name,shape in (("one",(12,6)),("two",(12,6,5))):
            data=ratings(shape)
            data.to_csv(output/f"gd_interval_{name}.csv",index=False)
            a=mfrmpy.analyze(data,workflow="gd",gd_settings=dict(object_facet="Person",
                random_facets=list(data.columns[1:-1]),coefficient_intervals=dict(method="normal_delta",assumption="normal")))
            python_results[name]=a["gd_analysis"]["result"]
            mfrmpy.save_project(output/f"gd_interval_{name}.mfrm",a)
        subprocess.run(["Rscript","validation/gd_coefficient_intervals.R",args.r_library,str(output)],check=True)
        for r in json.loads((output/"gd_interval_mfrmr.json").read_text()):
            p=python_results[r["name"]]
            names=["Residual" if name.endswith(",e") else name for name in r["covariance_names"]]
            pcov=p["coefficient_intervals"]["covariance"].loc[names,names].to_numpy()
            np.testing.assert_allclose(pcov,r["covariance"],rtol=1e-9,atol=1e-9)
            bounds=np.array([[p["coefficient_intervals"]["observed"][k+"_"+s] for s in ("lower","upper")] for k in ("G","Phi")])
            np.testing.assert_allclose(bounds,r["intervals"],rtol=1e-9,atol=1e-9)
            reference.append(dict(design=r["name"],max_covariance_difference=float(np.max(np.abs(pcov-r["covariance"]))),
                                  max_interval_difference=float(np.max(np.abs(bounds-r["intervals"])))))
    frames=[]; summaries=[]
    for i,shape in enumerate(((60,12),(20,3),(80,12,10),(20,3,2))):
        frame, summary=simulation(shape,args.repetitions,20261126+i)
        frames.append(frame); summaries.extend(summary)
        print(json.dumps(summary),flush=True)
    pd.concat(frames,ignore_index=True).to_csv(output/"gd_interval_replicates.csv.gz",index=False)
    sources=[Path(p) for p in ("mfrm_app/gd_intervals.py","mfrm_app/generalizability.py",__file__)]
    evidence=dict(python=sys.version,seed=20261126,repetitions=args.repetitions,
        basis="Exact independent chi-square mean squares under balanced independent Gaussian random effects; all true variance components one",
        prespecified_criterion="95% Wilson return lower bound >= .80; coverage interval wholly within .92-.98",
        r_reference=reference,simulations=summaries,
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    (output/"gd_coefficient_intervals.json").write_text(json.dumps(evidence,indent=2)+"\n")


if __name__=="__main__":
    main()
