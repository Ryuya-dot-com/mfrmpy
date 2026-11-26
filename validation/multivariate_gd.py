"""Compare balanced covariance components with mfrmr 0.3.2 ANOVA/Henderson.

PYTHONPATH=. python3 validation/multivariate_gd.py --r-library <mfrmr-0.3.2-library>
This validates point calculations, not interval coverage or selected weights.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import platform
import subprocess

import numpy as np
import pandas as pd

import mfrmpy


def ratings(shape):
    rng = np.random.default_rng(24)
    y = np.zeros((*shape, 3))
    covariance = np.array([[1., .6, -.2], [.6, 1.5, .1], [-.2, .1, .8]])
    for mask in itertools.product([False, True], repeat=len(shape)):
        if any(mask):
            y += rng.multivariate_normal(np.zeros(3), covariance,
                                         size=tuple(n if v else 1 for n,v in zip(shape, mask)))
    data = pd.DataFrame(list(np.ndindex(shape)), columns=["Person", "Rater", "Task"][:len(shape)])
    for i,name in enumerate(("Content", "Language", "Style")):
        data[name] = y[..., i].ravel()
    return data


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--r-library",required=True)
    args=parser.parse_args()
    output=Path("validation/evidence"); output.mkdir(exist_ok=True)
    computed={}
    for name,shape in (("one",(12,6)),("two",(12,6,5))):
        data=ratings(shape); data.to_csv(output/f"multivariate_gd_{name}.csv",index=False)
        facets=[f for f in ("Rater","Task") if f in data]
        a=mfrmpy.analyze(data, workflow="gd",gd_settings=dict(object_facet="Person",random_facets=facets,
            score_cols=["Content","Language","Style"],composite_weights={"Content":.2,"Language":.5,"Style":.3},
            selected_plan={f:13 for f in facets},n_max_per_facet=2))
        mfrmpy.save_project(output/f"multivariate_gd_{name}.mfrm",a)
        computed[name]=a["gd_analysis"]["result"]
    subprocess.run(["Rscript","validation/multivariate_gd.R",args.r_library,str(output)],check=True)
    differences=[]
    for r in json.loads((output/"multivariate_gd_mfrmr.json").read_text()):
        p=computed[r["name"]]; multi=p["multivariate"]
        errors=[]
        for source,matrix in multi["covariance_components"].items():
            reference=np.asarray(r["components"][source])
            np.testing.assert_allclose(matrix,reference,atol=1e-10,rtol=1e-10)
            errors.append(float(np.max(np.abs(matrix.to_numpy()-reference))))
        np.testing.assert_array_equal(multi["component_diagnostics"].PositiveSemidefinite,
                                      [row["PositiveSemidefinite"] for row in r["component_diagnostics"]])
        component_reference=np.array([row["RawEstimate"] for row in r["composite_components"]])
        np.testing.assert_allclose(p["variance_components"].RawVariance,component_reference,atol=1e-9,rtol=1e-9)
        point_errors=[]
        names=dict(UniverseVar="UniverseVariance",RelErrorVar="RelativeErrorVariance",AbsErrorVar="AbsoluteErrorVariance",
                   RelSEM="RelativeSEM",AbsSEM="AbsoluteSEM",G="G",Phi="Phi")
        for reference in r["plans"]:
            subset=p["d_study"]
            for f in p["design"]["random_facets"]:subset=subset.loc[subset[f].eq(reference[f])]
            row=subset.iloc[0]
            for key,field in names.items():
                np.testing.assert_allclose(row[field],reference[key],atol=1e-9,rtol=1e-9)
                point_errors.append(abs(float(row[field])-reference[key]))
        for reference in r["univariate"]:
            row=multi["observed"].loc[lambda t:t.TargetType.eq("score") & t.Target.eq(reference["Score"])].iloc[0]
            for key,field in names.items():
                np.testing.assert_allclose(row[field],reference[key],atol=1e-9,rtol=1e-9)
                point_errors.append(abs(float(row[field])-reference[key]))
        differences.append(dict(design=r["name"],score_count=3,ratings=p["design"]["n_observations"],
            covariance_max_difference=max(errors),composite_component_max_difference=float(np.max(np.abs(p["variance_components"].RawVariance-component_reference))),
            metric_max_difference=max(point_errors),non_psd_components=multi["component_diagnostics"].loc[lambda t:~t.PositiveSemidefinite,"Source"].tolist()))
    paths=[Path(p) for p in ("mfrm_app/multivariate_gd.py","validation/multivariate_gd.py","validation/multivariate_gd.R")]
    evidence=dict(python=platform.python_version(),platform=platform.platform(),mfrmr="0.3.2",
        mfrmr_archive_sha256="611ac538da670a58bd9a245d97fcc202b02524c5dc3dbcf03c071b48be68867e",
        covariance_reference="Balanced ANOVA raw covariance matrices",
        scalar_reference="Henderson scalar composite and univariate components with negative scalar estimates clipped",
        results=differences,source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    (output/"multivariate_gd.json").write_text(json.dumps(evidence,indent=2)+"\n")
    print(json.dumps(evidence,indent=2))


if __name__=="__main__":
    main()
