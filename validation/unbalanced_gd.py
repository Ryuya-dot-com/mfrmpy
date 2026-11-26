"""Point calculation checks against mfrmr 0.3.2; no coverage/efficiency claim.

PYTHONPATH=. python3 validation/unbalanced_gd.py --r-library <library>
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
from mfrm_app import unbalanced_gd


def dataset(name):
    rng = np.random.default_rng(761)
    data = pd.DataFrame(list(itertools.product(range(12), range(6), range(5))), columns=['Person', 'Rater', 'Task'])
    if name == 'one':
        data = data.loc[data.Task == 0].drop(columns='Task').reset_index(drop=True)
    nested = name.startswith('nested')
    if nested:
        cap = (data.Task % 3 + 3) if name != 'nested_balanced' else 3
        data = data.loc[data.Rater < cap].reset_index(drop=True)
        terms = [['Person'], ['Task'], ['Task', 'Rater'], ['Person', 'Task'], ['Person', 'Task', 'Rater']]
    else:
        factors = list(data.columns)
        terms = [list(s) for k in range(1, len(factors)+1) for s in itertools.combinations(factors, k)]
    y = np.zeros((len(data), 2))
    for columns in terms:
        codes = pd.factorize(pd.MultiIndex.from_frame(data[columns]), sort=True)[0]
        effects = rng.multivariate_normal([0, 0], [[2, .8], [.8, 1]], size=int(codes.max())+1)
        y += effects[codes]
    data[['A', 'B']] = y
    if name in ('one', 'crossed', 'nested_missing'):
        data = data.loc[data.index % 11 != 0].copy()
    if name == 'nested_missing':
        data.iloc[2, data.columns.get_loc('A')] = np.nan
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--r-library', required=True)
    args = parser.parse_args()
    output = Path('validation/evidence')
    computed = {}
    for name in ['one', 'crossed', 'nested_balanced', 'nested_unequal', 'nested_missing']:
        path = output/f'unbalanced_gd_{name}.csv'
        dataset(name).to_csv(path, index=False)
        data = pd.read_csv(path)
        facets = [f for f in ['Rater', 'Task'] if f in data]
        cfg = dict(object_facet='Person', random_facets=facets, method='minque0', missing='omit',
                   score_cols=['A', 'B'], composite_weights={'A': .3, 'B': .7}, n_max_per_facet=2,
                   selected_plan={f: 5 if f=='Rater' else 4 for f in facets})
        if name.startswith('nested'):
            cfg['nesting'] = {'Rater': 'Task'}
        analysis = mfrmpy.analyze(data, workflow='gd', gd_settings=cfg)
        assert analysis['gd_analysis']['result']['available']
        mfrmpy.save_project(output/f'unbalanced_gd_{name}.mfrm', analysis)
        computed[name] = analysis['gd_analysis']['result']
    subprocess.run(['Rscript', 'validation/unbalanced_gd.R', args.r_library, str(output)], check=True)
    differences = []
    for r in json.loads((output/'unbalanced_gd_mfrmr.json').read_text()):
        p = computed[r['name']]
        multi = p['multivariate']
        errors = []
        for source, matrix in multi['covariance_components'].items():
            ref = np.asarray(r['components'][source])
            np.testing.assert_allclose(matrix, ref, rtol=1e-10, atol=1e-10)
            errors.append(float(np.max(np.abs(matrix.to_numpy()-ref))))
        np.testing.assert_allclose(p['estimation']['kernel_gram'], r['gram'], atol=1e-9, rtol=1e-12)
        assert p['estimation']['rank'] == r['rank']
        np.testing.assert_array_equal(multi['component_diagnostics'].PositiveSemidefinite,
                                     [d['PositiveSemidefinite'] for d in r['diagnostics']])
        point_errors = []
        for reference in r['clipped_plans']:
            row = multi['selected'].loc[multi['selected'].Target.eq(reference['Target'])].iloc[0]
            for key in ['G', 'Phi', 'RelativeErrorVariance', 'AbsoluteErrorVariance', 'RelativeSEM', 'AbsoluteSEM']:
                np.testing.assert_allclose(row[key], reference[key], atol=1e-10, rtol=1e-10)
                point_errors.append(float(abs(row[key]-reference[key])))
        # R's native raw D-study uses signed components, a different policy from scalar clipping.
        raw_errors = []
        for reference in r['raw_plans']:
            comp = multi['variance_components'].loc[multi['variance_components'].Target.eq(reference['Score'])].copy()
            comp['Variance'] = comp.RawVariance
            c = unbalanced_gd.coefficients(comp, p['selected_plan'], p['design'])
            values = dict(G=c['G'], Phi=c['Phi'], RelativeErrorVariance=c['details']['sigma2_relative_error'],
                          AbsoluteErrorVariance=c['details']['sigma2_absolute_error'])
            for key, value in values.items():
                if reference[key] is not None:
                    np.testing.assert_allclose(value, reference[key], rtol=1e-10, atol=1e-10)
                    raw_errors.append(float(abs(value-reference[key])))
        differences.append(dict(design=r['name'], ratings=p['design']['n_observations'], omitted=p['design']['n_omitted_rows'],
            rank=p['estimation']['rank'], covariance_max_difference=max(errors),
            clipped_metric_max_difference=max(point_errors), native_raw_metric_max_difference=max(raw_errors),
            non_psd_components=multi['component_diagnostics'].loc[lambda t:~t.PositiveSemidefinite, 'Source'].tolist()))
    paths = ['mfrm_app/unbalanced_gd.py', 'mfrm_app/multivariate_gd.py', 'validation/unbalanced_gd.py', 'validation/unbalanced_gd.R']
    evidence = dict(python=platform.python_version(), platform=platform.platform(), mfrmr='0.3.2',
        mfrmr_archive_sha256='611ac538da670a58bd9a245d97fcc202b02524c5dc3dbcf03c071b48be68867e',
        comparison='Raw covariance, centered-kernel Gram, rank, PSD; native raw D-study plus explicit scalar-clipped R calculations. Point calculations only.',
        results=differences, source_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths})
    (output/'unbalanced_gd.json').write_text(json.dumps(evidence, indent=2)+'\n')
    print(json.dumps(differences, indent=2))


if __name__ == '__main__':
    main()


def large_design_check():
    """One 21,818-row public-API run; traced allocations, not total process RSS."""
    import time
    import tracemalloc
    data = pd.DataFrame(list(itertools.product(range(3000), range(4), range(2))),
                        columns=['Person', 'Rater', 'Task'])
    data = data.loc[data.index % 11 != 0].copy()
    rng = np.random.default_rng(98)
    data['Score'] = rng.normal(size=3000)[data.Person] + .2*data.Rater + .1*data.Task + rng.normal(size=len(data))
    tracemalloc.start()
    start = time.perf_counter()
    a = mfrmpy.analyze(data, workflow='gd', gd_settings=dict(object_facet='Person', random_facets=['Rater', 'Task'],
        score_col='Score', method='minque0', selected_plan={'Rater': 4, 'Task': 2}, n_max_per_facet=2))
    elapsed = time.perf_counter()-start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    result = a['gd_analysis']['result']
    assert result['available'] and result['estimation']['rank'] == 7
    assert result['estimation']['kernel_gram'].shape == (7, 7)
    assert np.isfinite(result['selected_plan']['G']) and not result['design']['observed_n']
    return dict(rows=len(data), persons=3000, raters=4, tasks=2, seed=98,
        omitted_cells='Every original row whose zero-based position is a multiple of 11',
        elapsed_seconds=elapsed, traced_peak_mib=peak/2**20, kernel_gram_shape=[7, 7],
        dense_observation_matrix_elements=len(data)**2, module_path=mfrmpy.__file__,
        selected_G=result['selected_plan']['G'], selected_Phi=result['selected_plan']['Phi'])
