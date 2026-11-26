"""MINQUE(0): independent kernels, design failures and retained public outputs."""
import itertools

import numpy as np
import pandas as pd
import pytest

import mfrmpy


def data_and_settings(nested=False, incomplete=True, multivariate=True):
    rng = np.random.default_rng(761)
    data = pd.DataFrame(list(itertools.product(range(8), range(4), range(3))), columns=['Person', 'Rater', 'Task'])
    p, r, t = (data[f].to_numpy() for f in ['Person', 'Rater', 'Task'])
    y = rng.normal(size=(8, 2))[p] * 3 + rng.normal(size=(4, 2))[r] + rng.normal(size=(3, 2))[t]
    y += rng.normal(size=(len(data), 2))
    data[['A', 'B']] = y
    if nested:
        data = data.loc[(data.Task != 0) | (data.Rater < 2)].copy()
    if incomplete:
        data = data.loc[data.index % 11 != 0].copy()
    cfg = dict(object_facet='Person', random_facets=['Rater', 'Task'], method='minque0', n_max_per_facet=2,
               selected_plan={'Rater': 5, 'Task': 4})
    if nested:
        cfg['nesting'] = {'Rater': 'Task'}
    if multivariate:
        cfg.update(score_cols=['A', 'B'], composite_weights={'A': .3, 'B': .7})
    else:
        cfg['score_col'] = 'A'
    return data, cfg


def run(data, cfg):
    return mfrmpy.analyze(data, workflow='gd', gd_settings=cfg)


@pytest.mark.parametrize('nested', [False, True])
def test_minque_dense_kernels_and_expectations(nested):
    data, cfg = data_and_settings(nested)
    a = run(data, cfg)
    result = a['gd_analysis']['result']
    assert result['available']
    design = result['design']
    n = len(data)
    P = np.eye(n) - np.ones((n, n))/n
    kernels = []
    for columns in design['component_factors'].values():
        kernels.append(np.logical_and.reduce([data[f].to_numpy()[:, None] == data[f].to_numpy()[None, :] for f in columns]).astype(float))
    H = np.array([[np.trace(P @ ki @ P @ kj) for kj in kernels] for ki in kernels])
    np.testing.assert_allclose(result['estimation']['kernel_gram'], H, rtol=1e-12, atol=1e-10)
    Y = data[['A', 'B']].to_numpy()
    Q = np.array([Y.T @ P @ K @ P @ Y for K in kernels])
    expected = np.linalg.solve(H, Q.reshape(len(kernels), -1)).reshape(-1, 2, 2)
    for observed, reference in zip(result['multivariate']['covariance_components'].values(), expected):
        np.testing.assert_allclose(observed, reference, rtol=1e-10, atol=1e-10)
    # Conditional expectation under a declared random-effects model recovers its components.
    truth = np.arange(1, len(kernels)+1, dtype=float)/3
    V = sum(v*K for v, K in zip(truth, kernels))
    expected_quadratics = np.array([np.trace(P @ K @ P @ V) for K in kernels])
    np.testing.assert_allclose(np.linalg.solve(H, expected_quadratics), truth, atol=2e-12)
    assert not design['observed_n']
    assert np.isnan(result['observed_coefficients']['G'])
    assert not result['d_study'].IsObservedDesign.any()
    assert result['selected_plan']['GStatus'] == 'point_estimate'
    # Independent D mean: average the covariance matrix over a complete future assignment.
    counts = cfg['selected_plan']
    units = list(itertools.product(range(counts['Rater']), range(counts['Task'])))
    weights = np.array([.3, .7])
    var = {s: max(0., float(weights @ m.to_numpy() @ weights)) for s, m in result['multivariate']['covariance_components'].items()}
    relative = absolute = 0.
    for source, factors in design['component_factors'].items():
        if source == 'Person':
            continue
        share = np.mean([all(u[['Rater', 'Task'].index(f)] == v[['Rater', 'Task'].index(f)] for f in factors if f != 'Person') for u in units for v in units])
        absolute += var[source]*share
        if 'Person' in factors:
            relative += var[source]*share
    assert result['selected_plan']['G'] == pytest.approx(var['Person']/(var['Person']+relative))
    assert result['selected_plan']['Phi'] == pytest.approx(var['Person']/(var['Person']+absolute))
    preview = mfrmpy.check_design(data, workflow='gd', gd_settings=cfg)
    assert preview['ok'] and preview['estimation']['identified']
    # Order, labels and score units must not determine the result.
    transformed = data.sample(frac=1, random_state=3).assign(B=lambda d: d.B*10)
    changed = run(transformed, dict(cfg, composite_weights={'A': .3, 'B': .07}))['gd_analysis']['result']
    for field in ['G', 'Phi', 'RelativeSEM', 'AbsoluteSEM']:
        assert changed['selected_plan'][field] == pytest.approx(result['selected_plan'][field], rel=1e-11)


@pytest.mark.parametrize('one_facet', [True, False])
def test_minque_balanced_matches_anova(one_facet):
    data, cfg = data_and_settings(incomplete=False)
    if one_facet:
        data = data.loc[data.Task == 0].drop(columns='Task')
        cfg.update(random_facets=['Rater'], selected_plan={'Rater': 5})
    a = run(data, cfg)['gd_analysis']['result']
    b = run(data, dict(cfg, method='anova'))['gd_analysis']['result']
    for source, matrix in a['multivariate']['covariance_components'].items():
        np.testing.assert_allclose(matrix, b['multivariate']['covariance_components'][source], atol=1e-11)
    for key in ['G', 'Phi', 'RelativeSEM', 'AbsoluteSEM']:
        assert a['selected_plan'][key] == pytest.approx(b['selected_plan'][key], rel=1e-11)
    assert a['design']['observed_n'] == b['design']['observed_n']
    cfg.pop('score_cols'); cfg.pop('composite_weights'); cfg['score_col'] = 'A'
    single = run(data, cfg)['gd_analysis']['result']
    direct = run(data, dict(cfg, method='anova'))['gd_analysis']['result']
    np.testing.assert_allclose(single['variance_components'].RawVariance, direct['variance_components'].RawVariance, atol=1e-11)


def test_minque_missing_and_invalid_inputs():
    data, cfg = data_and_settings()
    data.iloc[1, data.columns.get_loc('A')] = np.nan
    data.iloc[3, data.columns.get_loc('Task')] = np.nan
    failed = run(data, cfg)['gd_analysis']['result']
    assert failed['reason_code'] == 'missing_values'
    cfg['missing'] = 'omit'
    a = run(data, cfg)['gd_analysis']['result']
    assert a['design']['n_omitted_rows'] == 2
    assert a['excluded_rows'].RowPosition.tolist() == [1, 3]
    b = run(data.dropna(), cfg)['gd_analysis']['result']
    pd.testing.assert_frame_equal(a['variance_components'], b['variance_components'])
    assert a['design']['n_input_rows'] == len(data)
    assert a['design']['n_observations'] == len(data)-2
    for column, value, reason in [('B', np.inf, 'invalid_scores'), ('A', 'bad', 'invalid_scores'),
                                   ('Person', ' ', 'invalid_labels'), ('Person', np.inf, 'invalid_labels')]:
        bad = data.copy(); bad[column] = bad[column].astype(object); bad.iloc[0, bad.columns.get_loc(column)] = value
        assert run(bad, cfg)['gd_analysis']['result']['reason_code'] == reason
    duplicate = pd.concat([data, data.iloc[[0]]])
    assert run(duplicate, cfg)['gd_analysis']['result']['reason_code'] == 'duplicate_cells'
    duplicate.iloc[-1, duplicate.columns.get_loc('A')] = np.nan
    assert run(duplicate, cfg)['gd_analysis']['result']['reason_code'] == 'duplicate_cells'
    for update in [dict(method='reml'), dict(nesting={'Person': 'Task'}), dict(nesting={'Rater': 'Rater'}),
                   dict(missing='pairwise'), dict(fixed_condition={'facet':'X', 'mode':'mean', 'level':None}),
                   dict(coefficient_intervals={'method':'normal_delta', 'assumption':'normal', 'confidence':.95})]:
        with pytest.raises(ValueError):
            run(data, dict(cfg, **update))
    with pytest.raises(ValueError):
        run(data, dict(cfg, method='anova'))


@pytest.mark.parametrize('nested', [False, True])
def test_minque_unidentified_and_local_nested_labels(nested):
    data, cfg = data_and_settings(nested, incomplete=False)
    if nested:
        data = data.loc[data.Rater == 0]
    else:
        data = data.loc[data.Rater == data.Person % 4]
    a = run(data, cfg)
    r = a['gd_analysis']['result']
    assert not r['available'] and r['reason_code'] == 'unidentified_components'
    assert r['estimation']['rank'] < r['estimation']['n_components']
    assert not mfrmpy.check_design(data, workflow='gd', gd_settings=cfg)['ok']
    assert mfrmpy.tables(a, 'gd_kernel_gram')
    data, cfg = data_and_settings(True, incomplete=False)
    r = run(data, cfg)['gd_analysis']['result']
    assert r['design']['nested_child_units'] == 10  # local child labels count separately per parent
    assert not r['design']['balanced'] and r['design']['complete']
    data = data.loc[data.Rater < 2]
    r = run(data, cfg)['gd_analysis']['result']
    assert r['design']['observed_n'] == {'Rater': 2, 'Task': 3}


@pytest.mark.parametrize('multi', [False, True])
def test_minque_retained_outputs_and_components_only(tmp_path, monkeypatch, multi):
    from mfrm_app import unbalanced_gd
    data, cfg = data_and_settings(True, multivariate=multi)
    data.iloc[0, data.columns.get_loc('A')] = np.nan
    cfg['missing'] = 'omit'
    analyses = [run(data, cfg), run(data, dict(cfg, selected_plan=None))]
    monkeypatch.setattr(unbalanced_gd, 'study', lambda *a, **k: pytest.fail('must not re-estimate'))
    monkeypatch.setattr(unbalanced_gd, 'kernel_audit', lambda *a, **k: pytest.fail('must not re-audit'))
    for i, a in enumerate(analyses):
        path = tmp_path/f'{i}.mfrm'
        mfrmpy.save_project(path, a)
        restored = mfrmpy.load_project(path)
        assert restored['content_fingerprint'] == a['content_fingerprint']
        summary = mfrmpy.summary(restored).set_index('Field').Value
        assert summary['ObservedG'] == '' and summary['Method'] == 'minque0'
        assert summary['SupportsFormalInference'] == 'False'
        assert summary['OmittedRows'] == '1'
        tables = mfrmpy.tables(restored, 'gd')
        assert any('excluded_rows' in t[0] for t in tables)
        obs = mfrmpy.tables(restored, 'gd_observed')[0][1]
        assert obs.GStatus.eq('no_balanced_observed_design').all()
        paths = mfrmpy.export_csv(restored, tmp_path/str(i))
        assert any('RowPosition' in p.read_text() for p in paths[1:])
        if i:
            assert a['gd_analysis']['result']['available']
            assert not a['gd_analysis']['result']['selected_plan']
            assert a['gd_analysis']['result']['d_study'].empty


def test_minque_boundaries_and_archive_validation(tmp_path):
    from copy import deepcopy
    from mfrm_app import project
    data, cfg = data_and_settings(multivariate=False)
    for values in (np.zeros(len(data)), np.full(len(data), 4.)):
        a = run(data.assign(A=values), cfg)
        r = a['gd_analysis']['result']
        assert r['available'] and r['selected_plan']['GStatus'] == 'undefined_zero_denominator'
        assert r['variance_components'].RawVariance.eq(0).all()
    bad = data.assign(A=np.where(np.arange(len(data)) % 2, 1e308, -1e308))
    assert run(bad, cfg)['gd_analysis']['result']['reason_code'] == 'invalid_components'
    assert run(data.assign(A=data.A*1e-200), cfg)['gd_analysis']['result']['reason_code'] == 'invalid_components'
    _, weighted_cfg = data_and_settings()
    weighted_cfg['composite_weights'] = {'A': 1e-200, 'B': 2e-200}
    for method in ['minque0', 'anova']:
        full, _ = data_and_settings(incomplete=False)
        assert run(full, dict(weighted_cfg, method=method))['gd_analysis']['result']['reason_code'] == 'invalid_components'
    a = run(data, cfg)
    for field, value in [('nesting', {'Task':'Rater'}), ('missing', 'omit'), ('method', 'anova')]:
        altered = deepcopy(a)
        altered['gd_analysis']['result']['design'][field] = value
        altered['content_fingerprint'] = project.content_fingerprint(altered)
        with pytest.raises(project.ProjectError):
            mfrmpy.save_project(tmp_path/'bad.mfrm', altered)
    altered = deepcopy(a)
    altered['gd_analysis']['result']['estimation']['rank'] -= 1
    altered['content_fingerprint'] = project.content_fingerprint(altered)
    with pytest.raises(project.ProjectError):
        mfrmpy.save_project(tmp_path/'bad-rank.mfrm', altered)
    failed = run(data.iloc[:0], cfg)
    assert failed['gd_analysis']['result']['reason_code'] == 'missing_data'
    mfrmpy.save_project(tmp_path/'empty.mfrm', failed)
    assert not mfrmpy.load_project(tmp_path/'empty.mfrm')['gd_analysis']['result']['available']
    for update in [dict(nesting={'Rater':['Task']}), dict(method='reml'), dict(method='anova', missing='omit')]:
        with pytest.raises(ValueError):
            mfrmpy.check_design(data, workflow='gd', gd_settings=dict(cfg, **update))
    # Component labels must not collide with column names when exporting the Gram matrix.
    renamed = data.rename(columns={'Person':'Source'})
    unusual = run(renamed, dict(cfg, object_facet='Source'))
    assert len(mfrmpy.tables(unusual, 'gd_kernel_gram')[0][1]) == 49


@pytest.mark.figures
@pytest.mark.parametrize('multi', [False, True])
def test_minque_figures_and_html(tmp_path, monkeypatch, multi):
    import matplotlib.pyplot as plt
    from mfrm_app import unbalanced_gd
    data, cfg = data_and_settings(True, multivariate=multi)
    cases = [run(data, cfg), run(data, dict(cfg, selected_plan=None)), run(data.assign(A=0., B=0.), cfg)]
    monkeypatch.setattr(unbalanced_gd, 'study', lambda *a, **k: pytest.fail('must not re-estimate'))
    monkeypatch.setattr(unbalanced_gd, 'kernel_audit', lambda *a, **k: pytest.fail('must not re-audit'))
    for i, a in enumerate(cases):
        path = tmp_path/f'{i}.mfrm'
        mfrmpy.save_project(path, a)
        restored = mfrmpy.load_project(path)
        html = mfrmpy.report_html(restored)
        assert 'MINQUE(0)' in html and 'no_balanced_observed_design' in html
        for level in ([None, 'A'] if multi else [None]):
            fig = mfrmpy.plot(restored, 'gd', level=level)
            if i == 1:
                assert any('selected_plan' in text.get_text() for text in fig.axes[0].texts)
            elif i == 2:
                assert any('zero total variance' in text.get_text() for text in fig.axes[0].texts)
            plt.close(fig)
