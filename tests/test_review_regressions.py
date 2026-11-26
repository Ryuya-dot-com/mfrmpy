"""Review counterexamples: numerical availability is separate from inference."""
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

import mfrmpy
from mfrmpy._core import engine, project
from mfrmpy._core.design_rank import RowSpace


@pytest.mark.parametrize('number', range(4))
def test_vectorized_diagnostics_preserve_saved_reference(number):
    import json
    from pathlib import Path
    fixture = json.loads(Path(__file__).with_name('diagnostic_reference.json').read_text())['scenarios'][number]
    data = pd.DataFrame(**fixture['input'])
    data.Rater = pd.Categorical(data.Rater, categories=['X','Y','unused'])
    computed = dict(fit=engine.calc_facet_fit(data,['Person','Rater']),
                    bias=engine.calc_bias_facet(data,['Person','Rater']),
                    se=engine.calc_facet_se(data,['Person','Rater']),
                    agreement=engine.calc_interrater_agreement(data,['Person','Rater'],'Rater')['pairs'])
    for name, result in computed.items():
        expected = pd.DataFrame(**fixture['expected'][name])
        for column in result.select_dtypes(include='number'):
            expected[column] = pd.to_numeric(expected[column], errors='coerce')
        pd.testing.assert_frame_equal(result, expected, check_dtype=False, check_categorical=False, rtol=1e-11, atol=1e-11)


def test_repeated_rows_do_not_relax_estimability():
    row = sparse.csr_matrix([[1., 0.]])
    for count in (1, 100_000):
        space = RowSpace(sparse.vstack([row]*count))
        assert space.contains([1., 0.])
        assert not space.contains([1., 1e-12])
        assert space.tolerance == RowSpace(row).tolerance


def test_small_gd_interval_is_not_formal_inference(tmp_path):
    data = pd.DataFrame(dict(Person=['A','A','B','B'], Rater=['X','Y','X','Y'], Score=[0,1,3,5]))
    a = mfrmpy.fit(data, workflow='gd', gd_settings=dict(object_facet='Person', random_facets=['Rater'],
                    coefficient_intervals=dict(method='normal_delta', assumption='normal')))
    summary = mfrmpy.summary(a).set_index('Field').Value
    assert summary.SupportsFormalInference == summary.CoverageQualified == 'False'
    assert summary.CIEligible == 'True'
    meta = mfrmpy.tables(a, 'gd_metadata')[0][1].set_index('Field').Value
    assert meta.interval_inference_allowed == 'False'
    assert meta.interval_numerically_available == 'True'
    files = mfrmpy.export_csv(a, tmp_path/'nested'/'tables', kind='gd_observed')
    manifest = pd.read_csv(files[0])
    assert len(manifest) == 1 and not manifest.SupportsFormalInference.any()
    assert not manifest.CoverageQualified.any()
    path = tmp_path/'another'/'result.mfrm'
    mfrmpy.save_project(path, a)
    pd.testing.assert_frame_equal(mfrmpy.summary(mfrmpy.load_project(path)), mfrmpy.summary(a))


@pytest.fixture(scope='module')
def example_fits():
    cfg = dict(person='Person', facets=['Rater','Task','Criterion'], score='Score',
               rating_min=0, rating_max=5, method='MML')
    data = mfrmpy.load_example()
    return (mfrmpy.fit(data, **cfg),
            mfrmpy.fit(data, **cfg, quad_points=121, reltol=1e-12, maxit=1000))


def test_general_uncertainty_uses_retained_numerical_checks(example_fits, tmp_path, monkeypatch):
    coarse, fine = example_fits
    assert 'numerical_review' not in coarse['result']
    assert 'numerical_review' not in fine['result']
    monkeypatch.setattr(engine, 'mfrm_estimate', lambda *a, **k: pytest.fail('Unexpected refit'))
    for source, expected in ((coarse, False), (fine, True)):
        fingerprint = source['content_fingerprint']
        steps = source['result']['steps'].copy(deep=True)
        a = mfrmpy.diagnose(source, outputs=['measures'], interval_requests=[dict(facet='Rater')])
        assert a['result']['numerical_review']['passed'] == expected
        assert project.content_fingerprint(source) == fingerprint
        pd.testing.assert_frame_equal(source['result']['steps'], steps)
        table = mfrmpy.tables(a, 'measures')[0][1]
        assert table.SEUse.eq('screening_only').any() == expected
        assert pd.to_numeric(table.ModelSE, errors='coerce').notna().any() == expected
        reliability = mfrmpy.tables(a, 'reliability')[0][1]
        assert pd.to_numeric(reliability.Reliability, errors='coerce').notna().any() == expected
        path = tmp_path/f'{expected}.mfrm'
        mfrmpy.save_project(path, a)
        restored = mfrmpy.load_project(path)
        pd.testing.assert_frame_equal(mfrmpy.tables(restored,'measures')[0][1], table)
        if expected:
            raw = a['interval_analyses'][0]['record']
            np.testing.assert_allclose(a['result']['numerical_review']['covariance'],
                                      np.linalg.inv(raw['calculations'][0]['information']), rtol=1e-10)


def test_missing_numerical_evidence_holds_legacy_uncertainty(example_fits):
    source = example_fits[1]
    a = mfrmpy.diagnose(source, outputs=['measures'])
    a = dict(a, result={k:v for k,v in a['result'].items() if k != 'numerical_review'})
    a['content_fingerprint'] = project.content_fingerprint(a)
    table = mfrmpy.tables(a, 'measures')[0][1]
    assert pd.to_numeric(table.ModelSE, errors='coerce').isna().all() and table.Estimate.notna().any()


def test_diagnostic_failure_does_not_mutate_fit(example_fits, monkeypatch):
    source = example_fits[1]
    monkeypatch.setattr(engine, 'compute_mml_step_se_table', lambda *a, **k: (_ for _ in ()).throw(ValueError('step failure')))
    with pytest.raises(ValueError, match='step failure'):
        mfrmpy.diagnose(source, outputs=['measures'])
    assert project.content_fingerprint(source) == source['content_fingerprint']


def test_expected_agreement_uses_weighted_probability_means():
    data = pd.DataFrame(dict(Person=['A','A','A','B','B'], Rater=['X','X','Y','X','Y'],
                             Observed=[0.,1.,1.,0.,1.], Weight=[1.,3.,2.,1.,1.]))
    probs = np.array([[.8,.2],[.2,.8],[.3,.7],[.9,.1],[.4,.6]])
    got = engine.calc_interrater_agreement(data, ['Person','Rater'], 'Rater', res={}, _precomputed_probs=probs)
    expected = (np.dot((probs[0]+3*probs[1])/4,probs[2])+np.dot(probs[3],probs[4]))/2
    assert got['pairs'].ExpectedExact.item() == pytest.approx(expected)
    probs[0] = np.nan
    got = engine.calc_interrater_agreement(data.drop(columns='Weight'), ['Person','Rater'], 'Rater', res={}, _precomputed_probs=probs)
    assert got['pairs'].ExpectedExact.item() == pytest.approx(np.dot(probs[3],probs[4]))


def test_anchor_mean_and_calibration_metadata_select_same_model():
    from scipy.special import softmax
    rng = np.random.default_rng(81)
    data = pd.DataFrame([(str(p),r,int(rng.choice(3,p=softmax(np.arange(3)*(ability-shift)))))
                         for p,ability in enumerate(rng.normal(.7,1,60))
                         for r,shift in [('X',-.6),('Y',.4)]],columns=['Person','Rater','Score'])
    cfg = dict(person='Person',facets=['Rater'],score='Score',rating_min=0,rating_max=2,
               method='MML',quad_points=61,reltol=1e-12)
    fixed = [dict(Facet='Rater',Level='X',Anchor=-.6,Source='Known simulation coefficient')]
    anchored = mfrmpy.fit(data,**cfg,anchors=[{k:v for k,v in fixed[0].items() if k!='Source'}])
    assert anchored['fit_settings']['population_mean'] == 'estimate'
    assert anchored['fit_settings']['mml_engine'] == 'Direct'
    base = mfrmpy.fit(data,**cfg)
    a = mfrmpy.with_anchor_sensitivity(base,rater='Rater',difference=('Y','X'),intervals=True,
            scenarios={'No metadata':dict(anchors=fixed),'Known':dict(anchors=fixed,calibration={'kind':'known'})})
    table = mfrmpy.tables(a,'anchor_sensitivity')[0][1].set_index('Scenario')
    expected = anchored['result']['params']['facets']['Rater'][1] + .6
    assert table.loc['No metadata','Estimate'] == pytest.approx(expected)
    assert table.loc['Known','Estimate'] == pytest.approx(expected)
    levels = mfrmpy.tables(a,'Anchor sensitivity levels')[0][1]
    assert levels.loc[levels.Scenario.eq('No metadata'),'Origin'].str.contains('estimated person mean').all()
    constrained = mfrmpy.fit(data,**cfg,anchors=[dict(Facet='Rater',Level='X',Anchor=-.6)],population_mean='fixed_zero')
    assert not constrained['result']['config'].get('population_formula')


def test_memory_scope_allows_more_than_64_coordinates():
    from mfrmpy._core import mml_fixed_sd_review
    data = pd.DataFrame([(str(p),str(r),(p+r)%3) for p in range(6) for r in range(70)],
                        columns=['Person','Rater','Score'])
    a = mfrmpy.fit(data,person='Person',facets=['Rater'],score='Score',rating_min=0,rating_max=2,
                  method='MML',maxit=1,quad_points=3)
    prepared = mml_fixed_sd_review.prepare_fixed_sd_check(a['result'])
    assert len(prepared['coordinates']) > 64


def test_dff_extended_scale_and_early_method_validation(monkeypatch):
    data = pd.DataFrame([(f'{g}{p}',g,r,str(t),str(c),(p+r+t+c)%6)
                         for g in ['A','B'] for p in range(9) for r in range(2)
                         for t in range(2) for c in range(2)],
                        columns=['Person','Group','Rater','Task','Criterion','Score'])
    data.Rater = data.Rater.astype(str)
    data = mfrmpy.prepare_rater_dff_data(data,rater='Rater',group='Group',cell_facet='Cell')
    opts = dict(person='Person',facets=['Task','Criterion','Cell'],score='Score',rating_min=0,rating_max=5)
    args = dict(rater='Rater',group='Group',cell_facet='Cell',difference=('1','0'),group_difference=('B','A'))
    jmle = mfrmpy.fit(data,**opts)
    with pytest.raises(ValueError,match='MML'):
        mfrmpy.with_rater_dff(jmle,**args)
    a = mfrmpy.fit(data,**opts,method='MML',population_prior_sd=1.2,noncenter_facet='Cell',reltol=1e-12)
    monkeypatch.setattr(engine,'mfrm_estimate',lambda *a,**k: pytest.fail('Unexpected refit'))
    a = mfrmpy.diagnose(a,outputs=[],interval_requests=[dict(facet='Cell')])
    a = mfrmpy.with_rater_dff(a,**args)
    table = mfrmpy.tables(a,'rater_dff')[0][1]
    assert not table.CoverageQualified.any() and not table.SupportsFormalInference.any()


@pytest.mark.figures
def test_legacy_language_alias_emits_english(example_fits):
    import re
    from mfrmpy._core.desktop_outputs import figure_choices
    a = mfrmpy.diagnose(example_fits[1],outputs=['measures'])
    report = mfrmpy.report_html(a,language='ja')
    assert '<html lang="en">' in report
    assert not re.search('[一-龯ぁ-ゖァ-ヶ]',report)
    assert all(not re.search('[一-龯ぁ-ゖァ-ヶ]',title) for title,_ in figure_choices(a))
