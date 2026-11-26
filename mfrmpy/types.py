"""Types for public requests and retained analysis envelopes.

Analysis remains a dictionary for backwards compatibility. Nested estimator
records are private; use tables() for supported output schemas.
"""
from collections.abc import Callable, Sequence
from typing import Any, Literal, NotRequired, TypedDict
import pandas as pd


class IntervalRequest(TypedDict):
    facet: str
    difference: NotRequired[Sequence[str] | None]
    level: NotRequired[float]


class FitSettings(TypedDict, total=False):
    person: str
    person_col: str
    facets: Sequence[str]
    facet_cols: Sequence[str]
    score: str
    score_col: str
    rating_min: int
    rating_max: int
    model: Literal['RSM', 'PCM']
    method: Literal['JMLE', 'MML']
    step_facet: str | None
    maxit: int
    reltol: float
    quad_points: int | None
    population_prior_sd: float
    estimate_population_sd: Literal[False]
    population_mean: Literal['auto', 'fixed_zero', 'estimate']
    mml_engine: Literal['EM', 'Direct', 'Hybrid'] | None
    noncenter_facet: str
    positive_facets: Sequence[str] | None
    anchors: list[dict[str, Any]] | None
    group_anchors: list[dict[str, Any]] | None
    compute_pca: bool
    keep_original: Literal[True]
    umean: float
    uscale: float


class AnalysisSettings(FitSettings, total=False):
    workflow: Literal['mfrm', 'gd']
    interval_requests: Sequence[IntervalRequest] | None
    bias_requests: Sequence[dict[str, Any]] | None
    joint_settings: dict[str, Any] | None
    gd_settings: dict[str, Any] | None
    progress: Callable[[str], None] | None


class Analysis(TypedDict, total=False):
    kind: str
    analysis_id: str
    created_at: str
    content_fingerprint: str
    source: dict[str, Any]
    input_data: pd.DataFrame
    fit_settings: FitSettings
    computed_outputs: list[str]
    result: dict[str, Any]
    diagnostics: dict[str, Any]
    report_tables: dict[str, pd.DataFrame]
    scorefile: pd.DataFrame
    interval_analyses: list[dict[str, Any]]
    bias_analyses: list[dict[str, Any]]
    joint_analysis: dict[str, Any]
    gd_analysis: dict[str, Any]
    rater_dff: dict[str, Any]
    anchor_sensitivity: dict[str, Any]
    scope: dict[str, Any]
    qualification_at_save: dict[str, Any]
