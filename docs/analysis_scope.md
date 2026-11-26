# Analyses, interpretation and reporting limits

[Start here](../README.md) · [Windows guide](https://github.com/Ryuya-dot-com/MFRM_Python_Application/blob/main/docs/desktop_user_guide.en.md) · [日本語の操作・結果の読み方](https://github.com/Ryuya-dot-com/MFRM_Python_Application/blob/main/docs/desktop_user_guide.ja.md#分析を選ぶ)

Use this page to choose an analysis and understand what its results can support.
It describes the Web application and the desktop preview documented in this
source branch. The desktop preview includes RSM/PCM, bias diagnostics, selected
intervals, joint interactions and G/D studies; it is not yet a general desktop
release. The published Web release and this desktop preview do not have
identical controls or export formats.

## Choose the analysis for your question

| Question | Analysis | What to distinguish |
|---|---|---|
| How do person measures, rater severity and task difficulty compare? | Many-facet RSM or PCM | RSM shares category steps. PCM allows separate steps for levels of the chosen step facet. A main rater effect is not a rater-by-task interaction. |
| Does a particular combination of facet levels differ from the main-effects pattern? | Conditional bias diagnostics | These hold the original calibration fixed. They identify patterns to investigate, without confirming the presence or absence of bias. |
| What is the uncertainty in a facet level or a planned difference? | Specified structural intervals for fixed-SD MML | Only targets passing the checks receive intervals. A difference uses the full covariance, not the sum of marginal SEs. |
| What happens when an interaction is fitted together with main effects and steps? | Desktop joint interaction analysis | This refits a separate model. It does not turn the fixed-calibration diagnostic into a joint test. |
| How would numbers of raters or tasks affect score reliability? | G study and D study | These use observed scores and a declared random-facet design, not the MFRM logit scale. |

The Web interface also provides advanced workflows such as bounded GPCM,
latent regression, anchors and linking review, residual PCA, prediction and
simulation. Their availability does not establish that every displayed
statistic can support inference. Read the explanation for the selected output.

## Estimation and standard errors

JMLE jointly estimates person and facet parameters. Desktop MML uses EM with
a fixed normal population SD; the SD is a setting, not an estimated result.
The Web interface offers additional MML controls, including optional population
SD estimation, whose uncertainty and model-ranking uses remain unavailable.

A standard error must be interpreted for its stated target and method.
Conditional-information errors hold other parameters fixed. Structural
observed-information errors use a covariance calculation for the model
parameters. An MML person's EAP posterior SD describes a posterior distribution
with calibration fixed; it is not the ML standard error of a structural effect.
Do not treat these quantities as interchangeable because each is labelled SE.

The application does not support inferential use of general MML tables,
Person-score uncertainty, estimated population-SD uncertainty or automatic
equivalence conclusions. Selected structural intervals have the separate,
limited scope below. An available selected interval does not remove the
restrictions on other outputs.

## Assignment support for a rater comparison

`check_rater_contrast` traces shared works and checks whether the observed
allocation identifies a specified severity contrast. With a planned roster
it also reports missing cells and whether a specific cell would restore
structural support. In group comparisons, the same work ID across groups
requires an explicit decision to share its modeled location. The check uses
unit-weight ratings. Conditional likelihood information, finite estimates,
interval precision and coverage require separate checks. An anchor-dependent
result assumes the supplied fixed values. A size-limited result leaves
estimability unknown.
Large designs use person-column elimination and a bounded SVD of the reduced
system. This preserves the distinction between confounding and an unverified
rank. It does not establish finite estimates or interval coverage. Saving a fit
before diagnostics and selecting outputs do not relax inference holds.
[Examples and returned statuses](python_library.md#which-ratings-support-a-rater-comparison-development-version)

## Annual rater comparisons in the development Python API

An explicit calibration facet and level map can reuse a previous rater estimate
across years while retaining its covariance. Keep work IDs, rating-event IDs,
and rating occasions distinct. The report separates overlap with calibration
from works rescored across years, and compares the recording-linked estimate
with conditional and shared-calibration intervals. Shared calibration does
not correct bias from raters whose severity actually changed. These intervals
remain model approximations with `CoverageQualified=False`; annual coverage
findings cannot be borrowed from single-year calibration. If occasion and
rating mode are confounded, common works alone do not identify a mode-adjusted
annual contrast. [API, assumptions and examples](python_library.md#compare-a-returning-rater-across-years)

The Python comparison display defaults to a color-coded forest of individual
rater levels, separated by scenario and origin. Their pointwise intervals retain
the joint covariance, including estimated-origin and calibration terms. Interval
overlap is not a test of the difference, and validation of a contrast does not
qualify individual-level coverage. The difference display and normal-approximation
half-eyes require explicit selection. Older files without individual evidence
show recoverable points and hold reasons without refitting.
[Display choices](python_library.md#choose-the-comparison-display)

## Specified structural intervals

For supported unit-weight, fixed-SD RSM/PCM analyses, this output provides
pointwise asymptotic normal intervals for selected levels of a non-person
facet and a prespecified difference. It uses the full structural covariance.

The calculation checks termination, identification, parameter boundaries,
observed information and sensitivity to quadrature settings. These checks
assess numerical suitability. They do not prove that the response model fits
the population, give a strict quadrature error bound, or guarantee
finite-sample coverage.

The approximation assumes the specified response model and fixed normal
population, independent persons, ignorable assignment/missingness, an
identified interior parameter and growing information. It does not provide
simultaneous intervals, Person-score intervals or a test of equivalence.

The [English](https://github.com/Ryuya-dot-com/MFRM_Python_Application/blob/main/docs/fixed_sd_quadrature_check.en.md) and
[Japanese](https://github.com/Ryuya-dot-com/MFRM_Python_Application/blob/main/docs/fixed_sd_quadrature_check.ja.md) Web guides explain selecting,
saving and rechecking these intervals. In Desktop, register the requests in
**Settings → Bias / intervals** before running the analysis.

## Bias diagnostics and joint interactions

Conditional bias diagnostics describe cells with the original calibration
held fixed. Sparse cells, extreme scores and disconnected rating designs can
limit what can be estimated or compared. Displayed diagnostic values do not
establish statistical significance, absence of bias or assessment fairness.

The desktop joint-interaction option fits main effects, steps and one selected
structural-facet interaction together in a separate RSM/PCM model. It uses
unit weights and a normal person population fixed at mean 0 and SD 1.
Eligible requested cells and prespecified contrasts receive pointwise
approximate Wald intervals and p values. Holm adjustment applies to the
recorded family of requested tests; it does not make the intervals simultaneous.

This model assumes independent persons, the stated response/population model,
ignorable assignment/missingness and an identified interior solution with
growing information. Numerical checks are not a finite-sample error-rate
guarantee. An interaction is evidence about the fitted model, not a verdict
about fairness or a person. The original main-effects results remain separate.

## G studies and D studies

The default ANOVA design is completely crossed and balanced, with one or two
random facets besides the object of measurement, at least two levels of each
factor, one observation per cell and unit observation weights. Missing or
repeated cells and unsupported designs produce a reason rather than an
unsupported coefficient.

G describes relative decisions and Phi describes absolute decisions. The D
study changes planned numbers of observations under the stated design and
population assumptions. Negative estimated variance components are shown
alongside the zero-clipped components used for the coefficients. A coefficient
alone does not establish an acceptable assessment or validate a future design.

The desktop G/D entry uses the original observed scores without requiring an
MFRM fit. The development Python API
can explicitly request pointwise logit-delta intervals for G/Phi under independent
normal random effects in the supported balanced crossed designs. This includes
the full component covariance and random-facet uncertainty; it is a plug-in
approximation, not exact or distribution-robust inference. Relevant nonpositive
raw components and boundary coefficients withhold the corresponding interval.
Few facet levels can still give poor coverage. The old person-only bootstrap
remains withheld. The same conditions apply alongside an MFRM analysis; the
MFRM fit does not validate the normal model for the original scores.
[Interval assumptions and statuses](python_library.md#approximate-gphi-intervals-development-version)

The development multivariate entry accepts multiple outcomes on the same complete
balanced crossed cells and explicit nonnegative composite weights. It retains
raw covariance components and PSD diagnostics, then projects each score/composite
component before clipping negative scalar variances for G/Phi. This agrees with
directly forming the specified composite, but it does not repair the joint
covariance matrices or qualify estimated correlations. G/Phi and SEM are reported
separately for each outcome and the composite. Multivariate intervals, different
facet counts by outcome are not supported. Missing/unbalanced/nested designs require
the separate explicit MINQUE(0) method described below. Fixed-facet conditioning retains its existing elementary-cell checks.
[Multivariate assumptions and weights](python_library.md#multivariate-scores-and-a-weighted-composite-development-version)

The development `method="minque0"` supports one or two random facets, crossed or
with one non-object facet nested in the other. It estimates single-score variance
components or covariance components for the same retained rows of multiple scores.
`missing="error"` is the default; `missing="omit"` explicitly removes rows missing
any selected score or identifier, including outcomes with zero composite weight.
Absent cells need no placeholders. Duplicated cells, nonfinite/nonnumeric scores,
non-unit observation weights and invalid labels remain unsupported.

Identification uses the normalized Gram matrix of covariance kernels after removing
the intercept. Its numerical rank, tolerance, condition number and component support
are retained. Full rank supports component separation under the declared model; it
does not establish precise estimates or ignorable missingness. Independent random
effects with the same component covariance across levels are assumed conditional on
the assignment. Informative missingness can violate these assumptions; omission is
not a correction for selection. Negative and non-PSD component estimates are retained
and scalar projections are clipped as in the balanced multivariate entry.

An unbalanced design has no automatic observed G/Phi in this implementation. Its
pooled level counts are not replication counts. Without `selected_plan`, components
and diagnostics are available while D projections remain absent. An explicit plan
specifies the common counts for a future balanced mean; a nested child's count is
**per parent**, not its total across parents. These projections do not describe
the heterogeneous observed means. MINQUE(0) intervals, arbitrary or person-nested
layouts, cell replication and fixed-facet aggregation are not supported. Point
calculations have been checked against mfrmr 0.3.2 and independent dense-kernel
algebra; interval coverage and estimator efficiency have not been qualified.
[MINQUE(0) example and retained audit](python_library.md#missing-unbalanced-and-nested-designs-development-version)

## When a result cannot support inference

Read the message for the particular table or figure.

- **Input or design problem:** correct the indicated column mapping, score
  range, missingness or design problem. For example, unrelated rating groups
  may lack a common scale. Keep the original data and document any change.
- **Numerical or boundary problem:** review the stopping reason, identification
  and extreme scores. A finite displayed estimate or an optimizer stopping
  flag does not establish valid inference.
- **Unsupported uncertainty method or use:** the application has not
  established that use for this output. Repeatedly changing the data or
  rerunning the same settings does not resolve that limitation.
- **Changed settings or a saved result from another implementation:** check
  which analysis the displayed output belongs to. Reopened results retain the
  saved values but are checked against current output-specific rules.

A blank SE or interval means it is unavailable, not zero. Do not infer equality,
absence of bias or a reliable ranking from missing uncertainty or a
non-significant result. Use an eligible dedicated output only within its
stated scope.

## Comparing programs and reporting results

Match the model, data, weights, constraints, population distribution, scoring
method and numerical settings before comparing software.

In particular, this application's non-person PTMEA/PtMeaCorr correlates
observed scores with person measures. FACETS PtMea uses the signed sum of
measures for each observation. They are different diagnostics. The application's
RSM/PCM fit ZSTD follows a FACETS-style fourth-moment/Wilson–Hilferty convention
with an absolute cap of 9; the GPCM adaptation is approximate. Agreement in one
diagnostic does not establish equivalence of the programs.

Infit and Outfit describe squared residuals. Evaluate their magnitude alongside
the number of observations, category use and the person scoring method. The
fixed-SD MML fit table plugs EAP person measures estimated from the same ratings
into the category predictions. With few ratings per person, this reuse can lower
the mean squares even in data generated by the fitted model. The ZSTD
calculation carries that effect into the standardized value. Use fit statistics to
select ratings for inspection and use signed residuals or an explicit contrast
for questions about the direction of a rater effect.

AIC/AICc/BIC, model-choice LR values and automatic model recommendations remain
restricted to technical review. Residual PCA is a diagnostic requiring attention
to sparse observations, overlap and stability, not an automatic dimensionality
decision. Simulations describe their specified data-generating conditions;
they do not validate a different empirical design.

When reporting, retain the model, estimator, constraints, scoring rubric,
population-SD treatment, analysis version, uncertainty method and relevant
limitations. Preserve captions and notes with exported figures and tables.
Software citation does not replace citations for the statistical methods.

Implementation references for technical readers:
[output rules](../mfrm_app/output_qualification.py),
[structural intervals](../mfrm_app/structural_intervals.py),
[joint interactions](../mfrm_app/joint_interaction.py),
[G/D calculations](../mfrm_app/generalizability.py).
The [FACETS comparison](https://github.com/Ryuya-dot-com/MFRM_Python_Application/blob/main/validation/FACETS_WINDOWS_DESKTOP_REVIEW_20261029.md)
states the conditions and limits of its numerical checks.
