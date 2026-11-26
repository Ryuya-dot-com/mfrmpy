# mfrmpy: Many-facet Rasch analysis in Python

`mfrmpy` brings the supported MFRM Desktop calculations to Python scripts and
notebooks. It uses the same Python engine as the existing MFRM Web and desktop
applications. It does not require R, FACETS, Streamlit or Tkinter for analysis.

This checkout is **0.2.3.dev4**, an unpublished development candidate. It adds
fit/diagnostic separation, reduced design audits, G/Phi intervals under explicit
balanced normal-model assumptions, multivariate G/D, and explicit MINQUE(0) for
supported unbalanced/nested designs. Sections marked “development version” describe
this candidate, not an assurance that those features are on PyPI. It also exposes
Fair Average tables and plain-text/ZIP reporting without requiring a project archive.
The [package repository](https://github.com/Ryuya-dot-com/mfrmpy) hosts library documentation and issues. Available analyses are RSM/PCM with JMLE
or fixed-population-SD MML, diagnostics, conditional bias, selected structural
intervals, a separate joint-interaction model, and observed-score G/D
studies. Numerical eligibility checks and interpretation limits remain in
force. Availability on PyPI is not statistical validation or certification.

## Installation

Install the published package from PyPI:

```bash
python -m pip install mfrmpy
```

To use this local development candidate, install its built wheel explicitly:

```bash
python -m pip install dist/development-review-20261126/mfrmpy-0.2.3.dev5-py3-none-any.whl
```

The development version has not been uploaded. A plain PyPI installation does
not install this local candidate. Use `mfrmpy.__version__` to identify your version.

Python 3.11 or newer and pandas 2.2.3 or newer are required. Archives that use
pandas 3's NaN-string dtype semantics require pandas 3 to reopen without changing
those semantics; other supported pandas 2 string dtypes round-trip normally.
Package captions and report notes are English. `language="ja"` is retained as
an alias for English output; input labels are preserved verbatim. Save/export
functions create missing parent directories.

For a source checkout, run `pytest`. After installing a candidate wheel, run
`pytest --installed` to include subprocess tests that deliberately ignore the
source directory. Figure tests additionally require the `figures` extra.

For Matplotlib figures and offline HTML reports:

```bash
python -m pip install 'mfrmpy[figures]'
```

To upgrade an earlier installation, including an alpha release:

```bash
python -m pip install --upgrade mfrmpy
```

Python 3.11 and 3.12 are the initial validation targets. Core dependencies are
NumPy >=1.26, pandas >=3.0, SciPy >=1.10 and PyArrow >=16.0. pandas 3 and
PyArrow preserve the string storage and missing-value types recorded in Web
and desktop `.mfrm` projects, including their content fingerprints. This
distribution provides a library, not the
standalone desktop executable or the Web app launcher. Those applications
remain available from the
[application repository](https://github.com/Ryuya-dot-com/MFRM_Python_Application).

## First analysis

Ratings must be in long form: one row per observed rating, with separate
person, facet and score columns. Specify the full intended integer rating
range, even when an endpoint is absent from the observations. Missing ratings
are not zero scores. The engine rounds floating-point values within its integer tolerance to the nearest category. Fractional and infinite scores produce an input error with row numbers. ID labels are trimmed, and ambiguous labels that would collapse after trimming require clarification in the input. Use a separate name for a facet called `Weight`, the reserved observation-weight field. The bundled example is synthetic.

For a five-category rubric, `3, 3, 3, 3` and `1, 2, 4, 5` each contribute four
observed ratings on the declared 1–5 scale. A blank score contributes no
response. RSM pools category support across ratings, while PCM assesses it
within each level of the chosen step facet. If a category is absent from a
person's ratings, keep the rubric scale and assess the response pattern
separately from the person's ability estimate.
[Five-category validation examples](../validation/rating_category_support.md)

```python
import mfrmpy

data = mfrmpy.load_example()
preflight = mfrmpy.check_design(
    data,
    person="Person",
    facets=["Rater", "Task", "Criterion"],
    score="Score",
    rating_min=0,
    rating_max=5,
)
print(preflight["summary"])

analysis = mfrmpy.analyze(
    data,
    person="Person",
    facets=["Rater", "Task", "Criterion"],
    score="Score",
    rating_min=0,
    rating_max=5,
    model="RSM",
    method="JMLE",
)

print(mfrmpy.summary(analysis))
for title, frame, note in mfrmpy.tables(analysis, kind="measures"):
    print(title)
    print(frame.head())
    print(note)

mfrmpy.save_project("writing.mfrm", analysis)
restored = mfrmpy.load_project("writing.mfrm")  # no refitting
```

Use `pandas.read_csv("ratings.csv", dtype={"Person": str, "Rater": str})`
for your own data, adjusting the ID columns as needed. Explicit string types
preserve leading zeroes. The input/design audit remains part of the returned
analysis and tables; inspect it before interpreting estimates.

## Save the fit before diagnostics (development version)

`analyze()` still performs the complete analysis. For larger data, `fit()`
returns the same fitted model, settings, convergence record and identification
audit before observation diagnostics, agreement calculations or scorefile
construction. Save this result before requesting additional outputs:

```python
fitted = mfrmpy.fit(
    data, person="Person", facets=["Rater", "Task", "Criterion"],
    score="Score", rating_min=0, rating_max=5, method="MML",
)
mfrmpy.save_project("calibration.mfrm", fitted)

restored = mfrmpy.load_project("calibration.mfrm")
analysis = mfrmpy.diagnose(restored, outputs="measures")
print(mfrmpy.tables(analysis, "measures")[0][1])
mfrmpy.save_project("measures.mfrm", analysis)
```

`diagnose()` returns a new analysis envelope and never refits the base model.
If an output fails, the supplied fit remains available for saving or retrying.
Already computed sections are reused. Choose any combination of:

| `outputs` | Computed information |
|---|---|
| `"measures"` | Measures, fit statistics, conditional/qualified uncertainty, reliability and fair-average report tables |
| `"agreement"` | Agreement and Krippendorff's alpha |
| `"scorefile"` | Observation-level scores, predictions, residuals and category probabilities |
| `"pca"` | Residual PCA with its existing availability and interpretation checks |

The default is `("measures", "agreement", "scorefile")`; PCA is explicit.
The analysis retains computed sections, and `summary()` lists computed and
pending outputs. Table/CSV/HTML output uses available sections; requesting a
figure requiring a pending section explains which calculation is needed.
`tables(analysis, "measures")` builds that table without constructing step,
agreement or supplementary tables. Neither output selection nor completing
diagnostics changes the model's inference qualification.

`fit()` accepts the same supplementary `interval_requests`, `bias_requests`,
`joint_settings`, and G/D workflow settings as `analyze()`. Add or replace a
request later with `diagnose(fitted, outputs=[], interval_requests=[...])`.
Only explicitly requested bias-cell or joint-model analyses perform their
documented separate fits. An MFRM `fit()` with no requests saves the estimate
before ordinary numerical review or report diagnostics.

MML diagnostics retain a same-point check at fitted GH order `q` and `2*q-1`:
analytic observed information without regularization, positive definiteness,
inverse residuals, scaled gradients, covariance/score changes, and posterior
mean/SD stability. General SEs, interval bars, separation, reliability and strata
are withheld when this evidence is absent or fails. Retained point estimates
remain visible where structurally reportable. Inspect
`analysis["result"]["numerical_review"]["reason"]` for the failed checks.
Displaying or reopening an analysis never refits it or manufactures missing
evidence. An explicit refit may use a tighter `reltol`, more `quad_points`, or
`mml_engine="Direct"`; no setting guarantees passage. These checks are not a
continuous-integral error bound or a coverage certificate. Fixed-grid GH
remains the integration method; adaptive GH is not implemented.

For MML with structural anchors, `population_mean="auto"` estimates the
population mean and defaults to the Direct optimizer. Unanchored auto fixes
the mean at zero. Use `population_mean="fixed_zero"` explicitly to reproduce
the former anchored model, or `"estimate"` with an anchored origin facet.
Calibration metadata affects uncertainty propagation, not this model choice.
General anchored-model uncertainty remains withheld where the ordinary
numerical review does not support it; anchor-sensitivity intervals use their
dedicated mean-estimating information calculation.

Partial analyses use schema 3 `.mfrm` archives. This development version reads
schemas 1, 2 and 3. Existing complete `analyze()` results continue to save as
schema 2; older applications cannot open partial schema-3 analyses.

Large JMLE designs eliminate person columns before a reduced rank calculation
when the original dense audit would exceed its limit. Specified rater contrasts
use the same elimination, including fixed/group constraints, group differences
and planned assignments. The 5,000,000-element limit applies to the remaining
distinct-row/nonzero-column system, rather than the original person-expanded
matrix. This is an algebraic rank check, not evidence of interval coverage.
If even the reduced system exceeds the limit, identification remains unknown.
Coordinate-level null-vector summaries are not constructed on the reduced JMLE
path. Old archives keep their recorded audit state; loading does not rerun it.

These changes do not provide out-of-core estimation. The input and fitted state
remain in memory; large agreement/PCA matrices and explicit scorefiles can
still be expensive. Project limits remain 128 MiB compressed and 512 MiB
uncompressed. MML's separate covariance-size and numerical qualification checks
are unchanged.

## Which ratings support a rater comparison? (development version)

With a source installation of the development version, use `check_rater_contrast`
before fitting. Include every facet whose effect you want to separate. For the
bundled writing example, R1 and R3 directly scored the same 30 people, and
their severity difference is estimable with Task and Criterion in the design.

```python
import mfrmpy

comparison = mfrmpy.check_rater_contrast(
    mfrmpy.load_example(), person="Person", facets=["Rater", "Task", "Criterion"],
    score="Score", rating_min=0, rating_max=5, rater="Rater", difference=("R1", "R3"),
)
print(comparison["status"])               # observed
print(comparison["overlap"][0]["path"])    # rater pair and shared person IDs per link
```

The first rater minus the second is always expressed in **severity** logits.
A positive contrast means the first rater is stricter. The returned
`contrast_weights` apply to the model coefficients, so they change sign if
`positive_facets` codes the rater coefficient as leniency. The structural
test concerns the location shared by the adjacent logits of an RSM or PCM
with centered steps. Category use, precision and finite parameter estimates
each require separate checks.

### Does the rater difference change between two schools?

Both assignments below have six works, eight ratings and four ratings per
rater. The question is whether (R1 − R2) at school A minus (R1 − R2) at
school B can be distinguished from differences in the quality of the works.

```python
import pandas as pd
import mfrmpy

def school_comparison(assignments):
    rows = [(school, work, rater) for (school, work), raters in
            zip([("A", "a1"), ("A", "a2"), ("A", "a3"),
                 ("B", "b1"), ("B", "b2"), ("B", "b3")], assignments)
            for rater in raters]
    data = pd.DataFrame(rows, columns=["School", "Work", "Rater"])
    data["Score"] = [0, 1] * 4
    return mfrmpy.check_rater_contrast(
        data, person="Work", facets=["Rater", "School"], score="Score",
        rater="Rater", difference=("R1", "R2"), group="School", group_difference=("A", "B"),
    )

both, r1, r2 = ("R1", "R2"), ("R1",), ("R2",)
for assignment in [(both, r1, r2, both, r1, r2), (both, both, r1, r1, r2, r2)]:
    result = school_comparison(assignment)
    print(result["status"], [g["kind"] for g in result["overlap"]])
# observed              ['direct', 'direct']
# common_location_only  ['direct', 'none']
```

The second assignment leaves both B-school raters free to explain the score
difference through the quality of their assigned works. The group argument
adds a rater-by-group interaction. Its coefficient raises the coded score,
so the weights for interaction cells (R1,A), (R2,A), (R1,B), (R2,B) are
(−1, +1, +1, −1) for this severity contrast. These are design results for a
prespecified comparison. The function fits no interaction and tests no DFF null hypothesis.

| `status` | How to read it |
|---|---|
| `observed` | The contrast is a linear combination of observed design rows while each person or work has its own location. |
| `anchor_dependent` | Observations and the supplied fixed-level or group-mean constraints together identify the contrast. Inspect the assumed calibration values. |
| `fixed_by_anchors` | The constraints already fix the contrast before seeing the ratings. `observation_estimable` separately records whether the assignments could support it without the anchors. |
| `common_location_only` | The free-person check fails. A second, unanchored calculation imposing a common person location removes the structural ambiguity. A population model would need its own distributional and numerical checks. |
| `unsupported` | The observed allocation, any supplied anchor constraints with free person locations, and the separate common-location calculation each leave the contrast unresolved. For example, one task per rater can confound task difficulty with severity. |
| `audit_size_limit` | The reduced calculation still exceeds five million matrix entries after person elimination and duplicate-row removal. Estimability is unknown. The overlap paths are still available. |

The check treats retained rows as unit-weight ratings. When preparing data
for a weighted analysis, pass only the ratings that enter that analysis.
For example, remove a zero-weight row that the estimator would exclude.
`observed` describes structural support with free work locations. CML also
depends on the possible response patterns at each observed total. A work
with all minimum or all maximum scores has only one such pattern and
contributes no conditional information, even if two raters scored it.

When both rubric endpoints are supplied, inspect `comparison["score_support"]`.
`category_counts` counts retained ratings by score and `by_rater` counts them
by rater and, if specified, group. Unlisted categories have zero ratings.
Replicate ratings count separately in these distributions.
`works_at_minimum` and `works_at_maximum` use all retained ratings for each
work ID, including across groups for a shared work location.
`observation_estimable_without_endpoint_works` checks the contrast's
assignment support after dropping all such works, with no anchors.
This check uses all remaining assignments, including alternate paths.
It is a CML design diagnostic. Finite estimates, category-step support
within each calibration scope, and interval accuracy require their own checks.

`overlap` reports one shortest path per requested group (or one for the entire
additive design), with **all** shared person IDs on each link. Paths describe
assignments. Other facets in the rank calculation can still confound the rater
difference. Set `person` to a unique performance or recording ID to trace
rescoring of the same work. For comparisons by `group`, each work ID must
belong to one group by default. Apply that rule across both ratings and the
planned roster. If schools each use numbers such as `1` for different works,
use globally unique IDs such as `A:1` and `B:1` in both tables. When the same
recording is intentionally scored under different groups or conditions, pass
`shared_work_locations=True` to give each repeated ID one shared location.
The result records this setting. The reported paths still describe each
group's own assignments.

Missing scores and incomplete assignments contribute no links. Replicate
ratings of one work contribute one assignment to the rank calculation. The
design check also accepts a single observed score category. Supply both rubric
endpoints to check that every retained score is in range.

Supply `anchors`, `group_anchors`, `noncenter_facet` and `positive_facets`
using the same conventions as `analyze`. Interaction rows and columns have zero
margins under those constraints. The check treats external anchor values as exact.
`estimable_with_anchors` is `None` when no anchors are supplied or the rank
calculation is size-limited. Use the original ratings to run this preflight
independently of a fitted or archived analysis.

### Compare planned and observed scoring

Pass a long-form `planned_assignments` table with the work ID and all facets
used in the contrast. For example, both schools planned to double-score one
work. School B has one missing score. Use the original group and rater columns.

```python
import pandas as pd
import mfrmpy

roster = pd.DataFrame(
    [("a1", "A", "R1"), ("a1", "A", "R2"), ("b1", "B", "R1"),
     ("b1", "B", "R2"), ("b2", "B", "R2")],
    columns=["Work", "School", "Rater"],
)
scored = roster.assign(Score=[3, 4, 2, None, 4])
review = mfrmpy.check_rater_contrast(
    scored, person="Work", facets=["School", "Rater"], score="Score",
    rater="Rater", difference=("R1", "R2"), group="School", group_difference=("B", "A"),
    planned_assignments=roster,
)
print(review["status"])  # common_location_only
print(review["plan"]["planned_observation_estimable"])  # True
print(review["plan"]["missing"])
# [{'assignment': {'Work': 'b1', 'School': 'B', 'Rater': 'R2'}, 'restores_contrast': True}]
```

The roster comparison counts unique work-by-facet cells. A blank or absent
score for a planned cell counts as missing. A scored cell absent from the
roster is listed as unplanned and remains in the observed design. Directly
shared works are counted once per group, even when both raters scored several
tasks on the same work. The structural calculation includes all supplied
facets and can detect confounding that these counts cannot show.

`planned_observation_estimable` checks the proposed roster by itself.
`completed_observation_estimable` checks the observed design plus all its
missing planned cells. Both checks concern structural support from the
assignments without anchors. If observed support is absent,
`restores_contrast` shows whether a particular missing cell would suffice
on its own. Several missing cells may be needed together. `None` means the
single-cell check was unnecessary or unavailable, including when the observed
design already supports the contrast or the rank check exceeds its size limit.
These checks use cell identities and leave the observed status unchanged.

An observed comparison can still be biased when missing ratings depend on the
scores that would have been assigned. In a five-category DFF simulation,
assignments supported the comparison under both random omissions and low-score
omissions. With the same number of missing ratings, low-score omissions produced
more false positives under both fixed-SD MML and CML. When interpreting a
comparison with missing scores, check the retained assignment history and any
rules for which performances receive a second rating.
[Score-dependent omissions in the DFF simulation](../validation/m5a_group_dff.md#得点に依存する欠測の反復検証)

## Reproduce category probabilities and severity contrasts (development version)

For a particular task, criterion and rater, `category_design` shows how their
coefficients determine the probability of each rubric score at a specified
ability. Use the `analysis` from *First analysis* above or a restored `.mfrm`.

```python
from scipy.special import softmax
import numpy as np

profile = {"Rater": "R1", "Task": "Task1", "Criterion": "Content"}
d = mfrmpy.category_design(analysis, levels=profile, theta=0.4)
print(d["origins"], d["population"], d["anchors"], d["group_anchors"])
print(d["terms"])       # term number, facet, level, transition and coefficient in logits
print(d["A"])           # rows are rubric scores, columns are the numbered terms
print(d["categories"])  # B (ability weight) and conditional probability
z = d["A"] @ d["terms"]["Estimate"] + d["categories"]["B"] * d["theta"]
np.testing.assert_allclose(softmax(z), d["categories"]["Probability"])
```

For rubric score `s`, the ability weight is `k = s - rating_min`. A difficulty
or severity has weight `−k`. A facet in `positive_facets` and a jointly fitted
interaction cell have weight `+k`. Each step up to that score has weight `−1`.
RSM uses one shared ladder, and PCM uses the ladder of the selected step-facet
level. In both models the steps in each ladder sum to zero. The output uses
expanded level coefficients at calculation precision. The A and B weights
follow TAM's category-logit equation. [TAM model and design arrays](https://alexanderrobitzsch.github.io/TAM/reference/tam.mml.html#details)

A fixed effect gives each observed rater a coefficient. A centering constraint
sets its origin. With the JMLE defaults, rater severities sum to zero over
raters with equal weight and person locations are free. For MML, `population`
records the mean and fixed SD of the integrated normal ability distribution.
The probabilities above condition on the supplied `theta` in canonical logits
for either estimator. With `noncenter_facet="Person"` MML centers all unanchored
structural facets while also fixing the ability mean at zero. For the
`Item + Rater + step` comparison with ConQuest's `lconstraints=cases`, use
`noncenter_facet="Item"` and match the population variance. A level anchor or
group-mean anchor sets the origin of its facet. [ConQuest location constraints](https://conquestmanual.acer.org/s4-00.html)

To reconstruct the severity difference, combine the rater coefficients from the
two profiles using the weights from the assignment check.

```python
audit = mfrmpy.check_rater_contrast(
    data, person="Person", facets=["Rater", "Task", "Criterion"], score="Score",
    rater="Rater", difference=("R1", "R3"),
)
severity_difference = 0.0
for term in audit["contrast_weights"]:
    selected = dict(profile, Rater=term["rater_level"])
    design = mfrmpy.category_design(analysis, levels=selected)
    row = design["terms"].query("Kind == 'facet' and Facet == 'Rater'")
    severity_difference += term["weight"] * row.Estimate.item()
print(audit["status"], severity_difference)  # R1 minus R3 in severity logits
```

For a fitted `joint_settings` model, pass `joint=True` to use its own main
effects, steps and interaction cell. For a difference between two groups,
request the same pair of facets in `check_rater_contrast`, select each of
the four profiles and apply its weight to the `interaction` row. Use
`shared_work_locations=True` when the same works were scored at both levels
and share a modeled location. Main effects cancel in this contrast. For
example, cells with coefficients `(+h, −h)`
in the first group and `(−h, +h)` in the second give a severity contrast
of `−4h`, because a positive cell coefficient raises the score.

### Estimate a prespecified two-school rater contrast

When each participant belongs to a single school, `joint_settings` can withhold
the whole interaction fit even if `check_rater_contrast` reports `observed` for
the specified difference. In the two-school example, the same fixed-SD RSM can
be fitted by using one severity level per school–rater pair and centering the
shared item effects. Take the severity difference within each school, then
subtract the two differences. Each school's location cancels in this contrast.

From the repository root with a source installation of the development version,
run the example to check the observed ratings and reproduce the **prespecified
contrast**, its full-information SE, and its model-based interval.

```bash
python -m validation.m5a_public_example --project dff-example.mfrm
```

The example uses a synthetic assignment on a common 1–5 rubric.
`prepare_rater_dff_data(data, rater="Rater", group="School", cell_facet="SchoolRater")`
copies the data and builds the composite levels from the trimmed IDs.
Rows with a missing or blank rater or group keep a missing composite level.
Fit an unanchored unit-weight RSM
with fixed-SD MML (SD 1), one common task facet and the composite facet as the
non-centered facet. Keep the original rater and group columns in the input.
For a model-based interval, request `interval_requests=[dict(facet="SchoolRater")]`
when fitting. Check the assignment with `check_rater_contrast` before fitting.

```python
# model is the retained composite-facet fit described above.
analysis = mfrmpy.with_rater_dff(
    model, rater="Rater", group="School", cell_facet="SchoolRater",
    difference=("R3", "R1"), group_difference=("B", "A"),
)
mfrmpy.save_project("dff-example.mfrm", analysis)
restored = mfrmpy.load_project("dff-example.mfrm")
print(mfrmpy.tables(restored, kind="rater_dff")[0][1])
mfrmpy.plot(restored, kind="rater_dff")  # requires the figures extra
mfrmpy.export_csv(restored, "dff-tables")
```

`with_rater_dff` returns an analysis with one ordered contrast request and its
assignment evidence. Point estimates and model-based intervals are read from
the fit and its retained numerical evidence without refitting. The same
result and the shared-work table appear in the CSV and offline HTML report.
New requests also retain category counts by group and rater and show how many
shared works have all ratings at a rubric endpoint. The assignment check
after excluding those works describes support relevant to CML. Interpret
the MML estimate and interval with their own fit diagnostics.
Pass `planned_assignments=roster` to retain the planned-versus-observed
diagnostics in `.mfrm` and include them in its tables, CSV and HTML.
If the assignment cannot identify the contrast, the result and the figure
show why it is withheld. A supported fit with an unavailable information
review can return a point estimate and the reason the interval is withheld.
Individual composite-level intervals still follow their own connectivity
checks. The shown Wald result is a model approximation with
`CoverageQualified=False` and `SupportsFormalInference=False`. The validation
report gives the allocations and effect sizes evaluated for coverage.
In the 20-shared-person simulation, increasing one school's generating ability
SD from 1 to 1.5 reduced correct-direction detection of a true +0.5 logit
contrast from 29.4% to 14.6% under fixed-SD MML.
[School ability variance sensitivity analysis](../validation/m5a_school_variance.md#能力sd差と真のdffが共存する場合)

The [group DFF interpretation example](../validation/m5a_dff_report.md) uses
the same ratings to compare the MML interval with a conditional likelihood
interval from mfrmr. It traces the shared work IDs back to ratings by task and
reopens a second project with one school's shared ratings missing. An interval
spanning zero can still include differences large enough to matter.

For exploratory item checks, the worked example also requests
`bias_requests=[dict(facet_a="SchoolRater", facet_b="Item")]`. The conditional
cell shifts are in logits of the score predictor with the calibration held
fixed. Use them to inspect tasks and rubric use in the shared works. The
prespecified contrast has its own severity direction and full-information
SE. The mfrmr residual comparison instead reports differences in score units.
[Worked example and interpretation](../validation/m5a_direct_overlap.md#公開python-apiで指定対比の点推定を再現する)

### Compare a returning rater across years

For an annual comparison of a returning rater, keep one `Recording` ID per
performance, including when the same recording is scored again. Use a separate
`RaterYear` level for each rating year and rater. The annual example checks the
observed link and requests `B-R3` minus `A-R3` with `interval_requests`, while
allowing the location of new works to differ by their presentation year.
[Annual recording example](../validation/m5b_annual_drift.md)

When selecting recordings by their earlier scores, retain the original ratings
for all eligible recordings and record the selection by score band and rater
pair. In the 24-recording simulation, selecting the highest- or lowest-scoring
recordings within each rater pair increased false positives under ordinary CML
even though the assignments supported the contrast. With fixed-SD MML,
sampling across all three score bands met the prespecified coverage criterion
at both tested drift values.
[Replay selection and interval coverage](../validation/m5b_annual_drift.md#共通録画を片側の得点帯から選ぶ場合)

Compare category use across rater-year levels when interpreting a severity
change. In the five-category replay simulation, a change in one rater's use
of category 4 appeared as a change in severity under shared-step RSM. A
PCM with `step_facet="RaterYear"` can estimate separate steps subject to
category support in each rater-year. Use the retained category counts and
interval holds to check which step sets the observed scores can support.
[Annual category-use experiment](../validation/m5b_annual_drift.md#カテゴリの使い方が年度によって変わる場合)

The [annual anchor-sensitivity example](../validation/m5b_anchor_report.md)
compares the unanchored estimate with fixing two raters to their old values,
removing each anchor, and fixing only their mean within each year. From a
repository checkout, reopen the retained example without fitting:

```bash
python -m validation.m5b_anchor_report validation/evidence/m5b_anchor_sensitivity_example.mfrm --read
```

The development API can now reproduce the conditional and shared-calibration
intervals directly. Keep `Recording` stable for the same recorded performance,
`RatingYear` for the scoring occasion, `WorkYear` for the presentation cohort,
and a unique `Event` ID for each actual rating. A new rating of the same recording
has a new event ID. The following assumes `RaterYear` contains labels such as
`A-R1`, and that both years use the same 1–5 rubric:

```python
settings = dict(person="Recording", score="Score", rating_min=1, rating_max=5,
                method="MML", population_prior_sd=1., quad_points=61, reltol=1e-12)
source = mfrmpy.analyze(ratings.query("RatingYear == 'A'"), **settings,
                       facets=["Rater"], noncenter_facet="Rater")
base = mfrmpy.analyze(ratings, **settings, facets=["WorkYear", "RaterYear"],
                     noncenter_facet="RaterYear", positive_facets=["WorkYear"])
anchors = []
for rater in ("R1", "R2"):
    terms = mfrmpy.category_design(source, levels={"Rater": rater})["terms"]
    value = terms.loc[terms.Facet.eq("Rater"), "Estimate"].item()
    for year in ("A", "B"):
        anchors.append(dict(Facet="RaterYear", Level=f"{year}-{rater}",
                            Anchor=value, Source="Year A calibration"))
linked = mfrmpy.with_anchor_sensitivity(
    base, rater="RaterYear", difference=("B-R3", "A-R3"), occasion="RatingYear",
    intervals=True, important_difference=.5,
    scenarios={"Previous values": dict(anchors=anchors, calibration=dict(
        kind="shared", analysis=source, rating_id="Event", facet="Rater",
        level_map={f"{year}-{rater}": rater for year in ("A", "B")
                   for rater in ("R1", "R2", "R3")}))},
)
mfrmpy.export_out(linked, "annual.out", kind="anchor_sensitivity")
mfrmpy.export_bundle(linked, "annual.zip", kind="anchor_sensitivity",
                     plots=["anchor_sensitivity"])
mfrmpy.save_project("annual.mfrm", linked)
```

`calibration.facet` names the source facet. `level_map` maps **every receiving
rater level** to a source level, including the free target rater for event
matching. Only the entries in `anchors` are fixed. Mapping `A-R1` and `B-R1`
to `R1` uses the same estimated value twice, with its full covariance; these
are not independent calibration estimates. Source and receiving rater signs
and rating scales must agree. Other source facets must have matching names
and signs in the receiving model. The source must be an unanchored fixed-SD
RSM/MML; an explicit mapping permits a noncentered source rater facet.
The fixed values must match the source's unrounded native logit estimates.

The optional `occasion` column identifies the distinct occasions of the two
selected contrast levels and adds an all-work overlap table. It does not change
the estimator or certify that the design supports an annual contrast. With a
mapped source and `rating_id`, that occasion column must also be present in
the calibration input; the same event ID cannot refer to a different occasion,
work, mapped rater, other source facet or score.

The retained example distinguishes three counts: 60 recordings shared between
calibration and the receiving analysis, 120 reused rating events, and 6
recordings scored in both years. The R3-to-R3 comparison path uses 4 of those
6 recordings. They are different types of overlap. `OverlapWorks`,
`ReusedRatings`, `CalibrationLevelMap`, variance components and the occasion
table preserve these distinctions through saving and all report formats.

The [ordinary-API example](../validation/evidence/annual_api/annual_drift.out)
returns +0.203 logit without anchors and −0.703 with previous values fixed;
the conditional and shared-calibration SEs are 0.334 and 0.314. Its
[confounded counterpart](../validation/evidence/annual_api/annual_mode_confounded.out)
withholds the contrast when rating occasion and live/video mode coincide,
including when only previous-year rater values are fixed. The default plot is a
color-coded forest of the selected individual levels. An explicitly selected
difference plot also shows the signed important-difference line. A sampling-error adjustment does not
remove bias from anchored raters whose severity actually changes.

The public-API check reproduced 18 cases from the existing annual experiments;
it is not a new coverage study. `CoverageQualified=False` and
`SupportsFormalInference=False` remain explicit. Use the
[annual calibration conditions and coverage results](../validation/m5b_estimated_anchors.md),
not single-year evidence or the overlap count alone. Older files without these
interval records remain point-only in the ordinary API; the special validation
record in the earlier example is not silently promoted into an API interval.

### Choose the comparison display

The default for `anchor_sensitivity`, including the automatic HTML and ZIP
figures, is **only a color-coded forest of the selected rater levels**.
Colors and marker shapes identify the levels. Each scenario has its own panel
and states its origin: an unanchored fixed person mean and an estimated person
mean located by anchors are different reference systems. Conditional and
calibration-propagated uncertainty appear as separate labelled rows.

```python
# Default: individual B-R3 and A-R3 estimates and their retained intervals.
fig = mfrmpy.plot(linked, "anchor_sensitivity")

# Explicit alternatives, independent choices of target and display form:
delta = mfrmpy.plot(linked, "anchor_sensitivity", view="difference")
halfeye = mfrmpy.plot(linked, "anchor_sensitivity", style="halfeye")
three = mfrmpy.plot(linked, "anchor_sensitivity",
                    levels=["A-R1", "A-R3", "B-R3"])
level_table = mfrmpy.tables(linked, "Anchor sensitivity levels")[0][1]
mfrmpy.export_bundle(linked, "chosen_figures.zip", kind="anchor_sensitivity",
    plots=[{"kind": "anchor_sensitivity"},
           {"kind": "anchor_sensitivity", "view": "difference"}])
```

The x-axis uses native coefficients in logits. With `FacetSign=-1`, larger
values lower predicted scores (greater severity); with `+1`, the direction is
reversed. Pointwise intervals are normal approximations, not simultaneous
intervals. Fixed anchors appear as fixed points under conditional uncertainty;
propagation can give them nonzero calibration uncertainty. Half-eye shading is
explicitly a normal approximation from the estimate and SE, not posterior draws
or an empirical distribution. No additional density dependency is required.

Individual interval overlap does **not** determine whether the difference
excludes zero. The saved `level_estimates` contain the native estimates and
joint covariance for every fitted rater level, with anchor sensitivities and
joint work contributions when available. The public level table includes
`JointCovariance` and its `LevelOrder`; these recover the contrast variance as
`V[first, first] + V[second, second] - 2*V[first, second]`. The original difference
table remains the direct source for the prespecified comparison and its signed
important difference. The important-difference line is absent from the level
plot because it is a threshold for a difference, not for a native level.

These individual intervals have not received a separate coverage qualification;
`CoverageQualified=False` remains explicit. Existing contrast simulations are
not evidence for individual-level coverage. Old files without level records
recover baseline points from the saved fit, withhold unavailable alternate
levels/intervals with a reason, and never refit or split a contrast SE into two
individual SEs. Their retained differences remain available via
`view="difference"`. To customize beyond these choices, use the public tables
and the plotting library of your choice.

### Compare alternate anchor settings

`with_anchor_sensitivity` takes an unanchored fixed-SD MML RSM and fits
each supported alternative to the same ratings. Supply a label for each
scenario and document the source of each fixed value with `Source`. The
example below uses illustrative anchor values.

```python
import mfrmpy

reference = mfrmpy.analyze(
    mfrmpy.load_example(), person="Person", facets=["Rater", "Task", "Criterion"],
    score="Score", rating_min=0, rating_max=5, method="MML",
)
fixed = [
    dict(Facet="Rater", Level=level, Anchor=value, Source="Illustrative calibration")
    for level, value in (("R1", -0.4), ("R2", 0.2))
]
mean = [
    dict(Facet="Rater", Level=row["Level"], Group="reference", GroupValue=-0.1,
         Source="Illustrative calibration")
    for row in fixed
]
analysis = mfrmpy.with_anchor_sensitivity(
    reference, rater="Rater", difference=("R3", "R1"), scenarios={
        "Fixed R1 and R2": dict(anchors=fixed),
        "Remove R1": dict(anchors=fixed[1:]),
        "Remove R2": dict(anchors=fixed[:1]),
        "Group mean": dict(group_anchors=mean),
    },
)
for title, frame, note in mfrmpy.tables(analysis, "anchor_sensitivity"):
    print(title, frame)
mfrmpy.save_project("anchors.mfrm", analysis)
```

The contrast and changes from `No anchors` are in severity logits. The
constraints table gives the native model sign, value, source and target
levels. A group mean uses one equation for its listed levels. The shared-work
table traces the observed path. `AssignmentSupport` also checks the full
facet structure, so a shared-work link can coexist with an unidentified
contrast. If the constraints already determine the contrast, the table
gives `fixed_by_anchors` and a reason in place of an estimated difference.

Each alternative inherits the reference fit settings. Use
`noncenter_facet` in a scenario to change which facet carries the free
location, as in the annual example. The scenario table records that choice.
The project stores the fitted points and design evidence for reuse in tables,
CSV and HTML. The [annual validation report](../validation/m5b_anchor_report.md)
also documents its separate calculations of shared-calibration uncertainty.

### Compare conditional and independent-calibration intervals

Set `intervals=True` to add model-based 95% intervals to the sensitivity
report. An anchored scenario with individually fixed levels of the selected
rater facet can specify `calibration`. This uses those levels to set the
location and fits the receiving population mean. The baseline and alternatives
with only a change in SD use a fixed zero population mean. Inspect
`PersonMean` and `PersonSD` in the comparison table.

- `dict(kind="known")` gives a conditional interval for the specified fixed values.
- `dict(kind="independent", work_ids=calibration_ids, covariance=covariance)`
  gives the conditional and propagated intervals. Supply a pandas DataFrame
  in logit squared with the fixed rater levels as both row and column labels.
  Include the off-diagonal covariance. Declare independence based on how
  the calibration and receiving samples were collected. Supply work IDs as
  strings in the same namespace as the receiving work IDs.
- `dict(kind="shared", work_ids=calibration_ids)` records shared sampling
  units. The table can still show a conditional interval. It withholds the
  independent propagation when sampling units are shared or when the work
  IDs supplied for an independent calibration appear in the receiving fit.
- With the fitted calibration project, `dict(kind="shared", analysis=source)`
  also computes a person-cluster sandwich interval when both fits pass the
  numerical checks. An optional `rating_id` column identifies reused rating
  events. [Shared calibration example](#use-a-calibration-project-with-shared-works)

The following uses synthetic ratings and illustrative calibration inputs.

```python
import mfrmpy
import pandas as pd

ratings = pd.DataFrame([
    dict(Work=f"W{p:02}", Rater=f"R{r+1}", Task=f"T{t+1}", Score=(p+2*r+t) % 3)
    for p in range(30) for r in range(3) for t in range(2)
])
reference = mfrmpy.analyze(
    ratings, person="Work", facets=["Rater", "Task"], score="Score",
    rating_min=0, rating_max=2, method="MML", quad_points=121, reltol=1e-11,
)
fixed = [
    dict(Facet="Rater", Level=level, Anchor=value, Source="Illustrative calibration")
    for level, value in (("R1", -0.2), ("R2", 0.1))
]
covariance = pd.DataFrame(
    [[0.030, -0.012], [-0.012, 0.025]],
    index=["R1", "R2"], columns=["R1", "R2"],
)
independent = dict(
    anchors=fixed,
    calibration=dict(kind="independent", work_ids=["cal:001", "cal:002"], covariance=covariance),
)
analysis = mfrmpy.with_anchor_sensitivity(
    reference, rater="Rater", difference=("R3", "R1"),
    intervals=True, important_difference=0.5,
    scenarios={
        "Independent calibration, SD 1": independent,
        "Independent calibration, SD 1.5": dict(independent, population_prior_sd=1.5),
        "No anchors, SD 1.5": dict(population_prior_sd=1.5),
    },
)
_, table, _ = mfrmpy.tables(analysis, "Anchor sensitivity intervals")[0]
print(table[["Scenario", "Basis", "Estimate", "SE", "Lower", "Upper", "Reason"]])
```

Use `IncludesZero` and, optionally, `IncludesImportantDifference` to
compare the values still compatible with each model. The latter checks the
signed value supplied by the analyst. The conditional and propagated rows
share one point estimate. A zero calibration covariance recovers the
conditional SE and endpoints. The propagated row also gives the labelled
covariance and the contrast's derivatives with respect to the fixed values.
Group-mean scenarios remain available for point-estimate sensitivity. The
interval table labels available intervals as model approximations. If the
score or information changes too much between quadrature rules, `Reason`
records the failed numerical check.
See the recorded designs and coverage when interpreting them. [Independent
calibration example and validation](../validation/m3_anchor_interval_report.md)

### Use a calibration project with shared works

For the same facet names and levels, pass the unanchored, centered, fixed-SD MML RSM calibration project and its
unrounded rater coefficients in native logits. Both projects must use
the same person or work ID for the same sampling unit, matching facet names
and signs, the same rating scale, and unit observation weights. Explicitly mapped
annual levels use the [annual workflow](#compare-a-returning-rater-across-years).
This interval
assumes independence across sampling units. When two works by one person
have correlated scores, treating each work as an independent person changes
that assumption.

The following uses the synthetic `ratings` table above. All 30 works occur
in both stages. The receiving sample reuses their T1 ratings plus the T2
ratings for R1.

```python
events = ratings.assign(RatingID=["rating-" + str(i) for i in range(len(ratings))])
settings = dict(person="Work", facets=["Rater", "Task"], score="Score",
                rating_min=0, rating_max=2, method="MML", quad_points=121, reltol=1e-11)
source = mfrmpy.analyze(events, **settings)
receiving = mfrmpy.analyze(events.query("Task == 'T1' or Rater == 'R1'"), **settings)
anchors = []
for rater in ("R1", "R2"):
    terms = mfrmpy.category_design(source, levels=dict(Rater=rater, Task="T1"))["terms"]
    value = terms.loc[terms.Facet.eq("Rater"), "Estimate"].item()
    anchors.append(dict(Facet="Rater", Level=rater, Anchor=value, Source="Calibration sample"))
linked = mfrmpy.with_anchor_sensitivity(
    receiving, rater="Rater", difference=("R3", "R1"), intervals=True,
    scenarios={"Shared works": dict(anchors=anchors, calibration=dict(
        kind="shared", analysis=source, rating_id="RatingID"))},
)
_, intervals, _ = mfrmpy.tables(linked, "Anchor sensitivity intervals")[0]
print(intervals[["Basis", "ReusedRatings", "Estimate", "SE", "SharedCrossTerm", "Reason"]])
```

`RatingID` is an event identifier unique within each input. Separate ratings
of the same work have different event IDs, even if their scores agree.
Omitting `rating_id` leaves the reused rating count unknown. Person IDs still
determine which contributions are paired. With only a list of IDs or an
anchor covariance matrix, the conditional interval can be reported and the
shared interval carries a hold reason.

The `shared_calibration` row combines calibration and receiving variance
with their signed cross term. Its SE may increase or decrease. The report
retains the source identity and the contributions of each sampling unit, so
the interval, or its hold reason, survives saving and exporting. A common
source of uncertainty in annual comparisons is a change in the anchored
rater's severity itself. A sampling-error adjustment assumes the fixed
values apply to the receiving sample. [Single-year validation](../validation/m3_shared_anchors.md) · [Annual anchor sensitivity](../validation/m5b_anchor_report.md)

### Check coefficient availability

Inspect `withheld_reason` when estimates or probabilities are missing.
`baseline_state` records the baseline fit's qualification. For a joint model
`joint_interval_reason` reports the selected cell's interval hold separately.
Choosing an unobserved combination of known levels evaluates the model at
that combination. The assignment check describes the observations that can
support a rater comparison. The dense design is limited to five million entries
per requested profile.

## Models and supplementary requests

`analyze` is the shared application service, not a second implementation of
the estimators. It returns a retained-analysis dictionary and does not display
an interface. Its MFRM settings are:

| Setting | Meaning / default |
|---|---|
| `person_col` (`person`), `facet_cols` (`facets`), `score_col` (`score`) | Required column names; `facet_cols` / `facets` is a list |
| `rating_min`, `rating_max` | Required integer rubric endpoints (`int` or `numpy.integer`), minimum < maximum |
| `model`, `method` | `"RSM"` / `"PCM"`; `"JMLE"` / `"MML"`; defaults RSM and JMLE |
| `step_facet` | Required for PCM; one selected facet with separate step structures |
| `maxit`, `reltol` | Maximum iterations (400) and stopping tolerance (1e-6) |
| `quad_points`, `population_prior_sd` | Fixed-SD MML quadrature count (default `31` for MML) and normal population SD (1.0) |
| `noncenter_facet`, `positive_facets` | Non-centered reference facet (default `"Person"`) and optional list of facets given additive `+1` leniency/easiness orientation |
| `anchors`, `group_anchors` | Fixed-level / group-mean anchor records under the existing identification rules |
| `compute_pca` | Optional residual diagnostics; default `False` |
| `umean`, `uscale` | Reporting origin (0) and positive scale multiplier (1); canonical logits are retained |
| `progress` | Optional callback receiving phase names, not percentages |

Keep `keep_original=True`, `estimate_population_sd=False` and `mml_engine="EM"`.
Other choices, bounded GPCM and latent regression are outside this initial
public interface. A numerical termination flag alone is not evidence that
inference is valid; the returned output qualifications also matter.

Example requests, supplied **when fitting**, are:

```python
analysis = mfrmpy.analyze(
    data,
    person_col="Person",
    facet_cols=["Rater", "Task", "Criterion"],
    score_col="Score",
    rating_min=0,
    rating_max=5,
    model="PCM",
    step_facet="Criterion",
    method="MML",
    population_prior_sd=1.0,
    bias_requests=[{"facet_a": "Rater", "facet_b": "Task"}],
    interval_requests=[{"facet": "Rater", "difference": ["R1", "R2"], "level": 0.95}],
    joint_settings={"facet_a": "Rater", "facet_b": "Task",
                    "difference": [["R3", "T1"], ["R1", "T1"]]},
)
```

Replace the contrast labels with levels that actually occur in your data.
`bias_requests` estimates a score-increasing logit shift for each observed
cell, holding person estimates, main effects and steps fixed. Its SE and
likelihood-ratio statistic are for screening under that fixed calibration.
`joint_settings` refits main effects, steps and interaction terms together in
a fixed-N(0,1), unit-weight model. The example subtracts the R1 × T1 interaction
from the R3 × T1 interaction. A positive difference means that the interaction
raises R3's expected score on T1 more than R1's, relative to their own main
effects. Use the joint covariance to account for the uncertainty of both cells
and all other estimated parameters. The joint table includes the target type
and `Ratings` for each cell. With a fixed population mean and sum-to-zero
effects, a zero-count cell can have a model-implied estimate and interval. Selected
structural intervals require fixed-SD, unit-weight RSM/PCM and suitable
identification, termination, boundaries, information and quadrature checks.
They are pointwise asymptotic intervals, with no finite-sample coverage
guarantee. Holm adjustment of joint tests does not make intervals simultaneous.

Unavailable/failed interval or joint requests remain recorded with a reason;
they do not silently erase a completed baseline fit. For details, see the
[analysis scope](https://github.com/Ryuya-dot-com/mfrmpy/blob/main/docs/analysis_scope.md).

## G study and D study without an MFRM fit

By default these analyze observed scores in a complete, balanced design with one or two
random facets, at least two levels of each factor and one observation per cell.
The score need not be an integer for G/D analysis. The development MINQUE(0)
extension below handles supported unbalanced/nested designs. For example:

```python
import pandas as pd

ratings = pd.DataFrame([
    {"Person": str(p), "Rater": str(r), "Task": str(t),
     "Score": 0.4*p + 0.3*r + 0.2*t + 0.1*((p+r+t) % 3)}
    for p in range(8) for r in range(3) for t in range(2)
])
gd_analysis = mfrmpy.analyze(
    ratings,
    workflow="gd",
    gd_settings={
        "object_facet": "Person",
        "random_facets": ["Rater", "Task"],
        "score_col": "Score",
        "target_population": "Comparable persons, raters and tasks",
        "selected_plan": {"Rater": 4, "Task": 2},
    },
)
for title, frame, note in mfrmpy.tables(gd_analysis):
    print(title, frame, note, sep="\n")
```

Unsupported designs return an unavailable result with a reason. G is for
relative decisions and Phi for absolute decisions. These are not MFRM person
reliability. The development version also supports the multivariate workflow
below. A fixed
conditioning facet can be specified through `gd_settings["fixed_condition"]`
under the existing desktop rules; it does not add another random facet.

### Approximate G/Phi intervals (development version)

For the supported univariate complete balanced crossed designs, explicitly request a
normal random-effects approximation in `gd_settings`:

```python
"coefficient_intervals": {
    "method": "normal_delta",
    "assumption": "normal",
    "confidence": 0.95,
},
```

This setting assumes independent normal effects for sampled objects, each
random facet, and their interactions/residual. Declaring a target population
does not establish these assumptions. The approximation is not automatically
appropriate for ordinal scores, dependent raters, or very few facet levels.
The default remains point estimates only; the old person-only bootstrap is
still withheld because it omits random-facet sampling uncertainty.

The observed-design, selected-plan and D-study tables include `G_SE`,
`G_lower`, `G_upper`, `G_width`, `G_interval_status` and the corresponding `Phi` fields.
Each row also identifies the interval method, confidence and assumption.
The interval is computed on `log(object variance / error variance)` and
transformed back to [0, 1]. Its covariance includes all ANOVA components:
if expected mean squares satisfy `E[MS] = E v`, the covariance is
`E^-1 diag(2 E[MS]^2 / df) E^-T`, with nonnegative fitted components
substituted into the working expected mean squares. The full covariance,
assumption, confidence level and coefficient-specific status are saved.
`SE` is the delta standard error on the coefficient scale; the asymmetric
interval is not `estimate +/- 1.96 * SE`.

`approximate_normal_delta` means the model-based approximation was returned.
`nonpositive_component` withholds an interval when a relevant raw component
is zero or negative, where clipping invalidates the smooth approximation.
`boundary_coefficient` covers zero object or error variance; numerical range
failures are recorded separately. The point estimates retain the existing
zero-clipping policy. A negative facet main-effect component can therefore
withhold Phi while G remains available. A successful numerical calculation
does not establish nominal coverage, especially with few sampled facet levels.

D-study intervals describe uncertainty in the projected coefficient for the
same universe at the stated counts. They are pointwise intervals, not a
simultaneous band over searched plans or a prediction interval for a future
sample. Shading in the D-study figure shows these pointwise bounds; with two
random facets, shading is limited to the selected plan's count of the second
facet so that overlapping bands do not obscure all curves. Absent
bounds have their reasons in the tables. Fixed-facet slices/composites remain
conditional on those fixed levels. CSV and HTML retain these qualifications.
An updated reader is needed for projects containing this new interval record;
existing point-only archives remain readable without recalculation.

In the recorded normal-model check (3,000 replications per condition, every
true variance component equal to one), 60 objects x 12 levels of one random
facet and 80 objects x 12 x 10 levels of two random facets met the specified
return/coverage criteria for both coefficients at observed and doubled facet
counts. With 20 objects x 3 x 2 levels, Phi returned only 1,330/3,000 intervals;
1,311/1,330 contained the truth at observed counts. Its return and coverage
criteria were not met. With 20 objects x 3 levels, G was conservative and did
not meet the coverage criterion either. These results do not validate other
variance ratios, ordinal scores or nonnormal effects. The calculation is
available as an explicit approximation, not as an automatic coverage guarantee.

### Multivariate scores and a weighted composite (development version)

Use `score_cols` for two or more outcomes recorded on **the same complete
balanced crossed cells**, with one or two random facets. The score columns are
fixed outcome dimensions, not an additional sampled facet. Each selected score
must be finite on every retained cell. No pairwise deletion or silent averaging
of duplicate cells is performed. Different task/rater counts for different
scores, incomplete designs, and nested designs are outside this entry.

```python
# Reuse the balanced ratings example above; replace these illustrative columns
# with the actual outcomes recorded on each rating cell.
multi_ratings = ratings.assign(
    Content=ratings["Score"],
    Language=0.7*ratings["Score"] + 0.2*ratings["Rater"].astype(int),
)
multi = mfrmpy.analyze(
    multi_ratings,
    workflow="gd",
    gd_settings={
        "object_facet": "Person",
        "random_facets": ["Rater", "Task"],
        "score_cols": ["Content", "Language"],
        "composite_weights": {"Content": 0.6, "Language": 0.4},
        "selected_plan": {"Rater": 4, "Task": 2},
        "target_population": "Comparable persons, raters and tasks",
    },
)
print(mfrmpy.tables(multi, "gd_observed")[0][1])
print(mfrmpy.tables(multi, "gd_covariance")[0][1])
print(mfrmpy.tables(multi, "gd_component_diagnostics")[0][1])
mfrmpy.save_project("multivariate.mfrm", multi)
restored = mfrmpy.load_project("multivariate.mfrm")
mfrmpy.plot(restored, "gd")                  # specified composite
mfrmpy.plot(restored, "gd", level="Content") # retained individual outcome
```

Specify `score_col` for univariate analysis or `score_cols` for multivariate
analysis, not both. `composite_weights` is required and must name every selected
score, with finite nonnegative values and at least one positive weight. Weights
are used as supplied: `(1, 1)` is a sum and `(0.5, 0.5)` an average. Outcomes are
not standardized before combining them; units and ranges therefore matter.
Doubling every weight multiplies variance by four and SEM by two while leaving
G/Phi unchanged. A zero weight does not remove that score from input validation
or the individual-score results.

`gd_observed`, `gd_selected` and `gd_d_study` contain every individual score and
the specified composite, identified by `TargetType`, `Target` and `Weights`.
`gd_variance` shows their raw and used scalar components. `gd_covariance` stores
the signed component entries in original score-product units (diagonals are
variances). The highest-order interaction remains confounded with residual error.
For each component matrix `Sigma`, the composite raw variance is `w.T @ Sigma @ w`.
Negative scalar estimates are then set to zero for G/Phi, matching a univariate
analysis of the directly formed composite. Covariance entries are **not**
individually clipped, and component matrices are not silently made positive
semidefinite.

`gd_component_diagnostics` reports each matrix's minimum eigenvalue and rank
after scaling by outcome SDs, along with the numerical tolerance and
`PositiveSemidefinite` flag. A non-PSD estimate can arise from sampling variation;
it is retained for inspection rather than presented as a valid fitted joint
covariance. `ComponentPSD` and `ClippedComponents` also accompany the coefficient
rows. Scalar clipping for different targets does not create one corrected joint
covariance model. It should not be used to infer correlations or optimize weights
as if the component estimates were known population values.

`available=True` means the covariance decomposition was computed. A constant
score or a canceling composite can still have undefined G/Phi, recorded as
`undefined_zero_denominator`, while other outcomes remain available. SEM describes
measurement error, not the sampling SE of a reliability coefficient.

The existing fixed-facet slice/sum/mean option can be used when every selected
score satisfies the same elementary-cell checks; its interpretation stays
conditional on the fixed levels. Multivariate intervals are not yet supported:
combining `score_cols` with `coefficient_intervals` raises an explicit error.
The univariate interval coverage results do not qualify multivariate uncertainty.
CSV, HTML, plots and `.mfrm` retain the targets, weights and component diagnostics.
An updated reader is required for these new multivariate records; earlier
univariate archives remain readable without recomputation.

The weight convention follows the distinction between score units and nominal
weights in [Brennan's CASMA report on composite weights](https://education.uiowa.edu/sites/education.uiowa.edu/files/2026-04/casma-research-report-50-archived.pdf).

### Missing, unbalanced and nested designs (development version)

Use `method="minque0"` explicitly for missing cells, unequal observations or a
random facet nested in another random facet. The default ANOVA path and its
interval assumptions remain unchanged. For example, tasks can each have their
own raters. The same local rater label may occur in different tasks; it denotes
a different rater within each task when nesting is declared.

```python
import numpy as np
import pandas as pd
import mfrmpy

rng = np.random.default_rng(761)
person_effect = rng.normal(size=20)
task_effect = rng.normal(size=4)
rater_effect = rng.normal(size=(4, 3))
nested = pd.DataFrame([
    {"Person": p, "Task": t, "Rater": r,
     "Score": person_effect[p] + task_effect[t] + rater_effect[t, r] + rng.normal()}
    for p in range(20) for t in range(4) for r in range(2 + t % 2)
    if (p + t + r) % 9 != 0
])
nested.loc[1, "Score"] = np.nan
cfg = {
    "object_facet": "Person", "random_facets": ["Rater", "Task"],
    "score_col": "Score", "method": "minque0", "missing": "omit",
    "nesting": {"Rater": "Task"},
    "selected_plan": {"Rater": 3, "Task": 4},  # 3 raters per task, 4 tasks
    "target_population": "Comparable persons, tasks and task-specific raters",
}
preflight = mfrmpy.check_design(nested, workflow="gd", gd_settings=cfg)
analysis = mfrmpy.analyze(nested, workflow="gd", gd_settings=cfg)
print(mfrmpy.summary(analysis))
selected = mfrmpy.tables(analysis, "gd_selected")
audit = mfrmpy.tables(analysis, ["gd_estimation", "gd_component_support", "gd_excluded_rows"])
mfrmpy.save_project("nested.mfrm", analysis)
restored = mfrmpy.load_project("nested.mfrm")
```

Omit `nesting` for crossed facets. For multiple outcomes, replace `score_col` with
`score_cols` and explicit `composite_weights` as above; every selected outcome uses
the same retained rows. At most two non-object random facets are supported. The
object of measurement is crossed with the random design. Arbitrary nesting,
nesting within persons, repeated observations per cell, non-unit observation
weights and `fixed_condition` aggregation are not supported in this method.

Missing cells can be absent rows. Explicit missing values default to an unavailable
result (`missing="error"`); `"omit"` removes a row when any selected identifier or
score is missing, including a zero-weight outcome. It does not delete rows pairwise
or fill scores with zero. Infinite/nonnumeric scores and blank or ambiguous labels
remain errors, including on rows with another missing field. The original input is
retained. `gd_excluded_rows` lists zero-based `RowPosition`, the original index as
text, affected columns and the action (`omitted` or `rejected_missing`). Metadata
distinguishes input rows, rows with missing values and rows actually omitted.

MINQUE(0) fits an intercept and independent random effects with constant covariance
components across levels, conditional on the observed assignment. It solves
`H[i,j] = trace(P K[i] P K[j])` against `Y' P K[i] P Y`, where `P` removes the
intercept and each `K` indicates shared factor levels. The implementation uses
group counts and group score sums, not a dense observation-by-observation matrix.
The component system has at most seven rows. `gd_kernel_gram` and `gd_estimation`
retain that system, numerical rank, scaled eigenvalues, tolerance and condition
number. Insufficient rank returns `unidentified_components` without estimates.
This is a component-identification check under the model, not a guarantee of
precision or a correction for informative missingness. See [Rao's original MINQUE
theory](https://www.sciencedirect.com/science/article/pii/0047259X71900017).

`gd_covariance`, `gd_variance` and `gd_component_diagnostics` retain the raw
estimates and non-PSD diagnostics. Each outcome or composite uses `w' Sigma w`
before negative scalar components are clipped for G/Phi. Clipping changes the raw
estimator; it does not preserve unbiasedness or produce a common repaired covariance
model. The highest-order interaction is combined with residual error. ANOVA degrees
of freedom and sampling SEs are not assigned to MINQUE(0) component estimates.

For an unbalanced observed design, `gd_observed` retains
`GStatus=PhiStatus="no_balanced_observed_design"`, with the reason. Counts of distinct
observed raters/tasks are not automatically reused as replications. You can omit
`selected_plan` to obtain components only; no D table is generated and the figure
explains that a plan is required. An explicit plan projects a **future balanced
mean**, with common counts for all outcomes. For nested `Rater:Task`, the counts
mean raters **per task** and number of tasks. Under the same random-effect model,
relative error is `Var(Person:Task)/n_task + Var(Residual)/(n_task*n_rater)`;
absolute error additionally includes `Var(Task)/n_task + Var(Rater(Task))/(n_task*n_rater)`.
These projections do not estimate reliability of the unequal observed averages.
Complete balanced observations may also use MINQUE(0); crossed results agree
with the existing ANOVA point estimates.

MINQUE(0) coefficient intervals are unsupported, including for a single outcome;
`coefficient_intervals` is rejected. Missingness assumptions, unbiasedness of raw
components under the model, numerical identification and interval coverage are
separate questions. The validation checks point calculations against mfrmr 0.3.2
and independent small dense-kernel calculations; it does not qualify coverage or
claim efficiency over other estimators. Input data, omission, holds, components
and plan results survive save/load without re-estimation; new files require an
updated reader. CSV/HTML and plots use these retained results.

## Results, plots and saved projects

### Plain-text reports and ZIP bundles (development version)

For reading and sharing results, use a UTF-8 `.out` (or `.out.txt`) report and a
ZIP of tables and plots. A `.mfrm` project is optional and serves a different
purpose: reopening the retained fit without estimation. Text and ZIP reports
cannot be loaded as projects. These are mfrmpy reports, not FACETS output syntax,
input specifications, or a claim of identical estimators. FACETS also describes
a [text Report Output File](https://www.winsteps.com/facetman64/reportoutputfile.htm).

```python
mfrmpy.export_out(analysis, "writing.out")
mfrmpy.export_bundle(analysis, "writing_results.zip", plots=[
    "rater_severity",
    {"kind": "fair_average", "facet": "Rater", "statistic": "FairM"},
])
# Optional: retain the analysis for reopening later.
mfrmpy.save_project("writing.mfrm", analysis)
```

The ZIP contains `report.out`, `tables/*.csv`, a CSV provenance manifest,
`plots/*.png` and `plots/*.svg`, and `manifest.json` with the analysis ID, content
fingerprint, file SHA-256 digests, requested plot options and figure captions.
It does not contain the `.mfrm` project or a separate raw-data file; the output
tables can still contain work IDs and retained rating details. Files are written
to temporary storage and the destination is replaced only after success. Parent
directories must already exist. ZIP contents are written incrementally to disk;
the analysis and selected tables still reside in memory.

`plots=None` (the default) exports the available standard report figures,
including each PCM step-owner's Wright map and category curves. A list selects
specific plots; PCM requests can specify `level`. `plots=[]` omits figures and
needs no Matplotlib. The default figures use their usual row limits, including
`top_n=40`; pass a larger `top_n` when needed. The text and CSV tables include all
selected rows and columns. Figures retain their captions, plotted scope and
holds. A failed plot aborts the ZIP rather than silently omitting that output.

Use `kind="fair_average"`, `kind="rater_dff"` or a list of table kinds to limit
text/CSV construction before export. Plot selection is separate from `kind`.
`report_text(analysis, kind=...)` returns the same text as a string. No export
performs a fit or automatically fills pending diagnostics; a partial fit's text
lists its computed and pending outputs.

### Fair Average and DFF: what is available

```python
title, fair, note = mfrmpy.tables(analysis, "fair_average")[0]
print(fair.loc[fair.Facet.eq("Rater")])
print(note)
```

Fair Average is an expected rating on the original score scale. `FairM` evaluates
the fitted level at the unweighted means of the other fitted locations; for a
nonperson level this includes the fitted person mean. `FairZ` sets those other
locations to zero. Neither is the average of predictions across actual partners:
the expectation is nonlinear. PCM uses a target step-owner's own thresholds and
mean thresholds for other targets. The implemented reference convention is
described explicitly; numerical identity with FACETS for all designs and extreme
scores has not been established. See [FACETS' definition and cautions](https://www.winsteps.com/facetman64/fairaverage.htm).

The table retains observed means, both predictions, their availability states
and hold reasons. Free endpoint levels are withheld. An unbounded JMLE person
mean makes nonperson FairM unavailable; FairZ may still be available and is
labelled as a different reference. Calibration blocks withhold predictions in
the table and plot. For a partial fit, request `diagnose(analysis,
outputs="measures")` first. Fair Average does not establish that scoring is fair.
No fair-score SE/CI or explicit weighted reference-plan prediction is implemented;
a logit measure SE is not a fair-score SE. The reference-plan/interval separation
in mfrmr 0.3.2 was consulted, not imported as a Python capability.

DFF already has a [prespecified two-group contrast workflow](#estimate-a-prespecified-two-school-rater-contrast),
including observed shared works, score-category use, a plot and retained holds.
Use `tables(analysis, "rater_dff")` or select that kind in the exports. It currently
requires an unanchored RSM/MML with fixed positive population SD,
and a negative composite group-by-rater facet. Other rubric endpoints,
category counts and additional facets are computable subject to assignment
and numerical checks; the original coverage experiments used five-category
1–5 responses, SD 1 and a common task facet. Its model-based Wald
interval is distinct from validated coverage: `CoverageQualified=False` and
`SupportsFormalInference=False` remain explicit. Conditional `bias_requests`
cell diagnostics are not a substitute for this ordered difference of rater
differences. Missing observed overlap cannot be supplied by a planned roster.

### Public output API

The stable public surface is the names in `mfrmpy.__all__`. The bundled
`mfrmpy._core` modules are implementation details. The distribution excludes
the legacy `mfrm_app` namespace and unused application/research modules.
Use the documented API; retained research scripts in the checkout can use
source-only compatibility bridges. `mfrmpy.types` provides request/envelope
types and the distribution includes `py.typed` and public function stubs.

| Function | Result |
|---|---|
| `check_design(data, **settings)` | Optimization-free preflight dict checking MFRM connectivity / category support or G/D balance |
| `analyze(data, **settings)` | Retained analysis dictionary, including inputs, settings, identities and qualification |
| `summary(analysis)` | One-screen DataFrame of design counts, qualification state, hold reasons and available table/plot kinds |
| `tables(analysis, kind=None)` | List of `(title, pandas.DataFrame, note)` tuples (optionally filtered by `kind`, e.g. `"measures"`, `"steps"`/`"thresholds"`, `"reliability"`, `"agreement"`, `"bias"`, `"intervals"`, `"joint"`, `"anchor_sensitivity"`, `"gd"`) |
| `export_csv(analysis, directory, kind=None)` | Atomically write selected qualified CSV tables and `00_manifest.csv` (with current and saved reasons) |
| `report_text(analysis, kind=None)` | Plain-text string of selected qualified tables, notes and availability states; no optional dependency |
| `export_out(analysis, path, kind=None)` | Atomically write the plain-text report, typically `.out` or `.out.txt` |
| `export_bundle(analysis, path, kind=None, plots=None)` | ZIP of the text report, CSVs, PNG/SVG figures and provenance manifest; `plots=[]` needs no plotting extra |
| `load_example()` | A new DataFrame containing the bundled synthetic writing example |
| `save_project(path, analysis)` | Atomically write a validated `.mfrm` archive |
| `load_project(path)` | Validate and restore an archive without refitting; malformed archives raise `ProjectError` |
| `plot(analysis, kind="wright", **options)` | Matplotlib Figure; requires `figures` extra |
| `report_html(analysis, **options)` | Standalone HTML string (`facet`/`statistic` or `pathway_facet`/`pathway_statistic`); requires `figures` extra |

`analysis["result"]` and `analysis["diagnostics"]` retain raw MFRM output;
`analysis["gd_analysis"]` retains G/D results. The nested implementation
schema is internal; use `.mfrm` files to retain analyses across sessions rather
than serializing or modifying this dictionary yourself. For reporting, use
`tables` or `export_csv`, keeping the notes and `SEUse`/`CIUse`/`Use`/`StatusReason`
fields:

- **Distinct run qualification states:** `stat.jmle.identifiability_audit_size_limit`
  (rank audit still exceeds its size limit after person elimination), `stat.jmle.eta_rank_deficient`
  (proven structural rank deficiency / disconnected subsets), and
  `stat.estimation.optimizer_termination_unverified` (non-convergence) are reported
  separately rather than merged under one hold code.
- **Exploratory SEs and intervals (`screening_only` / `SupportsFormalInference = False`):**
  Converged JMLE (including size-limited dense audits without proven disconnection)
  and converged fixed-SD MML display finite `ModelSE`, `RealSE = ModelSE * sqrt(max(Infit, 1))`
  and exploratory normal bands with explicit `Basis`, `SEUse = "screening_only"`,
  `CIUse = "screening_only"` and `SupportsFormalInference = False`. Non-converged,
  structurally rank-deficient, free-SD MML, fixed-constraint and extreme-endpoint
  rows withhold SEs and intervals (`""`, not zero).
- **Four-series reliability and counts:** `Separation and Reliability by Facet`
  reports all four combinations of `SEMode` (`model` vs `fit_adjusted`) and
  `DistributionBasis` (`sample` ddof=1 vs `population` ddof=0), alongside `Levels`,
  `EstimateAvailable`, `SEAvailable`, `ExcludedLevels`, `AggregationTarget`,
  `Use` and `SupportsFormalInference = False`.
  Rater separation describes how distinguishable the fitted severity levels
  are relative to their SEs. Strata, `(4 * Separation + 1) / 3`, is a
  continuous conversion of that index. Use the agreement table to describe
  matching observed ratings. Dependence between ratings of the same work can
  make person separation and empirical EAP reliability look higher than the
  accuracy obtained on such data.
- **Observed-score agreement:** `tables(analysis, "agreement")` reports exact
  and adjacent agreement, model-expected exact agreement, ordinal, nominal
  and interval Krippendorff's alpha, and a table of shared contexts and score
  differences for each element pair. Alpha uses the pooled score distribution
  and only contexts with two or more distinct elements (`PairableUnits`).
  Model-expected agreement uses fitted probabilities conditional on the work
  measure under independent scoring and is subject to calibration holds.
  Distinguish separate scoring occasions in the design if ratings would
  otherwise be averaged within a context by element. For a five-category
  example, including pairwise Cohen's kappa calculations and the effect of
  concentrated scores, see the [agreement validation](../validation/m5_rater_agreement.md).
  For partially double-rated data, report the rule used to select the
  double-rated works, the number eligible and double-rated, and score counts
  for both samples. The selection rule can change alpha and kappa substantially
  even when the number of matched ratings and the exact agreement are similar.
- **Fit and boundary safeguards (`FitStatus` / `FitReason` / `ObservedCategories`):**
  Following `mfrmr` 0.3.0, ZSTD is withheld (`""`) when degrees of freedom are
  below 1 (where the Wilson–Hilferty approximation is unstable) or when a scope includes
  JMLE extreme-score boundary proxies; single-observation free JMLE levels
  withhold zero-d.f. mean squares (`unavailable_zero_df`); unbounded endpoint
  facet levels, zero-count internal/boundary score categories (`stat.category.*`),
  and disconnected MML cohorts (`stat.mml.population_assumption_linked`) withhold
  unqualified estimates, SEs, specified intervals, joint interactions and bias;
  and `ObservedCategories`, agreement notes and G/D metadata flag constant-score
  levels explicitly.
- **Shared re-display and saved provenance:** `tables`, `plot`, `report_html` and
  `export_csv` apply the same qualification rules to fresh and reopened `.mfrm`
  projects while retaining original archived values and save-time reasons (`Saved*`
  columns, `00_manifest.csv` and HTML archive notes).
- **Cross-package numerical alignment (`mfrmr` 0.3.0 / `TAM` / `ConQuest`):** Fixed-SD
  MML defaults to `quad_points=31`. Each requested interval or joint interaction has its own gradient and quadrature sensitivity checks at the fitted point. A converged fit can still have withheld intervals, including in the bundled writing example. Earlier archives retain their fitted quadrature count. `analyze` /
  `check_design` accept `noncenter_facet` (e.g. `noncenter_facet="Task"` to match
  `TAM` `constraint="cases"` and `ConQuest` `lconstraints=cases`) and
  `positive_facets`. `Step Thresholds` (`tables(analysis, "steps")`) reports centered
  adjacent-category thresholds, observed transition counts, disorder flags, and
  exploratory `screening_only` SEs/bands (full observed-information delta-method SEs
  including the sum-to-zero final step for MML; analytical conditional step-information
  SEs for JMLE). `Specified intervals` includes explicit `SupportsFormalInference = False`
  and `Reason` columns, and conditional `Bias` (`tables(analysis, "bias")`) reports
  `ObsScore`, `ExpScore`, `ObsExpAvg`, `BiasSize`, `SE`, `Z`, `LR_ChiSq`, and 1-d.f.
  conditional likelihood-ratio `LR_Prob` under `Use = "screening_only"`
  (`SupportsFormalInference = False`), withholding uncertainty under calibration or
  cell holds. Fixed-SD MML facet SEs propagate full joint observed-information
  covariance `sqrt(diag(J I^{-1} J^T))` across all levels including constrained
  sum-to-zero levels (matching `mfrmr` 0.3.0, whereas `TAM::tam.se()` and `ConQuest`
  `stderr=quick` compute diagonal curvature `1 / sqrt(diag(I))`, and `ConQuest`
  5.47.5 `stderr=empirical` matches on free coordinates but repeats the preceding SE
  on `*`-constrained levels); and MML Person reliability uses empirical EAP variance
  `Var(EAP) / (Var(EAP) + mean(SD^2))` (matching `TAM::EAP.rel` under
  `DistributionBasis = "population"`).

With the optional figures dependency installed:

```python
from pathlib import Path

step_facet = analysis["fit_settings"].get("step_facet")
level = str(data[step_facet].iloc[0]) if step_facet else None
figure = mfrmpy.plot(analysis, "wright", level=level)
figure.savefig("wright.png", dpi=300)
Path("report.html").write_text(mfrmpy.report_html(analysis), encoding="utf-8")
```

Other figure kinds include:
- Core maps & curves: `"pathway"` (alias `"fit_pathway"`; `statistic="Infit"|"Outfit"|"InfitZSTD"|"OutfitZSTD"`), `"category"` (alias `"ccc"`), `"expected"` (alias `"tcc"`), and `"information"` (category Fisher information and conditional SE).
- Facet & rubric profiles: `"facet"`, `"rater_severity"`, `"step"` (alias `"thresholds"`), `"bubble"`, and `"fair_average"` (`statistic="FairM"|"FairZ"`).
- Residual & supplementary diagnostics: `"unexpected"`, `"residual_qq"`, `"pca"` (alias `"residual_pca"`), `"pca_loadings"`, `"bias"`, `"bias_ranked"`, `"interval"`, `"joint"`, and `"gd"`.

PCM Wright maps and category curves require `level` (a step-facet level), while `"step"`, `"expected"`, and `"information"` accept `level` for a single step-facet level or `level=None` to overlay all step-facet levels; supplementary requests use their
zero-based `index`. Retained results are reused; plotting and reopening must
not recompute estimation. Unsupported requests can produce a reason-bearing
figure rather than a numerical result.

Saved `.mfrm` projects use the existing desktop/Web archive schema with
deterministic outer gzip compression (`\x1f\x8b`) by default; `load_project`
automatically reads both gzip-compressed and earlier uncompressed `.mfrm`
archives without changing `content_fingerprint` (up to 128 MiB on disk and
512 MiB uncompressed). Older `0.1.0` releases read uncompressed archives only.
New source or dependency versions can cause current qualification checks to
withhold old interval output without altering the archived estimates. Record
`mfrmpy.__version__` and `analysis["source"]` when reporting an analysis.
The package API version and the shared engine version are distinct.

## API and compatibility policy

The supported API is the exported functions documented above, `ProjectError`
and `__version__`. `analyze` accepts the settings and workflows documented in
this guide. `ProjectError` is a `ValueError` subclass for invalid/unsupported
project contents; filesystem errors such as a missing file remain normal
Python I/O errors.

Within **0.2.x**, patch releases preserve these public names, documented
arguments and return types. Additive options and bug fixes are allowed. Before
1.0, an incompatible public API change requires a new minor release, such as
0.3.0, with migration instructions in the changelog. Private `mfrmpy._core` helpers,
nested raw-result fields, table display order and exact wording are outside
this compatibility promise. Keep interpretation notes with exported results.

Version 0.2.1 reads the supported schema-1 RSM/JMLE and schema-2 `.mfrm` archives,
including those written by 0.1.0a2 and the corresponding Web/desktop workflows.
The 0.2.x series retains read compatibility with these supported archives;
malformed or unsupported records are rejected without refitting. Future archive
changes will be documented, and older applications may not read newer formats.

Numerical bug fixes or updated qualification rules can change reported results
or withhold an output; such changes belong in the changelog. For reproducible
research, retain the original `.mfrm` file and record the package, source and
dependency versions. Reopening preserves archived estimates and applies current reporting rules.
When a legacy fit used different score categories or different `score_k` facet IDs than the retained input, loading warns and `summary()` explains how to refit. Numerical tables, figures and exports from such a fit require a new analysis.
Stored specified intervals and joint interactions verify their numerical source and runtime; a mismatch withholds the corresponding output. For an older unavailable G/D result with zero object variance, rerun the G/D study to obtain the corrected coefficients.

Cell density counts unique person-by-facet cells. Repeated ratings count toward the total number of ratings. A size-limited rank check reports unknown nullity. In designs with disjoint rater pools, fixing one rater anchors only estimable levels; a group mean supplies one constraint. Inspect the status of the requested level or contrast.

For a CSV export, read `00_manifest.csv` and resolve each `File` relative to its directory. The manifest identifies the analysis, archived content, exporting version, and SHA-256 of each table. Re-export switches the manifest after all new tables have been written. Files added by the user are preserved.

## Privacy, license and citation

Calculations run locally. Saved projects and exported reports can retain
ratings and person/rater IDs; store and share them according to your research
requirements. Use synthetic data for bug reports.

MIT licensed; see
[LICENSE](https://github.com/Ryuya-dot-com/mfrmpy/blob/main/LICENSE)
and the
[responsible-use notice](https://github.com/Ryuya-dot-com/mfrmpy/blob/main/LICENSE_NOTICE.md).
Cite Ryuya Komuro, *mfrmpy*, the installed package version, and
the statistical methods used. See the [Python source](https://github.com/Ryuya-dot-com/mfrmpy) and [package issue tracker](https://github.com/Ryuya-dot-com/mfrmpy/issues).
