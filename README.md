# mfrmpy

Many-facet Rasch analysis (RSM / PCM with JMLE or fixed-population-SD MML) and observed-score generalizability / decision (G/D) studies in Python. The development version adds explicit MINQUE(0) for supported missing, unbalanced and nested designs; see the [scope and example](docs/python_library.md#missing-unbalanced-and-nested-designs-development-version).

This checkout is the unpublished **0.2.3.dev3** development candidate. Its version
distinguishes the new features from the earlier 0.2.1 development snapshots and
the 0.2.2 release recorded in [the release history](validation/m1_release_candidate.md).
See the [local installation instructions](docs/python_library.md#installation).

This is the Python package repository. Report library problems in [mfrmpy issues](https://github.com/Ryuya-dot-com/mfrmpy/issues). The Web and Desktop repository maintains the applications and shares their calculation engine with this package.

- **PyPI releases:** [mfrmpy](https://pypi.org/project/mfrmpy/)
- **Package documentation:** [docs/python_library.md](docs/python_library.md)
- **Statistical scope and interpretation boundaries:** [docs/analysis_scope.md](docs/analysis_scope.md)
- [Python package development roadmap (日本語)](ROADMAP.md)
- [Save a fit before computing diagnostics (development version)](docs/python_library.md#save-the-fit-before-diagnostics-development-version)
- **Changelog:** [CHANGELOG.md](CHANGELOG.md)
- **Web & Desktop applications:** [MFRM_Python_Application](https://github.com/Ryuya-dot-com/MFRM_Python_Application)

## Installation

```bash
python -m pip install mfrmpy
# Optional Matplotlib figures and offline HTML reports:
python -m pip install "mfrmpy[figures]"
```

## Quick start

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
    method="MML",
)

print(mfrmpy.summary(analysis))
for title, frame, note in mfrmpy.tables(analysis, kind=("measures", "steps")):
    print(title)
    print(frame.head())

mfrmpy.export_out(analysis, "writing.out")  # readable UTF-8 report
mfrmpy.export_csv(analysis, "results_csv")
# With the figures extra: text + CSV + PNG/SVG in one ZIP.
mfrmpy.export_bundle(analysis, "writing_results.zip", plots=["rater_severity", "fair_average"])
# Optional archive for resuming analysis:
mfrmpy.save_project("writing.mfrm", analysis)
restored = mfrmpy.load_project("writing.mfrm")  # zero refitting
```

## License

MIT License — see [LICENSE](LICENSE) and [LICENSE_NOTICE.md](LICENSE_NOTICE.md).
