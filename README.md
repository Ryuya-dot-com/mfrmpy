# mfrmpy

Many-facet Rasch analysis (RSM / PCM with JMLE or fixed-population-SD MML) and balanced observed-score generalizability / decision (G/D) studies in Python.

- **PyPI package (`0.2.0`):** [pypi.org/project/mfrmpy/0.2.0](https://pypi.org/project/mfrmpy/0.2.0/)
- **Package documentation:** [docs/python_library.md](docs/python_library.md)
- **Statistical scope and interpretation boundaries:** [docs/analysis_scope.md](docs/analysis_scope.md)
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

mfrmpy.export_csv(analysis, "results_csv")
mfrmpy.save_project("writing.mfrm", analysis)
restored = mfrmpy.load_project("writing.mfrm")  # zero refitting
```

## License

MIT License — see [LICENSE](LICENSE) and [LICENSE_NOTICE.md](LICENSE_NOTICE.md).
