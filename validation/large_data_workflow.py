"""Measure fit, audit and selected output phases; no performance assertions.

Run from the checkout with PYTHONPATH=. Each scenario uses a fresh process.
The binary crossed data exercise size scaling, not interval coverage.
"""
import argparse
import hashlib
import json
import platform
from pathlib import Path
import resource
import subprocess
import sys
import tempfile
import time
import tracemalloc

import numpy as np
import pandas as pd

import mfrmpy


def scenario(persons, recheck=False):
    data = pd.DataFrame({
        "Person": np.repeat(np.arange(persons).astype(str), 2),
        "Rater": np.tile(["A", "B"], persons),
        "Score": (np.repeat(np.arange(persons), 2) + np.tile([0, 1], persons)) % 2,
    })
    phases = []
    def measure(name, operation):
        if recheck and name not in {"load_fit", "one_table"}:
            return operation()
        tracemalloc.start()
        start = time.perf_counter()
        result = operation()
        seconds = time.perf_counter() - start
        _, allocated_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        process_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform != "darwin":
            process_peak *= 1024
        phases.append(dict(phase=name, seconds=seconds, traced_peak_mib=allocated_peak / 2**20,
                           process_peak_mib=process_peak / 2**20))
        return result
    fitted = measure("fit", lambda: mfrmpy.fit(
        data, person="Person", facets=["Rater"], score="Score", rating_min=0, rating_max=1))
    contrast = measure("contrast_audit", lambda: mfrmpy.check_rater_contrast(
        data, person="Person", facets=["Rater"], score="Score", rater="Rater", difference=["A", "B"]))
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "fit.mfrm"
        measure("save_fit", lambda: mfrmpy.save_project(path, fitted))
        restored = measure("load_fit", lambda: mfrmpy.load_project(path))
        fit_bytes = path.stat().st_size
        np.testing.assert_array_equal(restored["result"]["params"]["theta"], fitted["result"]["params"]["theta"])
        del fitted
        result = measure("measures", lambda: mfrmpy.diagnose(restored, outputs="measures"))
        measure("one_table", lambda: mfrmpy.tables(result, "measures"))
        if not recheck:
            result = measure("agreement", lambda: mfrmpy.diagnose(result, outputs="agreement"))
            result = measure("scorefile", lambda: mfrmpy.diagnose(result, outputs="scorefile"))
    audit = result["result"]["identifiability"]["summary"].iloc[0]
    return dict(persons=persons, ratings=len(data), fit_archive_bytes=fit_bytes,
                optimizer_success=bool(result["result"]["opt"].success),
                audit_status=audit.Status, contrast_status=contrast["status"],
                dense_design_cells=int(audit.EtaDesignRows * audit.EtaDesignColumns),
                reduced_cells=int(audit.ReducedRows * audit.ReducedColumns), phases=phases)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--persons", type=int)
    parser.add_argument("--recheck", action="store_true", help="Measure only changed file-read and single-table paths")
    parser.add_argument("--output", type=Path, default=Path("validation/evidence/large_data_workflow.json"))
    args = parser.parse_args()
    if args.persons:
        print(json.dumps(scenario(args.persons, args.recheck)))
    else:
        runs = []
        for persons in (2000, 10000):
            command = [sys.executable, __file__, "--persons", str(persons)]
            if args.recheck:
                command.append("--recheck")
            completed = subprocess.run(command,
                                       check=True, capture_output=True, text=True)
            runs.append(json.loads(completed.stdout))
            print(json.dumps(runs[-1]), flush=True)
        paths = [Path("mfrm_app") / name for name in (
            "basic_analysis.py", "engine.py", "design_rank.py", "rater_contrast.py",
            "output_qualification.py", "project.py", "desktop_analysis.py")]
        paths += [Path("mfrmpy/__init__.py"), Path(__file__)]
        evidence = dict(python=sys.version, platform=platform.platform(), numpy=np.__version__, pandas=pd.__version__,
                        measurement="Tracemalloc peak per phase; process RSS peak cumulative within each fresh scenario.",
                        source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}, runs=runs)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(evidence, indent=2) + "\n")
