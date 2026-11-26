"""Versioned basic-analysis files. Only explicitly supported data types are read.

This is a result archive, not an optimizer checkpoint or an executable object
format. JSON size and array shapes are checked before constructing arrays.
"""
from __future__ import annotations

import base64
from collections import OrderedDict
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
import tempfile
import warnings
import zlib

import numpy as np
import pandas as pd
from scipy.optimize import OptimizeResult

from mfrmpy._core import engine

FORMAT = "mfrm.basic-project"
SCHEMA_VERSION = 3
MAX_PROJECT_BYTES = 128 * 1024 * 1024
MAX_UNCOMPRESSED_PROJECT_BYTES = 512 * 1024 * 1024
GZIP_MAGIC = b"\x1f\x8b"


class ProjectError(ValueError):
    """The archive is unsupported, incomplete, changed, or malformed."""


def _dtype_encode(dtype):
    if isinstance(dtype, pd.CategoricalDtype):
        return ["category", _encode(dtype.categories), dtype.ordered]
    if isinstance(dtype, pd.StringDtype):
        return ["string", dtype.storage, _encode(dtype.na_value)]
    if isinstance(dtype, np.dtype):
        if dtype.fields or dtype.subdtype:
            raise ProjectError("Structured array dtypes are not supported in project files")
        return ["numpy", dtype.str]
    if str(dtype) in {"boolean", "Int8", "Int16", "Int32", "Int64", "UInt8", "UInt16", "UInt32", "UInt64", "Float32", "Float64"}:
        return ["nullable", str(dtype)]
    raise ProjectError(f"Unsupported dtype: {dtype}")


def _dtype_decode(spec):
    kind = spec[0]
    if kind == "category":
        return pd.CategoricalDtype(_decode(spec[1]), ordered=spec[2])
    if kind == "string" and spec[1] in {"python", "pyarrow"}:
        missing = _decode(spec[2])
        try:
            return pd.StringDtype(storage=spec[1], na_value=missing)
        except TypeError:
            if missing is not pd.NA:
                raise ProjectError("This archive uses NaN-string semantics requiring pandas >= 3.0") from None
            return pd.StringDtype(storage=spec[1])
    if kind == "nullable" and spec[1] in {"boolean", "Int8", "Int16", "Int32", "Int64", "UInt8", "UInt16", "UInt32", "UInt64", "Float32", "Float64"}:
        return pd.api.types.pandas_dtype(spec[1])
    if kind == "numpy" and isinstance(spec[1], str):
        dtype = np.dtype(spec[1])
        if not dtype.fields and not dtype.subdtype and dtype.kind in "biufUSOmM" and 0 < dtype.itemsize <= MAX_PROJECT_BYTES:
            return dtype
    raise ProjectError("Unsupported stored dtype")


def _encode(value):
    if value is pd.NA:
        return ["NA"]
    if value is pd.NaT:
        return ["NaT"]
    if value is None or type(value) in (str, int, bool):
        return value
    if type(value) is float:
        return ["float", value.hex()]
    if isinstance(value, np.ndarray):
        data = (_encode(value.ravel().tolist()) if value.dtype.hasobject
                else base64.b64encode(value.tobytes(order="C")).decode("ascii"))
        return ["array", _dtype_encode(value.dtype), list(value.shape), data]
    if isinstance(value, np.generic):
        return ["scalar", _encode(np.asarray(value))]
    if isinstance(value, pd.RangeIndex):
        return ["range", value.start, value.stop, value.step, _encode(value.name)]
    if isinstance(value, pd.MultiIndex):
        return ["multi-index", _encode(list(value.levels)), _encode(list(value.codes)), _encode(list(value.names))]
    if isinstance(value, pd.Index):
        return ["index", _dtype_encode(value.dtype), _encode(value.tolist()), _encode(value.name)]
    if isinstance(value, pd.Series):
        return ["series", _dtype_encode(value.dtype), _encode(value.tolist()), _encode(value.index), _encode(value.name), _encode(value.attrs)]
    if isinstance(value, pd.DataFrame):
        return ["frame", _encode(value.index), _encode(value.columns),
                [_encode(value.iloc[:, i]) for i in range(value.shape[1])], _encode(value.attrs)]
    if isinstance(value, OptimizeResult):
        # The inverse-Hessian operator is optimizer machinery, not a reported
        # uncertainty estimate. It is deliberately absent from result archives.
        return ["optimizer", _encode({k: v for k, v in value.items() if k != "hess_inv"})]
    if type(value) in (dict, OrderedDict):
        tag = "ordered-dict" if type(value) is OrderedDict else "dict"
        return [tag, [[_encode(k), _encode(v)] for k, v in value.items()]]
    if type(value) in (list, tuple):
        return [type(value).__name__, [_encode(v) for v in value]]
    if type(value) is slice:
        parts = (value.start, value.stop, value.step)
        if any(x is not None and type(x) is not int for x in parts) or value.step == 0:
            raise ProjectError("Only integer parameter slices are supported")
        return ["slice", *parts]
    if isinstance(value, pd.Timestamp):
        return ["timestamp", value.isoformat()]
    raise ProjectError(f"Unsupported project value: {type(value).__name__}")


def _decode(value, depth=0):
    if depth > 80:
        raise ProjectError("Project nesting exceeds project limit")
    if value is None or type(value) in (str, int, bool):
        return value
    if not isinstance(value, list) or not value or not isinstance(value[0], str):
        raise ProjectError("Expected a typed project value")
    tag, *parts = value
    decode = lambda item: _decode(item, depth + 1)
    if tag == "NA" and not parts:
        return pd.NA
    if tag == "NaT" and not parts:
        return pd.NaT
    if tag == "float" and len(parts) == 1:
        return float.fromhex(parts[0])
    if tag == "array" and len(parts) == 3:
        dtype, shape, data = _dtype_decode(parts[0]), parts[1], parts[2]
        if (not isinstance(shape, list) or len(shape) > 32
                or any(type(n) is not int or n < 0 or n > MAX_PROJECT_BYTES for n in shape)):
            raise ProjectError("Invalid array shape")
        size = math.prod(shape)
        if size * dtype.itemsize > MAX_PROJECT_BYTES:
            raise ProjectError("Stored array exceeds project limit")
        if dtype.hasobject:
            items = decode(data)
            if not isinstance(items, list) or len(items) != size:
                raise ProjectError("Array content/shape mismatch")
            array = np.empty(size, dtype=object)
            array[:] = items
        else:
            raw = base64.b64decode(data, validate=True)
            if len(raw) != size * dtype.itemsize:
                raise ProjectError("Array content/shape mismatch")
            array = np.frombuffer(raw, dtype=dtype).copy()
        return array.reshape(shape)
    if tag == "scalar" and len(parts) == 1:
        array = decode(parts[0])
        if not isinstance(array, np.ndarray) or array.shape != ():
            raise ProjectError("Invalid scalar")
        return array[()]
    if tag == "range" and len(parts) == 4:
        index = pd.RangeIndex(*parts[:3], name=decode(parts[3]))
        if len(index) > MAX_PROJECT_BYTES:
            raise ProjectError("Index exceeds project limit")
        return index
    if tag == "multi-index" and len(parts) == 3:
        return pd.MultiIndex(levels=decode(parts[0]), codes=decode(parts[1]), names=decode(parts[2]), verify_integrity=True)
    if tag == "index" and len(parts) == 3:
        return pd.Index(decode(parts[1]), dtype=_dtype_decode(parts[0]), name=decode(parts[2]), tupleize_cols=False)
    if tag == "series" and len(parts) == 5:
        items, dtype, index = decode(parts[1]), _dtype_decode(parts[0]), decode(parts[2])
        if not isinstance(items, list) or len(items) != len(index):
            raise ProjectError("Series axes/content mismatch")
        if len(items) * getattr(dtype, "itemsize", 8) > MAX_PROJECT_BYTES:
            raise ProjectError("Series exceeds project limit")
        series = pd.Series(items, dtype=dtype, index=index, name=decode(parts[3]))
        series.attrs = decode(parts[4])
        return series
    if tag == "frame" and len(parts) == 4:
        index, columns = decode(parts[0]), decode(parts[1])
        series = [decode(part) for part in parts[2]]
        if len(series) != len(columns) or any(not s.index.equals(index) for s in series):
            raise ProjectError("Table axes/content mismatch")
        frame = pd.concat(series, axis=1) if series else pd.DataFrame(index=index)
        frame.columns = columns
        frame.attrs = decode(parts[3])
        return frame
    if tag in {"list", "tuple"} and len(parts) == 1:
        items = [decode(item) for item in parts[0]]
        return tuple(items) if tag == "tuple" else items
    if tag == "slice" and len(parts) == 3:
        if any(x is not None and type(x) is not int for x in parts) or parts[2] == 0:
            raise ProjectError("Invalid parameter slice")
        return slice(*parts)
    if tag in {"dict", "ordered-dict"} and len(parts) == 1:
        result = OrderedDict() if tag == "ordered-dict" else {}
        for key, item in parts[0]:
            key = decode(key)
            if key in result:
                raise ProjectError("Duplicate mapping key")
            result[key] = decode(item)
        return result
    if tag == "optimizer" and len(parts) == 1:
        return OptimizeResult(decode(parts[0]))
    if tag == "timestamp" and len(parts) == 1:
        return pd.Timestamp(parts[0])
    raise ProjectError(f"Unsupported project tag: {tag}")


def _json_bytes(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")


def content_fingerprint(analysis):
    digest = hashlib.sha256()
    for chunk in _encoded_chunks({k: v for k, v in analysis.items() if k != "content_fingerprint"}):
        digest.update(chunk)
    return digest.hexdigest()


def _encoded_chunks(value):
    """Emit the existing typed JSON, retaining at most one encoded column."""
    if isinstance(value, OptimizeResult):
        yield b'["optimizer",'
        yield from _encoded_chunks({k: v for k, v in value.items() if k != "hess_inv"})
        yield b']'
    elif type(value) in (dict, OrderedDict):
        yield b'["ordered-dict",[' if type(value) is OrderedDict else b'["dict",['
        for i, (key, item) in enumerate(value.items()):
            yield b',' if i else b''
            yield b'[' + _json_bytes(_encode(key)) + b','
            yield from _encoded_chunks(item)
            yield b']'
        yield b']]'
    elif isinstance(value, pd.DataFrame):
        yield b'["frame",' + _json_bytes(_encode(value.index)) + b',' + _json_bytes(_encode(value.columns)) + b',['
        for i in range(value.shape[1]):
            yield b',' if i else b''
            yield _json_bytes(_encode(value.iloc[:, i]))
        yield b'],' + _json_bytes(_encode(value.attrs)) + b']'
    elif type(value) in (list, tuple):
        yield b'["list",[' if type(value) is list else b'["tuple",['
        for i, item in enumerate(value):
            yield b',' if i else b''
            yield from _encoded_chunks(item)
        yield b']]'
    else:
        yield _json_bytes(_encode(value))


def gd_analysis_identity(data, settings, calculation_version):
    return "gd_" + engine.stable_json_fingerprint(dict(
        data=engine.dataframe_fingerprint(data, length=64), settings=settings,
        calculation_version=calculation_version), length=24)


def _validate_gd_entry(data, entry):
    if not isinstance(entry, dict) or not isinstance(entry.get("settings"), dict) or not isinstance(entry.get("result"), dict):
        raise ProjectError("Incomplete G/D record")
    settings, result = entry["settings"], entry["result"]
    multivariate = "score_cols" in settings
    required = {"object_facet", "random_facets", "n_max_per_facet", "target_population", "selected_plan"}
    required |= {"score_cols", "composite_weights"} if multivariate else {"score_col"}
    allowed = required | {"fixed_condition", "coefficient_intervals", "method", "missing", "nesting"}
    if (not required.issubset(settings) or set(settings) - allowed
            or type(result.get("available")) is not bool or not result.get("calculation_version")):
        raise ProjectError("Incomplete G/D settings or availability")
    if "fixed_condition" in settings:
        fc = settings["fixed_condition"]
        if (not isinstance(fc, dict) or set(fc) != {"facet", "mode", "level"}
                or not isinstance(fc.get("facet"), str) or not fc["facet"] or ":" in fc["facet"]
                or fc["facet"] in {settings["object_facet"], *settings.get("score_cols", [settings.get("score_col")]), *settings["random_facets"]}
                or fc.get("mode") not in ("level", "sum", "mean")
                or (fc["mode"] == "level" and (not isinstance(fc.get("level"), str) or not fc["level"].strip()))
                or (fc["mode"] != "level" and fc.get("level") is not None)):
            raise ProjectError("Invalid G/D fixed-facet conditioning settings")
    if gd_analysis_identity(data, settings, result["calculation_version"]) != entry.get("analysis_id"):
        raise ProjectError("G/D results belong to different input data or settings")
    design = result.get("design", {})
    if design.get("score_basis") != "original_observed_scores":
        raise ProjectError("Desktop G/D requires original observed scores")
    if "fixed_condition" in settings:
        if (design.get("raw_data_fingerprint") != engine.dataframe_fingerprint(data, length=64)
                or design.get("fixed_condition") != settings["fixed_condition"]):
            raise ProjectError("G/D input fingerprint does not match")
    elif ("data_fingerprint" in design
            and design["data_fingerprint"] != engine.dataframe_fingerprint(data, length=64)):
        raise ProjectError("G/D input fingerprint does not match")
    if result.get("bootstrap_ci", {}).get("inference_allowed") is not False:
        raise ProjectError("G/D interval qualification is missing or unsupported")
    from mfrmpy._core import unbalanced_gd
    try:
        unbalanced_gd.validate_settings(settings)
    except (ValueError, TypeError, KeyError) as exc:
        raise ProjectError(str(exc)) from exc
    minque = settings.get("method") == "minque0"
    if minque:
        if (result.get("calculation_version") != unbalanced_gd.VERSION or design.get("method") != "minque0"
                or design.get("nesting") != (settings.get("nesting") or {})
                or design.get("missing") != settings.get("missing", "error")
                or not isinstance(result.get("excluded_rows"), pd.DataFrame)):
            raise ProjectError("MINQUE(0) design or missing-data record is inconsistent")
        if result["available"]:
            audit = result.get("estimation", {})
            sources = list(design.get("component_factors", {}))
            gram = audit.get("kernel_gram")
            if (not audit.get("identified") or audit.get("rank") != len(sources)
                    or not isinstance(gram, pd.DataFrame) or list(gram.index) != sources or list(gram.columns) != sources
                    or design.get("planning_status") not in ("balanced_mean_projection", "selected_plan_required")):
                raise ProjectError("MINQUE(0) component identification record is incomplete")
    from mfrmpy._core import gd_intervals
    try:
        request = gd_intervals.validate_request(settings.get("coefficient_intervals"))
    except ValueError as exc:
        raise ProjectError(str(exc)) from exc
    recorded = result.get("coefficient_intervals")
    if request is not None:
        if (not isinstance(recorded, dict) or any(recorded.get(k) != v for k, v in request.items())
                or not isinstance(recorded.get("scope"), str) or not recorded["scope"].strip()
                or recorded.get("covariance_basis") != "normal_expected_mean_squares_nonnegative_plugin"):
            raise ProjectError("G/D coefficient interval assumptions are missing or inconsistent")
        if result["available"]:
            covariance = recorded.get("covariance")
            names = result["variance_components"].Source.tolist()
            if (recorded.get("status") != "computed" or not isinstance(covariance, pd.DataFrame)
                    or list(covariance.index) != names or list(covariance.columns) != names
                    or not set(gd_intervals.FIELDS).issubset(recorded.get("observed", {}))
                    or not set(gd_intervals.FIELDS).issubset(result.get("selected_plan", {}))
                    or not set(gd_intervals.FIELDS).issubset(result["d_study"].columns)):
                raise ProjectError("G/D coefficient interval record is incomplete")
    elif recorded is not None:
        raise ProjectError("G/D coefficient intervals require explicit assumptions in settings")
    if multivariate or minque:
        from mfrmpy._core import multivariate_gd
        try:
            scores, weights = (multivariate_gd.validate_settings(settings) if multivariate else ([settings["score_col"]], {}))
        except ValueError as exc:
            raise ProjectError(str(exc)) from exc
        record = result.get("multivariate" if multivariate else "design_study", {})
        if (not isinstance(record, dict) or record.get("score_columns") != scores or record.get("weights") != weights
                or design.get("score_columns") != scores or design.get("composite_weights", {}) != weights):
            raise ProjectError("Multivariate score identities or weights do not match")
        if result["available"]:
            matrices = record.get("covariance_components")
            sources = result["variance_components"].Source.tolist()
            if not isinstance(matrices, dict) or list(matrices) != sources:
                raise ProjectError("Multivariate covariance components are incomplete")
            for matrix in matrices.values():
                if (not isinstance(matrix, pd.DataFrame) or list(matrix.index) != scores or list(matrix.columns) != scores
                        or not np.isfinite(matrix.to_numpy(dtype=float)).all()
                        or not np.allclose(matrix, matrix.T, rtol=1e-12, atol=0)):
                    raise ProjectError("Multivariate covariance score labels or symmetry are invalid")
            for key in ("observed", "selected", "variance_components", "d_study", "component_diagnostics"):
                empty_allowed = minque and not settings["selected_plan"] and not design["observed_n"] and key in ("selected", "d_study")
                if not isinstance(record.get(key), pd.DataFrame) or (record[key].empty and not empty_allowed):
                    raise ProjectError("Multivariate retained output is incomplete")
    elif "multivariate" in result:
        raise ProjectError("Multivariate results require score_cols settings")
    if result["available"]:
        counts = settings["selected_plan"] or design["observed_n"]
        selected = result.get("selected_plan", {})
        components_only = minque and not counts and design.get("planning_status") == "selected_plan_required"
        if (not components_only and set(counts) != set(settings["random_facets"])) or any(selected.get(f) != n for f, n in counts.items()):
            raise ProjectError("Selected D plan does not match its settings")
        if components_only and (selected or not result["d_study"].empty):
            raise ProjectError("Unbalanced MINQUE(0) projections require an explicit selected plan")
    elif not result.get("reason_code"):
        raise ProjectError("Unavailable G/D results must retain their reason")


def validate_analysis(analysis, *, _computed_fingerprint=None):
    kinds = ("basic-rsm-jmle-v1", "mfrm-analysis-v2", "gd-analysis-v2", "mfrm-analysis-v3")
    if not isinstance(analysis, dict) or analysis.get("kind") not in kinds:
        raise ProjectError("Unsupported desktop analysis kind")
    expected_fp = content_fingerprint(analysis) if _computed_fingerprint is None else _computed_fingerprint
    if analysis.get("content_fingerprint") != expected_fp:
        raise ProjectError("Analysis content changed: input, settings, results or tables no longer match")
    if not isinstance(analysis.get("input_data"), pd.DataFrame):
        raise ProjectError("Projects require the retained original input data")
    if "view_settings" in analysis:
        view = analysis["view_settings"]
        if (not isinstance(view, dict) or not {"figure", "facet", "statistic"}.issubset(view)
                or set(view) - {"figure", "facet", "statistic", "presentation", "language", "gd_swap"}
                or view.get("presentation", "explained") not in ("explained", "manuscript")
                or view.get("language", "en") not in ("en", "ja")
                or view.get("gd_swap", "0") not in ("0", "1")
                or any(not isinstance(value, str) for value in view.values())):
            raise ProjectError("Invalid saved view settings")
    if analysis["kind"] == "gd-analysis-v2":
        if "anchor_sensitivity" in analysis:
            raise ProjectError("Anchor sensitivity requires an MFRM calibration")
        _validate_gd_entry(analysis["input_data"], analysis.get("gd_analysis"))
        if analysis["analysis_id"] != analysis["gd_analysis"]["analysis_id"] or "result" in analysis:
            raise ProjectError("Standalone G/D identity or workflow does not match")
        return
    result = analysis["result"]
    if analysis["kind"] == "mfrm-analysis-v3":
        outputs = analysis.get("computed_outputs")
        if (not isinstance(outputs, list) or any(not isinstance(name, str) for name in outputs)
                or outputs != sorted(set(outputs))
                or set(outputs) - {"measures", "agreement", "scorefile", "pca"}):
            raise ProjectError("Invalid computed output record")
        diag = analysis.get("diagnostics", {})
        for name, present in {
            "measures": isinstance(diag.get("measures"), pd.DataFrame) and not diag["measures"].empty,
            "agreement": isinstance(diag.get("interrater_agreement"), dict) and bool(diag["interrater_agreement"]),
            "scorefile": isinstance(analysis.get("scorefile"), pd.DataFrame),
            "pca": bool(diag.get("pca_enabled")),
        }.items():
            if (name in outputs) != present:
                raise ProjectError(f"Stored {name} does not match computed_outputs")
    cfg = result["config"]
    legacy = analysis["kind"] == "basic-rsm-jmle-v1"
    if (cfg.get("model") not in ("RSM", "PCM") or cfg.get("method") not in ("JMLE", "MML")
            or cfg.get("estimate_population_sd")
            or (legacy and (cfg["model"], cfg["method"]) != ("RSM", "JMLE"))):
        raise ProjectError("Project model/method does not match the supported workflow")
    if engine.build_result_analysis_identity(result).analysis_id != analysis.get("analysis_id"):
        raise ProjectError("Analysis identity does not match the stored result")
    if not isinstance(result["prep"].get("data"), pd.DataFrame):
        raise ProjectError("Projects require the retained estimation data")
    data, settings = analysis.get("input_data"), analysis.get("fit_settings")
    required = {"person_col", "facet_cols", "score_col", "rating_min", "rating_max"}
    if not isinstance(data, pd.DataFrame) or not isinstance(settings, dict) or not required.issubset(settings):
        raise ProjectError("Projects require input data and complete column/score settings")
    if not isinstance(settings["facet_cols"], list) or not settings["facet_cols"]:
        raise ProjectError("Project facet settings are missing")
    columns = [settings["person_col"], *settings["facet_cols"], settings["score_col"]]
    if len(set(columns)) != len(columns) or any(column not in data for column in columns):
        raise ProjectError("Project column settings do not match its input data")
    if not legacy:
        for key in ("model", "method", "rating_min", "rating_max"):
            if settings.get(key) != cfg.get(key):
                raise ProjectError(f"Stored {key} disagrees with the fitted result")
        if settings["facet_cols"] != cfg["facet_names"]:
            raise ProjectError("Stored facets disagree with the fitted result")
        if cfg["model"] == "PCM" and settings.get("step_facet") != cfg.get("step_facet"):
            raise ProjectError("Stored PCM step facet disagrees with the fitted result")
        expected_nc = settings.get("noncenter_facet", "Person")
        if "noncenter_facet" in settings and (
            not isinstance(expected_nc, str) or expected_nc == "Person" or expected_nc not in cfg["facet_names"]
        ):
            raise ProjectError("Invalid stored noncenter_facet setting")
        if cfg.get("noncenter_facet", "Person") != expected_nc:
            raise ProjectError("Stored noncenter_facet disagrees with the fitted result")
        expected_pos = settings.get("positive_facets", [])
        if "positive_facets" in settings and (
            not isinstance(expected_pos, list)
            or not expected_pos
            or any(not isinstance(f, str) or f not in cfg["facet_names"] for f in expected_pos)
            or expected_pos != [f for f in cfg["facet_names"] if f in set(expected_pos)]
        ):
            raise ProjectError("Invalid stored positive_facets setting")
        actual_pos = [f for f in cfg["facet_names"] if int((cfg.get("facet_signs") or {}).get(f, -1)) == 1]
        if expected_pos != actual_pos:
            raise ProjectError("Stored positive_facets disagree with the fitted result")
        if cfg["method"] == "MML":
            if "population_mean" in settings:
                mode = settings["population_mean"]
                if mode not in ("fixed_zero", "estimate") or bool(cfg.get("population_formula")) != (mode == "estimate"):
                    raise ProjectError("Stored population mean setting disagrees with the fitted model")
            s_eng = settings.get("mml_engine")
            c_eng = cfg.get("mml_engine")
            if (settings.get("quad_points") != cfg.get("quad_points")
                    or settings.get("population_prior_sd") != cfg.get("population_prior_sd")
                    or settings.get("estimate_population_sd") is not False
                    or s_eng not in ("EM", "Hybrid", "Direct")
                    or c_eng not in ("em", "hybrid", "direct")
                    or s_eng.lower() != c_eng):
                raise ProjectError("Stored fixed-SD MML settings disagree with the fitted result")
    if legacy and (
        any(analysis.get(key) for key in ("bias_analyses", "interval_analyses", "gd_analysis"))
        or settings.get("anchors") or settings.get("group_anchors")
        or settings.get("compute_pca") or "umean" in settings or "uscale" in settings
        or "noncenter_facet" in settings or "positive_facets" in settings
    ):
        raise ProjectError("Supplementary analyses, anchors, PCA and rescaling require schema 2")
    if "compute_pca" in settings and settings["compute_pca"] is not True:
        raise ProjectError("Stored compute_pca flag must be True when present")
    want_pca = ("pca" in analysis["computed_outputs"] if analysis["kind"] == "mfrm-analysis-v3"
                else bool(settings.get("compute_pca", False)))
    diag = analysis.get("diagnostics")
    scope = analysis.get("scope")
    if isinstance(diag, dict) and "pca_enabled" in diag and bool(diag.get("pca_enabled")) != want_pca:
        raise ProjectError("Stored diagnostics PCA state disagrees with fit_settings")
    if isinstance(scope, dict) and "pca" in scope and bool(scope.get("pca")) != want_pca:
        raise ProjectError("Stored scope PCA flag disagrees with fit_settings")
    if ("umean" in settings) != ("uscale" in settings):
        raise ProjectError("umean and uscale must be stored together")
    if "umean" in settings:
        raw_umean, raw_uscale = settings["umean"], settings["uscale"]
        if (isinstance(raw_umean, (bool, np.bool_)) or isinstance(raw_uscale, (bool, np.bool_))
                or not isinstance(raw_umean, (int, float)) or not isinstance(raw_uscale, (int, float))
                or not math.isfinite(raw_umean) or not math.isfinite(raw_uscale) or raw_uscale <= 0
                or (float(raw_umean), float(raw_uscale)) == (0.0, 1.0)):
            raise ProjectError("Invalid stored umean/uscale rescaling settings")
    s_umean = float(settings.get("umean", 0.0))
    s_uscale = float(settings.get("uscale", 1.0))
    report_tables = analysis.get("report_tables")
    diag_measures = diag.get("measures") if isinstance(diag, dict) else None
    if isinstance(report_tables, dict) and isinstance(diag_measures, pd.DataFrame) and {"Facet", "Level", "Estimate"}.issubset(diag_measures.columns):
        for r_facet, r_tbl in report_tables.items():
            if isinstance(r_tbl, pd.DataFrame) and {"Level", "Measure"}.issubset(r_tbl.columns):
                m_sub = diag_measures.loc[diag_measures["Facet"].astype(str).eq(str(r_facet)), ["Level", "Estimate"]].copy()
                if m_sub.empty:
                    continue
                m_sub["Level"] = m_sub["Level"].astype(str)
                est_map = dict(zip(m_sub["Level"], pd.to_numeric(m_sub["Estimate"], errors="coerce")))
                est = pd.Series([est_map.get(str(lv), np.nan) for lv in r_tbl["Level"]], dtype=float).to_numpy()
                meas = pd.to_numeric(r_tbl["Measure"], errors="coerce").to_numpy(dtype=float)
                mask = np.isfinite(est)
                if np.any(mask) and (
                    not np.all(np.isfinite(meas[mask]))
                    or not np.allclose(meas[mask], est[mask] * s_uscale + s_umean, rtol=1e-9, atol=1e-9)
                ):
                    raise ProjectError(f"Stored report_tables[{r_facet!r}] Measure disagrees with umean/uscale")
    if "anchors" in settings or "group_anchors" in settings:
        from mfrmpy._core import basic_analysis
        try:
            norm_a = basic_analysis.normalize_anchor_records(
                settings.get("anchors"), person_col=settings["person_col"], facet_cols=settings["facet_cols"]
            )
            norm_g = basic_analysis.normalize_group_anchor_records(
                settings.get("group_anchors"),
                person_col=settings["person_col"],
                facet_cols=settings["facet_cols"],
                fixed_pairs={(r["Facet"], r["Level"]) for r in norm_a},
            )
        except ValueError as exc:
            raise ProjectError(f"Invalid stored anchor settings: {exc}") from exc
        if settings.get("anchors", []) != norm_a or settings.get("group_anchors", []) != norm_g:
            raise ProjectError("Stored anchor settings are not in canonical form")
        for facet in ["Person", *cfg["facet_names"]]:
            spec = cfg["theta_spec"] if facet == "Person" else cfg["facet_specs"][facet]
            levels = list(map(str, spec["levels"]))
            expected_fixed = {r["Level"]: r["Anchor"] for r in norm_a if r["Facet"] == facet}
            actual_fixed = {
                levels[i]: float(v)
                for i, v in enumerate(spec.get("anchors", []))
                if np.isfinite(v)
            }
            if set(expected_fixed) != set(actual_fixed) or any(
                abs(expected_fixed[k] - actual_fixed[k]) > 1e-12 for k in expected_fixed
            ):
                raise ProjectError(f"Stored fixed anchors for {facet} disagree with the fitted result")
            expected_groups = {r["Level"]: (r["Group"], r["GroupValue"]) for r in norm_g if r["Facet"] == facet}
            actual_groups = {
                levels[i]: (str(g), float(spec.get("group_values", {}).get(str(g), np.nan)))
                for i, g in enumerate(spec.get("groups", []))
                if g not in (None, "") and not pd.isna(g)
            }
            if set(expected_groups) != set(actual_groups) or any(
                expected_groups[k][0] != actual_groups[k][0]
                or not np.isfinite(actual_groups[k][1])
                or abs(expected_groups[k][1] - actual_groups[k][1]) > 1e-12
                for k in expected_groups
            ):
                raise ProjectError(f"Stored group anchors for {facet} disagree with the fitted result")
    else:
        for facet in ["Person", *cfg["facet_names"]]:
            spec = cfg["theta_spec"] if facet == "Person" else cfg["facet_specs"][facet]
            if np.isfinite(spec.get("anchors", [])).any() or any(
                g not in (None, "") and not pd.isna(g) for g in spec.get("groups", [])
            ):
                raise ProjectError(f"Fitted result contains {facet} anchors missing from fit_settings")
    for entry in analysis.get("bias_analyses", []):
        if entry.get("analysis_id") != analysis["analysis_id"] or not isinstance(entry.get("result"), dict):
            raise ProjectError("Bias results belong to a different calibration")
        request = entry.get("settings", {})
        if (set(request) != {"facet_a", "facet_b", "max_abs", "omit_extreme"}
                or request["facet_a"] == request["facet_b"]
                or any(request[f] not in ["Person", *cfg["facet_names"]] for f in ("facet_a", "facet_b"))):
            raise ProjectError("Incomplete bias facet-pair settings")
        table = entry["result"].get("table")
        if isinstance(table, pd.DataFrame) and not table.empty:
            if not table["FacetA"].eq(request["facet_a"]).all() or not table["FacetB"].eq(request["facet_b"]).all():
                raise ProjectError("Bias table does not match its requested facet pair")
        elif not entry["result"].get("_skip_reason"):
            raise ProjectError("Bias results must contain cells or an explicit skip reason")
    for entry in analysis.get("interval_analyses", []):
        if entry.get("analysis_id") != analysis["analysis_id"] or cfg["method"] != "MML":
            raise ProjectError("Intervals belong to a different calibration or method")
        request = entry.get("settings", {})
        if set(request) != {"facet", "difference", "level"} or request["facet"] not in cfg["facet_names"]:
            raise ProjectError("Incomplete specified interval settings")
        if entry.get("status") == "computed":
            from mfrmpy._core import structural_intervals
            record = entry.get("record")
            if not isinstance(record, dict) or record.get("record_hash") != structural_intervals._fingerprint(record):
                raise ProjectError("Stored interval evidence changed")
            differences = [row for row in record["targets"] if row["Kind"] == "difference"]
            expected = request["difference"]
            if (record["level"] != request["level"]
                    or any(row["Facet"] != request["facet"] for row in record["targets"])
                    or ([(row["Level"], row["Reference"]) for row in differences]
                        != ([tuple(expected)] if expected is not None else []))):
                raise ProjectError("Stored intervals do not match the requested targets")
        elif entry.get("status") not in ("unavailable", "failed") or entry.get("record") is not None or not entry.get("reason"):
            raise ProjectError("Interval review must retain a record or an explicit failure reason")
    if entry := analysis.get('joint_analysis'):
        from mfrmpy._core import joint_interaction
        if entry.get('analysis_id') != analysis['analysis_id']:
            raise ProjectError('Joint interaction belongs to a different baseline analysis')
        try:
            request = joint_interaction.settings(entry.get('settings'), cfg['facet_names'], cfg['method'])
        except (ValueError, TypeError) as exc:
            raise ProjectError(str(exc)) from exc
        if request is None or request != entry['settings']:
            raise ProjectError('Incomplete joint interaction settings')
        if entry.get('status') == 'computed':
            record = entry.get('record')
            if (not isinstance(record, dict) or record.get('record_hash') != joint_interaction._fingerprint(record)
                    or record.get('settings') != request):
                raise ProjectError('Joint interaction evidence does not match its request')
        elif entry.get('status') not in ('unavailable', 'failed') or entry.get('record') is not None or not entry.get('reason'):
            raise ProjectError('Joint interaction must retain evidence or an explicit failure reason')
    if "rater_dff" in analysis:
        from mfrmpy._core import rater_dff
        rater_dff.validate_entry(analysis, analysis["rater_dff"])
    if "anchor_sensitivity" in analysis:
        from mfrmpy._core import anchor_sensitivity
        anchor_sensitivity.validate_entry(analysis, analysis["anchor_sensitivity"])
    if entry := analysis.get("gd_analysis"):
        _validate_gd_entry(data, entry)


def serialize_project(analysis, *, compress=True, assign_fingerprint=False):
    """Validate and stream typed JSON without a full intermediate JSON tree."""
    if not isinstance(analysis, dict):
        validate_analysis(analysis)
    fp = content_fingerprint(analysis)
    if assign_fingerprint:
        analysis["content_fingerprint"] = fp
    validate_analysis(analysis, _computed_fingerprint=fp)
    schema = SCHEMA_VERSION if analysis["kind"] == "mfrm-analysis-v3" else 2
    header = (f'{{"format":{_json_bytes(FORMAT).decode("utf-8")},'
              f'"schema_version":{schema},"analysis":').encode("utf-8")
    buf = io.BytesIO()
    stream = gzip.GzipFile(fileobj=buf, mode="wb", compresslevel=6, mtime=0) if compress else buf
    uncompressed_len = 0
    def write(chunk):
        nonlocal uncompressed_len
        uncompressed_len += len(chunk)
        if uncompressed_len > MAX_UNCOMPRESSED_PROJECT_BYTES:
            raise ProjectError("Project exceeds the uncompressed project size limit (512 MiB)")
        stream.write(chunk)
        if buf.tell() > MAX_PROJECT_BYTES:
            raise ProjectError("Project exceeds the project size limit (128 MiB)")
    try:
        write(header)
        for chunk in _encoded_chunks(analysis):
            write(chunk)
        write(b"}")
    finally:
        if compress:
            stream.close()
    data = buf.getvalue()
    if len(data) > MAX_PROJECT_BYTES:
        raise ProjectError("Project exceeds the project size limit (128 MiB)")
    return data


def save_project(path, analysis, *, compress=True):
    """Atomically save a validated snapshot; never damage an existing file on failure."""
    data = serialize_project(analysis, compress=compress)
    write_bytes_atomic(path, data)


def write_bytes_atomic(path, data):
    """Write a completed project/report, retaining any old file if replacement fails."""
    path = Path(path)
    temporary = None
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _read_bounded(stream, limit):
    data = bytearray()
    while len(data) <= limit:
        chunk = stream.read(min(1024 * 1024, limit + 1 - len(data)))
        if not chunk:
            break
        data.extend(chunk)
    return data


def _unpack_project_bytes(data):
    if not isinstance(data, (bytes, bytearray)):
        raise ProjectError("Project payload must be raw bytes")
    raw = bytes(data)
    if len(raw) > MAX_PROJECT_BYTES:
        raise ProjectError("Project exceeds the project size limit (128 MiB)")
    if raw.startswith(GZIP_MAGIC):
        try:
            with gzip.GzipFile(fileobj=io.BytesIO(raw), mode="rb") as gz:
                uncompressed = _read_bounded(gz, MAX_UNCOMPRESSED_PROJECT_BYTES)
        except (OSError, EOFError, zlib.error) as exc:
            raise ProjectError(f"Malformed project: {exc}") from exc
        if len(uncompressed) > MAX_UNCOMPRESSED_PROJECT_BYTES:
            raise ProjectError("Project exceeds the uncompressed project size limit (512 MiB)")
        return uncompressed
    if len(raw) > MAX_UNCOMPRESSED_PROJECT_BYTES:
        raise ProjectError("Project exceeds the uncompressed project size limit (512 MiB)")
    return raw


def input_refit_reason(analysis):
    """Detect the two legacy preparation errors from retained rows without refitting."""
    prep = analysis.get("result", {}).get("prep", {})
    settings = analysis.get("fit_settings", {})
    raw, fitted = analysis.get("input_data"), prep.get("data")
    audit = prep.get("row_audit")
    if not isinstance(fitted, pd.DataFrame) or not isinstance(audit, pd.DataFrame):
        return ""
    if "IncludedInLikelihood" not in audit or len(audit) != len(raw):
        return ""
    included = raw.iloc[audit["IncludedInLikelihood"].to_numpy(dtype=bool)]
    if len(included) != len(fitted):
        return ""
    scores = pd.to_numeric(included[settings["score_col"]], errors="coerce").to_numpy(dtype=float)
    if not np.array_equal(np.rint(scores), fitted["Score"].to_numpy(dtype=float)):
        return "Saved fitted categories differ from rounded input scores. Refit the original ratings with mfrmpy 0.2.1 or later."
    if "score_k" in settings["facet_cols"] and "score_index_col" not in prep:
        if not np.array_equal(included["score_k"].astype(str), fitted["score_k"].astype(str)):
            return "Saved score_k facet levels differ from the input IDs. Refit the original ratings with mfrmpy 0.2.1 or later."
    return ""


def require_reportable_input(analysis):
    reason = input_refit_reason(analysis)
    if reason:
        raise ProjectError(reason)


def loads_project(data):
    """Restore a validated `.mfrm` payload (gzip or legacy uncompressed JSON) with zero estimation."""
    try:
        raw_json = _unpack_project_bytes(data)
        document = json.loads(raw_json)
        if not isinstance(document, dict) or document.get("format") != FORMAT:
            raise ProjectError("Not an MFRM basic project")
        if type(document.get("schema_version")) is not int or document["schema_version"] not in (1, 2, SCHEMA_VERSION):
            raise ProjectError("Unsupported project schema version")
        if set(document) != {"format", "schema_version", "analysis"}:
            raise ProjectError("Unknown project document fields")
        analysis = _decode(document["analysis"])
        validate_analysis(analysis)
        if document["schema_version"] < 3 and analysis["kind"] == "mfrm-analysis-v3":
            raise ProjectError("Partial analyses require schema 3")
        if document["schema_version"] == 1 and analysis["kind"] != "basic-rsm-jmle-v1":
            raise ProjectError("Schema 1 supports only the original RSM/JMLE workflow")
        reason = input_refit_reason(analysis)
        if reason:
            warnings.warn(reason, UserWarning, stacklevel=2)
        return analysis
    except (ValueError, TypeError, KeyError, IndexError, AttributeError, OverflowError, RecursionError) as exc:
        if isinstance(exc, ProjectError):
            raise
        raise ProjectError(f"Malformed project: {exc}") from exc


def load_project(path):
    """Restore a completed or failed result without running the estimator."""
    with Path(path).open("rb") as stream:
        data = _read_bounded(stream, MAX_PROJECT_BYTES)
    return loads_project(data)
