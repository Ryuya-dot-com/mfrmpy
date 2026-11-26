"""Category design for one profile on the fitted logit scale."""
from collections.abc import Mapping
from copy import deepcopy
from numbers import Real

import numpy as np
import pandas as pd
from scipy.special import softmax

from mfrm_app import desktop_analysis, engine, joint_interaction, output_qualification, project, reporting_facts


def _baseline_values(analysis, profile, step_facet):
    result = analysis["result"]
    cfg, params = result["config"], result["params"]
    # Follow the public point-estimate masks, then read eligible values at
    # calculation precision. The displayed tables round to three decimals.
    measures, note = reporting_facts.publication_measure_table(result, analysis["diagnostics"], fit_note="")
    if measures.empty:
        return [], note or "Facet estimates unavailable"
    allowed = pd.to_numeric(measures.set_index(["Facet", "Level"])["Estimate"], errors="coerce")
    values = []
    for facet, level in profile.items():
        if not np.isfinite(allowed.get((facet, level), np.nan)):
            return [], f"Facet estimate withheld for {facet} / {level}"
        values.append(params["facets"][facet][cfg["facet_levels"][facet].index(level)])
    table = desktop_analysis._step_thresholds_table(analysis)
    owner = profile[step_facet] if step_facet else "Common"
    if table is None:
        return [], "Step estimates unavailable"
    steps = table[1].loc[table[1]["StepFacet"].eq(owner), "Estimate"]
    if len(steps) != cfg["n_cat"] - 1 or not np.isfinite(pd.to_numeric(steps, errors="coerce")).all():
        return [], f"Step estimates withheld for {owner}"
    ladder = (params["steps_mat"][cfg["facet_levels"][step_facet].index(owner)]
              if step_facet else params["steps"])
    return [*values, *ladder], ""


def category_design(analysis, *, levels, theta=0.0, joint=False):
    """Return category weights and conditional probabilities for one facet profile.

    A uses expanded logit coefficients in the order of the terms table.
    B is the category index starting at zero. Probabilities are the softmax
    of A @ terms.Estimate + B * theta. Joint models use their own calibration.
    """
    project.validate_analysis(analysis)
    if "result" not in analysis:
        raise ValueError("Category designs require an RSM or PCM analysis")
    project.require_reportable_input(analysis)
    if not isinstance(theta, Real) or isinstance(theta, (bool, np.bool_)) or not np.isfinite(theta):
        raise ValueError("theta must be a finite ability in logits")
    if not isinstance(joint, (bool, np.bool_)):
        raise ValueError("joint must be True or False")
    result = analysis["result"]
    cfg = result["config"]
    facets = cfg["facet_names"]
    if not isinstance(levels, Mapping) or set(levels) != set(facets):
        raise ValueError(f"Select one level for every structural facet: {facets}")
    profile = {}
    for f in facets:
        level = levels[f]
        if not isinstance(level, str) or level.strip() not in cfg["facet_levels"][f]:
            raise ValueError(f"Unknown fitted level for {f}: {level!r}")
        profile[f] = level.strip()

    step_facet = cfg.get("step_facet") if cfg["model"] == "PCM" else None
    owner = profile[step_facet] if step_facet else None
    n = cfg["n_cat"]
    if n * (len(facets) + int(joint) + n - 1) > 5_000_000:
        raise ValueError("Selected category design exceeds five million dense entries")
    k = np.arange(n)
    columns = [cfg.get("facet_signs", {}).get(f, -1) * k for f in facets]
    terms = [dict(Kind="facet", Facet=f, Level=profile[f]) for f in facets]
    entry = analysis.get("joint_analysis")
    state = output_qualification.run_qualification_state(result)
    hold = state["screening_block_reason"]
    joint_interval_reason = None
    values = []
    if joint:
        if not entry or entry["status"] != "computed":
            raise ValueError(entry["reason"] if entry else "No joint interaction was fitted")
        record = entry["record"]
        cells = joint_interaction.read_joint_interaction(result, record)
        a, b = entry["settings"]["facet_a"], entry["settings"]["facet_b"]
        cell = cells.loc[cells.Kind.eq("cell") & cells.LevelA.eq(profile[a]) & cells.LevelB.eq(profile[b])].iloc[0]
        terms.append(dict(Kind="interaction", Facet=a, Level=profile[a], OtherFacet=b, OtherLevel=profile[b]))
        columns.append(k)
        main = pd.DataFrame(record["main_effects_and_steps"])
        effects = main.loc[main.kind.eq("facet")].set_index(["facet", "level"])["Estimate"]
        ladder = main.loc[main.kind.eq("step") & main.level.eq(owner if step_facet else "shared")].sort_values("step")
        values = [*(effects.at[f, profile[f]] for f in facets), cell.Estimate, *ladder.Estimate]
        hold = "" if record["optimizer_success"] else "Joint optimizer termination unverified"
        joint_interval_reason = cell.Reason
    elif not hold:
        values, hold = _baseline_values(analysis, profile, step_facet)

    first = int(result["prep"]["rating_min"])
    for step in range(1, n):
        terms.append(dict(Kind="step", Facet=step_facet, Level=owner, ToScore=first + step))
        columns.append(-(k >= step).astype(int))
    terms = pd.DataFrame(terms, columns=["Kind", "Facet", "Level", "OtherFacet", "OtherLevel", "ToScore"])
    terms["ToScore"] = terms["ToScore"].astype("Int64")
    terms.index.name = "Term"
    terms["Estimate"] = np.nan if hold else values
    categories = pd.DataFrame({"B": k}, index=pd.Index(first + k, name="Score"))
    A = pd.DataFrame(np.column_stack(columns), index=categories.index, columns=terms.index)
    categories["Probability"] = np.nan
    if not hold:
        logits = A @ terms["Estimate"] + k * float(theta)
        if not np.isfinite(logits).all():
            raise ValueError("Nonfinite category logits for the supplied ability and coefficients")
        categories["Probability"] = softmax(logits.to_numpy())

    origins = {}
    for facet, spec in {"Person": cfg["theta_spec"], **cfg["facet_specs"]}.items():
        origins[facet] = spec.get("origin_constraint", "sum_to_zero" if spec["centered"] else "free")
    population = None
    if cfg["method"] == "MML":
        origins["Person"] = "normal_population"
        population = dict(mean=0.0, sd=engine.get_population_prior_sd(cfg))
    settings = analysis["fit_settings"]
    return dict(A=A, categories=categories, terms=terms, theta=float(theta), analysis_id=analysis["analysis_id"],
                model=cfg["model"], method=cfg["method"], calibration="joint" if joint else "baseline",
                origins=origins, population=population, step_constraint="sum_to_zero_per_ladder",
                interaction_constraint="zero_row_and_column_sums" if joint else None,
                anchors=deepcopy(settings.get("anchors", [])), group_anchors=deepcopy(settings.get("group_anchors", [])),
                baseline_state=state["state"], withheld_reason=hold, joint_interval_reason=joint_interval_reason)
