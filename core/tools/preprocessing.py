from __future__ import annotations

from typing import Any, Dict, List, Tuple, Optional


def normalize_preprocess_rules(rules: Any) -> Dict[str, Any]:
    """
    Normalize user-provided preprocessing rules into a standard schema.

    Supported input:
    - dict with keys: filters (list), logic ("and"/"or"), drop_na (bool)
    - list of filters (auto-wrapped as {"filters": [...]})
    """
    if rules is None:
        return {"filters": [], "logic": "and", "drop_na": True}

    if isinstance(rules, list):
        rules = {"filters": rules}

    if not isinstance(rules, dict):
        raise ValueError("preprocess rules must be a dict or list of filters")

    filters = rules.get("filters") or []
    if not isinstance(filters, list):
        raise ValueError("preprocess rules.filters must be a list")
    filters = [f for f in filters if isinstance(f, dict)]

    logic = str(rules.get("logic", "and")).strip().lower()
    if logic not in ("and", "or"):
        logic = "and"

    drop_na = bool(rules.get("drop_na", True))

    return {
        "filters": filters,
        "logic": logic,
        "drop_na": drop_na,
    }


def _value_exists(val: Any) -> bool:
    if val is None:
        return False
    if isinstance(val, (list, tuple)):
        return len(val) > 0
    try:
        import numpy as np
        if isinstance(val, np.ndarray):
            return val.size > 0
    except Exception:
        pass
    return True


def _coerce_numeric(val: Any) -> Optional[float]:
    if val is None:
        return None
    if isinstance(val, bool):
        return None
    if isinstance(val, (int, float)):
        return float(val)
    try:
        # numpy scalar
        if hasattr(val, "item"):
            v = val.item()
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                return float(v)
    except Exception:
        return None
    return None


def _match_filter(rec: Dict[str, Any], flt: Dict[str, Any], drop_na: bool) -> bool:
    field = flt.get("field")
    if not field:
        return True

    op = str(flt.get("op", "==")).strip().lower()
    val = rec.get(field, None)

    if op == "exists":
        return _value_exists(val)

    if val is None:
        return False if drop_na else True

    if op in ("in", "not_in"):
        values = flt.get("values", flt.get("value", []))
        if not isinstance(values, list):
            values = [values]
        hit = val in values
        return hit if op == "in" else (not hit)

    if op == "contains":
        target = flt.get("value")
        if isinstance(val, str) and isinstance(target, str):
            return target in val
        return False if drop_na else True

    if op == "between":
        v = _coerce_numeric(val)
        min_v = _coerce_numeric(flt.get("min"))
        max_v = _coerce_numeric(flt.get("max"))
        if v is None or min_v is None or max_v is None:
            return False if drop_na else True
        return min_v <= v <= max_v

    # Numeric comparisons (fallback to raw for ==/!=)
    if op in ("==", "!=", "<", ">", "<=", ">="):
        target = flt.get("value")
        v = _coerce_numeric(val)
        t = _coerce_numeric(target)

        if v is None or t is None:
            if op in ("==", "!="):
                hit = val == target
                return hit if op == "==" else (not hit)
            return False if drop_na else True

        if op == "==":
            return v == t
        if op == "!=":
            return v != t
        if op == "<":
            return v < t
        if op == ">":
            return v > t
        if op == "<=":
            return v <= t
        if op == ">=":
            return v >= t

    # Unknown op -> do not filter out
    return True


def apply_preprocess_rules(
    trials: List[Dict[str, Any]],
    rules: Any,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Apply preprocessing rules to a list of trial dicts.
    Returns (filtered_trials, summary).
    """
    normalized = normalize_preprocess_rules(rules)
    filters = normalized.get("filters", [])
    logic = normalized.get("logic", "and")
    drop_na = normalized.get("drop_na", True)

    if not filters:
        return trials, {
            "original": len(trials),
            "kept": len(trials),
            "filtered": 0,
            "logic": logic,
            "drop_na": drop_na,
            "filters": [],
        }

    failed_counts = [0 for _ in filters]
    kept: List[Dict[str, Any]] = []

    for rec in trials:
        results = []
        for idx, flt in enumerate(filters):
            ok = _match_filter(rec, flt, drop_na=drop_na)
            results.append(ok)
            if not ok:
                failed_counts[idx] += 1

        if logic == "or":
            if any(results):
                kept.append(rec)
        else:
            if all(results):
                kept.append(rec)

    summary_filters = []
    for flt, failed in zip(filters, failed_counts):
        summary_filters.append({
            "field": flt.get("field"),
            "op": flt.get("op", "=="),
            "value": flt.get("value", flt.get("values")),
            "failed": int(failed),
        })

    summary = {
        "original": len(trials),
        "kept": len(kept),
        "filtered": len(trials) - len(kept),
        "logic": logic,
        "drop_na": drop_na,
        "filters": summary_filters,
    }

    return kept, summary


def set_preprocess_rules_tool(
    *,
    state: Any,
    ctx: Any,
    params: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Store preprocessing rules in state.extra for downstream tools.
    """
    rules_raw = params.get("rules")
    if rules_raw is None:
        raise RuntimeError("rules parameter is required")

    mode = str(params.get("mode", "set")).strip().lower()

    if mode not in ("set", "append", "clear"):
        raise RuntimeError("mode must be one of: set, append, clear")

    if mode == "clear":
        state.extra.pop("preprocess_rules", None)
        state.extra.pop("preprocess_summary", None)
        return {"status": "cleared"}

    normalized = normalize_preprocess_rules(rules_raw)

    if mode == "append":
        existing = state.extra.get("preprocess_rules")
        if existing:
            existing_norm = normalize_preprocess_rules(existing)
            merged_filters = (existing_norm.get("filters") or []) + (normalized.get("filters") or [])
            normalized = {
                "filters": merged_filters,
                "logic": normalized.get("logic", existing_norm.get("logic", "and")),
                "drop_na": normalized.get("drop_na", existing_norm.get("drop_na", True)),
            }

    state.extra["preprocess_rules"] = normalized

    return {
        "status": "ok",
        "mode": mode,
        "rules": normalized,
    }
    import os
