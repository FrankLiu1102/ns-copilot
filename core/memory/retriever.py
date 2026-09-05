"""
Memory retrieval for Code Generator.

Loads strategy_bank.json and retrieves relevant strategies
based on (modality, model) matching for injection into Code Generator prompts.
"""
import json
import os
from typing import List, Dict, Any, Optional

_STRATEGY_BANK_PATH = os.path.join(os.path.dirname(__file__), "strategy_bank.json")
_cached_bank: Optional[List[Dict[str, Any]]] = None


def _load_bank() -> List[Dict[str, Any]]:
    """Load strategy bank from JSON file (cached after first load)."""
    global _cached_bank
    if _cached_bank is not None:
        return _cached_bank
    if not os.path.isfile(_STRATEGY_BANK_PATH):
        _cached_bank = []
        return _cached_bank
    with open(_STRATEGY_BANK_PATH, "r", encoding="utf-8") as f:
        _cached_bank = json.load(f)
    return _cached_bank


def retrieve_strategies(
    modality: Optional[str] = None,
    model: Optional[str] = None,
    max_results: int = 5,
) -> List[Dict[str, Any]]:
    """
    Retrieve relevant strategies from the memory bank.

    Matching priority:
    1. Exact (modality, model) match
    2. Same modality, any model
    3. All entries (fallback)

    Args:
        modality: "eeg" or "spike"
        model: tool name like "decode_with_reve" or "vanilla"
        max_results: max entries to return

    Returns:
        List of strategy entries, sorted by primary metric descending.
    """
    bank = _load_bank()
    if not bank:
        return []

    # Score each entry
    scored = []
    for entry in bank:
        score = 0
        if modality and entry.get("modality") == modality:
            score += 10
        if model and entry.get("model") == model:
            score += 20
        scored.append((score, entry.get("metric", {}).get("value", 0), entry))

    # Sort: highest relevance score first, then by metric value
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)

    return [entry for _, _, entry in scored[:max_results]]


def format_memory_prompt(strategies: List[Dict[str, Any]]) -> str:
    """
    Format retrieved strategies into a prompt section for Code Generator.

    Returns empty string if no strategies available.
    """
    if not strategies:
        return ""

    lines = ["## Memory: Successful Strategies from Past Runs",
             "The following strategies worked well in previous analyses. "
             "Use them as reference — adapt to the current task, do not copy blindly.",
             ""]

    for i, entry in enumerate(strategies, 1):
        metric = entry.get("metric", {})
        strategy = entry.get("strategy", {})
        hint = entry.get("dataset_hint", "")

        header = f"{i}. **{entry.get('model', '?')}** ({entry.get('stage', '?')}) — {metric.get('name', '?')}={metric.get('value', '?')}"
        if hint:
            header += f"  [{hint}]"
        lines.append(header)

        if strategy.get("features"):
            lines.append(f"   - Features: {strategy['features']}")
        if strategy.get("classifier") and strategy["classifier"] != "unknown":
            lines.append(f"   - Classifier: {strategy['classifier']}")
        if strategy.get("cv") and strategy["cv"] != "unknown":
            lines.append(f"   - CV: {strategy['cv']}")
        if strategy.get("scaler"):
            lines.append(f"   - Scaler: {strategy['scaler']}")
        if strategy.get("dim_reduction"):
            lines.append(f"   - Dim reduction: {strategy['dim_reduction']}")
        if strategy.get("key_decisions"):
            lines.append(f"   - Key decisions: {', '.join(strategy['key_decisions'])}")
        lines.append("")

    return "\n".join(lines)


def get_memory_section(modality: Optional[str] = None, model: Optional[str] = None) -> str:
    """
    One-call convenience: retrieve + format memory for Code Generator prompt.
    Returns empty string if no relevant strategies found.
    """
    strategies = retrieve_strategies(modality=modality, model=model, max_results=3)
    return format_memory_prompt(strategies)
