"""
Configuration for NS-Copilot Pipeline
"""

import os

# =============================================================================
# LLM Backend Selection
# =============================================================================
# Options: "openai" (GPT-4), "ollama" (Llama, local models)
# 
# Recommendation:
#   - For public/open-source datasets: use "openai" (GPT-4) for best quality
#   - For private/sensitive datasets: use "ollama" (Llama) for local deployment
# 
LLM_BACKEND = os.getenv("LLM_BACKEND", "openai")

# =============================================================================
# OpenAI Configuration (when LLM_BACKEND="openai")
# =============================================================================
# Set via web UI (Configuration tab) or OPENAI_API_KEY environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

# =============================================================================
# Ollama Configuration (when LLM_BACKEND="ollama")
# =============================================================================
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# =============================================================================
# Model Configuration
# =============================================================================
# Default models based on backend:
#   - openai: gpt-4o (general), gpt-5.1 (advanced reasoning)
#   - ollama: llama3.1:8b, llama3.1:70b
#
# These can be overridden via environment variables.
# Each pipeline component can use a different model to balance
# cost, latency, and reasoning quality.
_DEFAULT_OPENAI_MODEL = "gpt-4o"
_DEFAULT_OPENAI_MODEL_ADVANCED = "gpt-5.2"
_DEFAULT_OLLAMA_MODEL = "llama3.1:8b"
_DEFAULT_OLLAMA_MODEL_ADVANCED = "llama3.1:70b"

# Export OPENAI_MODEL for backward compatibility
OPENAI_MODEL = os.getenv("OPENAI_MODEL", _DEFAULT_OPENAI_MODEL)

def _get_default_model() -> str:
    """Get default model based on selected backend."""
    if LLM_BACKEND.lower() == "openai":
        return _DEFAULT_OPENAI_MODEL
    return _DEFAULT_OLLAMA_MODEL

def _get_advanced_model() -> str:
    """Get advanced model for components that benefit from stronger reasoning."""
    if LLM_BACKEND.lower() == "openai":
        return _DEFAULT_OPENAI_MODEL_ADVANCED
    return _DEFAULT_OLLAMA_MODEL_ADVANCED

# ---------------------------------------------------------------------------
# Per-component model configuration
#
# Change the model for any component by setting the corresponding
# environment variable, or by editing the defaults below.
#
#   Component                    Env Variable                     Default (OpenAI)
#   ─────────                    ────────────                     ────────────────
#   Router                       ROUTER_MODEL                     gpt-5.1
#   Task Agent (QA/DA)           TASK_AGENT_MODEL                 gpt-5.1
#   Planner                      PLANNER_MODEL                    gpt-5.1
#   Code Generator               CODE_GENERATOR_MODEL             gpt-5.1
#   Controller (legacy/QA)       CONTROLLER_MODEL                 gpt-5.1
#   Controller (Diagnosis)       CONTROLLER_DIAGNOSIS_MODEL       gpt-5.1
#   Controller (Prompt Refine)   CONTROLLER_PROMPT_REFINE_MODEL   gpt-5.1
#   Analysis Agent               ANALYSIS_MODEL                   gpt-5.1
# ---------------------------------------------------------------------------
ROUTER_MODEL = os.getenv("ROUTER_MODEL", _get_advanced_model())
TASK_AGENT_MODEL = os.getenv("TASK_AGENT_MODEL", _get_advanced_model())
PLANNER_MODEL = os.getenv("PLANNER_MODEL", _get_advanced_model())
CODE_GENERATOR_MODEL = os.getenv("CODE_GENERATOR_MODEL", _get_advanced_model())
CONTROLLER_MODEL = os.getenv("CONTROLLER_MODEL", _get_advanced_model())
CONTROLLER_DIAGNOSIS_MODEL = os.getenv("CONTROLLER_DIAGNOSIS_MODEL", _get_advanced_model())
CONTROLLER_PROMPT_REFINE_MODEL = os.getenv("CONTROLLER_PROMPT_REFINE_MODEL", _get_advanced_model())
ANALYSIS_MODEL = os.getenv("ANALYSIS_MODEL", _get_advanced_model())

# ---------------------------------------------------------------------------
# DA Agent Temperature & Execution Settings
# ---------------------------------------------------------------------------
PLANNER_TEMPERATURE = float(os.getenv("PLANNER_TEMPERATURE", "0.2"))
CODE_GENERATOR_TEMPERATURE = float(os.getenv("CODE_GENERATOR_TEMPERATURE", "0.2"))
ANALYSIS_TEMPERATURE = float(os.getenv("ANALYSIS_TEMPERATURE", "0.3"))
DA_MAX_STEPS = int(os.getenv("DA_MAX_STEPS", "50"))
DA_SEED = int(os.getenv("DA_SEED", "0"))

# =============================================================================
# Legacy Configuration (backward compatibility)
# =============================================================================
USE_GPT4_FOR_NO_DATASET = os.getenv("USE_GPT4_FOR_NO_DATASET", "true").lower() == "true"
CLOUD_LLM_BACKEND = os.getenv("CLOUD_LLM_BACKEND", "gpt4")

# KB Configuration
KB_PATH = os.getenv("KB_PATH", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "knowledge_base", "index", "kb.sqlite"))

# Retrieval Configuration
KB_TOP_K = int(os.getenv("KB_TOP_K", "5"))

# Pipeline Configuration
ENABLE_DEVELOPER_MODE = os.getenv("ENABLE_DEVELOPER_MODE", "false").lower() == "true"

# =============================================================================
# Controller Mode
# =============================================================================
# Controls the Controller's behavior after initial execution:
#
#   CONTROLLER_MODE=off    — Skip Controller entirely; output initial results
#   CONTROLLER_MODE=auto   — Controller evaluates and decides whether to retry
#   CONTROLLER_MODE=force  — Controller always retries FORCE_IMPROVEMENT_COUNT times
#
#   FORCE_IMPROVEMENT_COUNT=2  — Number of forced iterations (only in "force" mode)
#
# Backward compatibility: if CONTROLLER_MODE is not set but the legacy
# FORCE_IMPROVEMENT_LOOPS=true is present, we infer "force" mode.
# =============================================================================
_legacy_force = os.getenv("FORCE_IMPROVEMENT_LOOPS", "false").lower() == "true"
_controller_mode_raw = os.getenv("CONTROLLER_MODE", "")

if _controller_mode_raw:
    CONTROLLER_MODE = _controller_mode_raw.strip().lower()
elif _legacy_force:
    CONTROLLER_MODE = "force"
else:
    CONTROLLER_MODE = "auto"

if CONTROLLER_MODE not in ("off", "auto", "force"):
    raise ValueError(
        f"Invalid CONTROLLER_MODE={CONTROLLER_MODE!r}. "
        f"Must be 'off', 'auto', or 'force'."
    )

FORCE_IMPROVEMENT_COUNT = int(os.getenv("FORCE_IMPROVEMENT_COUNT", "2"))
AUTO_MAX_IMPROVEMENT_COUNT = int(os.getenv("AUTO_MAX_IMPROVEMENT_COUNT", "2"))

# =============================================================================
# Debate MoE Configuration
# =============================================================================
# Controls the Debate-style Mixture of Experts reasoning system.
# When enabled, multiple LLM experts (Advocate, Critic, Synthesizer, Judge)
# collaborate through structured debate to produce higher-quality answers.
#
#   ENABLE_DEBATE_MOE=true   — Use debate-based reasoning for QA
#   ENABLE_DEBATE_MOE=false  — Use single-model reasoning (default)
#
# Expert models can be customized per role:
#   ADVOCATE_MODEL    — Proposes and defends answers
#   CRITIC_MODEL      — Challenges and scrutinizes
#   SYNTHESIZER_MODEL — Integrates perspectives
#   JUDGE_MODEL       — Makes final determination
#
# You can use different providers for different roles:
#   - OpenAI: gpt-4o, gpt-4-turbo, gpt-4o-mini
#   - Anthropic: claude-sonnet-4-20250514, claude-opus-4-20250514
#   - Ollama: llama3.1:8b, llama3.1:70b, mistral:7b
# =============================================================================
ENABLE_DEBATE_MOE = os.getenv("ENABLE_DEBATE_MOE", "false").lower() == "true"
DEBATE_MAX_ROUNDS = int(os.getenv("DEBATE_MAX_ROUNDS", "5"))
DEBATE_CONSENSUS_THRESHOLD = float(os.getenv("DEBATE_CONSENSUS_THRESHOLD", "0.9"))

# Expert Model Configuration
# Each expert can use a different model/provider for diverse perspectives
ADVOCATE_MODEL = os.getenv("ADVOCATE_MODEL", _get_default_model())
CRITIC_MODEL = os.getenv("CRITIC_MODEL", _get_default_model())
SYNTHESIZER_MODEL = os.getenv("SYNTHESIZER_MODEL", _get_default_model())
JUDGE_MODEL = os.getenv("JUDGE_MODEL", _get_default_model())

# Expert Temperature Configuration
# Lower temperature = more deterministic, Higher = more creative
ADVOCATE_TEMPERATURE = float(os.getenv("ADVOCATE_TEMPERATURE", "0.7"))
CRITIC_TEMPERATURE = float(os.getenv("CRITIC_TEMPERATURE", "0.7"))
SYNTHESIZER_TEMPERATURE = float(os.getenv("SYNTHESIZER_TEMPERATURE", "0.6"))
JUDGE_TEMPERATURE = float(os.getenv("JUDGE_TEMPERATURE", "0.3"))

# Anthropic API Key (optional, for Claude models)
# Set via web UI (Configuration tab) or ANTHROPIC_API_KEY environment variable
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# =============================================================================
# Metric Aliases Configuration
# =============================================================================
# Unified registry that maps canonical metric names to:
#   - aliases: possible key names in result dicts or user natural-language text
#   - direction: "higher" = higher is better, "lower" = lower is better
#   - nl_patterns: extra patterns for natural-language detection (optional)
#
# This table is used by:
#   1. _extract_metric_from_result  — finding metric values in execution results
#   2. _parse_target_metric         — detecting which metric the user is asking about
#   3. _parse_target_threshold      — extracting threshold values from user text
#
# To add support for a new metric, simply append an entry here.
# A fuzzy-match fallback in pipeline.py handles metrics not listed here.
# =============================================================================
METRIC_ALIASES = {
    "accuracy": {
        "aliases": ["accuracy", "acc", "test_accuracy", "accuracy_mean", "cv_accuracy_mean", "mean_accuracy", "final_accuracy", "accuracy_oof"],
        "direction": "higher",
    },
    "balanced_accuracy": {
        "aliases": ["balanced_accuracy", "test_balanced_accuracy", "balanced_acc", "balanced_accuracy_mean", "cv_balanced_accuracy_mean", "mean_balanced_accuracy", "final_balanced_accuracy", "balanced_accuracy_oof"],
        "direction": "higher",
    },
    "f1_macro": {
        "aliases": ["f1_macro", "test_f1_macro", "cv_f1_macro_mean", "f1_macro_mean", "macro_f1", "f1_score", "mean_f1_macro", "final_f1_macro", "f1_macro_oof"],
        "direction": "higher",
    },
    "f1_micro": {
        "aliases": ["f1_micro", "test_f1_micro", "f1_micro_mean", "cv_f1_micro_mean", "micro_f1", "mean_f1_micro", "final_f1_micro", "f1_micro_oof"],
        "direction": "higher",
    },
    "precision": {
        "aliases": ["precision", "test_precision"],
        "direction": "higher",
    },
    "recall": {
        "aliases": ["recall", "test_recall", "sensitivity"],
        "direction": "higher",
    },
    "auroc": {
        "aliases": ["auroc", "test_auroc", "auroc_mean", "cv_auroc_mean", "cv_auroc_macro_mean", "macro_auroc", "auc", "roc_auc", "auc_roc", "mean_auroc", "final_auroc", "auroc_oof"],
        "direction": "higher",
    },
    "r2_score": {
        "aliases": ["r2_score", "r2", "r_squared", "r-squared", "r squared", "r2score", "r^2"],
        "direction": "higher",
    },
    "mean_poisson_deviance": {
        "aliases": [
            "mean_poisson_deviance", "poisson_deviance",
            "poisson deviance", "mean deviance", "deviance",
        ],
        "direction": "lower",
    },
    "correlation": {
        "aliases": ["correlation", "pearson", "spearman", "rho", "pearson_r"],
        "direction": "higher",
    },
    "mse": {
        "aliases": ["mse", "mean_squared_error"],
        "direction": "lower",
    },
    "mae": {
        "aliases": ["mae", "mean_absolute_error"],
        "direction": "lower",
    },
    "log_loss": {
        "aliases": ["log_loss", "logloss", "cross_entropy"],
        "direction": "lower",
    },
}
