#!/usr/bin/env python3
"""
NS-Copilot Web Application

Web interface over the unified pipeline:
User → Task Agent → Controller → User

This release is data-analysis (DA) only. The router, KB retrieval, and the
QA/ER task agents ship in the tree but are not instantiated.
"""

import json
import os
from pathlib import Path

import gradio as gr
from neuro_copilot.core.web_adapter import WebAdapter
from neuro_copilot.core.config_manager import (
    get_config_manager,
    load_and_apply_config,
    NeuroCopilotConfig,
)


# Global state
_state = {
    "developer_mode": True,
    "adapter": None,
    "last_context": {
        "user_idea": None,
        "dataset_path": None,
        "preprocess_rules": None,
        "last_output": None,
    },
    # NEW: Diagnosis pending review
    "pending_diagnosis": None,  # Stores diagnosis data when waiting for user decision
}


def initialize_adapter():
    """Initialize web adapter"""
    if _state["adapter"] is None:
        _state["adapter"] = WebAdapter(developer_mode=_state["developer_mode"])
    else:
        _state["adapter"].developer_mode = _state["developer_mode"]


def toggle_developer_mode(enable: bool):
    """Toggle developer mode"""
    _state["developer_mode"] = enable
    initialize_adapter()
    return f"Developer Mode: {'ON' if enable else 'OFF'}"


def process_request(
    user_idea: str,
    dataset_files,
    task_type_choice: str,
    show_debug: bool
):
    """
    Process user request through pipeline
    
    Args:
        user_idea: User's question/request
        dataset_files: Uploaded dataset file(s) - can be single file, list of files, or None
        show_debug: Whether to show debug information
        
    Returns:
        Tuple of (main_output, status, debug_info)
    """
    import traceback
    import sys
    
    print("\n" + "="*80, flush=True)
    print("🌐 WEB REQUEST RECEIVED", flush=True)
    print("="*80, flush=True)
    print(f"User idea: {user_idea[:100]}...", flush=True)
    print(f"Has dataset: {dataset_files is not None}", flush=True)
    print(f"Show debug: {show_debug}", flush=True)
    
    if not user_idea.strip():
        print("⚠️ Empty input", flush=True)
        return "Please enter a question or request.", "⚠️ No input provided", "", False
    
    # Initialize adapter
    print("Initializing adapter...", flush=True)
    initialize_adapter()
    
    # Get dataset file path(s) if provided
    dataset_paths = None
    additional_paths = None

    # Fall back to DEFAULT_DATASET_PATH when nothing is uploaded
    if dataset_files is None or (isinstance(dataset_files, list) and len(dataset_files) == 0):
        default_path = os.environ.get("DEFAULT_DATASET_PATH")
        if default_path and os.path.exists(default_path):
            dataset_files = default_path
            print(f"📂 Using default dataset: {default_path}", flush=True)

    if dataset_files is not None:
        # Handle both single file and list of files
        if isinstance(dataset_files, list):
            if len(dataset_files) > 0:
                # Extract paths from file objects
                paths = []
                for f in dataset_files:
                    if isinstance(f, str):
                        paths.append(f)
                    elif hasattr(f, 'name'):
                        paths.append(f.name)
                
                if paths:
                    dataset_paths = paths[0]  # Primary path
                    if len(paths) > 1:
                        additional_paths = paths[1:]  # Additional paths
                    print(f"Dataset paths: {paths}", flush=True)
        elif isinstance(dataset_files, str):
            dataset_paths = dataset_files
            print(f"Dataset path: {dataset_paths}", flush=True)
        elif hasattr(dataset_files, 'name'):
            dataset_paths = dataset_files.name
            print(f"Dataset path: {dataset_paths}", flush=True)

        if dataset_paths:
            allowed = (".mat", ".nwb", ".h5", ".npy", ".csv", ".set", ".tar.gz", ".tgz", ".zip")
            lower_name = dataset_paths.lower()
            # Allow directories (e.g., BIDS datasets) to pass validation
            if not os.path.isdir(dataset_paths) and not lower_name.endswith(allowed):
                err = f"Invalid file type. Allowed: {', '.join(allowed)}"
                print(err, flush=True)
                return err, "❌ Dataset type error", "", False

            # Validate additional paths if any
            if additional_paths:
                for ap in additional_paths:
                    if not os.path.isdir(ap) and not ap.lower().endswith(allowed):
                        err = f"Invalid file type for {os.path.basename(ap)}. Allowed: {', '.join(allowed)}"
                        print(err, flush=True)
                        return err, "❌ Dataset type error", "", False
    
    preprocess_rules = None

    # Process request
    try:
        print("Calling adapter.process_request()...", flush=True)
        sys.stdout.flush()
        sys.stderr.flush()
        
        task_type_override = None
        if task_type_choice and task_type_choice != "Auto":
            task_type_override = task_type_choice
        response = _state["adapter"].process_request(
            user_idea=user_idea,
            dataset_file=dataset_paths,
            additional_files=additional_paths,
            preprocess_rules=preprocess_rules,
            task_type_override=task_type_override,
        )
        
        print("✅ Adapter returned successfully", flush=True)
        print(f"Status: {response.status_message}", flush=True)
        print(f"Output length: {len(response.main_output)} chars", flush=True)
        
        # Store visualization paths for later use
        viz_paths = response.visualization_paths or []
        _state["last_visualization_paths"] = viz_paths
        if viz_paths:
            print(f"Visualizations found: {len(viz_paths)}", flush=True)
            for vp in viz_paths:
                print(f"  - {vp}", flush=True)
        
        # Check if diagnosis needs user review
        if response.needs_user_review:
            print(f"⚠️ Diagnosis needs user review: {response.diagnosis_summary[:100]}...", flush=True)
            # Extract previous plan and metrics from diagnosis_raw
            diagnosis_raw = response.diagnosis_raw or {}
            previous_plan = diagnosis_raw.get("plan")
            exec_report = diagnosis_raw.get("exec_report", {})
            
            # Extract metrics from execution report
            metrics = {}
            for step in exec_report.get("steps", []):
                out_path = step.get("output_path")
                if out_path:
                    try:
                        import json
                        from pathlib import Path
                        out_json = json.loads(Path(out_path).read_text(encoding="utf-8"))
                        step_metrics = out_json.get("metrics", {})
                        if step_metrics:
                            metrics.update(step_metrics)
                    except Exception:
                        pass
            
            _state["pending_diagnosis"] = {
                "summary": response.diagnosis_summary,
                "recommendations": response.diagnosis_recommendations,
                "suggested_action": response.suggested_action,
                "raw": response.diagnosis_raw,
                "dataset_path": dataset_paths,
                "user_idea": user_idea,
                "dataset_metadata": response.dataset_metadata,  # Include dataset metadata
                "previous_plan": previous_plan,  # NEW: Store previous plan for retry
                "metrics": metrics,  # NEW: Store metrics for retry
                "attempt_number": 1,  # Track attempt count
            }
        else:
            _state["pending_diagnosis"] = None
        
        # Format debug info if requested
        debug_output = ""
        if show_debug:
            debug_output = _state["adapter"].format_debug_display(response.debug_info)
        
        print("Returning response to Gradio...", flush=True)
        if dataset_paths is not None:
            _state["last_context"] = {
                "user_idea": user_idea,
                "dataset_path": dataset_paths,
                "preprocess_rules": None,
                "last_output": response.main_output,
                "dataset_metadata": response.dataset_metadata,  # NEW: Save dataset metadata
            }
        return response.main_output, response.status_message, debug_output, response.needs_user_review
        
    except Exception as e:
        print("="*80, flush=True)
        print("❌ EXCEPTION CAUGHT IN WEB LAYER", flush=True)
        print("="*80, flush=True)
        print(f"Exception type: {type(e).__name__}", flush=True)
        print(f"Exception message: {str(e)}", flush=True)
        print("\nFull traceback:", flush=True)
        traceback.print_exc()
        print("="*80, flush=True)
        sys.stdout.flush()
        sys.stderr.flush()
        
        error_msg = f"❌ Error: {str(e)}"
        return error_msg, "❌ Pipeline error", "", False


def _looks_like_metric_reply(text: str) -> bool:
    if not text:
        return False
    lower = text.lower()
    keys = [
        "r2_score",
        "r2",
        "r^2",
        "r-squared",
        "mean_poisson_deviance",
        "poisson deviance",
        "deviance",
        "correlation",
        "corr",
        "pearson",
        "spearman",
    ]
    return any(k in lower for k in keys)


def create_config_tab():
    """Create the Configuration tab UI with sub-tabs: General, QA Settings, DA Settings"""
    
    # Model choices for dropdowns
    openai_models = ["gpt-5.1", "gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo", "o1-preview", "o1-mini"]
    anthropic_models = ["claude-sonnet-4-20250514", "claude-opus-4-20250514", "claude-3-5-sonnet-latest"]
    ollama_models = ["llama3.1:8b", "llama3.1:70b", "mistral:7b", "mixtral:8x7b", "qwen2:7b"]
    all_models = openai_models + anthropic_models + ollama_models
    
    with gr.Column():
        gr.Markdown("## ⚙️ System Configuration for NS-Copilot")
        gr.Markdown("Configure API keys, QA reasoning, and Data Analysis settings. Changes are saved and applied immediately.")
        
        with gr.Tabs() as config_sub_tabs:
            # =====================================================================
            # SUB-TAB 1: General (API Keys & Endpoints)
            # =====================================================================
            with gr.Tab("🔑 General", id="general_tab"):
                with gr.Row():
                    # === API Keys Section ===
                    with gr.Column(scale=1):
                        gr.Markdown("### 🔑 API Keys")
                        
                        openai_key_input = gr.Textbox(
                            label="OpenAI API Key",
                            type="password",
                            placeholder="sk-...",
                            info="Required for GPT-4 models"
                        )
                        openai_key_status = gr.Textbox(
                            label="Status",
                            interactive=False,
                            value="Not validated",
                            max_lines=1
                        )
                        validate_openai_btn = gr.Button("Validate OpenAI Key", size="sm")
                        
                        gr.Markdown("---")
                        
                        anthropic_key_input = gr.Textbox(
                            label="Anthropic API Key",
                            type="password",
                            placeholder="sk-ant-...",
                            info="Required for Claude models"
                        )
                        anthropic_key_status = gr.Textbox(
                            label="Status",
                            interactive=False,
                            value="Not validated",
                            max_lines=1
                        )
                        validate_anthropic_btn = gr.Button("Validate Anthropic Key", size="sm")
                    
                    # === Base URLs Section ===
                    with gr.Column(scale=1):
                        gr.Markdown("### 🌐 API Endpoints")
                        
                        openai_base_url = gr.Textbox(
                            label="OpenAI Base URL",
                            value="https://api.openai.com/v1",
                            info="Change for Azure or proxy"
                        )
                        
                        ollama_base_url = gr.Textbox(
                            label="Ollama Base URL",
                            value="http://localhost:11434",
                            info="Local Ollama server"
                        )
            
            # =====================================================================
            # SUB-TAB 2: QA Settings (Debate MoE + Pipeline)
            # =====================================================================
            with gr.Tab("💬 QA Settings", id="qa_tab"):
                # === Debate MoE Section ===
                gr.Markdown("### 🎭 Debate MoE Configuration")
                gr.Markdown("Enable multi-expert reasoning where different LLMs debate to produce better answers.")
                
                with gr.Row():
                    debate_moe_enabled = gr.Checkbox(
                        label="Enable Debate MoE",
                        value=False,
                        info="Use multi-expert debate for QA tasks"
                    )
                    debate_max_rounds = gr.Slider(
                        label="Max Rounds",
                        minimum=3,
                        maximum=7,
                        step=1,
                        value=5,
                        info="Number of debate rounds"
                    )
                    debate_consensus_threshold = gr.Slider(
                        label="Consensus Threshold",
                        minimum=0.5,
                        maximum=1.0,
                        step=0.05,
                        value=0.9,
                        info="Confidence threshold for consensus"
                    )
                
                with gr.Row():
                    advocate_model = gr.Dropdown(
                        label="Advocate Model",
                        choices=all_models,
                        value="gpt-4o",
                        info="Proposes and defends answers",
                        allow_custom_value=True
                    )
                    critic_model = gr.Dropdown(
                        label="Critic Model",
                        choices=all_models,
                        value="claude-sonnet-4-20250514",
                        info="Challenges and scrutinizes",
                        allow_custom_value=True
                    )
                    synthesizer_model = gr.Dropdown(
                        label="Synthesizer Model",
                        choices=all_models,
                        value="gpt-4o",
                        info="Integrates perspectives",
                        allow_custom_value=True
                    )
                    judge_model = gr.Dropdown(
                        label="Judge Model",
                        choices=all_models,
                        value="gpt-4o",
                        info="Makes final determination",
                        allow_custom_value=True
                    )
                
                with gr.Accordion("Advanced: Temperature Settings", open=False):
                    with gr.Row():
                        advocate_temp = gr.Slider(label="Advocate Temp", minimum=0, maximum=1, step=0.1, value=0.7)
                        critic_temp = gr.Slider(label="Critic Temp", minimum=0, maximum=1, step=0.1, value=0.7)
                        synthesizer_temp = gr.Slider(label="Synthesizer Temp", minimum=0, maximum=1, step=0.1, value=0.6)
                        judge_temp = gr.Slider(label="Judge Temp", minimum=0, maximum=1, step=0.1, value=0.3)
                
                gr.Markdown("---")
                
                # === QA Pipeline Settings ===
                gr.Markdown("### 🔧 QA Pipeline Settings")
                
                with gr.Row():
                    router_model = gr.Dropdown(
                        label="Router Model",
                        choices=all_models,
                        value="gpt-4o",
                        info="Task classification",
                        allow_custom_value=True
                    )
                    task_agent_model = gr.Dropdown(
                        label="Task Agent Model",
                        choices=all_models,
                        value="gpt-4o",
                        info="Main reasoning (when not using Debate MoE)",
                        allow_custom_value=True
                    )
                    controller_model = gr.Dropdown(
                        label="Controller Model",
                        choices=all_models,
                        value="gpt-4o",
                        info="Quality validation",
                        allow_custom_value=True
                    )
                
                with gr.Row():
                    kb_top_k = gr.Slider(
                        label="KB Top-K",
                        minimum=1,
                        maximum=20,
                        step=1,
                        value=5,
                        info="Number of papers to retrieve"
                    )
                    enable_online_search = gr.Checkbox(
                        label="Enable Online Search",
                        value=True,
                        info="Search PubMed for additional context"
                    )
                    online_search_top_k = gr.Slider(
                        label="Online Search Top-K",
                        minimum=1,
                        maximum=10,
                        step=1,
                        value=3,
                        info="Number of online papers to retrieve"
                    )
            
            # =====================================================================
            # SUB-TAB 3: DA Settings (Data Analysis)
            # =====================================================================
            with gr.Tab("📊 DA Settings", id="da_tab"):
                gr.Markdown("### 📊 Data Analysis Pipeline Configuration")
                gr.Markdown("Configure the LLM backbones, temperatures, and execution parameters for the Data Analysis pipeline.")
                
                # === DA Agent Models ===
                gr.Markdown("#### Agent LLM Backbones")
                
                with gr.Row():
                    da_planner_model = gr.Dropdown(
                        label="Planner Agent LLM Backbone",
                        choices=all_models,
                        value="gpt-5.1",
                        info="Plans analysis steps from user request",
                        allow_custom_value=True
                    )
                    da_code_gen_model = gr.Dropdown(
                        label="Code Generator Agent LLM Backbone",
                        choices=all_models,
                        value="gpt-5.1",
                        info="Generates Python code for each step",
                        allow_custom_value=True
                    )

                with gr.Row():
                    da_controller_diag_model = gr.Dropdown(
                        label="Controller (Diagnosis) Agent LLM Backbone",
                        choices=all_models,
                        value="gpt-5.1",
                        info="Diagnoses execution results and identifies anomalies",
                        allow_custom_value=True
                    )
                    da_controller_refine_model = gr.Dropdown(
                        label="Controller (Prompt Refine) Agent LLM Backbone",
                        choices=all_models,
                        value="gpt-5.1",
                        info="Refines prompt with diagnosis recommendations for retry",
                        allow_custom_value=True
                    )
                    da_analysis_model = gr.Dropdown(
                        label="Analysis Agent LLM Backbone",
                        choices=all_models,
                        value="gpt-5.1",
                        info="Interprets results and generates reports",
                        allow_custom_value=True
                    )
                
                gr.Markdown("#### Temperature Settings")
                
                with gr.Row():
                    da_planner_temp = gr.Slider(
                        label="Planner Temperature",
                        minimum=0,
                        maximum=1,
                        step=0.1,
                        value=0.2,
                        info="Lower = more deterministic plans"
                    )
                    da_code_gen_temp = gr.Slider(
                        label="Code Generator Temperature",
                        minimum=0,
                        maximum=1,
                        step=0.1,
                        value=0.2,
                        info="Lower = more reliable code"
                    )
                
                with gr.Row():
                    da_controller_diag_temp = gr.Slider(
                        label="Controller (Diagnosis) Temperature",
                        minimum=0,
                        maximum=1,
                        step=0.1,
                        value=0.1,
                        info="Lower = more consistent diagnosis"
                    )
                    da_controller_refine_temp = gr.Slider(
                        label="Controller (Prompt Refine) Temperature",
                        minimum=0,
                        maximum=1,
                        step=0.1,
                        value=0.2,
                        info="Lower = more faithful prompt refinement"
                    )
                    da_analysis_temp = gr.Slider(
                        label="Analysis Temperature",
                        minimum=0,
                        maximum=1,
                        step=0.1,
                        value=0.3,
                        info="Lower = more factual interpretation"
                    )
                
                gr.Markdown("---")
                
                # === Execution Settings ===
                gr.Markdown("#### Execution Settings")
                
                with gr.Row():
                    da_max_steps = gr.Slider(
                        label="Max Execution Steps",
                        minimum=1,
                        maximum=100,
                        step=1,
                        value=50,
                        info="Maximum number of analysis steps to execute"
                    )
                    da_seed = gr.Number(
                        label="Random Seed",
                        value=0,
                        precision=0,
                        info="Seed for reproducibility (0 = default)"
                    )

                gr.Markdown("---")

                # === Controller Settings ===
                gr.Markdown("#### Controller Settings")
                gr.Markdown(
                    "The Controller evaluates execution results after each run and decides "
                    "whether to retry with improvements. Without Controller, only the Initial "
                    "round runs (all models execute once, no early stopping)."
                )

                da_controller_enabled = gr.Checkbox(
                    label="Enable Controller",
                    value=True,
                    info="Enable the Controller for retry rounds after the Initial run."
                )

                with gr.Group(visible=True) as da_controller_mode_group:
                    da_controller_retry_mode = gr.Radio(
                        label="Controller Mode",
                        choices=["Auto Mode", "Force Mode"],
                        value="Auto Mode",
                    )
                    gr.Markdown(
                        "<small>"
                        "**Auto Mode**: After the Initial round, if any model exceeds the Codex baseline → early stop. "
                        "Otherwise, the Controller reorders models and retries. Early stopping triggers whenever a model "
                        "beats the baseline in any retry round.<br>"
                        "**Force Mode**: Run a fixed number of retry rounds regardless of results (no early stopping). "
                        "Useful for ablation experiments."
                        "</small>",
                        elem_id="controller-mode-desc",
                    )
                    da_controller_retry_count = gr.Number(
                        label="Max Retry Rounds",
                        value=2,
                        minimum=1,
                        maximum=10,
                        step=1,
                        precision=0,
                        info="Number of retry rounds after Initial. In Auto Mode, early stopping may end before this limit."
                    )

                # Hidden components for backward compatibility with save/load
                da_controller_force_count = gr.Number(value=2, visible=False)
                da_controller_auto_max_count = gr.Number(value=2, visible=False)

                # --- Visibility logic ---
                def _toggle_controller(enabled):
                    return gr.update(visible=enabled)

                da_controller_enabled.change(
                    fn=_toggle_controller,
                    inputs=[da_controller_enabled],
                    outputs=[da_controller_mode_group],
                )

                # Sync retry count to hidden components
                da_controller_retry_count.change(
                    fn=lambda v: (gr.update(value=v), gr.update(value=v)),
                    inputs=[da_controller_retry_count],
                    outputs=[da_controller_force_count, da_controller_auto_max_count],
                )
        
        gr.Markdown("---")
        
        # === Save/Load Buttons (shared across all sub-tabs) ===
        with gr.Row():
            save_config_btn = gr.Button("💾 Save Configuration", variant="primary", size="lg")
            load_config_btn = gr.Button("📂 Load Saved Config", variant="secondary", size="lg")
        
        config_status = gr.Textbox(
            label="Configuration Status",
            interactive=False,
            value="Configuration not saved yet",
            max_lines=2
        )
    
    # === Event Handlers ===
    
    def validate_openai_key(key):
        if not key or len(key) < 10:
            return "❌ API key is empty or too short"
        manager = get_config_manager()
        valid, msg = manager.validate_api_key("openai", key)
        return f"{'✅' if valid else '❌'} {msg}"
    
    def validate_anthropic_key(key):
        if not key or len(key) < 10:
            return "❌ API key is empty or too short"
        manager = get_config_manager()
        valid, msg = manager.validate_api_key("anthropic", key)
        return f"{'✅' if valid else '❌'} {msg}"
    
    def clean_api_key(key: str) -> str:
        """Clean API key by removing invisible Unicode characters"""
        import re
        if not key:
            return ""
        key = key.strip()
        key = re.sub(r'[\u2028\u2029\ufeff\u200b\u200c\u200d\xa0]', '', key)
        key = ''.join(char for char in key if ord(char) < 128)
        return key.strip()
    
    def save_configuration(
        openai_key, anthropic_key, openai_url, ollama_url,
        debate_enabled, max_rounds, consensus_thresh,
        adv_model, crit_model, synth_model, jdg_model,
        adv_temp, crit_temp, synth_temp, jdg_temp,
        rtr_model, task_model, ctrl_model,
        kb_k, online_search, online_k,
        planner_model, code_gen_model,
        ctrl_diag_model, ctrl_refine_model, analysis_model,
        planner_temp, code_gen_temp,
        ctrl_diag_temp, ctrl_refine_temp, analysis_temp,
        max_steps, seed,
        controller_enabled, controller_retry_mode,
        controller_force_count, controller_auto_max_count,
    ):
        try:
            from neuro_copilot.core.config_manager import DAConfig
            manager = get_config_manager()
            
            # Build config object
            config = NeuroCopilotConfig()
            
            # API settings (clean invisible characters)
            config.api.openai_api_key = clean_api_key(openai_key)
            config.api.anthropic_api_key = clean_api_key(anthropic_key)
            config.api.openai_base_url = openai_url
            config.api.ollama_base_url = ollama_url
            
            # Debate MoE settings
            config.debate_moe.enabled = debate_enabled
            config.debate_moe.max_rounds = int(max_rounds)
            config.debate_moe.consensus_threshold = float(consensus_thresh)
            config.debate_moe.advocate_model = adv_model
            config.debate_moe.critic_model = crit_model
            config.debate_moe.synthesizer_model = synth_model
            config.debate_moe.judge_model = jdg_model
            config.debate_moe.advocate_temperature = float(adv_temp)
            config.debate_moe.critic_temperature = float(crit_temp)
            config.debate_moe.synthesizer_temperature = float(synth_temp)
            config.debate_moe.judge_temperature = float(jdg_temp)
            
            # Pipeline settings
            config.pipeline.router_model = rtr_model
            config.pipeline.task_agent_model = task_model
            config.pipeline.controller_model = ctrl_model
            config.pipeline.kb_top_k = int(kb_k)
            config.pipeline.enable_online_search = online_search
            config.pipeline.online_search_top_k = int(online_k)
            
            # Derive controller_mode from UI controls
            # Controller enabled + Auto Mode → "auto"
            # Controller enabled + Force Mode → "force"
            # Controller disabled → "off"
            if not controller_enabled:
                _ctrl_mode = "off"
            elif controller_retry_mode == "Force Mode":
                _ctrl_mode = "force"
            else:
                _ctrl_mode = "auto"

            # DA settings
            config.da = DAConfig(
                planner_model=planner_model,
                planner_temperature=float(planner_temp),
                code_generator_model=code_gen_model,
                code_generator_temperature=float(code_gen_temp),
                controller_diagnosis_model=ctrl_diag_model,
                controller_diagnosis_temperature=float(ctrl_diag_temp),
                controller_prompt_refine_model=ctrl_refine_model,
                controller_prompt_refine_temperature=float(ctrl_refine_temp),
                analysis_model=analysis_model,
                analysis_temperature=float(analysis_temp),
                max_steps=int(max_steps),
                seed=int(seed),
                controller_mode=_ctrl_mode,
                controller_force_count=int(controller_force_count),
                controller_auto_max_count=int(controller_auto_max_count),
            )
            
            # Save and apply
            if manager.save(config):
                manager.apply(config)
                
                # Reset the adapter to pick up new config
                _state["adapter"] = None
                
                status_parts = []
                status_parts.append(f"Debate MoE: {'Enabled' if debate_enabled else 'Disabled'}")
                status_parts.append(f"DA Planner: {planner_model}")
                status_parts.append(f"Controller: {_ctrl_mode}")
                return f"✅ Configuration saved and applied successfully!\n" + " | ".join(status_parts)
            else:
                return "❌ Failed to save configuration"
                
        except Exception as e:
            return f"❌ Error saving configuration: {str(e)}"
    
    def load_saved_configuration():
        try:
            manager = get_config_manager()
            config = manager.load()
            
            return (
                # General
                config.api.openai_api_key,
                config.api.anthropic_api_key,
                config.api.openai_base_url,
                config.api.ollama_base_url,
                # QA - Debate MoE
                config.debate_moe.enabled,
                config.debate_moe.max_rounds,
                config.debate_moe.consensus_threshold,
                config.debate_moe.advocate_model,
                config.debate_moe.critic_model,
                config.debate_moe.synthesizer_model,
                config.debate_moe.judge_model,
                config.debate_moe.advocate_temperature,
                config.debate_moe.critic_temperature,
                config.debate_moe.synthesizer_temperature,
                config.debate_moe.judge_temperature,
                # QA - Pipeline
                config.pipeline.router_model,
                config.pipeline.task_agent_model,
                config.pipeline.controller_model,
                config.pipeline.kb_top_k,
                getattr(config.pipeline, 'enable_online_search', True),
                getattr(config.pipeline, 'online_search_top_k', 3),
                # DA
                getattr(config.da, 'planner_model', 'gpt-4o'),
                getattr(config.da, 'code_generator_model', 'gpt-4o'),
                getattr(config.da, 'controller_diagnosis_model', 'gpt-4o'),
                getattr(config.da, 'controller_prompt_refine_model', 'gpt-4o'),
                getattr(config.da, 'analysis_model', 'gpt-4o'),
                getattr(config.da, 'planner_temperature', 0.2),
                getattr(config.da, 'code_generator_temperature', 0.2),
                getattr(config.da, 'controller_diagnosis_temperature', 0.1),
                getattr(config.da, 'controller_prompt_refine_temperature', 0.2),
                getattr(config.da, 'analysis_temperature', 0.3),
                getattr(config.da, 'max_steps', 50),
                getattr(config.da, 'seed', 0),
                # Controller settings — derive UI state from controller_mode
                getattr(config.da, 'controller_mode', 'auto') != "off",  # enabled checkbox
                "Force Mode" if getattr(config.da, 'controller_mode', 'auto') == "force" else "Auto Mode",  # radio
                getattr(config.da, 'controller_force_count', 2),  # retry count (hidden)
                getattr(config.da, 'controller_auto_max_count', 2),  # retry count (hidden)
                # Status
                "✅ Configuration loaded successfully"
            )
        except Exception as e:
            return (
                "", "", "https://api.openai.com/v1", "http://localhost:11434",
                False, 5, 0.9,
                "gpt-4o", "claude-sonnet-4-20250514", "gpt-4o", "gpt-4o",
                0.7, 0.7, 0.6, 0.3,
                "gpt-4o", "gpt-4o", "gpt-4o",
                5, True, 3,
                "gpt-4o", "gpt-4o", "gpt-4o", "gpt-4o", "gpt-4o",
                0.2, 0.2, 0.1, 0.2, 0.3,
                50, 0,
                # Controller defaults
                True, "Auto Mode",
                2, 2,
                f"⚠️ Could not load config: {str(e)}"
            )
    
    # Wire up events
    validate_openai_btn.click(
        fn=validate_openai_key,
        inputs=[openai_key_input],
        outputs=[openai_key_status]
    )
    
    validate_anthropic_btn.click(
        fn=validate_anthropic_key,
        inputs=[anthropic_key_input],
        outputs=[anthropic_key_status]
    )
    
    save_config_btn.click(
        fn=save_configuration,
        inputs=[
            openai_key_input, anthropic_key_input, openai_base_url, ollama_base_url,
            debate_moe_enabled, debate_max_rounds, debate_consensus_threshold,
            advocate_model, critic_model, synthesizer_model, judge_model,
            advocate_temp, critic_temp, synthesizer_temp, judge_temp,
            router_model, task_agent_model, controller_model,
            kb_top_k, enable_online_search, online_search_top_k,
            da_planner_model, da_code_gen_model,
            da_controller_diag_model, da_controller_refine_model, da_analysis_model,
            da_planner_temp, da_code_gen_temp,
            da_controller_diag_temp, da_controller_refine_temp, da_analysis_temp,
            da_max_steps, da_seed,
            da_controller_enabled, da_controller_retry_mode,
            da_controller_force_count, da_controller_auto_max_count,
        ],
        outputs=[config_status]
    )
    
    load_config_btn.click(
        fn=load_saved_configuration,
        inputs=[],
        outputs=[
            openai_key_input, anthropic_key_input, openai_base_url, ollama_base_url,
            debate_moe_enabled, debate_max_rounds, debate_consensus_threshold,
            advocate_model, critic_model, synthesizer_model, judge_model,
            advocate_temp, critic_temp, synthesizer_temp, judge_temp,
            router_model, task_agent_model, controller_model,
            kb_top_k, enable_online_search, online_search_top_k,
            da_planner_model, da_code_gen_model,
            da_controller_diag_model, da_controller_refine_model, da_analysis_model,
            da_planner_temp, da_code_gen_temp,
            da_controller_diag_temp, da_controller_refine_temp, da_analysis_temp,
            da_max_steps, da_seed,
            da_controller_enabled, da_controller_retry_mode,
            da_controller_force_count, da_controller_auto_max_count,
            config_status
        ]
    )

    return {
        "load_fn": load_saved_configuration,
        "components": [
            openai_key_input, anthropic_key_input, openai_base_url, ollama_base_url,
            debate_moe_enabled, debate_max_rounds, debate_consensus_threshold,
            advocate_model, critic_model, synthesizer_model, judge_model,
            advocate_temp, critic_temp, synthesizer_temp, judge_temp,
            router_model, task_agent_model, controller_model,
            kb_top_k, enable_online_search, online_search_top_k,
            da_planner_model, da_code_gen_model,
            da_controller_diag_model, da_controller_refine_model, da_analysis_model,
            da_planner_temp, da_code_gen_temp,
            da_controller_diag_temp, da_controller_refine_temp, da_analysis_temp,
            da_max_steps, da_seed,
            da_controller_enabled, da_controller_retry_mode,
            da_controller_force_count, da_controller_auto_max_count,
            config_status
        ]
    }


def create_web_interface():
    """Create Gradio web interface"""
    print("Creating Gradio interface...", flush=True)
    
    # Load and apply saved configuration on startup
    try:
        load_and_apply_config()
        print("✅ Loaded saved configuration", flush=True)
    except Exception as e:
        print(f"⚠️ Could not load saved config: {e}", flush=True)
    
    custom_css = """
    /* Footer repositioned by JS */
    /* Space for icons moved above chat box by JS */
    #chat-history {
        margin-top: 28px !important;
    }
    /* File indicator styling */
    #file-indicator {
        font-size: 13px;
        color: #555;
        padding: 2px 8px;
        min-height: 0 !important;
    }
    #file-indicator .prose {
        font-size: 13px !important;
    }
    /* Textarea auto-resize constraints */
    #chat-input-bar textarea {
        max-height: 50vh !important;
        overflow-y: auto !important;
    }
    """
    with gr.Blocks(title="NS-Copilot", css=custom_css) as app:
        welcome_message = (
            "Hello! I'm NS-Copilot, an autonomous neuroscience data analysis agent.\n\n"
            "Upload your neural dataset (EEG or spike data) and describe your analysis task. "
            "I will automatically select the best pretrained foundation model, generate analysis code, "
            "and optimize results through iterative refinement.\n\n"
            "**How to use:**\n"
            "1. Upload your dataset files (CSV, .set, .nwb, .mat, etc.)\n"
            "2. Describe your analysis task in natural language\n"
            "3. NS-Copilot handles the rest — model selection, code generation, and evaluation"
        )
        welcome_history = [{"role": "assistant", "content": welcome_message}]

        with gr.Tabs() as tabs:
            # === MAIN CHAT TAB ===
            with gr.Tab("🤖 NS-Copilot", id="main_tab"):
                with gr.Column(elem_id="chat-container"):
                    gr.Markdown("## 🤖 NS-Copilot")
                    chat_history = gr.Chatbot(label="History", show_label=False, height=600, value=welcome_history, elem_id="chat-history")
                    
                    # Gallery for displaying generated visualizations
                    with gr.Accordion("📊 Generated Visualizations", open=False, visible=False) as viz_accordion:
                        viz_gallery = gr.Gallery(
                            label="Analysis Visualizations",
                            show_label=False,
                            columns=2,
                            height=300,
                            object_fit="contain"
                        )
                    
                    # NEW: Diagnosis review UI (hidden by default)
                    with gr.Column(visible=False) as diagnosis_container:
                        gr.Markdown("### 🔍 Potential issues detected in analysis results. Would you like to try to improve?")
                        
                        # Recommended fix as a clickable button (styled as a card)
                        recommended_fix_btn = gr.Button(
                            "💡 Click to apply recommended fix",
                            variant="primary",
                            size="lg",
                            elem_classes=["fix-recommendation-btn"]
                        )
                        
                        # No button below
                        no_btn = gr.Button("❌ No - Accept current results", variant="secondary")
                    
                    debug_output = gr.Markdown(label="🔍 Debug Information", visible=False)

                    with gr.Column(elem_id="chat-input-bar"):
                        with gr.Row():
                            task_type_input = gr.Dropdown(
                                label="Task Type",
                                choices=["Auto", "QA", "DA", "ER"],
                                value="DA",
                                interactive=True,
                                scale=1,
                                min_width=80,
                            )
                            multimodal_input = gr.MultimodalTextbox(
                                placeholder="Describe your request...",
                                file_count="multiple",
                                lines=1,
                                max_lines=100,
                                scale=8,
                                show_label=False,
                                submit_btn="➤",
                                sources=["upload"],
                            )
                        file_indicator = gr.Markdown(value="", visible=False, elem_id="file-indicator")
                        # Hidden components to keep compatibility with existing handlers
                        dataset_input = gr.File(visible=False, file_count="multiple")
                        user_idea_input = gr.Textbox(visible=False)
                        submit_btn = gr.Button(visible=False)

                        developer_mode_checkbox = gr.Checkbox(
                            label="🔍 Developer Mode (Show pipeline internals)",
                            value=False,
                            visible=False
                        )
                        developer_mode_status = gr.Textbox(
                            label="Developer Mode Status",
                            value="Developer Mode: OFF",
                            interactive=False,
                            visible=False
                        )
            
            # === CONFIGURATION TAB (same level as main tab) ===
            with gr.Tab("⚙️ Configuration", id="config_tab"):
                config_tab_data = create_config_tab()
        
        # Event handlers (outside of Tabs context)
        def on_submit(multimodal_data, dataset_files_unused, task_type_choice, show_debug, history):
            """Generator that streams progress events in real-time, then yields final result."""
            import threading
            import time

            # Extract text and files from MultimodalTextbox
            user_idea = ""
            dataset_files = None
            if isinstance(multimodal_data, dict):
                user_idea = multimodal_data.get("text", "")
                files = multimodal_data.get("files", [])
                if files:
                    # Deduplicate by filename (keep first occurrence)
                    seen_names = set()
                    deduped = []
                    for f in files:
                        path = f if isinstance(f, str) else f.get("path", "")
                        name = Path(path).name if path else ""
                        if name and name not in seen_names:
                            seen_names.add(name)
                            deduped.append(path)
                    dataset_files = deduped
            elif isinstance(multimodal_data, str):
                user_idea = multimodal_data

            # Original logic follows
            ctx = _state.get("last_context", {})
            reused_dataset = False
            if (dataset_files is None or (isinstance(dataset_files, list) and len(dataset_files) == 0)) and ctx.get("dataset_path"):
                dataset_files = ctx.get("dataset_path")
                reused_dataset = True
            base_user_idea = ctx.get("user_idea") or ""
            exec_user_idea = user_idea
            if _looks_like_metric_reply(user_idea) and base_user_idea:
                exec_user_idea = f"{base_user_idea}\n{user_idea}"

            history = history or []

            # Build user message
            user_lines = [user_idea.strip()]
            if dataset_files is not None and not reused_dataset:
                if isinstance(dataset_files, list):
                    file_names = []
                    for f in dataset_files:
                        name = f.name if hasattr(f, "name") else str(f)
                        file_names.append(Path(name).name)
                    user_lines.append(f"[datasets] {', '.join(file_names)}")
                else:
                    name = dataset_files.name if hasattr(dataset_files, "name") else str(dataset_files)
                    user_lines.append(f"[dataset] {Path(name).name}")
            user_message = "\n".join([line for line in user_lines if line])

            # Add user message + "thinking" indicator
            history = history + [{"role": "user", "content": user_message}]
            history = history + [{"role": "assistant", "content": "⏳ Processing your request..."}]

            # Yield initial state: show user message + thinking indicator
            yield (
                history,
                gr.update(value=None),  # clear multimodal_input
                None,  # clear hidden dataset_input
                gr.update(),  # debug_output unchanged
                gr.update(),  # viz_accordion unchanged
                gr.update(),  # viz_gallery unchanged
                gr.update(),  # diagnosis_container unchanged
                gr.update(),  # recommended_fix_btn unchanged
                gr.update(value="", visible=False),  # clear file_indicator
            )

            # Run pipeline in background thread
            result_holder = {"output": None, "status": None, "debug": None, "needs_review": None, "done": False}

            def _run_pipeline():
                try:
                    result_holder["output"], result_holder["status"], result_holder["debug"], result_holder["needs_review"] = process_request(
                        exec_user_idea, dataset_files, task_type_choice, show_debug
                    )
                except Exception as e:
                    import traceback
                    result_holder["output"] = f"❌ Error: {e}\n\n{traceback.format_exc()}"
                    result_holder["status"] = "❌ Pipeline error"
                    result_holder["debug"] = ""
                    result_holder["needs_review"] = False
                finally:
                    result_holder["done"] = True

            pipeline_thread = threading.Thread(target=_run_pipeline, daemon=True)
            pipeline_thread.start()

            # Poll for progress events and yield updates
            from neuro_copilot.core.progress_events import get_tracker
            last_event_count = 0

            while not result_holder["done"]:
                time.sleep(0.5)
                current_count = get_tracker().event_count()
                if current_count > last_event_count:
                    # New events arrived — rebuild history with all events so far
                    events = get_tracker().get_events()
                    # Remove the "thinking" placeholder, rebuild with events
                    streaming_history = history[:-1]  # remove "⏳ Processing..."
                    for evt in events:
                        streaming_history = streaming_history + [
                            {"role": "assistant", "content": f"**{evt.title}**\n\n{evt.content}"}
                        ]
                    # Add thinking indicator at the end
                    streaming_history = streaming_history + [
                        {"role": "assistant", "content": "⏳ Processing..."}
                    ]
                    last_event_count = current_count
                    yield (
                        streaming_history,
                        gr.update(),  # multimodal_input already cleared
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                    )

            # Pipeline complete — wait for thread to finish
            pipeline_thread.join(timeout=5)

            if _looks_like_metric_reply(user_idea) and base_user_idea and ctx.get("dataset_path"):
                _state["last_context"]["user_idea"] = base_user_idea

            output = result_holder["output"]
            status = result_holder["status"]
            debug = result_holder["debug"]
            needs_review = result_holder["needs_review"]

            assistant_text = output
            if status:
                assistant_text = f"{assistant_text}\n\n_Status: {status}_"
            if show_debug and debug and debug.strip():
                assistant_text = f"{assistant_text}\n\n{debug}"

            # Get visualization paths
            viz_paths = _state.get("last_visualization_paths", [])
            valid_viz_paths = [p for p in viz_paths if Path(p).exists()]

            # Build final history: user message + all progress events + final response
            final_history = history[:-1]  # remove "⏳ Processing..."

            # Add all progress events
            for evt in get_tracker().get_events():
                final_history = final_history + [
                    {"role": "assistant", "content": f"**{evt.title}**\n\n{evt.content}"}
                ]

            # Add final assistant response
            final_history = final_history + [{"role": "assistant", "content": assistant_text}]

            # Add images
            if valid_viz_paths:
                for viz_path in valid_viz_paths:
                    filename = Path(viz_path).name
                    final_history = final_history + [
                        {
                            "role": "assistant",
                            "content": {
                                "path": viz_path,
                                "alt_text": f"Visualization: {filename}"
                            }
                        }
                    ]

            # Prepare outputs
            debug_visible = show_debug and bool(debug.strip() if debug else False)
            gallery_images = valid_viz_paths if valid_viz_paths else None
            gallery_visible = len(valid_viz_paths) > 0
            show_diagnosis = needs_review

            fix_btn_text = "💡 Click to apply recommended fix"
            if needs_review:
                pending = _state.get("pending_diagnosis", {})
                recommendations = pending.get("recommendations", [])
                if recommendations:
                    top_fix = recommendations[0]
                    fix_btn_text = f"💡 Apply Fix: {top_fix}"

            yield (
                final_history,
                gr.update(value=None),
                None,
                gr.update(visible=debug_visible, value=debug or ""),
                gr.update(visible=gallery_visible),
                gr.update(value=gallery_images) if gallery_images else gr.update(value=None),
                gr.update(visible=show_diagnosis),
                gr.update(value=fix_btn_text),
                gr.update(value="", visible=False),
            )
        
        def on_apply_fix(history):
            """Handle Apply Fix button - attempt improved analysis with the recommended fix"""
            import traceback
            
            pending = _state.get("pending_diagnosis")
            if not pending:
                history = history + [{"role": "assistant", "content": "⚠️ No pending diagnosis information."}]
                return history, gr.update(visible=False), gr.update(value="")
            
            # Get the top recommendation
            recommendations = pending.get("recommendations", [])
            top_fix = recommendations[0] if recommendations else "Apply recommended improvements"
            
            # Add user decision to history
            history = history + [{"role": "user", "content": f"💡 Applying fix: {top_fix}"}]
            
            # Get diagnosis info
            suggested_action = pending.get("suggested_action", "replan")
            recommendations = pending.get("recommendations", [])
            original_prompt = pending.get("user_idea", "")
            dataset_path = pending.get("dataset_path")
            
            # Build progress message
            retry_msg = f"🔄 Attempting to improve analysis based on diagnosis recommendations...\n\n"
            retry_msg += f"**Suggested action**: {suggested_action}\n"
            if recommendations:
                retry_msg += "**Improvements to apply**:\n"
                for rec in recommendations[:3]:
                    retry_msg += f"- {rec}\n"
            
            history = history + [{"role": "assistant", "content": retry_msg}]
            
            try:
                # Step 1: Use Controller Agent to refine the prompt
                from neuro_copilot.core.controller.controller_agent import ControllerAgent
                controller = ControllerAgent()
                
                # Get dataset info for context - prefer from pending_diagnosis
                dataset_info = pending.get("dataset_metadata") or {}
                if not dataset_info:
                    # Fallback to last_context
                    last_ctx = _state.get("last_context", {})
                    dataset_info = last_ctx.get("dataset_metadata", {})
                
                refined_prompt = controller.refine_prompt_with_diagnosis(
                    original_prompt=original_prompt,
                    recommendations=recommendations,
                    suggested_action=suggested_action,
                    dataset_info=dataset_info,
                )
                
                # Show refined prompt to user
                refine_msg = f"📝 **Refined Analysis Request**:\n\n> {refined_prompt}\n\n⏳ Re-running analysis..."
                history = history + [{"role": "assistant", "content": refine_msg}]
                
                # Build retry context with previous plan and diagnosis feedback
                retry_context = {
                    "previous_plan": pending.get("previous_plan"),
                    "diagnosis_feedback": "\n".join(recommendations),
                    "metrics": pending.get("metrics", {}),
                    "attempt_number": pending.get("attempt_number", 1) + 1,
                }
                
                # Step 2: Re-run pipeline with refined prompt AND retry context
                initialize_adapter()
                response = _state["adapter"].process_request(
                    user_idea=refined_prompt,
                    dataset_file=dataset_path,
                    task_type_override="DA",  # Retry is always DA since we're doing data analysis
                    preprocess_rules=None,
                    retry_context=retry_context,  # Pass retry context to pipeline
                )
                
                # Step 3: Process response
                main_output = response.main_output
                
                # Check if new execution also has issues
                if response.needs_user_review:
                    # Still has issues - show results with warning
                    result_msg = f"⚠️ **Refined analysis completed, but issues may remain**\n\n{main_output}"
                    history = history + [{"role": "assistant", "content": result_msg}]
                    
                    # Extract plan and metrics from new response for next retry
                    diagnosis_raw = response.diagnosis_raw or {}
                    new_plan = diagnosis_raw.get("plan")
                    new_exec_report = diagnosis_raw.get("exec_report", {})
                    new_metrics = {}
                    for step in new_exec_report.get("steps", []):
                        out_path = step.get("output_path")
                        if out_path:
                            try:
                                import json
                                from pathlib import Path
                                out_json = json.loads(Path(out_path).read_text(encoding="utf-8"))
                                step_metrics = out_json.get("metrics", {})
                                if step_metrics:
                                    new_metrics.update(step_metrics)
                            except Exception:
                                pass
                    
                    # Update pending diagnosis for potential second retry
                    current_attempt = pending.get("attempt_number", 1)
                    _state["pending_diagnosis"] = {
                        "summary": response.diagnosis_summary,
                        "recommendations": response.diagnosis_recommendations,
                        "suggested_action": response.suggested_action,
                        "raw": response.diagnosis_raw,
                        "dataset_path": dataset_path,
                        "user_idea": refined_prompt,  # Use refined prompt for next iteration
                        "dataset_metadata": response.dataset_metadata,  # Keep dataset metadata
                        "previous_plan": new_plan,  # Store new plan for next retry
                        "metrics": new_metrics,  # Store new metrics
                        "attempt_number": current_attempt + 1,
                    }
                    # Keep UI visible for another attempt, update fix button text
                    new_fix = response.diagnosis_recommendations[0] if response.diagnosis_recommendations else "Apply recommended improvements"
                    return history, gr.update(visible=True), gr.update(value=f"💡 Apply Fix: {new_fix}")
                else:
                    # Success!
                    result_msg = f"✅ **Refined analysis completed successfully!**\n\n{main_output}"
                    history = history + [{"role": "assistant", "content": result_msg}]
                    
                    # Clear pending diagnosis
                    _state["pending_diagnosis"] = None
                    
                    return history, gr.update(visible=False), gr.update(value="")
                
            except Exception as e:
                # Error during retry
                error_msg = f"❌ **Error during retry**:\n\n```\n{str(e)}\n```\n\n"
                error_msg += "Please try modifying the request manually or contact the developer."
                print(f"Error during retry: {e}")
                traceback.print_exc()
                history = history + [{"role": "assistant", "content": error_msg}]
                
                # Clear pending diagnosis
                _state["pending_diagnosis"] = None
                
                return history, gr.update(visible=False), gr.update(value="")
        
        def on_no_click(history):
            """Handle No button - accept current results and run full analysis"""
            pending = _state.get("pending_diagnosis")
            if not pending:
                history = history + [{"role": "assistant", "content": "⚠️ No pending diagnosis information."}]
                return history, gr.update(visible=False), gr.update(value="")
            
            # Add user decision to history
            history = history + [{"role": "user", "content": "❌ No - Accept current results"}]
            
            # Acknowledge and explain limitations
            accept_msg = (
                "📊 **Accepting Current Results**\n\n"
                "The system will output a complete analysis report, including:\n"
                "- Current metric results\n"
                "- **Known Limitations**\n"
                "- **Improvement Recommendations**\n"
                "- Available visualizations\n\n"
            )
            
            # Add limitations based on diagnosis
            recommendations = pending.get("recommendations", [])
            if recommendations:
                accept_msg += "### ⚠️ Known Limitations\n"
                for rec in recommendations[:5]:
                    accept_msg += f"- {rec}\n"
            
            accept_msg += "\n### 📌 Recommendations\n"
            accept_msg += "For more accurate results, consider in future analyses:\n"
            accept_msg += "1. Specify a time reference field for alignment (check your dataset's event columns)\n"
            accept_msg += "2. Specify the analysis time window (e.g., a task-relevant epoch)\n"
            accept_msg += "3. Filter for high-quality data points (check your dataset's quality columns)\n"
            
            history = history + [{"role": "assistant", "content": accept_msg}]
            
            # Clear pending diagnosis
            _state["pending_diagnosis"] = None
            
            return history, gr.update(visible=False), gr.update(value="")

        # Show uploaded file names below input + deduplicate files
        _dedup_just_fired = {"v": False}

        def update_file_indicator(multimodal_data):
            if _dedup_just_fired["v"]:
                # This is the re-trigger after we wrote back deduplicated files — skip
                _dedup_just_fired["v"] = False
                if not isinstance(multimodal_data, dict):
                    return gr.update(), gr.update(value="", visible=False)
                files = multimodal_data.get("files", [])
                if not files:
                    return gr.update(), gr.update(value="", visible=False)
                names = []
                for f in files:
                    if isinstance(f, dict):
                        name = f.get("orig_name") or f.get("name") or "file"
                    elif hasattr(f, "orig_name"):
                        name = f.orig_name or "file"
                    else:
                        name = str(f).split("/")[-1]
                    names.append(name)
                return gr.update(), gr.update(value="📎 " + ", ".join(names), visible=True)

            if not isinstance(multimodal_data, dict):
                return gr.update(), gr.update(value="", visible=False)
            files = multimodal_data.get("files", [])
            text = multimodal_data.get("text", "")
            if not files:
                return gr.update(), gr.update(value="", visible=False)

            # Detect pasted text files and convert to text input
            real_files = []
            pasted_texts = []
            for f in files:
                # Get file path and name
                if isinstance(f, dict):
                    fpath = f.get("path", "")
                    fname = f.get("orig_name") or f.get("name") or ""
                elif hasattr(f, "orig_name"):
                    fpath = str(getattr(f, "path", f))
                    fname = f.orig_name or ""
                else:
                    fpath = str(f)
                    fname = str(f).split("/")[-1]

                if fname == "pasted_text.txt" or (fname.endswith(".txt") and "pasted" in fname.lower()):
                    # Read pasted text content and merge into text field
                    try:
                        with open(fpath, "r", encoding="utf-8") as pf:
                            pasted_texts.append(pf.read().strip())
                    except Exception:
                        real_files.append(f)
                else:
                    real_files.append(f)

            if pasted_texts:
                # Merge pasted text into the text field
                merged_text = text
                for pt in pasted_texts:
                    if merged_text:
                        merged_text = merged_text + "\n" + pt
                    else:
                        merged_text = pt
                _dedup_just_fired["v"] = True
                if real_files:
                    names = []
                    for f in real_files:
                        if isinstance(f, dict):
                            names.append(f.get("orig_name") or f.get("name") or "file")
                        elif hasattr(f, "orig_name"):
                            names.append(f.orig_name or "file")
                        else:
                            names.append(str(f).split("/")[-1])
                    return {"text": merged_text, "files": real_files}, gr.update(value="📎 " + ", ".join(names), visible=True)
                else:
                    return {"text": merged_text, "files": []}, gr.update(value="", visible=False)

            files = real_files

            # Extract names and deduplicate
            seen = set()
            unique_names = []
            unique_files = []
            has_dupes = False
            for f in files:
                if isinstance(f, dict):
                    name = f.get("orig_name") or f.get("name") or "file"
                elif hasattr(f, "orig_name"):
                    name = f.orig_name or "file"
                elif hasattr(f, "name"):
                    name = f.name.split("/")[-1] if "/" in str(f.name) else str(f.name)
                else:
                    name = str(f).split("/")[-1]
                if name in seen:
                    has_dupes = True
                else:
                    seen.add(name)
                    unique_names.append(name)
                    unique_files.append(f)

            label = "📎 " + ", ".join(unique_names)
            indicator = gr.update(value=label, visible=True)

            if has_dupes:
                _dedup_just_fired["v"] = True
                gr.Warning("Duplicate file detected and removed. Please do not upload files with the same name.")
                return {"text": text, "files": unique_files}, indicator
            else:
                return gr.update(), indicator

        multimodal_input.change(
            fn=update_file_indicator,
            inputs=[multimodal_input],
            outputs=[multimodal_input, file_indicator],
        )

        # MultimodalTextbox has built-in submit (submit_btn="➤")
        multimodal_input.submit(
            fn=on_submit,
            inputs=[
                multimodal_input,
                dataset_input,
                task_type_input,
                developer_mode_checkbox,
                chat_history,
            ],
            outputs=[
                chat_history,
                multimodal_input,
                dataset_input,
                debug_output,
                viz_accordion,
                viz_gallery,
                diagnosis_container,
                recommended_fix_btn,
                file_indicator,
            ]
        )
        
        recommended_fix_btn.click(
            fn=on_apply_fix,
            inputs=[chat_history],
            outputs=[chat_history, diagnosis_container, recommended_fix_btn]
        )
        
        no_btn.click(
            fn=on_no_click,
            inputs=[chat_history],
            outputs=[chat_history, diagnosis_container, recommended_fix_btn]
        )
        
        developer_mode_checkbox.change(
            fn=toggle_developer_mode,
            inputs=[developer_mode_checkbox],
            outputs=[developer_mode_status]
        )

        # Auto-resize textarea via client-side JS
        app.load(
            fn=None,
            inputs=None,
            outputs=None,
            js="""
            () => {
                console.log('[auto-resize] JS loaded!');
                let lastVal = '';
                setInterval(() => {
                    const all = document.querySelectorAll('textarea');
                    let ta = null;
                    for (let i = 0; i < all.length; i++) {
                        if (all[i].placeholder && all[i].placeholder.indexOf('Describe') >= 0) {
                            ta = all[i]; break;
                        }
                    }
                    if (!ta) return;
                    if (ta.value !== lastVal) {
                        lastVal = ta.value;
                        ta.style.height = 'auto';
                        const maxH = window.innerHeight * 0.38;
                        const newH = Math.min(ta.scrollHeight + 2, maxH);
                        ta.style.height = newH + 'px';
                        ta.style.overflowY = ta.scrollHeight > maxH ? 'auto' : 'hidden';
                        console.log('[auto-resize] h=' + newH + ' scrollH=' + ta.scrollHeight);
                    }
                }, 300);

                // Move Chatbot action icons above the chat box
                function moveIcons() {
                    var chat = document.querySelector('#chat-history');
                    if (!chat) return;
                    var icons = chat.querySelector('.icon-buttons') || chat.querySelector('[class*="icon-button"]');
                    if (!icons) {
                        // Try finding by button with specific aria/title
                        var btns = chat.querySelectorAll('button');
                        for (var b of btns) {
                            if (b.parentElement && b.parentElement.children.length >= 2) {
                                var p = b.parentElement;
                                if (p.querySelector('button[title], button[aria-label]')) {
                                    icons = p; break;
                                }
                            }
                        }
                    }
                    if (icons && !icons.dataset.moved) {
                        icons.dataset.moved = '1';
                        icons.style.position = 'absolute';
                        icons.style.top = '-28px';
                        icons.style.right = '0';
                        icons.style.zIndex = '10';
                        chat.style.position = 'relative';
                    }
                }
                moveIcons();
                setTimeout(moveIcons, 1000);
                setTimeout(moveIcons, 3000);

                // Move Gradio footer to top-right
                function moveFooter() {
                    var f = document.querySelector('footer');
                    if (!f) return;
                    f.style.position = 'fixed';
                    f.style.top = '4px';
                    f.style.right = '12px';
                    f.style.bottom = 'auto';
                    f.style.left = 'auto';
                    f.style.width = 'auto';
                    f.style.zIndex = '9999';
                    f.style.background = 'transparent';
                    f.style.padding = '2px 8px';
                    f.style.fontSize = '11px';
                    f.style.opacity = '0.5';
                }
                moveFooter();
                setTimeout(moveFooter, 1000);
                setTimeout(moveFooter, 3000);
            }
            """
        )

    return app


def main():
    """Launch web application"""
    print("="*80, flush=True)
    print("🧠 NS-Copilot (New Architecture)", flush=True)
    print("="*80, flush=True)
    print("\nInitializing web interface...", flush=True)
    
    app = create_web_interface()
    
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    
    print("✅ Gradio interface created!", flush=True)
    print("\nLaunching web server...", flush=True)
    print(f"Access the interface at: http://localhost:{port}", flush=True)
    print("\nPress Ctrl+C to stop the server.", flush=True)
    
    app.launch(
        server_name=host,
        server_port=port,
        share=False,
        show_error=True,
        quiet=False,  # Show startup messages
        css="""
        #chat-container {
            max-width: 1200px;
            margin: 0 auto;
            padding-bottom: 100px;
        }
        #chat-history {
            height: calc(100vh - 260px) !important;
        }
        #chat-input-bar {
            position: fixed !important;
            bottom: 30px;
            left: 50%;
            transform: translateX(-50%);
            width: min(1200px, calc(100% - 40px));
            background: #ffffff;
            padding: 8px 8px 4px 8px;
            border-top: 1px solid #e6e6e6;
            border-radius: 12px;
            box-shadow: 0 -2px 10px rgba(0,0,0,0.05);
            z-index: 100;
        }
        #chat-input-bar .gr-form {
            gap: 8px;
        }
        """
    )


if __name__ == "__main__":
    main()





