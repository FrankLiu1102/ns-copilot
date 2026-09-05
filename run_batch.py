#!/usr/bin/env python3
"""
Batch runner: run a prompt N times through the pipeline and collect results.

Usage:
    python run_batch.py --prompt "Your prompt here" --dataset /path/to/dataset --n 10

    # Or use a prompt file:
    python run_batch.py --prompt-file prompt/PD/PD_ds004584_auto_prompt.md --dataset dataset/Parkinson/ds004584 --n 10

Runs inside the Docker container. To execute:
    docker exec neuro-copilot-web python run_batch.py --prompt-file prompt/PD/PD_ds004584_auto_prompt.md --dataset /app/dataset/Parkinson/ds004584 --n 10
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Add project root to path (works both in Docker /app and local environments)
_project_root = str(Path(__file__).resolve().parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Pre-defined seed list for reproducible variability across runs
# Each run uses a different seed, but the same run index always uses the same seed
SEED_LIST = [42, 123, 456, 789, 1024, 2048, 3090, 4096, 5555, 6789,
             7777, 8888, 9999, 1111, 2222, 3333, 4444, 5678, 6543, 7890]


def run_once(prompt: str, dataset_path: str, run_index: int, seed: int):
    """Run the pipeline once and return success status and elapsed time."""
    from neuro_copilot.core.pipeline import NeuroCopilotPipeline

    print(f"\n{'=' * 60}")
    print(f"  RUN {run_index + 1} (seed={seed})")
    print(f"{'=' * 60}\n")

    pipeline = NeuroCopilotPipeline()
    dataset_info = {"dataset_path": dataset_path, "_run_seed": seed}

    start = time.time()
    try:
        output = pipeline._run_da_execution(prompt, dataset_info)
        elapsed = time.time() - start
        print(f"\n✅ Run {run_index + 1} completed in {elapsed:.0f}s")
        return True, elapsed
    except Exception as e:
        elapsed = time.time() - start
        print(f"\n❌ Run {run_index + 1} failed in {elapsed:.0f}s: {e}")
        return False, elapsed


def collect_results(logs_dir: str, start_time: str):
    """Collect results from all runs that started after start_time."""
    results = []
    logs_path = Path(logs_dir)

    for run_dir in sorted(logs_path.iterdir()):
        if not run_dir.is_dir() or run_dir.name < start_time:
            continue
        # Check for auto mode (has model subdirectories)
        audit_path = run_dir / 'audit.json'
        model_dirs = [d for d in run_dir.iterdir() if d.is_dir() and (d / 'audit.json').exists()]

        if model_dirs:
            # Auto mode run
            run_result = {"run_dir": run_dir.name, "auto_mode": True, "models": {}}
            for model_dir in sorted(model_dirs):
                with open(model_dir / 'audit.json') as f:
                    data = json.load(f)
                run_result["models"][model_dir.name] = data
            results.append(run_result)
        elif audit_path.exists():
            with open(audit_path) as f:
                data = json.load(f)
            if data.get('auto_mode'):
                # Parent auto mode audit (skip, models are in subdirs)
                continue
            results.append({"run_dir": run_dir.name, "auto_mode": False, "data": data})

    return results


def summarize(results: list):
    """Print summary table across all runs."""
    print(f"\n{'=' * 80}")
    print(f"  SUMMARY ({len(results)} runs)")
    print(f"{'=' * 80}\n")

    if not results:
        print("No results found.")
        return

    # Check if auto mode
    if results[0].get("auto_mode"):
        # Collect per-model, per-stage metrics across runs
        from collections import defaultdict
        model_metrics = defaultdict(lambda: defaultdict(list))

        for run in results:
            for model_name, data in run.get("models", {}).items():
                for i, att in enumerate(data.get("attempts", [])):
                    er = att.get("execution_result") or {}
                    mk = er.get("metrics") or {}
                    stage = ["initial", "retry1", "retry2"][i] if i < 3 else f"attempt{i}"
                    for key in ["cv_accuracy_mean", "cv_balanced_accuracy_mean",
                                "cv_f1_macro_mean", "cv_auroc_macro_mean"]:
                        v = mk.get(key)
                        if v is not None:
                            model_metrics[model_name][(stage, key)].append(v)

        # Print per-model summary
        import numpy as np
        models = sorted(model_metrics.keys())
        stages = ["initial", "retry1", "retry2"]
        metric_keys = ["cv_accuracy_mean", "cv_balanced_accuracy_mean",
                       "cv_f1_macro_mean", "cv_auroc_macro_mean"]
        short_names = {"cv_accuracy_mean": "Acc", "cv_balanced_accuracy_mean": "BalAcc",
                       "cv_f1_macro_mean": "F1", "cv_auroc_macro_mean": "AUROC"}

        for stage in stages:
            print(f"\n--- {stage.upper()} ---")
            print("%-12s | %12s | %12s | %12s | %12s" % (
                "Model", "Acc", "BalAcc", "F1", "AUROC"))
            print("-" * 70)
            for model in models:
                vals = []
                for mk in metric_keys:
                    v_list = model_metrics[model].get((stage, mk), [])
                    if v_list:
                        mean = np.mean(v_list) * 100
                        std = np.std(v_list) * 100
                        vals.append(f"{mean:.1f}±{std:.1f}")
                    else:
                        vals.append("—")
                print("%-12s | %12s | %12s | %12s | %12s" % (model, *vals))

        # Best per stage
        print(f"\n--- BEST PER STAGE (by mean BalAcc) ---")
        for stage in stages:
            best_model = None
            best_mean = -1
            for model in models:
                v_list = model_metrics[model].get((stage, "cv_balanced_accuracy_mean"), [])
                if v_list and np.mean(v_list) > best_mean:
                    best_mean = np.mean(v_list)
                    best_model = model
            if best_model:
                print(f"  {stage}: {best_model} (BalAcc={best_mean*100:.1f}%)")

    # Save raw results
    summary_path = Path(results[0]["run_dir"]).parent if not results[0].get("auto_mode") \
        else Path("outputs/da_run_logs")
    summary_file = f"batch_summary_{time.strftime('%Y%m%d_%H%M%S')}.json"
    # Just print, don't save to avoid path issues
    print(f"\nDone. {len(results)} runs collected.")


def main():
    parser = argparse.ArgumentParser(description="Batch run pipeline")
    parser.add_argument("--prompt", type=str, help="Prompt text")
    parser.add_argument("--prompt-file", type=str, help="Path to prompt file")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset path")
    parser.add_argument("--n", type=int, default=10, help="Number of successful runs required")
    parser.add_argument("--seeds", type=str, default=None, help="Comma-separated list of specific seeds to run (e.g., '42,123,1024')")
    args = parser.parse_args()

    if args.prompt_file:
        with open(args.prompt_file) as f:
            prompt = f.read().strip()
    elif args.prompt:
        prompt = args.prompt
    else:
        parser.error("Either --prompt or --prompt-file is required")

    # Determine seed list
    if args.seeds:
        run_seeds = [int(s.strip()) for s in args.seeds.split(",")]
        args.n = len(run_seeds)
    else:
        run_seeds = None  # Use default SEED_LIST
        if args.n > len(SEED_LIST):
            parser.error(f"--n cannot exceed {len(SEED_LIST)} (available seeds). Use --seeds for custom seeds.")

    print(f"Prompt: {prompt[:100]}...")
    print(f"Dataset: {args.dataset}")
    print(f"Runs: {args.n}")
    if run_seeds:
        print(f"Seeds: {run_seeds}")
    print()

    logs_dir = "/app/outputs/da_run_logs"
    start_time = time.strftime("%Y%m%d_%H%M%S")

    total_start = time.time()
    successes = 0
    attempts = 0
    codex_failures = 0

    while successes < args.n:
        if run_seeds:
            seed = run_seeds[successes]
        else:
            seed_idx = successes % len(SEED_LIST)
            seed = SEED_LIST[seed_idx]

        ok, elapsed = run_once(prompt, args.dataset, successes, seed)

        if ok:
            successes += 1
        else:
            codex_failures += 1
            print(f"⚠️ Run failed (attempt {attempts + 1}), not counting. "
                  f"Successes so far: {successes}/{args.n}")

        attempts += 1

        # Safety: stop if too many consecutive failures
        if codex_failures > args.n * 2:
            print(f"\n⚠️ Too many failures ({codex_failures}), stopping.")
            break

    total_elapsed = time.time() - total_start
    print(f"\n{'=' * 60}")
    print(f"  BATCH COMPLETE: {successes}/{args.n} succeeded in {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")
    if codex_failures > 0:
        print(f"  ({codex_failures} failed runs skipped)")
    print(f"{'=' * 60}")

    # Collect and summarize
    results = collect_results(logs_dir, start_time)
    summarize(results)


if __name__ == "__main__":
    main()
