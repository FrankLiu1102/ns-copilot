# NS-Copilot

**An LLM-driven agent system for autonomous neuroscience analysis**

<p align="center">
  Findings of the Association for Computational Linguistics: EMNLP 2026
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10%2B-blue" alt="python">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="license">
  <a href="https://arxiv.org/abs/2609.01971"><img src="https://img.shields.io/badge/arXiv-2609.01971-607d8b" alt="arXiv"></a>
</p>

NS-Copilot is a multi-agent system that autonomously performs end-to-end neuroscience data analysis. Given raw neural data and a natural-language task description, it orchestrates four LLM agents (Planner, Coder, Controller, Interpreter) to select pre-trained models, generate analysis code, iteratively optimize results, and produce comprehensive reports, all within a sandboxed Docker environment.

## Features

- **Auto Model Selection**: Automatically establishes a performance threshold, searches through a prioritized queue of pre-trained models, and triggers early stopping when the threshold is exceeded
- **9 Pre-trained Models**: Integrates neural decoders as callable tools via declarative tool specifications
  - **Spike models**: NDT3, MtM, POYO-1 (working memory decoding)
  - **EEG models**: LaBraM, REVE, CBraMod, EEGPT, EEGMamba, BrainOmni (AD/PD classification)
- **Closed-loop Optimization**: Controller agent diagnoses suboptimal results and routes to Replan (new strategy) or Modify (targeted code patching)
- **Multi-Format Data**: NWB, EEGLAB (.set), MATLAB (.mat), CSV, compressed archives (auto-detected)
- **Docker Sandboxed Execution**: All generated code runs in isolated containers with GPU support
- **Web Interface**: Gradio-based UI with real-time progress streaming

## Quick Start

```bash
# 1. Clone and configure
git clone https://github.com/FrankLiu1102/ns-copilot.git && cd ns-copilot
cp .env.example .env   # Add your OPENAI_API_KEY (and HF_TOKEN for REVE)

# 2. Download models + datasets
pip install openneuro-py   # Host dependency for dataset download
./setup.sh

# 3. Build and start Docker container (first build ~15-20min)
docker-compose up -d --build

# 4. Authenticate Codex CLI (required for auto mode)
docker exec -u root neuro-copilot-web sh -c 'printenv OPENAI_API_KEY | codex login --with-api-key'

# 5. Open web UI at http://localhost:8001
# Or run reproducibility tests (see below)
```

## Reproducibility

Run experiments with random seeds for reproducibility:

```bash
# Auto mode (default): system selects the best pre-trained model
./run_test.sh --domain wm --dataset dandi_000006 --count 10
./run_test.sh --domain ad --dataset ds004504 --count 10
./run_test.sh --domain pd --dataset ds004584 --count 10

# Single model mode: run a specific model
./run_test.sh --domain wm --dataset dandi_000006 --model POYO --count 3

# Specific seeds
./run_test.sh --domain ad --dataset ds004504 --seeds 42,123,456
```

**Arguments:**

| Argument | Values | Description |
|----------|--------|-------------|
| `--domain` | `wm`, `ad`, `pd` | Task domain (required) |
| `--dataset` | `dandi_000006`, `ds004504`, `ds004584` | Dataset name (required) |
| `--count` | integer | Number of runs with random seeds (required unless `--seeds`) |
| `--model` | `vanilla`, `NDT3`, `MtM`, `POYO`, `REVE`, `EEGPT`, `EEGMamba`, `BrainOmni`, `LaBraM`, `CBraMod` | Specific model (omit for auto mode) |
| `--seeds` | comma-separated | Specific seeds, e.g., `42,123,456` (overrides `--count`) |

**Benchmarks:**

| Domain | Dataset | Task | Pre-trained Models |
|--------|---------|------|--------------------|
| `wm` | `dandi_000006` | Spike decoding (left vs right lick) | NDT3, MtM, POYO-1 |
| `ad` | `ds004504` | EEG classification (AD/FTD/Healthy) | LaBraM, REVE, CBraMod, EEGPT, EEGMamba, BrainOmni |
| `pd` | `ds004584` | EEG classification (PD/Control) | LaBraM, REVE, CBraMod, EEGPT, EEGMamba, BrainOmni |

**Results** are saved to `outputs/da_run_logs/<timestamp>/`, each containing `audit.json` with per-attempt metrics and generated code.

## Auto Mode Pipeline

When no specific model is specified, NS-Copilot runs the full Auto Model Selection protocol:

1. **Parallel initialization**: The Planner analyzes dataset metadata to determine the primary metric and construct a model priority queue. Concurrently, a performance threshold is established: by default an independent Codex agent produces a reference score, or you can set `AUTO_TARGET_VALUE` to supply one directly (see [Environment Variables](#environment-variables)).
2. **Model search**: Each model in the queue is evaluated sequentially. If any model exceeds the performance threshold, early stopping is triggered.
3. **Controller retry** (up to 2 rounds): If no model exceeds the threshold, the Controller diagnoses each model's results and routes to either Replan (full strategy resynthesis via Planner) or Modify (targeted code patching via Coder). Early stopping applies within retry rounds as well.
4. **Interpreter**: Generates a comprehensive report with metric interpretations, key findings, limitations, and recommendations.

## Project Structure

```
ns-copilot/
├── run_test.sh              # Reproducibility runner (entry point)
├── run_batch.py             # Batch execution engine (called by run_test.sh)
├── setup.sh                 # Download models + datasets
├── web_app.py               # Gradio web interface
├── Dockerfile / docker-compose.yml
│
├── core/                    # Core pipeline
│   ├── pipeline.py          # Main orchestration (auto mode logic)
│   ├── DA/                  # Planner + Coder agents
│   ├── controller/          # Controller agent (closed-loop optimization)
│   ├── models/              # Pre-trained model wrappers + tool specifications
│   └── tools/               # Callable analysis tools
│
├── prompt/                  # Experiment prompts (WM / AD / PD)
├── dataset/                 # Datasets (auto-downloaded via setup.sh)
├── models/                  # Pre-trained model repos + checkpoints
└── outputs/                 # Run logs and results
```

The Python package and the Docker container are named `neuro_copilot` and
`neuro-copilot-web`, retained from the project's original name for backward
compatibility. The system is referred to as NS-Copilot throughout.

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes | OpenAI API key (also used by Codex CLI) |
| `HF_TOKEN` | For REVE | HuggingFace token ([request access](https://huggingface.co/brain-bzh/reve-base)) |
| `MODELS_DIR` | No | Custom model storage path (default: `./models`) |
| `AUTO_TARGET_VALUE` | No | Early-stopping threshold for auto mode; replaces the Codex-derived baseline (see below) |

### Setting the early-stopping threshold

Auto mode stops the model search as soon as a model exceeds a performance
threshold. That threshold is derived from an independent Codex baseline run by
default. Set `AUTO_TARGET_VALUE` to supply your own instead:

```bash
AUTO_TARGET_VALUE=0.85 ./run_test.sh --domain ad --dataset ds004504 --count 3
```

For the web UI, put it in `.env` and recreate the container. Note that
`./run.sh restart` will **not** pick it up, because `docker-compose restart`
reuses the existing container's environment:

```bash
echo "AUTO_TARGET_VALUE=0.85" >> .env
docker-compose up -d
```

The threshold applies to the primary metric the Planner selects for the run,
which is the same metric the Codex baseline is read on. A model must exceed it
to stop the search early.


## Citation

If you use NS-Copilot in your research, please cite:

```bibtex
@misc{liu2026nscopilot,
  title  = {NS-Copilot: An LLM-Driven Agent System for Autonomous Neuroscience Analysis},
  author = {Wuche Liu and Yiran Qiao and Linlin Hou and Rui Yang and
            Shusen Pu and Song Wang and Jing Ma},
  year   = {2026},
  eprint = {2609.01971},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url    = {https://arxiv.org/abs/2609.01971},
  note   = {Findings of the Association for Computational Linguistics: EMNLP 2026}
}
```

## License

MIT

### Pre-trained Model Licenses

All integrated pre-trained models are publicly released for research use:

| Model | License | Source |
|-------|---------|--------|
| NDT3 | MIT (code), CC-BY-NC-4.0 (weights) | [joel99/ndt3](https://github.com/joel99/ndt3) |
| MtM | MIT | [colehurwitz/IBL_MtM_model](https://github.com/colehurwitz/IBL_MtM_model) |
| POYO-1 | Apache 2.0 | [neuro-galaxy/torch_brain](https://github.com/neuro-galaxy/torch_brain) |
| LaBraM | MIT | [935963004/LaBraM](https://github.com/935963004/LaBraM) |
| REVE | MIT (code), Custom (weights) | [elouayas/reve_eeg](https://github.com/elouayas/reve_eeg) / [brain-bzh/reve-base](https://huggingface.co/brain-bzh/reve-base) |
| CBraMod | MIT | [wjq-learning/CBraMod](https://github.com/wjq-learning/CBraMod) |
| EEGPT | Apache 2.0 | [BINE022/EEGPT](https://github.com/BINE022/EEGPT) |
| EEGMamba | MIT | [wjq-learning/EEGMamba](https://github.com/wjq-learning/EEGMamba) |
| BrainOmni | MIT | [OpenTSLab/BrainOmni](https://github.com/OpenTSLab/BrainOmni) |

When using these models in publications, please cite the original papers (see [MODEL_SETUP.md](MODEL_SETUP.md#citations)).

### Dataset Licenses

| Dataset | License | Included in this repository | Source |
|---------|---------|------------------------------|--------|
| DANDI 000006 (WM) | CC-BY-4.0 | Yes, as derived spike/trial CSVs | Economo, M. N., & Svoboda, K. (2022). *Mouse anterior lateral motor cortex (ALM) in delay response task.* DANDI Archive. [doi:10.48324/dandi.000006/0.220126.1855](https://doi.org/10.48324/dandi.000006/0.220126.1855) |
| OpenNeuro ds004504 (AD) | CC0 | No, downloaded by `setup.sh` | Miltiadous, A., et al. (2024). OpenNeuro. [doi:10.18112/openneuro.ds004504.v1.0.8](https://doi.org/10.18112/openneuro.ds004504.v1.0.8) |
| OpenNeuro ds004584 (PD) | CC0 | No, downloaded by `setup.sh` | Singh, A., et al. (2023). OpenNeuro. [doi:10.18112/openneuro.ds004584.v1.0.0](https://doi.org/10.18112/openneuro.ds004584.v1.0.0) |

The spike and trial CSVs under `dataset/Working_Memory/dandi_000006/` are derived
from DANDI 000006 and are redistributed here under CC-BY-4.0.
