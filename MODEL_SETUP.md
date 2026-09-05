# Foundation Models Setup Guide

NS-Copilot integrates 9 pretrained foundation models for neural data analysis (3 spike, 6 EEG). This guide covers setup, paths, and troubleshooting.

## Quick Start

```bash
# 1. Configure
cp .env.example .env   # Add OPENAI_API_KEY, HF_TOKEN

# 2. Download all models + datasets
./setup.sh

# 3. Build and start
docker-compose up -d --build

# 4. (For auto mode) Authenticate Codex CLI inside container
docker exec -u root neuro-copilot-web sh -c 'printenv OPENAI_API_KEY | codex login --with-api-key'
```

## Supported Models

### Spike Models

| Model | Params | Pretrained On | Output Dim | Use Case |
|-------|--------|---------------|------------|----------|
| NDT3 | 45M | 2000h multi-species spike data | 1024 | General spike decoding |
| MtM | ~10M | Mouse Neuropixels (10 sessions) | 512 | Mouse silicon probe data |
| POYO-1 | ~24M | Multi-species, variable neuron counts | 128 | Flexible neural populations |

### EEG Models

| Model | Params | Pretrained On | Output Dim | Channel Requirement |
|-------|--------|---------------|------------|---------------------|
| LaBraM | ~100M | 2500h EEG | 200 | Standard 10-20 channels (uppercase names) |
| REVE | ~100M | 60,000h EEG (25,000 subjects) | 1536 | Any electrode montage |
| CBraMod | ~50M | 9000h EEG | 600 | Channel-agnostic |
| EEGPT | ~50M | Multi-dataset EEG | 512 | 10-10/10-20 system (up to 62 channels) |
| EEGMamba | ~30M | Multi-dataset EEG | 512 | Channel-agnostic |
| BrainOmni | ~10M | Multi-modal neural data | 256 | Channel-agnostic |

## Model Paths

All models are stored in `$MODELS_DIR` (default: `./models/`). Each model has:
- **Code directory**: cloned from GitHub
- **Checkpoint file**: downloaded from HuggingFace or Figshare

```
models/
├── ndt3/                  # NDT3
│   ├── context_general_bci/
│   └── data/pretrained/753jmg4u/checkpoints/val-epoch=397-val_loss=0.4987.ckpt
├── mtm/                   # MtM
│   ├── src/
│   └── checkpoints/multi-NDT1-MtM-10-sessions/model_best.pt
├── poyo/                  # POYO-1
│   ├── poyo_note/
│   └── checkpoints/poyo_1.ckpt
├── labram/                # LaBraM
│   ├── labram/
│   └── checkpoints/labram-base.pth  (requires git lfs pull)
├── cbramod/               # CBraMod
│   ├── models/
│   └── pretrained_weights/pretrained_weights.pth
├── eegpt/                 # EEGPT
│   ├── models/
│   └── checkpoint/eegpt_mcae_58chs_4s_large4E.ckpt  (manual download from Figshare)
├── eegmamba/              # EEGMamba
│   ├── models/
│   └── pretrained_weights/pretrained_EEGMamba.pth
└── brainomni/             # BrainOmni
    ├── models/
    └── pretrained_weights/tiny/BrainOmni.pt
```

**REVE** is loaded from HuggingFace at runtime (no local files needed, requires `HF_TOKEN`).

## Manual Setup

If `./setup.sh` fails for a specific model, use these manual instructions:

### NDT3
```bash
cd models
git clone https://github.com/joel99/ndt3.git
mkdir -p ndt3/data/pretrained/753jmg4u/checkpoints
wget -O ndt3/data/pretrained/753jmg4u/checkpoints/val-epoch=397-val_loss=0.4987.ckpt \
  "https://huggingface.co/joel99/ndt3/resolve/main/val-epoch%3D397-val_loss%3D0.4987.ckpt"
```

### MtM
```bash
cd models
git clone https://github.com/colehurwitz/IBL_MtM_model.git mtm
mkdir -p mtm/checkpoints/multi-NDT1-MtM-10-sessions
wget -O mtm/checkpoints/multi-NDT1-MtM-10-sessions/model_best.pt \
  "https://huggingface.co/ibl-foundation-model/multi-NDT1-MtM-10-sessions/resolve/main/model_best.pt"
```

### POYO-1
```bash
cd models
mkdir -p poyo/checkpoints
wget -O poyo/checkpoints/poyo_1.ckpt \
  "https://nyu1.osn.mghpcc.org/brainsets-public/model-zoo/poyo_1.ckpt"
# Note: POYO-1 code is bundled in core/models/poyo_wrapper.py, no separate repo needed
```

### LaBraM
```bash
cd models
git clone https://github.com/935963004/LaBraM.git labram
cd labram && git lfs pull   # Required — checkpoint stored via Git LFS
```

### CBraMod
```bash
cd models
git clone https://github.com/wjq-learning/CBraMod.git cbramod
mkdir -p cbramod/pretrained_weights
wget -O cbramod/pretrained_weights/pretrained_weights.pth \
  "https://huggingface.co/weighting666/CBraMod/resolve/main/pretrained_weights.pth"
```

### EEGPT
```bash
cd models
git clone https://github.com/BINE022/EEGPT.git eegpt
mkdir -p eegpt/checkpoint
# Manual download required from Figshare:
# Visit https://figshare.com/ndownloader/articles/25866970/versions/2
# Download eegpt_mcae_58chs_4s_large4E.ckpt and place in eegpt/checkpoint/
```

### EEGMamba
```bash
cd models
git clone https://github.com/wjq-learning/EEGMamba.git eegmamba
mkdir -p eegmamba/pretrained_weights
wget -O eegmamba/pretrained_weights/pretrained_EEGMamba.pth \
  "https://huggingface.co/weighting666/EEGMamba/resolve/main/pretrained_EEGMamba.pth"
```

### BrainOmni
```bash
cd models
git clone https://github.com/OpenTSLab/BrainOmni.git brainomni
mkdir -p brainomni/pretrained_weights/tiny
wget -O brainomni/pretrained_weights/tiny/BrainOmni.pt \
  "https://huggingface.co/Ego4D/BrainOmni/resolve/main/tiny/BrainOmni.pt"
```

### REVE (no local download needed)
REVE loads from HuggingFace at runtime. Ensure `HF_TOKEN` is set in `.env`:
1. Create account at https://huggingface.co
2. Request access at https://huggingface.co/brain-bzh/reve-base
3. Create a Read token at https://huggingface.co/settings/tokens

## Environment Variable Overrides

Each model's path can be overridden via environment variables (see `.env.example`):

```bash
# Example: use models stored in a different location
NDT3_PATH=/data/shared/models/ndt3
NDT3_CHECKPOINT_PATH=/data/shared/models/ndt3/data/pretrained/753jmg4u/checkpoints/val-epoch=397-val_loss=0.4987.ckpt
```

The wrapper search order is: environment variable → `/app/<model>` (Docker mount) → `./models/<model>` → `../models/<model>`

## Codex CLI Baseline (Auto Mode)

When `CONTROLLER_MODE=auto`, NS-Copilot runs a full Codex CLI agent as baseline before foundation model selection.

**Prerequisites:**
1. Node.js 20+ and npm
2. Codex CLI: `npm install -g @openai/codex`
3. Authentication: `codex login` (or `printenv OPENAI_API_KEY | codex login --with-api-key`)

**Environment variables:**
- `CODEX_BASELINE_MODEL` (default: `gpt-5.2-codex`)
- `AUTO_MAX_RETRIES` (default: `2`)

## Troubleshooting

### "Model X is not available" error
1. Verify model directory exists: `ls models/<model>/`
2. Check `.env` has correct `MODELS_DIR`
3. Rebuild container: `docker-compose down && docker-compose build && docker-compose up -d`

### Checkpoint not found
1. Re-run `./setup.sh`
2. For LaBraM: run `cd models/labram && git lfs pull`
3. For EEGPT: manually download from Figshare (see above)

### Out of memory
- Foundation models run on GPU by default; ensure sufficient VRAM (8GB+ recommended)
- Models automatically fall back to CPU if no GPU is available
- Reduce number of EEG segments via `max_seconds` parameter

### Codex CLI 401 Unauthorized
- Run `codex login` inside the container (or use `--with-api-key`)
- Ensure `OPENAI_API_KEY` has access to the Responses API

## Adding New Models

1. Create wrapper in `core/models/<name>_wrapper.py` (extend `BaseEEGDecoder` or write custom)
2. Create tool spec in `core/models/<name>_tool_spec.py`
3. Register the tool in `core/tools/generic_tools.py`
4. Add download step in `setup.sh`
5. Add volume mount in `docker-compose.yml`
6. The Planner will automatically discover the new model via its tool spec

## Citations

When using foundation models, please cite the original papers:

- **NDT3**: Ye et al. (2025). A Generalist Intracortical Motor Decoder. bioRxiv.
- **MtM**: Zhang et al. (2024). Towards a Universal Translator for Neural Dynamics. NeurIPS 2024.
- **POYO-1**: Azabou et al. (2023). A Unified, Scalable Framework for Neural Population Decoding. NeurIPS 2023.
- **LaBraM**: Jiang et al. (2024). Large Brain Model for Learning Generic Representations. ICLR 2024.
- **REVE**: El Ouahidi et al. (2025). REVE: A Foundation Model for EEG. arXiv.
- **CBraMod**: Wang et al. (2025). CBraMod: A Criss-Cross Brain Foundation Model. ICLR 2025.
- **EEGPT**: Yue et al. (2024). EEGPT: Pretrained Transformer for Universal EEG. NeurIPS 2024.
- **EEGMamba**: Wang et al. (2024). EEGMamba: Bidirectional Mamba with Cross-Domain Transfer. arXiv.
- **BrainOmni**: Chen et al. (2025). BrainOmni: A Multimodal Brain Foundation Model. arXiv.
