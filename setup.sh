#!/bin/bash
# NS-Copilot Setup Script
# Downloads foundation models, checkpoints, and datasets

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Load environment variables from .env if exists
if [ -f .env ]; then
    set -a
    . .env
    set +a
fi

# Default models directory (can be overridden in .env)
MODELS_DIR="${MODELS_DIR:-./models}"

echo -e "${GREEN}=== NS-Copilot Setup ===${NC}"
echo ""

# =============================================================================
# Prerequisites Check
# =============================================================================
MISSING_PREREQS=0

# Check Docker
if command -v docker &> /dev/null; then
    echo "✓ Docker $(docker --version 2>/dev/null | grep -oP '\d+\.\d+\.\d+')"
else
    echo -e "${RED}✗ Docker not found${NC}"
    echo "  Install Docker Engine:"
    echo "    sudo apt-get update"
    echo "    sudo apt-get install -y docker.io docker-compose-v2"
    echo "    sudo systemctl enable --now docker"
    echo "    sudo usermod -aG docker \$USER"
    echo "  Then log out and back in, and re-run this script."
    MISSING_PREREQS=1
fi

# Check Docker Compose (v2 plugin or standalone)
if docker compose version &> /dev/null; then
    echo "✓ Docker Compose $(docker compose version --short 2>/dev/null)"
elif command -v docker-compose &> /dev/null; then
    echo "✓ docker-compose $(docker-compose --version 2>/dev/null | grep -oP '\d+\.\d+\.\d+')"
elif [ "$MISSING_PREREQS" -eq 0 ]; then
    echo -e "${RED}✗ Docker Compose not found${NC}"
    echo "  Install: sudo apt-get install -y docker-compose-v2"
    MISSING_PREREQS=1
fi

# Check NVIDIA Container Toolkit (optional, for GPU)
if command -v nvidia-smi &> /dev/null; then
    if dpkg -l nvidia-container-toolkit &> /dev/null 2>&1 || command -v nvidia-container-runtime &> /dev/null; then
        echo "✓ NVIDIA Container Toolkit (GPU support)"
    else
        echo -e "${YELLOW}⚠ GPU detected but nvidia-container-toolkit not found${NC}"
        echo "  For GPU support in Docker:"
        echo "    sudo apt-get install -y nvidia-container-toolkit"
        echo "    sudo systemctl restart docker"
    fi
else
    echo -e "${YELLOW}⚠ No NVIDIA GPU detected (CPU-only mode)${NC}"
fi

# Check Node.js (required for Codex CLI baseline in auto mode)
if command -v node &> /dev/null; then
    echo "✓ Node.js $(node --version)"
else
    echo -e "${YELLOW}⚠ Node.js not found (required for Codex CLI baseline in auto mode)${NC}"
    echo "  Install: curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -"
    echo "           sudo apt-get install -y nodejs"
fi

# Check Codex CLI
if command -v codex &> /dev/null; then
    echo "✓ Codex CLI $(codex --version 2>/dev/null | head -1)"
else
    echo -e "${YELLOW}⚠ Codex CLI not found (required for auto mode baseline)${NC}"
    echo "  Install: npm install -g @openai/codex"
    echo "  Then authenticate: codex login"
fi

if [ "$MISSING_PREREQS" -eq 1 ]; then
    echo ""
    echo -e "${RED}Please install missing prerequisites and re-run this script.${NC}"
    exit 1
fi

echo ""
echo "Models directory: $MODELS_DIR"
echo ""

# Create models directory if it doesn't exist
mkdir -p "$MODELS_DIR"

# =============================================================================
# NDT3 (Neural Data Transformer 3)
# =============================================================================
echo -e "${YELLOW}[1/9] Setting up NDT3...${NC}"

NDT3_DIR="$MODELS_DIR/ndt3"

if [ -d "$NDT3_DIR" ] && [ -d "$NDT3_DIR/context_general_bci" ]; then
    echo "✓ NDT3 code already exists, skipping clone..."
else
    echo "Cloning NDT3 repository from GitHub..."
    git clone https://github.com/joel99/ndt3.git "$NDT3_DIR"
    echo -e "${GREEN}✓ NDT3 code downloaded${NC}"
fi

# Download NDT3 checkpoint
CHECKPOINT_DIR="$NDT3_DIR/data/pretrained/753jmg4u/checkpoints"
CHECKPOINT_FILE="$CHECKPOINT_DIR/val-epoch=397-val_loss=0.4987.ckpt"

if [ -f "$CHECKPOINT_FILE" ]; then
    echo "✓ NDT3 checkpoint already exists, skipping download..."
else
    echo "Downloading NDT3 checkpoint from Hugging Face..."
    mkdir -p "$CHECKPOINT_DIR"

    # Download using wget or curl
    if command -v wget &> /dev/null; then
        wget -O "$CHECKPOINT_FILE" \
            "https://huggingface.co/joel99/ndt3/resolve/main/753jmg4u/checkpoints/val-epoch%3D397-val_loss%3D0.4987.ckpt" \
            || echo -e "${RED}✗ Checkpoint download failed. Please download manually from:${NC}\n  https://huggingface.co/joel99/ndt3"
    elif command -v curl &> /dev/null; then
        curl -L -o "$CHECKPOINT_FILE" \
            "https://huggingface.co/joel99/ndt3/resolve/main/753jmg4u/checkpoints/val-epoch%3D397-val_loss%3D0.4987.ckpt" \
            || echo -e "${RED}✗ Checkpoint download failed. Please download manually from:${NC}\n  https://huggingface.co/joel99/ndt3"
    else
        echo -e "${RED}✗ Neither wget nor curl found. Please install one or download checkpoint manually:${NC}"
        echo "  https://huggingface.co/joel99/ndt3"
        exit 1
    fi

    if [ -f "$CHECKPOINT_FILE" ] && [ -s "$CHECKPOINT_FILE" ]; then
        echo -e "${GREEN}✓ NDT3 checkpoint downloaded ($(du -h "$CHECKPOINT_FILE" | cut -f1))${NC}"
    else
        rm -f "$CHECKPOINT_FILE"
        echo -e "${RED}✗ NDT3 checkpoint download failed or file is empty${NC}"
    fi
fi

echo ""

# =============================================================================
# MtM (Multi-task Masking) — NDT1 with multi-task masking strategy
# =============================================================================
echo -e "${YELLOW}[2/9] Setting up MtM...${NC}"

MTM_DIR="$MODELS_DIR/mtm"

if [ -d "$MTM_DIR" ] && [ -d "$MTM_DIR/src" ]; then
    echo "✓ MtM code already exists, skipping clone..."
else
    echo "Cloning MtM repository from GitHub..."
    git clone https://github.com/colehurwitz/IBL_MtM_model.git "$MTM_DIR"
    echo -e "${GREEN}✓ MtM code downloaded${NC}"
fi

# Download MtM checkpoint (10-session model)
MTM_CKPT_DIR="$MTM_DIR/checkpoints/multi-NDT1-MtM-10-sessions"
MTM_CKPT_FILE="$MTM_CKPT_DIR/model_best.pt"
MTM_CFG_FILE="$MTM_CKPT_DIR/config.yaml"
MTM_HF_BASE="https://huggingface.co/ibl-foundation-model/multi-NDT1-MtM-10-sessions/resolve/main"

if [ -f "$MTM_CKPT_FILE" ]; then
    echo "✓ MtM checkpoint already exists, skipping download..."
else
    echo "Downloading MtM checkpoint from Hugging Face..."
    mkdir -p "$MTM_CKPT_DIR"

    if command -v wget &> /dev/null; then
        wget -O "$MTM_CKPT_FILE" "${MTM_HF_BASE}/model_best.pt" \
            || echo -e "${RED}✗ MtM checkpoint download failed. Please download manually from:${NC}\n  https://huggingface.co/ibl-foundation-model/multi-NDT1-MtM-10-sessions"
    elif command -v curl &> /dev/null; then
        curl -L -o "$MTM_CKPT_FILE" "${MTM_HF_BASE}/model_best.pt" \
            || echo -e "${RED}✗ MtM checkpoint download failed. Please download manually from:${NC}\n  https://huggingface.co/ibl-foundation-model/multi-NDT1-MtM-10-sessions"
    else
        echo -e "${RED}✗ Neither wget nor curl found. Please install one or download checkpoint manually:${NC}"
        echo "  https://huggingface.co/ibl-foundation-model/multi-NDT1-MtM-10-sessions"
        exit 1
    fi

    if [ -f "$MTM_CKPT_FILE" ]; then
        echo -e "${GREEN}✓ MtM checkpoint downloaded ($(du -h "$MTM_CKPT_FILE" | cut -f1))${NC}"
    fi
fi

# Download MtM config (needed for model loading)
if [ -f "$MTM_CFG_FILE" ]; then
    echo "✓ MtM config already exists, skipping download..."
else
    echo "Downloading MtM config..."
    mkdir -p "$MTM_CKPT_DIR"

    if command -v wget &> /dev/null; then
        wget -O "$MTM_CFG_FILE" "${MTM_HF_BASE}/config.yaml" 2>/dev/null \
            || echo -e "${YELLOW}⚠ MtM config download failed (non-critical)${NC}"
    elif command -v curl &> /dev/null; then
        curl -L -o "$MTM_CFG_FILE" "${MTM_HF_BASE}/config.yaml" 2>/dev/null \
            || echo -e "${YELLOW}⚠ MtM config download failed (non-critical)${NC}"
    fi

    if [ -f "$MTM_CFG_FILE" ]; then
        echo -e "${GREEN}✓ MtM config downloaded${NC}"
    fi
fi

echo ""

# =============================================================================
# POYO-1 (PerceiverIO-based Foundation Model)
# =============================================================================
echo -e "${YELLOW}[3/9] Setting up POYO-1...${NC}"

POYO_DIR="$MODELS_DIR/poyo"
POYO_CKPT_DIR="$POYO_DIR/checkpoints"
POYO_CKPT_FILE="$POYO_CKPT_DIR/poyo_1.ckpt"
POYO_OSN_BASE="https://nyu1.osn.mghpcc.org/brainsets-public/model-zoo"

mkdir -p "$POYO_CKPT_DIR"

if [ -f "$POYO_CKPT_FILE" ]; then
    echo "✓ POYO-1 checkpoint already exists, skipping download..."
else
    echo "Downloading POYO-1 checkpoint from NYU OSN..."

    if command -v wget &> /dev/null; then
        wget -O "$POYO_CKPT_FILE" "${POYO_OSN_BASE}/poyo_1.ckpt" \
            || echo -e "${RED}✗ POYO-1 checkpoint download failed. Please download manually from:${NC}\n  ${POYO_OSN_BASE}/poyo_1.ckpt"
    elif command -v curl &> /dev/null; then
        curl -L -o "$POYO_CKPT_FILE" "${POYO_OSN_BASE}/poyo_1.ckpt" \
            || echo -e "${RED}✗ POYO-1 checkpoint download failed. Please download manually from:${NC}\n  ${POYO_OSN_BASE}/poyo_1.ckpt"
    else
        echo -e "${RED}✗ Neither wget nor curl found. Please install one or download checkpoint manually:${NC}"
        echo "  ${POYO_OSN_BASE}/poyo_1.ckpt"
        exit 1
    fi

    if [ -f "$POYO_CKPT_FILE" ]; then
        echo -e "${GREEN}✓ POYO-1 checkpoint downloaded ($(du -h "$POYO_CKPT_FILE" | cut -f1))${NC}"
    fi
fi

echo ""

# =============================================================================
# LaBraM (Large Brain Model for EEG)
# =============================================================================
echo -e "${YELLOW}[4/9] Setting up LaBraM...${NC}"

LABRAM_DIR="$MODELS_DIR/labram"

if [ -d "$LABRAM_DIR" ] && [ -f "$LABRAM_DIR/modeling_finetune.py" ]; then
    echo "✓ LaBraM code already exists, skipping clone..."
else
    echo "Cloning LaBraM repository from GitHub..."
    git clone https://github.com/935963004/LaBraM.git "$LABRAM_DIR"
    echo -e "${GREEN}✓ LaBraM code downloaded${NC}"
fi

# LaBraM checkpoints are included in the repo (via Git LFS)
LABRAM_CKPT_FILE="$LABRAM_DIR/checkpoints/labram-base.pth"

if [ -f "$LABRAM_CKPT_FILE" ]; then
    echo -e "${GREEN}✓ LaBraM checkpoint present ($(du -h "$LABRAM_CKPT_FILE" | cut -f1))${NC}"
else
    echo -e "${YELLOW}⚠ LaBraM checkpoint not found. Try: cd $LABRAM_DIR && git lfs pull${NC}"
fi

echo ""

# =============================================================================
# REVE (Representation for EEG with Versatile Embeddings)
# =============================================================================
echo -e "${YELLOW}[5/9] Setting up REVE...${NC}"

# REVE is loaded from HuggingFace at runtime via transformers AutoModel.
# It requires:
#   1. A HuggingFace account with access granted to brain-bzh/reve-base
#   2. A HuggingFace token set as HF_TOKEN environment variable
#
# The model (~277MB) and position bank are cached automatically on first use.

if [ -n "$HF_TOKEN" ]; then
    echo "✓ HF_TOKEN is set"

    # Pre-download the position bank (not gated, always works)
    echo "Pre-caching REVE position bank..."
    python -c "
from transformers import AutoModel
try:
    pos_bank = AutoModel.from_pretrained('brain-bzh/reve-positions', trust_remote_code=True)
    print('✓ REVE position bank cached')
except Exception as e:
    print(f'⚠ Position bank download failed: {e}')
" 2>/dev/null || echo -e "${YELLOW}⚠ Position bank pre-cache failed (non-critical, will retry at runtime)${NC}"

    # Pre-download the REVE model (gated, needs HF_TOKEN)
    echo "Pre-caching REVE model (brain-bzh/reve-base, ~277MB)..."
    python -c "
from transformers import AutoModel
try:
    model = AutoModel.from_pretrained('brain-bzh/reve-base', trust_remote_code=True)
    print('✓ REVE model cached')
except Exception as e:
    print(f'⚠ REVE model download failed: {e}')
    print('  Make sure you have access at: https://huggingface.co/brain-bzh/reve-base')
" 2>/dev/null || echo -e "${YELLOW}⚠ REVE model pre-cache failed. See instructions below.${NC}"

else
    echo -e "${YELLOW}⚠ HF_TOKEN not set. REVE model cannot be downloaded.${NC}"
    echo "  To set up REVE:"
    echo "    1. Create a HuggingFace account at https://huggingface.co"
    echo "    2. Request access at https://huggingface.co/brain-bzh/reve-base"
    echo "    3. Create a Read token at https://huggingface.co/settings/tokens"
    echo "    4. Set HF_TOKEN in your .env file or environment:"
    echo "       export HF_TOKEN=\"hf_your_token_here\""
    echo "    5. Re-run this script"
fi

echo ""

# =============================================================================
# CBraMod (Criss-Cross Brain Foundation Model for EEG)
# =============================================================================
echo -e "${YELLOW}[6/9] Setting up CBraMod...${NC}"

CBRAMOD_DIR="$MODELS_DIR/cbramod"

if [ -d "$CBRAMOD_DIR" ] && [ -f "$CBRAMOD_DIR/models/cbramod.py" ]; then
    echo "✓ CBraMod code already exists, skipping clone..."
else
    echo "Cloning CBraMod repository from GitHub..."
    git clone https://github.com/wjq-learning/CBraMod.git "$CBRAMOD_DIR"
    echo -e "${GREEN}✓ CBraMod code downloaded${NC}"
fi

# Download CBraMod checkpoint
CBRAMOD_CKPT_DIR="$CBRAMOD_DIR/pretrained_weights"
CBRAMOD_CKPT_FILE="$CBRAMOD_CKPT_DIR/pretrained_weights.pth"
CBRAMOD_HF_BASE="https://huggingface.co/weighting666/CBraMod/resolve/main"

if [ -f "$CBRAMOD_CKPT_FILE" ]; then
    echo "✓ CBraMod checkpoint already exists, skipping download..."
else
    echo "Downloading CBraMod checkpoint from Hugging Face..."
    mkdir -p "$CBRAMOD_CKPT_DIR"

    if command -v wget &> /dev/null; then
        wget -O "$CBRAMOD_CKPT_FILE" "${CBRAMOD_HF_BASE}/pretrained_weights.pth" \
            || echo -e "${RED}✗ CBraMod checkpoint download failed. Please download manually from:${NC}\n  https://huggingface.co/weighting666/CBraMod"
    elif command -v curl &> /dev/null; then
        curl -L -o "$CBRAMOD_CKPT_FILE" "${CBRAMOD_HF_BASE}/pretrained_weights.pth" \
            || echo -e "${RED}✗ CBraMod checkpoint download failed. Please download manually from:${NC}\n  https://huggingface.co/weighting666/CBraMod"
    else
        echo -e "${RED}✗ Neither wget nor curl found. Please install one or download checkpoint manually:${NC}"
        echo "  https://huggingface.co/weighting666/CBraMod"
        exit 1
    fi

    if [ -f "$CBRAMOD_CKPT_FILE" ]; then
        echo -e "${GREEN}✓ CBraMod checkpoint downloaded ($(du -h "$CBRAMOD_CKPT_FILE" | cut -f1))${NC}"
    fi
fi

echo ""

# =============================================================================
# EEGPT (EEG Pretrained Transformer, NeurIPS 2024)
# =============================================================================
echo -e "${YELLOW}[7/9] Setting up EEGPT...${NC}"

EEGPT_DIR="$MODELS_DIR/eegpt"

if [ -d "$EEGPT_DIR" ] && [ -f "$EEGPT_DIR/pretrain/modeling_pretraining.py" ]; then
    echo "✓ EEGPT code already exists, skipping clone..."
else
    echo "Cloning EEGPT repository from GitHub..."
    git clone https://github.com/BINE022/EEGPT.git "$EEGPT_DIR"
    echo -e "${GREEN}✓ EEGPT code downloaded${NC}"
fi

# EEGPT checkpoint (must be downloaded manually from Figshare)
EEGPT_CKPT_DIR="$EEGPT_DIR/checkpoint"
EEGPT_CKPT_FILE="$EEGPT_CKPT_DIR/eegpt_mcae_58chs_4s_large4E.ckpt"

if [ -f "$EEGPT_CKPT_FILE" ]; then
    echo -e "${GREEN}✓ EEGPT checkpoint present ($(du -h "$EEGPT_CKPT_FILE" | cut -f1))${NC}"
else
    echo -e "${YELLOW}⚠ EEGPT checkpoint not found.${NC}"
    echo "  The checkpoint must be downloaded manually from Figshare (WAF blocks automated downloads)."
    echo "  Steps:"
    echo "    1. Download from: https://figshare.com/ndownloader/articles/25866970/versions/2"
    echo "    2. Extract the zip file"
    echo "    3. Place eegpt_mcae_58chs_4s_large4E.ckpt in: $EEGPT_CKPT_DIR/"
    mkdir -p "$EEGPT_CKPT_DIR"
fi

echo ""

# =============================================================================
# EEGMamba (Bidirectional Mamba Foundation Model for EEG)
# =============================================================================
echo -e "${YELLOW}[8/9] Setting up EEGMamba...${NC}"

EEGMAMBA_DIR="$MODELS_DIR/eegmamba"

if [ -d "$EEGMAMBA_DIR" ] && [ -f "$EEGMAMBA_DIR/models/eegmamba.py" ]; then
    echo "✓ EEGMamba code already exists, skipping clone..."
else
    echo "Cloning EEGMamba repository from GitHub..."
    git clone https://github.com/wjq-learning/EEGMamba.git "$EEGMAMBA_DIR"
    echo -e "${GREEN}✓ EEGMamba code downloaded${NC}"
fi

# Download EEGMamba checkpoint
EEGMAMBA_CKPT_DIR="$EEGMAMBA_DIR/pretrained_weights"
EEGMAMBA_CKPT_FILE="$EEGMAMBA_CKPT_DIR/pretrained_EEGMamba.pth"
EEGMAMBA_HF_BASE="https://huggingface.co/weighting666/EEGMamba/resolve/main"

if [ -f "$EEGMAMBA_CKPT_FILE" ]; then
    echo "✓ EEGMamba checkpoint already exists, skipping download..."
else
    echo "Downloading EEGMamba checkpoint from Hugging Face..."
    mkdir -p "$EEGMAMBA_CKPT_DIR"

    if command -v wget &> /dev/null; then
        wget -O "$EEGMAMBA_CKPT_FILE" "${EEGMAMBA_HF_BASE}/pretrained_EEGMamba.pth" \
            || echo -e "${RED}✗ EEGMamba checkpoint download failed. Please download manually from:${NC}\n  https://huggingface.co/weighting666/EEGMamba"
    elif command -v curl &> /dev/null; then
        curl -L -o "$EEGMAMBA_CKPT_FILE" "${EEGMAMBA_HF_BASE}/pretrained_EEGMamba.pth" \
            || echo -e "${RED}✗ EEGMamba checkpoint download failed. Please download manually from:${NC}\n  https://huggingface.co/weighting666/EEGMamba"
    else
        echo -e "${RED}✗ Neither wget nor curl found. Please install one or download checkpoint manually:${NC}"
        echo "  https://huggingface.co/weighting666/EEGMamba"
        exit 1
    fi

    if [ -f "$EEGMAMBA_CKPT_FILE" ]; then
        echo -e "${GREEN}✓ EEGMamba checkpoint downloaded ($(du -h "$EEGMAMBA_CKPT_FILE" | cut -f1))${NC}"
    fi
fi

echo ""

# =============================================================================
# BrainOmni (Multi-modal Brain Foundation Model)
# =============================================================================
echo -e "${YELLOW}[9/9] Setting up BrainOmni...${NC}"

BRAINOMNI_DIR="$MODELS_DIR/brainomni"

if [ -d "$BRAINOMNI_DIR" ] && [ -f "$BRAINOMNI_DIR/brainomni/model.py" ]; then
    echo "✓ BrainOmni code already exists, skipping clone..."
else
    echo "Cloning BrainOmni repository from GitHub..."
    git clone https://github.com/OpenTSLab/BrainOmni.git "$BRAINOMNI_DIR"
    echo -e "${GREEN}✓ BrainOmni code downloaded${NC}"
fi

# BrainOmni checkpoint (tiny model)
BRAINOMNI_CKPT_DIR="$BRAINOMNI_DIR/pretrained_weights/tiny"
BRAINOMNI_CKPT_FILE="$BRAINOMNI_CKPT_DIR/BrainOmni.pt"

BRAINOMNI_HF_BASE="https://huggingface.co/OpenTSLab/BrainOmni/resolve/main/tiny"

if [ -f "$BRAINOMNI_CKPT_FILE" ]; then
    echo -e "${GREEN}✓ BrainOmni checkpoint present ($(du -h "$BRAINOMNI_CKPT_FILE" | cut -f1))${NC}"
else
    echo "Downloading BrainOmni checkpoint from Hugging Face..."
    mkdir -p "$BRAINOMNI_CKPT_DIR"

    if command -v wget &> /dev/null; then
        wget -O "$BRAINOMNI_CKPT_FILE" "${BRAINOMNI_HF_BASE}/BrainOmni.pt" \
            || echo -e "${RED}✗ BrainOmni checkpoint download failed. Please download manually from:${NC}\n  https://huggingface.co/OpenTSLab/BrainOmni"
        wget -O "$BRAINOMNI_CKPT_DIR/model_cfg.json" "${BRAINOMNI_HF_BASE}/model_cfg.json" 2>/dev/null \
            || echo -e "${YELLOW}⚠ BrainOmni config download failed (non-critical)${NC}"
    elif command -v curl &> /dev/null; then
        curl -L -o "$BRAINOMNI_CKPT_FILE" "${BRAINOMNI_HF_BASE}/BrainOmni.pt" \
            || echo -e "${RED}✗ BrainOmni checkpoint download failed. Please download manually from:${NC}\n  https://huggingface.co/OpenTSLab/BrainOmni"
        curl -L -o "$BRAINOMNI_CKPT_DIR/model_cfg.json" "${BRAINOMNI_HF_BASE}/model_cfg.json" 2>/dev/null \
            || echo -e "${YELLOW}⚠ BrainOmni config download failed (non-critical)${NC}"
    else
        echo -e "${RED}✗ Neither wget nor curl found. Please install one or download checkpoint manually:${NC}"
        echo "  https://huggingface.co/OpenTSLab/BrainOmni"
        exit 1
    fi

    if [ -f "$BRAINOMNI_CKPT_FILE" ] && [ -s "$BRAINOMNI_CKPT_FILE" ]; then
        echo -e "${GREEN}✓ BrainOmni checkpoint downloaded ($(du -h "$BRAINOMNI_CKPT_FILE" | cut -f1))${NC}"
    else
        rm -f "$BRAINOMNI_CKPT_FILE"
        echo -e "${RED}✗ BrainOmni checkpoint download failed or file is empty${NC}"
    fi
fi

echo ""

# =============================================================================
# Datasets (OpenNeuro)
# =============================================================================
echo -e "${GREEN}=== Dataset Setup ===${NC}"
echo ""

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATASET_DIR="$SCRIPT_DIR/dataset"

# --- Ensure openneuro-py is available ---
if ! python -c "import openneuro" 2>/dev/null; then
    echo "Installing openneuro-py..."
    pip install openneuro-py || {
        echo -e "${RED}✗ Failed to install openneuro-py. Please install manually: pip install openneuro-py${NC}"
        echo "  Then re-run this script."
    }
fi

# --- ds004504 (AD/FTD/Healthy EEG, 88 subjects, ~5.4GB) ---
echo -e "${YELLOW}[Dataset 1/2] ds004504 (AD/FTD/Healthy EEG classification)${NC}"

DS004504_DIR="$DATASET_DIR/AD/ds004504"

if [ -d "$DS004504_DIR" ] && [ -f "$DS004504_DIR/participants.tsv" ]; then
    echo "✓ ds004504 already exists, skipping download..."
else
    echo "Downloading ds004504 from OpenNeuro (~5.4GB)..."
    mkdir -p "$DATASET_DIR/AD"
    python -c "import openneuro; openneuro.download(dataset='ds004504', target_dir='$DS004504_DIR')" \
        || echo -e "${RED}✗ ds004504 download failed. Please download manually:${NC}\n  python -c \"import openneuro; openneuro.download(dataset='ds004504', target_dir='$DS004504_DIR')\""

    if [ -f "$DS004504_DIR/participants.tsv" ]; then
        echo -e "${GREEN}✓ ds004504 downloaded ($(du -sh "$DS004504_DIR" | cut -f1))${NC}"
    fi
fi

echo ""

# --- ds004584 (PD/Control EEG, 149 subjects, ~2.9GB) ---
echo -e "${YELLOW}[Dataset 2/2] ds004584 (PD/Control EEG classification)${NC}"

DS004584_DIR="$DATASET_DIR/Parkinson/ds004584"

if [ -d "$DS004584_DIR" ] && [ -f "$DS004584_DIR/participants.tsv" ]; then
    echo "✓ ds004584 already exists, skipping download..."
else
    echo "Downloading ds004584 from OpenNeuro (~2.9GB)..."
    mkdir -p "$DATASET_DIR/Parkinson"
    python -c "import openneuro; openneuro.download(dataset='ds004584', target_dir='$DS004584_DIR')" \
        || echo -e "${RED}✗ ds004584 download failed. Please download manually:${NC}\n  python -c \"import openneuro; openneuro.download(dataset='ds004584', target_dir='$DS004584_DIR')\""

    if [ -f "$DS004584_DIR/participants.tsv" ]; then
        echo -e "${GREEN}✓ ds004584 downloaded ($(du -sh "$DS004584_DIR" | cut -f1))${NC}"
    fi
fi

echo ""

# --- WM dataset (DANDI 000006, included in repo) ---
WM_DIR="$DATASET_DIR/Working_Memory/dandi_000006"
if [ -d "$WM_DIR" ]; then
    echo "✓ WM dataset (dandi_000006) already present in repo"
else
    echo -e "${YELLOW}⚠ WM dataset (dandi_000006) not found at $WM_DIR${NC}"
    echo "  This dataset should be included in the git repo. Try: git checkout -- dataset/Working_Memory/"
fi

echo ""

# =============================================================================
# Summary
# =============================================================================
echo -e "${GREEN}=== Setup Complete ===${NC}"
echo ""
echo "Models installed in: $MODELS_DIR"
echo ""
echo "Installed models:"
if [ -d "$NDT3_DIR/context_general_bci" ]; then
    echo "  ✓ NDT3 (Neural Data Transformer 3)"
    if [ -f "$CHECKPOINT_FILE" ]; then
        echo "    ✓ Checkpoint: val-epoch=397-val_loss=0.4987.ckpt"
    else
        echo "    ✗ Checkpoint not found (download failed)"
    fi
else
    echo "  ✗ NDT3 (setup failed)"
fi
if [ -d "$MTM_DIR/src" ]; then
    echo "  ✓ MtM (Multi-task Masking / NDT1)"
    if [ -f "$MTM_CKPT_FILE" ]; then
        echo "    ✓ Checkpoint: multi-NDT1-MtM-10-sessions/model_best.pt"
    else
        echo "    ✗ Checkpoint not found (download failed)"
    fi
else
    echo "  ✗ MtM (setup failed)"
fi
if [ -d "$POYO_DIR" ] && [ -f "$POYO_CKPT_FILE" ]; then
    echo "  ✓ POYO-1 (PerceiverIO Foundation Model)"
    echo "    ✓ Checkpoint: poyo_1.ckpt"
else
    echo "  ✗ POYO-1 (setup failed or checkpoint not downloaded)"
fi
if [ -d "$LABRAM_DIR" ] && [ -f "$LABRAM_DIR/modeling_finetune.py" ]; then
    echo "  ✓ LaBraM (Large Brain Model for EEG)"
    if [ -f "$LABRAM_CKPT_FILE" ]; then
        echo "    ✓ Checkpoint: labram-base.pth"
    else
        echo "    ✗ Checkpoint not found (run git lfs pull in $LABRAM_DIR)"
    fi
else
    echo "  ✗ LaBraM (setup failed)"
fi
if [ -n "$HF_TOKEN" ]; then
    echo "  ✓ REVE (loaded from HuggingFace at runtime)"
    echo "    Model: brain-bzh/reve-base (69.2M params, 512-dim features)"
else
    echo "  ⚠ REVE (HF_TOKEN not set — see instructions above)"
fi
if [ -d "$CBRAMOD_DIR" ] && [ -f "$CBRAMOD_DIR/models/cbramod.py" ]; then
    echo "  ✓ CBraMod (Criss-Cross Brain Foundation Model)"
    if [ -f "$CBRAMOD_CKPT_FILE" ]; then
        echo "    ✓ Checkpoint: pretrained_weights.pth"
    else
        echo "    ✗ Checkpoint not found (download failed)"
    fi
else
    echo "  ✗ CBraMod (setup failed)"
fi
if [ -d "$EEGPT_DIR" ] && [ -f "$EEGPT_DIR/pretrain/modeling_pretraining.py" ]; then
    echo "  ✓ EEGPT (EEG Pretrained Transformer, NeurIPS 2024)"
    if [ -f "$EEGPT_CKPT_FILE" ]; then
        echo "    ✓ Checkpoint: eegpt_mcae_58chs_4s_large4E.ckpt"
    else
        echo "    ✗ Checkpoint not found (manual download required from Figshare)"
    fi
else
    echo "  ✗ EEGPT (setup failed)"
fi
if [ -d "$EEGMAMBA_DIR" ] && [ -f "$EEGMAMBA_DIR/models/eegmamba.py" ]; then
    echo "  ✓ EEGMamba (Bidirectional Mamba Foundation Model)"
    if [ -f "$EEGMAMBA_CKPT_FILE" ]; then
        echo "    ✓ Checkpoint: pretrained_EEGMamba.pth"
    else
        echo "    ✗ Checkpoint not found (download failed)"
    fi
else
    echo "  ✗ EEGMamba (setup failed)"
fi
if [ -d "$BRAINOMNI_DIR" ] && [ -f "$BRAINOMNI_DIR/brainomni/model.py" ]; then
    echo "  ✓ BrainOmni (Multi-modal Brain Foundation Model)"
    if [ -f "$BRAINOMNI_CKPT_FILE" ]; then
        echo "    ✓ Checkpoint: tiny/BrainOmni.pt"
    else
        echo "    ✗ Checkpoint not found (download failed)"
    fi
else
    echo "  ✗ BrainOmni (setup failed)"
fi
echo ""

# Verify .env exists
if [ ! -f .env ]; then
    echo -e "${YELLOW}⚠ .env file not found. Creating from .env.example...${NC}"
    if [ -f .env.example ]; then
        cp .env.example .env
        echo "Please edit .env to configure API keys and other settings."
    else
        echo -e "${RED}✗ .env.example not found. Please create .env manually.${NC}"
    fi
    echo ""
fi

echo ""
echo "Datasets:"
if [ -f "$DS004504_DIR/participants.tsv" ]; then
    echo "  ✓ ds004504 (AD/FTD/Healthy, 88 subjects)"
else
    echo "  ✗ ds004504 (not downloaded)"
fi
if [ -f "$DS004584_DIR/participants.tsv" ]; then
    echo "  ✓ ds004584 (PD/Control, 149 subjects)"
else
    echo "  ✗ ds004584 (not downloaded)"
fi
if [ -d "$WM_DIR" ]; then
    echo "  ✓ dandi_000006 (Working Memory, included in repo)"
else
    echo "  ✗ dandi_000006 (missing — check git)"
fi
echo ""

echo "Next steps:"
echo "  1. Edit .env if needed (API keys, model paths, HF_TOKEN)"
echo "  2. Run: docker-compose build"
echo "  3. Run: docker-compose up -d"
echo ""
echo "For REVE (EEG foundation model):"
echo "  - Request access: https://huggingface.co/brain-bzh/reve-base"
echo "  - Set HF_TOKEN in .env or environment"
echo ""
echo -e "${GREEN}Ready to use NS-Copilot with foundation models!${NC}"
