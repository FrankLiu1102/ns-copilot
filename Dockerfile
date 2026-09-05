# Dockerfile for NS-Copilot Web Application with CUDA support for foundation models (NDT3, MtM, POYO-1)
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

# Install Python 3.10
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js 20 (required for Codex CLI baseline)
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs && \
    echo "=== Node.js $(node --version) installed ==="

# Install Codex CLI (OpenAI's coding agent, used as baseline in auto mode)
RUN npm install -g @openai/codex && \
    echo "=== Codex CLI $(codex --version 2>/dev/null || echo 'installed') ==="

# ---- Dependencies layer (cached unless requirements-docker.txt changes) ----
COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt && \
    echo "=== Package installation complete ==="

# Enforce numpy < 2 (some older deps may not support numpy 2.x)
RUN pip install --no-cache-dir "numpy>=1.24.0,<2.0.0" && \
    python -c "import numpy; print(f'numpy {numpy.__version__}')"

# Install flash-attn (requires CUDA, separate step for better caching)
# Pinned to 2.7.4 for NDT3 RotaryEmbedding compatibility (pos_idx_in_fp32 param)
RUN pip install --no-cache-dir flash-attn==2.7.4.post1 --no-build-isolation && \
    echo "=== Flash Attention 2.7.4.post1 installed successfully ==="

# Install mamba-ssm + causal-conv1d for EEGMamba (requires CUDA compilation)
# --no-build-isolation ensures compilation uses the installed PyTorch headers
# (avoids ABI mismatch between base image CUDA 12.1 and PyTorch cu124)
RUN pip install --no-cache-dir causal-conv1d --no-build-isolation && \
    pip install --no-cache-dir mamba-ssm --no-build-isolation && \
    python -c "from mamba_ssm import Mamba; print('=== Mamba SSM installed and verified ===')"

# Install dependencies for BrainOmni (pure Python, no CUDA)
RUN pip install --no-cache-dir vector-quantize-pytorch einx && \
    python -c "from vector_quantize_pytorch import VectorQuantize; print('=== BrainOmni deps installed ===')"

# ---- Code layer (only this rebuilds when code changes) ----
COPY . .

# Create symbolic link to make neuro_copilot package accessible
# This allows "from neuro_copilot.core import ..." to work
RUN ln -sf /app /app/neuro_copilot && \
    echo "=== Created neuro_copilot symlink ===" && \
    python -c "from neuro_copilot.core import web_adapter; print('✓ neuro_copilot.core imported successfully')" || echo "✗ Failed to import"

# Expose port for web service (overridden to 8001 in docker-compose.yml)
EXPOSE 8001

# Environment variables (can be overridden)
ENV ANTHROPIC_API_KEY=""
ENV OPENAI_API_KEY=""
ENV HF_TOKEN=""
ENV HOST="0.0.0.0"
ENV PORT=8001
ENV PYTHONPATH="/app:${PYTHONPATH}"

# Health check - uses $PORT so it works regardless of port override
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8001/', timeout=5)" || exit 1

# Run web application
CMD ["python", "web_app.py"]
