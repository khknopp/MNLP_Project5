# Sourced by every job and by setup.sh. Snellius environment: uv-managed venv, HF caches inside the repo.
export PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$PROJECT_DIR"

export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR="$PROJECT_DIR/.cache/uv"
export HF_HOME="$PROJECT_DIR/.cache/huggingface"
export XDG_CACHE_HOME="$PROJECT_DIR/.cache"
export TOKENIZERS_PARALLELISM=false

# Compute nodes have no internet: jobs only use what slurm/setup.sh cached on the login node.
if [ -n "${SLURM_JOB_ID:-}" ]; then
    export UV_FROZEN=1 UV_OFFLINE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
fi
mkdir -p logs
