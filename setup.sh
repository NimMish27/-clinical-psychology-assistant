#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
# Clinical Psychology Assistant — Environment Setup
# Usage: bash setup.sh [--dev]
#
# Flags:
#   --dev     Install dev dependencies (ruff, mypy, pre-commit, ipykernel)
# ══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

# ── Colour helpers ────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
info()    { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

DEV_MODE=false
for arg in "$@"; do [[ "$arg" == "--dev" ]] && DEV_MODE=true; done

# ── 1. Python version check ───────────────────────────────────────────────────
info "Checking Python version..."
PYTHON_BIN=""
for candidate in python3.11 python3 python; do
    if command -v "$candidate" &>/dev/null; then
        VERSION=$("$candidate" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
        MAJOR=$(echo "$VERSION" | cut -d. -f1)
        MINOR=$(echo "$VERSION" | cut -d. -f2)
        if [[ "$MAJOR" -eq 3 && "$MINOR" -ge 11 ]]; then
            PYTHON_BIN="$candidate"
            info "Found $candidate ($VERSION) ✓"
            break
        fi
    fi
done

[[ -z "$PYTHON_BIN" ]] && error "Python 3.11+ is required. Install it from https://python.org and retry."

# ── 2. Virtual environment ────────────────────────────────────────────────────
VENV_DIR=".venv"
if [[ -d "$VENV_DIR" ]]; then
    warn "Virtual environment already exists at $VENV_DIR — skipping creation."
else
    info "Creating virtual environment at $VENV_DIR ..."
    "$PYTHON_BIN" -m venv "$VENV_DIR"
    info "Virtual environment created ✓"
fi

# Activate
# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"
info "Virtual environment activated ✓"

# ── 3. Upgrade pip / setuptools ───────────────────────────────────────────────
info "Upgrading pip and setuptools..."
pip install --quiet --upgrade pip setuptools wheel

# ── 4. Install dependencies ───────────────────────────────────────────────────
if [[ "$DEV_MODE" == true ]]; then
    info "Installing dev dependencies from requirements-dev.txt..."
    pip install --quiet -r requirements-dev.txt
else
    info "Installing production dependencies from requirements.txt..."
    pip install --quiet -r requirements.txt
fi
info "Dependencies installed ✓"

# ── 5. Create .env if missing ─────────────────────────────────────────────────
if [[ ! -f ".env" ]]; then
    cp .env.example .env
    warn ".env created from .env.example — update SECRET_KEY and review all values before running."
else
    info ".env already exists — skipping."
fi

# ── 6. Create runtime directories ────────────────────────────────────────────
info "Creating runtime directories..."
mkdir -p data/{raw,processed,chroma} logs
info "Directories created ✓"

# ── 7. Pre-commit hooks (dev only) ────────────────────────────────────────────
if [[ "$DEV_MODE" == true ]] && command -v pre-commit &>/dev/null; then
    if [[ -f ".pre-commit-config.yaml" ]]; then
        info "Installing pre-commit hooks..."
        pre-commit install
        info "Pre-commit hooks installed ✓"
    fi
fi

# ── 8. Ollama check ───────────────────────────────────────────────────────────
echo ""
if command -v ollama &>/dev/null; then
    info "Ollama detected ✓"
    info "To pull the model:  ollama pull llama3.1:8b"
else
    warn "Ollama not found. Install from https://ollama.com, then run:"
    warn "  ollama pull llama3.1:8b"
fi

# ── 9. Summary ────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}══════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  Setup complete!${NC}"
echo -e "${GREEN}══════════════════════════════════════════════════════${NC}"
echo ""
echo "  Activate environment : source .venv/bin/activate"
echo "  Start API server     : uvicorn api.main:app --reload"
echo "  Run ingestion        : python ingestion/run_ingestion.py"
echo "  Run tests            : pytest"
echo ""
