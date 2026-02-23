#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# setup.sh — One-time environment setup for the Phage Therapy
#             Prediction System.
#
# Creates a Python virtual environment, installs backend and
# frontend dependencies.
#
# Usage:
#   chmod +x setup.sh
#   ./setup.sh
# ──────────────────────────────────────────────────────────────
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$PROJECT_ROOT/.venv"

echo "═══════════════════════════════════════════════════════"
echo "  Phage Therapy Prediction System — Setup"
echo "═══════════════════════════════════════════════════════"

# ──── 1. Python virtual environment ─────────────────────────
echo ""
echo "▸ Step 1/3: Python virtual environment"

if [ -d "$VENV_DIR" ]; then
    echo "  Virtual environment already exists at .venv/"
else
    echo "  Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
    echo "  ✓ Created .venv/"
fi

# Activate
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
echo "  ✓ Activated ($(python --version))"

# ──── 2. Backend (Python) dependencies ──────────────────────
echo ""
echo "▸ Step 2/3: Installing Python dependencies"
pip install --upgrade pip -q
pip install -r "$PROJECT_ROOT/requirements.txt" -q
echo "  ✓ All Python packages installed"

# ──── 3. Frontend (Node.js) dependencies ────────────────────
echo ""
echo "▸ Step 3/3: Installing frontend dependencies"

if ! command -v node &>/dev/null; then
    echo "  ⚠  Node.js not found. Please install Node.js (v18+) and re-run."
    echo "     https://nodejs.org/"
    exit 1
fi

cd "$PROJECT_ROOT/frontend"
npm install --silent
echo "  ✓ Frontend packages installed"

cd "$PROJECT_ROOT"

# ──── Done ──────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════"
echo "  Setup complete!"
echo ""
echo "  Next steps:"
echo "    1. Prepare data:   python scripts/prepare_dataset.py"
echo "    2. Train models:   python scripts/train_models.py"
echo "    3. Launch the app: ./run.sh"
echo "═══════════════════════════════════════════════════════"
