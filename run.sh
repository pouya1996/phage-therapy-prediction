#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# run.sh — Start the Phage Therapy Prediction System
#           (backend API + frontend dev server).
#
# Usage:
#   ./run.sh            Start both servers
#   ./run.sh --backend  Start backend only
#   ./run.sh --stop     Stop all running servers
# ──────────────────────────────────────────────────────────────
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$PROJECT_ROOT/.venv"
BACKEND_PORT=8000
FRONTEND_PORT=3000
BACKEND_PID_FILE="$PROJECT_ROOT/.backend.pid"
FRONTEND_PID_FILE="$PROJECT_ROOT/.frontend.pid"

# ──── Helpers ───────────────────────────────────────────────
stop_servers() {
    local stopped=0
    for pidfile in "$BACKEND_PID_FILE" "$FRONTEND_PID_FILE"; do
        if [ -f "$pidfile" ]; then
            pid=$(cat "$pidfile")
            if kill -0 "$pid" 2>/dev/null; then
                kill "$pid" 2>/dev/null || true
                stopped=1
                echo "  Stopped PID $pid"
            fi
            rm -f "$pidfile"
        fi
    done
    # Also kill anything on the ports
    lsof -ti:"$BACKEND_PORT" 2>/dev/null | xargs kill -9 2>/dev/null || true
    lsof -ti:"$FRONTEND_PORT" 2>/dev/null | xargs kill -9 2>/dev/null || true
    if [ $stopped -eq 1 ]; then
        echo "  ✓ Servers stopped"
    else
        echo "  No running servers found"
    fi
}

# ──── Handle --stop ─────────────────────────────────────────
if [ "${1:-}" = "--stop" ]; then
    echo "Stopping servers..."
    stop_servers
    exit 0
fi

# ──── Pre-flight checks ────────────────────────────────────
if [ ! -d "$VENV_DIR" ]; then
    echo "Error: Virtual environment not found. Run ./setup.sh first."
    exit 1
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# Check that models exist
if [ ! -d "$PROJECT_ROOT/models" ] || [ -z "$(ls "$PROJECT_ROOT/models" 2>/dev/null)" ]; then
    echo "⚠  No trained models found in models/."
    echo "   Run 'python scripts/train_models.py' first to train models."
    echo "   The prediction API requires trained models to function."
    echo ""
    read -rp "Continue anyway? [y/N] " yn
    case "$yn" in
        [Yy]*) ;;
        *) exit 1 ;;
    esac
fi

# Stop any already-running servers
echo "Checking for existing servers…"
stop_servers 2>/dev/null
sleep 1   # let ports fully release after kill

echo "═══════════════════════════════════════════════════════"
echo "  Phage Therapy Prediction System"
echo "═══════════════════════════════════════════════════════"

# ──── Start backend ─────────────────────────────────────────
echo ""
echo "▸ Starting backend API on http://localhost:$BACKEND_PORT"
cd "$PROJECT_ROOT"
uvicorn backend.app:app \
    --host 0.0.0.0 \
    --port "$BACKEND_PORT" \
    > "$PROJECT_ROOT/.backend.log" 2>&1 &
echo $! > "$BACKEND_PID_FILE"

# Wait for backend to be ready
echo -n "  Waiting for backend"
for i in $(seq 1 30); do
    if curl -s "http://localhost:$BACKEND_PORT/api/health" >/dev/null 2>&1; then
        echo ""
        echo "  ✓ Backend ready (PID $(cat "$BACKEND_PID_FILE"))"
        break
    fi
    echo -n "."
    sleep 1
    if [ "$i" -eq 30 ]; then
        echo ""
        echo "  ✗ Backend failed to start. Check .backend.log"
        exit 1
    fi
done

# ──── Start frontend (unless --backend only) ────────────────
if [ "${1:-}" != "--backend" ]; then
    echo ""
    echo "▸ Starting frontend on http://localhost:$FRONTEND_PORT"
    cd "$PROJECT_ROOT/frontend"
    npm run dev > "$PROJECT_ROOT/.frontend.log" 2>&1 &
    echo $! > "$FRONTEND_PID_FILE"
    sleep 2
    echo "  ✓ Frontend ready (PID $(cat "$FRONTEND_PID_FILE"))"
fi

# ──── Summary ───────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════"
echo "  Application is running!"
echo ""
echo "  Frontend : http://localhost:$FRONTEND_PORT"
echo "  API      : http://localhost:$BACKEND_PORT/api/health"
echo ""
echo "  Stop all : ./run.sh --stop"
echo "  Logs     : .backend.log / .frontend.log"
echo "═══════════════════════════════════════════════════════"
