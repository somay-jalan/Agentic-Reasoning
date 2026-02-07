#!/usr/bin/env bash
set -e

SESSION="webui"
ENV_NAME="webui_env"

# --------------------------------------------------
# Enforce Python version (3.11 or 3.12 only)
# --------------------------------------------------

if command -v python3.11 >/dev/null 2>&1; then
    PYTHON_BIN="python3.11"
elif command -v python3.12 >/dev/null 2>&1; then
    PYTHON_BIN="python3.12"
else
    echo "❌ Python 3.11 or 3.12 is required to run Open-WebUI."
    echo "   Installed versions:"
    python3 --version || true
    exit 1
fi

PY_VERSION=$($PYTHON_BIN - <<'EOF'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
EOF
)

echo "✅ Using Python $PY_VERSION ($PYTHON_BIN)"

# --------------------------------------------------
# Virtual environment
# --------------------------------------------------

$PYTHON_BIN -m venv "$ENV_NAME"
source "$ENV_NAME/bin/activate"

pip install --upgrade pip setuptools wheel

pip install -r requirements.txt || true
pip install open-webui

# --------------------------------------------------
# System dependencies
# --------------------------------------------------

sudo apt-get update
sudo apt-get install -y \
    texlive-xetex \
    texlive-fonts-recommended \
    texlive-latex-extra

# --------------------------------------------------
# tmux setup
# --------------------------------------------------

tmux new-session -d -s "$SESSION"

export OPEN_WEBUI_PORT=8080

tmux send-keys -t "$SESSION" "
source $ENV_NAME/bin/activate
open-webui serve
" C-m

tmux split-window -h -t "$SESSION"

tmux send-keys -t "$SESSION" "
source $ENV_NAME/bin/activate
uvicorn agent_server:app --host 0.0.0.0 --port 8000
" C-m

tmux attach -t "$SESSION"

WEBUI_PORT=\${OPEN_WEBUI_PORT:-8080}
echo \"Application is live at http://localhost:\$WEBUI_PORT\"
