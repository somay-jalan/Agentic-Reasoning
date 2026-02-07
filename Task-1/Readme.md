# LLM Agent Proxy for arXiv → Beamer

**(OpenWebUI + OpenRouter)**

This repository implements a **tool-augmented LLM agent** that can:

* 🔍 Search arXiv for research papers
* 📄 Fetch paper text (LaTeX preferred, PDF fallback)
* 🎞️ Generate **LaTeX Beamer presentations** from arXiv papers
* ✅ **Automatically verify** presentation quality and accuracy
* 🔁 **Auto-improve** presentations that score below threshold
* 📑 **Compile presentations to PDF** using `xelatex`
* 🌊 Stream responses in **OpenAI-compatible SSE format**
* 🔌 Plug directly into **OpenWebUI** as a custom OpenAI-compatible backend

The server is built with **FastAPI**, uses **OpenRouter** for model access, and is served via **Uvicorn**.

---

## 1. Architecture Overview

```
OpenWebUI
   |
   |  (OpenAI-compatible /v1/chat/completions)
   v
FastAPI Agent Server
   |
   |-- LLM (via OpenRouter)
   |-- Tool Calling
       |-- arxiv_search
       |-- arxiv_to_text
       |-- read_file
       |-- create_beamer_presentation
       |-- verify_beamer_presentation (automatic)
       |-- compile_beamer_to_pdf
```

### Key Features

* Fully **OpenAI API compatible**
* Supports **streaming (SSE)** responses
* Automatic **tool calling** with multi-step reasoning
* **Intelligent search discipline** (max 3 searches per request)
* **Organized file structure**
* **Automatic verification** of generated presentations
* **Auto-improvement loop** for low-quality outputs
* **PDF compilation** with `xelatex`
* **Context-aware truncation** to prevent token overflow
* **Coherent summaries** when reaching interaction limits

---

## 2. Requirements

### System

* Python **3.11+**
* **xelatex** (for PDF compilation)

#### Linux

```bash
sudo apt-get install texlive-xetex texlive-fonts-extra
```

#### macOS

```bash
brew install --cask mactex
```

#### Windows

Install **MiKTeX** or **TeX Live**

---

### Python Dependencies

```bash
pip install -r requirements.txt
```

---

## 3. Environment Variables

Create a `.env` file in the project root:

```env
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

This key is used by:

* The **main agent**
* The **Beamer generation tool**
* The **verification system**

---

## 4. Project Structure

```
.
├── agent_server.py          # FastAPI server (main entrypoint)
├── arxiv_tool.py            # arXiv tools + Beamer generator + verifier
├── setup_run_tmux.sh        # One-command setup + launch script
├── .env                     # OpenRouter API key
├── output/
│   ├── presentations_tex/   # Generated LaTeX files
│   ├── presentations_pdf/   # Compiled PDF presentations
│   └── papers/              # Downloaded arXiv papers
├── requirements.txt
├── README.md
```

All generated files are automatically organized under `output/`.

---

## 5. Running the Agent Manually (Uvicorn)

```bash
uvicorn agent_server:app --host 0.0.0.0 --port 8000
```

* OpenAI-compatible endpoint:

```
http://localhost:8000/v1/chat/completions
```

---

## 6. One-Command Setup with tmux (Recommended) - Only for linux

The repository includes a helper script:

```
setup_run_tmux.sh
```

This script **sets up everything and launches both OpenWebUI and the agent server inside a tmux session**.

### What the Script Does

1. Creates and activates a Python virtual environment
2. Installs all required Python packages
3. Installs LaTeX dependencies for PDF compilation
4. Starts a tmux session named `webui`
5. Launches:

   * **OpenWebUI** on port `8080`
   * **FastAPI agent server** on port `8000`
6. Attaches you to the tmux session

---

### Script Contents

```bash
#!/usr/bin/env bash
set -e

SESSION="webui"
ENV_NAME="webui_env"
PYTHON_BIN="python3"

$PYTHON_BIN -m venv $ENV_NAME
source $ENV_NAME/bin/activate
pip install -r requirements.txt || true
pip install uvicorn open-webui

sudo apt-get update
sudo apt-get install -y texlive-xetex texlive-fonts-recommended texlive-latex-extra

tmux new-session -d -s $SESSION
export OPEN_WEBUI_PORT=8080
tmux send-keys -t $SESSION "
source $ENV_NAME/bin/activate
open-webui serve
" C-m

tmux split-window -h -t $SESSION

tmux send-keys -t $SESSION "
source $ENV_NAME/bin/activate
uvicorn agent_server:app --host 0.0.0.0 --port 8000
" C-m

tmux attach -t $SESSION

WEBUI_PORT=\${OPEN_WEBUI_PORT:-8080}
echo \"Application is live at http://localhost:\$WEBUI_PORT\"
```

---

### How to Use

```bash
chmod +x setup_run_tmux.sh
./setup_run_tmux.sh
```

After launch:

* **OpenWebUI** → [http://localhost:8080](http://localhost:8080)
* **Agent API** → [http://localhost:8000/v1/chat/completions](http://localhost:8000/v1/chat/completions)

#### tmux Tips

* Detach: `Ctrl + B`, then `D`
* Reattach: `tmux attach -t webui`
* Kill session: `tmux kill-session -t webui`

---

## 7. Connecting to OpenWebUI

1. Go to **Settings → Admin Settings → Connections**
2. Add a **Custom OpenAI-Compatible API**

| Field        | Value                       |
| ------------ | --------------------------- |
| API Base URL | `http://localhost:8000/v1`  |
| API Key      | `dummy` (ignored by server) |
| Streaming    | Enabled                     |

---

## 8. Supported Models

Default model:

```python
DEFAULT_MODEL = "openai/gpt-oss-120b"
```

You can switch to any OpenRouter-supported model:

* `openai/gpt-4o`
* `anthropic/claude-3.5-sonnet`
* `google/gemini-pro`

---

## 9. Available Tools (Auto-Invoked)

### `arxiv_search`

Searches arXiv with a **max of 3 searches per request**.

---

### `arxiv_to_text`

Fetches paper content:

* LaTeX source preferred
* PDF fallback
* Saves to `output/papers/`

---

### `create_beamer_presentation`

Generates LaTeX Beamer slides:

* Auto-fetches paper if needed
* Saves to `output/presentations_tex/`
* Auto-verifies quality
* Auto-improves if score < 7/10

---

### `verify_beamer_presentation`

Scores:

* Accuracy
* Completeness
* Clarity
* Technical correctness

Triggers regeneration if below threshold.

---

### `compile_beamer_to_pdf`

* Uses `xelatex` (run twice)
* Saves to `output/presentations_pdf/`
* **Always asks user permission**

---

## 10. Streaming Behavior

* Uses **Server-Sent Events (SSE)**
* Matches OpenAI streaming format
* Tool calls are hidden from user
* Only final coherent output is streamed

---

## 11. Agent Intelligence Features

* **Search discipline** (never loops endlessly)
* **Context-aware truncation**
* **15 tool-interaction limit**
* Graceful summarization at limits
* **No chain-of-thought leakage**

---

## 12. Output Files

```
output/
├── presentations_tex/
│   └── presentation_1412.6980.tex
├── presentations_pdf/
│   └── presentation_1412.6980.pdf
└── papers/
    ├── 1412.6980.pdf
    └── 1412.6980_source.tar.gz
```

---

## 13. Configuration

### `agent_server.py`

```python
DEFAULT_MODEL = "openai/gpt-oss-120b"
MAX_TURNS = 15
VERIFICATION_THRESHOLD = 7
```

### `arxiv_tool.py`

```python
FOLDERS = {
    "presentations_tex": Path("output/presentations_tex"),
    "presentations_pdf": Path("output/presentations_pdf"),
    "papers": Path("output/papers"),
}
```

---

## 14. Troubleshooting

* **xelatex not found** → install TeX Live / MiKTeX
* **Context too large** → use larger-context model
* **Verification always fails** → lower threshold
* **PDF compile errors** → inspect `.log` file

---

## 15. Advanced Usage

* Customize verification prompts
* Change Beamer theme & slide density
* Add IEEE / ACM formats
* Improve figure & table handling
* Multi-language support

---

## 16. API Reference

### `POST /v1/chat/completions`

```json
{
  "model": "openai/gpt-oss-120b",
  "messages": [{"role": "user", "content": "Find papers on LLMs"}],
  "stream": true
}
```

---

### `GET /v1/models`

Returns available models.

---

## 17. License

**MIT License**

---

## 18. Contributing

Contributions welcome:

* Presentation themes
* Verification metrics
* Dataset-aware slides
* Citation automation

---

## 19. Acknowledgments

Built with:

* FastAPI
* OpenRouter
* arXiv API
* OpenWebUI
