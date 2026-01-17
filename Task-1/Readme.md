# LLM Agent Proxy for arXiv → Beamer (OpenWebUI + OpenRouter)

This repository implements a **tool-augmented LLM agent** that can:

* Search arXiv
* Fetch paper text (LaTeX preferred, PDF fallback)
* Generate a **LaTeX Beamer presentation** from an arXiv paper
* Stream responses in **OpenAI-compatible SSE format**
* Plug directly into **OpenWebUI** as a custom OpenAI-compatible backend

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
       |-- create_beamer_presentation
```

### Key Features

* Fully **OpenAI API compatible**
* Supports **streaming (SSE)** responses
* Automatic **tool calling**
* Handles **multi-step reasoning** (search → fetch → generate)
* Saves generated Beamer `.tex` files locally

---

## 2. Requirements

### System

* Python **3.11+**
* Linux / macOS (Windows works with minor tweaks)

### Python Dependencies

Install all required packages:

```bash
pip install fastapi uvicorn openai python-dotenv requests feedparser pymupdf
```

> `pymupdf` (imported as `fitz`) is required for PDF fallback text extraction.

---

## 3. Environment Variables

Create a `.env` file in the project root:

```env
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

This key is used by:

* The **main agent**
* The **Beamer generation tool** (separate internal client)

---

## 4. Project Structure

```
.
├── agent_server.py          # FastAPI server (main entrypoint)
├── arxiv_tool.py            # arXiv tools + Beamer generator
├── .env                     # OpenRouter API key
├── README.md
```

> Your file may be named differently, but `agent_server:app` must point to the FastAPI app.

---

## 5. Running the Server (Uvicorn)

```bash
uvicorn agent_server:app --host 0.0.0.0 --port 8000
```

What this does:

* Starts the FastAPI server
* Exposes an **OpenAI-compatible API** at:

  ```
  http://localhost:8000/v1/chat/completions
  ```

---

## 6. Connecting to OpenWebUI


Please visit [open-webui](https://github.com/open-webui/open-webui) to start the OpenWebUI application. Once the application is running, proceed with the next steps.
### Step 1: Open OpenWebUI Settings

* Go to **Settings → Models / Providers**
* Add a **Custom OpenAI-Compatible API**

### Step 2: Provider Configuration

| Field        | Value                                  |
| ------------ | -------------------------------------- |
| API Base URL | `http://localhost:8000/v1`             |
| API Key      | `dummy` (ignored, required by UI only) |
| Model Name   | `xiaomi/mimo-v2-flash:free`            |
| Streaming    | Enabled                                |

> The API key here is not used. Authentication happens via OpenRouter inside the server.

---

## 7. Supported Models

Default model (configurable in code):

```python
DEFAULT_MODEL = "xiaomi/mimo-v2-flash:free"
```

You can switch to any OpenRouter-supported model, for example:

* `openai/gpt-4o`
* `anthropic/claude-3.5-sonnet`
* `google/gemini-pro`

---

## 8. Available Tools (Auto-Invoked)

### 1. `arxiv_search`

Search arXiv papers using arXiv API endpoint.

Example user prompt:

```
Find recent papers on agentic reasoning
```

---

### 2. `arxiv_to_text`

Fetches paper content:

* Tries **LaTeX source first**
* Falls back to **PDF text extraction** if LaTeX is not found

Example user prompt:

```
Please provide content for arXiv:1412.6980
```

---

### 3. `create_beamer_presentation`

Generates a **LaTeX Beamer presentation** from an arXiv ID.

Example user prompt:

```
Create a Beamer presentation for arXiv:1412.6980
```

What happens:

1. Paper text is fetched
2. LLM generates Beamer slides
3. Output is saved locally as:

   ```
   presentation_1412.6980.tex
   ```

If the source was PDF, the title slide includes:

```
(Generated from PDF source via arxiv_to_text)
```

---

## 9. Streaming Behavior (Important for OpenWebUI)

* The server uses **Server-Sent Events (SSE)**
* Matches OpenAI streaming format
* Tool calls are:

  1. Buffered during streaming
  2. Executed after the first pass
  3. Followed by a second streamed response

This ensures:

* OpenWebUI renders partial tokens
* Tool results are correctly injected
* Final answer is coherent

---

## 10. Example End-to-End Usage (OpenWebUI)

User prompt:

```
Search arXiv for papers on Deep Learning Models and make a Beamer presentation for the best one
```

Agent flow:

1. `arxiv_search`
2. Model selects paper
3. `create_beamer_presentation`
4. `.tex` file saved locally
5. Confirmation streamed back to UI

---

## 11. Output Files

Generated files are saved in the **server working directory**:

```
presentation_<arxiv_id>.tex
```

You can compile manually:

```bash
pdflatex presentation_2305.01234.tex
```

---
