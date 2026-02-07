import os
from typing import List, Dict, Any
import json
from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
from fastapi.responses import StreamingResponse

# Import the new function
from arxiv_tool import arxiv_search, arxiv_to_text, create_beamer_presentation

load_dotenv() 

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# --- TOOL DEFINITIONS ---

ARXIV_TOOL = {
    "type": "function",
    "function": {
        "name": "arxiv_search",
        "description": "Search arXiv for research papers",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query for research papers"
                },
                "max_results": {
                    "type": "integer",
                    "default": 5
                }
            },
            "required": ["query"]
        },
    },
}

ARXIV_TO_TEXT_TOOL = {
    "type": "function",
    "function": {
        "name": "arxiv_to_text",
        "description": "Fetch a research paper from arXiv and return its text. Tries LaTeX first, otherwise PDF.",
        "parameters": {
            "type": "object",
            "properties": {
                "arxiv_id": {
                    "type": "string",
                    "description": "The arXiv paper ID, e.g. 2305.00001"
                }
            },
            "required": ["arxiv_id"]
        }
    }
}

ARXIV_BEAMER_TOOL = {
    "type": "function",
    "function": {
        "name": "create_beamer_presentation",
        "description": "Generates a .tex Beamer presentation file from an arXiv ID. Handles text extraction and LaTeX generation internally.",
        "parameters": {
            "type": "object",
            "properties": {
                "arxiv_id": {
                    "type": "string",
                    "description": "The arXiv paper ID to turn into a presentation"
                }
            },
            "required": ["arxiv_id"]
        }
    }
}

# Add the new tool to the list
AVAILABLE_TOOLS = [ARXIV_TOOL, ARXIV_TO_TEXT_TOOL, ARXIV_BEAMER_TOOL]

DEFAULT_MODEL = "openai/gpt-oss-120b:free"

SYSTEM_PROMPT_TEXT = '''
You are an expert autonomous research agent and tool-using LLM.

Your primary goal is to solve research-oriented tasks accurately, efficiently,
and using tools correctly when appropriate.

────────────────────────────────────────────────────
CORE RESPONSIBILITIES
────────────────────────────────────────────────────
- You are an expert at structured tool calling and multi-step reasoning.
- You ALWAYS prefer using provided tools over relying on prior knowledge.
- You NEVER hallucinate arXiv papers, arXiv IDs, datasets, or results.
- You optimize for correctness, research quality, and minimal tool usage.

────────────────────────────────────────────────────
AVAILABLE TOOLS AND WHEN TO USE THEM
────────────────────────────────────────────────────
1. arxiv_search
   Use when the user asks to:
   - find research papers
   - search the literature
   - discover relevant or recent work
   - identify benchmarks, datasets, or surveys

   If the user does NOT provide a specific arXiv ID, you MUST start with
   arxiv_search.

2. arxiv_to_text
   Use when the user asks to:
   - summarize a paper
   - explain methods, results, or equations
   - analyze or critique a specific paper
   - extract technical details

   Only call this tool when you already have a concrete arXiv ID.

3. create_beamer_presentation
   Use when the user asks to:
   - create slides
   - generate a presentation
   - produce a Beamer / LaTeX deck

   Once an arXiv ID is known, call this tool directly.
   Do NOT manually write LaTeX slides if this tool applies.

────────────────────────────────────────────────────
SEARCH DISCIPLINE (CRITICAL)
────────────────────────────────────────────────────
- Do NOT call arxiv_search more than 3 times per user request.
- If results are insufficient after retries, respond with:
  "Relevant benchmarking papers are limited on arXiv. Here are the closest matches."
- Do NOT endlessly refine queries.
- Prefer partial but relevant results over repeated searching.

────────────────────────────────────────────────────
CRITICAL REASONING PRIVACY POLICY
────────────────────────────────────────────────────
- NEVER reveal chain-of-thought, planning, retries, or internal reasoning.
- NEVER narrate failed searches, query refinement, or decision-making.
- All reasoning and retries MUST be done silently.
- The user should only see:
  - tool calls
  - final answers
  - concise summaries of results

If a tool fails or returns weak results:
- Retry silently (within limits)
- Present the best available output without explanation of the retry process

────────────────────────────────────────────────────
RESEARCH & DOMAIN EXPERTISE
────────────────────────────────────────────────────
You are an expert in:
- Machine Learning and Deep Learning
- Large Language Models (LLMs)
- Physiological signal processing (PPG, EDA, EEG, ECG)
- Wearable sensor datasets (e.g., WESAD, DEAP, PhysioNet)
- Benchmarking and evaluation methodologies
- Optimization, statistics, and linear algebra
- Academic paper analysis and synthesis
- Research-quality slide and presentation design

────────────────────────────────────────────────────
OUTPUT STYLE
────────────────────────────────────────────────────
- Be concise, precise, and well-structured.
- Use bullet points and clear sections when appropriate.
- Prefer factual, source-backed claims.
- If uncertainty exists, state it clearly and propose the next correct action.
- STOP once the task is complete unless the user asks to continue.

────────────────────────────────────────────────────
SAFETY & RELIABILITY
────────────────────────────────────────────────────
- Never fabricate citations, arXiv IDs, datasets, or results.
- Ask a clarifying question ONLY if the request is genuinely ambiguous.
- If a task cannot be completed with available tools, explain why succinctly.

You are operating inside an automated multi-step agent loop.
Behave deterministically, silently reason, and produce research-grade outputs.

'''


def call_tool(name: str, arguments):
    import json
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            # Fallback if the model sends malformed json (rare but happens)
            return f"Error decoding arguments for tool {name}"

    try:
        if name == "arxiv_search":
            print(f"Tool Call: Searching {arguments.get('query')}")
            return arxiv_search(**arguments)
        if name == "arxiv_to_text":
            print(f"Tool Call: Fetching text for {arguments.get('arxiv_id')}")
            # We return a truncated preview to the chat context to save tokens, 
            # unless the specific full text is requested.
            result = arxiv_to_text(**arguments)
            # Preview only for chat context
            return {
                "arxiv_id": result["arxiv_id"],
                "source_used": result["source_used"],
                "text_preview": result["full_text"][:2000] + "... [Text Truncated for Chat Context]"
            }
        if name == "create_beamer_presentation":
            print(f"Tool Call: Generating Beamer for {arguments.get('arxiv_id')}")
            return create_beamer_presentation(**arguments)
            
    except Exception as e:
        return f"Tool execution error: {str(e)}"

    raise ValueError(f"Unknown tool: {name}")

# ---------------- CONFIG ----------------

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

# ---------------- SCHEMA ----------------
class ChatRequest(BaseModel):
    messages: List[Dict[str, Any]]
    model: str | None = None
    temperature: float | None = 0.7
    stream: bool = True

app = FastAPI(title="LLM Agent Proxy")

# ---------------- MULTI-STEP STREAMING LOGIC ----------------

def stream_generator(req: ChatRequest):
    SYSTEM_PROMPT = {
        "role": "system",
        "content": SYSTEM_PROMPT_TEXT,
    }
    model = req.model or DEFAULT_MODEL
    current_messages = req.messages
    if not current_messages or current_messages[0]["role"] != "system":
        current_messages = [SYSTEM_PROMPT] + current_messages
    
    # Safety limit to prevent infinite loops (e.g., model keeps searching forever)
    MAX_TURNS = 10 
    turn_count = 0

    while turn_count < MAX_TURNS:
        turn_count += 1
        print(f"--- Turn {turn_count} ---")

        # 1. Request Stream
        stream = client.chat.completions.create(
            model=model,
            messages=current_messages,
            temperature=req.temperature,
            tools=AVAILABLE_TOOLS,
            tool_choice="auto",
            stream=True, 
        )

        tool_calls_buffer = []
        
        # 2. Process Stream Chunks
        for chunk in stream:
            delta = chunk.choices[0].delta
            
            # A. Buffer Tool Calls (don't yield to user yet)
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    if tc.index >= len(tool_calls_buffer):
                        tool_calls_buffer.append({
                            "id": tc.id,
                            "function": {"name": tc.function.name, "arguments": ""},
                            "type": "function"
                        })
                    if tc.function.arguments:
                        tool_calls_buffer[tc.index]["function"]["arguments"] += tc.function.arguments
                    if tc.function.name:
                        tool_calls_buffer[tc.index]["function"]["name"] = tc.function.name

            # B. Yield Content (Text Response) to User
            elif delta.content is not None:
                yield f"data: {json.dumps(chunk.model_dump(mode='json'))}\n\n"

        # 3. Decision Point: Did the model ask for tools?
        if not tool_calls_buffer:
            print("No tools called. Conversation finished.")
            yield "data: [DONE]\n\n"
            break # Exit the while loop
        
        # 4. Execute Tools (Backstage)
        print(f"Executing {len(tool_calls_buffer)} tool(s)...")
        
        # Add the Assistant's "Tool Request" message to history
        current_messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": tool_calls_buffer
        })

        # Run every tool requested and add results to history
        for tc in tool_calls_buffer:
            func_name = tc["function"]["name"]
            func_args = tc["function"]["arguments"]
            
            tool_result = call_tool(func_name, func_args)
            
            current_messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "name": func_name,
                "content": str(tool_result),
            })
        
        # LOOP CONTINUES: The 'while' loop restarts, sending updated history to LLM

# ---------------- ENDPOINT ----------------

@app.post("/v1/chat/completions")
async def chat(req: ChatRequest):
    SYSTEM_PROMPT = {
        "role": "system",
        "content": SYSTEM_PROMPT_TEXT,
    }
    if req.stream:
        return StreamingResponse(
            stream_generator(req),
            media_type="text/event-stream"
        )

    # ---------------- MULTI-STEP NON-STREAMING LOGIC ----------------
    current_messages = req.messages
    if not current_messages or current_messages[0]["role"] != "system":
        current_messages = [SYSTEM_PROMPT] + current_messages

    MAX_TURNS = 10
    turn_count = 0

    while turn_count < MAX_TURNS:
        turn_count += 1
        
        response = client.chat.completions.create(
            model=req.model or DEFAULT_MODEL,
            messages=current_messages,
            temperature=req.temperature,
            tools=AVAILABLE_TOOLS,
            tool_choice="auto",
        )
        
        message = response.choices[0].message

        # If no tools called, we are done. Return response.
        if not message.tool_calls:
            return response

        # If tools called, execute them
        print(f"Turn {turn_count}: Executing tools...")
        current_messages.append(message) # Append assistant's request

        for tool_call in message.tool_calls:
            result = call_tool(
                tool_call.function.name,
                tool_call.function.arguments,
            )
            current_messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": str(result),
            })
        
        # Loop restarts to send results back to model
        
    return response

AVAILABLE_MODELS = [
    {
        "id": DEFAULT_MODEL,
        "object": "model",
        "created": 0,
        "owned_by": "openrouter",
    }
]

@app.get("/v1/models")
async def list_models():
    """
    OpenAI-compatible models endpoint (required by OpenWebUI)
    """
    return {
        "object": "list",
        "data": AVAILABLE_MODELS,
    }