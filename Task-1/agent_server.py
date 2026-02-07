import os
from typing import List, Dict, Any
import json
from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
from fastapi.responses import StreamingResponse
import requests

# Import the new functions
from arxiv_tool import (
    arxiv_search, 
    arxiv_to_text, 
    create_beamer_presentation,
    verify_beamer_presentation,
    compile_beamer_to_pdf
)

load_dotenv() 

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

DEFAULT_MODEL = "openai/gpt-oss-120b"


def get_context_length(model_id: str) -> int | None:
    resp = requests.get(
        "https://openrouter.ai/api/v1/models",
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        },
        timeout=30,
    )
    resp.raise_for_status()

    for model in resp.json()["data"]:
        if model["id"] == model_id:
            return model.get("context_length")

    return None

ctx_len = get_context_length(DEFAULT_MODEL)
print(f"{DEFAULT_MODEL} context length = {ctx_len}")

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
        "description": "Fetch a research paper from arXiv and return its full text. Tries LaTeX first, otherwise PDF.",
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
                },
                "model": {
                    "type": "string",
                    "description": "The LLM model to use for generation"
                },
                "verification_feedback": {
                    "type": "object",
                    "description": "Optional verification feedback for improvement"
                },
                "previous_tex": {
                    "type": "string",
                    "description": "Previous presentation LaTeX code to improve upon"
                }
            },
            "required": ["arxiv_id"]
        }
    }
}

VERIFY_BEAMER_TOOL = {
    "type": "function",
    "function": {
        "name": "verify_beamer_presentation",
        "description": "Verify the quality and accuracy of a generated Beamer presentation against the original paper. Pass the full paper text and LaTeX content.",
        "parameters": {
            "type": "object",
            "properties": {
                "arxiv_id": {
                    "type": "string",
                    "description": "The arXiv paper ID"
                },
                "tex_content": {
                    "type": "string",
                    "description": "The generated LaTeX content (read from the .tex file)"
                },
                "paper_text": {
                    "type": "string",
                    "description": "The full original paper text"
                },
                "model": {
                    "type": "string",
                    "description": "The LLM model to use for verification"
                }
            },
            "required": ["arxiv_id", "tex_content", "paper_text"]
        }
    }
}

COMPILE_PDF_TOOL = {
    "type": "function",
    "function": {
        "name": "compile_beamer_to_pdf",
        "description": "Compile a LaTeX Beamer presentation to PDF. Requires pdflatex to be installed.",
        "parameters": {
            "type": "object",
            "properties": {
                "arxiv_id": {
                    "type": "string",
                    "description": "The arXiv paper ID (used to locate the .tex file)"
                }
            },
            "required": ["arxiv_id"]
        }
    }
}

READ_FILE_TOOL = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Read the content of a file. Use this to read generated presentation files or other text files.",
        "parameters": {
            "type": "object",
            "properties": {
                "filepath": {
                    "type": "string",
                    "description": "Path to the file to read (e.g., 'output/presentations_tex/presentation_2201.12150.tex')"
                }
            },
            "required": ["filepath"]
        }
    }
}

# Add all tools to the list
AVAILABLE_TOOLS = [
    ARXIV_TOOL, 
    ARXIV_TO_TEXT_TOOL, 
    ARXIV_BEAMER_TOOL,
    VERIFY_BEAMER_TOOL,
    COMPILE_PDF_TOOL,
    READ_FILE_TOOL
]


# Verification threshold and retry tracker
VERIFICATION_THRESHOLD = 7  # Minimum average score out of 10 for auto-acceptance
presentation_retry_tracker = {}  # Track retries per arxiv_id

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
   Returns the FULL text of the paper.

3. create_beamer_presentation
   Use when the user asks to:
   - create slides
   - generate a presentation
   - produce a Beamer / LaTeX deck

   Once an arXiv ID is known, call this tool directly.
   Do NOT manually write LaTeX slides if this tool applies.

4. read_file
   Use when you need to:
   - Read the content of a generated presentation file
   - Access any text file on disk
   - Get the LaTeX content for verification

   Required for verification workflow to read the generated .tex file.

5. verify_beamer_presentation
   Use AUTOMATICALLY after creating a presentation to:
   - verify accuracy against the original paper
   - check for completeness and quality
   - identify potential issues

   Always verify presentations before asking about PDF compilation.
   Requires: arxiv_id, tex_content (from read_file), and paper_text (from arxiv_to_text)

6. compile_beamer_to_pdf
   Use when:
   - A presentation has been created and verified
   - The user confirms they want a PDF
   - You need to offer PDF compilation as next step

   IMPORTANT: Always ASK the user if they want to compile to PDF.
   Do NOT compile automatically without permission.

────────────────────────────────────────────────────
SEARCH DISCIPLINE (CRITICAL)
────────────────────────────────────────────────────
- Do NOT call arxiv_search more than 3 times per user request.
- If results are insufficient after 3 searches:
  1. STOP searching immediately
  2. Present the best available results you found
  3. Explain: "I found [N] papers matching your criteria. These are the most relevant results available on arXiv."
  4. Summarize what you DID find
  5. Offer to help with what you found OR adjust the search query

- NEVER say "I couldn't find anything" without presenting what you DID find
- NEVER end the conversation after failed searches - always provide value
- Prefer partial but relevant results over repeated searching

────────────────────────────────────────────────────
PRESENTATION WORKFLOW
────────────────────────────────────────────────────
When creating presentations, follow this sequence:
1. Search for or identify the paper
2. Fetch the paper text (arxiv_to_text) - returns FULL text
3. Generate presentation (create_beamer_presentation) - creates .tex file
4. Read the generated .tex file (read_file with path from step 3)
5. Verify presentation (verify_beamer_presentation with tex_content and paper_text)
6. Present verification results to user
7. ASK user: "Would you like me to compile this to PDF?"
8. Only if user agrees: compile_beamer_to_pdf

All generated files are organized in folders:
- output/presentations_tex/: LaTeX source files
- output/presentations_pdf/: Compiled PDF files  
- output/papers/: Downloaded research papers

────────────────────────────────────────────────────
AUTOMATIC PRESENTATION IMPROVEMENT
────────────────────────────────────────────────────
When a presentation is verified with an average score below 7/10:
1. The system AUTOMATICALLY regenerates the presentation with feedback
2. This happens ONCE without user intervention
3. The previous presentation is provided to the model for comparison
4. The improved version is re-verified
5. User receives BOTH original and improved verification results
6. Present results as: "Initial version scored X/10, improved version scores Y/10"

If the improved version still scores low, accept it and inform the user.
Do NOT attempt further improvements without explicit user request.

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
- Focus on what WAS accomplished, not what failed

────────────────────────────────────────────────────
HANDLING MAXIMUM INTERACTION DEPTH
────────────────────────────────────────────────────
If you reach the maximum number of tool interactions:
1. DO NOT ask the user to start a new conversation
2. Analyze what was accomplished in the current session
3. Provide a COHERENT SUMMARY of:
   - What tasks were completed successfully
   - What information was gathered
   - What files were created (with paths)
   - What remains to be done (if anything)
4. Offer clear next steps based on the current state
5. Make your response useful and actionable

Example: "I've successfully completed the following:
- Found 5 relevant papers on [topic]
- Generated a presentation for paper [ID] (saved to output/presentations_tex/)
- Verified the presentation (scored 8/10 for accuracy)

The presentation is ready. Would you like me to compile it to PDF in your next message?"

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
- Always provide value even if searches return limited results.
- NEVER end abruptly - always offer next steps or alternatives.

────────────────────────────────────────────────────
SAFETY & RELIABILITY
────────────────────────────────────────────────────
- Never fabricate citations, arXiv IDs, datasets, or results.
- Ask a clarifying question ONLY if the request is genuinely ambiguous.
- If a task cannot be completed with available tools, explain why succinctly.
- Always deliver actionable output based on what information IS available.

You are operating inside an automated multi-step agent loop.
Behave deterministically, silently reason, and produce research-grade outputs.

'''


def call_tool(name: str, arguments, model: str = None):
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
            arxiv_id = arguments.get('arxiv_id')
            print(f"Tool Call: Fetching text for {arxiv_id}")
            result = arxiv_to_text(arxiv_id)
            
            # Return the FULL TEXT, not truncated
            return {
                "arxiv_id": result["arxiv_id"],
                "source_used": result["source_used"],
                "full_text": result["full_text"],
                "text_length": len(result["full_text"])
            }
        
        if name == "read_file":
            filepath = arguments.get('filepath')
            print(f"Tool Call: Reading file {filepath}")
            
            from pathlib import Path
            file_path = Path(filepath)
            
            if not file_path.exists():
                return {
                    "status": "error",
                    "message": f"File not found: {filepath}"
                }
            
            try:
                content = file_path.read_text(encoding='utf-8')
                return {
                    "status": "success",
                    "filepath": str(filepath),
                    "content": content,
                    "content_length": len(content)
                }
            except Exception as e:
                return {
                    "status": "error",
                    "message": f"Error reading file: {str(e)}"
                }
        
        if name == "create_beamer_presentation":
            arxiv_id = arguments.get('arxiv_id')
            print(f"Tool Call: Generating Beamer for {arxiv_id}")
            
            # Pass the model, feedback, AND previous tex to the function
            result = create_beamer_presentation(
                arxiv_id=arxiv_id,
                model=model or DEFAULT_MODEL,
                verification_feedback=arguments.get('verification_feedback'),
                previous_tex=arguments.get('previous_tex')
            )
            
            # Mark if this was a retry
            if arguments.get('verification_feedback'):
                result['is_retry'] = True
            
            return result
        
        if name == "verify_beamer_presentation":
            arxiv_id = arguments.get('arxiv_id')
            print(f"Tool Call: Verifying presentation for {arxiv_id}")
            
            # Get tex_content and paper_text from arguments
            tex_content = arguments.get('tex_content')
            paper_text = arguments.get('paper_text')
            
            if not tex_content:
                return {
                    "status": "error",
                    "message": f"No tex_content provided. Use read_file to get the presentation content first."
                }
            
            if not paper_text:
                return {
                    "status": "error",
                    "message": f"No paper_text provided. Use arxiv_to_text to get the paper content first."
                }
            
            # TRUNCATE if needed to prevent context overflow
            paper_text_truncated, tex_content_truncated = truncate_for_verification(paper_text, tex_content)
            
            if len(paper_text) != len(paper_text_truncated):
                print(f"WARNING: Paper text truncated from {len(paper_text)} to {len(paper_text_truncated)} chars")
            
            # Pass (potentially truncated) data to verification
            result = verify_beamer_presentation(
                arxiv_id=arxiv_id,
                tex_content=tex_content_truncated,
                paper_text=paper_text_truncated,
                model=model or DEFAULT_MODEL
            )
    
            
            # AUTO-IMPROVEMENT LOGIC
            # Check if scores are below threshold and we haven't retried yet
            if arxiv_id not in presentation_retry_tracker:
                accuracy = result.get('accuracy_score', 10)
                completeness = result.get('completeness_score', 10)
                clarity = result.get('clarity_score', 10)
                avg_score = (accuracy + completeness + clarity) / 3
                
                if avg_score < VERIFICATION_THRESHOLD:
                    print(f"Verification score {avg_score:.1f}/10 below threshold. Auto-improving...")
                    presentation_retry_tracker[arxiv_id] = True
                    
                    # Use the current tex_content as previous
                    previous_tex_content = tex_content
                    
                    # Regenerate with feedback AND previous presentation
                    improved_result = create_beamer_presentation(
                        arxiv_id=arxiv_id,
                        model=model or DEFAULT_MODEL,
                        verification_feedback=result,
                        previous_tex=previous_tex_content
                    )
                    
                    # Read the improved version
                    from pathlib import Path
                    tex_file_new = Path("output/presentations_tex") / f"presentation_{arxiv_id}.tex"
                    if tex_file_new.exists():
                        improved_tex = tex_file_new.read_text(encoding='utf-8')
                        
                        # Re-verify the improved version
                        new_verification = verify_beamer_presentation(
                            arxiv_id=arxiv_id,
                            tex_content=improved_tex,
                            paper_text=paper_text,
                            model=model or DEFAULT_MODEL
                        )
                        
                        # Calculate improvements
                        new_accuracy = new_verification.get('accuracy_score', 0)
                        new_completeness = new_verification.get('completeness_score', 0)
                        new_clarity = new_verification.get('clarity_score', 0)
                        new_avg = (new_accuracy + new_completeness + new_clarity) / 3
                        
                        # Return both original and improved results
                        return {
                            "original_verification": result,
                            "auto_improved": True,
                            "improved_verification": new_verification,
                            "message": f"Initial presentation scored {avg_score:.1f}/10. Auto-generated improved version which scored {new_avg:.1f}/10.",
                            "improvement_delta": {
                                "accuracy": new_accuracy - accuracy,
                                "completeness": new_completeness - completeness,
                                "clarity": new_clarity - clarity,
                                "average": new_avg - avg_score
                            }
                        }
            
            return result
        
        if name == "compile_beamer_to_pdf":
            print(f"Tool Call: Compiling PDF for {arguments.get('arxiv_id')}")
            result = compile_beamer_to_pdf(**arguments)
            return result
            
    except Exception as e:
        return {
            "status": "error",
            "message": f"Tool execution error: {str(e)}"
        }

    raise ValueError(f"Unknown tool: {name}")

# ---------------- CONFIG ----------------

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

# ---------------- SCHEMA ----------------
class ChatRequest(BaseModel):
    session_id: str | None = None
    messages: List[Dict[str, Any]]
    model: str | None = None
    temperature: float | None = 0.7
    stream: bool = True

app = FastAPI(title="LLM Agent Proxy")

# ---------------- MULTI-STEP STREAMING LOGIC ----------------

def truncate_for_verification(paper_text: str, tex_content: str, max_total_chars: int = ctx_len) -> tuple:
    """
    Intelligently truncate paper text and tex content to fit within context limits.
    Prioritizes keeping the tex_content complete and truncates paper_text if needed.
    """
    tex_len = len(tex_content)
    paper_len = len(paper_text)
    
    # If both fit comfortably, return as is
    if tex_len + paper_len <= max_total_chars:
        return paper_text, tex_content
    
    # Keep tex_content complete (presentations are typically <20k chars)
    # Truncate paper_text to fit
    available_for_paper = max_total_chars - tex_len - 1000  # 1000 char buffer
    
    if available_for_paper < 10000:
        # If we can't keep enough of the paper, truncate both
        paper_truncated = paper_text[:40000] + "\n\n...[Paper truncated due to length constraints]"
        tex_truncated = tex_content[:30000] + "\n\n...[Presentation truncated due to length constraints]"
        return paper_truncated, tex_truncated
    
    # Truncate paper intelligently - keep beginning and end
    if paper_len > available_for_paper:
        keep_chars = available_for_paper // 2
        paper_truncated = (
            paper_text[:keep_chars] + 
            "\n\n...[MIDDLE SECTION TRUNCATED TO FIT CONTEXT]...\n\n" + 
            paper_text[-keep_chars:]
        )
        return paper_truncated, tex_content
    
    return paper_text, tex_content


def stream_generator(req: ChatRequest):
    SYSTEM_PROMPT = {
        "role": "system",
        "content": SYSTEM_PROMPT_TEXT,
    }
    model = req.model or DEFAULT_MODEL
    current_messages = req.messages
    if not current_messages or current_messages[0]["role"] != "system":
        current_messages = [SYSTEM_PROMPT] + current_messages
    
    # Safety limit to prevent infinite loops
    MAX_TURNS = 15
    turn_count = 0

    while turn_count < MAX_TURNS:
        turn_count += 1
        print(f"--- Turn {turn_count} ---")

        try:
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
            content_buffer = []
            
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
                    content_buffer.append(delta.content)
                    yield f"data: {json.dumps(chunk.model_dump(mode='json'))}\n\n"

            # 3. Decision Point: Did the model ask for tools?
            if not tool_calls_buffer:
                print("No tools called. Conversation finished.")
                if content_buffer:
                    print(f"Model provided response: {len(''.join(content_buffer))} chars")
                yield "data: [DONE]\n\n"
                break

        except Exception as e:
            print(f"ERROR in stream: {str(e)}")
            # If streaming fails, try to recover by truncating the last message
            if "list index out of range" in str(e) or "context" in str(e).lower():
                # Context too large - truncate tool results
                print("Context too large, truncating last tool result...")
                if current_messages and current_messages[-1].get("role") == "tool":
                    content = current_messages[-1].get("content", "")
                    if len(content) > 50000:  # If content is very large
                        try:
                            content_dict = json.loads(content)
                            # Truncate full_text or content fields
                            if "full_text" in content_dict:
                                content_dict["full_text"] = content_dict["full_text"][:30000] + "\n\n...[TRUNCATED DUE TO LENGTH]"
                            if "content" in content_dict:
                                content_dict["content"] = content_dict["content"][:30000] + "\n\n...[TRUNCATED DUE TO LENGTH]"
                            current_messages[-1]["content"] = json.dumps(content_dict)
                            print("Truncated and retrying...")
                            continue  # Retry with truncated content
                        except:
                            pass
            
            # Send error message to user
            error_chunk = {
                "choices": [{
                    "delta": {
                        "content": f"\n\n[Error: {str(e)}. The content may be too large. Try with a shorter paper or contact support.]"
                    }
                }]
            }
            yield f"data: {json.dumps(error_chunk)}\n\n"
            yield "data: [DONE]\n\n"
            break
        
        # 4. Execute Tools (Backstage)
        print(f"Executing {len(tool_calls_buffer)} tool(s)...")
        
        # Add the Assistant's "Tool Request" message to history
        current_messages.append({
            "role": "assistant",
            "content": ''.join(content_buffer) if content_buffer else None,
            "tool_calls": tool_calls_buffer
        })

        # Run every tool requested and add results to history
        for tc in tool_calls_buffer:
            func_name = tc["function"]["name"]
            func_args = tc["function"]["arguments"]
            
            tool_result = call_tool(func_name, func_args, model=model)
            
            # Truncate large results to prevent context overflow
            result_str = json.dumps(tool_result) if isinstance(tool_result, dict) else str(tool_result)
            
            # If result is too large (>100k chars), truncate intelligently
            if len(result_str) > 100000:
                print(f"WARNING: Tool result is {len(result_str)} chars, truncating...")
                if isinstance(tool_result, dict):
                    if "full_text" in tool_result:
                        tool_result["full_text"] = tool_result["full_text"][:30000] + "\n\n...[TRUNCATED - Text too long for context]"
                        tool_result["was_truncated"] = True
                    if "content" in tool_result:
                        tool_result["content"] = tool_result["content"][:30000] + "\n\n...[TRUNCATED - Content too long for context]"
                        tool_result["was_truncated"] = True
                    result_str = json.dumps(tool_result)
                else:
                    result_str = str(tool_result)[:50000] + "\n...[TRUNCATED]"
            
            current_messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "name": func_name,
                "content": result_str,
            })
        
        # LOOP CONTINUES

    # If we hit MAX_TURNS, force a coherent final summary
    if turn_count >= MAX_TURNS:
        print(f"WARNING: Hit MAX_TURNS ({MAX_TURNS}), generating coherent summary")
        
        summary_prompt = {
            "role": "user",
            "content": "You've reached the maximum interaction depth. Please provide a coherent summary of what you've accomplished in this conversation, including any files created, data gathered, and next steps the user should take. Be specific about file paths and completed tasks."
        }
        current_messages.append(summary_prompt)
        
        try:
            final_stream = client.chat.completions.create(
                model=model,
                messages=current_messages,
                temperature=0.3,
                stream=True,
            )
            
            for chunk in final_stream:
                delta = chunk.choices[0].delta
                if delta.content is not None:
                    yield f"data: {json.dumps(chunk.model_dump(mode='json'))}\n\n"
        except Exception as e:
            print(f"ERROR in final summary: {str(e)}")
            error_chunk = {
                "choices": [{
                    "delta": {
                        "content": "\n\nTask completed. Check the output/ folder for generated files."
                    }
                }]
            }
            yield f"data: {json.dumps(error_chunk)}\n\n"
        
        yield "data: [DONE]\n\n"
    

# ---------------- ENDPOINT ----------------

SESSION_STATE = {}

@app.post("/v1/chat/completions")
async def chat(req: ChatRequest):
    # Use a hash of the first user message or generate a UUID
    # if req.session_id:
    #     sid = req.session_id
    # else:
    #     # Generate from first user message for consistency across retries
    #     first_msg = next((m for m in req.messages if m.get("role") == "user"), None)
    #     if first_msg:
    #         import hashlib
    #         sid = hashlib.md5(first_msg.get("content", "")[:100].encode()).hexdigest()
    #     else:
    #         import uuid
    #         sid = str(uuid.uuid4())
    # print("SESSION ID:",sid)
    # if sid not in SESSION_STATE:
    #     SESSION_STATE[sid] = {
    #         "messages": [],
    #         "completed": False,
    #     }

    # session = SESSION_STATE[sid]

    # # If this session already finished, don't rerun
    # if session["completed"]:
    #     print(f"Session {sid} already completed. Ignoring duplicate request.")
    #     return {"choices": []}
    
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

    MAX_TURNS = 15
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
                model=req.model or DEFAULT_MODEL
            )
            current_messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result) if isinstance(result, dict) else str(result),
            })
        
        # Loop restarts to send results back to model
    
    # If we hit MAX_TURNS in non-streaming mode, generate a coherent summary
    if turn_count >= MAX_TURNS:
        print(f"WARNING: Hit MAX_TURNS ({MAX_TURNS}) in non-streaming mode, generating summary")
        
        summary_prompt = {
            "role": "user",
            "content": "You've reached the maximum interaction depth. Please provide a coherent summary of what you've accomplished in this conversation, including any files created, data gathered, and next steps the user should take. Be specific about file paths and completed tasks."
        }
        current_messages.append(summary_prompt)
        
        response = client.chat.completions.create(
            model=req.model or DEFAULT_MODEL,
            messages=current_messages,
            temperature=0.3,
        )
        
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