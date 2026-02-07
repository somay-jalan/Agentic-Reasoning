import os
import requests
import feedparser
from typing import List, Dict, Any, Optional
import tarfile
import tempfile
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
import subprocess
import shutil

load_dotenv()

# We need a client inside the tool to generate the specific content
# independent of the main agent's conversation flow.
tool_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    default_headers={
        "HTTP-Referer": "https://somay-research-tool",
        "X-Title": "Arxiv Beamer Tool",
    },
)

# Create organized folder structure
FOLDERS = {
    "presentations_tex": Path("output/presentations_tex"),
    "presentations_pdf": Path("output/presentations_pdf"),
    "papers": Path("output/papers"),
}

for folder in FOLDERS.values():
    folder.mkdir(parents=True, exist_ok=True)

def canonical_id(arxiv_id: str) -> str:
    return arxiv_id.split("v")[0]

def arxiv_search(
    query: str,
    max_results: int = 5,
    sort_by: str = "relevance",
) -> List[Dict]:
    """
    Fetch papers from arXiv.
    """
    base_url = "http://export.arxiv.org/api/query"
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results,
        "sortBy": sort_by,
        "sortOrder": "descending",
    }

    response = requests.get(base_url, params=params, timeout=10)
    response.raise_for_status()

    feed = feedparser.parse(response.text)

    papers = []
    for entry in feed.entries:
        # Extract arXiv ID from the link
        arxiv_id = entry.id.split('/abs/')[-1]
        papers.append({
            "title": entry.title,
            "authors": [a.name for a in entry.authors],
            "summary": entry.summary,
            "published": entry.published,
            "link": entry.link,
            "arxiv_id": arxiv_id,
        })

    return papers


def download_arxiv_tex(arxiv_id: str) -> str:
    """
    Downloads the arXiv source .tar.gz and extracts .tex file.
    Returns the content of the main .tex file as string.
    """
    url = f"https://arxiv.org/e-print/{arxiv_id}"
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    arxiv_id = canonical_id(arxiv_id)
    # Save source to papers folder
    source_path = FOLDERS["papers"] / f"{arxiv_id}_source.tar.gz"
    source_path.write_bytes(r.content)

    with tempfile.TemporaryDirectory() as tmpdir:
        tar_path = Path(tmpdir) / "source.tar.gz"
        tar_path.write_bytes(r.content)

        try:
            with tarfile.open(tar_path) as tar:
                # Look for .tex files
                tex_files = [m for m in tar.getmembers() if m.name.endswith(".tex")]
                
                if not tex_files:
                    raise ValueError("No .tex file found in arXiv source")

                # Heuristic: The largest .tex file is usually the main one
                # or look for specific names like main.tex, mns.tex
                main_file = max(tex_files, key=lambda m: m.size)
                
                tar.extract(main_file, path=tmpdir)
                tex_path = Path(tmpdir) / main_file.name
                return tex_path.read_text(errors='replace')
        except tarfile.ReadError:
             raise ValueError("File is not a valid tar archive (likely PDF only)")


def arxiv_to_text(arxiv_id: str) -> dict:
    """
    Fetch a research paper from arXiv and return its text.
    
    Tries LaTeX source first, otherwise falls back to PDF text.
    """
    used_source = None
    text_content = ""
    try:
        text_content = download_arxiv_tex(arxiv_id)
        used_source = "tex"
    except Exception:
        # fallback to PDF
        print(f"DEBUG: Failed to get TeX for {arxiv_id}, trying PDF...")
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        r = requests.get(pdf_url, timeout=10)
        r.raise_for_status()
        
        # Save PDF to papers folder
        pdf_path = FOLDERS["papers"] / f"{arxiv_id}.pdf"
        pdf_path.write_bytes(r.content)
        
        # Save momentarily to read
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(r.content)
            tmp_path = tmp.name

        try:
            import fitz  # PyMuPDF
            doc = fitz.open(tmp_path)
            text_content = "\n".join([page.get_text() for page in doc])
            doc.close()
        except Exception as e:
            text_content = f"[ERROR extracting text from PDF: {e}]"
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        used_source = "pdf"

    return {
        "arxiv_id": arxiv_id,
        "source_used": used_source,
        "full_text": text_content,
    }


BEAMER_SYSTEM_PROMPT = """You are an expert academic presentation designer and LaTeX specialist.

Your mission is to create compelling, well-structured Beamer presentations that:
1. Clearly communicate complex research to academic audiences
2. Follow best practices in slide design and information hierarchy
3. Produce valid, compilable LaTeX code

CORE PRINCIPLES:
- Clarity over completeness: Slides should highlight key insights, not reproduce the entire paper
- Visual hierarchy: Use frames, blocks, and emphasis strategically
- Technical accuracy: Preserve mathematical notation, terminology, and citations
- Compilation readiness: Escape all special characters properly (%, _, &, #, $, {, })

PRESENTATION STRUCTURE:
1. Title slide with paper metadata
2. Introduction/Motivation (2-3 slides)
3. Problem Statement & Contributions (1-2 slides)
4. Methodology (3-5 slides, depending on complexity)
5. Results & Analysis (2-4 slides)
6. Conclusion & Future Work (1-2 slides)

STYLE GUIDELINES:
- Use bullet points for clarity (3-5 per slide maximum)
- Include mathematical equations where central to the work
- Convert figures/tables to textual descriptions
- Use \\alert{} for emphasis sparingly
- Add slide transitions and pauses only if beneficial

OUTPUT REQUIREMENTS:
- Complete, valid LaTeX code only
- Use \\documentclass{beamer}
- Include proper preamble with packages
- Ensure all content is properly escaped
- No placeholder text or TODOs
"""


def create_beamer_presentation(
    arxiv_id: str, 
    model: str, 
    verification_feedback: Optional[dict] = None, 
    previous_tex: Optional[str] = None
) -> dict:
    """
    Generates a LaTeX Beamer presentation for the given arXiv ID.
    Can optionally accept verification feedback and previous presentation to improve.
    """
    
    # 1. Get the content
    print(f"Fetching content for {arxiv_id}...")
    data = arxiv_to_text(arxiv_id)
    arxiv_id = canonical_id(arxiv_id)
    
    source = data['source_used']
    text = data['full_text']

    # Check if this is a regeneration based on feedback
    if verification_feedback and previous_tex:
        print(f"Regenerating presentation based on verification feedback...")
        feedback_section = f"""

**THIS IS A REVISION - IMPROVE THE PREVIOUS PRESENTATION:**

**Previous Presentation LaTeX Code:**
```latex
{previous_tex}
**Verification Results:**
- Overall Quality: {verification_feedback.get('overall_quality', 'N/A')}
- Accuracy Score: {verification_feedback.get('accuracy_score', 'N/A')}/10
- Completeness Score: {verification_feedback.get('completeness_score', 'N/A')}/10
- Clarity Score: {verification_feedback.get('clarity_score', 'N/A')}/10

**Issues Found:**
{chr(10).join('- ' + issue for issue in verification_feedback.get('issues', []))}

**Specific Recommendations:**
{chr(10).join('- ' + rec for rec in verification_feedback.get('recommendations', []))}

**Your Task:**
Review the previous presentation above and the verification feedback. 
Generate an IMPROVED version that addresses all the issues while keeping what worked well.
Focus on fixing the specific problems identified.
"""
    else:
        feedback_section = ""

    print(f"Generating Slides via LLM (Source: {source})...")

    # 2. Construct Prompt
    user_prompt = f"""
Create a detailed LaTeX Beamer presentation for the following paper.

**Instructions:**
1. Use the `beamer` document class.
2. Include sections for: Introduction, Methodology, Experiments/Results, and Conclusion.
3. Ensure the LaTeX code compiles (escape special characters like %, _, &, #, $ properly).
4. Use `\\usetheme{{metropolis}}` or `\\usetheme{{Madrid}}`.
5. DO NOT INCLUDE IMAGES IN THE PRESENTATION.

If the paper contains figures, tables, or plots:
- Convert them into bullet-point textual descriptions.
- Do NOT reference filenames, paths, or figure numbers.

**CRITICAL REQUIREMENT:**
The source content for this generation came from: **{source.upper()}**.

If the source was **PDF**, you MUST include a visible footnote or note on the Title Slide that says:
"\\footnotesize{{(Generated from PDF source via arxiv_to_text)}}"

If the source was **TEX**, do not add that disclaimer.
{feedback_section}

**Paper Content:**
{text}

Return ONLY the raw LaTeX code inside a markdown code block (```latex ... ```).
"""

    # 3. Call LLM to generate LaTeX
    response = tool_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": BEAMER_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.2
    )
    
    raw_response = response.choices[0].message.content
    
    # 4. Extract Code block
    latex_code = raw_response
    if "```latex" in raw_response:
        latex_code = raw_response.split("```latex")[1].split("```")[0].strip()
    elif "```" in raw_response:
        latex_code = raw_response.split("```")[1].split("```")[0].strip()

    # 5. Save to file in organized folder
    filename = FOLDERS["presentations_tex"] / f"presentation_{arxiv_id}.tex"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(latex_code)

    return {
        "status": "success",
        "message": f"Presentation generated successfully based on {source} source.",
        "tex_file": str(filename),
        "arxiv_id": arxiv_id,
    }


CHECKER_SYSTEM_PROMPT = """You are an expert peer reviewer for academic presentations.

Your task is to verify that a LaTeX Beamer presentation accurately represents the research paper it's based on.

EVALUATION CRITERIA:
1. **Accuracy**: Does the presentation correctly represent the paper's contributions, methods, and results?
2. **Completeness**: Are key concepts, equations, and findings included?
3. **Clarity**: Is the content well-organized and understandable?
4. **Technical Correctness**: Are mathematical notations, terminology, and citations accurate?
5. **LaTeX Quality**: Is the code well-structured and likely to compile?

OUTPUT FORMAT:
Provide a structured JSON response with:
{
  "overall_quality": "excellent|good|fair|poor",
  "accuracy_score": 0-10,
  "completeness_score": 0-10,
  "clarity_score": 0-10,
  "issues": ["list of specific problems found"],
  "strengths": ["list of what was done well"],
  "recommendations": ["specific suggestions for improvement"],
  "passes_verification": true/false
}

Be thorough but fair. Minor LaTeX formatting issues are acceptable if content is accurate.
"""


def verify_beamer_presentation(
    arxiv_id: str, 
    tex_content: str, 
    paper_text: str, 
    model: str
) -> dict:
    """
    Uses an LLM to verify the quality and accuracy of a generated Beamer presentation.
    Passes the full paper text for comprehensive verification.
    """
    arxiv_id = canonical_id(arxiv_id)
    print(f"Verifying presentation for {arxiv_id}...")
    
    user_prompt = f"""
Please verify this Beamer presentation against the original paper.

**Full Paper Text:**
{paper_text}

**Generated LaTeX Presentation:**
{tex_content}

Evaluate the presentation according to the criteria and provide your assessment in JSON format.
"""

    response = tool_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": CHECKER_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.1
    )
    
    raw_response = response.choices[0].message.content
    
    # Try to extract JSON
    import json
    try:
        if "```json" in raw_response:
            json_str = raw_response.split("```json")[1].split("```")[0].strip()
        elif "```" in raw_response:
            json_str = raw_response.split("```")[1].split("```")[0].strip()
        else:
            json_str = raw_response
        
        verification_result = json.loads(json_str)
    except:
        # Fallback if parsing fails
        verification_result = {
            "overall_quality": "unknown",
            "raw_response": raw_response,
            "passes_verification": True  # Default to passing if we can't parse
        }
    
    return verification_result


def compile_beamer_to_pdf(arxiv_id: str) -> dict:
    """
    Compiles a LaTeX Beamer presentation to PDF using xelatex.
    """
    arxiv_id = canonical_id(arxiv_id)
    tex_file = FOLDERS["presentations_tex"] / f"presentation_{arxiv_id}.tex"
    
    if not tex_file.exists():
        return {
            "status": "error",
            "message": f"LaTeX file not found: {tex_file}"
        }
    
    print(f"Compiling {tex_file} to PDF with xelatex...")
    
    # Check if xelatex is available
    if not shutil.which("xelatex"):
        return {
            "status": "error",
            "message": "xelatex not found. Please install a LaTeX distribution (e.g., TeX Live, MiKTeX)."
        }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_tex = Path(tmpdir) / tex_file.name
        shutil.copy(tex_file, temp_tex)
        
        try:
            # Run xelatex twice (TOC, references)
            for _ in range(2):
                result = subprocess.run(
                    [
                        "xelatex",
                        "-interaction=nonstopmode",
                        "-halt-on-error",
                        temp_tex.name,
                    ],
                    cwd=tmpdir,
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                
                if result.returncode != 0:
                    return {
                        "status": "error",
                        "message": "XeLaTeX compilation failed",
                        "log": result.stdout[-2000:] + result.stderr[-2000:],
                    }
            
            temp_pdf = Path(tmpdir) / temp_tex.with_suffix(".pdf").name
            
            if temp_pdf.exists():
                output_pdf = (
                    FOLDERS["presentations_pdf"]
                    / f"presentation_{arxiv_id}.pdf"
                )
                shutil.copy(temp_pdf, output_pdf)
                
                return {
                    "status": "success",
                    "message": "PDF compiled successfully with xelatex",
                    "pdf_file": str(output_pdf),
                }
            else:
                return {
                    "status": "error",
                    "message": "PDF was not produced by xelatex",
                    "log": result.stdout[-2000:],
                }
        
        except subprocess.TimeoutExpired:
            return {
                "status": "error",
                "message": "PDF compilation timed out (>60s)",
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Compilation error: {str(e)}",
            }
