import os
import requests
import feedparser
from typing import List, Dict, Any
import tarfile
import tempfile
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

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

TOOL_MODEL = "openai/gpt-oss-120b:free"  # Or use a stronger model like gpt-4o

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
        papers.append({
            "title": entry.title,
            "authors": [a.name for a in entry.authors],
            "summary": entry.summary,
            "published": entry.published,
            "link": entry.link,
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
    Fetch a research paper from arXiv.
    
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


def create_beamer_presentation(arxiv_id: str) -> str:
    """
    Generates a LaTeX Beamer presentation for the given arXiv ID.
    """
    # 1. Get the content
    print(f"Fetching content for {arxiv_id}...")
    data = arxiv_to_text(arxiv_id)
    
    source = data['source_used']
    text = data['full_text']
    
    # Truncate text if strictly necessary for the model context, 
    # though modern flash models handle 100k+ tokens.
    # text = text[:100000] 

    print(f"Generating Slides via LLM (Source: {source})...")

    # 2. Construct Prompt
    system_prompt = (
        "You are an expert academic assistant. "
        "Your goal is to write a complete, valid LaTeX Beamer presentation based on the provided research paper text."
    )

    user_prompt = f"""
    Create a detailed LaTeX Beamer presentation for the following paper.
    
    **Instructions:**
    1. Use the `beamer` document class.
    2. Include sections for: Introduction, Methodology, Experiments/Results, and Conclusion.
    3. Ensure the LaTeX code compiles (escape special characters like %, _, & properly).
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

    **Paper Content:**
    {text}
    
    Return ONLY the raw LaTeX code inside a markdown code block (```latex ... ```).
    """

    # 3. Call LLM to generate LaTeX
    response = tool_client.chat.completions.create(
        model=TOOL_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
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

    # 5. Save to file
    filename = f"presentation_{arxiv_id}.tex"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(latex_code)

    return f"Presentation generated successfully based on {source} source. Saved to file: {filename}"