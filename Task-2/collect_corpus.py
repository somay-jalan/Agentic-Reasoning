# =============================================================================
# Physics Python Code Corpus Collector  —  v2
#
# Key changes from v1:
#   1. Inferred samples are DROPPED — only PR- or issue-backed samples kept.
#   2. Pre-fix file is fetched for every sample:
#        • PR   → file at pr.base.sha
#        • Issue → file at parent of the closing commit
#   3. CodeSample gains: pre_fix_code, task_type ("fix" | "generate")
#   4. Prompt template includes the pre-fix file when it exists.
#   5. Retry + exponential back-off on secondary-rate-limit errors.
# =============================================================================

from __future__ import annotations

import os
import re
import time
import json
import base64
import logging
import hashlib
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Optional
from collections import defaultdict

from dotenv import load_dotenv
from github import (
    Github,
    Auth,
    RateLimitExceededException,
    GithubException,
)
from tqdm import tqdm
from urllib3.util.retry import Retry


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

load_dotenv(dotenv_path=Path(__file__).parent / ".env")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")

OUTPUT_DIR        = Path("physics_corpus")
LOG_LEVEL         = logging.INFO

FILES_PER_LIBRARY = 10          # target samples per (library, sub_domain)
MIN_LINES         = 20          # discard files shorter than this
MAX_BYTES         = 80_000      # discard files larger than this

# How many recent commits to scan when hunting for a closing commit
CLOSING_COMMIT_SCAN_DEPTH = 40

# Retry settings for secondary-rate-limit / transient errors
MAX_RETRIES      = 4
BASE_BACKOFF_SEC = 15           # doubles each retry
MAX_SKIPS_PER_QUERY = 600   # give up on a query after this many skips

REPO_BLOCKLIST = {
    "numpy/numpy-tutorials",
    "scipy/scipy.org",
}

# Prompt-source labels
PROMPT_SOURCE_PR    = "pull_request"
PROMPT_SOURCE_ISSUE = "issue"

# Task-type labels stored on CodeSample
TASK_FIX      = "fix"       # pre-fix file existed  → LLM must repair/extend it
TASK_GENERATE = "generate"  # PR creates a new file  → LLM generates from scratch

# ---------------------------------------------------------------------------
# Library search targets
# (library, sub_domain, [search_queries])
# ---------------------------------------------------------------------------

LIBRARY_TARGETS: list[tuple[str, str, list[str]]] = [
    ("scipy", "ode_integration",
     ["scipy.integrate solve_ivp physics simulation language:Python",
      "scipy odeint classical mechanics pendulum language:Python"]),

    ("scipy", "signal_processing",
     ["scipy.signal fft physics waveform language:Python",
      "scipy signal spectrogram frequency physics language:Python"]),

    ("scipy", "optimization",
     ["scipy.optimize minimize energy physics language:Python",
      "scipy curve_fit experimental data physics language:Python"]),

    ("numpy", "linear_algebra",
     ["numpy linalg eig eigenvalue quantum mechanics language:Python",
      "numpy matrix physics simulation Hamiltonian language:Python"]),

    ("numpy", "fft_spectral",
     ["numpy fft physics spectral analysis language:Python",
      "numpy rfft signal physics power spectrum language:Python"]),

    ("sympy", "classical_mechanics",
     ["sympy Lagrangian mechanics physics derive language:Python",
      "sympy symbols equations of motion physics language:Python"]),

    ("sympy", "quantum_symbolic",
     ["sympy quantum operators commutator physics language:Python",
      "sympy solve Schrodinger equation symbolic language:Python"]),

    ("astropy", "coordinates_units",
     ["astropy units coordinates astrophysics language:Python",
      "astropy SkyCoord galactic coordinates simulation language:Python"]),

    ("astropy", "cosmology",
     ["astropy cosmology FlatLambdaCDM Hubble language:Python",
      "astropy cosmology luminosity distance redshift language:Python"]),

    ("qutip", "quantum_dynamics",
     ["qutip mesolve Lindblad master equation language:Python",
      "qutip sesolve quantum state evolution language:Python"]),

    ("qutip", "quantum_optics",
     ["qutip cavity QED Jaynes-Cummings language:Python",
      "qutip Wigner function qubit simulation language:Python"]),

    ("fenics", "pde_fem",
     ["FEniCS FunctionSpace TrialFunction physics PDE language:Python",
      "dolfinx finite element fluid dynamics language:Python"]),

    ("plasmapy", "plasma_physics",
     ["plasmapy plasma_frequency Alfven speed language:Python",
      "PlasmaPy particle simulation magnetic field language:Python"]),

    ("pint", "unit_physics",
     ["pint UnitRegistry Quantity physics calculation language:Python",
      "pint dimensional analysis physics simulation language:Python"]),

    ("pybamm", "electrochemistry",
     ["PyBaMM battery model electrochemistry language:Python",
      "pybamm lithium ion SPM simulation language:Python"]),

    ("general", "nbody_gravity",
     ["n-body gravitational simulation python numpy language:Python",
      "gravitational orbit simulation velocity verlet python language:Python"]),

    ("general", "fluid_dynamics",
     ["Navier Stokes python simulation numpy scipy language:Python",
      "lattice Boltzmann fluid simulation python language:Python"]),

    ("general", "monte_carlo",
     ["Monte Carlo physics simulation python numpy language:Python",
      "Ising model Monte Carlo python simulation language:Python"]),

    ("general", "molecular_dynamics",
     ["molecular dynamics simulation python lennard jones language:Python",
      "MD simulation velocity verlet python physics language:Python"]),
]

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class CodeSample:
    # ── Identity ─────────────────────────────────────────────────────────────
    sample_id:      str
    library:        str
    sub_domain:     str

    # ── Source location ──────────────────────────────────────────────────────
    repo_full_name:  str
    repo_stars:      int
    repo_description: str
    file_path:       str
    file_url:        str
    raw_url:         str

    # ── Code ────────────────────────────────────────────────────────────────
    code:        str        # post-fix / current file (ground-truth reference)
    lines:       int
    size_bytes:  int

    # ── Pre-fix code (the "broken" or "incomplete" file before the fix) ──────
    pre_fix_code: Optional[str]   # None when the PR/issue created a brand-new file

    # ── Task classification ──────────────────────────────────────────────────
    task_type:    str       # TASK_FIX | TASK_GENERATE

    # ── Prompt provenance ────────────────────────────────────────────────────
    prompt_source:       str    # PROMPT_SOURCE_PR | PROMPT_SOURCE_ISSUE
    has_real_prompt:     bool   # always True in v2 (inferred samples dropped)

    # ── Issue / PR context ───────────────────────────────────────────────────
    related_issue_title: Optional[str]
    related_issue_body:  Optional[str]
    related_pr_title:    Optional[str]
    related_pr_body:     Optional[str]

    # ── Provenance detail ────────────────────────────────────────────────────
    closing_commit_sha:  Optional[str]   # SHA of the commit that closed the issue
    pr_base_sha:         Optional[str]   # SHA of the PR base branch head
    search_query:        str

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("corpus_collector")

# ---------------------------------------------------------------------------
# Low-level GitHub helpers
# ---------------------------------------------------------------------------
SEARCH_RATE_LIMIT_MAX = 30   # GitHub's hard cap for authenticated search
SEARCH_RATE_LIMIT_BUFFER = 5  # only sleep when genuinely almost empty

def _wait_for_rate_limit(gh: Github, buffer: int = SEARCH_RATE_LIMIT_BUFFER) -> None:
    rl        = gh.get_rate_limit()
    remaining = rl.resources.search.remaining
    reset_ts  = rl.resources.search.reset.timestamp()

    if remaining < buffer:
        wait_secs = max(0, reset_ts - time.time()) + 2
        log.warning(
            "Search rate-limit genuinely low (%d / %d remaining). Sleeping %.0f s …",
            remaining, SEARCH_RATE_LIMIT_MAX, wait_secs,
        )
        time.sleep(wait_secs)
    else:
        # Pace requests to stay under 30/min without hitting the wall
        # 60s / 30 requests = 2s minimum between search calls
        time.sleep(2.0)


def _core_rate_check(gh: Github, buffer: int = 20) -> None:
    rl        = gh.get_rate_limit()
    remaining = rl.resources.core.remaining
    if remaining < buffer:
        reset_ts  = rl.resources.core.reset.timestamp()
        wait_secs = max(0, reset_ts - time.time()) + 5
        log.warning(
            "Core rate-limit low (%d remaining). Sleeping %.0f s …",
            remaining, wait_secs,
        )
        time.sleep(wait_secs)

def _github_call(fn, *args, **kwargs):
    for attempt in range(MAX_RETRIES):
        try:
            return fn(*args, **kwargs)
        except RateLimitExceededException:
            raise
        except GithubException as exc:
            status = getattr(exc, "status", 0)
            headers = getattr(exc, "headers", {}) or {}

            # Secondary rate limit has this header; plain 403 access-denied does NOT
            is_secondary_rate_limit = (
                status == 403
                and "retry-after" in headers
            ) or status in (429, 500, 502, 503)

            if is_secondary_rate_limit:
                wait = BASE_BACKOFF_SEC * (2 ** attempt)
                log.warning("Secondary rate-limit hit — backing off %d s …", wait)
                time.sleep(wait)
                if attempt == MAX_RETRIES - 1:
                    raise
            else:
                raise   # permanent error — don't retry, let caller handle it

def _sha(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()[:10]


def _decode_content(content_file) -> Optional[str]:
    try:
        return base64.b64decode(content_file.content).decode("utf-8", errors="replace")
    except Exception:
        return None


def _is_physics_file(code: str) -> bool:
    PHYSICS_KEYWORDS = [
        "scipy", "numpy", "sympy", "astropy", "qutip",
        "fenics", "dolfinx", "plasmapy", "pint", "pybamm",
        "matplotlib", "simulation", "odeint", "solve_ivp",
        "Hamiltonian", "Lagrangian", "quantum",
    ]
    code_lower = code.lower()
    return sum(1 for kw in PHYSICS_KEYWORDS if kw in code_lower) >= 2

# ---------------------------------------------------------------------------
# Closing-commit detection
# ---------------------------------------------------------------------------

# Patterns like "fixes #42", "closes #42", "resolves #42"
_CLOSE_PATTERN = re.compile(
    r"(fix|fixes|fixed|close|closes|closed|resolve|resolves|resolved)"
    r"\s*#\s*(\d+)\b",
    re.IGNORECASE,
)



def _find_closing_commit(repo, issue_number: int, file_path: str) -> Optional[str]:
    try:
        commits = _github_call(repo.get_commits, path=file_path)
        for i, commit in enumerate(tqdm(
            commits,
            desc="    Commits",
            unit="commit",
            total=CLOSING_COMMIT_SCAN_DEPTH,
            colour="cyan",
            leave=False,
        )):
            if i >= CLOSING_COMMIT_SCAN_DEPTH:
                break
            msg = commit.commit.message or ""
            for match in _CLOSE_PATTERN.finditer(msg):
                if int(match.group(2)) == issue_number:
                    return commit.sha
    except GithubException:
        pass
    return None

# ---------------------------------------------------------------------------
# Pre-fix file fetcher
# ---------------------------------------------------------------------------

def _fetch_pre_fix_code(gh: Github, repo, file_path: str, ref_sha: str) -> Optional[str]:
    _core_rate_check(gh)   # gh passed directly, no internal excavation
    try:
        content = _github_call(repo.get_contents, file_path, ref=ref_sha)
        if isinstance(content, list):
            return None
        return _decode_content(content)
    except GithubException as exc:
        if getattr(exc, "status", 0) == 404:
            return None
        log.debug("Could not fetch pre-fix file at %s: %s", ref_sha[:8], exc)
        return None


def _fetch_related_context(gh: Github, repo, file_path: str) -> dict:
    ctx = {
        "issue_title":         None,
        "issue_body":          None,
        "pr_title":            None,
        "pr_body":             None,
        "pre_fix_code":        None,
        "closing_commit_sha":  None,
        "pr_base_sha":         None,
        "prompt_source":       None,
        "has_real_prompt":     False,
    }

    stem = Path(file_path).stem.lower()

    try:
        issues = _github_call(repo.get_issues, state="closed")
        for issue in tqdm(
            issues,
            desc=f"    Issues ({Path(file_path).stem})",
            unit="issue",
            total=None,
            colour="yellow",
            leave=False,
        ):
            title_hit = stem in (issue.title or "").lower()
            body_hit  = stem in (issue.body  or "").lower()
            if not (title_hit or body_hit):
                continue

            if issue.pull_request is not None:
                pr = _github_call(repo.get_pull, issue.number)
                ctx["pr_title"]        = issue.title
                ctx["pr_body"]         = (issue.body or "")[:800]
                ctx["pr_base_sha"]     = pr.base.sha
                ctx["prompt_source"]   = PROMPT_SOURCE_PR
                ctx["has_real_prompt"] = True
                ctx["pre_fix_code"]    = _fetch_pre_fix_code(
                    gh, repo, file_path, pr.base.sha
                )
            else:
                ctx["issue_title"]     = issue.title
                ctx["issue_body"]      = (issue.body or "")[:800]
                ctx["prompt_source"]   = PROMPT_SOURCE_ISSUE
                ctx["has_real_prompt"] = True
                closing_sha = _find_closing_commit(repo, issue.number, file_path)
                ctx["closing_commit_sha"] = closing_sha
                if closing_sha:
                    try:
                        closing_commit = _github_call(repo.get_commit, closing_sha)
                        if closing_commit.parents:
                            parent_sha = closing_commit.parents[0].sha
                            ctx["pre_fix_code"] = _fetch_pre_fix_code(
                                gh, repo, file_path, parent_sha
                            )
                    except GithubException:
                        pass
            break

    except GithubException:
        pass

    return ctx


# ---------------------------------------------------------------------------
# Core collector
# ---------------------------------------------------------------------------

def collect_for_target(
    gh:           Github,
    library:      str,
    sub_domain:   str,
    queries:      list[str],
    files_wanted: int,
    seen_hashes:  set[str],
) -> list[CodeSample]:

    samples: list[CodeSample] = []

    for query in queries:
        if len(samples) >= files_wanted:
            break

        tqdm.write(f"[{library}/{sub_domain}] Query: {query!r}")
        _wait_for_rate_limit(gh)

        try:
            results = _github_call(gh.search_code, query, order="desc")
        except RateLimitExceededException:
            _wait_for_rate_limit(gh, buffer=0)
            results = gh.search_code(query, order="desc")
        except GithubException as exc:
            tqdm.write(f"Search failed: {exc}")
            continue

        skips = 0          # ← total skips for this entire query (never reset on success)

        for file_result in tqdm(
            results,
            desc=f"  {library}/{sub_domain}",
            unit="file",
            total=MAX_SKIPS_PER_QUERY,
            colour="blue",
            leave=False,
        ):
            if len(samples) >= files_wanted:
                break

            # ── Total-skip guard (checked before ANY API call) ────────────
            if skips >= MAX_SKIPS_PER_QUERY:
                tqdm.write(
                    f"  ↯ Reached {skips} total skips — "
                    f"abandoning query and trying next"
                )
                break

            # ── Cheap filters first (no API calls) ───────────────────────
            repo = file_result.repository
            if repo.full_name in REPO_BLOCKLIST:
                skips += 1; continue
            if repo.fork:
                skips += 1; continue
            if repo.stargazers_count < 5:
                skips += 1; continue
            if not file_result.path.endswith(".py"):
                skips += 1; continue
            if file_result.size > MAX_BYTES:
                skips += 1; continue

            # ── First API call: fetch file contents ───────────────────────
            try:
                _core_rate_check(gh)
                content_file = _github_call(repo.get_contents, file_result.path)
            except GithubException as exc:
                if getattr(exc, "status", 0) == 403:
                    tqdm.write(f"  ✗ Skipping {repo.full_name} — access forbidden")
                skips += 1
                continue

            if isinstance(content_file, list):
                skips += 1; continue

            code = _decode_content(content_file)
            if code is None:
                skips += 1; continue

            lines = code.count("\n") + 1
            if lines < MIN_LINES:
                skips += 1; continue
            if not _is_physics_file(code):
                skips += 1; continue

            code_hash = _sha(code)
            if code_hash in seen_hashes:
                skips += 1; continue
            seen_hashes.add(code_hash)

            # ── Expensive API call: fetch PR/issue context ────────────────
            ctx = _fetch_related_context(gh, repo, file_result.path)

            if not ctx["has_real_prompt"]:
                tqdm.write(f"  ✗ Skipping {file_result.path} — no PR/issue found")
                skips += 1
                continue

            # ── Successful sample (skips NOT reset — total-skip model) ────
            task_type = TASK_GENERATE if ctx["pre_fix_code"] is None else TASK_FIX

            sample = CodeSample(
                sample_id            = f"{library}__{sub_domain}__{code_hash}",
                library              = library,
                sub_domain           = sub_domain,
                repo_full_name       = repo.full_name,
                repo_stars           = repo.stargazers_count,
                repo_description     = repo.description or "",
                file_path            = file_result.path,
                file_url             = file_result.html_url,
                raw_url              = content_file.download_url or "",
                code                 = code,
                lines                = lines,
                size_bytes           = file_result.size,
                pre_fix_code         = ctx["pre_fix_code"],
                task_type            = task_type,
                prompt_source        = ctx["prompt_source"],
                has_real_prompt      = True,
                related_issue_title  = ctx["issue_title"],
                related_issue_body   = ctx["issue_body"],
                related_pr_title     = ctx["pr_title"],
                related_pr_body      = ctx["pr_body"],
                closing_commit_sha   = ctx["closing_commit_sha"],
                pr_base_sha          = ctx["pr_base_sha"],
                search_query         = query,
            )
            samples.append(sample)

            src_tag  = "PR"  if ctx["prompt_source"] == PROMPT_SOURCE_PR else "ISS"
            task_tag = "FIX" if task_type == TASK_FIX else "GEN"
            tqdm.write(
                f"  ✓ [{len(samples)}/{files_wanted}] {file_result.path} "
                f"({lines} lines, ★{repo.stargazers_count}) "
                f"[{src_tag}|{task_tag}] skips so far: {skips}"
            )

            time.sleep(0.5)

    return samples

# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

_MAX_PRE_FIX_LINES = 3000   # truncate very long pre-fix files in the prompt


def _truncate_code(code: str, max_lines: int = _MAX_PRE_FIX_LINES) -> str:
    lines = code.splitlines()
    if len(lines) <= max_lines:
        return code
    kept = lines[:max_lines]
    kept.append(f"# ... ({len(lines) - max_lines} more lines truncated)")
    return "\n".join(kept)


def build_benchmark_prompt(sample: CodeSample) -> str:
    """
    Construct a zero-shot benchmark prompt.

    • PR samples:    spec = PR title + body.
    • Issue samples: spec = issue title + body.
    • Pre-fix code is injected when available (task_type == TASK_FIX).
    • No pre-fix code means the PR/issue *created* a new file (task_type == TASK_GENERATE).
    """
    lib_hint = f"Use the `{sample.library}` Python library."

    # ── Task specification ─────────────────────────────────────────────────
    if sample.related_pr_body:
        spec = (
            f"A pull request titled:\n"
            f"  '{sample.related_pr_title}'\n\n"
            f"was opened with this description:\n\n"
            f"{sample.related_pr_body}"
        )
    else:
        spec = (
            f"A GitHub issue titled:\n"
            f"  '{sample.related_issue_title}'\n\n"
            f"was filed with this description:\n\n"
            f"{sample.related_issue_body}"
        )

    # ── Pre-fix file section ───────────────────────────────────────────────
    if sample.pre_fix_code:
        pre_fix_block = (
            f"\nORIGINAL FILE (the version that existed before the fix):\n"
            f"```python\n"
            f"{_truncate_code(sample.pre_fix_code)}\n"
            f"```\n\n"
            f"Your task is to produce a corrected / improved version of this file "
            f"that addresses the issue or pull-request description above.\n"
        )
    else:
        pre_fix_block = (
            f"\nThis pull request creates a new file from scratch — "
            f"no prior version exists.\n"
            f"Your task is to implement the functionality described above.\n"
        )

    # ── Full prompt ────────────────────────────────────────────────────────
    return (
        f"You are an expert Python physicist and scientific programmer.\n\n"
        f"TASK:\n{spec}\n"
        f"{pre_fix_block}\n"
        f"REQUIREMENTS:\n"
        f"- {lib_hint}\n"
        f"- Write clean, well-commented Python code.\n"
        f"- Include a runnable `main()` block or example usage.\n"
        f"- Handle units and physical constants correctly.\n"
        f"- Do NOT reproduce the original file verbatim; write your own implementation.\n\n"
        f"Generate only the Python code, with no explanations outside of inline comments."
    )

# ---------------------------------------------------------------------------
# Save corpus
# ---------------------------------------------------------------------------

def save_corpus(samples: list[CodeSample], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    all_records = []

    for s in samples:
        d = asdict(s)

        sub_dir = output_dir / s.library / s.sub_domain
        sub_dir.mkdir(parents=True, exist_ok=True)

        # Current (post-fix / reference) file
        py_path = sub_dir / f"{s.sample_id}.py"
        py_path.write_text(s.code, encoding="utf-8")

        # Pre-fix file (ground-truth input for FIX tasks)
        if s.pre_fix_code:
            pre_py_path = sub_dir / f"{s.sample_id}.pre_fix.py"
            pre_py_path.write_text(s.pre_fix_code, encoding="utf-8")
            d["local_pre_fix_py_path"] = str(pre_py_path)
        else:
            d["local_pre_fix_py_path"] = None

        # Prompt
        prompt = build_benchmark_prompt(s)
        prompt_path = sub_dir / f"{s.sample_id}.prompt.txt"
        prompt_path.write_text(prompt, encoding="utf-8")

        d["prompt"]            = prompt
        d["local_py_path"]     = str(py_path)
        d["local_prompt_path"] = str(prompt_path)
        all_records.append(d)

    index_path = output_dir / "corpus_index.json"
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(all_records, f, indent=2, ensure_ascii=False)

    log.info("Saved %d samples → %s", len(samples), output_dir.resolve())
    log.info("Benchmark index → %s", index_path)

# ---------------------------------------------------------------------------
# Coverage report
# ---------------------------------------------------------------------------

def print_prompt_coverage_report(samples: list[CodeSample]) -> None:
    """
    Breakdown by library and task type.
    Also writes:
      - physics_corpus/prompts_pr.json
      - physics_corpus/prompts_issue.json
      - physics_corpus/tasks_fix.json
      - physics_corpus/tasks_generate.json
    """
    pr_samples    = [s for s in samples if s.prompt_source == PROMPT_SOURCE_PR]
    issue_samples = [s for s in samples if s.prompt_source == PROMPT_SOURCE_ISSUE]
    fix_samples   = [s for s in samples if s.task_type == TASK_FIX]
    gen_samples   = [s for s in samples if s.task_type == TASK_GENERATE]

    W = 72
    print("\n" + "=" * W)
    print("  PROMPT COVERAGE REPORT  (v2 — no inferred samples)")
    print("=" * W)
    print(f"  Total samples collected   : {len(samples)}")
    print(f"  Source → Pull Request     : {len(pr_samples)}")
    print(f"  Source → Issue            : {len(issue_samples)}")
    print(f"  Task   → Fix/improve      : {len(fix_samples)}  "
          f"(pre-fix file found)")
    print(f"  Task   → Generate new     : {len(gen_samples)}  "
          f"(PR creates new file)")
    print("-" * W)

    # Per-library breakdown
    lib_stats: dict[str, dict] = defaultdict(
        lambda: {"pr": 0, "issue": 0, "fix": 0, "generate": 0, "total": 0}
    )
    for s in samples:
        lib_stats[s.library]["total"]             += 1
        lib_stats[s.library][s.prompt_source.split("_")[0]] += 1
        lib_stats[s.library][s.task_type]         += 1

    header = f"  {'Library':<15}  {'Total':>5}  {'PR':>4}  {'Issue':>5}  " \
             f"{'Fix':>4}  {'Gen':>4}"
    print(f"\n{header}")
    print(f"  {'-'*15}  {'-'*5}  {'-'*4}  {'-'*5}  {'-'*4}  {'-'*4}")
    for lib, st in sorted(lib_stats.items()):
        print(
            f"  {lib:<15}  {st['total']:>5}  {st['pull']:>4}  "
            f"{st['issue']:>5}  {st['fix']:>4}  {st['generate']:>4}"
        )

    # Detailed sample lists
    print(f"\n  {'─'*W}")
    print("  ALL SAMPLES")
    print(f"  {'─'*W}")
    for s in samples:
        src  = "PR " if s.prompt_source == PROMPT_SOURCE_PR else "ISS"
        task = "FIX" if s.task_type == TASK_FIX else "GEN"
        title = (s.related_pr_title or s.related_issue_title or "")[:60]
        print(f"  [{src}|{task}]  {s.library}/{s.sub_domain}")
        print(f"            file  : {s.file_path}")
        print(f"            title : {title}")
        print()

    print("=" * W + "\n")

    # Save manifests
    def _manifest(lst: list[CodeSample]) -> list[dict]:
        return [
            {
                "sample_id":           s.sample_id,
                "library":             s.library,
                "sub_domain":          s.sub_domain,
                "prompt_source":       s.prompt_source,
                "task_type":           s.task_type,
                "has_pre_fix_code":    s.pre_fix_code is not None,
                "file_path":           s.file_path,
                "repo_full_name":      s.repo_full_name,
                "closing_commit_sha":  s.closing_commit_sha,
                "pr_base_sha":         s.pr_base_sha,
                "related_pr_title":    s.related_pr_title,
                "related_issue_title": s.related_issue_title,
            }
            for s in lst
        ]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    manifests = {
        "prompts_pr.json":       pr_samples,
        "prompts_issue.json":    issue_samples,
        "tasks_fix.json":        fix_samples,
        "tasks_generate.json":   gen_samples,
    }
    for fname, lst in manifests.items():
        with open(OUTPUT_DIR / fname, "w", encoding="utf-8") as f:
            json.dump(_manifest(lst), f, indent=2)

    log.info("Manifests saved → %s", ", ".join(manifests))

# ---------------------------------------------------------------------------
# General corpus summary
# ---------------------------------------------------------------------------

def print_summary(samples: list[CodeSample]) -> None:
    by_lib: dict[str, list[CodeSample]] = defaultdict(list)
    for s in samples:
        by_lib[s.library].append(s)

    W = 72
    print("\n" + "=" * W)
    print(f"  CORPUS SUMMARY  —  {len(samples)} total samples")
    print("=" * W)
    print(
        f"  {'Library':<15}  {'Sub-domains':>11}  {'Files':>5}  "
        f"{'Avg lines':>9}  {'Fix':>4}  {'Gen':>4}"
    )
    print(f"  {'-'*15}  {'-'*11}  {'-'*5}  {'-'*9}  {'-'*4}  {'-'*4}")
    for lib, slist in sorted(by_lib.items()):
        sub_domains = len({s.sub_domain for s in slist})
        avg_lines   = sum(s.lines for s in slist) // max(len(slist), 1)
        fixes       = sum(1 for s in slist if s.task_type == TASK_FIX)
        gens        = sum(1 for s in slist if s.task_type == TASK_GENERATE)
        print(
            f"  {lib:<15}  {sub_domains:>11}  {len(slist):>5}  "
            f"{avg_lines:>9}  {fixes:>4}  {gens:>4}"
        )
    print("=" * W)
    print(f"  Output directory : {OUTPUT_DIR.resolve()}")
    print("=" * W + "\n")

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    if not GITHUB_TOKEN:
        raise EnvironmentError("GITHUB_TOKEN not found.")

    auth = Auth.Token(GITHUB_TOKEN)
    gh   = Github(auth=auth, per_page=30, retry=Retry(total=0))

    log.info("Authenticated as : %s", gh.get_user().login)

    all_samples: list[CodeSample] = []
    seen_hashes: set[str]         = set()

    for library, sub_domain, queries in tqdm(
        LIBRARY_TARGETS,
        desc="Targets",
        unit="target",
        colour="green",
    ):
        samples = collect_for_target(
            gh           = gh,
            library      = library,
            sub_domain   = sub_domain,
            queries      = queries,
            files_wanted = FILES_PER_LIBRARY,
            seen_hashes  = seen_hashes,
        )
        all_samples.extend(samples)
        tqdm.write(   # use tqdm.write instead of log.info to avoid bar corruption
            f"  → {library}/{sub_domain}: "
            f"collected {len(samples)} / {FILES_PER_LIBRARY}"
        )
        time.sleep(3)

    if not all_samples:
        tqdm.write("No samples collected — check your search queries and token.")
        return

    save_corpus(all_samples, OUTPUT_DIR)
    print_summary(all_samples)
    print_prompt_coverage_report(all_samples)

if __name__ == "__main__":
    main()