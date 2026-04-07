# domain_weights.py
"""
Builds domain-specific token weights for CodeBLEU by:
  1. Parsing all reference (_after.py) files into ASTs
  2. Extracting meaningful node categories (calls, attributes, classes, etc.)
  3. Computing TF-IDF-style weights across the corpus
  4. Optionally augmenting with weights from SymPy's own source tree

Node categories and their base importance multipliers:
  - Class instantiation (Matrix(), Symbol())         → high
  - Method/attribute chains (expr.diff(), .evalf())  → high
  - Module-level function calls (solve(), eigenvals) → high
  - Exception types raised/caught                    → medium
  - Decorator names                                  → medium
  - String constants in assertions                   → low
  - Plain Name loads                                 → low
"""

import ast
import math
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

# ───────────────────────── CONFIG ─────────────────────────────────

ROOT = Path(__file__).resolve().parent

# Base importance score per AST node category
# These reflect how domain-discriminative each category is
CATEGORY_BASE_WEIGHT = {
    "class_instantiation":   3.0,   # Matrix(...), Symbol(...), FpGroup(...)
    "method_call":           2.5,   # expr.diff(), m.eigenvals(), T.kernel()
    "attribute_access":      2.0,   # numpy.isnan, S.Infinity, S.Zero
    "function_call":         2.0,   # solve(), linprog(), homomorphism()
    "exception_type":        1.8,   # NotImplementedError, ValueError
    "decorator":             1.5,   # @property, @cacheit
    "import_name":           1.3,   # from sympy import ..., import numpy
    "comparison_target":     1.2,   # rhs of assert x == Symbol(...)
    "name_load":             1.0,   # bare name references
    "string_constant":       0.5,   # string literals (low signal)
}

# ──────────────────────── AST VISITOR ─────────────────────────────

class DomainTokenExtractor(ast.NodeVisitor):
    """
    Walks a Python AST and extracts (token, category) pairs.
    Each pair gets a base weight from CATEGORY_BASE_WEIGHT.
    """

    def __init__(self):
        self.tokens: List[Tuple[str, str]] = []   # (token, category)

    # ── Imports ──────────────────────────────────────────────────

    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            root_mod = alias.name.split(".")[0]
            self.tokens.append((root_mod, "import_name"))
            if alias.asname:
                self.tokens.append((alias.asname, "import_name"))
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        if node.module:
            root_mod = node.module.split(".")[0]
            self.tokens.append((root_mod, "import_name"))
        for alias in node.names:
            self.tokens.append((alias.name, "import_name"))
            if alias.asname:
                self.tokens.append((alias.asname, "import_name"))
        self.generic_visit(node)

    # ── Function / Method Calls ───────────────────────────────────

    def visit_Call(self, node: ast.Call):
        func = node.func

        if isinstance(func, ast.Name):
            # Plain call: solve(...), Matrix(...), Symbol(...)
            name = func.id
            # Heuristic: if name starts with uppercase → likely class instantiation
            cat = "class_instantiation" if name[0].isupper() else "function_call"
            self.tokens.append((name, cat))

        elif isinstance(func, ast.Attribute):
            # a.b(...) — method call
            attr = func.attr
            self.tokens.append((attr, "method_call"))

            # Also record the object being called on, as an attribute access
            if isinstance(func.value, ast.Name):
                self.tokens.append((func.value.id, "attribute_access"))
            elif isinstance(func.value, ast.Attribute):
                self.tokens.append((func.value.attr, "attribute_access"))

        self.generic_visit(node)

    # ── Attribute Access (non-call) ───────────────────────────────

    def visit_Attribute(self, node: ast.Attribute):
        # Only record if NOT inside a Call (those are caught above)
        if not isinstance(node.ctx, ast.Load):
            self.generic_visit(node)
            return
        parent_is_call = False   # we rely on visit_Call handling call attrs
        self.tokens.append((node.attr, "attribute_access"))
        if isinstance(node.value, ast.Name):
            self.tokens.append((node.value.id, "attribute_access"))
        self.generic_visit(node)

    # ── Exception handling ────────────────────────────────────────

    def visit_Raise(self, node: ast.Raise):
        if node.exc:
            exc = node.exc
            if isinstance(exc, ast.Call) and isinstance(exc.func, ast.Name):
                self.tokens.append((exc.func.id, "exception_type"))
            elif isinstance(exc, ast.Name):
                self.tokens.append((exc.id, "exception_type"))
        self.generic_visit(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler):
        if node.type:
            if isinstance(node.type, ast.Name):
                self.tokens.append((node.type.id, "exception_type"))
            elif isinstance(node.type, ast.Tuple):
                for elt in node.type.elts:
                    if isinstance(elt, ast.Name):
                        self.tokens.append((elt.id, "exception_type"))
        self.generic_visit(node)

    # ── Decorators ───────────────────────────────────────────────

    def visit_FunctionDef(self, node: ast.FunctionDef):
        for dec in node.decorator_list:
            if isinstance(dec, ast.Name):
                self.tokens.append((dec.id, "decorator"))
            elif isinstance(dec, ast.Attribute):
                self.tokens.append((dec.attr, "decorator"))
        self.generic_visit(node)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node: ast.ClassDef):
        for dec in node.decorator_list:
            if isinstance(dec, ast.Name):
                self.tokens.append((dec.id, "decorator"))
            elif isinstance(dec, ast.Attribute):
                self.tokens.append((dec.attr, "decorator"))
        # Record base classes
        for base in node.bases:
            if isinstance(base, ast.Name):
                self.tokens.append((base.id, "class_instantiation"))
        self.generic_visit(node)

    # ── Assert comparisons ────────────────────────────────────────

    def visit_Assert(self, node: ast.Assert):
        # Pull names from the right-hand side of comparisons in asserts
        if isinstance(node.test, ast.Compare):
            for comp in node.test.comparators:
                for child in ast.walk(comp):
                    if isinstance(child, ast.Name):
                        self.tokens.append((child.id, "comparison_target"))
        self.generic_visit(node)

    # ── Bare Name loads ───────────────────────────────────────────

    def visit_Name(self, node: ast.Name):
        if isinstance(node.ctx, ast.Load) and len(node.id) > 1:
            self.tokens.append((node.id, "name_load"))
        self.generic_visit(node)


# ──────────────────── CORPUS-LEVEL ANALYSIS ───────────────────────

def extract_tokens_from_file(path: Path) -> List[Tuple[str, str]]:
    """Parse a single Python file and return (token, category) pairs."""
    try:
        source = path.read_text(encoding="utf-8")
        tree   = ast.parse(source, filename=str(path))
    except SyntaxError as e:
        print(f"[WARN] SyntaxError in {path}: {e}")
        return []
    extractor = DomainTokenExtractor()
    extractor.visit(tree)
    return extractor.tokens


def compute_corpus_weights(
    file_paths: List[Path],
    min_doc_freq: int = 1,
) -> Dict[str, float]:
    """
    TF-IDF over AST tokens across all reference files.

    For each token t:
      TF(t, d)  = (weighted count in doc d) / (total weighted tokens in d)
                  where weight = CATEGORY_BASE_WEIGHT[category]
      IDF(t)    = log((N + 1) / (df(t) + 1)) + 1       [smoothed]
      Final(t)  = max TF-IDF across documents * category_base_weight

    This means tokens that appear heavily in a few files AND belong to
    high-importance categories get the highest weights.
    """
    N = len(file_paths)

    # Per-document weighted counts and document frequency
    doc_weighted_counts: List[Dict[str, float]] = []
    doc_totals:          List[float]             = []
    doc_freq:            Counter                 = Counter()

    for path in file_paths:
        tokens = extract_tokens_from_file(path)
        weighted: Dict[str, float] = defaultdict(float)
        total = 0.0
        seen_in_doc = set()

        for (tok, cat) in tokens:
            w = CATEGORY_BASE_WEIGHT.get(cat, 1.0)
            weighted[tok] += w
            total += w
            seen_in_doc.add(tok)

        for tok in seen_in_doc:
            doc_freq[tok] += 1

        doc_weighted_counts.append(dict(weighted))
        doc_totals.append(max(total, 1.0))

    # Collect all tokens that meet min_doc_freq
    all_tokens = {t for t, df in doc_freq.items() if df >= min_doc_freq}

    # Compute TF-IDF
    token_scores: Dict[str, float] = {}
    for tok in all_tokens:
        idf = math.log((N + 1) / (doc_freq[tok] + 1)) + 1.0
        max_tf_idf = 0.0
        for d_idx in range(N):
            tf = doc_weighted_counts[d_idx].get(tok, 0.0) / doc_totals[d_idx]
            max_tf_idf = max(max_tf_idf, tf * idf)
        token_scores[tok] = max_tf_idf

    return token_scores


def normalise_weights(
    raw: Dict[str, float],
    scale_max: float = 3.0,
    floor: float = 0.1,
) -> Dict[str, float]:
    """
    Linear-scale raw TF-IDF scores into [floor, scale_max].
    Tokens at or below the median get floor; the top token gets scale_max.
    """
    if not raw:
        return {}
    values = sorted(raw.values())
    v_max  = values[-1]
    v_min  = values[0]
    span   = v_max - v_min if v_max != v_min else 1.0

    normalised = {}
    for tok, v in raw.items():
        normalised[tok] = floor + (v - v_min) / span * (scale_max - floor)
    return normalised


# ──────────────────── SYMPY SOURCE AUGMENTATION ───────────────────

def augment_from_sympy_source(
    sympy_src_root: Path,
    existing_weights: Dict[str, float],
    augment_weight: float = 1.5,
    max_files: int = 200,
) -> Dict[str, float]:
    """
    Optionally walk SymPy's own source tree to find more domain tokens.
    Tokens found in SymPy source but NOT in existing_weights get augment_weight.
    Tokens already in existing_weights are boosted by 10%.

    sympy_src_root: path to a local clone of sympy/sympy
    """
    if not sympy_src_root.exists():
        print(f"[INFO] SymPy source not found at {sympy_src_root}, skipping augmentation.")
        return existing_weights

    py_files = list(sympy_src_root.rglob("*.py"))[:max_files]
    print(f"[INFO] Augmenting from {len(py_files)} SymPy source files ...")

    sympy_freq: Counter = Counter()
    for path in py_files:
        for (tok, cat) in extract_tokens_from_file(path):
            if CATEGORY_BASE_WEIGHT.get(cat, 0) >= 1.5:   # only high-signal cats
                sympy_freq[tok] += 1

    augmented = dict(existing_weights)
    for tok, freq in sympy_freq.items():
        if freq < 3:           # ignore very rare tokens
            continue
        if tok in augmented:
            augmented[tok] *= 1.1
        else:
            augmented[tok] = augment_weight

    return augmented


# ─────────────────────── TOP-K DISPLAY ───────────────────────────

def top_k_weights(weights: Dict[str, float], k: int = 40) -> List[Tuple[str, float]]:
    return sorted(weights.items(), key=lambda x: x[1], reverse=True)[:k]


# ──────────────────────────── MAIN ───────────────────────────────

def build_domain_weights(
    config_path: Path = ROOT / "manual_corpus/benchmark_config.json",
    sympy_src: Path = Path("/tmp/sympy_src/sympy"),   # local sympy clone (optional)
    save_path: Path = ROOT / "domain_weights.json",
) -> Dict[str, float]:

    config = json.loads(config_path.read_text())
    items  = config["items"]

    # Collect all reference (_after) files
    ref_files: List[Path] = []
    for item_str in items:
        item_dir  = ROOT / "manual_corpus" /item_str
        meta      = json.loads((item_dir / "metadata.json").read_text())
        after_f   = item_dir / meta["after_file"]
        if after_f.exists():
            ref_files.append(after_f)
            print(f"[INFO] Added reference: {after_f.name}")

    print(f"\n[INFO] Building weights from {len(ref_files)} reference files ...")
    raw_weights  = compute_corpus_weights(ref_files, min_doc_freq=1)
    norm_weights = normalise_weights(raw_weights, scale_max=3.0, floor=0.1)

    # Optional: augment from SymPy source tree
    norm_weights = augment_from_sympy_source(sympy_src, norm_weights)

    # Save
    save_path.write_text(json.dumps(norm_weights, indent=2, sort_keys=True))
    print(f"[INFO] Saved {len(norm_weights)} token weights → {save_path}")

    # Show top tokens
    print("\n── Top 40 domain tokens by weight ──")
    print(f"{'Token':<35} {'Weight':>8}")
    print("-" * 45)
    for tok, w in top_k_weights(norm_weights, k=40):
        print(f"{tok:<35} {w:>8.4f}")

    return norm_weights


if __name__ == "__main__":
    build_domain_weights()