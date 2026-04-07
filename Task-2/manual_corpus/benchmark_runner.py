import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from datetime import datetime
from typing import List, Tuple

ROOT = Path(__file__).resolve().parent
CONFIG_FILE = ROOT / "benchmark_config.json"
RESULTS_FILE = ROOT / "benchmark_results.json"


def log(msg: str):
    print(f"[INFO] {msg}")


def log_error(msg: str):
    print(f"[ERROR] {msg}")


def run_cmd(cmd, cwd=None, env=None, capture=True, step_name="command"):
    log(f"Running {step_name}: {' '.join(cmd)}")
    if cwd:
        log(f"  cwd = {cwd}")
    result = subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        text=True,
        capture_output=capture,
    )
    log(f"Finished {step_name} with return code {result.returncode}")
    return result


# ─────────────────────── TEST FILE LOADER ─────────────────────────

def load_test_files(item_dir: Path, meta: dict) -> List[Path]:
    """
    Supports both old single 'test_file' and new 'test_files' list.
    All paths are relative to item_dir.
    Raises clearly if any file is missing.
    """
    if "test_files" in meta:
        paths = [item_dir / f for f in meta["test_files"]]
    elif "test_file" in meta:
        paths = [item_dir / meta["test_file"]]
    else:
        raise KeyError(
            f"metadata.json at {item_dir} must have 'test_file' or 'test_files'"
        )

    missing = [p for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Test file(s) not found: {[str(p) for p in missing]}"
        )

    return paths


# ─────────────────────── INSTALL / CLONE ──────────────────────────

def install_repo(repo_dir: Path):
    log(f"Installing dependencies for repo: {repo_dir}")

    log("Upgrading pip/setuptools/wheel ...")
    pip_upgrade = run_cmd(
        [sys.executable, "-m", "pip", "install", "--upgrade",
         "pip", "setuptools", "wheel"],
        step_name="pip upgrade"
    )
    if pip_upgrade.returncode != 0:
        log_error("pip upgrade failed (continuing anyway)")

    log("Installing pytest ...")
    pytest_res = run_cmd(
        [sys.executable, "-m", "pip", "install", "pytest"],
        step_name="install pytest"
    )
    if pytest_res.returncode != 0:
        log_error("pytest installation failed")
        return False, pytest_res

    log("Installing repository in editable mode: pip install -e .")
    install_res = run_cmd(
        [sys.executable, "-m", "pip", "install", "-e", "."],
        cwd=repo_dir,
        step_name="pip install -e ."
    )
    if install_res.returncode != 0:
        log_error("Editable install failed")
        return False, install_res

    log("Repository installation completed successfully")
    return True, install_res


def clone_repo(repo_url: str, checkout: str, workdir: Path):
    repo_dir = workdir / "repo"

    log(f"Cloning repository from {repo_url}")
    log(f"Temporary working directory: {workdir}")
    clone_res = run_cmd(
        ["git", "clone", repo_url, str(repo_dir)],
        step_name="git clone"
    )
    if clone_res.returncode != 0:
        log_error("git clone failed")
        return None, clone_res

    if checkout:
        log(f"Checking out revision: {checkout}")
        checkout_res = run_cmd(
            ["git", "checkout", checkout],
            cwd=repo_dir,
            step_name="git checkout"
        )
        if checkout_res.returncode != 0:
            log_error(f"git checkout failed for revision: {checkout}")
            return None, checkout_res
        log(f"Checkout successful: {checkout}")
        return repo_dir, checkout_res

    return repo_dir, clone_res


# ─────────────────────── VARIANT RUNNER ───────────────────────────

def run_one_variant(
    item_dir:       Path,
    repo_dir:       Path,
    target_relpath: str,
    candidate_file: Path,
    test_files:     List[Path],          # ← now a list
    label:          str,
) -> dict:
    log(f"Preparing to run variant: {label}")

    target_path = repo_dir / target_relpath
    log(f"Target file inside cloned repo: {target_path}")
    log(f"Candidate replacement file: {candidate_file}")
    log(f"Test files to execute: {[str(f) for f in test_files]}")

    # ── Sanity checks ──────────────────────────────────────────────
    if not target_path.exists():
        msg = f"Target file not found: {target_path}"
        log_error(msg)
        return _error_result(label, msg)

    if not candidate_file.exists():
        msg = f"Candidate file not found: {candidate_file}"
        log_error(msg)
        return _error_result(label, msg)

    # ── Backup original ────────────────────────────────────────────
    backup_path = target_path.with_suffix(target_path.suffix + ".bak")
    log(f"Backing up original target file -> {backup_path}")
    shutil.copy2(target_path, backup_path)

    # Accumulated results across all test files
    all_passed   = True
    per_file_results = []
    combined_stdout  = []
    combined_stderr  = []

    try:
        # ── Swap in candidate ──────────────────────────────────────
        log(f"Replacing target file with {label} candidate")
        shutil.copy2(candidate_file, target_path)

        # ── Build env ──────────────────────────────────────────────
        env = os.environ.copy()
        existing = env.get("PYTHONPATH", "")
        repo_pythonpath = str(repo_dir)
        env["PYTHONPATH"] = (
            repo_pythonpath if not existing
            else repo_pythonpath + os.pathsep + existing
        )
        log(f"Using PYTHONPATH={env['PYTHONPATH']}")

        # ── Run each test file ─────────────────────────────────────
        for test_file in test_files:
            log(f"Running pytest on: {test_file.name}")

            cmd = [
                sys.executable, "-m", "pytest",
                str(test_file),
                "-v",           # one line per test
                "--tb=short",   # compact tracebacks
                "-q",           # keep summary short
            ]

            res = run_cmd(
                cmd,
                cwd=ROOT,
                env=env,
                capture=True,
                step_name=f"pytest [{label}] {test_file.name}",
            )

            file_passed = res.returncode == 0
            all_passed  = all_passed and file_passed

            # Print per-file output block
            print("\n" + "=" * 40)
            print(f"PYTEST STDOUT [{label}] [{test_file.name}]")
            print("=" * 40)
            print(res.stdout if res.stdout else "(no stdout)")

            if res.stderr:
                print("\n" + "=" * 40)
                print(f"PYTEST STDERR [{label}] [{test_file.name}]")
                print("=" * 40)
                print(res.stderr)
            print("=" * 40 + "\n")

            file_status = "PASS" if file_passed else "FAIL"
            log(f"  {test_file.name} → {file_status}")

            # Collect failed test names for this file
            failed_tests = _extract_failed_test_names(res.stdout)
            if failed_tests:
                log(f"  Failed tests in {test_file.name}:")
                for t in failed_tests:
                    log(f"    ✗ {t}")

            per_file_results.append({
                "test_file":    test_file.name,
                "status":       file_status,
                "returncode":   res.returncode,
                "failed_tests": failed_tests,
                "stdout":       res.stdout,
                "stderr":       res.stderr,
            })

            combined_stdout.append(
                f"[{test_file.name}]\n{res.stdout}"
            )
            if res.stderr:
                combined_stderr.append(
                    f"[{test_file.name}]\n{res.stderr}"
                )

        overall_status = "PASS" if all_passed else "FAIL"
        log(f"Variant {label} overall status: {overall_status} "
            f"({sum(1 for r in per_file_results if r['status']=='PASS')}/"
            f"{len(per_file_results)} files passed)")

        return {
            "label":            label,
            "status":           overall_status,
            "all_files_passed": all_passed,
            "per_file":         per_file_results,
            # Flattened for backward compatibility
            "returncode":       0 if all_passed else 1,
            "stdout":           "\n\n".join(combined_stdout),
            "stderr":           "\n\n".join(combined_stderr),
        }

    finally:
        if backup_path.exists():
            log(f"Restoring original target file from backup")
            shutil.move(backup_path, target_path)
            log(f"Restore complete for variant: {label}")


def _error_result(label: str, msg: str) -> dict:
    return {
        "label":            label,
        "status":           "ERROR",
        "reason":           msg,
        "all_files_passed": False,
        "per_file":         [],
        "returncode":       None,
        "stdout":           "",
        "stderr":           "",
    }


def _extract_failed_test_names(stdout: str) -> List[str]:
    """
    Pull failed test names from pytest output.
    Matches lines like:
      FAILED test_eigen.py::test_float_eigenvals - AssertionError
    """
    failed = []
    for line in stdout.splitlines():
        line = line.strip()
        if line.startswith("FAILED"):
            # "FAILED path::test_name - reason"  →  "test_name"
            parts = line.split("::")
            if len(parts) >= 2:
                test_name = parts[-1].split(" - ")[0].strip()
                failed.append(test_name)
            else:
                failed.append(line)
    return failed


# ─────────────────────────── ITEM RUNNER ──────────────────────────

def run_item(item_path_str: str) -> dict:
    log("=" * 80)
    log(f"Starting benchmark item: {item_path_str}")
    log("=" * 80)

    item_dir  = ROOT / item_path_str
    meta_path = item_dir / "metadata.json"

    log(f"Item directory: {item_dir}")
    log(f"Metadata path:  {meta_path}")

    if not meta_path.exists():
        msg = f"Missing metadata.json at {meta_path}"
        log_error(msg)
        return {"item": item_path_str, "status": "ERROR", "reason": msg}

    log("Reading metadata.json ...")
    meta = json.loads(meta_path.read_text())

    name           = meta["name"]
    repo_url       = meta["repo_url"]
    checkout       = meta.get("checkout", "master")
    target_relpath = meta["target_relpath"]
    before_file    = item_dir / meta["before_file"]
    after_file     = item_dir / meta["after_file"]

    # ── Load test files (single or multiple) ──────────────────────
    try:
        test_files = load_test_files(item_dir, meta)
    except (KeyError, FileNotFoundError) as e:
        log_error(str(e))
        return {"item": item_path_str, "status": "ERROR", "reason": str(e)}

    log("Parsed metadata:")
    log(f"  name           = {name}")
    log(f"  repo_url       = {repo_url}")
    log(f"  checkout       = {checkout}")
    log(f"  target_relpath = {target_relpath}")
    log(f"  before_file    = {before_file}")
    log(f"  after_file     = {after_file}")
    log(f"  test_files     = {[f.name for f in test_files]}")

    result = {
        "item":           item_path_str,
        "name":           name,
        "repo_url":       repo_url,
        "checkout":       checkout,
        "target_relpath": target_relpath,
        "test_files":     [f.name for f in test_files],
        "before":         None,
        "after":          None,
        "status":         "OK",
    }

    # ── BEFORE ────────────────────────────────────────────────────
    log(f"Starting BEFORE run for item: {name}")
    with tempfile.TemporaryDirectory(prefix=f"{name}_before_") as tmp:
        workdir = Path(tmp)
        repo_dir, clone_res = clone_repo(repo_url, checkout, workdir)

        if repo_dir is None:
            log_error("BEFORE run failed during clone/checkout")
            result["before"] = {
                "label": "before", "status": "ERROR",
                "returncode": clone_res.returncode,
                "stdout": clone_res.stdout, "stderr": clone_res.stderr,
            }
        else:
            ok, install_res = install_repo(repo_dir)
            if not ok:
                log_error("BEFORE run failed during install")
                result["before"] = {
                    "label": "before", "status": "ERROR",
                    "reason": "install_failed",
                    "returncode": install_res.returncode,
                    "stdout": install_res.stdout, "stderr": install_res.stderr,
                }
            else:
                result["before"] = run_one_variant(
                    item_dir=item_dir,
                    repo_dir=repo_dir,
                    target_relpath=target_relpath,
                    candidate_file=before_file,
                    test_files=test_files,
                    label="before",
                )

    log(f"Completed BEFORE run for item: {name}")

    # ── AFTER ─────────────────────────────────────────────────────
    log(f"Starting AFTER run for item: {name}")
    with tempfile.TemporaryDirectory(prefix=f"{name}_after_") as tmp:
        workdir = Path(tmp)
        repo_dir, clone_res = clone_repo(repo_url, checkout, workdir)

        if repo_dir is None:
            log_error("AFTER run failed during clone/checkout")
            result["after"] = {
                "label": "after", "status": "ERROR",
                "returncode": clone_res.returncode,
                "stdout": clone_res.stdout, "stderr": clone_res.stderr,
            }
        else:
            ok, install_res = install_repo(repo_dir)
            if not ok:
                log_error("AFTER run failed during install")
                result["after"] = {
                    "label": "after", "status": "ERROR",
                    "reason": "install_failed",
                    "returncode": install_res.returncode,
                    "stdout": install_res.stdout, "stderr": install_res.stderr,
                }
            else:
                result["after"] = run_one_variant(
                    item_dir=item_dir,
                    repo_dir=repo_dir,
                    target_relpath=target_relpath,
                    candidate_file=after_file,
                    test_files=test_files,
                    label="after",
                )

    log(f"Completed AFTER run for item: {name}")

    before_status = result.get("before", {}).get("status", "N/A")
    after_status  = result.get("after",  {}).get("status", "N/A")
    log(f"Final item summary: before={before_status}, after={after_status}")

    return result


# ─────────────────────────────── MAIN ─────────────────────────────

def main():
    log("Starting benchmark runner")
    log(f"Root directory: {ROOT}")
    log(f"Config file:    {CONFIG_FILE}")
    log(f"Results file:   {RESULTS_FILE}")

    if not CONFIG_FILE.exists():
        log_error(f"Missing config file: {CONFIG_FILE}")
        sys.exit(1)

    log("Reading benchmark_config.json ...")
    config = json.loads(CONFIG_FILE.read_text())
    items  = config["items"]
    log(f"Loaded {len(items)} benchmark item(s)")

    all_results = {
        "timestamp": datetime.now().isoformat(),
        "results":   [],
    }

    for idx, item in enumerate(items, start=1):
        log("")
        log("#" * 80)
        log(f"Running item {idx}/{len(items)}: {item}")
        log("#" * 80)

        res = run_item(item)
        all_results["results"].append(res)

        before_status = res.get("before", {}).get("status", "N/A")
        after_status  = res.get("after",  {}).get("status", "N/A")

        # Per-file summary
        print("\n" + "=" * 80)
        print(f"SUMMARY FOR {item}")
        print(f"  BEFORE overall : {before_status}")
        for pf in res.get("before", {}).get("per_file", []):
            failed = pf.get("failed_tests", [])
            mark   = "✓" if pf["status"] == "PASS" else "✗"
            print(f"    {mark} {pf['test_file']:<40} {pf['status']}")
            for t in failed:
                print(f"        ✗ {t}")

        print(f"  AFTER  overall : {after_status}")
        for pf in res.get("after", {}).get("per_file", []):
            failed = pf.get("failed_tests", [])
            mark   = "✓" if pf["status"] == "PASS" else "✗"
            print(f"    {mark} {pf['test_file']:<40} {pf['status']}")
            for t in failed:
                print(f"        ✗ {t}")
        print("=" * 80 + "\n")

    log("Writing benchmark results to JSON ...")
    RESULTS_FILE.write_text(json.dumps(all_results, indent=2))
    log(f"Saved results to: {RESULTS_FILE}")
    log("Benchmark runner finished successfully")


if __name__ == "__main__":
    main()