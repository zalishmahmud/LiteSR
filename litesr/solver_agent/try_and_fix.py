"""
try_and_fix.py — Self-healing function executor.

On exception:
    1. Captures the error + traceback
    2. Sends the broken program + error to local LLM (via Ollama)
    3. Re-runs the fixed program once
    4. If still fails — gives up and returns (None, False)
"""

import os
import re
import ast
import textwrap
import traceback
from litesr.clients import get_ollama_client

# ── Ollama client (local LLM) ─────────────────────────────────────────────────
OLLAMA_URL = "http://localhost:11434/v1"
FIX_MODEL  = "mistral:latest"

_llm = get_ollama_client(OLLAMA_URL)

FIX_PROMPT = """\
You are an expert Python debugger. The following Python program raised an exception.
Fix ONLY the bug — do not change logic, variable names, or function signatures.

Rules:
- Return the COMPLETE corrected program, not just the fixed function.
- Do NOT include any explanation, preamble, or commentary.
- Do NOT wrap the code in markdown fences (no ```python, no ```).
- The very first character of your response must be the start of valid Python code.

--- PROGRAM ---
{program}

--- ERROR ---
{error}

Complete corrected program:"""


# ── Helper: strip prose/fences from LLM output ───────────────────────────────

def _extract_code(raw: str) -> str:
    """Pull clean Python out of LLM output that may contain prose or markdown fences."""
    fence_match = re.search(r"```(?:python)?\s*\n(.*?)```", raw, re.DOTALL)
    if fence_match:
        return fence_match.group(1).strip()

    PYTHON_STARTS = (
        "import ", "from ", "def ", "class ", "async def ", "@",
        "#", "if ", "for ", "while ", "try:", "with ",
        "return ", "raise ", "pass", "break", "continue",
        "yield ", "lambda ",
    )
    lines = raw.splitlines()
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        if any(stripped.startswith(kw) for kw in PYTHON_STARTS):
            return "\n".join(lines[i:]).strip()

    return raw.strip()


# ── Helper: fix indentation in LLM-returned code ─────────────────────────────

def _sanitize_indentation(program: str) -> str:
    """Re-indent function bodies to exactly 4 spaces."""
    try:
        ast.parse(program)
        return program
    except IndentationError:
        pass
    except SyntaxError:
        return program

    def _reindent_bodies(src: str) -> str:
        lines = src.splitlines(keepends=True)
        out = []
        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.lstrip()
            if stripped.startswith(("def ", "class ", "async def ")):
                base_indent = len(line) - len(stripped)
                out.append(line)
                i += 1
                body = []
                while i < len(lines):
                    bl  = lines[i]
                    bls = bl.lstrip()
                    if bls.rstrip() == "":
                        body.append(bl)
                        i += 1
                        continue
                    cur = len(bl) - len(bls)
                    if cur <= base_indent:
                        break
                    body.append(bl)
                    i += 1
                body_text = textwrap.dedent("".join(body))
                for bl in body_text.splitlines(keepends=True):
                    out.append(" " * (base_indent + 4) + bl.lstrip())
            else:
                out.append(line)
                i += 1
        return "".join(out)

    fixed = _reindent_bodies(program)
    try:
        ast.parse(fixed)
        return fixed
    except Exception:
        return program


# ── Internal: call LLM and return cleaned+sanitized program ──────────────────

def _ask_llm_to_fix(program: str, error: str) -> str:
    """Send broken program + error to local LLM, return fixed program string."""
    prompt   = FIX_PROMPT.format(program=program, error=error)
    response = _llm.chat.completions.create(
        model=FIX_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    raw = response.choices[0].message.content.strip()
    return _sanitize_indentation(_extract_code(raw))


# ── Internal: single execution attempt ───────────────────────────────────────

def _execute_once(
    program:            str,
    function_to_run:    str,
    function_to_evolve: str,
    dataset,
    numba_accelerate:   bool,
    attempt:            int = 1,
) -> tuple:
    """
    Try to exec `program` and call `function_to_run(dataset)`.
    Returns (result, True, None) on success.
    Returns (None, False, traceback_str) on failure.
    """
    try:
        if numba_accelerate:
            from . import evaluator_accelerate
            program = evaluator_accelerate.add_numba_decorator(
                program=program,
                function_to_evolve=function_to_evolve,
            )

        namespace = {}
        exec(program, namespace)
        fn     = namespace[function_to_run]
        result = fn(dataset)

        if not isinstance(result, (int, float)):
            msg = f"Return value is not int or float — got {type(result).__name__}"
            return None, False, msg

        return result, True, None

    except Exception:
        tb = traceback.format_exc()
        return None, False, tb


# ── Public: called from evaluator._compile_and_run_function ──────────────────

def _is_fixable(program: str) -> tuple[bool, str]:
    """
    Return (fixable, reason) before wasting an LLM call.

    Un-fixable cases: empty function bodies or placeholder markers.
    """
    PLACEHOLDER_MARKERS = (
        "# your existing code",
        "# ... your",
        "... (your existing",
        "# replace this",
        "# add your",
        "pass  # ",
    )
    lower = program.lower()
    for marker in PLACEHOLDER_MARKERS:
        if marker in lower:
            return False, f"Program contains placeholder marker: '{marker}'"

    try:
        tree = ast.parse(program)
    except SyntaxError:
        return True, "syntax error — worth trying LLM fix"

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        real_stmts = []
        for stmt in node.body:
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str):
                continue
            if isinstance(stmt, ast.Pass):
                continue
            real_stmts.append(stmt)
        if not real_stmts:
            return False, f"Function '{node.name}' has no real implementation (only docstrings/pass)"

    return True, "ok"


def ask_llm_to_fix(program: str, error: Exception) -> str | None:
    """
    Entry point called directly from evaluator on exception.
    Returns fixed program string, or None if not fixable.
    """
    fixable, reason = _is_fixable(program)
    if not fixable:
        print(f"[try_and_fix] Skipping LLM fix — not fixable: {reason}")
        return None

    print(f"[try_and_fix] Exception caught — asking LLM to fix ({FIX_MODEL})…")
    try:
        fixed = _ask_llm_to_fix(program, traceback.format_exc())
        print("[try_and_fix] LLM returned a fix — retrying execution…")
        return fixed
    except Exception as llm_err:
        print(f"[try_and_fix] LLM call failed: {llm_err} — giving up.")
        return None


# ── Public: standalone full pipeline ─────────────────────────────────────────

def compile_and_run_with_fix(
    program:            str,
    function_to_run:    str,
    function_to_evolve: str,
    dataset,
    numba_accelerate:   bool = False,
) -> tuple[object, bool]:
    """
    Run the program; if it fails, ask the local LLM to fix it and retry once.
    Returns (result, success).
    """
    result, ok, error = _execute_once(
        program, function_to_run, function_to_evolve, dataset, numba_accelerate,
        attempt=1,
    )
    if ok:
        return result, True

    print("[try_and_fix] Attempt 1 failed — asking LLM to fix…")
    try:
        fixed_program = _ask_llm_to_fix(program, error)
        print("[try_and_fix] LLM returned a fix — retrying…")
    except Exception as llm_err:
        print(f"[try_and_fix] LLM call failed: {llm_err} — giving up.")
        return None, False

    result, ok, error = _execute_once(
        fixed_program, function_to_run, function_to_evolve, dataset, numba_accelerate,
        attempt=2,
    )
    if ok:
        print("[try_and_fix] Fixed program ran successfully.")
        return result, True

    print("[try_and_fix] Fixed program also failed — giving up.")
    return None, False


# ── Drop-in multiprocessing wrapper ──────────────────────────────────────────

def compile_and_run_with_fix_mp(
    program:            str,
    function_to_run:    str,
    function_to_evolve: str,
    dataset,
    numba_accelerate:   bool,
    result_queue,
) -> None:
    """Same as compile_and_run_with_fix() but puts result into a Queue."""
    result, ok = compile_and_run_with_fix(
        program, function_to_run, function_to_evolve, dataset, numba_accelerate
    )
    result_queue.put((result, ok))
