"""
subquery_prompt.py
------------------
Builds the subquery-generation prompt for the RAG equation search agent.

Usage
-----
    from subquery_prompt import build_subquery_prompt

    prompt = build_subquery_prompt(user_query)
    # pass `prompt` to your LLM call
"""

from __future__ import annotations

import re

try:
    from .query_utils import split_researcher_input as _split_researcher_input
except ImportError:
    from query_utils import split_researcher_input as _split_researcher_input  # type: ignore[no-redef]


def _highest_equation_version(code_block: str) -> int | None:
    versions = re.findall(r"def\s+equation_v(\d+)", code_block)
    return max(int(v) for v in versions) if versions else None


def _build_code_instruction(code_block: str) -> str:
    if not code_block:
        return ""

    highest = _highest_equation_version(code_block)
    only_v0 = highest is None or highest == 0

    version_rule = (
        "  - Only equation_v0 is present. Ignore all code entirely.\n"
        if only_v0
        else (
            f"  - The highest version present is equation_v{highest}. "
            "Use it ONLY as a\n"
            "    secondary signal to identify which terms or species are "
            "still active.\n"
            "    Do NOT treat its functional form as physically correct.\n"
        )
    )

    return (
        "If the input contains code, follow these rules strictly:\n"
        "  - Functions named 'equation_v0' are STARTING SKELETONS ONLY.\n"
        "    They may contain incorrect physics. DO NOT infer math from them.\n"
        + version_rule
        + "  - The NATURAL LANGUAGE DESCRIPTION is always authoritative.\n"
        "    NL always wins any conflict with code.\n"
        "  - Never include code syntax, variable names, or programming terms.\n\n"
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_subquery_prompt(user_query: str) -> str:
    """
    Build the full subquery-generation prompt from a raw researcher input.

    Parameters
    ----------
    user_query : str
        The complete researcher input — may contain a natural-language
        description, code blocks, or both.

    Returns
    -------
    str
        A fully-formed prompt ready to send to the LLM.
    """
    nl_description, code_block = _split_researcher_input(user_query)
    code_instruction = _build_code_instruction(code_block)

    spec_section = (
        "=== SPECIFICATION (authoritative) ===\n"
        f"{nl_description}\n\n"
    )

    code_section = (
        "=== IMPLEMENTATION ATTEMPTS (informational only — may contain errors) ===\n"
        f"{code_block}\n\n"
        if code_block
        else ""
    )

    prompt = (
        "You are a scientific equation search assistant.\n"
        "A researcher needs to find mathematical equations from scientific papers.\n\n"
        "Their input may be:\n"
        "  a) A natural language question about equations or a physical/mathematical problem.\n"
        "  b) Python code or pseudocode that implements a model or equation.\n\n"
        + code_instruction
        + "Your task: decompose the underlying mathematical problem into 2-5 short, "
        "focused sub-queries. Each sub-query must target a single distinct equation, "
        "formula, or mathematical concept from the relevant scientific domain.\n\n"
        "Rules:\n"
        "- Return between 2 and 5 sub-queries. Never exceed 5.\n"
        "- Each sub-query must be a concise keyword phrase (under 10 words).\n"
        "- Use standard mathematical / scientific terminology only — no code terms.\n"
        "- Focus on equation names, model names, physical law names, and domain terms.\n"

        # SPECIFICITY — domain-agnostic examples
        "- Sub-queries must be keyword-dense and specific enough to retrieve a single "
        "named formula. Use the name of the law, the key variables, and the "
        "mathematical relationship — not a description of what it does.\n"
        "  BAD : 'how forces affect motion over time'\n"
        "  GOOD: 'Newton second law force mass acceleration'\n"
        "  BAD : 'how a quantity grows and saturates'\n"
        "  GOOD: 'Michaelis-Menten saturation kinetics substrate concentration'\n"
        "  BAD : 'decay of a quantity over time'\n"
        "  GOOD: 'exponential decay rate constant half-life ODE'\n"

        # SHARED STRUCTURE — one query per law, not one per term
        "- If multiple terms share the same mathematical structure "
        "(e.g. all coefficients follow the same law), emit ONE sub-query "
        "for that shared structure — not one per term.\n"

        # STANDALONE — detach from parent equation/domain
        "- Query each mathematical concept as a standalone named formula. "
        "Do NOT attach a concept to the parent equation, system name, or domain. "
        "Ask yourself: could this sub-query retrieve the right formula "
        "with no surrounding context? If not, rewrite it.\n"
        "  BAD : 'parameter dependence on temperature for all terms in model'\n"
        "  GOOD: 'temperature dependent coefficient exponential inverse absolute temperature'\n"

        # NO OVERLAP
        "- Do NOT include sub-queries that overlap heavily with each other.\n"

        # NO TRAILING ANCHORS
        "- Do NOT end a sub-query with the system name, equation name, or "
        "trailing phrases like 'for all terms', 'in the system', "
        "'for the model', 'of the equation'.\n"

        "- Return ONLY a JSON array of strings. No explanation, no markdown.\n\n"
        + spec_section
        + code_section
        + "JSON array of equation-focused sub-queries:"
    )

    return prompt