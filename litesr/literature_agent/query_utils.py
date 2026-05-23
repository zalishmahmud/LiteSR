"""
query_utils.py — Shared utilities for parsing researcher inputs.
"""

from __future__ import annotations

import re


def split_researcher_input(user_query: str) -> tuple[str, str]:
    """Split a researcher's input into a natural-language description and a code block.

    Scans for the first line that looks like Python code (import, def equation_,
    MAX_NPARAMS) and splits there.

    Returns
    -------
    (nl_description, code_block)
        nl_description : the natural-language portion (may be empty string)
        code_block     : the code portion (empty string if no code detected)
    """
    patterns = [r"^import\s", r"^def\s+equation_", r"^MAX_NPARAMS"]

    earliest_pos = len(user_query)
    for pattern in patterns:
        match = re.search(pattern, user_query, re.MULTILINE)
        if match and match.start() < earliest_pos:
            earliest_pos = match.start()

    if earliest_pos == len(user_query):
        return user_query.strip(), ""

    return user_query[:earliest_pos].strip(), user_query[earliest_pos:].strip()
