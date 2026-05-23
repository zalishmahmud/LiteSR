"""
equation_indexer.py — Equation-only indexing pipeline.

Extracts mathematical equations (with descriptions) from PDF pages using:
  1. Claude / Ollama vision OCR on each rendered page image
  2. LaTeX pattern regex on pdfplumber text layer (fallback / supplement)

Stores equation+description pairs in Milvus collection `equation_index`,
separate from the full-text `rag_documents` collection used by RAGPipeline.

Usage:
    from litesr.literature_agent.equation_indexer import EquationIndexer

    indexer = EquationIndexer()
    indexer.index("./papers")

    hits = indexer.get_equation_context("damped harmonic oscillator with nonlinear term")
    for h in hits:
        print(h["equation"], "|", h["description"], "|", h["source"])
"""

from __future__ import annotations

import base64
import io
import json
import re
import time
from pathlib import Path
from typing import Union

import pdfplumber
from pymilvus import MilvusClient

from litesr.clients import get_ollama_client, llm_call, llm_call_vision, llm_call_pdf_document

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_MILVUS_URI     = "http://localhost:19530"
DEFAULT_OLLAMA_URL     = "http://localhost:11434/v1"
DEFAULT_EMBED_MODEL    = "nomic-embed-text"
DEFAULT_COLLECTION     = "equation_index_rseos" # "equation_index" has hco3 context
DEFAULT_TOP_K          = 5
PAGE_RENDER_RESOLUTION = 150   # DPI for page → image rendering

# eq_type filter sets — used in get_equation_context()
_PREFERRED_TYPES = {"ODE", "definition", "equilibrium", "conservation"}
_BLOCKED_TYPES   = {"asymptotic", "dimensionless"}
_VALID_EQ_TYPES  = {
    "ODE", "conservation", "equilibrium",
    "definition", "asymptotic", "dimensionless",
}

# LaTeX / math patterns to scan in the text layer
_LATEX_PATTERNS = [
    re.compile(r'\$\$(.+?)\$\$', re.DOTALL),
    re.compile(r'\$([^\$\n]{3,}?)\$'),
    re.compile(r'\\begin\{equation\*?\}(.+?)\\end\{equation\*?\}', re.DOTALL),
    re.compile(r'\\begin\{align\*?\}(.+?)\\end\{align\*?\}', re.DOTALL),
    re.compile(r'\\begin\{eqnarray\*?\}(.+?)\\end\{eqnarray\*?\}', re.DOTALL),
    re.compile(r'((?:\\frac|\\sum|\\int|\\prod|\\lim|\\partial|\\nabla|\\Delta)'
               r'[^\n]{5,80})'),
]

_VISION_PROMPT = """\
You are a mathematical equation extractor. Look at this PDF page image.

Extract every distinct mathematical equation, formula, or expression visible on the page — \
including those in figures, tables, or image regions.

For each equation:
  - Write it in LaTeX if possible, otherwise plain text.
  - Write a description that contains: (1) what the full equation represents, followed by \
(2) a term-by-term breakdown explaining what each part does. \
This helps retrieve the equation even when only one term is searched for.

Description format: "<overall name>. Terms: <term1> = <role>, <term2> = <role>, ..."

Return ONLY a valid JSON array. No prose, no markdown fences. Example:
[
  {
    "equation": "\\ddot{x} + 2\\gamma\\dot{x} + \\omega_0^2 x = F_0\\cos(\\omega t)",
    "description": "driven damped harmonic oscillator. Terms: \\ddot{x} = acceleration, \
2\\gamma\\dot{x} = linear damping force, \\omega_0^2 x = restoring force, \
F_0\\cos(\\omega t) = periodic driving force"
  }
]

If no equations are present on this page, return an empty array: []
"""

_DOCUMENT_PROMPT = """\
You are a mathematical equation extractor. Look at this entire PDF document (all pages).

Extract every distinct mathematical equation, formula, or expression visible in the document — \
including those in figures, tables, or image regions.

For each equation:
  - Write it in LaTeX exactly as typeset in the source document.
  - CRITICAL — expansion rule: Always write coefficient expressions in fully \
expanded form with explicit values. Never use a factored form that hides individual \
coefficients. For example:
      CORRECT  : 3x^2 - 2y + \\frac{1}{2}z
      INCORRECT: \\frac{1}{2}(6x^2 - 4y + z)
    This applies to ALL equations — exponential arguments, polynomial terms, \
numerators, denominators, and any grouped expression.
  - Write a description that contains: (1) what the full equation represents, followed by \
(2) a term-by-term breakdown explaining what each part does. \
This helps retrieve the equation even when only one term is searched for.

Description format: "<overall name> in <domain>. Terms: <term1> = <role>, <term2> = <role>, ..."

Return ONLY a valid JSON array. No prose, no markdown fences. Example:
[
  {
    "equation": "E = \\frac{1}{2} m v^2 + m g h",
    "description": "total mechanical energy in classical mechanics. \
Terms: E = total mechanical energy (joules), \
\\frac{1}{2}mv^2 = kinetic energy term proportional to mass and velocity squared, \
mgh = gravitational potential energy proportional to mass height and gravitational acceleration"
  }
]

If no equations are present, return an empty array: []
"""

# ── UPDATED: now requires eq_type field ───────────────────────────────────────
_CHUNKS_PROMPT = """\
You are a mathematical equation extractor. Below are text excerpts from a scientific paper.

Extract every distinct mathematical equation, formula, or expression present in the text.

For each equation write THREE fields:

"equation" — the equation in LaTeX (or plain text if LaTeX is not possible).
  NOTATION RULES — critical, follow exactly:
  - Copy every symbol EXACTLY as it appears in the source text.
  - Do NOT add subscripts, superscripts, or decorations that are not
    in the source. If the source writes Y, write Y — not Y_{{-3}} or Y_i.
  - Do NOT rename or disambiguate symbols across equations. If the same
    symbol appears in multiple equations, use the same symbol in all of them.
  - Do NOT invent notation to make equations look more systematic.

"eq_type" — classify as exactly one of:
  "ODE"            — a differential equation governing a species over time (d[X]/dt = ...)
  "conservation"   — a mass/charge/atom balance identity (sum = constant)
  "equilibrium"    — an equilibrium or rate constant expression (K = ...)
  "definition"     — defines a variable or auxiliary quantity (X ≡ ...)
  "asymptotic"     — an approximation valid only in a specific time scale or limit
  "dimensionless"  — a dimensionless parameter definition or scaled ODE

"description" — a rich natural-language description used for semantic search. It must include:
  1. The specific scientific domain and context where this equation applies.
  2. What the equation governs or predicts.
  3. A term-by-term breakdown: what each symbol represents, its units if known, and its role.
  If eq_type is "asymptotic", you MUST state which time scale it applies to and
  that it is NOT the governing ODE.
  DESCRIPTION NOTATION RULES:
  - Describe each symbol using the SAME notation as in the source equation.
  - Do NOT introduce subscripted variants in the description that do not
    appear in the source (e.g. if the source has Y, describe it as Y —
    not as Y_{{-3}} or Y_{{stoich}}).
  - If the same symbol appears in multiple equations with the same physical
    meaning, state that explicitly: e.g. "Y = proton concentration (mol/L),
    same symbol used across all reaction rate equations in this system".

Description format:
"<eq_type> in <domain context>. <what it governs>. Terms: <sym1> = <meaning and role>, ..."

Return ONLY a valid JSON array of objects with keys: equation, eq_type, description.
No prose, no markdown fences. If no equations are found, return: []

Text excerpts:
{context}
"""
# ── Generic equation discovery queries — work across any scientific domain ────
# These are intentionally broad so the pre-filter populates equation_index
# with all equation types regardless of the target domain.
# Swap these out for domain-specific queries during debugging if needed.
_EQUATION_SEARCH_QUERIES = [
    "differential equation rate of change governing species",
    "mass action kinetics reaction rate constant expression",
    "equilibrium constant dissociation formula chemical reaction",
    "conservation law mass balance identity constraint",
    "Arrhenius temperature dependent rate constant activation energy",
]


class EquationIndexer:
    """Equation-only indexer: extracts and stores PDF equations in Milvus `equation_index`."""

    def __init__(
        self,
        milvus_uri:      str  = DEFAULT_MILVUS_URI,
        ollama_url:      str  = DEFAULT_OLLAMA_URL,
        embedding_model: str  = DEFAULT_EMBED_MODEL,
        use_ollama:      bool = False,
        ollama_model:    str  = "mistral:latest",
        vision_model:    str  = "llava:latest",
        anthropic_model: str  = "claude-sonnet-4-6",
        collection_name: str  = DEFAULT_COLLECTION,
        top_k:           int  = DEFAULT_TOP_K,
    ):
        self._embed_client    = get_ollama_client(ollama_url)
        self._embed_model     = embedding_model
        self._use_ollama      = use_ollama
        self._ollama_model    = ollama_model
        self._vision_model    = vision_model
        self._anthropic_model = anthropic_model
        self._collection      = collection_name
        self._top_k           = top_k
        self._milvus          = self._connect_milvus(milvus_uri)
        backend = f"Ollama ({vision_model})" if use_ollama else f"Anthropic ({anthropic_model})"
        print(f"[EquationIndexer] Vision backend : {backend}")
        print(f"[EquationIndexer] Embedding      : {embedding_model} via Ollama")
        print(f"[EquationIndexer] Collection     : {collection_name}")

    @property
    def collection_name(self) -> str:
        return self._collection

    # ── Milvus helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _connect_milvus(uri: str, max_wait: float = 120.0, interval: float = 3.0) -> MilvusClient:
        from pymilvus.exceptions import MilvusException
        client   = MilvusClient(uri=uri)
        deadline = time.monotonic() + max_wait
        while True:
            try:
                client.list_collections()
                return client
            except MilvusException as exc:
                if time.monotonic() > deadline:
                    raise RuntimeError(f"Milvus at {uri} not ready after {max_wait}s") from exc
                time.sleep(interval)

    def _get_embedding_dim(self) -> int:
        resp = self._embed_client.embeddings.create(model=self._embed_model, input=["test"])
        return len(resp.data[0].embedding)

    def _ensure_collection(self, force_update: bool = False) -> None:
        exists = self._collection in self._milvus.list_collections()
        if exists and not force_update:
            # Schema migration: rebuild if 'text' or 'eq_type' fields are missing
            try:
                sample = self._milvus.query(
                    collection_name = self._collection,
                    filter          = "",
                    output_fields   = ["text", "eq_type"],
                    limit           = 1,
                )
                if sample and ("text" not in sample[0] or "eq_type" not in sample[0]):
                    print(f"[EquationIndexer] Collection '{self._collection}' missing "
                          f"'text' or 'eq_type' field — rebuilding…")
                    force_update = True
            except Exception:
                # Field doesn't exist at all — rebuild
                force_update = True

        if exists and force_update:
            self._milvus.drop_collection(self._collection)
            exists = False

        if not exists:
            dim = self._get_embedding_dim()
            self._milvus.create_collection(
                collection_name = self._collection,
                dimension       = dim,
                metric_type     = "COSINE",
                auto_id         = True,
            )
            print(f"[EquationIndexer] Created collection '{self._collection}' (dim={dim})")

    def _get_indexed_sources(self) -> set[str]:
        try:
            results = self._milvus.query(
                collection_name = self._collection,
                filter          = "",
                output_fields   = ["source"],
                limit           = 10_000,
            )
            return {r["source"] for r in results}
        except Exception:
            return set()

    def _embed(self, text: str) -> list[float]:
        resp = self._embed_client.embeddings.create(model=self._embed_model, input=[text])
        return resp.data[0].embedding

    def _embed_and_store(self, equations: list[dict]) -> None:
        if not equations:
            return

        print(f"\n[EquationIndexer] ── STORING IN MILVUS ({len(equations)} equations) ─")
        for i, eq in enumerate(equations, 1):
            print(f"  [{i}] eq_type : {eq.get('eq_type', 'unknown')}")
            print(f"       eq      : {eq['equation'][:120]}")
            print(f"       desc    : {eq['description'][:120]}")
            print(f"       source  : {eq.get('source', '')}")
        print(f"[EquationIndexer] ─────────────────────────────────────────────\n")

        batch_size = 50
        for i in range(0, len(equations), batch_size):
            batch = equations[i: i + batch_size]
            embed_texts = [e["description"] for e in batch]
            resp = self._embed_client.embeddings.create(
                model=self._embed_model, input=embed_texts
            )
            records = [
                {
                    "vector":      r.embedding,
                    "text":        f"{eq['equation']} — {eq['description']}"[:4000],
                    "equation":    eq["equation"][:2000],
                    "eq_type":     eq.get("eq_type", "definition")[:64],
                    "description": eq["description"][:2000],
                    "source":      eq["source"][:512],
                }
                for r, eq in zip(resp.data, batch)
            ]
            self._milvus.insert(collection_name=self._collection, data=records)

    # ── Extraction helpers ────────────────────────────────────────────────────

    @staticmethod
    def _page_to_base64(page) -> str:
        img = page.to_image(resolution=PAGE_RENDER_RESOLUTION).original
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    @staticmethod
    def _parse_equation_json(raw: str) -> list[dict]:
        """Parse LLM JSON output into a list of equation dicts.

        Accepts both the old 2-field format (equation, description) and the
        new 3-field format (equation, eq_type, description).  Missing eq_type
        is defaulted to "definition" so old chunks stay compatible.
        """
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        parsed = json.loads(raw)
        if not isinstance(parsed, list):
            return []

        results = []
        for e in parsed:
            if not isinstance(e, dict):
                continue
            equation    = str(e.get("equation",    "")).strip()
            description = str(e.get("description", "")).strip()
            eq_type     = str(e.get("eq_type",     "definition")).strip()

            if not equation:
                continue

            # Normalise eq_type — fall back to "definition" if unrecognised
            if eq_type not in _VALID_EQ_TYPES:
                eq_type = "definition"

            results.append({
                "equation":    equation,
                "eq_type":     eq_type,
                "description": description,
            })
        return results

    def _vision_extract(self, page) -> list[dict]:
        try:
            image_b64 = self._page_to_base64(page)
            raw = llm_call_vision(
                prompt          = _VISION_PROMPT,
                image_b64       = image_b64,
                max_tokens      = 1500,
                use_ollama      = self._use_ollama,
                ollama_model    = self._vision_model,
                anthropic_model = self._anthropic_model,
            )
            return self._parse_equation_json(raw)
        except Exception as exc:
            print(f"[EquationIndexer]   vision extraction failed: {exc}")
        return []

    def _document_extract(self, pdf_path: Path, pages_per_chunk: int = 30) -> list[dict]:
        """Extract equations from a PDF by sending page chunks to Claude."""
        import io as _io
        import base64 as _b64
        from pypdf import PdfReader, PdfWriter

        reader     = PdfReader(str(pdf_path))
        total      = len(reader.pages)
        n_chunks   = (total + pages_per_chunk - 1) // pages_per_chunk
        label      = f"({n_chunks} chunk{'s' if n_chunks > 1 else ''})"
        print(f"[EquationIndexer] Extracting from {pdf_path.name} — {total} pages {label}…")

        equations: list[dict] = []
        seen: set[str] = set()

        for chunk_idx, start in enumerate(range(0, total, pages_per_chunk), start=1):
            end = min(start + pages_per_chunk, total)
            print(f"[EquationIndexer]   chunk {chunk_idx}/{n_chunks}: pages {start+1}–{end}")

            writer = PdfWriter()
            for i in range(start, end):
                writer.add_page(reader.pages[i])
            buf = _io.BytesIO()
            writer.write(buf)
            pdf_b64 = _b64.b64encode(buf.getvalue()).decode()

            try:
                raw = llm_call_pdf_document(
                    pdf_b64         = pdf_b64,
                    prompt          = _DOCUMENT_PROMPT,
                    max_tokens      = 100000,
                    anthropic_model = self._anthropic_model,
                )
                for eq in self._parse_equation_json(raw):
                    if eq["equation"] not in seen:
                        seen.add(eq["equation"])
                        equations.append(eq)
            except Exception as exc:
                print(f"[EquationIndexer]   chunk {chunk_idx} failed: {exc}")

        return equations

    @staticmethod
    def _text_extract(page) -> list[dict]:
        text = page.extract_text() or ""
        results: list[dict] = []
        seen: set[str] = set()
        for pattern in _LATEX_PATTERNS:
            for match in pattern.finditer(text):
                eq = match.group(1).strip() if pattern.groups else match.group(0).strip()
                if eq and eq not in seen and len(eq) >= 3:
                    seen.add(eq)
                    results.append({
                        "equation":    eq,
                        "eq_type":     "definition",   # text-layer fallback
                        "description": "",
                    })
        return results

    def _extract_equations_from_pdf(
        self,
        pdf_path: Path,
        pages_per_chunk: int = 2,
        pages_overlap: int = 1,
    ) -> list[dict]:
        """Extract equations in overlapping page chunks."""
        with pdfplumber.open(pdf_path) as pdf:
            pages_text = [page.extract_text() or "" for page in pdf.pages]

        total = len(pages_text)
        if not any(pages_text):
            print(f"[EquationIndexer] {pdf_path.name}: no extractable text, skipping")
            return []

        stride   = max(1, pages_per_chunk - pages_overlap)
        starts   = list(range(0, total, stride))
        n_chunks = len(starts)
        print(f"[EquationIndexer] {pdf_path.name}: {total} pages → "
              f"{n_chunks} chunk(s) of {pages_per_chunk} (overlap={pages_overlap})")

        equations: list[dict] = []
        seen: set[str] = set()

        for chunk_idx, start in enumerate(starts, start=1):
            end     = min(start + pages_per_chunk, total)
            context = "\n\n".join(pages_text[start:end]).strip()
            if not context:
                print(f"[EquationIndexer]   chunk {chunk_idx}/{n_chunks} "
                      f"(pp {start+1}–{end}): empty, skipping")
                continue

            print(f"\n[EquationIndexer] ── chunk {chunk_idx}/{n_chunks}  "
                  f"pages {start+1}–{end}  ({len(context)} chars) ──────────────")
            prompt = _CHUNKS_PROMPT.format(context=context)

            try:
                raw = llm_call(
                    prompt          = prompt,
                    max_tokens      = 8192,
                    use_ollama      = self._use_ollama,
                    ollama_model    = self._ollama_model,
                    anthropic_model = self._anthropic_model,
                )
                print(f"[EquationIndexer]   raw response ({len(raw)} chars):")
                print(raw[:2000])
                if len(raw) > 2000:
                    print(f"  ... [{len(raw)-2000} chars truncated for display]")

                chunk_eqs = self._parse_equation_json(raw)
                new_eqs   = [eq for eq in chunk_eqs if eq["equation"] not in seen]
                dup_count = len(chunk_eqs) - len(new_eqs)
                for eq in new_eqs:
                    seen.add(eq["equation"])
                equations.extend(new_eqs)

                print(f"[EquationIndexer]   parsed {len(chunk_eqs)} equations "
                      f"({len(new_eqs)} new, {dup_count} duplicates from overlap)")
                for i, eq in enumerate(new_eqs, 1):
                    print(f"    [{i}] [{eq.get('eq_type','?'):>13}] "
                          f"{eq['equation'][:100]}")
                    print(f"              {eq['description'][:100]}")

            except Exception as exc:
                print(f"[EquationIndexer]   chunk {chunk_idx}/{n_chunks} FAILED: {exc}")

        # ── Summary with eq_type breakdown ────────────────────────────────────
        from collections import Counter
        type_counts = Counter(eq.get("eq_type", "unknown") for eq in equations)
        print(f"\n[EquationIndexer] ── TOTAL: {len(equations)} equations from "
              f"{pdf_path.name} ─────────────────────────────")
        for eq_type, count in sorted(type_counts.items()):
            print(f"  {eq_type:<15} : {count}")
        print(f"[EquationIndexer] ──────────────────────────────────────────────────\n")
        return equations

    # ── Public API ────────────────────────────────────────────────────────────

    def index(self, papers_dir: Union[str, Path] = "./papers", force_update: bool = False) -> None:
        papers_dir = Path(papers_dir)
        pdfs = sorted(papers_dir.rglob("*.pdf"))
        if not pdfs:
            print(f"[EquationIndexer] No PDFs found in {papers_dir}")
            return

        self._ensure_collection(force_update=force_update)
        already_indexed = set() if force_update else self._get_indexed_sources()

        all_equations: list[dict] = []
        for pdf_path in pdfs:
            if pdf_path.name in already_indexed:
                print(f"[EquationIndexer] Skipping (already indexed): {pdf_path.name}")
                continue

            print(f"[EquationIndexer] Extracting equations from: {pdf_path.name}")

            # ✅ Use document extract (sends real PDF to Claude) instead of
            #    text-layer extract (pdfplumber plain text → LLM reconstruction)
            eqs = self._document_extract(pdf_path)

            print(f"[EquationIndexer]   → {len(eqs)} equations found")
            for eq in eqs:
                eq["source"] = pdf_path.name
            all_equations.extend(eqs)

        if all_equations:
            self._embed_and_store(all_equations)
    def index_from_rag(
        self,
        rag,
        user_query: str | None = None,
        top_k_per_query: int = 10,
    ) -> None:
        """Extract equations from rag_documents text chunks and store in equation_index."""
        from collections import defaultdict

        self._ensure_collection()
        already_indexed = self._get_indexed_sources()

        base    = user_query[:200] if user_query else ""
        queries = [
            f"mathematical equation formula {base}",
            f"rate constant parameter coefficient {base}",
            f"model derivation expression {base}",
        ]

        seen_texts: set[str] = set()
        chunks_by_source: dict[str, list[str]] = defaultdict(list)

        orig_top_k = rag.top_k
        rag.top_k  = top_k_per_query
        try:
            for q in queries:
                for chunk in rag.get_context(q):
                    if chunk["source"] in already_indexed:
                        continue
                    if chunk["text"] not in seen_texts:
                        seen_texts.add(chunk["text"])
                        chunks_by_source[chunk["source"]].append(chunk["text"])
        finally:
            rag.top_k = orig_top_k

        if not chunks_by_source:
            print("[EquationIndexer] No new chunks to extract equations from.")
            return

        total_chunks = sum(len(v) for v in chunks_by_source.values())
        print(f"[EquationIndexer] {total_chunks} chunks across "
              f"{len(chunks_by_source)} source(s) → extracting equations…")

        all_equations: list[dict] = []
        for source, texts in chunks_by_source.items():
            context = "\n\n---\n\n".join(texts)
            prompt  = _CHUNKS_PROMPT.format(context=context)
            try:
                raw = llm_call(
                    prompt          = prompt,
                    max_tokens      = 4096,
                    use_ollama      = self._use_ollama,
                    ollama_model    = self._ollama_model,
                    anthropic_model = self._anthropic_model,
                )
                eqs = self._parse_equation_json(raw)
                print(f"[EquationIndexer]   {source}: {len(eqs)} equations")
                for eq in eqs:
                    eq["source"] = source
                all_equations.extend(eqs)
            except Exception as exc:
                print(f"[EquationIndexer]   extraction failed for {source}: {exc}")

        if all_equations:
            print(f"[EquationIndexer] Embedding and storing {len(all_equations)} equations…")
            self._embed_and_store(all_equations)
            print("[EquationIndexer] Done.")
        else:
            print("[EquationIndexer] No equations extracted.")

    @staticmethod
    def _filter_by_eq_type(
        hits: list[dict],
        preferred: set[str] = _PREFERRED_TYPES,
        blocked:   set[str] = _BLOCKED_TYPES,
        min_preferred: int  = 3,
    ) -> list[dict]:
        """Suppress asymptotic/dimensionless noise; prefer ODE/definition/equilibrium.

        Falls back to non-blocked results if fewer than `min_preferred` preferred
        results are available, so retrieval never returns empty-handed.
        """
        preferred_hits = [h for h in hits if h.get("eq_type") in preferred]
        fallback_hits  = [h for h in hits if h.get("eq_type") not in blocked]

        filtered = preferred_hits if len(preferred_hits) >= min_preferred else fallback_hits
        removed  = len(hits) - len(filtered)

        if removed:
            print(f"[EquationIndexer] eq_type filter: "
                  f"{len(hits)} → {len(filtered)} hits "
                  f"(removed {removed} asymptotic/dimensionless)")
        return filtered

    def get_equation_context(self, query: str) -> list[dict]:
        """Return top-k equations relevant to `query`, filtered by eq_type.

        Returns list of dicts with keys: equation, eq_type, description, source, score.
        """
        q_emb = self._embed(query)
        results = self._milvus.search(
            collection_name = self._collection,
            data            = [q_emb],
            limit           = self._top_k,
            output_fields   = ["equation", "eq_type", "description", "source"],
        )

        hits = []
        for r in results[0]:
            hits.append({
                "equation":    r["entity"]["equation"],
                "eq_type":     r["entity"].get("eq_type", "unknown"),
                "description": r["entity"]["description"],
                "source":      r["entity"]["source"],
                "score":       round(r["distance"], 4),
            })

        return self._filter_by_eq_type(hits)