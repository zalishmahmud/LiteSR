"""
rag.py — Agentic RAG module using Milvus + Ollama embeddings (local).

Usage:
    from litesr.literature_agent.rag import LiteratureAgent

    rag = LiteratureAgent(milvus_uri="http://localhost:19530")
    rag.index_pdfs("./papers")

    result = rag.agentic_query("your question or code snippet", return_sources=True)
    print(result["answer"])
    print(result["sources"])
    print(result["sub_queries"])
"""
from litesr.literature_agent.query_utils import split_researcher_input as _split_researcher_input
from pathlib import Path
from typing import Union
import json
import re

import pdfplumber
from pymilvus import MilvusClient
from litesr.clients import get_ollama_client, llm_call
from litesr.literature_agent.subquery_prompt import build_subquery_prompt

# ── Defaults ───────────────────────────────────────────────────────────────────
DEFAULT_COLLECTION  = "rag_documents"
DEFAULT_EMBED_MODEL = "nomic-embed-text"
DEFAULT_GEN_MODEL   = "mistral:latest"
DEFAULT_OLLAMA_URL  = "http://localhost:11434/v1"
DEFAULT_TOP_K       = 5
CHUNK_SIZE          = 500
CHUNK_OVERLAP       = 100

# Minimum cosine similarity score for a chunk to be considered useful.
REFLECT_SCORE_THRESHOLD = 0.7

AGENTIC_SYNTHESIS_TEMPLATE = """\
You are a scientific equation extraction assistant.

The researcher is working on this mathematical problem:
=== SPECIFICATION (authoritative — describes the target equation) ===
{nl_description}

=== FUNCTION SIGNATURE (the Python function that must be implemented) ===
{code_block}

The retrieved chunks below already contain extracted equations with term descriptions. \
Your job is to select and present the equations most relevant to the researcher's query.

STRICT OUTPUT RULES:
- Output ONLY mathematical equations and their descriptions.
- DO NOT output any Python, pseudocode, or programming constructs of any kind.
- DO NOT output prose paragraphs or derivations.
- Preserve the original equation text and term descriptions exactly — do NOT rephrase or restate them.
- Select only equations relevant to the query; skip chunks with no usable equation.
- DO NOT construct, assemble, or infer new equations from components in the chunks. \
  If a complete equation does not appear verbatim in a chunk, do NOT output it.
- DO NOT add any commentary, recommendations, instructions, or closing remarks \
  after the equations. Your output ends with the last equation entry.
- Format each entry as:
    [Equation Name]
    <equation>
    → <description with term breakdown>

─────────────────────────────────────────────────────────────────────
Context from retrieved papers:
{context}

Relevant equations:"""


class LiteratureAgent:
    """
    RAG pipeline with built-in local Ollama embedding.

    Parameters
    ----------
    milvus_uri : str
        Milvus server URI, e.g. "http://localhost:19530".
    local_embedding : bool
        If True (default), uses Ollama locally for embedding.
    embedding_model : str
        Ollama embedding model. Default: "nomic-embed-text".
    ollama_url : str
        Ollama base URL. Default: "http://localhost:11434/v1".
    collection_name : str
        Milvus collection name (created automatically if absent).
    top_k : int
        Number of chunks to retrieve per query.
    """

    def __init__(
        self,
        milvus_uri:        str,
        local_embedding:   bool = True,
        embedding_model:   str  = DEFAULT_EMBED_MODEL,
        ollama_url:        str  = DEFAULT_OLLAMA_URL,
        collection_name:   str  = DEFAULT_COLLECTION,
        top_k:             int  = DEFAULT_TOP_K,
        use_ollama:        bool = False,
        ollama_model:      str  = "mistral:latest",
    ):
        if local_embedding:
            self.embed_client = get_ollama_client(ollama_url)
        else:
            self.embed_client = None

        self.embed_model     = embedding_model
        self.collection_name = collection_name
        self.top_k           = top_k
        self._use_ollama     = use_ollama
        self._ollama_model   = ollama_model

        self.milvus = self._connect_milvus(milvus_uri)
        print(f"[RAG] Connected to Milvus at {milvus_uri}")
        print(f"[RAG] Embedding  : {embedding_model} via {'Ollama (local)' if local_embedding else 'external'}")
        synthesis_backend = f"Ollama ({ollama_model})" if use_ollama else "Anthropic claude-sonnet-4-6"
        print(f"[RAG] Synthesis  : {synthesis_backend}")

    # ── Milvus readiness helpers ───────────────────────────────────────────────

    @staticmethod
    def _connect_milvus(
        uri:         str,
        max_wait:    float = 120.0,
        interval:    float = 3.0,
    ) -> MilvusClient:
        """
        Create a MilvusClient and wait until the proxy is actually ready.

        Retries list_collections until it succeeds or max_wait seconds elapse.
        """
        import time
        from pymilvus.exceptions import MilvusException

        client    = MilvusClient(uri=uri)
        deadline  = time.monotonic() + max_wait
        attempt   = 0

        while True:
            attempt += 1
            try:
                client.list_collections()
                if attempt > 1:
                    print(f"[RAG] Milvus ready after {attempt} attempt(s).")
                return client
            except MilvusException as exc:
                msg = str(exc)
                if "not ready yet" not in msg and "service unavailable" not in msg:
                    raise
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RuntimeError(
                        f"Milvus at {uri!r} did not become ready within {max_wait}s. "
                        f"Last error: {exc}"
                    ) from exc
                wait = min(interval, remaining)
                print(
                    f"[RAG] Milvus proxy not ready yet (attempt {attempt}), "
                    f"retrying in {wait:.0f}s… ({remaining:.0f}s left)"
                )
                time.sleep(wait)

    # ── Public API ─────────────────────────────────────────────────────────────

    def index_pdfs(self, source: Union[str, Path, list], force_update: bool = False):
        """
        Parse PDFs, chunk them, embed with Ollama, and store in Milvus.

        Parameters
        ----------
        source : str | Path | list
            A folder path containing PDFs, or a list of individual PDF paths.
        force_update : bool
            False (default) — skip PDFs already indexed, only add new ones.
            True — drop everything and re-index all PDFs from scratch.
        """
        if isinstance(source, list):
            pdf_paths = [Path(p) for p in source]
        else:
            pdf_paths = list(Path(source).glob("**/*.pdf"))

        if not pdf_paths:
            print("[RAG] No PDF files found. Nothing to index.")
            return

        sample_dim = self._get_embedding_dim()

        if force_update:
            if self.milvus.has_collection(self.collection_name):
                self.milvus.drop_collection(self.collection_name)
                print(f"[RAG] Force update — dropped collection '{self.collection_name}'")
            self.milvus.create_collection(
                collection_name=self.collection_name,
                dimension=sample_dim,
                metric_type="COSINE",
                auto_id=True,
            )
            pdfs_to_index = pdf_paths
        else:
            if not self.milvus.has_collection(self.collection_name):
                self.milvus.create_collection(
                    collection_name=self.collection_name,
                    dimension=sample_dim,
                    metric_type="COSINE",
                    auto_id=True,
                )
                print(f"[RAG] Created collection '{self.collection_name}' (dim={sample_dim})")
                pdfs_to_index = pdf_paths
            else:
                already_indexed = self._get_indexed_sources()
                pdfs_to_index = [p for p in pdf_paths if p.name not in already_indexed]
                skipped = len(pdf_paths) - len(pdfs_to_index)
                if skipped:
                    print(f"[RAG] Skipping {skipped} already indexed PDF(s). Use force_update=True to re-index.")

        if not pdfs_to_index:
            print("[RAG] All PDFs already indexed. Nothing to do.")
            return

        print(f"[RAG] Indexing {len(pdfs_to_index)} PDF(s)...")
        all_chunks  = []
        failed_pdfs = []
        for pdf_path in pdfs_to_index:
            try:
                chunks = self._parse_and_chunk_pdf(pdf_path)
                all_chunks.extend(chunks)
                print(f"[RAG]   {pdf_path.name} -> {len(chunks)} chunks")
            except Exception as e:
                failed_pdfs.append(pdf_path.name)
                print(f"[RAG]   SKIPPED {pdf_path.name} — corrupt or unreadable PDF: {e}")
                continue

        if failed_pdfs:
            print(f"[RAG] WARNING: {len(failed_pdfs)} PDF(s) skipped due to errors: {failed_pdfs}")

        self._embed_and_store(all_chunks)
        print(f"[RAG] Indexed {len(all_chunks)} chunks total.")

    def get_context(self, question: str) -> list[dict]:
        """
        Embed the question locally with Ollama, search Milvus, return raw chunks.

        Returns list of {"text": ..., "source": ..., "score": ...}
        """
        q_embedding = self._embed(question)
        results = self.milvus.search(
            collection_name=self.collection_name,
            data=[q_embedding],
            limit=self.top_k,
            output_fields=["text", "source"],
        )
        return [
            {
                "text":   hit["entity"]["text"],
                "source": hit["entity"]["source"],
                "score":  round(hit["distance"], 4),
            }
            for hit in results[0]
        ]

    def agentic_query(
        self,
        user_query:     str,
        return_sources: bool = False,
        max_retries:    int  = 5,
    ) -> Union[str, dict]:
        """
        Agentic RAG loop:
          1. Decompose  — Claude splits the query into focused sub-queries.
          2. Retrieve   — run get_context() for each sub-query independently.
          3. Reflect    — Claude inspects weak/empty results and rewrites them.
          4. Retry      — re-retrieve for any rewritten sub-queries (up to max_retries).
          5. Synthesize — Claude generates a final answer over the merged context.
        """
        # ── Step 1: Decompose ──────────────────────────────────────────────
        print("\n[RAG:agent] Step 1 — Decomposing query into sub-queries…")
        sub_queries = self._decompose_query(user_query)
        print(f"[RAG:agent]   {len(sub_queries)} sub-queries generated:")
        for i, sq in enumerate(sub_queries):
            print(f"    [{i+1}] {sq}")

        # ── Steps 2–4: Retrieve → Reflect → Retry ─────────────────────────
        all_chunks:   list[dict]       = []
        final_sub_queries              = sub_queries
        all_strong:   dict[str, list]  = {}

        for attempt in range(max_retries + 1):
            if attempt > 0:
                print(f"\n[RAG:agent] Retry round {attempt}/{max_retries}…")

            round_chunks: dict[str, list[dict]] = {}

            for sq in sub_queries:
                chunks = self.get_context(sq)
                round_chunks[sq] = chunks
                best_score = max((c["score"] for c in chunks), default=0.0)
                print(f"[RAG:agent]   '{sq[:60]}' → {len(chunks)} chunks, best score={best_score:.3f}")

            # Merge into all_chunks (deduplicate by text)
            seen_texts = {c["text"] for c in all_chunks}
            for chunks in round_chunks.values():
                for c in chunks:
                    if c["text"] not in seen_texts:
                        all_chunks.append(c)
                        seen_texts.add(c["text"])

            # ── Step 3: Reflect ────────────────────────────────────────────
            if attempt < max_retries:
                weak = {
                    sq: chunks
                    for sq, chunks in round_chunks.items()
                    if not chunks or max((c["score"] for c in chunks), default=0.0) < REFLECT_SCORE_THRESHOLD
                }
                all_strong.update({
                    sq: chunks
                    for sq, chunks in round_chunks.items()
                    if sq not in weak
                })

                if not weak:
                    print("[RAG:agent]   All sub-queries returned strong results — skipping reflection.")
                    break

                print(f"[RAG:agent]   {len(weak)} weak / {len(all_strong)} strong — rephrasing weak only…")
                rephrased_weak = self._reflect_and_rephrase(user_query, weak, all_strong)
                rephrased_weak = rephrased_weak[: len(weak)]

                sub_queries       = rephrased_weak
                final_sub_queries = list(all_strong.keys()) + rephrased_weak

                print(f"[RAG:agent]   Keeping {len(all_strong)} strong, retrying {len(rephrased_weak)} rephrased:")
                for sq in rephrased_weak:
                    print(f"    → {sq}")
            else:
                break

        if not all_chunks:
            answer = "No relevant documents found for this query."
            if return_sources:
                return {"answer": answer, "sources": [], "sub_queries": final_sub_queries}
            return answer

        all_chunks.sort(key=lambda c: c["score"], reverse=True)
        top_chunks = all_chunks[: self.top_k * 3]
        print(f"\n[RAG:agent] Step 5 — Synthesizing over {len(top_chunks)} deduplicated chunks…")

        # ── Step 5: Synthesize ─────────────────────────────────────────────
        context = "\n\n---\n\n".join(
            f"[source: {c['source']} | score: {c['score']}]\n{c['text']}"
            for c in top_chunks
        )
        sources = list({c["source"] for c in top_chunks})
        answer  = self._synthesize(user_query, context)

        if return_sources:
            return {"answer": answer, "sources": sources, "sub_queries": final_sub_queries}
        return answer

    # ── Internals ──────────────────────────────────────────────────────────────

    def _decompose_query(self, user_query: str) -> list[str]:
        """Break the user query into focused sub-queries. Falls back to [user_query] on error."""
        prompt = build_subquery_prompt(user_query)
        try:
            raw = llm_call(prompt, 300, self._use_ollama, self._ollama_model)
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
            sub_queries = json.loads(raw)
            if isinstance(sub_queries, list) and all(isinstance(s, str) for s in sub_queries):
                return [s.strip() for s in sub_queries if s.strip()]
            raise ValueError("Unexpected JSON structure")
        except Exception as e:
            print(f"[RAG] Decomposition failed ({e}), falling back to original query.")
            return [user_query]

    def _reflect_and_rephrase(
        self,
        user_query: str,
        weak: dict[str, list[dict]],
        strong: dict[str, list[dict]] | None = None,
    ) -> list[str]:
        """Rephrase weak sub-queries using strong ones as style guidance."""
        weak_scores = {
            sq: round(max((c["score"] for c in chunks), default=0.0), 3)
            for sq, chunks in weak.items()
        }
        strong_scores = {
            sq: round(max((c["score"] for c in chunks), default=0.0), 3)
            for sq, chunks in (strong or {}).items()
        }

        strong_section = ""
        if strong_scores:
            strong_lines = "\n".join(
                f"  ✓ [{score:.3f}] {sq}"
                for sq, score in strong_scores.items()
            )
            strong_section = (
                "The following sub-queries already returned STRONG results "
                f"(score ≥ {REFLECT_SCORE_THRESHOLD}) — do NOT change these, "
                "but use their terminology and specificity as a guide:\n"
                f"{strong_lines}\n\n"
            )

        weak_lines = "\n".join(
            f"  ✗ [{weak_scores[sq]:.3f}] {sq}"
            for sq in weak
        )

        prompt = (
            "You are a scientific equation search assistant helping to improve retrieval "
            "from a vector database of scientific papers.\n\n"
            f"The researcher's original question was:\n  {user_query}\n\n"
            + strong_section
            + "The following sub-queries returned WEAK or no results and must be rephrased:\n"
            f"{weak_lines}\n\n"
            "Task: rewrite ONLY the weak sub-queries so they are more likely to match "
            "relevant text in scientific papers. Study what made the strong queries work "
            "(specificity, named formulas, domain terms) and apply the same approach.\n\n"
            "Rules:\n"
            "- Return exactly one rephrased sub-query per weak sub-query listed above, "
            "in the same order.\n"
            "- Do NOT include the strong sub-queries in your output.\n"
            "- Try different terminology, synonyms, or a more specific / general framing.\n"
            "- Each rephrased query must be a concise keyword phrase (under 10 words).\n"
            "- Return ONLY a JSON array of strings. No explanation, no markdown.\n\n"
            "JSON array of rephrased weak sub-queries:"
        )
        try:
            raw = llm_call(prompt, 300, self._use_ollama, self._ollama_model)
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
            rephrased = json.loads(raw)
            if isinstance(rephrased, list) and all(isinstance(s, str) for s in rephrased):
                return [s.strip() for s in rephrased if s.strip()]
            raise ValueError("Unexpected JSON structure")
        except Exception as e:
            print(f"[RAG] Reflection failed ({e}), reusing original weak sub-queries.")
            return list(weak.keys())

    def _synthesize(self, user_query: str, context: str) -> str:
        """Generate a final equation-focused answer from the merged context chunks."""
        nl_description, code_block = _split_researcher_input(user_query)
        prompt = AGENTIC_SYNTHESIS_TEMPLATE.format(
            nl_description=nl_description,
            code_block=code_block,
            context=context,
        )
        try:
            return llm_call(prompt, 1500, self._use_ollama, self._ollama_model)
        except Exception as e:
            print(f"[RAG] Synthesis failed ({e}), returning raw context.")
            return context

    def _get_indexed_sources(self) -> set[str]:
        """Return the set of PDF filenames already stored in Milvus."""
        results = self.milvus.query(
            collection_name=self.collection_name,
            filter="source != ''",
            output_fields=["source"],
            limit=16384,
        )
        return {r["source"] for r in results}

    def _parse_and_chunk_pdf(self, pdf_path: Path) -> list[dict]:
        """Extract text from a PDF and split into overlapping chunks."""
        file_size = pdf_path.stat().st_size
        if file_size < 1024:
            raise ValueError(f"File too small ({file_size} bytes) — likely a failed/partial download")

        with pdfplumber.open(pdf_path) as pdf:
            full_text = "\n".join(
                page.extract_text() or "" for page in pdf.pages
            )

        chunks = []
        start  = 0
        while start < len(full_text):
            end  = start + CHUNK_SIZE
            text = full_text[start:end].strip()
            if text:
                chunks.append({
                    "text":   text,
                    "source": pdf_path.name,
                })
            start += CHUNK_SIZE - CHUNK_OVERLAP

        return chunks

    def _embed(self, text: str) -> list[float]:
        """Embed using the internal Ollama embedding client."""
        if self.embed_client is None:
            raise RuntimeError(
                "Embedding client not initialized. "
                "Set local_embedding=True (default) to use Ollama."
            )
        response = self.embed_client.embeddings.create(
            model=self.embed_model,
            input=text,
        )
        return response.data[0].embedding

    def _get_embedding_dim(self) -> int:
        """Probe the embedding model to get vector dimension."""
        return len(self._embed("hello"))

    def _embed_and_store(self, chunks: list[dict], batch_size: int = 100):
        """Embed chunks in batches and insert into Milvus."""
        for i in range(0, len(chunks), batch_size):
            batch  = chunks[i: i + batch_size]
            texts  = [c["text"] for c in batch]

            response   = self.embed_client.embeddings.create(
                model=self.embed_model,
                input=texts,
            )
            embeddings = [r.embedding for r in response.data]

            records = [
                {
                    "vector": emb,
                    "text":   chunk["text"],
                    "source": chunk["source"],
                }
                for emb, chunk in zip(embeddings, batch)
            ]

            self.milvus.insert(
                collection_name=self.collection_name,
                data=records,
            )
            print(f"[RAG]   Stored batch {i // batch_size + 1} ({len(records)} chunks)")
