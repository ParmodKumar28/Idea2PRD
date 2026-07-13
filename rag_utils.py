"""
rag_utils.py — RAG pipeline for Idea2PRD.

Handles:
  - Parsing uploaded files (PDF and TXT)
  - Chunking text with LangChain's RecursiveCharacterTextSplitter
  - Building a FAISS vector store with embeddings
  - Retrieving top-k relevant chunks for agent context
  - Formatting retrieved chunks for display and prompt injection
"""

import os
import tempfile
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from config import CHUNK_SIZE, CHUNK_OVERLAP, TOP_K_RETRIEVAL


# ─── Data Classes ─────────────────────────────────────────────────────

@dataclass
class ParsedDocument:
    """Represents a parsed document with its text content and metadata."""
    filename: str
    content: str
    file_type: str  # "pdf" or "txt"


@dataclass
class RetrievedChunk:
    """A single retrieved chunk with its source and relevance score."""
    content: str
    source: str
    score: float = 0.0
    chunk_index: int = 0


@dataclass
class RAGResult:
    """Complete RAG retrieval result for an agent query."""
    chunks: List[RetrievedChunk] = field(default_factory=list)
    formatted_context: str = ""
    has_context: bool = False


# ─── File Parsing ─────────────────────────────────────────────────────

def parse_uploaded_files(uploaded_files: list) -> List[ParsedDocument]:
    """Parse uploaded PDF and TXT files into ParsedDocument objects.

    Args:
        uploaded_files: List of Streamlit UploadedFile objects.

    Returns:
        List of ParsedDocument with extracted text content.
    """
    documents = []

    for uploaded_file in uploaded_files:
        filename = uploaded_file.name
        file_extension = os.path.splitext(filename)[1].lower()

        try:
            if file_extension == ".pdf":
                content = _parse_pdf(uploaded_file)
            elif file_extension == ".txt":
                content = _parse_txt(uploaded_file)
            else:
                # Skip unsupported file types
                continue

            if content.strip():
                documents.append(ParsedDocument(
                    filename=filename,
                    content=content,
                    file_type=file_extension.lstrip("."),
                ))
        except Exception as e:
            # Log the error but continue processing other files
            print(f"Warning: Failed to parse {filename}: {e}")
            continue

    return documents


def _parse_pdf(uploaded_file) -> str:
    """Extract text from a PDF file using PyPDF2."""
    from PyPDF2 import PdfReader

    # Write to a temp file because PdfReader needs a file path or file-like object
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    try:
        reader = PdfReader(tmp_path)
        text_parts = []
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
        return "\n\n".join(text_parts)
    finally:
        os.unlink(tmp_path)


def _parse_txt(uploaded_file) -> str:
    """Extract text from a TXT file."""
    return uploaded_file.getvalue().decode("utf-8", errors="replace")


# ─── Text Chunking ───────────────────────────────────────────────────

def chunk_documents(
    documents: List[ParsedDocument],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[dict]:
    """Split parsed documents into chunks with metadata.

    Uses LangChain's RecursiveCharacterTextSplitter for intelligent
    splitting at paragraph/sentence boundaries.

    Args:
        documents: List of ParsedDocument objects.
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Overlap between consecutive chunks.

    Returns:
        List of dicts with 'content', 'source', and 'chunk_index' keys.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    all_chunks = []

    for doc in documents:
        chunks = splitter.split_text(doc.content)
        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "content": chunk,
                "source": doc.filename,
                "chunk_index": i,
            })

    return all_chunks


# ─── Vector Store ─────────────────────────────────────────────────────

def build_vector_store(
    chunks: List[dict],
    provider: str = "gemini",
    api_key: str = "",
) -> Optional[FAISS]:
    """Build a FAISS vector store from document chunks.

    Args:
        chunks: List of chunk dicts with 'content' and 'source' keys.
        provider: LLM provider — "gemini" or "openai".
        api_key: API key for the embedding model.

    Returns:
        FAISS vector store, or None if no chunks provided.
    """
    if not chunks:
        return None

    # Select embedding model based on provider
    embeddings = _get_embeddings(provider, api_key)

    # Prepare texts and metadatas for FAISS
    texts = [chunk["content"] for chunk in chunks]
    metadatas = [
        {"source": chunk["source"], "chunk_index": chunk["chunk_index"]}
        for chunk in chunks
    ]

    # Build the FAISS index
    vector_store = FAISS.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
    )

    return vector_store


def _get_embeddings(provider: str, api_key: str):
    """Get the appropriate embedding model based on provider."""
    if provider == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        return GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=api_key,
        )
    elif provider == "groq":
        # Groq doesn't offer an embedding API — use Gemini's free embeddings
        # Note: This requires a Gemini API key for RAG when using Groq for chat.
        # For simplicity, we skip embeddings and RAG won't be available with Groq alone.
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=api_key,
            openai_api_base="https://api.groq.com/openai/v1",
        )
    else:
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=api_key,
        )


# ─── Retrieval ────────────────────────────────────────────────────────

def retrieve_context(
    query: str,
    vector_store: Optional[FAISS] = None,
    top_k: int = TOP_K_RETRIEVAL,
) -> RAGResult:
    """Retrieve the most relevant chunks for a given query.

    Args:
        query: The search query (e.g., agent-specific question).
        vector_store: FAISS vector store to search.
        top_k: Number of top results to return.

    Returns:
        RAGResult with retrieved chunks and formatted context string.
    """
    if vector_store is None:
        return RAGResult(chunks=[], formatted_context="", has_context=False)

    # Retrieve with scores (lower distance = more relevant)
    results = vector_store.similarity_search_with_score(query, k=top_k)

    chunks = []
    for doc, score in results:
        chunks.append(RetrievedChunk(
            content=doc.page_content,
            source=doc.metadata.get("source", "Unknown"),
            score=round(float(score), 4),
            chunk_index=doc.metadata.get("chunk_index", 0),
        ))

    # Format for prompt injection
    formatted = format_context_for_prompt(chunks)

    return RAGResult(
        chunks=chunks,
        formatted_context=formatted,
        has_context=len(chunks) > 0,
    )


def format_context_for_prompt(chunks: List[RetrievedChunk]) -> str:
    """Format retrieved chunks into a string for LLM prompt injection.

    Each chunk is labeled with its source document for citation tracking.
    """
    if not chunks:
        return ""

    parts = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(
            f"--- Excerpt {i} [Source: {chunk.source}] ---\n"
            f"{chunk.content}\n"
        )

    return "\n".join(parts)


# ─── Display Helpers ──────────────────────────────────────────────────

def get_source_summary(chunks: List[RetrievedChunk]) -> Dict[str, int]:
    """Get a summary of how many chunks came from each source document."""
    summary: Dict[str, int] = {}
    for chunk in chunks:
        summary[chunk.source] = summary.get(chunk.source, 0) + 1
    return summary
