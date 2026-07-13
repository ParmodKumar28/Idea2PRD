# Idea2PRD — Architecture Document

## System Architecture

### High-Level Data Flow

```
┌──────────┐    ┌───────────────┐    ┌──────────────┐    ┌───────────┐    ┌────────────┐
│   User   │───▶│  Streamlit UI │───▶│ RAG Retriever│───▶│  Agent    │───▶│ PRD Output │
│  (Input) │    │  (app.py)     │    │ (rag_utils)  │    │ Workflow  │    │ (Markdown) │
└──────────┘    └───────────────┘    └──────────────┘    │ (agents)  │    └────────────┘
                       │                     │           └───────────┘
                       │                     │                  │
                       ▼                     ▼                  ▼
                ┌─────────────┐      ┌─────────────┐    ┌─────────────┐
                │  File Upload│      │   FAISS     │    │  LLM API    │
                │  (PDF/TXT)  │      │  Vector DB  │    │ (Gemini/    │
                └─────────────┘      └─────────────┘    │  OpenAI)    │
                                                        └─────────────┘
```

---

## Component Architecture

### 1. Frontend Layer — `app.py`

The Streamlit application serves as the user interface and orchestration layer.

```
app.py
├── Sidebar
│   ├── LLM Provider Selector (Gemini / OpenAI)
│   ├── API Key Input (password field)
│   ├── File Uploader (PDF, TXT — multi-file)
│   └── Sample Input Loader (3 pre-built ideas)
│
├── Main Page
│   ├── Header (branding, description)
│   ├── Input Form (idea, users, domain, constraints)
│   ├── Generate Button + Progress Indicator
│   └── Results Tabs
│       ├── Tab 1: Final PRD (merged, downloadable)
│       ├── Tab 2: Retrieved Context (chunks + sources)
│       ├── Tab 3: Agent Outputs (raw per-agent output)
│       └── Tab 4: Evaluation (checklist + rubric)
│
└── Session State
    ├── pipeline_result (PipelineResult)
    ├── vector_store (FAISS index)
    ├── parsed_docs (list of ParsedDocument)
    └── doc_chunks (list of chunk dicts)
```

### 2. RAG Pipeline — `rag_utils.py`

Handles document ingestion, chunking, embedding, and retrieval.

```
RAG Pipeline Flow:

  Uploaded Files (PDF/TXT)
         │
         ▼
  ┌──────────────────┐
  │   File Parser     │  PyPDF2 for PDF, UTF-8 decode for TXT
  │   parse_uploaded  │  → List[ParsedDocument]
  └────────┬─────────┘
           │
           ▼
  ┌──────────────────┐
  │   Text Chunker    │  RecursiveCharacterTextSplitter
  │   chunk_documents │  chunk_size=1000, overlap=200
  │                   │  → List[{content, source, chunk_index}]
  └────────┬─────────┘
           │
           ▼
  ┌──────────────────┐
  │   Embedder +      │  Google embedding-001 or OpenAI text-embedding-3-small
  │   Vector Store    │  → FAISS.from_texts()
  │   build_vector    │  → FAISS index (in-memory)
  └────────┬─────────┘
           │
           ▼
  ┌──────────────────┐
  │   Retriever       │  similarity_search_with_score(query, top_k=5)
  │   retrieve_context│  → RAGResult(chunks, formatted_context)
  └──────────────────┘
```

**Key Parameters:**
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Chunk Size | 1000 chars | Balances context richness with embedding quality |
| Chunk Overlap | 200 chars | Prevents loss of context at chunk boundaries |
| Top-K | 5 | Provides sufficient context without overwhelming the prompt |
| Embedding Model (Gemini) | embedding-001 | Best available free embedding model |
| Embedding Model (OpenAI) | text-embedding-3-small | Cost-effective, high quality |

### 3. Multi-Agent Workflow — `agents.py`

Three specialized agents run sequentially, each building on prior outputs.

```
Sequential Agent Pipeline:

  ┌─────────────────────────────────────────────────────────┐
  │                                                         │
  │  ┌───────────────────┐     Outputs:                    │
  │  │ Business Analyst   │──→  • Problem Statement         │
  │  │ Agent              │     • Goals & Objectives        │
  │  │                    │     • Target Users               │
  │  │ Input: idea,       │     • User Personas (×2)        │
  │  │  users, domain,    │     • Key Pain Points           │
  │  │  constraints,      │     • Assumptions               │
  │  │  RAG context       │                                 │
  │  └────────┬──────────┘                                 │
  │           │ passes output ▼                             │
  │  ┌───────────────────┐     Outputs:                    │
  │  │ Product Manager    │──→  • Proposed Solution         │
  │  │ Agent              │     • User Stories (6-10)       │
  │  │                    │     • Functional Requirements    │
  │  │ Input: idea,       │     • Non-Functional Reqs       │
  │  │  BA output,        │     • Success Metrics           │
  │  │  RAG context       │     • MVP vs Future Scope       │
  │  └────────┬──────────┘                                 │
  │           │ passes output ▼                             │
  │  ┌───────────────────┐     Outputs:                    │
  │  │ Risk Reviewer      │──→  • Risks & Mitigation        │
  │  │ Agent              │     • Ambiguities & Gaps        │
  │  │                    │     • Missing Requirements       │
  │  │ Input: idea,       │     • Future Enhancements       │
  │  │  BA output,        │     • Clarifying Questions      │
  │  │  PM output,        │                                 │
  │  │  RAG context       │                                 │
  │  └────────┬──────────┘                                 │
  │           │ all outputs ▼                               │
  │  ┌───────────────────┐                                 │
  │  │ PRD Compiler       │──→  Final 12-section PRD       │
  │  │ (Technical Writer) │     (merged, formatted,         │
  │  │                    │      with citations)            │
  │  └───────────────────┘                                 │
  │                                                         │
  └─────────────────────────────────────────────────────────┘
```

**Agent Communication Pattern:**
- **BA → PM**: PM receives BA's analysis to build requirements on top of identified personas and pain points
- **BA + PM → Risk**: Risk reviewer sees both prior outputs to identify cross-cutting concerns
- **All → Compiler**: Technical writer merges all three into a cohesive document

### 4. Prompt Engineering — `prompts.py`

Each agent has a carefully structured system prompt.

```
Prompt Structure:

  ┌────────────────────────────────────┐
  │  System Prompt (per agent)         │
  │  ├── Role Assignment               │
  │  ├── Output Format Specification   │
  │  │   (exact markdown sections)     │
  │  ├── Section Descriptions          │
  │  │   (what each section contains)  │
  │  └── Rules & Guardrails            │
  │      (citation, anti-hallucination)│
  ├────────────────────────────────────┤
  │  Context Block (dynamic)           │
  │  ├── If docs uploaded:             │
  │  │   Retrieved chunks with sources │
  │  │   + citation instructions       │
  │  └── If no docs:                   │
  │      Mark-as-assumption directive  │
  ├────────────────────────────────────┤
  │  Prior Agent Outputs (if any)      │
  ├────────────────────────────────────┤
  │  User Input                        │
  │  (idea, users, domain, constraints)│
  └────────────────────────────────────┘
```

**Guardrail Strategies:**
1. **Citation enforcement**: "When you use information from excerpts, add [Source: filename]"
2. **Assumption marking**: "Mark ALL factual claims as [Assumption]: when no docs provided"
3. **Negative instructions**: "Do NOT invent market statistics or competitor data"
4. **Format locking**: "Produce output in the EXACT markdown format specified"
5. **Scope containment**: "Do NOT add features that contradict stated constraints"

---

## Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Frontend | Streamlit | Web UI framework |
| LLM Orchestration | LangChain | LLM abstraction and chaining |
| LLM (Option 1) | Google Gemini 2.0 Flash | Text generation (free tier) |
| LLM (Option 2) | OpenAI GPT-4o-mini | Text generation (paid) |
| Embeddings (Option 1) | Google embedding-001 | Document embedding (free) |
| Embeddings (Option 2) | OpenAI text-embedding-3-small | Document embedding (paid) |
| Vector Database | FAISS (in-memory) | Similarity search |
| PDF Parsing | PyPDF2 | PDF text extraction |
| Tokenization | tiktoken | Token counting |

---

## Design Decisions

### Why FAISS over ChromaDB?
- **Zero setup**: No external server or database to run
- **In-memory**: Perfect for session-scoped document retrieval
- **Student-friendly**: `pip install faiss-cpu` and done
- **Sufficient scale**: Handles hundreds of document chunks easily

### Why Sequential Agents over Framework (CrewAI, AutoGen)?
- **Transparency**: Easy to understand and debug the pipeline
- **Readability**: Clear code flow for a course submission
- **Control**: Explicit prompt engineering is visible in the code
- **No overhead**: No additional framework dependencies

### Why LangChain?
- **Provider abstraction**: Same code works with Gemini and OpenAI
- **Built-in splitters**: `RecursiveCharacterTextSplitter` handles edge cases well
- **FAISS integration**: Direct `from_texts` and `similarity_search_with_score`
- **Industry standard**: Widely used, well-documented
