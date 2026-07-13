"""
agents.py — Multi-agent workflow for Idea2PRD.

Implements three specialized agents that run sequentially:
  1. Business Analyst Agent — extracts problem context, personas, pain points
  2. Product Manager Agent — generates requirements, user stories, MVP scope
  3. Risk Reviewer Agent — identifies gaps, risks, clarifying questions

Each agent receives relevant RAG context and produces structured markdown output.
A final compilation step merges all outputs into a cohesive PRD.
"""

from dataclasses import dataclass, field

from config import GEMINI_MODEL, OPENAI_MODEL, GROQ_MODEL, GROQ_BASE_URL, LLM_TEMPERATURE, LLM_MAX_TOKENS
from prompts import (
    get_business_analyst_prompt,
    get_product_manager_prompt,
    get_risk_reviewer_prompt,
    get_compilation_prompt,
)
from rag_utils import retrieve_context, RAGResult


# ─── Data Classes ─────────────────────────────────────────────────────

@dataclass
class AgentOutput:
    """Output from a single agent run."""
    agent_name: str = ""
    output: str = ""
    rag_result: RAGResult = field(default_factory=RAGResult)
    success: bool = True
    error: str = ""


@dataclass
class PipelineResult:
    """Complete result from the multi-agent pipeline."""
    ba_output: AgentOutput = field(default_factory=lambda: AgentOutput(agent_name="Business Analyst"))
    pm_output: AgentOutput = field(default_factory=lambda: AgentOutput(agent_name="Product Manager"))
    risk_output: AgentOutput = field(default_factory=lambda: AgentOutput(agent_name="Risk Reviewer"))
    final_prd: str = ""
    success: bool = True
    error: str = ""


# ─── LLM Initialization ──────────────────────────────────────────────

def _get_llm(provider: str, api_key: str):
    """Initialize the LLM based on the selected provider.

    Args:
        provider: "gemini", "openai", or "grok"
        api_key: API key for the chosen provider

    Returns:
        LangChain ChatModel instance
    """
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            google_api_key=api_key,
            temperature=LLM_TEMPERATURE,
            max_output_tokens=LLM_MAX_TOKENS,
        )
    elif provider == "groq":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=GROQ_MODEL,
            openai_api_key=api_key,
            openai_api_base=GROQ_BASE_URL,
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
        )
    else:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=OPENAI_MODEL,
            openai_api_key=api_key,
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
        )


def _call_llm(llm, prompt: str) -> str:
    """Call the LLM with a prompt and return the response text.

    Uses LangChain's invoke method for consistent interface across providers.
    """
    response = llm.invoke(prompt)
    return response.content


# ─── Individual Agent Runners ─────────────────────────────────────────

def run_business_analyst_agent(
    llm,
    idea: str,
    target_users: str,
    domain: str,
    constraints: str,
    vector_store=None,
) -> AgentOutput:
    """Run the Business Analyst Agent.

    Retrieves relevant context and analyzes the product idea to extract
    problem statement, goals, personas, and pain points.
    """
    agent_name = "Business Analyst"

    try:
        # Retrieve context relevant to business analysis
        rag_query = f"Business analysis for: {idea}. Target users: {target_users}. Domain: {domain}"
        rag_result = retrieve_context(rag_query, vector_store)

        # Build prompt with context
        prompt = get_business_analyst_prompt(
            idea=idea,
            target_users=target_users,
            domain=domain,
            constraints=constraints,
            retrieved_context=rag_result.formatted_context if rag_result.has_context else None,
        )

        # Call LLM
        output = _call_llm(llm, prompt)

        return AgentOutput(
            agent_name=agent_name,
            output=output,
            rag_result=rag_result,
            success=True,
        )

    except Exception as e:
        return AgentOutput(
            agent_name=agent_name,
            output="",
            success=False,
            error=str(e),
        )


def run_product_manager_agent(
    llm,
    idea: str,
    ba_output: str,
    vector_store=None,
) -> AgentOutput:
    """Run the Product Manager Agent.

    Takes the BA output and generates user stories, requirements, MVP scope,
    and success metrics.
    """
    agent_name = "Product Manager"

    try:
        # Retrieve context relevant to product requirements
        rag_query = f"Product requirements, user stories, and features for: {idea}"
        rag_result = retrieve_context(rag_query, vector_store)

        # Build prompt with BA output and context
        prompt = get_product_manager_prompt(
            idea=idea,
            ba_output=ba_output,
            retrieved_context=rag_result.formatted_context if rag_result.has_context else None,
        )

        # Call LLM
        output = _call_llm(llm, prompt)

        return AgentOutput(
            agent_name=agent_name,
            output=output,
            rag_result=rag_result,
            success=True,
        )

    except Exception as e:
        return AgentOutput(
            agent_name=agent_name,
            output="",
            success=False,
            error=str(e),
        )


def run_risk_reviewer_agent(
    llm,
    idea: str,
    ba_output: str,
    pm_output: str,
    vector_store=None,
) -> AgentOutput:
    """Run the Risk Reviewer Agent.

    Reviews BA and PM outputs to identify risks, gaps, ambiguities,
    and clarifying questions.
    """
    agent_name = "Risk Reviewer"

    try:
        # Retrieve context relevant to risks and compliance
        rag_query = f"Risks, compliance, constraints, and challenges for: {idea}"
        rag_result = retrieve_context(rag_query, vector_store)

        # Build prompt with prior agent outputs and context
        prompt = get_risk_reviewer_prompt(
            idea=idea,
            ba_output=ba_output,
            pm_output=pm_output,
            retrieved_context=rag_result.formatted_context if rag_result.has_context else None,
        )

        # Call LLM
        output = _call_llm(llm, prompt)

        return AgentOutput(
            agent_name=agent_name,
            output=output,
            rag_result=rag_result,
            success=True,
        )

    except Exception as e:
        return AgentOutput(
            agent_name=agent_name,
            output="",
            success=False,
            error=str(e),
        )


def compile_final_prd(
    llm,
    idea: str,
    ba_output: str,
    pm_output: str,
    risk_output: str,
) -> str:
    """Compile the final PRD from all three agent outputs.

    A technical-writer LLM pass that merges outputs into a cohesive document
    with consistent formatting and smooth transitions.
    """
    prompt = get_compilation_prompt(
        idea=idea,
        ba_output=ba_output,
        pm_output=pm_output,
        risk_output=risk_output,
    )

    return _call_llm(llm, prompt)


# ─── Full Pipeline Orchestration ─────────────────────────────────────

def run_full_pipeline(
    idea: str,
    target_users: str,
    domain: str,
    constraints: str,
    provider: str,
    api_key: str,
    vector_store=None,
    progress_callback=None,
) -> PipelineResult:
    """Run the complete multi-agent PRD generation pipeline.

    Orchestrates all three agents sequentially (BA → PM → Risk),
    then compiles the final PRD. Reports progress via callback.

    Args:
        idea: Product idea description.
        target_users: Target user description.
        domain: Industry/domain (optional).
        constraints: Known constraints (optional).
        provider: LLM provider ("gemini" or "openai").
        api_key: API key for the LLM provider.
        vector_store: FAISS vector store (or None if no docs uploaded).
        progress_callback: Optional callable(step: str, progress: float)
            to report progress to the UI.

    Returns:
        PipelineResult with all agent outputs and the final compiled PRD.
    """
    result = PipelineResult()

    try:
        # Initialize LLM
        llm = _get_llm(provider, api_key)

        # ── Step 1: Business Analyst Agent ────────────────────────
        if progress_callback:
            progress_callback("🔍 Business Analyst Agent is analyzing...", 0.1)

        result.ba_output = run_business_analyst_agent(
            llm=llm,
            idea=idea,
            target_users=target_users,
            domain=domain,
            constraints=constraints,
            vector_store=vector_store,
        )

        if not result.ba_output.success:
            result.success = False
            result.error = f"Business Analyst failed: {result.ba_output.error}"
            return result

        # ── Step 2: Product Manager Agent ─────────────────────────
        if progress_callback:
            progress_callback("📋 Product Manager Agent is generating requirements...", 0.4)

        result.pm_output = run_product_manager_agent(
            llm=llm,
            idea=idea,
            ba_output=result.ba_output.output,
            vector_store=vector_store,
        )

        if not result.pm_output.success:
            result.success = False
            result.error = f"Product Manager failed: {result.pm_output.error}"
            return result

        # ── Step 3: Risk Reviewer Agent ───────────────────────────
        if progress_callback:
            progress_callback("⚠️ Risk Reviewer Agent is reviewing...", 0.65)

        result.risk_output = run_risk_reviewer_agent(
            llm=llm,
            idea=idea,
            ba_output=result.ba_output.output,
            pm_output=result.pm_output.output,
            vector_store=vector_store,
        )

        if not result.risk_output.success:
            result.success = False
            result.error = f"Risk Reviewer failed: {result.risk_output.error}"
            return result

        # ── Step 4: Compile Final PRD ─────────────────────────────
        if progress_callback:
            progress_callback("📄 Compiling final PRD...", 0.85)

        result.final_prd = compile_final_prd(
            llm=llm,
            idea=idea,
            ba_output=result.ba_output.output,
            pm_output=result.pm_output.output,
            risk_output=result.risk_output.output,
        )

        if progress_callback:
            progress_callback("✅ PRD generated successfully!", 1.0)

        result.success = True
        return result

    except Exception as e:
        result.success = False
        result.error = f"Pipeline error: {str(e)}"
        return result
