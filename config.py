"""
config.py — Application configuration and constants for Idea2PRD.

Contains chunk sizes, model defaults, sample test inputs, and UI settings.
"""

# ─── RAG Configuration ───────────────────────────────────────────────
CHUNK_SIZE = 1000          # Characters per chunk for text splitting
CHUNK_OVERLAP = 200        # Overlap between consecutive chunks
TOP_K_RETRIEVAL = 5        # Number of top chunks to retrieve per query

# ─── Model Defaults ──────────────────────────────────────────────────
GEMINI_MODEL = "gemini-2.0-flash"
OPENAI_MODEL = "gpt-4o-mini"
GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GEMINI_EMBEDDING_MODEL = "models/embedding-001"
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"

# ─── LLM Parameters ──────────────────────────────────────────────────
LLM_TEMPERATURE = 0.4      # Balanced: creative but grounded
LLM_MAX_TOKENS = 4096      # Max output tokens per agent call

# ─── Sample Test Inputs ──────────────────────────────────────────────
# Three diverse product ideas for demo and evaluation purposes.

SAMPLE_INPUTS = [
    {
        "name": "🎓 AI Study Planner",
        "idea": (
            "An AI-powered study planner app for college students that automatically "
            "creates personalized study schedules based on their course syllabi, exam "
            "dates, and learning pace. The app should adapt the schedule when students "
            "fall behind or get ahead, suggest optimal study techniques for different "
            "subjects, and integrate with university LMS platforms like Canvas or Moodle."
        ),
        "target_users": (
            "College and university students (18-25 years old), particularly those "
            "juggling multiple courses, part-time jobs, or extracurricular activities."
        ),
        "domain": "EdTech / Education Technology",
        "constraints": (
            "Must work offline for basic scheduling. Free tier required for student "
            "adoption. Must comply with FERPA for any university data integration. "
            "Should not require more than 2 minutes to set up initially."
        ),
    },
    {
        "name": "🏘️ Neighborhood Services Marketplace",
        "idea": (
            "A hyperlocal services marketplace that connects neighbors for everyday "
            "tasks like pet sitting, lawn mowing, tutoring, handyman work, and grocery "
            "pickups. Unlike TaskRabbit, this focuses on building community trust through "
            "neighborhood-level verification, reputation scores, and a community bulletin "
            "board. Includes a simple booking and payment system."
        ),
        "target_users": (
            "Suburban and urban homeowners (30-55 years old), retirees looking for "
            "supplemental income, college students seeking flexible gigs, and busy "
            "parents who need reliable local help."
        ),
        "domain": "Local Services / Gig Economy",
        "constraints": (
            "Must support in-app payments with escrow. Need identity verification for "
            "service providers. Must comply with local labor regulations. Should work "
            "in areas with as few as 50 active users to avoid the cold-start problem."
        ),
    },
    {
        "name": "👴 Senior Health Companion",
        "idea": (
            "A simplified health tracking app designed specifically for elderly users "
            "(65+) that monitors daily vitals (blood pressure, blood sugar, medication "
            "adherence), sends gentle reminders, and shares weekly health summaries with "
            "designated family caregivers. Features extra-large UI elements, voice input, "
            "and emergency SOS functionality. Should integrate with common home health "
            "devices via Bluetooth."
        ),
        "target_users": (
            "Elderly individuals (65+ years old) managing chronic conditions, and their "
            "adult children or caregivers (35-55 years old) who want remote visibility "
            "into their parent's health."
        ),
        "domain": "HealthTech / Elder Care",
        "constraints": (
            "Must comply with HIPAA for health data. UI must pass WCAG 2.1 AA "
            "accessibility standards. Must work on Android devices (high elderly "
            "adoption). Battery usage must be minimal for background monitoring. "
            "Cannot replace medical advice — must include appropriate disclaimers."
        ),
    },
]

# ─── Evaluation Rubric ───────────────────────────────────────────────
EVALUATION_CRITERIA = [
    {
        "criterion": "Relevance",
        "description": "Are the PRD sections directly relevant to the stated product idea?",
        "scale": "1 (Off-topic) → 5 (Perfectly aligned)",
    },
    {
        "criterion": "Completeness",
        "description": "Does the PRD cover all expected sections without major gaps?",
        "scale": "1 (Many gaps) → 5 (Comprehensive)",
    },
    {
        "criterion": "Clarity",
        "description": "Is the language clear, unambiguous, and well-structured?",
        "scale": "1 (Confusing) → 5 (Crystal clear)",
    },
    {
        "criterion": "Consistency",
        "description": "Are there contradictions between sections or agent outputs?",
        "scale": "1 (Contradictory) → 5 (Fully consistent)",
    },
]

QUALITY_CHECKLIST = [
    "Problem statement is specific and actionable",
    "Target users are clearly defined with demographics",
    "At least 2 user personas are provided",
    "User stories follow 'As a [user], I want [goal], so that [benefit]' format",
    "Functional requirements are testable and measurable",
    "Non-functional requirements cover performance, security, accessibility",
    "Success metrics are quantifiable (not vague)",
    "MVP scope is realistic and clearly separated from future scope",
    "Risks are specific with mitigation strategies",
    "Clarifying questions identify genuine ambiguities",
    "RAG context is cited where applicable",
    "Assumptions are explicitly stated when no documents are uploaded",
]
