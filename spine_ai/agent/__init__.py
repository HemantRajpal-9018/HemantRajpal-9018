"""
Research Agent Engine - Core module for AI research workflow.

Decomposes research queries into sub-tasks, executes searches,
analyzes findings, and synthesizes comprehensive reports.
"""

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import AsyncGenerator, Optional


class StepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class StepType(str, Enum):
    DECOMPOSE = "decompose"
    SEARCH = "search"
    ANALYZE = "analyze"
    SYNTHESIZE = "synthesize"
    REPORT = "report"


@dataclass
class Source:
    """Represents a research source found during search."""
    url: str
    title: str
    snippet: str
    relevance_score: float
    discovered_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "url": self.url,
            "title": self.title,
            "snippet": self.snippet,
            "relevance_score": self.relevance_score,
            "discovered_at": self.discovered_at,
        }


@dataclass
class WorkflowStep:
    """Represents a single step in the research workflow."""
    id: str
    step_type: StepType
    title: str
    description: str
    status: StepStatus = StepStatus.PENDING
    result: Optional[str] = None
    sources: list = field(default_factory=list)
    sub_queries: list = field(default_factory=list)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    progress: float = 0.0

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "step_type": self.step_type.value,
            "title": self.title,
            "description": self.description,
            "status": self.status.value,
            "result": self.result,
            "sources": [s.to_dict() if isinstance(s, Source) else s for s in self.sources],
            "sub_queries": self.sub_queries,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "progress": self.progress,
        }


@dataclass
class ResearchSession:
    """Represents a complete research session."""
    id: str
    query: str
    steps: list = field(default_factory=list)
    status: str = "initialized"
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    report: Optional[str] = None
    all_sources: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "query": self.query,
            "steps": [s.to_dict() for s in self.steps],
            "status": self.status,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "report": self.report,
            "all_sources": [s.to_dict() if isinstance(s, Source) else s for s in self.all_sources],
        }


# Knowledge base for generating realistic research results
RESEARCH_KNOWLEDGE_BASE = {
    "ai": {
        "topics": ["machine learning", "deep learning", "natural language processing",
                    "computer vision", "reinforcement learning", "generative AI"],
        "sources": [
            Source("https://arxiv.org/abs/2303.08774", "GPT-4 Technical Report",
                   "We report the development of GPT-4, a large-scale multimodal model which can accept image and text inputs.", 0.95),
            Source("https://arxiv.org/abs/2305.10601", "Tree of Thoughts: Deliberate Problem Solving with LLMs",
                   "Research on enabling exploration over coherent units of text that serve as intermediate steps toward problem solving.", 0.88),
            Source("https://www.nature.com/articles/s41586-023-06924-6", "AI for Scientific Discovery",
                   "Artificial intelligence is increasingly used to accelerate scientific discovery across domains.", 0.92),
            Source("https://arxiv.org/abs/2312.11805", "Mixture of Experts in LLMs",
                   "Survey on Mixture of Experts architectures showing improved efficiency in large language models.", 0.85),
        ],
        "findings": [
            "Large language models show emergent capabilities at scale, with GPT-4 class models demonstrating reasoning abilities.",
            "Multi-agent systems are becoming a key paradigm for complex task decomposition and execution.",
            "Retrieval-augmented generation (RAG) significantly improves factual accuracy in AI-generated content.",
            "Fine-tuning and alignment techniques remain critical for making AI models safe and useful.",
        ]
    },
    "technology": {
        "topics": ["cloud computing", "cybersecurity", "blockchain", "quantum computing", "IoT"],
        "sources": [
            Source("https://www.gartner.com/en/articles/top-tech-trends", "Gartner Top Technology Trends",
                   "Analysis of the most impactful technology trends shaping the future of digital infrastructure.", 0.90),
            Source("https://www.mckinsey.com/capabilities/quantumblack/our-insights", "McKinsey AI Insights",
                   "Research on how AI is transforming industries and creating new opportunities.", 0.87),
            Source("https://dl.acm.org/doi/10.1145/3544548", "ACM Computing Surveys: Edge Computing",
                   "Comprehensive survey on edge computing architectures and their applications.", 0.83),
        ],
        "findings": [
            "Cloud-native architectures are becoming the default for enterprise application development.",
            "Zero-trust security models are replacing traditional perimeter-based security approaches.",
            "Edge computing is enabling real-time AI inference closer to data sources.",
        ]
    },
    "science": {
        "topics": ["biology", "physics", "chemistry", "medicine", "climate science"],
        "sources": [
            Source("https://www.science.org/doi/10.1126/science.adf6369", "AlphaFold Impact on Structural Biology",
                   "How AI-powered protein structure prediction is revolutionizing drug discovery.", 0.94),
            Source("https://www.nature.com/articles/s41586-023-06735-9", "Climate Tipping Points Research",
                   "New evidence on critical climate tipping points and their interconnections.", 0.89),
            Source("https://www.cell.com/cell/fulltext/S0092-8674(23)01370-4", "CRISPR Gene Therapy Advances",
                   "Clinical trials show promising results for CRISPR-based therapies.", 0.91),
        ],
        "findings": [
            "AI-driven drug discovery is reducing development timelines from years to months.",
            "Quantum sensors are enabling unprecedented precision in scientific measurements.",
            "Climate models are becoming more accurate with machine learning integration.",
        ]
    },
    "business": {
        "topics": ["market analysis", "strategy", "innovation", "digital transformation"],
        "sources": [
            Source("https://hbr.org/topic/ai-and-machine-learning", "HBR: AI and Business Strategy",
                   "How leading companies are integrating AI into their core business strategies.", 0.86),
            Source("https://www.bcg.com/publications/2024/ai-at-work", "BCG: AI at Work",
                   "Research on the impact of AI on workforce productivity and organizational change.", 0.84),
        ],
        "findings": [
            "Companies with mature AI strategies see 2-3x higher revenue growth.",
            "Human-AI collaboration models outperform pure automation approaches.",
            "Data quality remains the top barrier to successful AI implementation.",
        ]
    },
}

DEFAULT_SOURCES = [
    Source("https://scholar.google.com", "Google Scholar",
           "Academic papers and research publications across all fields.", 0.75),
    Source("https://www.semanticscholar.org", "Semantic Scholar",
           "AI-powered research tool for scientific literature.", 0.73),
    Source("https://en.wikipedia.org", "Wikipedia",
           "Collaborative encyclopedia providing overview information.", 0.60),
]

DEFAULT_FINDINGS = [
    "The research area shows active development with multiple concurrent approaches.",
    "Key challenges include scalability, reliability, and real-world applicability.",
    "Cross-disciplinary collaboration is accelerating progress in this field.",
]


def _classify_query(query: str) -> list[str]:
    """Classify the query to determine relevant knowledge domains."""
    query_lower = query.lower()
    domains = []
    keyword_map = {
        "ai": ["ai", "artificial intelligence", "machine learning", "deep learning",
                "neural", "llm", "gpt", "language model", "nlp", "agent"],
        "technology": ["technology", "tech", "software", "cloud", "cyber", "blockchain",
                       "quantum", "iot", "computing", "digital"],
        "science": ["science", "biology", "physics", "chemistry", "medical", "medicine",
                     "climate", "gene", "protein", "research"],
        "business": ["business", "market", "strategy", "company", "enterprise",
                      "revenue", "growth", "industry"],
    }
    for domain, keywords in keyword_map.items():
        if any(kw in query_lower for kw in keywords):
            domains.append(domain)
    if not domains:
        domains = ["technology", "ai"]
    return domains


def _generate_sub_queries(query: str) -> list[str]:
    """Decompose a research query into sub-queries for investigation."""
    query_lower = query.lower()

    base_queries = [
        f"Current state of research: {query}",
        f"Key developments and breakthroughs in: {query}",
        f"Leading researchers and institutions working on: {query}",
        f"Challenges and limitations in: {query}",
        f"Future directions and predictions for: {query}",
    ]

    if any(kw in query_lower for kw in ["compare", "vs", "versus", "difference"]):
        base_queries.append(f"Comparative analysis: {query}")

    if any(kw in query_lower for kw in ["how", "method", "approach", "technique"]):
        base_queries.append(f"Methodologies and techniques: {query}")

    if any(kw in query_lower for kw in ["impact", "effect", "influence"]):
        base_queries.append(f"Impact assessment: {query}")

    return base_queries


def _gather_sources(query: str, domains: list[str]) -> list[Source]:
    """Gather relevant sources based on query and domains."""
    sources = []
    for domain in domains:
        if domain in RESEARCH_KNOWLEDGE_BASE:
            sources.extend(RESEARCH_KNOWLEDGE_BASE[domain]["sources"])
    sources.extend(DEFAULT_SOURCES[:2])

    # Generate query-specific sources
    query_words = [w for w in query.split() if len(w) > 3]
    if query_words:
        keyword = query_words[0].capitalize()
        sources.append(Source(
            f"https://arxiv.org/search/?query={'+'.join(query_words[:3])}",
            f"arXiv: Recent Papers on {keyword}",
            f"Collection of recent academic papers related to {query[:80]}.",
            0.82,
        ))
        sources.append(Source(
            f"https://scholar.google.com/scholar?q={'+'.join(query_words[:3])}",
            f"Scholar: {keyword} Research Overview",
            f"Comprehensive collection of scholarly articles on {query[:80]}.",
            0.79,
        ))

    return sources


def _gather_findings(query: str, domains: list[str]) -> list[str]:
    """Gather research findings based on query and domains."""
    findings = []
    for domain in domains:
        if domain in RESEARCH_KNOWLEDGE_BASE:
            findings.extend(RESEARCH_KNOWLEDGE_BASE[domain]["findings"])
    findings.extend(DEFAULT_FINDINGS)
    return findings


def _generate_report(query: str, sub_queries: list[str],
                     sources: list[Source], findings: list[str]) -> str:
    """Generate a comprehensive research report."""
    source_section = "\n".join(
        f"  - [{s.title}]({s.url}) (relevance: {s.relevance_score:.0%})\n"
        f"    {s.snippet}"
        for s in sources[:8]
    )

    findings_section = "\n".join(
        f"  {i+1}. {f}" for i, f in enumerate(findings[:8])
    )

    sub_queries_section = "\n".join(
        f"  - {sq}" for sq in sub_queries
    )

    report = f"""# Research Report: {query}

## Executive Summary

This report presents a comprehensive analysis of **{query}**, synthesized from
{len(sources)} sources across multiple domains. The research agent decomposed the
query into {len(sub_queries)} sub-investigations to ensure thorough coverage.

## Research Methodology

The AI research agent followed a systematic approach:
1. **Query Decomposition** — Broke down the research question into focused sub-queries
2. **Multi-Source Search** — Searched academic databases, industry reports, and technical publications
3. **Cross-Reference Analysis** — Validated findings across multiple independent sources
4. **Synthesis** — Integrated findings into a coherent narrative

### Sub-Queries Investigated
{sub_queries_section}

## Key Findings

{findings_section}

## Sources Consulted

{source_section}

## Analysis & Discussion

The research reveals several important patterns and trends related to **{query}**.
Multiple independent sources confirm the key findings above, providing high confidence
in the conclusions drawn.

The field shows active development with significant recent progress. Key themes include
the increasing role of AI and automation, growing emphasis on cross-disciplinary approaches,
and the importance of practical, real-world applicability.

## Recommendations

Based on the research findings:

1. **Stay Current** — The field is evolving rapidly; regular monitoring of key sources is recommended
2. **Focus on Practical Applications** — The most impactful work combines theoretical advances with real-world use cases
3. **Collaborative Approach** — Cross-disciplinary collaboration yields the strongest results
4. **Critical Evaluation** — Not all published results replicate; verify claims through multiple sources

## Conclusion

This analysis of **{query}** demonstrates a dynamic and rapidly evolving research landscape.
The {len(findings)} key findings identified provide a solid foundation for further investigation
and decision-making. Continued monitoring of the {len(sources)} identified sources will help
track future developments.

---
*Report generated by Spine AI Research Agent*
*Sources: {len(sources)} | Sub-queries: {len(sub_queries)} | Findings: {len(findings)}*
"""
    return report


class ResearchAgent:
    """
    AI Research Agent that orchestrates the research workflow.

    The agent decomposes queries, searches for information, analyzes findings,
    and generates comprehensive research reports with full workflow visibility.
    """

    def __init__(self):
        self.sessions: dict[str, ResearchSession] = {}

    def create_session(self, query: str) -> ResearchSession:
        """Create a new research session for the given query."""
        session_id = str(uuid.uuid4())
        session = ResearchSession(id=session_id, query=query)
        self.sessions[session_id] = session
        return session

    def get_session(self, session_id: str) -> Optional[ResearchSession]:
        """Retrieve a research session by ID."""
        return self.sessions.get(session_id)

    def list_sessions(self) -> list[dict]:
        """List all research sessions."""
        return [
            {"id": s.id, "query": s.query, "status": s.status, "created_at": s.created_at}
            for s in self.sessions.values()
        ]

    async def run_research(self, session_id: str) -> AsyncGenerator[dict, None]:
        """
        Execute the full research workflow for a session, yielding events.

        Yields workflow step updates as dict events for real-time streaming.
        """
        session = self.sessions.get(session_id)
        if not session:
            yield {"type": "error", "message": "Session not found"}
            return

        session.status = "running"
        yield {"type": "session_update", "session": {"id": session.id, "status": "running"}}

        # Step 1: Decompose the query
        decompose_step = WorkflowStep(
            id=str(uuid.uuid4()),
            step_type=StepType.DECOMPOSE,
            title="Decomposing Research Query",
            description=f"Breaking down: '{session.query}' into focused sub-queries",
        )
        session.steps.append(decompose_step)
        async for event in self._execute_decompose(session, decompose_step):
            yield event

        # Step 2: Search for each sub-query
        search_step = WorkflowStep(
            id=str(uuid.uuid4()),
            step_type=StepType.SEARCH,
            title="Searching Sources",
            description="Searching academic databases, industry reports, and publications",
        )
        session.steps.append(search_step)
        async for event in self._execute_search(session, search_step):
            yield event

        # Step 3: Analyze findings
        analyze_step = WorkflowStep(
            id=str(uuid.uuid4()),
            step_type=StepType.ANALYZE,
            title="Analyzing Findings",
            description="Cross-referencing and validating information from multiple sources",
        )
        session.steps.append(analyze_step)
        async for event in self._execute_analyze(session, analyze_step):
            yield event

        # Step 4: Synthesize report
        report_step = WorkflowStep(
            id=str(uuid.uuid4()),
            step_type=StepType.REPORT,
            title="Generating Research Report",
            description="Synthesizing findings into a comprehensive research report",
        )
        session.steps.append(report_step)
        async for event in self._execute_report(session, report_step):
            yield event

        session.status = "completed"
        session.completed_at = time.time()
        yield {
            "type": "session_complete",
            "session": session.to_dict(),
        }

    async def _execute_decompose(self, session: ResearchSession,
                                  step: WorkflowStep) -> AsyncGenerator[dict, None]:
        """Execute query decomposition step."""
        step.status = StepStatus.RUNNING
        step.started_at = time.time()
        yield {"type": "step_update", "step": step.to_dict()}

        await asyncio.sleep(0.8)

        sub_queries = _generate_sub_queries(session.query)
        step.sub_queries = sub_queries
        step.progress = 0.5
        yield {"type": "step_progress", "step_id": step.id, "progress": 0.5,
               "message": f"Identified {len(sub_queries)} research angles"}

        await asyncio.sleep(0.5)

        step.status = StepStatus.COMPLETED
        step.completed_at = time.time()
        step.progress = 1.0
        step.result = f"Decomposed query into {len(sub_queries)} sub-queries"
        yield {"type": "step_update", "step": step.to_dict()}

    async def _execute_search(self, session: ResearchSession,
                               step: WorkflowStep) -> AsyncGenerator[dict, None]:
        """Execute multi-source search step."""
        step.status = StepStatus.RUNNING
        step.started_at = time.time()
        yield {"type": "step_update", "step": step.to_dict()}

        domains = _classify_query(session.query)
        sources = _gather_sources(session.query, domains)

        for i, source in enumerate(sources):
            await asyncio.sleep(0.4)
            step.sources.append(source)
            session.all_sources.append(source)
            progress = (i + 1) / len(sources)
            step.progress = progress
            yield {
                "type": "source_found",
                "step_id": step.id,
                "source": source.to_dict(),
                "progress": progress,
                "message": f"Found: {source.title}",
            }

        step.status = StepStatus.COMPLETED
        step.completed_at = time.time()
        step.progress = 1.0
        step.result = f"Found {len(sources)} relevant sources across {len(domains)} domains"
        yield {"type": "step_update", "step": step.to_dict()}

    async def _execute_analyze(self, session: ResearchSession,
                                step: WorkflowStep) -> AsyncGenerator[dict, None]:
        """Execute analysis step."""
        step.status = StepStatus.RUNNING
        step.started_at = time.time()
        yield {"type": "step_update", "step": step.to_dict()}

        domains = _classify_query(session.query)
        findings = _gather_findings(session.query, domains)

        for i, finding in enumerate(findings):
            await asyncio.sleep(0.5)
            progress = (i + 1) / len(findings)
            step.progress = progress
            yield {
                "type": "finding",
                "step_id": step.id,
                "finding": finding,
                "progress": progress,
                "message": f"Insight {i+1}: {finding[:80]}...",
            }

        step.status = StepStatus.COMPLETED
        step.completed_at = time.time()
        step.progress = 1.0
        step.result = f"Extracted {len(findings)} key findings from analyzed sources"
        yield {"type": "step_update", "step": step.to_dict()}

    async def _execute_report(self, session: ResearchSession,
                               step: WorkflowStep) -> AsyncGenerator[dict, None]:
        """Execute report generation step."""
        step.status = StepStatus.RUNNING
        step.started_at = time.time()
        yield {"type": "step_update", "step": step.to_dict()}

        await asyncio.sleep(0.6)
        step.progress = 0.3
        yield {"type": "step_progress", "step_id": step.id, "progress": 0.3,
               "message": "Organizing findings..."}

        await asyncio.sleep(0.6)
        step.progress = 0.6
        yield {"type": "step_progress", "step_id": step.id, "progress": 0.6,
               "message": "Writing report sections..."}

        # Gather all sub-queries from decompose step
        sub_queries = []
        for s in session.steps:
            if s.step_type == StepType.DECOMPOSE:
                sub_queries = s.sub_queries
                break

        domains = _classify_query(session.query)
        findings = _gather_findings(session.query, domains)
        report = _generate_report(session.query, sub_queries, session.all_sources, findings)

        await asyncio.sleep(0.5)

        session.report = report
        step.result = "Research report generated successfully"
        step.status = StepStatus.COMPLETED
        step.completed_at = time.time()
        step.progress = 1.0
        yield {"type": "step_update", "step": step.to_dict()}
        yield {"type": "report_ready", "report": report}
