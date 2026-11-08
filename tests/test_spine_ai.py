"""
Tests for the Spine AI Research Agent.
"""

import asyncio
import pytest

from spine_ai.agent import (
    ResearchAgent,
    ResearchSession,
    WorkflowStep,
    Source,
    StepStatus,
    StepType,
    _classify_query,
    _generate_sub_queries,
    _gather_sources,
    _gather_findings,
    _generate_report,
)


# ===== Unit Tests for Helper Functions =====

class TestClassifyQuery:
    def test_ai_keywords(self):
        domains = _classify_query("How do large language models work?")
        assert "ai" in domains

    def test_technology_keywords(self):
        domains = _classify_query("What is cloud computing?")
        assert "technology" in domains

    def test_science_keywords(self):
        domains = _classify_query("Latest advances in biology research")
        assert "science" in domains

    def test_business_keywords(self):
        domains = _classify_query("Market strategy for enterprise growth")
        assert "business" in domains

    def test_multiple_domains(self):
        domains = _classify_query("AI technology in business")
        assert len(domains) >= 2

    def test_default_domains(self):
        domains = _classify_query("random topic with no keywords")
        assert len(domains) >= 1  # Falls back to defaults


class TestGenerateSubQueries:
    def test_basic_decomposition(self):
        queries = _generate_sub_queries("AI agents")
        assert len(queries) >= 5
        assert any("AI agents" in q for q in queries)

    def test_comparison_query(self):
        queries = _generate_sub_queries("Compare GPT-4 vs Claude")
        assert any("Comparative" in q or "compare" in q.lower() for q in queries)

    def test_method_query(self):
        queries = _generate_sub_queries("How does reinforcement learning approach work?")
        assert any("Methodolog" in q or "technique" in q.lower() for q in queries)

    def test_impact_query(self):
        queries = _generate_sub_queries("Impact of AI on healthcare")
        assert any("Impact" in q or "impact" in q.lower() for q in queries)


class TestGatherSources:
    def test_returns_sources(self):
        sources = _gather_sources("AI research", ["ai"])
        assert len(sources) > 0
        assert all(isinstance(s, Source) for s in sources)

    def test_includes_domain_sources(self):
        sources = _gather_sources("AI agents", ["ai"])
        urls = [s.url for s in sources]
        assert any("arxiv" in u for u in urls)

    def test_generates_query_specific_sources(self):
        sources = _gather_sources("quantum computing breakthroughs", ["technology"])
        assert len(sources) > 2  # Should have domain + query-specific sources

    def test_source_has_required_fields(self):
        sources = _gather_sources("test query", ["ai"])
        for source in sources:
            assert source.url
            assert source.title
            assert source.snippet
            assert 0 <= source.relevance_score <= 1


class TestGatherFindings:
    def test_returns_findings(self):
        findings = _gather_findings("AI research", ["ai"])
        assert len(findings) > 0
        assert all(isinstance(f, str) for f in findings)

    def test_includes_default_findings(self):
        findings = _gather_findings("random topic", [])
        assert len(findings) >= 3  # At least the defaults


class TestGenerateReport:
    def test_generates_report(self):
        sources = [Source("http://example.com", "Test", "A test source", 0.9)]
        report = _generate_report("AI test", ["sub query 1"], sources, ["finding 1"])
        assert "AI test" in report
        assert "Research Report" in report
        assert "Key Findings" in report
        assert "Sources Consulted" in report

    def test_report_includes_sources(self):
        sources = [Source("http://example.com", "My Source", "Snippet text", 0.85)]
        report = _generate_report("test", ["q1"], sources, ["f1"])
        assert "My Source" in report

    def test_report_includes_findings(self):
        report = _generate_report("test", ["q1"], [], ["Important finding here"])
        assert "Important finding here" in report


# ===== Unit Tests for Data Models =====

class TestSource:
    def test_to_dict(self):
        source = Source("http://ex.com", "Title", "Snippet", 0.9)
        d = source.to_dict()
        assert d["url"] == "http://ex.com"
        assert d["title"] == "Title"
        assert d["snippet"] == "Snippet"
        assert d["relevance_score"] == 0.9

    def test_default_timestamp(self):
        source = Source("http://ex.com", "Title", "Snippet", 0.9)
        assert source.discovered_at > 0


class TestWorkflowStep:
    def test_default_status(self):
        step = WorkflowStep(id="1", step_type=StepType.SEARCH,
                            title="Test", description="Desc")
        assert step.status == StepStatus.PENDING

    def test_to_dict(self):
        step = WorkflowStep(id="1", step_type=StepType.ANALYZE,
                            title="Analyze", description="Analyzing data")
        d = step.to_dict()
        assert d["id"] == "1"
        assert d["step_type"] == "analyze"
        assert d["status"] == "pending"
        assert d["progress"] == 0.0


class TestResearchSession:
    def test_creation(self):
        session = ResearchSession(id="test-id", query="test query")
        assert session.id == "test-id"
        assert session.query == "test query"
        assert session.status == "initialized"
        assert session.steps == []

    def test_to_dict(self):
        session = ResearchSession(id="test-id", query="test query")
        d = session.to_dict()
        assert d["id"] == "test-id"
        assert d["query"] == "test query"
        assert d["status"] == "initialized"
        assert d["steps"] == []


# ===== Tests for ResearchAgent =====

class TestResearchAgent:
    def test_create_session(self):
        agent = ResearchAgent()
        session = agent.create_session("AI research")
        assert session.query == "AI research"
        assert session.id is not None
        assert session.status == "initialized"

    def test_get_session(self):
        agent = ResearchAgent()
        session = agent.create_session("test")
        retrieved = agent.get_session(session.id)
        assert retrieved is not None
        assert retrieved.id == session.id

    def test_get_nonexistent_session(self):
        agent = ResearchAgent()
        assert agent.get_session("nonexistent") is None

    def test_list_sessions(self):
        agent = ResearchAgent()
        agent.create_session("query 1")
        agent.create_session("query 2")
        sessions = agent.list_sessions()
        assert len(sessions) == 2
        assert sessions[0]["query"] == "query 1"
        assert sessions[1]["query"] == "query 2"

    @pytest.mark.asyncio
    async def test_run_research_full_workflow(self):
        agent = ResearchAgent()
        session = agent.create_session("AI agents in research")
        events = []

        async for event in agent.run_research(session.id):
            events.append(event)

        # Should have multiple events
        assert len(events) > 5

        # Should have session_update, step_updates, and session_complete
        event_types = [e["type"] for e in events]
        assert "session_update" in event_types
        assert "step_update" in event_types
        assert "session_complete" in event_types
        assert "report_ready" in event_types

        # Session should be completed
        assert session.status == "completed"
        assert session.report is not None
        assert len(session.steps) == 4  # decompose, search, analyze, report
        assert len(session.all_sources) > 0

    @pytest.mark.asyncio
    async def test_run_research_nonexistent_session(self):
        agent = ResearchAgent()
        events = []
        async for event in agent.run_research("nonexistent"):
            events.append(event)
        assert len(events) == 1
        assert events[0]["type"] == "error"

    @pytest.mark.asyncio
    async def test_run_research_sources_found(self):
        agent = ResearchAgent()
        session = agent.create_session("machine learning advances")
        source_events = []

        async for event in agent.run_research(session.id):
            if event["type"] == "source_found":
                source_events.append(event)

        assert len(source_events) > 0
        for se in source_events:
            assert "source" in se
            assert "url" in se["source"]
            assert "title" in se["source"]

    @pytest.mark.asyncio
    async def test_run_research_findings(self):
        agent = ResearchAgent()
        session = agent.create_session("deep learning applications")
        finding_events = []

        async for event in agent.run_research(session.id):
            if event["type"] == "finding":
                finding_events.append(event)

        assert len(finding_events) > 0
        for fe in finding_events:
            assert "finding" in fe
            assert len(fe["finding"]) > 0


# ===== Tests for the API =====

class TestAPI:
    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from spine_ai.api.app import app
        return TestClient(app)

    def test_home_page(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert "Spine AI" in response.text

    def test_start_research(self, client):
        response = client.post("/api/research", json={"query": "AI agents"})
        assert response.status_code == 200
        data = response.json()
        assert "id" in data
        assert data["query"] == "AI agents"
        assert data["status"] == "initialized"

    def test_start_research_short_query(self, client):
        response = client.post("/api/research", json={"query": "ab"})
        assert response.status_code == 422  # Validation error

    def test_get_session(self, client):
        # First create a session
        create_resp = client.post("/api/research", json={"query": "test query"})
        session_id = create_resp.json()["id"]

        # Then get it
        response = client.get(f"/api/research/{session_id}")
        assert response.status_code == 200
        assert response.json()["query"] == "test query"

    def test_get_nonexistent_session(self, client):
        response = client.get("/api/research/nonexistent-id")
        assert response.status_code == 404

    def test_list_sessions(self, client):
        response = client.get("/api/sessions")
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    def test_stream_research(self, client):
        # Create session first
        create_resp = client.post("/api/research", json={"query": "streaming test"})
        session_id = create_resp.json()["id"]

        # Stream events
        with client.stream("GET", f"/api/research/{session_id}/stream") as response:
            assert response.status_code == 200
            events = []
            for line in response.iter_lines():
                if line.startswith("data: "):
                    events.append(line)
                if len(events) >= 3:
                    break
            assert len(events) >= 3

    def test_stream_nonexistent_session(self, client):
        response = client.get("/api/research/nonexistent-id/stream")
        assert response.status_code == 404
