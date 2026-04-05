"""
FastAPI application for the Spine AI Research Platform.

Provides REST endpoints and Server-Sent Events (SSE) for real-time
research workflow streaming.
"""

import asyncio
import json
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from spine_ai.agent import ResearchAgent

BASE_DIR = Path(__file__).resolve().parent.parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(
    title="Spine AI Research Platform",
    description="AI-powered research agent with real-time workflow visualization",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Global research agent instance
agent = ResearchAgent()


class ResearchRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=500,
                       description="The research query to investigate")


class SessionResponse(BaseModel):
    id: str
    query: str
    status: str


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the main research interface."""
    return templates.TemplateResponse(request, "index.html")


@app.post("/api/research", response_model=SessionResponse)
async def start_research(req: ResearchRequest):
    """Create a new research session."""
    session = agent.create_session(req.query)
    return SessionResponse(id=session.id, query=session.query, status=session.status)


@app.get("/api/research/{session_id}/stream")
async def stream_research(session_id: str):
    """Stream research workflow events via Server-Sent Events."""
    session = agent.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    async def event_generator():
        async for event in agent.run_research(session_id):
            data = json.dumps(event)
            yield f"data: {data}\n\n"
        yield "data: {\"type\": \"done\"}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/research/{session_id}")
async def get_session(session_id: str):
    """Get the current state of a research session."""
    session = agent.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session.to_dict()


@app.get("/api/sessions")
async def list_sessions():
    """List all research sessions."""
    return agent.list_sessions()
