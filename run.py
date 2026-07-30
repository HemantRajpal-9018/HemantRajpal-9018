"""
Run the Spine AI Research Platform server.

Usage:
    python run.py
"""

import uvicorn


def main():
    """Start the Spine AI server."""
    uvicorn.run(
        "spine_ai.api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )


if __name__ == "__main__":
    main()
