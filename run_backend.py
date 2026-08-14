#!/usr/bin/env python3
"""Start the FastAPI backend.

    python run_backend.py            # http://localhost:8000, docs at /docs
    python run_backend.py --reload   # auto-reload for development
"""

from __future__ import annotations

import argparse

import uvicorn
from dotenv import load_dotenv

from app.core.config import get_settings
from app.core.logging_config import configure_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Home Loan RAG API")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true", help="reload on code changes")
    arguments = parser.parse_args()

    load_dotenv()
    settings = get_settings()
    configure_logging(settings.app.log_level)

    uvicorn.run(
        "app.api.main:app",
        host=arguments.host,
        port=arguments.port,
        reload=arguments.reload,
        log_level=settings.app.log_level.lower(),
    )


if __name__ == "__main__":
    main()
