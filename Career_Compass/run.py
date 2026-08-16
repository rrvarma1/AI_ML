"""Run Career Compass locally without Docker.

Install dependencies once with:  python -m pip install -r requirements.txt
Install Chromium once with:       python -m playwright install chromium
Then start the portal with:       python run.py
"""

import asyncio
import sys
import uvicorn


if __name__ == "__main__":
    if sys.platform == "win32" and hasattr(asyncio, "WindowsProactorEventLoopPolicy"):
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

    uvicorn.run(
        "app.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        loop="asyncio",
    )
