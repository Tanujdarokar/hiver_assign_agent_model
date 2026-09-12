"""
AppleSupport AI Agent Web Server
================================
FastAPI Web Application providing an interactive testing dashboard and API endpoints
for the AppleSupport Customer Support AI Agent.
"""

import sys
import json
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))

from src.agent import AppleSupportAgent

app = FastAPI(
    title="AppleSupport AI Support Agent",
    description="Interactive Web Testing Dashboard for AppleSupport Customer Service AI Agent",
    version="1.0.0"
)

# Mount static files
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")

# Initialize global agent instance
print("Initializing AppleSupport AI Agent for Web Server ...")
agent = AppleSupportAgent()


class ChatRequest(BaseModel):
    message: str


@app.get("/", response_class=FileResponse)
async def serve_dashboard():
    """Serves the main web dashboard."""
    return FileResponse(ROOT / "templates" / "index.html")


@app.post("/api/chat")
async def handle_chat(req: ChatRequest):
    """Processes incoming customer inquiry and returns structured agent output."""
    message = req.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Customer message cannot be empty.")

    try:
        result = agent.handle(message)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent processing error: {str(e)}")


@app.get("/api/intents")
async def get_intents():
    """Returns the full intent taxonomy definitions."""
    intents_file = ROOT / "data" / "processed" / "intents.json"
    if intents_file.exists():
        with open(intents_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return JSONResponse(content=data)
    raise HTTPException(status_code=404, detail="Intents definition file not found.")


if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 70)
    print("STARTING APPLESUPPORT AI AGENT WEB SERVER")
    print("Open http://127.0.0.1:8000 in your browser to test the agent interactively!")
    print("=" * 70 + "\n")
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
