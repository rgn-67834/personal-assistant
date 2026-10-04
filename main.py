import os
import sys
from dotenv import load_dotenv

# Force UTF-8 so Windows doesn't crash on emojis in Claude's responses
os.environ["PYTHONIOENCODING"] = "utf-8"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
import anthropic
from fastapi import FastAPI, Header
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from typing import Literal

from pydantic import BaseModel

from agents.client import use_api_key
from agents.loop import collect_trace
from agents.orchestrator import run_orchestrator

load_dotenv()

app = FastAPI()

# Serve the static folder so FastAPI can send your HTML page to the browser
app.mount("/static", StaticFiles(directory="static"), name="static")


# --- Request/response models ---
# Pydantic models define the shape of data coming in and going out.
# FastAPI uses these to automatically validate requests.
class HistoryTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str

class ChatRequest(BaseModel):
    message: str
    # Earlier turns of this chat. The server keeps no conversation state, so
    # the browser sends the history back with every message.
    history: list[HistoryTurn] = []

class ChatResponse(BaseModel):
    response: str
    # Every model call and tool call the agents made to produce the response
    trace: list[dict] = []


MAX_HISTORY_TURNS = 20


# --- Routes ---
# A "route" is a URL path that FastAPI listens on.

@app.get("/", response_class=HTMLResponse)
def serve_homepage():
    # When you visit the root URL, serve the HTML page
    with open("static/index.html", encoding="utf-8") as f:
        return f.read()

# --- Bring your own key ---
# The page is open to anyone, but every chat request must carry the visitor's
# own Anthropic API key in the X-Anthropic-Key header. The key is used for that
# one request and is never stored or logged. There is deliberately no fallback
# to a server-side key, so a deployment can never spend the owner's credits.
@app.post("/chat")
def chat(request: ChatRequest, x_anthropic_key: str | None = Header(default=None)):
    api_key = (x_anthropic_key or "").strip()
    if not api_key:
        return JSONResponse(status_code=401, content={"response": "Add your Anthropic API key to start."})
    # Only the most recent turns go back to Claude, which bounds the cost of a
    # long chat. Dropping from the front can leave an assistant turn first, and
    # a conversation has to open with a user turn.
    history = [turn.model_dump() for turn in request.history[-MAX_HISTORY_TURNS:]]
    while history and history[0]["role"] != "user":
        history.pop(0)

    with collect_trace() as trace:
        try:
            with use_api_key(api_key):
                response = run_orchestrator(request.message, history)
            # Encode to UTF-8 and back to strip any characters Windows can't handle
            response = response.encode("utf-8", errors="replace").decode("utf-8")
            return ChatResponse(response=response, trace=trace)
        except anthropic.AuthenticationError:
            return JSONResponse(status_code=401, content={"response": "That API key was rejected by Anthropic. Check it and try again."})
        except anthropic.RateLimitError:
            return ChatResponse(response="Anthropic rate-limited this API key. Wait a moment and try again.", trace=trace)
        except anthropic.APIStatusError as e:
            return ChatResponse(response=f"Anthropic returned an error ({e.status_code}): {e.message}", trace=trace)
        except Exception as e:
            return ChatResponse(response=f"Error: {str(e)}", trace=trace)
