import os
import sys
from dotenv import load_dotenv

# Force UTF-8 so Windows doesn't crash on emojis in Claude's responses
os.environ["PYTHONIOENCODING"] = "utf-8"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
from fastapi import FastAPI, Request, Depends, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel
import secrets

from agents.orchestrator import run_orchestrator

load_dotenv()

app = FastAPI()

# Serve the static folder so FastAPI can send your HTML page to the browser
app.mount("/static", StaticFiles(directory="static"), name="static")

# --- Simple password protection ---
# This is how you keep the page private. Only someone with the right
# username and password can access it.
security = HTTPBasic()

VALID_USERNAME = "admin"
VALID_PASSWORD = "changeme"  # Change this to something strong

def require_auth(credentials: HTTPBasicCredentials = Depends(security)):
    # secrets.compare_digest is used instead of == to prevent timing attacks
    valid_user = secrets.compare_digest(credentials.username, VALID_USERNAME)
    valid_pass = secrets.compare_digest(credentials.password, VALID_PASSWORD)
    if not (valid_user and valid_pass):
        raise HTTPException(status_code=401, detail="Unauthorized")
    return credentials


# --- Request/response models ---
# Pydantic models define the shape of data coming in and going out.
# FastAPI uses these to automatically validate requests.
class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str


# --- Routes ---
# A "route" is a URL path that FastAPI listens on.

@app.get("/", response_class=HTMLResponse)
def serve_homepage(credentials: HTTPBasicCredentials = Depends(require_auth)):
    # When you visit the root URL, serve the HTML page
    with open("static/index.html") as f:
        return f.read()

@app.post("/chat")
def chat(request: ChatRequest, credentials: HTTPBasicCredentials = Depends(require_auth)):
    try:
        response = run_orchestrator(request.message)
        # Encode to UTF-8 and back to strip any characters Windows can't handle
        response = response.encode("utf-8", errors="replace").decode("utf-8")
        return ChatResponse(response=response)
    except Exception as e:
        return ChatResponse(response=f"Error: {str(e)}")
