# main.py
# To run this server: uvicorn main:app --reload
# Required packages are listed in the project_setup_guide artifact.

import asyncio
import base64
import re
import os
import shutil
import tempfile
from pathlib import Path
from datetime import datetime, timedelta
import logging

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Request
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel, HttpUrl
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
import httpx
from decouple import config
import google.generativeai as genai
from git import Repo, GitCommandError
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.docstore.document import Document
from jose import JWTError, jwt
from itsdangerous import URLSafeTimedSerializer

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration ---
GITHUB_CLIENT_ID = config('GITHUB_CLIENT_ID', default=None)
GITHUB_CLIENT_SECRET = config('GITHUB_CLIENT_SECRET', default=None)
SECRET_KEY = config('SECRET_KEY', default=None) 
GEMINI_API_KEY = config('GEMINI_API_KEY', default=None)
GITHUB_API_TOKEN = config('GITHUB_API_TOKEN', default=None) # Server's fallback token for public repos

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 120 
FRONTEND_URL = "http://localhost:3000"

# Validate essential configs
if not all([GEMINI_API_KEY, SECRET_KEY, GITHUB_CLIENT_ID, GITHUB_CLIENT_SECRET]):
    logger.error("FATAL: Required environment variables are missing. Check .env setup.")
    # This will prevent the app from starting if run directly, but helps in debugging.
    # In a real production deployment, this check should be more robust.
    raise ValueError("Required environment variables are missing. Check README for setup.")

genai.configure(api_key=GEMINI_API_KEY)
state_serializer = URLSafeTimedSerializer(SECRET_KEY)

# --- In-Memory Storage for Vector Stores & Status ---
vector_stores = {}
indexing_statuses = {}

# --- Pydantic Models ---
class AnalyzeRequest(BaseModel):
    url: HttpUrl

class QnaRequest(BaseModel):
    url: HttpUrl
    question: str

class TokenData(BaseModel):
    username: str | None = None

class CommitInfo(BaseModel):
    sha: str
    message: str
    author: str
    url: str

class IssueInfo(BaseModel):
    number: int
    title: str
    url: str

class AnalysisResponse(BaseModel):
    projectName: str
    description: str
    readmeContent: str
    techStack: list[str]
    recentCommits: list[CommitInfo]
    openIssues: list[IssueInfo]

# --- FastAPI App Initialization ---
app = FastAPI(
    title="GitHub Project Intelligence Assistant API",
    description="Backend service with authentication to analyze public and private repositories.",
    version="3.2.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token", auto_error=False)

# --- Security & JWT Functions ---
def create_access_token(data: dict, expires_delta: timedelta | None = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(token: str | None = Depends(oauth2_scheme)):
    if token is None: return None
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        if payload.get("sub") is None or payload.get("github_token") is None:
            return None
        return payload # Contains 'sub' (username) and 'github_token'
    except JWTError:
        return None

# --- GitHub API Helpers ---
def get_auth_headers(current_user: dict | None):
    if current_user and current_user.get("github_token"):
        return {"Authorization": f"token {current_user['github_token']}"}
    if GITHUB_API_TOKEN:
        return {"Authorization": f"token {GITHUB_API_TOKEN}"}
    return {}

def parse_github_url(url: str):
    match = re.search(r"github\.com/([\w\-]+)/([\w\-]+)", url)
    if not match: raise HTTPException(status_code=400, detail="Invalid GitHub URL format.")
    return match.groups()

# --- Authentication Endpoints ---
@app.get("/api/auth/login")
async def login():
    state = state_serializer.dumps("login_state")
    return RedirectResponse(f"https://github.com/login/oauth/authorize?client_id={GITHUB_CLIENT_ID}&scope=repo,read:user&state={state}")

@app.get("/api/auth/callback")
async def auth_callback(code: str, state: str):
    try:
        state_serializer.loads(state, max_age=300)
    except Exception:
        raise HTTPException(status_code=403, detail="Invalid state or state expired.")

    async with httpx.AsyncClient() as client:
        token_response = await client.post("https://github.com/login/oauth/access_token",
            data={"client_id": GITHUB_CLIENT_ID, "client_secret": GITHUB_CLIENT_SECRET, "code": code},
            headers={"Accept": "application/json"})
        token_data = token_response.json()
        github_token = token_data.get("access_token")
        if not github_token: raise HTTPException(status_code=400, detail="Could not retrieve access token.")
        
        user_response = await client.get("https://api.github.com/user", headers={"Authorization": f"token {github_token}"})
        user_data = user_response.json()

    jwt_data = {"sub": user_data.get("login"), "name": user_data.get("name"), "avatar_url": user_data.get("avatar_url"), "github_token": github_token}
    app_token = create_access_token(jwt_data, timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    return RedirectResponse(f"{FRONTEND_URL}?token={app_token}")

@app.get("/api/me")
async def read_users_me(current_user: dict = Depends(get_current_user)):
    if not current_user: return {"authenticated": False}
    payload = {k: v for k, v in current_user.items() if k != 'github_token'}
    payload["authenticated"] = True
    return payload

# --- Code Indexing Logic (Background Task) ---
def index_repository_task(repo_url_str: str, user_token: str | None):
    repo_url_key = repo_url_str
    try:
        owner, repo_name = parse_github_url(repo_url_str)
        indexing_statuses[repo_url_key] = {"status": "indexing", "detail": "Starting..."}
        
        temp_dir = tempfile.mkdtemp()
        
        clone_url = f"https://github.com/{owner}/{repo_name}.git"
        if user_token:
             clone_url = f"https://{user_token}@github.com/{owner}/{repo_name}.git"

        indexing_statuses[repo_url_key]["detail"] = "Cloning repository..."
        Repo.clone_from(clone_url, temp_dir, depth=1)
        
        docs = []
        root_path = Path(temp_dir)
        
        # Add more extensions if needed
        file_extensions = ['.py', '.js', '.jsx', '.ts', '.tsx', '.java', '.go', '.rs', '.md', '.json']
        
        indexing_statuses[repo_url_key]["detail"] = "Processing files..."
        for file_path in root_path.rglob('*'):
            if file_path.is_file() and file_path.suffix in file_extensions and '.git' not in str(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    docs.append(Document(page_content=content, metadata={"source": str(file_path.relative_to(root_path))}))
                except Exception:
                    continue # Skip files that can't be read

        if not docs:
            vector_stores[repo_url_key] = None
            indexing_statuses[repo_url_key] = {"status": "empty", "detail": "No indexable files found."}
            return

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
        split_docs = text_splitter.split_documents(docs)
        
        indexing_statuses[repo_url_key]["detail"] = "Creating embeddings..."
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.from_documents(split_docs, embeddings)
        
        vector_stores[repo_url_key] = vector_store
        indexing_statuses[repo_url_key] = {"status": "completed", "detail": "Indexing complete."}
        
    except GitCommandError as e:
        logger.error(f"Git error for {repo_url_key}: {e}")
        indexing_statuses[repo_url_key] = {"status": "failed", "detail": "Failed to clone. Ensure URL is correct and you have access."}
    except Exception as e:
        logger.error(f"Indexing failed for {repo_url_key}: {e}")
        indexing_statuses[repo_url_key] = {"status": "failed", "detail": f"An unexpected error occurred: {str(e)}"}
    finally:
        if 'temp_dir' in locals() and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

# --- Core API Endpoints ---
@app.post("/api/analyze", response_model=AnalysisResponse)
async def analyze_repository(req: AnalyzeRequest, background_tasks: BackgroundTasks, current_user: dict = Depends(get_current_user)):
    try:
        owner, repo_name = parse_github_url(str(req.url))
    except HTTPException as e:
        return JSONResponse(status_code=e.status_code, content={"detail": e.detail})
        
    headers = get_auth_headers(current_user)
    if not headers:
        raise HTTPException(status_code=401, detail="Authentication required for this operation or server GITHUB_API_TOKEN not set.")

    async with httpx.AsyncClient() as client:
        api_url = f"https://api.github.com/repos/{owner}/{repo_name}"
        try:
            # Fire all API calls concurrently
            repo_details_task = client.get(api_url, headers=headers)
            languages_task = client.get(f"{api_url}/languages", headers=headers)
            readme_task = client.get(f"{api_url}/readme", headers=headers)
            commits_task = client.get(f"{api_url}/commits?per_page=5", headers=headers)
            issues_task = client.get(f"{api_url}/issues?state=open&per_page=5", headers=headers)

            responses = await asyncio.gather(
                repo_details_task, languages_task, readme_task, commits_task, issues_task,
                return_exceptions=True
            )
            
            repo_details_resp, languages_resp, readme_resp, commits_resp, issues_resp = responses

            # Check for errors in critical responses
            if isinstance(repo_details_resp, Exception) or repo_details_resp.status_code != 200:
                raise HTTPException(status_code=404, detail="Repository not found or access denied.")

            # Process successful responses
            repo_data = repo_details_resp.json()
            languages = languages_resp.json() if isinstance(languages_resp, httpx.Response) and languages_resp.status_code == 200 else {}
            readme_data = readme_resp.json() if isinstance(readme_resp, httpx.Response) and readme_resp.status_code == 200 else {}
            commits_data = commits_resp.json() if isinstance(commits_resp, httpx.Response) and commits_resp.status_code == 200 else []
            issues_data = issues_resp.json() if isinstance(issues_resp, httpx.Response) and issues_resp.status_code == 200 else []

            readme_content = base64.b64decode(readme_data.get('content', '')).decode('utf-8') if 'content' in readme_data else "No README found."

        except httpx.RequestError as e:
            raise HTTPException(status_code=503, detail=f"Error communicating with GitHub: {e}")

    # Kick off background indexing task
    user_github_token = current_user.get("github_token") if current_user else None
    background_tasks.add_task(index_repository_task, str(req.url), user_github_token)

    return AnalysisResponse(
        projectName=repo_data.get('full_name', 'N/A'),
        description=repo_data.get('description', 'No description provided.'),
        readmeContent=readme_content,
        techStack=list(languages.keys()),
        recentCommits=[CommitInfo(sha=c['sha'], message=c['commit']['message'].split('\n')[0], author=c['commit']['author']['name'], url=c['html_url']) for c in commits_data],
        openIssues=[IssueInfo(number=i['number'], title=i['title'], url=i['html_url']) for i in issues_data if 'pull_request' not in i]
    )

@app.get("/api/index_status")
async def get_index_status(url: HttpUrl):
    return indexing_statuses.get(str(url), {"status": "idle"})

@app.post("/api/qna")
async def Youtube(req: QnaRequest):
    repo_url_key = str(req.url)
    if repo_url_key not in vector_stores or vector_stores[repo_url_key] is None:
        raise HTTPException(status_code=404, detail="Repository not indexed or index is empty. Please analyze it first.")
    
    vector_store = vector_stores[repo_url_key]
    retriever = vector_store.as_retriever()
    relevant_docs = retriever.get_relevant_documents(req.question)

    if not relevant_docs:
        return {"answer": "I couldn't find any relevant information in the codebase to answer that question."}

    context = "\n---\n".join([f"Source: {doc.metadata['source']}\n\n{doc.page_content}" for doc in relevant_docs])
    prompt = f"""
    You are an expert developer assistant. Answer the user's question based *only* on the following context from the codebase.
    Be concise and clear. If the context does not contain the answer, state that you couldn't find the information in the provided code.

    CONTEXT:
    {context}

    QUESTION: {req.question}

    ANSWER:
    """
    
    try:
        model = genai.GenerativeModel('gemini-pro')
        response = await model.generate_content_async(prompt)
        return {"answer": response.text}
    except Exception as e:
        logger.error(f"Gemini API error: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate answer from the AI model.")
@app.get("/api/health")
async def health_check():
    return {"status": "ok", "version": "3.2.0", "timestamp": datetime.utcnow().isoformat()}

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)
# # This is to ensure the server runs correctly when executed directly.
# # In production, you would typically use a WSGI server like Gunicorn or similar.    