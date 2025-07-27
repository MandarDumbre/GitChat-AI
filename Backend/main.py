# main.py
# To run this server: uvicorn main:app --reload
# Required packages: pip install fastapi uvicorn httpx python-decouple google-generativeai gitpython langchain langchain-community langchain-google-genai faiss-cpu python-jose[cryptography] itsdangerous python-multipart

import asyncio
import base64
import re
import os
import shutil
import tempfile
import stat
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
from langchain_community.vectorstores import FAISS
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
GITHUB_API_TOKEN = config('GITHUB_API_TOKEN', default=None)

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 120 
FRONTEND_URL = "http://localhost:3000"

# Validate essential configs
if not all([GEMINI_API_KEY, SECRET_KEY]):
    logger.error("FATAL: GEMINI_API_KEY and SECRET_KEY are required.")
    raise ValueError("Required environment variables are missing. Check README for setup.")

# For OAuth, both client ID and secret are needed
if GITHUB_CLIENT_ID and not GITHUB_CLIENT_SECRET:
    logger.error("FATAL: GITHUB_CLIENT_SECRET is required when GITHUB_CLIENT_ID is set.")
    raise ValueError("GITHUB_CLIENT_SECRET is required when GITHUB_CLIENT_ID is set.")

genai.configure(api_key=GEMINI_API_KEY)
if SECRET_KEY:
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
    version="3.2.3"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
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
    if token is None: 
        return None
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        if payload.get("sub") is None:
            return None
        return payload 
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
    # Handle both github.com and www.github.com
    match = re.search(r"(?:www\.)?github\.com/([\w\-.]+)/([\w\-.]+)", url)
    if not match: 
        raise HTTPException(status_code=400, detail="Invalid GitHub URL format.")
    owner, repo = match.groups()
    # Remove .git suffix if present
    if repo.endswith('.git'):
        repo = repo[:-4]
    return owner, repo

# --- Authentication Endpoints ---
@app.get("/api/auth/login")
async def login():
    if not GITHUB_CLIENT_ID:
        raise HTTPException(status_code=501, detail="GitHub OAuth not configured. Set GITHUB_CLIENT_ID and GITHUB_CLIENT_SECRET.")
    
    state = state_serializer.dumps("login_state")
    return RedirectResponse(f"https://github.com/login/oauth/authorize?client_id={GITHUB_CLIENT_ID}&scope=repo,read:user&state={state}")

@app.get("/api/auth/callback")
async def auth_callback(code: str, state: str):
    if not GITHUB_CLIENT_ID or not GITHUB_CLIENT_SECRET:
        raise HTTPException(status_code=501, detail="GitHub OAuth not configured.")
    
    try:
        state_serializer.loads(state, max_age=300)
    except Exception:
        raise HTTPException(status_code=403, detail="Invalid state or state expired.")

    async with httpx.AsyncClient() as client:
        try:
            token_response = await client.post("https://github.com/login/oauth/access_token",
                data={"client_id": GITHUB_CLIENT_ID, "client_secret": GITHUB_CLIENT_SECRET, "code": code},
                headers={"Accept": "application/json"})
            token_data = token_response.json()
            github_token = token_data.get("access_token")
            if not github_token: 
                raise HTTPException(status_code=400, detail="Could not retrieve access token.")
            
            user_response = await client.get("https://api.github.com/user", 
                headers={"Authorization": f"token {github_token}"})
            if user_response.status_code != 200:
                raise HTTPException(status_code=400, detail="Could not retrieve user information.")
            user_data = user_response.json()
        except httpx.RequestError as e:
            raise HTTPException(status_code=503, detail=f"Error communicating with GitHub: {e}")

    jwt_data = {
        "sub": user_data.get("login"), 
        "name": user_data.get("name"), 
        "avatar_url": user_data.get("avatar_url"), 
        "github_token": github_token
    }
    app_token = create_access_token(jwt_data, timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    return RedirectResponse(f"{FRONTEND_URL}?token={app_token}")

@app.get("/api/me")
async def read_users_me(current_user: dict = Depends(get_current_user)):
    if not current_user: 
        return {"authenticated": False}
    payload = {k: v for k, v in current_user.items() if k != 'github_token'}
    payload["authenticated"] = True
    return payload

# --- Error handler for shutil.rmtree on Windows ---
def remove_readonly(func, path, excinfo):
    """Error handler for shutil.rmtree on Windows."""
    if not os.access(path, os.W_OK):
        os.chmod(path, stat.S_IWUSR | stat.S_IWRITE)
        func(path)
    else:
        raise

# --- Code Indexing Logic (Background Task) ---
def index_repository_task(repo_url_str: str, user_token: str | None):
    repo_url_key = repo_url_str
    temp_dir = None
    try:
        owner, repo_name = parse_github_url(repo_url_str)
        indexing_statuses[repo_url_key] = {"status": "indexing", "detail": "Starting..."}
        
        temp_dir = tempfile.mkdtemp()
        
        clone_url = f"https://github.com/{owner}/{repo_name}.git"
        if user_token:
            clone_url = f"https://{user_token}@github.com/{owner}/{repo_name}.git"

        indexing_statuses[repo_url_key]["detail"] = "Cloning repository..."
        logger.info(f"Cloning repository: {clone_url}")
        
        # Clone with error handling
        try:
            Repo.clone_from(clone_url, temp_dir, depth=1)
        except GitCommandError as e:
            logger.error(f"Git clone failed: {e}")
            if "authentication failed" in str(e).lower():
                indexing_statuses[repo_url_key] = {"status": "failed", "detail": "Authentication failed. Repository may be private and require login."}
            elif "not found" in str(e).lower():
                indexing_statuses[repo_url_key] = {"status": "failed", "detail": "Repository not found. Check the URL."}
            else:
                indexing_statuses[repo_url_key] = {"status": "failed", "detail": "Failed to clone repository."}
            return
        
        docs = []
        root_path = Path(temp_dir)
        
        # Expanded file extensions
        file_extensions = ['.py', '.js', '.jsx', '.ts', '.tsx', '.java', '.go', '.rs', '.md', '.json', '.html', '.css', '.scss', '.cpp', '.c', '.h', '.php', '.rb', '.swift', '.kt', '.scala', '.sh', '.yml', '.yaml', '.xml', '.sql']
        
        indexing_statuses[repo_url_key]["detail"] = "Processing files..."
        file_count = 0
        
        for file_path in root_path.rglob('*'):
            if (file_path.is_file() and 
                file_path.suffix.lower() in file_extensions and 
                '.git' not in str(file_path) and
                'node_modules' not in str(file_path) and
                '__pycache__' not in str(file_path)):
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    if content.strip():  # Only add non-empty files
                        docs.append(Document(
                            page_content=content, 
                            metadata={"source": str(file_path.relative_to(root_path))}
                        ))
                        file_count += 1
                except Exception as e:
                    logger.warning(f"Could not read file {file_path}: {e}")
                    continue

        logger.info(f"Processed {file_count} files")
        
        if not docs:
            vector_stores[repo_url_key] = None
            indexing_statuses[repo_url_key] = {"status": "empty", "detail": "No indexable files found."}
            return

        indexing_statuses[repo_url_key]["detail"] = "Splitting documents..."
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
        split_docs = text_splitter.split_documents(docs)
        
        indexing_statuses[repo_url_key]["detail"] = "Creating embeddings..."
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001", 
            google_api_key=GEMINI_API_KEY
        )
        vector_store = FAISS.from_documents(split_docs, embeddings)
        
        vector_stores[repo_url_key] = vector_store
        indexing_statuses[repo_url_key] = {"status": "completed", "detail": "Indexing complete."}
        logger.info(f"Successfully indexed repository: {repo_url_key}")
        
    except Exception as e:
        logger.error(f"Indexing failed for {repo_url_key}: {e}")
        indexing_statuses[repo_url_key] = {"status": "failed", "detail": f"An unexpected error occurred: {str(e)}"}
    finally:
        if temp_dir and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir, onerror=remove_readonly)
            except Exception as e:
                logger.warning(f"Could not clean up temp directory {temp_dir}: {e}")

# --- Core API Endpoints ---
@app.post("/api/analyze", response_model=AnalysisResponse)
async def analyze_repository(req: AnalyzeRequest, background_tasks: BackgroundTasks, current_user: dict = Depends(get_current_user)):
    try:
        owner, repo_name = parse_github_url(str(req.url))
    except HTTPException as e:
        raise e
        
    headers = get_auth_headers(current_user)
    
    # For public repos, we can proceed without auth headers
    # For private repos, auth is required
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        api_url = f"https://api.github.com/repos/{owner}/{repo_name}"
        try:
            # Make all requests concurrently
            tasks = [
                client.get(api_url, headers=headers),
                client.get(f"{api_url}/languages", headers=headers),
                client.get(f"{api_url}/readme", headers=headers),
                client.get(f"{api_url}/commits?per_page=5", headers=headers),
                client.get(f"{api_url}/issues?state=open&per_page=5", headers=headers)
            ]
            
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            repo_details_resp, languages_resp, readme_resp, commits_resp, issues_resp = responses

            # Check if main repo request failed
            if isinstance(repo_details_resp, Exception):
                raise HTTPException(status_code=503, detail=f"Error communicating with GitHub: {repo_details_resp}")
            
            if repo_details_resp.status_code == 404:
                raise HTTPException(status_code=404, detail="Repository not found or access denied.")
            elif repo_details_resp.status_code == 403:
                raise HTTPException(status_code=403, detail="Access denied. Repository may be private - please login.")
            elif repo_details_resp.status_code != 200:
                raise HTTPException(status_code=repo_details_resp.status_code, detail="Failed to fetch repository details.")

            repo_data = repo_details_resp.json()
            
            # Process other responses safely
            languages = {}
            if isinstance(languages_resp, httpx.Response) and languages_resp.status_code == 200:
                languages = languages_resp.json()
            
            readme_data = {}
            if isinstance(readme_resp, httpx.Response) and readme_resp.status_code == 200:
                readme_data = readme_resp.json()
            
            commits_data = []
            if isinstance(commits_resp, httpx.Response) and commits_resp.status_code == 200:
                commits_data = commits_resp.json()
            
            issues_data = []
            if isinstance(issues_resp, httpx.Response) and issues_resp.status_code == 200:
                issues_data = issues_resp.json()

            # Process README
            readme_content = "No README found."
            if 'content' in readme_data:
                try:
                    readme_content = base64.b64decode(readme_data['content']).decode('utf-8')
                except Exception as e:
                    logger.warning(f"Could not decode README: {e}")
                    readme_content = "README found but could not be decoded."

        except httpx.RequestError as e:
            raise HTTPException(status_code=503, detail=f"Error communicating with GitHub: {e}")

    # Start background indexing
    user_github_token = current_user.get("github_token") if current_user else None
    background_tasks.add_task(index_repository_task, str(req.url), user_github_token)

    return AnalysisResponse(
        projectName=repo_data.get('full_name', 'N/A'),
        description=repo_data.get('description') or 'No description provided.',
        readmeContent=readme_content,
        techStack=list(languages.keys()) if languages else [],
        recentCommits=[
            CommitInfo(
                sha=c['sha'], 
                message=c['commit']['message'].split('\n')[0][:100], 
                author=c['commit']['author']['name'], 
                url=c['html_url']
            ) for c in commits_data[:5]
        ],
        openIssues=[
            IssueInfo(
                number=i['number'], 
                title=i['title'][:100], 
                url=i['html_url']
            ) for i in issues_data[:5] if 'pull_request' not in i
        ]
    )

@app.get("/api/index_status")
async def get_index_status(url: HttpUrl):
    return indexing_statuses.get(str(url), {"status": "idle"})

@app.post("/api/qna")
async def ask_question(req: QnaRequest):
    repo_url_key = str(req.url)
    
    if repo_url_key not in vector_stores:
        raise HTTPException(status_code=404, detail="Repository not indexed. Please analyze it first.")
    
    if vector_stores[repo_url_key] is None:
        raise HTTPException(status_code=404, detail="Repository index is empty. No code files were found to analyze.")
    
    vector_store = vector_stores[repo_url_key]
    
    try:
        retriever = vector_store.as_retriever(search_kwargs={"k": 5})
        relevant_docs = retriever.get_relevant_documents(req.question)

        if not relevant_docs:
            return {"answer": "I couldn't find any relevant information in the codebase to answer that question."}

        # Create context from relevant documents
        context = "\n---\n".join([
            f"File: {doc.metadata['source']}\n\n{doc.page_content[:1000]}" 
            for doc in relevant_docs
        ])
        
        prompt = f"""You are an expert code analyst. Answer the user's question based strictly on the provided codebase context.

Guidelines:
- Be precise and technical when discussing code
- Reference specific files when relevant
- If the context doesn't contain enough information to answer, say so clearly
- Keep responses concise but comprehensive
- Use code examples from the context when helpful

CODEBASE CONTEXT:
{context}

QUESTION: {req.question}

ANSWER:"""
        
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(prompt)
        
        return {"answer": response.text}
        
    except Exception as e:
        logger.error(f"Q&A failed for {repo_url_key}: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate answer from the AI model.")

@app.get("/api/health")
async def health_check():
    return {
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat(),
        "features": {
            "github_oauth": bool(GITHUB_CLIENT_ID and GITHUB_CLIENT_SECRET),
            "github_api_fallback": bool(GITHUB_API_TOKEN),
            "gemini_ai": bool(GEMINI_API_KEY)
        }
    }

# Root endpoint
@app.get("/")
async def root():
    return {"message": "GitHub Intelligence Assistant API", "docs": "/docs"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)