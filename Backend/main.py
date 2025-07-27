# main.py
# To run this server: uvicorn main:app --reload
# Required packages: pip install fastapi uvicorn httpx python-decouple google-generativeai gitpython langchain langchain-community langchain-google-genai faiss-cpu python-jose[cryptography] itsdangerous python-multipart slowapi

import asyncio
import base64
import re
import os
import shutil
import tempfile
import stat
import time
import threading
import urllib.parse
from pathlib import Path
from datetime import datetime, timedelta, timezone
from collections import OrderedDict
from typing import Optional
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
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Configuration ---
GITHUB_CLIENT_ID = config('GITHUB_CLIENT_ID', default=None)
GITHUB_CLIENT_SECRET = config('GITHUB_CLIENT_SECRET', default=None)
SECRET_KEY = config('SECRET_KEY', default='your-secret-key-here-change-this-in-production') 
GEMINI_API_KEY = config('GEMINI_API_KEY', default=None)
GITHUB_API_TOKEN = config('GITHUB_API_TOKEN', default=None)

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 120 
FRONTEND_URL = config('FRONTEND_URL', default="http://localhost:3000")

# --- Enhanced Environment Validation ---
def validate_environment():
    """Validate all required environment variables"""
    errors = []
    
    if not GEMINI_API_KEY or GEMINI_API_KEY.strip() == "":
        errors.append("GEMINI_API_KEY is required and cannot be empty")
    
    if not SECRET_KEY or SECRET_KEY.strip() == "" or len(SECRET_KEY) < 32:
        errors.append("SECRET_KEY is required and must be at least 32 characters")
    
    if GITHUB_CLIENT_ID and (not GITHUB_CLIENT_SECRET or GITHUB_CLIENT_SECRET.strip() == ""):
        errors.append("GITHUB_CLIENT_SECRET is required when GITHUB_CLIENT_ID is set")
    
    if errors:
        logger.error("Configuration errors: " + "; ".join(errors))
        raise ValueError("Configuration validation failed: " + "; ".join(errors))

# Call at startup
try:
    validate_environment()
except ValueError as e:
    logger.warning(f"Configuration validation failed: {e}")

genai.configure(api_key=GEMINI_API_KEY)
if SECRET_KEY:
    state_serializer = URLSafeTimedSerializer(SECRET_KEY)

# --- Thread-Safe Vector Store Manager ---
class ThreadSafeVectorStore:
    def __init__(self, max_size: int = 50, ttl_hours: int = 24):
        self._stores = OrderedDict()
        self._statuses = OrderedDict()
        self._access_times = {}
        self._lock = threading.RLock()
        self.max_size = max_size
        self.ttl_seconds = ttl_hours * 3600
    
    def set_store(self, key: str, store):
        with self._lock:
            self._cleanup_expired()
            
            if len(self._stores) >= self.max_size and key not in self._stores:
                # Remove oldest entry
                oldest_key = next(iter(self._stores))
                self._remove_key(oldest_key)
            
            self._stores[key] = store
            self._access_times[key] = time.time()
            if key in self._stores:
                self._stores.move_to_end(key)  # Mark as recently used
    
    def get_store(self, key: str):
        with self._lock:
            if key in self._stores:
                self._access_times[key] = time.time()
                self._stores.move_to_end(key)  # Mark as recently used
                return self._stores[key]
            return None
    
    def set_status(self, key: str, status: dict):
        with self._lock:
            self._statuses[key] = {**status, "timestamp": time.time()}
    
    def get_status(self, key: str) -> dict:
        with self._lock:
            return self._statuses.get(key, {"status": "idle"})
    
    def _cleanup_expired(self):
        current_time = time.time()
        expired_keys = [
            key for key, access_time in self._access_times.items()
            if current_time - access_time > self.ttl_seconds
        ]
        for key in expired_keys:
            self._remove_key(key)
            logger.info(f"Cleaned up expired entry: {key}")
    
    def _remove_key(self, key: str):
        self._stores.pop(key, None)
        self._statuses.pop(key, None)
        self._access_times.pop(key, None)
    
    def get_stats(self):
        with self._lock:
            return {
                "total_stores": len(self._stores),
                "total_statuses": len(self._statuses),
                "status_breakdown": {}
            }

# Initialize vector store manager
vector_store_manager = ThreadSafeVectorStore()

# --- Rate Limiter ---
limiter = Limiter(key_func=get_remote_address)

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
    version="4.0.0"
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", FRONTEND_URL],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token", auto_error=False)

# --- Enhanced Security & JWT Functions ---
def create_access_token(data: dict, expires_delta: timedelta | None = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(token: str | None = Depends(oauth2_scheme)):
    if not token:
        return None
    
    try:
        # Validate token format
        if not token.startswith('eyJ'):  # Basic JWT format check
            logger.warning("Invalid token format received")
            return None
            
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        
        # Check expiration manually for better error handling
        exp = payload.get("exp")
        if exp and datetime.fromtimestamp(exp, tz=timezone.utc) < datetime.now(timezone.utc):
            logger.info("Token expired")
            return None
            
        # Validate required fields
        if not payload.get("sub"):
            logger.warning("Token missing required 'sub' field")
            return None
            
        return payload
        
    except jwt.ExpiredSignatureError:
        logger.info("Token expired")
        return None
    except jwt.InvalidTokenError as e:
        logger.warning(f"Invalid token: {e}")
        return None
    except Exception as e:
        logger.error(f"Token validation error: {e}")
        return None

# --- Enhanced GitHub API Helpers ---
def get_auth_headers(current_user: dict | None):
    if current_user and current_user.get("github_token"):
        return {"Authorization": f"token {current_user['github_token']}"}
    if GITHUB_API_TOKEN:
        return {"Authorization": f"token {GITHUB_API_TOKEN}"}
    return {}

def parse_github_url(url: str):
    """Parse GitHub URL with comprehensive validation"""
    try:
        # Normalize URL
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
            
        parsed = urllib.parse.urlparse(url)
        
        # Validate domain
        if parsed.netloc.lower() not in ['github.com', 'www.github.com']:
            raise HTTPException(status_code=400, detail="URL must be from github.com")
        
        # Parse path
        path_parts = [part for part in parsed.path.strip('/').split('/') if part]
        
        if len(path_parts) < 2:
            raise HTTPException(status_code=400, detail="Invalid GitHub URL: missing owner or repository")
        
        owner, repo = path_parts[0], path_parts[1]
        
        # Remove .git extension
        if repo.endswith('.git'):
            repo = repo[:-4]
        
        # Validate owner and repo names (GitHub rules)
        if not re.match(r'^[a-zA-Z0-9]([a-zA-Z0-9\-_.])*[a-zA-Z0-9]$', owner) and len(owner) > 0:
            raise HTTPException(status_code=400, detail="Invalid GitHub username format")
        
        if not re.match(r'^[a-zA-Z0-9]([a-zA-Z0-9\-_.])*[a-zA-Z0-9]$', repo) and len(repo) > 0:
            raise HTTPException(status_code=400, detail="Invalid GitHub repository name format")
        
        return owner, repo
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid GitHub URL: {str(e)}")

# --- Error handler for shutil.rmtree on Windows ---
def remove_readonly(func, path, excinfo):
    """Error handler for shutil.rmtree on Windows."""
    if not os.access(path, os.W_OK):
        os.chmod(path, stat.S_IWUSR | stat.S_IWRITE)
        func(path)
    else:
        raise

# --- Enhanced Repository Indexing Logic ---
def index_repository_task(repo_url_str: str, user_token: str | None):
    repo_url_key = repo_url_str
    temp_dir = None
    
    try:
        owner, repo_name = parse_github_url(repo_url_str)
        vector_store_manager.set_status(repo_url_key, {
            "status": "indexing", 
            "detail": "Starting...", 
            "progress": 0
        })
        
        temp_dir = tempfile.mkdtemp()
        
        # Enhanced clone with timeout
        clone_url = f"https://github.com/{owner}/{repo_name}.git"
        if user_token:
            clone_url = f"https://{user_token}@github.com/{owner}/{repo_name}.git"

        vector_store_manager.set_status(repo_url_key, {
            "status": "indexing", 
            "detail": "Cloning repository...", 
            "progress": 10
        })
        
        try:
            # Clone with timeout and specific options
            repo = Repo.clone_from(
                clone_url, 
                temp_dir, 
                depth=1,
                single_branch=True
            )
        except GitCommandError as e:
            error_msg = str(e).lower()
            if "authentication failed" in error_msg or "403" in error_msg:
                status = {"status": "failed", "detail": "Authentication failed. Repository may be private."}
            elif "not found" in error_msg or "404" in error_msg:
                status = {"status": "failed", "detail": "Repository not found."}
            elif "timeout" in error_msg:
                status = {"status": "failed", "detail": "Repository clone timed out. Repository may be too large."}
            else:
                status = {"status": "failed", "detail": f"Clone failed: {str(e)[:100]}"}
            
            vector_store_manager.set_status(repo_url_key, status)
            return

        # Enhanced file processing with size limits and progress
        docs = []
        root_path = Path(temp_dir)
        
        # More comprehensive file extensions
        file_extensions = {
            '.py', '.js', '.jsx', '.ts', '.tsx', '.java', '.go', '.rs', 
            '.md', '.json', '.html', '.css', '.scss', '.cpp', '.c', '.h', 
            '.php', '.rb', '.swift', '.kt', '.scala', '.sh', '.yml', '.yaml', 
            '.xml', '.sql', '.dockerfile', '.tf', '.vue', '.dart', '.r'
        }
        
        # Directories to skip
        skip_dirs = {
            '.git', 'node_modules', '__pycache__', '.venv', 'venv', 
            'build', 'dist', 'target', '.idea', '.vscode', 'vendor',
            'coverage', '.nyc_output', 'test_reports'
        }
        
        vector_store_manager.set_status(repo_url_key, {
            "status": "indexing", 
            "detail": "Scanning files...", 
            "progress": 30
        })
        
        # Collect all eligible files first
        eligible_files = []
        total_size = 0
        MAX_FILE_SIZE = 1024 * 1024  # 1MB per file
        MAX_TOTAL_SIZE = 100 * 1024 * 1024  # 100MB total
        
        for file_path in root_path.rglob('*'):
            if (file_path.is_file() and 
                not file_path.is_symlink() and
                file_path.suffix.lower() in file_extensions and
                not any(skip_dir in file_path.parts for skip_dir in skip_dirs)):
                
                try:
                    file_size = file_path.stat().st_size
                    if file_size > MAX_FILE_SIZE:
                        continue
                    
                    if total_size + file_size > MAX_TOTAL_SIZE:
                        logger.warning(f"Repository size limit reached. Skipping remaining files.")
                        break
                    
                    eligible_files.append(file_path)
                    total_size += file_size
                    
                except (OSError, IOError):
                    continue
        
        if not eligible_files:
            vector_store_manager.set_status(repo_url_key, {
                "status": "empty", 
                "detail": "No indexable files found."
            })
            vector_store_manager.set_store(repo_url_key, None)
            return
        
        # Process files with progress tracking
        processed_files = 0
        for i, file_path in enumerate(eligible_files):
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                if content.strip():  # Only add non-empty files
                    docs.append(Document(
                        page_content=content, 
                        metadata={
                            "source": str(file_path.relative_to(root_path)),
                            "file_type": file_path.suffix.lower(),
                            "size": len(content)
                        }
                    ))
                    processed_files += 1
                
                # Update progress
                progress = 30 + (i / len(eligible_files)) * 40  # 30-70%
                vector_store_manager.set_status(repo_url_key, {
                    "status": "indexing", 
                    "detail": f"Processing files... ({processed_files}/{len(eligible_files)})", 
                    "progress": int(progress)
                })
                
            except Exception as e:
                logger.warning(f"Could not read file {file_path}: {e}")
                continue

        if not docs:
            vector_store_manager.set_status(repo_url_key, {
                "status": "empty", 
                "detail": "No valid content found in files."
            })
            vector_store_manager.set_store(repo_url_key, None)
            return

        vector_store_manager.set_status(repo_url_key, {
            "status": "indexing", 
            "detail": "Splitting documents...", 
            "progress": 70
        })
        
        # Enhanced text splitting
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500, 
            chunk_overlap=150,
            separators=["\n\n", "\n", " ", ""]
        )
        split_docs = text_splitter.split_documents(docs)
        
        vector_store_manager.set_status(repo_url_key, {
            "status": "indexing", 
            "detail": "Creating embeddings...", 
            "progress": 85
        })
        
        # Create embeddings with error handling
        try:
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001", 
                google_api_key=GEMINI_API_KEY
            )
            vector_store = FAISS.from_documents(split_docs, embeddings)
            
            vector_store_manager.set_store(repo_url_key, vector_store)
            vector_store_manager.set_status(repo_url_key, {
                "status": "completed", 
                "detail": f"Indexing complete. Processed {processed_files} files.", 
                "progress": 100,
                "files_processed": processed_files,
                "total_chunks": len(split_docs)
            })
            
            logger.info(f"Successfully indexed repository: {repo_url_key} ({processed_files} files, {len(split_docs)} chunks)")
            
        except Exception as e:
            logger.error(f"Embedding creation failed: {e}")
            vector_store_manager.set_status(repo_url_key, {
                "status": "failed", 
                "detail": f"Failed to create embeddings: {str(e)[:100]}"
            })
        
    except Exception as e:
        logger.error(f"Indexing failed for {repo_url_key}: {e}")
        vector_store_manager.set_status(repo_url_key, {
            "status": "failed", 
            "detail": f"Unexpected error: {str(e)[:100]}"
        })
    finally:
        # Enhanced cleanup
        if temp_dir and os.path.exists(temp_dir):
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    shutil.rmtree(temp_dir, onerror=remove_readonly)
                    break
                except Exception as e:
                    if attempt == max_retries - 1:
                        logger.warning(f"Could not clean up temp directory {temp_dir} after {max_retries} attempts: {e}")
                    else:
                        time.sleep(1)  # Wait before retry

# --- Authentication Endpoints ---
@app.get("/api/health")
async def health_check():
    """Enhanced health check with system status"""
    try:
        # Test Gemini API
        test_model = genai.GenerativeModel('gemini-pro')
        test_response = test_model.generate_content("Hello")
        gemini_status = bool(test_response.text)
    except:
        gemini_status = False
    
    # Memory usage info
    stats = vector_store_manager.get_stats()
    
    return {
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat(),
        "features": {
            "github_oauth": bool(GITHUB_CLIENT_ID and GITHUB_CLIENT_SECRET),
            "github_api_fallback": bool(GITHUB_API_TOKEN),
            "gemini_ai": gemini_status
        },
        "stats": {
            "indexed_repositories": stats["total_stores"],
            "active_indexing": len([
                s for s in vector_store_manager._statuses.values() 
                if s.get("status") == "indexing"
            ])
        }
    }

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

    async with httpx.AsyncClient(timeout=30.0) as client:
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

# --- Enhanced Core API Endpoints ---
@app.post("/api/analyze", response_model=AnalysisResponse)
@limiter.limit("10/minute")
async def analyze_repository(
    request: Request,
    req: AnalyzeRequest, 
    background_tasks: BackgroundTasks, 
    current_user: dict = Depends(get_current_user)
):
    try:
        owner, repo_name = parse_github_url(str(req.url))
    except HTTPException as e:
        raise e
        
    headers = get_auth_headers(current_user)
    
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
async def get_index_status(url: str):
    """Fixed: Accept URL as query parameter string instead of HttpUrl"""
    try:
        # Parse the URL to validate it
        parse_github_url(url)
        return vector_store_manager.get_status(url)
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid URL: {str(e)}")

@app.post("/api/qna")
@limiter.limit("20/minute")
async def ask_question(request: Request, req: QnaRequest):
    repo_url_key = str(req.url)
    
    # Input validation
    if len(req.question.strip()) < 3:
        raise HTTPException(status_code=400, detail="Question must be at least 3 characters long.")
    
    if len(req.question) > 1000:
        raise HTTPException(status_code=400, detail="Question is too long. Maximum 1000 characters.")
    
    vector_store = vector_store_manager.get_store(repo_url_key)
    if vector_store is None:
        status = vector_store_manager.get_status(repo_url_key)
        if status["status"] == "indexing":
            raise HTTPException(status_code=202, detail="Repository is still being indexed. Please wait.")
        elif status["status"] == "empty":
            raise HTTPException(status_code=404, detail="Repository index is empty.")
        else:
            raise HTTPException(status_code=404, detail="Repository not indexed. Please analyze it first.")
    
    try:
        # Enhanced retrieval with better search
        retriever = vector_store.as_retriever(
            search_kwargs={"k": 8}
        )
        relevant_docs = retriever.get_relevant_documents(req.question)

        if not relevant_docs:
            return {"answer": "I couldn't find any relevant information in the codebase to answer that question."}

        # Enhanced context creation with better formatting
        context_parts = []
        for i, doc in enumerate(relevant_docs[:5]):  # Limit to top 5
            file_path = doc.metadata.get('source', f'file_{i}')
            content = doc.page_content[:800]  # Limit content length
            context_parts.append(f"=== {file_path} ===\n{content}")
        
        context = "\n\n".join(context_parts)
        
        # Enhanced prompt with better instructions
        prompt = f"""You are an expert software engineer analyzing a codebase. Answer the user's question based strictly on the provided code context.

INSTRUCTIONS:
- Be precise and technical when discussing code
- Reference specific files when possible
- If the context doesn't contain enough information, say so clearly
- Provide code examples from the context when helpful
- Keep responses concise but comprehensive
- Focus on the most relevant information

CODEBASE CONTEXT:
{context}

QUESTION: {req.question}

ANALYSIS:"""
        
        # Enhanced AI generation with parameters
        model = genai.GenerativeModel(
            'gemini-1.5-flash',
            generation_config=genai.types.GenerationConfig(
                temperature=0.3,
                top_p=0.8,
                max_output_tokens=1024,
            )
        )
        
        response = model.generate_content(prompt)
        
        if not response.text:
            return {"answer": "I apologize, but I couldn't generate a proper response. Please try rephrasing your question."}
        
        return {
            "answer": response.text,
            "sources": [doc.metadata.get('source', 'unknown') for doc in relevant_docs[:3]]
        }
        
    except Exception as e:
        logger.error(f"Q&A failed for {repo_url_key}: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate answer. Please try again.")

# --- Root Endpoint ---
@app.get("/")
async def root():
    return {
        "message": "GitHub Intelligence Assistant API",
        "version": "4.0.0",
        "docs": "/docs",
        "health": "/api/health",
        "features": {
            "github_oauth": bool(GITHUB_CLIENT_ID and GITHUB_CLIENT_SECRET),
            "github_api_fallback": bool(GITHUB_API_TOKEN),
            "gemini_ai": bool(GEMINI_API_KEY)
        }
    }

# --- Error Handlers ---
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": True,
            "message": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.utcnow().isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": True,
            "message": "An internal server error occurred",
            "status_code": 500,
            "timestamp": datetime.utcnow().isoformat()
        }
    )

# --- Startup and Shutdown Events ---
@app.on_event("startup")
async def startup_event():
    logger.info("GitHub Intelligence Assistant API starting up...")
    logger.info(f"Features enabled:")
    logger.info(f"  - GitHub OAuth: {bool(GITHUB_CLIENT_ID and GITHUB_CLIENT_SECRET)}")
    logger.info(f"  - GitHub API Fallback: {bool(GITHUB_API_TOKEN)}")
    logger.info(f"  - Gemini AI: {bool(GEMINI_API_KEY)}")
    logger.info(f"  - Vector Store TTL: {vector_store_manager.ttl_seconds / 3600} hours")
    logger.info(f"  - Max Repositories: {vector_store_manager.max_size}")

@app.on_event("shutdown") 
async def shutdown_event():
    logger.info("GitHub Intelligence Assistant API shutting down...")
    # Cleanup any resources if needed
    with vector_store_manager._lock:
        store_count = len(vector_store_manager._stores)
        vector_store_manager._stores.clear()
        vector_store_manager._statuses.clear()
        vector_store_manager._access_times.clear()
        logger.info(f"Cleaned up {store_count} vector stores")

# --- Main Application Entry Point ---
if __name__ == "__main__":
    import uvicorn
    
    # Configuration for development
    config = {
        "host": "0.0.0.0",
        "port": 8000,
        "reload": os.getenv("DEBUG", "").lower() == "true",
        "log_level": "info"
    }
    
    logger.info("Starting GitHub Intelligence Assistant API...")
    logger.info(f"Server configuration: {config}")
    
    uvicorn.run("main:app", **config)