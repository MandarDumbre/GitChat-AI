# main.py
# To run this server: uvicorn main:app --reload
# Required packages: pip install fastapi uvicorn httpx python-decouple groq gitpython langchain langchain-community langchain-groq faiss-cpu python-jose[cryptography] itsdangerous python-multipart slowapi redis sentence-transformers

import asyncio
import base64
import re
import os
import hashlib
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
from groq import Groq, AsyncGroq
from git import Repo, GitCommandError
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from jose import JWTError, jwt
from itsdangerous import URLSafeTimedSerializer
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import redis
from functools import wraps
import hashlib

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
GROQ_API_KEY = config('GROQ_API_KEY', default=None)
GITHUB_API_TOKEN = config('GITHUB_API_TOKEN', default=None)
REDIS_URL = config('REDIS_URL', default='redis://localhost:6379')

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 120 
FRONTEND_URL = config('FRONTEND_URL', default="http://localhost:3000")

# Groq Configuration
GROQ_MODEL = config('GROQ_MODEL', default='llama-3.3-70b-versatile')  # Latest and most capable model
EMBEDDING_MODEL = config('EMBEDDING_MODEL', default='sentence-transformers/all-MiniLM-L6-v2')  # Free local embeddings

# Rate Limiting Configuration
RATE_LIMIT_REQUESTS_PER_MINUTE = int(config('RATE_LIMIT_RPM', default='30'))
RATE_LIMIT_REQUESTS_PER_HOUR = int(config('RATE_LIMIT_RPH', default='500'))
RATE_LIMIT_REQUESTS_PER_DAY = int(config('RATE_LIMIT_RPD', default='2000'))

# --- Enhanced Environment Validation ---
def validate_environment():
    """Validate all required environment variables"""
    errors = []
    
    if not GROQ_API_KEY or GROQ_API_KEY.strip() == "":
        errors.append("GROQ_API_KEY is required and cannot be empty")
    
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

# Initialize Groq clients
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
async_groq_client = AsyncGroq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

if SECRET_KEY:
    state_serializer = URLSafeTimedSerializer(SECRET_KEY)

# --- Enhanced Rate Limiting with Redis ---
try:
    redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)
    redis_client.ping()  # Test connection
    logger.info("Redis connection established")
except Exception as e:
    logger.warning(f"Redis connection failed: {e}. Using in-memory rate limiting.")
    redis_client = None

class EnhancedRateLimiter:
    def __init__(self, redis_client=None):
        self.redis_client = redis_client
        self.memory_store = {}  # Fallback for when Redis is unavailable
        
    def _get_key(self, identifier: str, window: str) -> str:
        return f"rate_limit:{identifier}:{window}"
    
    def _get_identifier(self, request: Request, user: dict = None) -> str:
        """Get rate limiting identifier (user or IP-based)"""
        if user and user.get('sub'):
            return f"user:{user['sub']}"
        return f"ip:{get_remote_address(request)}"
    
    def check_rate_limit(self, request: Request, user: dict = None) -> dict:
        """Check rate limits across multiple time windows"""
        identifier = self._get_identifier(request, user)
        current_time = int(time.time())
        
        windows = {
            'minute': (60, RATE_LIMIT_REQUESTS_PER_MINUTE, current_time // 60),
            'hour': (3600, RATE_LIMIT_REQUESTS_PER_HOUR, current_time // 3600),
            'day': (86400, RATE_LIMIT_REQUESTS_PER_DAY, current_time // 86400)
        }
        
        for window_name, (window_seconds, limit, window_key) in windows.items():
            key = self._get_key(identifier, f"{window_name}:{window_key}")
            
            if self.redis_client:
                try:
                    current_count = self.redis_client.get(key)
                    current_count = int(current_count) if current_count else 0
                    
                    if current_count >= limit:
                        return {
                            'allowed': False,
                            'window': window_name,
                            'limit': limit,
                            'current': current_count,
                            'reset_time': (window_key + 1) * window_seconds
                        }
                    
                    # Increment counter
                    pipe = self.redis_client.pipeline()
                    pipe.incr(key, 1)
                    pipe.expire(key, window_seconds)
                    pipe.execute()
                    
                except Exception as e:
                    logger.warning(f"Redis rate limiting failed: {e}")
                    # Fallback to memory-based limiting
                    return self._memory_rate_limit(identifier, window_name, limit, window_key, window_seconds)
            else:
                return self._memory_rate_limit(identifier, window_name, limit, window_key, window_seconds)
        
        return {'allowed': True}
    
    def _memory_rate_limit(self, identifier: str, window_name: str, limit: int, window_key: int, window_seconds: int) -> dict:
        """Fallback memory-based rate limiting"""
        key = f"{identifier}:{window_name}:{window_key}"
        current_time = time.time()
        
        if key not in self.memory_store:
            self.memory_store[key] = {'count': 0, 'expire': current_time + window_seconds}
        
        entry = self.memory_store[key]
        
        # Clean expired entries
        if current_time > entry['expire']:
            entry['count'] = 0
            entry['expire'] = current_time + window_seconds
        
        if entry['count'] >= limit:
            return {
                'allowed': False,
                'window': window_name,
                'limit': limit,
                'current': entry['count'],
                'reset_time': entry['expire']
            }
        
        entry['count'] += 1
        return {'allowed': True}

# Initialize enhanced rate limiter
enhanced_limiter = EnhancedRateLimiter(redis_client)

# Custom rate limiting decorator
def advanced_rate_limit(fallback_limit: str = "10/minute"):
    def decorator(func):
        @wraps(func)
        async def wrapper(request: Request, *args, **kwargs):
            # Extract current_user from kwargs if available
            current_user = kwargs.get('current_user')
            
            rate_limit_result = enhanced_limiter.check_rate_limit(request, current_user)
            
            if not rate_limit_result['allowed']:
                reset_time = datetime.fromtimestamp(rate_limit_result['reset_time'])
                raise HTTPException(
                    status_code=429,
                    detail={
                        "error": "Rate limit exceeded",
                        "window": rate_limit_result['window'],
                        "limit": rate_limit_result['limit'],
                        "current": rate_limit_result['current'],
                        "reset_time": reset_time.isoformat(),
                        "retry_after": int(rate_limit_result['reset_time'] - time.time())
                    }
                )
            
            return await func(request, *args, **kwargs)
        return wrapper
    return decorator

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

# --- Rate Limiter for SlowAPI (fallback) ---
limiter = Limiter(key_func=get_remote_address)

# --- Pydantic Models ---
class AnalyzeRequest(BaseModel):
    url: HttpUrl

class QnaRequest(BaseModel):
    url: HttpUrl
    question: str

class FollowupRequest(BaseModel):
    url: HttpUrl
    question: str
    previous_context: str = ""

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

class QnaResponse(BaseModel):
    answer: str
    sources: list[dict] = []
    metadata: dict = {}
    is_followup: bool = False

# FIXED: Added missing models for README generation
class RepoURL(BaseModel):
    url: str

class ReadmeResponse(BaseModel):
    readme_content: str

class UserResponse(BaseModel):
    authenticated: bool
    name: Optional[str] = None
    avatar_url: Optional[str] = None
    sub: Optional[str] = None

# --- FastAPI App Initialization ---
app = FastAPI(
    title="GitHub Project Intelligence Assistant API",
    description="Backend service with authentication to analyze public and private repositories using Groq.",
    version="6.0.0"
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

# --- Groq Helper Functions ---
async def call_groq_chat(messages: list, model: str = GROQ_MODEL, temperature: float = 0.4, max_tokens: int = 2048) -> str:
    """Enhanced Groq API call with error handling and retries"""
    if not async_groq_client:
        raise HTTPException(status_code=503, detail="Groq API client not initialized. Check GROQ_API_KEY.")
    
    max_retries = 3
    # Fallback models in order of preference
    fallback_models = [
        model,  # User specified model
        'llama-3.3-70b-versatile',  # Latest 70B model
        'llama-3.1-8b-instant',     # Fast 8B model
        'gemma2-9b-it'              # Alternative model
    ]
    
    for attempt in range(max_retries):
        # Try each model in the fallback chain
        for current_model in fallback_models:
            try:
                response = await async_groq_client.chat.completions.create(
                    model=current_model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=30
                )
                
                if response.choices and response.choices[0].message:
                    content = response.choices[0].message.content.strip()
                    if current_model != model:
                        logger.info(f"Successfully used fallback model: {current_model} (requested: {model})")
                    return content
                else:
                    raise Exception("No response content from Groq")
                    
            except Exception as e:
                error_str = str(e).lower()
                
                if "model" in error_str and ("decommissioned" in error_str or "deprecated" in error_str):
                    logger.warning(f"Model {current_model} is deprecated, trying next fallback...")
                    continue  # Try next model in fallback chain
                
                elif "rate limit" in error_str:
                    logger.warning(f"Groq rate limit hit (attempt {attempt + 1}): {e}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2 ** attempt)  # Exponential backoff
                        break  # Break inner loop to retry with same model
                    raise HTTPException(status_code=429, detail="Groq API rate limit exceeded. Please try again later.")
                
                elif "invalid" in error_str or "bad request" in error_str:
                    if "model" in error_str:
                        continue  # Try next model
                    logger.error(f"Invalid Groq request: {e}")
                    raise HTTPException(status_code=400, detail=f"Invalid request to AI service: {str(e)}")
                
                else:
                    logger.error(f"Groq API error with {current_model} (attempt {attempt + 1}): {e}")
                    if current_model == fallback_models[-1] and attempt == max_retries - 1:
                        # Last model and last attempt
                        raise HTTPException(status_code=503, detail="All AI models temporarily unavailable. Please try again later.")
                    elif current_model == fallback_models[-1]:
                        # Last model but not last attempt
                        await asyncio.sleep(1)
                        break  # Break inner loop to retry
                    else:
                        continue  # Try next model
    
    raise HTTPException(status_code=503, detail="Failed to get response from AI service after multiple attempts with all available models.")

# --- FIXED: Repository Analysis Helper Functions ---
def get_repo_local_path(repo_url: str) -> str:
    """Get local path for a cloned repository based on its URL"""
    repo_hash = hashlib.md5(repo_url.encode()).hexdigest()
    return os.path.join(tempfile.gettempdir(), f"gitchat_repo_{repo_hash}")

def get_file_tree(path: str, indent: str = "", max_depth: int = 3, current_depth: int = 0) -> str:
    """Recursively gets the file tree for a given path, ignoring common junk."""
    if current_depth > max_depth:
        return f"{indent}...\n"
    
    tree = ""
    try:
        items = sorted(os.listdir(path))
        for item in items:
            item_path = os.path.join(path, item)
            # Ignore hidden files/dirs, pycache, venv, etc.
            if item.startswith('.') or item in ['__pycache__', 'node_modules', 'venv', '.git']:
                continue
            
            if os.path.isdir(item_path):
                tree += f"{indent}{item}/\n"
                tree += get_file_tree(item_path, indent + "  ", max_depth, current_depth + 1)
            else:
                # Limit file list to common/important file types
                if item.endswith(('.py', '.js', '.ts', '.go', '.java', 'Dockerfile', 'requirements.txt', 'package.json')):
                    tree += f"{indent}{item}\n"
    except FileNotFoundError:
        return ""
    return tree

def analyze_code_structure(file_path: str, content: str) -> dict:
    """Analyze code structure based on file type"""
    file_type = Path(file_path).suffix.lower()
    structure = {"functions": [], "classes": [], "imports": [], "exports": []}
    
    try:
        if file_type == '.py':
            # Python analysis
            structure["functions"] = re.findall(r'def\s+(\w+)', content)
            structure["classes"] = re.findall(r'class\s+(\w+)', content)
            structure["imports"] = re.findall(r'(?:from\s+\S+\s+)?import\s+([^\n]+)', content)
        
        elif file_type in ['.js', '.jsx', '.ts', '.tsx']:
            # JavaScript/TypeScript analysis
            structure["functions"].extend(re.findall(r'function\s+(\w+)', content))
            structure["functions"].extend(re.findall(r'const\s+(\w+)\s*=.*=>', content))
            structure["functions"].extend(re.findall(r'(\w+)\s*:\s*function', content))
            structure["classes"] = re.findall(r'class\s+(\w+)', content)
            structure["imports"] = re.findall(r'import.*from\s+[\'"]([^\'"]+)[\'"]', content)
            structure["exports"] = re.findall(r'export\s+(?:default\s+)?(\w+)', content)
        
        elif file_type == '.java':
            # Java analysis
            structure["functions"] = re.findall(r'(?:public|private|protected).*?\s+(\w+)\s*\(', content)
            structure["classes"] = re.findall(r'(?:public\s+)?class\s+(\w+)', content)
            structure["imports"] = re.findall(r'import\s+([^\n;]+)', content)
        
        elif file_type == '.go':
            # Go analysis
            structure["functions"] = re.findall(r'func\s+(\w+)', content)
            structure["imports"] = re.findall(r'import\s+"([^"]+)"', content)
    
    except Exception as e:
        logger.warning(f"Code structure analysis failed for {file_path}: {e}")
    
    return structure

# --- Enhanced Repository Indexing Logic ---
def enhanced_index_repository_task(repo_url_str: str, user_token: str | None):
    """Enhanced indexing with better code structure analysis using local embeddings"""
    repo_url_key = repo_url_str
    temp_dir = None
    
    try:
        owner, repo_name = parse_github_url(repo_url_str)
        vector_store_manager.set_status(repo_url_key, {
            "status": "indexing", 
            "detail": "Starting enhanced analysis...", 
            "progress": 0
        })
        
        temp_dir = tempfile.mkdtemp()
        
        # Clone repository
        clone_url = f"https://github.com/{owner}/{repo_name}.git"
        if user_token:
            clone_url = f"https://{user_token}@github.com/{owner}/{repo_name}.git"

        vector_store_manager.set_status(repo_url_key, {
            "status": "indexing", 
            "detail": "Cloning repository...", 
            "progress": 10
        })
        
        try:
            repo = Repo.clone_from(clone_url, temp_dir, depth=1, single_branch=True)
        except GitCommandError as e:
            error_msg = str(e).lower()
            if "authentication failed" in error_msg or "403" in error_msg:
                status = {"status": "failed", "detail": "Authentication failed. Repository may be private."}
            elif "not found" in error_msg or "404" in error_msg:
                status = {"status": "failed", "detail": "Repository not found."}
            else:
                status = {"status": "failed", "detail": f"Clone failed: {str(e)[:100]}"}
            
            vector_store_manager.set_status(repo_url_key, status)
            return

        # Enhanced file processing with code structure analysis
        docs = []
        root_path = Path(temp_dir)
        
        # FIXED: Store repo path for README generation
        repo_local_path = get_repo_local_path(repo_url_str)
        if os.path.exists(repo_local_path):
            shutil.rmtree(repo_local_path, onerror=remove_readonly)
        shutil.copytree(temp_dir, repo_local_path)
        
        # Prioritized file extensions with weights
        file_priorities = {
            # Core code files (highest priority)
            '.py': 10, '.js': 10, '.jsx': 9, '.ts': 9, '.tsx': 9,
            '.java': 8, '.go': 8, '.rs': 8, '.cpp': 7, '.c': 7,
            
            # Configuration and important files
            '.json': 6, '.yml': 6, '.yaml': 6, '.xml': 5, '.sql': 5,
            
            # Documentation and markup
            '.md': 8, '.rst': 6, '.txt': 4,
            
            # Web and styling
            '.html': 5, '.css': 4, '.scss': 4,
            
            # Others
            '.php': 6, '.rb': 6, '.swift': 6, '.kt': 6, '.scala': 6,
            '.sh': 5, '.dockerfile': 6, '.tf': 5, '.vue': 7, '.dart': 6, '.r': 5
        }
        
        skip_dirs = {
            '.git', 'node_modules', '__pycache__', '.venv', 'venv', 
            'build', 'dist', 'target', '.idea', '.vscode', 'vendor',
            'coverage', '.nyc_output', 'test_reports', 'logs', 'tmp'
        }
        
        vector_store_manager.set_status(repo_url_key, {
            "status": "indexing", 
            "detail": "Analyzing code structure...", 
            "progress": 30
        })
        
        # Collect and prioritize files
        eligible_files = []
        total_size = 0
        MAX_FILE_SIZE = 2 * 1024 * 1024  # 2MB per file
        MAX_TOTAL_SIZE = 150 * 1024 * 1024  # 150MB total
        
        for file_path in root_path.rglob('*'):
            if (file_path.is_file() and 
                not file_path.is_symlink() and
                file_path.suffix.lower() in file_priorities and
                not any(skip_dir in file_path.parts for skip_dir in skip_dirs)):
                
                try:
                    file_size = file_path.stat().st_size
                    if file_size > MAX_FILE_SIZE:
                        continue
                    
                    if total_size + file_size > MAX_TOTAL_SIZE:
                        break
                    
                    priority = file_priorities.get(file_path.suffix.lower(), 1)
                    eligible_files.append((file_path, priority, file_size))
                    total_size += file_size
                    
                except (OSError, IOError):
                    continue
        
        # Sort by priority (highest first)
        eligible_files.sort(key=lambda x: x[1], reverse=True)
        
        if not eligible_files:
            vector_store_manager.set_status(repo_url_key, {
                "status": "empty", 
                "detail": "No indexable files found."
            })
            return

        # Process files with enhanced metadata extraction
        processed_files = 0
        for i, (file_path, priority, file_size) in enumerate(eligible_files):
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                if content.strip():
                    # Enhanced metadata extraction
                    file_type = file_path.suffix.lower()
                    relative_path = str(file_path.relative_to(root_path))
                    
                    # Extract code structure information
                    metadata = {
                        "source": relative_path,
                        "file_type": file_type,
                        "size": len(content),
                        "priority": priority,
                        "lines": len(content.split('\n'))
                    }
                    
                    # Add language-specific analysis
                    if file_type == '.py':
                        functions = re.findall(r'def\s+(\w+)', content)
                        classes = re.findall(r'class\s+(\w+)', content)
                        imports = re.findall(r'(?:from\s+\S+\s+)?import\s+([^\n]+)', content)
                        metadata.update({
                            "functions": len(functions),
                            "classes": len(classes),
                            "imports": len(imports),
                            "language": "python"
                        })
                    elif file_type in ['.js', '.jsx', '.ts', '.tsx']:
                        functions = len(re.findall(r'function\s+\w+|const\s+\w+\s*=.*=>|\w+\s*:\s*function', content))
                        classes = len(re.findall(r'class\s+\w+', content))
                        imports = len(re.findall(r'import.*from|require\(', content))
                        metadata.update({
                            "functions": functions,
                            "classes": classes,
                            "imports": imports,
                            "language": "javascript"
                        })
                    
                    docs.append(Document(page_content=content, metadata=metadata))
                    processed_files += 1
                
                # Update progress
                progress = 30 + (i / len(eligible_files)) * 40
                vector_store_manager.set_status(repo_url_key, {
                    "status": "indexing", 
                    "detail": f"Processing files... ({processed_files}/{len(eligible_files)})", 
                    "progress": int(progress)
                })
                
            except Exception as e:
                logger.warning(f"Could not read file {file_path}: {e}")
                continue

        # Enhanced text splitting with code-aware splitting
        vector_store_manager.set_status(repo_url_key, {
            "status": "indexing", 
            "detail": "Creating intelligent code chunks...", 
            "progress": 70
        })
        
        # Code-aware text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,  # Larger chunks for better context
            chunk_overlap=200,
            separators=[
                "\n\nclass ", "\n\ndef ", "\n\nfunction ",  # Code structure
                "\n\n", "\n", " ", ""  # General separators
            ]
        )
        split_docs = text_splitter.split_documents(docs)
        
        vector_store_manager.set_status(repo_url_key, {
            "status": "indexing", 
            "detail": "Creating semantic embeddings (local)...", 
            "progress": 85
        })
        
        # Create embeddings using local HuggingFace model
        try:
            embeddings = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            vector_store = FAISS.from_documents(split_docs, embeddings)
            
            vector_store_manager.set_store(repo_url_key, vector_store)
            vector_store_manager.set_status(repo_url_key, {
                "status": "completed", 
                "detail": f"Enhanced analysis complete. Processed {processed_files} files with intelligent chunking.", 
                "progress": 100,
                "files_processed": processed_files,
                "total_chunks": len(split_docs),
                "analysis_type": "enhanced"
            })
            
            logger.info(f"Enhanced indexing completed: {repo_url_key} ({processed_files} files, {len(split_docs)} chunks)")
            
        except Exception as e:
            logger.error(f"Enhanced embedding creation failed: {e}")
            vector_store_manager.set_status(repo_url_key, {
                "status": "failed", 
                "detail": f"Failed to create embeddings: {str(e)[:100]}"
            })
        
    except Exception as e:
        logger.error(f"Enhanced indexing failed for {repo_url_key}: {e}")
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
        # Test Groq API
        test_response = await call_groq_chat([
            {"role": "user", "content": "Hello"}
        ], max_tokens=10)
        groq_status = bool(test_response)
    except:
        groq_status = False
    
    # Test Redis connection
    redis_status = False
    if redis_client:
        try:
            redis_client.ping()
            redis_status = True
        except:
            pass
    
    # Memory usage info
    stats = vector_store_manager.get_stats()
    
    return {
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat(),
        "features": {
            "github_oauth": bool(GITHUB_CLIENT_ID and GITHUB_CLIENT_SECRET),
            "github_api_fallback": bool(GITHUB_API_TOKEN),
            "groq": groq_status,
            "redis_cache": redis_status,
            "local_embeddings": True
        },
        "rate_limits": {
            "per_minute": RATE_LIMIT_REQUESTS_PER_MINUTE,
            "per_hour": RATE_LIMIT_REQUESTS_PER_HOUR,
            "per_day": RATE_LIMIT_REQUESTS_PER_DAY
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
@advanced_rate_limit("5/minute")
async def login(request: Request):
    if not GITHUB_CLIENT_ID:
        raise HTTPException(status_code=501, detail="GitHub OAuth not configured. Set GITHUB_CLIENT_ID and GITHUB_CLIENT_SECRET.")
    
    state = state_serializer.dumps("login_state")
    return RedirectResponse(f"https://github.com/login/oauth/authorize?client_id={GITHUB_CLIENT_ID}&scope=repo,read:user&state={state}")

@app.get("/api/auth/callback")
@advanced_rate_limit("10/minute")
async def auth_callback(request: Request, code: str, state: str):
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
@advanced_rate_limit("20/minute")
async def read_users_me(request: Request, current_user: dict = Depends(get_current_user)):
    if not current_user: 
        return {"authenticated": False}
    payload = {k: v for k, v in current_user.items() if k != 'github_token'}
    payload["authenticated"] = True
    return payload

# --- Enhanced Core API Endpoints ---
@app.post("/api/analyze", response_model=AnalysisResponse)
@advanced_rate_limit("10/minute")
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
                if not headers.get("Authorization"):
                    raise HTTPException(status_code=403, detail="Access to public repository failed. This may be due to GitHub's API rate limits for unauthenticated users. Please try again later or log in with GitHub to increase your rate limit.")
                else:
                    raise HTTPException(status_code=403, detail="Access denied. The repository may be private or you may not have sufficient permissions.")
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
    background_tasks.add_task(enhanced_index_repository_task, str(req.url), user_github_token)

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

# FIXED: Added the missing README generation endpoint
@app.post("/api/generate_readme", response_model=ReadmeResponse)
@advanced_rate_limit("5/minute")  # Lower rate limit for resource-intensive operation
async def generate_readme(
    request: Request,
    payload: RepoURL, 
    current_user: dict = Depends(get_current_user)
):
    """
    Generates a README.md file for a given repository based on its file structure.
    """
    repo_url = payload.url.strip()
    
    try:
        # Validate GitHub URL
        owner, repo_name = parse_github_url(repo_url)
    except HTTPException as e:
        raise e
    
    # Get the local repository path
    repo_local_path = get_repo_local_path(repo_url)
    
    if not os.path.exists(repo_local_path):
        raise HTTPException(
            status_code=404, 
            detail="Repository not found locally. Please analyze the repository first to clone it."
        )

    try:
        # Get file tree structure
        logger.info(f"Generating README for {repo_url} using local path: {repo_local_path}")
        file_tree = get_file_tree(repo_local_path, max_depth=4)
        
        if not file_tree.strip():
            raise HTTPException(
                status_code=404,
                detail="No analyzable files found in the repository structure."
            )
        
        # Analyze key files for better README generation
        key_files_analysis = ""
        key_file_names = [
            'package.json', 'requirements.txt', 'Dockerfile', 'docker-compose.yml',
            'main.py', 'app.py', 'index.js', 'server.js', 'pom.xml', 'build.gradle'
        ]
        
        for root, dirs, files in os.walk(repo_local_path):
            # Skip hidden directories and common build/cache dirs
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['node_modules', '__pycache__', 'venv']]
            
            for file in files:
                if file in key_file_names:
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()[:1000]  # First 1000 chars
                            relative_path = os.path.relpath(file_path, repo_local_path)
                            key_files_analysis += f"\n--- {relative_path} ---\n{content}\n"
                    except Exception as e:
                        logger.warning(f"Could not read key file {file_path}: {e}")
                        continue
        
        # Enhanced prompt for README generation
        prompt = f"""
You are a technical documentation expert. Generate a comprehensive, professional README.md file for this GitHub repository.

## Repository Information:
- Repository: {owner}/{repo_name}
- Analysis based on file structure and key configuration files

## File Structure:
```
{file_tree}
```

## Key Configuration Files Found:
{key_files_analysis if key_files_analysis.strip() else "No key configuration files detected."}

## Instructions:
Generate a complete README.md file with the following sections (adapt based on what you can infer from the structure):

1. **Project Title** - Clear, descriptive title
2. **Description** - What the project does and why it's useful
3. **Features** - Key capabilities (bullet points)
4. **Tech Stack** - Technologies, frameworks, languages used
5. **Prerequisites** - System requirements, dependencies
6. **Installation** - Step-by-step setup instructions
7. **Usage** - How to run/use the application with examples
8. **Project Structure** - Brief explanation of main directories/files
9. **Contributing** - How others can contribute
10. **License** - Mention license (if detectable) or suggest adding one

## Guidelines:
- Use proper markdown formatting
- Be specific and actionable
- Include code examples where appropriate
- Make installation steps clear and sequential
- Infer the project type (web app, library, CLI tool, etc.) from structure
- Use modern, professional language
- Include badges or shields where relevant
- Make it beginner-friendly but comprehensive

Generate ONLY the markdown content. Do not include any explanatory text before or after the README content.
"""

        # Generate README using Groq
        readme_content = await call_groq_chat(
            messages=[
                {"role": "system", "content": "You are a technical documentation expert specializing in creating comprehensive, professional README files for GitHub repositories."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,  # Lower temperature for more consistent, structured output
            max_tokens=3000   # Allow for longer, more detailed README
        )
        
        if not readme_content or len(readme_content.strip()) < 100:
            raise HTTPException(
                status_code=500,
                detail="Generated README content is too short or empty. Please try again."
            )
        
        logger.info(f"Successfully generated README for {repo_url} ({len(readme_content)} characters)")
        
        return ReadmeResponse(readme_content=readme_content)
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.error(f"README generation failed for {repo_url}: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to generate README: {str(e)[:100]}..."
        )

@app.get("/api/index_status")
@advanced_rate_limit("30/minute")
async def get_index_status(request: Request, url: str):
    """Get indexing status for a repository"""
    try:
        # Parse the URL to validate it
        parse_github_url(url)
        return vector_store_manager.get_status(url)
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid URL: {str(e)}")

@app.post("/api/qna", response_model=QnaResponse)
@advanced_rate_limit("15/minute")
async def ask_question(request: Request, req: QnaRequest, current_user: dict = Depends(get_current_user)):
    repo_url_key = str(req.url)
    
    # Enhanced input validation
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
        # 1. ENHANCED MULTI-STAGE RETRIEVAL
        # First, get a larger pool of potentially relevant documents
        retriever = vector_store.as_retriever(search_kwargs={"k": 20})
        candidate_docs = retriever.get_relevant_documents(req.question)
        
        if not candidate_docs:
            return QnaResponse(answer="I couldn't find any relevant information in the codebase to answer that question.")

        # 2. SEMANTIC RE-RANKING using Groq
        # Use Groq to rank the relevance of documents
        doc_summaries = []
        for i, doc in enumerate(candidate_docs[:15]):  # Limit for ranking
            summary = doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content
            doc_summaries.append(f"Document {i}: {doc.metadata.get('source', 'unknown')}\n{summary}")
        
        ranking_messages = [
            {"role": "system", "content": "You are a code analysis assistant. Rank documents by relevance to the user's question."},
            {"role": "user", "content": f"""Rank these code documents by relevance to the question: "{req.question}"

Documents:
{chr(10).join(doc_summaries)}

Return only the document numbers (0-{len(doc_summaries)-1}) in order of relevance, separated by commas. Most relevant first.
Example: 3,1,7,2"""}
        ]
        
        try:
            ranking_response = await call_groq_chat(ranking_messages, max_tokens=100)
            ranked_indices = [int(x.strip()) for x in ranking_response.strip().split(',') if x.strip().isdigit()]
            # Reorder documents based on AI ranking
            relevant_docs = [candidate_docs[i] for i in ranked_indices[:8] if i < len(candidate_docs)]
        except:
            # Fallback to original ordering if ranking fails
            relevant_docs = candidate_docs[:8]

        # 3. ENHANCED CONTEXT CREATION WITH CODE STRUCTURE ANALYSIS
        context_parts = []
        file_types = {}
        total_functions = 0
        
        for i, doc in enumerate(relevant_docs):
            file_path = doc.metadata.get('source', f'file_{i}')
            file_type = doc.metadata.get('file_type', 'unknown')
            content = doc.page_content
            
            # Analyze code structure
            if file_type in ['.py', '.js', '.jsx', '.ts', '.tsx', '.java', '.go']:
                # Extract function/class definitions
                if file_type == '.py':
                    functions = re.findall(r'def\s+(\w+)', content)
                    classes = re.findall(r'class\s+(\w+)', content)
                elif file_type in ['.js', '.jsx', '.ts', '.tsx']:
                    functions = re.findall(r'function\s+(\w+)|const\s+(\w+)\s*=|(\w+)\s*:\s*function', content)
                    classes = re.findall(r'class\s+(\w+)', content)
                else:
                    functions = []
                    classes = []
                
                total_functions += len(functions)
                structure_info = ""
                if functions:
                    function_names = [name for tpl in functions for name in tpl if name]
                    structure_info += f"\nFunctions: {', '.join(function_names)[:100]}"
                if classes:
                    structure_info += f"\nClasses: {', '.join(classes)[:100]}"
            else:
                structure_info = ""
            
            file_types[file_type] = file_types.get(file_type, 0) + 1
            
            # Truncate content intelligently - preserve complete functions/classes
            if len(content) > 1200:
                lines = content.split('\n')
                if file_type in ['.py', '.js', '.jsx', '.ts', '.tsx']:
                    # Try to keep complete functions
                    truncated_lines = []
                    char_count = 0
                    for line in lines:
                        if char_count + len(line) > 1200:
                            break
                        truncated_lines.append(line)
                        char_count += len(line)
                    content = '\n'.join(truncated_lines)
                else:
                    content = content[:1200]
            
            context_parts.append(f"=== {file_path} ==={structure_info}\n{content}")
        
        context = "\n\n".join(context_parts)
        
        # 4. QUESTION CLASSIFICATION AND SPECIALIZED PROMPTING
        classification_messages = [
            {"role": "system", "content": "You are a code analysis expert. Classify questions about codebases into specific categories."},
            {"role": "user", "content": f"""Classify this question about a codebase into one of these categories:
1. ARCHITECTURE - about overall structure, design patterns, organization
2. FUNCTIONALITY - about what the code does, features, behavior
3. IMPLEMENTATION - about how specific parts work, algorithms, logic
4. DEBUGGING - about errors, issues, problems in the code
5. IMPROVEMENT - about optimization, refactoring, best practices
6. DOCUMENTATION - about usage, setup, configuration
7. GENERAL - general questions about the project

Question: "{req.question}"

Return only the category name."""}
        ]
        
        try:
            classification_response = await call_groq_chat(classification_messages, max_tokens=20)
            question_category = classification_response.strip().upper()
        except:
            question_category = "GENERAL"
        
        # 5. SPECIALIZED PROMPTS BASED ON QUESTION TYPE
        specialized_instructions = {
            "ARCHITECTURE": """Focus on:
- Overall code organization and structure
- Design patterns and architectural decisions  
- Module/component relationships
- Separation of concerns
- Code organization principles""",
            
            "FUNCTIONALITY": """Focus on:
- What the code accomplishes
- Key features and capabilities
- User-facing functionality
- Business logic and workflows
- Input/output behavior""",
            
            "IMPLEMENTATION": """Focus on:
- Specific algorithms and logic
- Implementation details
- Code flow and execution paths
- Technical approaches used
- Low-level mechanics""",
            
            "DEBUGGING": """Focus on:
- Potential issues or bugs
- Error handling mechanisms
- Common failure points
- Debugging strategies
- Code quality concerns""",
            
            "IMPROVEMENT": """Focus on:
- Optimization opportunities
- Refactoring suggestions
- Best practice recommendations
- Performance considerations
- Code quality improvements""",
            
            "DOCUMENTATION": """Focus on:
- Setup and installation steps
- Usage instructions
- Configuration options
- API documentation
- Examples and tutorials""",
            
            "GENERAL": """Provide a comprehensive overview addressing the question from multiple angles."""
        }
        
        category_instruction = specialized_instructions.get(question_category, specialized_instructions["GENERAL"])
        
        # 6. DYNAMIC PROMPT CONSTRUCTION
        project_context = ""
        if file_types:
            main_languages = sorted(file_types.items(), key=lambda x: x[1], reverse=True)[:3]
            project_context = f"This appears to be primarily a {', '.join([lang for lang, _ in main_languages])} project with {total_functions} functions analyzed."
        
        # 7. ENHANCED GROQ PROMPT WITH STRUCTURED OUTPUT
        enhanced_messages = [
            {"role": "system", "content": f"""You are a senior software engineer and code architect analyzing a codebase. {project_context}

ANALYSIS FOCUS: {category_instruction}

RESPONSE REQUIREMENTS:
- Be precise and technical when discussing code
- Reference specific files and line numbers when possible
- Provide concrete examples from the codebase
- Structure your response with clear sections
- Include code snippets when helpful
- If information is incomplete, state what additional context would help"""},
            {"role": "user", "content": f"""CODEBASE CONTEXT:
{context}

QUESTION: {req.question}

STRUCTURED ANALYSIS:"""}
        ]
        
        # 8. ENHANCED AI GENERATION WITH BETTER PARAMETERS
        answer_text = await call_groq_chat(
            enhanced_messages,
            model=GROQ_MODEL,
            temperature=0.1,
            max_tokens=2048
        )
        
        if not answer_text:
            return QnaResponse(answer="I apologize, but I couldn't generate a proper response. Please try rephrasing your question.")
        
        # 9. POST-PROCESS RESPONSE FOR BETTER FORMATTING
        # Add section breaks for better readability
        answer_text = re.sub(r'\n\n+', '\n\n', answer_text)  # Normalize line breaks
        
        # 10. EXTRACT MORE DETAILED SOURCE INFORMATION
        source_details = []
        for doc in relevant_docs[:5]:
            source_path = doc.metadata.get('source', 'unknown')
            file_type = doc.metadata.get('file_type', '')
            size = doc.metadata.get('size', 0)
            source_details.append({
                "file": source_path,
                "type": file_type,
                "relevance": "high" if doc in relevant_docs[:3] else "medium"
            })
        
        return QnaResponse(
            answer=answer_text,
            sources=source_details,
            metadata={
                "question_category": question_category.lower(),
                "files_analyzed": len(relevant_docs),
                "primary_languages": [lang for lang, _ in sorted(file_types.items(), key=lambda x: x[1], reverse=True)[:3]],
                "functions_found": total_functions
            }
        )
        
    except Exception as e:
        logger.error(f"Enhanced Q&A failed for {repo_url_key}: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate answer. Please try again.")

@app.post("/api/qna/followup", response_model=QnaResponse)
@advanced_rate_limit("20/minute")
async def ask_followup_question(request: Request, req: FollowupRequest, current_user: dict = Depends(get_current_user)):
    """Enhanced endpoint for follow-up questions that maintains conversation context"""
    repo_url_key = str(req.url)
    
    vector_store = vector_store_manager.get_store(repo_url_key)
    if vector_store is None:
        raise HTTPException(status_code=404, detail="Repository not indexed.")
    
    try:
        # Use previous context to inform the search
        enhanced_query = f"{req.previous_context[:200]}... {req.question}" if req.previous_context else req.question
        
        retriever = vector_store.as_retriever(search_kwargs={"k": 10})
        relevant_docs = retriever.get_relevant_documents(enhanced_query)
        
        if not relevant_docs:
            return QnaResponse(
                answer="I couldn't find relevant information for this follow-up question.",
                is_followup=True
            )
        
        context = "\n\n".join([
            f"=== {doc.metadata.get('source', 'unknown')} ===\n{doc.page_content[:800]}" 
            for doc in relevant_docs[:5]
        ])
        
        followup_messages = [
            {"role": "system", "content": "You are continuing a conversation about this codebase. Provide focused answers that build on previous discussion."},
            {"role": "user", "content": f"""PREVIOUS CONTEXT: {req.previous_context[:300] if req.previous_context else "None"}

CURRENT QUESTION: {req.question}

CODEBASE CONTEXT:
{context}

Provide a focused answer that builds on the previous discussion:"""}
        ]
        
        response_text = await call_groq_chat(
            followup_messages,
            temperature=0.2,
            max_tokens=1536
        )
        
        return QnaResponse(
            answer=response_text,
            sources=[{"file": doc.metadata.get('source', 'unknown'), "type": doc.metadata.get('file_type', ''), "relevance": "medium"} for doc in relevant_docs[:3]],
            is_followup=True
        )
        
    except Exception as e:
        logger.error(f"Follow-up Q&A failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to process follow-up question.")

# --- Additional Utility Endpoints ---
@app.delete("/api/index/{repo_hash}")
@advanced_rate_limit("10/minute")
async def delete_repository_index(request: Request, repo_hash: str, current_user: dict = Depends(get_current_user)):
    """Delete a repository index (admin or owner only)"""
    if not current_user:
        raise HTTPException(status_code=401, detail="Authentication is required for this action.")

    # Simple hash-based lookup (you could implement more sophisticated logic)
    found_key = None
    for key in vector_store_manager._stores.keys():
        if hashlib.md5(key.encode()).hexdigest()[:8] == repo_hash:
            found_key = key
            break
    
    if not found_key:
        raise HTTPException(status_code=404, detail="Repository index not found.")
    
    vector_store_manager._remove_key(found_key)
    return {"message": "Repository index deleted successfully.", "repo": found_key}

@app.get("/api/stats")
@advanced_rate_limit("30/minute")
async def get_system_stats(request: Request):
    """Get system statistics"""
    stats = vector_store_manager.get_stats()
    
    # Enhanced stats
    status_breakdown = {}
    for status_info in vector_store_manager._statuses.values():
        status = status_info.get("status", "unknown")
        status_breakdown[status] = status_breakdown.get(status, 0) + 1
    
    stats["status_breakdown"] = status_breakdown
    stats["uptime_seconds"] = time.time() - app.state.start_time if hasattr(app.state, 'start_time') else 0
    
    return stats

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
            "message": "An unexpected error occurred. Please try again later.",
            "status_code": 500,
            "timestamp": datetime.utcnow().isoformat()
        }
    )

# --- Startup Events ---
@app.on_event("startup")
async def startup_event():
    """Initialize application state on startup"""
    app.state.start_time = time.time()
    logger.info("GitHub Project Intelligence Assistant API (Groq-powered) started successfully")
    
    # Log configuration status
    logger.info(f"GitHub OAuth: {'Enabled' if GITHUB_CLIENT_ID and GITHUB_CLIENT_SECRET else 'Disabled'}")
    logger.info(f"GitHub API Token: {'Set' if GITHUB_API_TOKEN else 'Not Set'}")
    logger.info(f"Redis Cache: {'Available' if redis_client else 'Unavailable'}")
    logger.info(f"Groq Model: {GROQ_MODEL}")
    logger.info(f"Embedding Model: {EMBEDDING_MODEL} (Local)")
    
    # Test Groq connection and log available models
    if async_groq_client:
        try:
            test_response = await call_groq_chat([{"role": "user", "content": "Hi"}], max_tokens=5)
            logger.info("Groq API connection test successful")
        except Exception as e:
            logger.warning(f"Groq API connection test failed: {e}")
    else:
        logger.warning("Groq API client not initialized - check GROQ_API_KEY")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on application shutdown"""
    logger.info("Shutting down GitHub Project Intelligence Assistant API")
    
    # Close Redis connection if available
    if redis_client:
        try:
            redis_client.close()
        except:
            pass

# --- Main Entry Point ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )