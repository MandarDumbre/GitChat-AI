import React, { useState, useEffect, useRef } from 'react';
import { marked } from 'marked';

// Configure marked for better rendering (code blocks, etc.)
marked.setOptions({
  gfm: true,
  breaks: true,
  sanitize: false,
});

// --- Icon Components (inline SVGs) ---
const GithubIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M9 19c-5 1.5-5-2.5-7-3m14 6v-3.87a3.37 3.37 0 0 0-.94-2.61c3.14-.35 6.44-1.54 6.44-7A5.44 5.44 0 0 0 20 4.77 5.07 5.07 0 0 0 19.91 1S18.73.65 16 2.48a13.38 13.38 0 0 0-7 0C6.27.65 5.09 1 5.09 1A5.07 5.07 0 0 0 5 4.77a5.44 5.44 0 0 0-1.5 3.78c0 5.42 3.3 6.61 6.44 7A3.37 3.37 0 0 0 9 18.13V22"></path>
  </svg>
);

const SearchIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <circle cx="11" cy="11" r="8"></circle>
    <line x1="21" y1="21" x2="16.65" y2="16.65"></line>
  </svg>
);

const SparklesIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M12 3L9.27 9.27L3 12l6.27 2.73L12 21l2.73-6.27L21 12l-6.27-2.73L12 3z"/>
    <path d="M3 21l1.5-3.5L8 16l-3.5-1.5L1 11l1.5 3.5L6 18l-3.5 1.5z"/>
  </svg>
);

const LoadingSpinner = () => (
  <svg className="animate-spin h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
  </svg>
);

const LogoutIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4"></path>
    <polyline points="16 17 21 12 16 7"></polyline>
    <line x1="21" y1="12" x2="9" y2="12"></line>
  </svg>
);

const AlertIcon = () => (
  <svg className="w-12 h-12 text-red-400" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"></path>
  </svg>
);

// --- Diagnostic Error Component ---
const ConnectionError = ({ apiUrl }) => (
  <div className="max-w-3xl mx-auto mt-10 p-6 bg-red-900/70 border-2 border-red-600 rounded-xl text-center shadow-lg">
    <div className="flex justify-center items-center mb-4">
      <AlertIcon />
      <h3 className="text-2xl font-bold text-red-200 ml-4">Backend Connection Failed</h3>
    </div>
    <p className="mt-2 text-red-300">
      The application could not connect to the backend server at{' '}
      <code className="bg-red-800/50 px-1.5 py-1 rounded-md font-mono">{apiUrl}</code>.
    </p>
    <p className="mt-6 text-gray-200">Please perform the following checks:</p>
    <ul className="text-left list-disc list-inside mt-2 text-gray-300 inline-block">
      <li>Is the Python backend server running? Start it with: <code className="bg-gray-700 px-1.5 py-1 rounded-md font-mono">uvicorn main:app --reload</code></li>
      <li>Is the backend running on the correct port (8000)?</li>
      <li>Are there any CORS errors in your browser's developer console?</li>
    </ul>
  </div>
);

// --- Main Application Component ---
export default function App() {
  // Auth state
  const [user, setUser] = useState(null);
  const [token, setToken] = useState(null);

  // Core state
  const [repoUrl, setRepoUrl] = useState('');
  const [analysis, setAnalysis] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState({ type: null, message: '' });

  // Indexing state
  const [indexingStatus, setIndexingStatus] = useState('idle');
  const [indexingDetail, setIndexingDetail] = useState('');
  const [indexingProgress, setIndexingProgress] = useState(0);
  const pollingIntervalRef = useRef(null);

  // Q&A state
  const [question, setQuestion] = useState('');
  const [answer, setAnswer] = useState('');
  const [isAsking, setIsAsking] = useState(false);
  const [qnaError, setQnaError] = useState('');
  const [sources, setSources] = useState([]);

  // README Generation state
  const [generatedReadme, setGeneratedReadme] = useState('');
  const [isGeneratingReadme, setIsGeneratingReadme] = useState(false);
  const [readmeError, setReadmeError] = useState('');

  const API_BASE_URL = 'http://127.0.0.1:8000';

  // --- Auth Effects ---
  useEffect(() => {
    const urlParams = new URLSearchParams(window.location.search);
    const urlToken = urlParams.get('token');
    if (urlToken) {
      setToken(urlToken);
      window.history.replaceState({}, document.title, "/");
    }
  }, []);

  useEffect(() => {
    if (token) {
      const fetchUser = async () => {
        try {
          const response = await fetch(`${API_BASE_URL}/api/me`, { 
            headers: { 'Authorization': `Bearer ${token}` } 
          });
          if (response.ok) {
            const userData = await response.json();
            if (userData.authenticated) {
              setUser(userData);
            } else {
              handleLogout();
            }
          } else {
            handleLogout();
          }
        } catch (error) {
          console.error('Error fetching user data:', error);
          handleLogout();
        }
      };
      fetchUser();
    }
  }, [token]);
  
  useEffect(() => {
    // Cleanup polling on unmount
    return () => {
      if (pollingIntervalRef.current) {
        clearInterval(pollingIntervalRef.current);
      }
    };
  }, []);

  // --- Core Logic ---
  const getAuthHeaders = () => token ? { 'Authorization': `Bearer ${token}` } : {};
  
  const handleLogin = () => {
    window.location.href = `${API_BASE_URL}/api/auth/login`;
  };
  
  const handleLogout = () => {
    setToken(null);
    setUser(null);
  };
  
  const stopPolling = () => {
    if (pollingIntervalRef.current) {
      clearInterval(pollingIntervalRef.current);
      pollingIntervalRef.current = null;
    }
  };

  const handleGenerateReadme = async () => {
    setReadmeError('');
    setGeneratedReadme('');
    setIsGeneratingReadme(true);

    try {
      const response = await fetch(`${API_BASE_URL}/api/generate_readme`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...getAuthHeaders()
        },
        body: JSON.stringify({ url: repoUrl.trim() }),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `Failed to generate README. Status: ${response.status}`);
      }

      const data = await response.json();
      setGeneratedReadme(data.readme_content);
    } catch (err) {
      console.error("README Generation failed:", err);
      setReadmeError(err.message || 'An unknown error occurred while generating the README.');
    } finally {
      setIsGeneratingReadme(false);
    }
  };

  const handleAnalyze = async () => {
    if (!repoUrl.trim()) {
      setError({ type: 'generic', message: 'Please enter a GitHub repository URL.' });
      return;
    }

    // Reset all states
    setError({ type: null, message: '' });
    setAnalysis(null);
    setAnswer('');
    setQuestion('');
    setQnaError('');
    setSources([]);
    setIndexingStatus('idle');
    setIndexingDetail('');
    setIndexingProgress(0);
    setGeneratedReadme('');
    setReadmeError('');
    stopPolling();
    setIsLoading(true);

    try {
      const response = await fetch(`${API_BASE_URL}/api/analyze`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...getAuthHeaders()
        },
        body: JSON.stringify({ url: repoUrl.trim() }),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.message || errorData.detail || `HTTP error! Status: ${response.status}`);
      }

      const data = await response.json();
      setAnalysis(data);
      startPollingForIndexStatus();
    } catch (err) {
      console.error("Analysis failed:", err);
      if (err instanceof TypeError && err.message.includes('Failed to fetch')) {
        setError({ 
          type: 'connection', 
          message: `Network Error: Could not connect to the backend server at ${API_BASE_URL}` 
        });
      } else {
        setError({ 
          type: 'generic', 
          message: err.message || 'Failed to analyze repository.' 
        });
      }
    } finally {
      setIsLoading(false);
    }
  };

  const startPollingForIndexStatus = () => {
    const pollStatus = async () => {
      try {
        const statusResponse = await fetch(
          `${API_BASE_URL}/api/index_status?url=${encodeURIComponent(repoUrl.trim())}`
        );
        
        if (!statusResponse.ok) {
          throw new Error(`Status check failed: ${statusResponse.status}`);
        }
        
        const statusData = await statusResponse.json();
        setIndexingStatus(statusData.status || 'idle');
        setIndexingDetail(statusData.detail || '');
        setIndexingProgress(statusData.progress || 0);
        
        // Stop polling if indexing is complete
        if (['completed', 'failed', 'empty'].includes(statusData.status)) {
          stopPolling();
        }
      } catch (err) {
        console.error("Polling error:", err);
        setIndexingStatus('failed');
        setIndexingDetail('Error fetching indexing status.');
        stopPolling();
      }
    };

    // Initial status check
    pollStatus();
    
    // Start polling every 2.5 seconds
    pollingIntervalRef.current = setInterval(pollStatus, 2500);
  };

  const handleAskQuestion = async () => {
    const trimmedQuestion = question.trim();
    if (!trimmedQuestion) {
      setQnaError('Please enter a question.');
      return;
    }

    if (trimmedQuestion.length < 3) {
      setQnaError('Question must be at least 3 characters long.');
      return;
    }

    setQnaError('');
    setAnswer('');
    setSources([]);
    setIsAsking(true);

    try {
      const response = await fetch(`${API_BASE_URL}/api/qna`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...getAuthHeaders()
        },
        body: JSON.stringify({ 
          url: repoUrl.trim(), 
          question: trimmedQuestion 
        }),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        if (response.status === 202) {
          throw new Error('Repository is still being indexed. Please wait for indexing to complete.');
        } else if (response.status === 404) {
          throw new Error('Repository not indexed yet. Please wait for the analysis to complete.');
        } else {
          throw new Error(errorData.message || errorData.detail || `HTTP error! Status: ${response.status}`);
        }
      }

      const data = await response.json();
      setAnswer(data.answer || 'No answer received.');
      setSources(data.sources || []);
    } catch (err) {
      console.error("Q&A failed:", err);
      setQnaError(err.message || 'Failed to get an answer. Please try again.');
    } finally {
      setIsAsking(false);
    }
  };

  const getIndexingMessage = () => {
    switch (indexingStatus) {
      case 'indexing':
        return indexingDetail || 'Cloning and analyzing codebase...';
      case 'completed':
        return 'Codebase analysis complete. You can now ask questions about the code.';
      case 'failed':
        return `Error: ${indexingDetail || 'Codebase analysis failed.'}`;
      case 'empty':
        return 'Analysis complete, but no indexable code files were found in this repository.';
      default:
        return 'Initializing code analysis...';
    }
  };

  const getIndexingStatusColor = () => {
    switch (indexingStatus) {
      case 'completed':
        return 'bg-green-900/50 border-green-600';
      case 'failed':
        return 'bg-red-900/50 border-red-600';
      case 'indexing':
        return 'bg-blue-900/50 border-blue-600';
      default:
        return 'bg-yellow-900/50 border-yellow-600';
    }
  };
  
  const getRelevanceColor = (relevance) => {
    if (relevance === 'high') return 'border-green-500';
    if (relevance === 'medium') return 'border-yellow-500';
    return 'border-gray-600';
  };

  // --- Conditional Rendering ---

  // 1. If no token, show the login screen.
  if (!token) {
    return (
      <div className="bg-gray-900 text-white min-h-screen font-sans flex flex-col justify-center items-center p-4 text-center">
        <div className="space-y-8 max-w-lg">
          <div>
            <h1 className="text-4xl md:text-5xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-purple-400 to-indigo-600">
              GitChat AI
            </h1>
            <h2 className="text-xl md:text-2xl text-gray-400 mt-2">The Intelligent Assistant and Readme Generator</h2>
          </div>
          <p className="text-gray-300 text-xl">
            Login with GitHub to continue
          </p>
          <button 
            onClick={handleLogin} 
            className="bg-indigo-600 hover:bg-indigo-700 text-white font-bold py-3 px-6 rounded-lg flex items-center transition-colors text-lg mx-auto shadow-lg hover:shadow-indigo-500/50"
          >
            <GithubIcon />
            <span className="ml-3">Login</span>
          </button>
        </div>
      </div>
    );
  }

  // 2. If there's a token but no user object yet, we are authenticating.
  if (!user) {
    return (
      <div className="bg-gray-900 text-white min-h-screen font-sans flex flex-col justify-center items-center">
        <div className="flex items-center">
          <LoadingSpinner />
          <p className="ml-4 text-lg text-gray-300">Authenticating...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-gray-900 text-white min-h-screen font-sans">
      <div className="container mx-auto p-4 md:p-8">
        <header className="flex flex-col md:flex-row justify-between items-center mb-8 gap-4">
          <h1 className="text-2xl md:text-3xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-purple-400 to-indigo-600">
            GitChat AI
          </h1>
          <div className="flex items-center space-x-4">
            <img 
              src={user.avatar_url} 
              alt={user.name || user.sub} 
              className="w-10 h-10 rounded-full border-2 border-indigo-500"
            />
            <span className="hidden md:inline text-gray-300">
              {user.name || user.sub}
            </span>
            <button 
              onClick={handleLogout} 
              className="flex items-center space-x-2 bg-gray-700 hover:bg-gray-600 text-white font-semibold py-2 px-4 rounded-lg transition-colors"
            >
              <LogoutIcon />
              <span>Logout</span>
            </button>
          </div>
        </header>

        <div className="text-center mb-12">
          <p className="text-gray-400 mt-4 max-w-2xl mx-auto">
            Welcome, {user.name || user.sub}! Analyze any public repository with AI-powered interactions.
          </p>
        </div>

        <div className="max-w-2xl mx-auto">
          <div className="flex flex-col md:flex-row items-stretch bg-gray-800 border border-gray-700 rounded-lg p-2 shadow-lg gap-2">
            <div className="flex items-center flex-1">
              <span className="pl-2 text-gray-500">
                <GithubIcon />
              </span>
              <input 
                type="text" 
                value={repoUrl} 
                onChange={(e) => setRepoUrl(e.target.value)}
                placeholder="e.g., https://github.com/owner-name/repo-name"
                className="w-full p-2 bg-transparent text-white focus:outline-none"
                onKeyPress={(e) => e.key === 'Enter' && !isLoading && handleAnalyze()}
                disabled={isLoading}
              />
            </div>
            <button 
              onClick={handleAnalyze} 
              disabled={isLoading} 
              className="bg-indigo-600 hover:bg-indigo-700 disabled:bg-indigo-900 disabled:cursor-not-allowed text-white font-bold py-2 px-4 rounded-md flex items-center justify-center transition-colors duration-300 min-w-[120px]"
            >
              {isLoading ? <LoadingSpinner /> : <SearchIcon />}
              <span className="ml-2">{isLoading ? "Analyzing..." : "Analyze"}</span>
            </button>
          </div>
        </div>

        {error.type === 'connection' ? (
          <ConnectionError apiUrl={API_BASE_URL} />
        ) : (
          <>
            {error.type === 'generic' && (
              <div className="max-w-2xl mx-auto mt-4 p-4 bg-red-900/50 border border-red-600 rounded-lg">
                <p className="text-red-200 text-center">{error.message}</p>
              </div>
            )}
            
            {analysis && (
              <div className="mt-12 space-y-8">
                {/* Project Header */}
                <div className="bg-gray-800/50 border border-gray-700 rounded-lg p-6">
                  <h2 className="text-3xl font-bold text-white mb-2">
                    {analysis.projectName}
                  </h2>
                  <p className="text-gray-400">{analysis.description}</p>
                  {analysis.techStack && analysis.techStack.length > 0 && (
                    <div className="mt-4">
                      <h4 className="text-sm font-semibold text-gray-300 mb-2">Tech Stack:</h4>
                      <div className="flex flex-wrap gap-2">
                        {analysis.techStack.map((tech, index) => (
                          <span 
                            key={index} 
                            className="bg-indigo-600/20 text-indigo-200 px-2 py-1 rounded-md text-sm"
                          >
                            {tech}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
                
                {/* Main Content Grid */}
                <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
                  {/* README Section */}
                  <div className="lg:col-span-2 bg-gray-800/50 border border-gray-700 rounded-lg p-6">
                    <div className="flex justify-between items-center mb-4">
                      <h3 className="text-xl font-bold text-white">README.md</h3>
                      <button
                        onClick={handleGenerateReadme}
                        disabled={isGeneratingReadme}
                        className="bg-teal-600 hover:bg-teal-700 disabled:bg-teal-900/50 disabled:cursor-not-allowed text-white font-semibold py-2 px-4 rounded-lg flex items-center transition-colors text-sm"
                      >
                        {isGeneratingReadme ? <LoadingSpinner /> : <SparklesIcon />}
                        <span className="ml-2">{isGeneratingReadme ? 'Generating...' : 'Generate README'}</span>
                      </button>
                    </div>
                    <div 
                      className="prose prose-invert prose-sm md:prose-base max-w-none text-gray-300 h-96 overflow-auto border border-gray-700 rounded-md p-4 bg-gray-900/50" 
                      dangerouslySetInnerHTML={{ __html: marked(analysis.readmeContent) }} 
                    />
                  </div>
                  
                  {/* Sidebar */}
                  <div className="lg:col-span-1 space-y-6">
                    {/* Recent Commits */}
                    <div className="bg-gray-800/50 border border-gray-700 rounded-lg p-6">
                      <h3 className="text-xl font-bold text-white mb-4">Recent Commits</h3>
                      {analysis.recentCommits && analysis.recentCommits.length > 0 ? (
                        <ul className="space-y-3">
                          {analysis.recentCommits.map(commit => (
                            <li key={commit.sha} className="text-sm">
                              <a 
                                href={commit.url} 
                                target="_blank" 
                                rel="noopener noreferrer"
                                className="block hover:bg-gray-700/50 p-2 rounded-md transition-colors"
                              >
                                <p className="font-mono text-indigo-400 truncate" title={commit.message}>
                                  {commit.message}
                                </p>
                                <p className="text-gray-500 mt-1">by {commit.author}</p>
                              </a>
                            </li>
                          ))}
                        </ul>
                      ) : (
                        <p className="text-sm text-gray-500">No recent commits found.</p>
                      )}
                    </div>
                    
                    {/* Open Issues */}
                    <div className="bg-gray-800/50 border border-gray-700 rounded-lg p-6">
                      <h3 className="text-xl font-bold text-white mb-4">Open Issues</h3>
                      {analysis.openIssues && analysis.openIssues.length > 0 ? (
                        <ul className="space-y-3">
                          {analysis.openIssues.map(issue => (
                            <li key={issue.number}>
                              <a 
                                href={issue.url} 
                                target="_blank" 
                                rel="noopener noreferrer" 
                                className="text-sm text-gray-300 hover:text-indigo-400 transition-colors block p-2 hover:bg-gray-700/50 rounded-md"
                                title={issue.title}
                              >
                                #{issue.number}: {issue.title}
                              </a>
                            </li>
                          ))}
                        </ul>
                      ) : (
                        <p className="text-sm text-gray-500">No open issues found.</p>
                      )}
                    </div>
                  </div>
                </div>
                
                {/* Generated README Section */}
                {isGeneratingReadme && (
                  <div className="mt-8 p-6 bg-gray-800/50 border border-gray-700 rounded-lg flex items-center justify-center">
                    <LoadingSpinner />
                    <p className="ml-4 text-gray-300">Generating new README, this may take a moment...</p>
                  </div>
                )}
                {readmeError && (
                  <div className="mt-8 p-4 bg-red-900/50 border border-red-600 rounded-lg">
                    <p className="text-red-200 text-center">{readmeError}</p>
                  </div>
                )}
                {generatedReadme && (
                  <div className="mt-8 bg-gray-800/50 border border-gray-700 rounded-lg p-6">
                    <div className="flex justify-between items-center mb-4">
                      <h3 className="text-xl font-bold text-white flex items-center">
                        <SparklesIcon />
                        <span className="ml-2">Generated README</span>
                      </h3>
                    </div>
                    <div
                      className="prose prose-invert max-w-none text-gray-300 border border-gray-700 rounded-md p-4 bg-gray-900/50"
                      dangerouslySetInnerHTML={{ __html: marked(generatedReadme) }}
                    />
                  </div>
                )}

                {/* Q&A Section */}
                <div className="bg-gray-800/50 border border-gray-700 rounded-lg p-6">
                  <h2 className="text-2xl font-bold text-white mb-4 flex items-center">
                    <SparklesIcon />
                    <span className="ml-2">AI Code Analysis & Q&A</span>
                  </h2>
                  
                  {/* Indexing Status */}
                  <div className={`p-4 rounded-md mb-6 flex items-center text-sm border ${getIndexingStatusColor()}`}>
                    {indexingStatus === 'indexing' && (
                      <div className="flex items-center mr-3">
                        <LoadingSpinner />
                        {indexingProgress > 0 && (
                          <div className="ml-3 bg-gray-700 rounded-full h-2 w-24">
                            <div 
                              className="bg-blue-500 h-2 rounded-full transition-all duration-300" 
                              style={{ width: `${Math.min(100, Math.max(0, indexingProgress))}%` }}
                            />
                          </div>
                        )}
                      </div>
                    )}
                    <div className="flex-1">
                      <p className="text-gray-200 font-medium">{getIndexingMessage()}</p>
                      {indexingProgress > 0 && indexingStatus === 'indexing' && (
                        <p className="text-gray-400 text-xs mt-1">
                          Progress: {Math.round(indexingProgress)}%
                        </p>
                      )}
                    </div>
                  </div>
                  
                  {/* Question Input */}
                  <div className="space-y-4">
                    <div className="flex flex-col md:flex-row gap-3">
                      <input 
                        type="text" 
                        value={question} 
                        onChange={(e) => setQuestion(e.target.value)}
                        placeholder={
                          indexingStatus === 'completed' 
                            ? "Ask about the code structure, functions, dependencies..." 
                            : "Waiting for code analysis to complete..."
                        }
                        className="flex-1 p-3 bg-gray-700 border border-gray-600 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent disabled:opacity-50 disabled:cursor-not-allowed"
                        onKeyPress={(e) => e.key === 'Enter' && !isAsking && indexingStatus === 'completed' && handleAskQuestion()}
                        disabled={indexingStatus !== 'completed' || isAsking}
                      />
                      <button 
                        onClick={handleAskQuestion} 
                        disabled={indexingStatus !== 'completed' || isAsking || !question.trim()} 
                        className="bg-purple-600 hover:bg-purple-700 disabled:bg-purple-900/50 disabled:cursor-not-allowed text-white font-bold py-3 px-6 rounded-md flex items-center justify-center transition-colors min-w-[120px]"
                      >
                        {isAsking ? <LoadingSpinner /> : <SparklesIcon />}
                        <span className="ml-2">
                          {isAsking ? "Thinking..." : "Ask AI"}
                        </span>
                      </button>
                    </div>

                    {/* Q&A Error */}
                    {qnaError && (
                      <div className="p-3 bg-red-900/50 border border-red-600 rounded-md">
                        <p className="text-red-200 text-sm">{qnaError}</p>
                      </div>
                    )}

                    {/* Loading State */}
                    {isAsking && (
                      <div className="p-4 bg-blue-900/30 border border-blue-600 rounded-md">
                        <div className="flex items-center">
                          <LoadingSpinner />
                          <p className="text-blue-200 ml-3">
                            Analyzing codebase and generating answer...
                          </p>
                        </div>
                      </div>
                    )}

                    {/* Answer Display */}
                    {answer && (
                      <div className="space-y-4">
                        <div className="p-6 bg-gray-900 rounded-lg border border-gray-700">
                          <div className="flex items-center mb-3">
                            <SparklesIcon />
                            <h4 className="text-lg font-semibold text-white ml-2">AI Analysis</h4>
                          </div>
                          <div 
                            className="prose prose-invert max-w-none text-gray-300"
                            dangerouslySetInnerHTML={{ __html: marked(answer) }}
                          />
                        </div>

                        {/* Sources */}
                        {sources && sources.length > 0 && (
                          <div className="p-4 bg-gray-800/50 rounded-lg border border-gray-600">
                            <h5 className="text-sm font-semibold text-gray-300 mb-2">
                              Referenced Files:
                            </h5>
                            <div className="flex flex-wrap gap-2">
                              {sources.map((source, index) => (
                                <span 
                                  key={index}
                                  className={`bg-gray-700 text-gray-300 px-2 py-1 rounded text-xs font-mono border-l-2 ${getRelevanceColor(source.relevance)}`}
                                  title={`Type: ${source.type} | Relevance: ${source.relevance}`}
                                >
                                  {source.file}
                                </span>
                              ))}
                            </div>
                          </div>
                        )}
                      </div>
                    )}

                    {/* Example Questions */}
                    {indexingStatus === 'completed' && !answer && !isAsking && (
                      <div className="p-4 bg-gray-700/30 rounded-lg border border-gray-600">
                        <h5 className="text-sm font-semibold text-gray-300 mb-3">
                          💡 Example Questions:
                        </h5>
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-2 text-xs">
                          {[
                            "What is the main purpose of this project?",
                            "How is the code structured?",
                            "What are the key dependencies?",
                            "Are there any security concerns?",
                            "How can I contribute to this project?",
                            "What testing frameworks are used?"
                          ].map((example, index) => (
                            <button
                              key={index}
                              onClick={() => setQuestion(example)}
                              className="text-left p-2 bg-gray-600/50 hover:bg-gray-600 rounded text-gray-300 hover:text-white transition-colors"
                            >
                              {example}
                            </button>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            )}
          </>
        )}

        {/* Footer */}
        <footer className="mt-16 text-center text-gray-500 text-sm border-t border-gray-800 pt-8">
          <p>
            GitChat AI - Powered by AI for Smarter Code Analysis
          </p>
        </footer>
      </div>
    </div>
  );
}
