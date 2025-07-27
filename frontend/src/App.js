import React, { useState, useEffect, useRef } from 'react';
import { marked } from 'marked';

// --- Icon Components (inline SVGs) ---
const GithubIcon = () => <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M9 19c-5 1.5-5-2.5-7-3m14 6v-3.87a3.37 3.37 0 0 0-.94-2.61c3.14-.35 6.44-1.54 6.44-7A5.44 5.44 0 0 0 20 4.77 5.07 5.07 0 0 0 19.91 1S18.73.65 16 2.48a13.38 13.38 0 0 0-7 0C6.27.65 5.09 1 5.09 1A5.07 5.07 0 0 0 5 4.77a5.44 5.44 0 0 0-1.5 3.78c0 5.42 3.3 6.61 6.44 7A3.37 3.37 0 0 0 9 18.13V22"></path></svg>;
const SearchIcon = () => <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="11" cy="11" r="8"></circle><line x1="21" y1="21" x2="16.65" y2="16.65"></line></svg>;
const SparklesIcon = () => <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M12 3L9.27 9.27L3 12l6.27 2.73L12 21l2.73-6.27L21 12l-6.27-2.73L12 3z"/><path d="M3 21l1.5-3.5L8 16l-3.5-1.5L1 11l1.5 3.5L6 18l-3.5 1.5z"/></svg>;
const LoadingSpinner = () => <svg className="animate-spin h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg>;
const LogoutIcon = () => <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4"></path><polyline points="16 17 21 12 16 7"></polyline><line x1="21" y1="12" x2="9" y2="12"></line></svg>;

// --- Diagnostic Error Component ---
const ConnectionError = ({ apiUrl }) => (
    <div className="max-w-3xl mx-auto mt-10 p-6 bg-red-900/70 border-2 border-red-600 rounded-xl text-center shadow-lg">
        <div className="flex justify-center items-center mb-4">
            <svg className="w-12 h-12 text-red-400" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"></path></svg>
            <h3 className="text-2xl font-bold text-red-200 ml-4">Backend Connection Failed</h3>
        </div>
        <p className="mt-2 text-red-300">
            The application could not connect to the backend server at <code className="bg-red-800/50 px-1.5 py-1 rounded-md font-mono">{apiUrl}</code>.
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
    const pollingIntervalRef = useRef(null);

    // Q&A state
    const [question, setQuestion] = useState('');
    const [answer, setAnswer] = useState('');
    const [isAsking, setIsAsking] = useState(false);
    const [qnaError, setQnaError] = useState('');

    const API_BASE_URL = 'http://127.0.0.1:8000';

    // --- Auth Effects ---
    useEffect(() => {
        const urlParams = new URLSearchParams(window.location.search);
        const urlToken = urlParams.get('token');
        if (urlToken) {
            localStorage.setItem('github-assistant-token', urlToken);
            setToken(urlToken);
            window.history.replaceState({}, document.title, "/");
        } else {
            const storedToken = localStorage.getItem('github-assistant-token');
            if (storedToken) setToken(storedToken);
        }
    }, []);

    useEffect(() => {
        if (token) {
            const fetchUser = async () => {
                const response = await fetch(`${API_BASE_URL}/api/me`, { headers: { 'Authorization': `Bearer ${token}` } });
                if (response.ok) {
                    const userData = await response.json();
                    if (userData.authenticated) setUser(userData); else handleLogout();
                } else { handleLogout(); }
            };
            fetchUser().catch(() => handleLogout());
        }
    }, [token]);
    
    useEffect(() => () => { if (pollingIntervalRef.current) clearInterval(pollingIntervalRef.current); }, []);

    // --- Core Logic ---
    const getAuthHeaders = () => token ? { 'Authorization': `Bearer ${token}` } : {};
    const handleLogin = () => window.location.href = `${API_BASE_URL}/api/auth/login`;
    const handleLogout = () => {
        localStorage.removeItem('github-assistant-token');
        setToken(null);
        setUser(null);
    };
    
    const stopPolling = () => {
        if (pollingIntervalRef.current) {
            clearInterval(pollingIntervalRef.current);
            pollingIntervalRef.current = null;
        }
    };

    const handleAnalyze = async () => {
        if (!repoUrl) { setError({ type: 'generic', message: 'Please enter a GitHub repository URL.' }); return; }
        setError({ type: null, message: '' }); setAnalysis(null); setAnswer(''); setQuestion(''); setQnaError('');
        setIndexingStatus('idle'); stopPolling(); setIsLoading(true);

        try {
            const response = await fetch(`${API_BASE_URL}/api/analyze`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json', ...getAuthHeaders() },
                body: JSON.stringify({ url: repoUrl }),
            });
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || `HTTP error! Status: ${response.status}`);
            }
            const data = await response.json();
            setAnalysis(data);
            startPollingForIndexStatus();
        } catch (err) {
            console.error("Analysis failed:", err);
            if (err instanceof TypeError && err.message.includes('Failed to fetch')) {
                setError({ type: 'connection', message: `Network Error: Could not connect to the backend.` });
            } else {
                setError({ type: 'generic', message: err.message || 'Failed to analyze repository.' });
            }
        } finally {
            setIsLoading(false);
        }
    };

    const startPollingForIndexStatus = () => {
        pollingIntervalRef.current = setInterval(async () => {
            try {
                const statusResponse = await fetch(`${API_BASE_URL}/api/index_status?url=${encodeURIComponent(repoUrl)}`);
                if (!statusResponse.ok) throw new Error('Could not fetch indexing status.');
                const statusData = await statusResponse.json();
                setIndexingStatus(statusData.status);
                setIndexingDetail(statusData.detail || '');
                if (['completed', 'failed', 'empty'].includes(statusData.status)) {
                    stopPolling();
                }
            } catch (err) {
                console.error("Polling error:", err);
                setIndexingStatus('failed');
                setIndexingDetail('Error fetching status.');
                stopPolling();
            }
        }, 2500);
    };

    const handleAskQuestion = async () => {
        if (!question) { setQnaError('Please enter a question.'); return; }
        setQnaError(''); setAnswer(''); setIsAsking(true);
        try {
            const response = await fetch(`${API_BASE_URL}/api/qna`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json', ...getAuthHeaders() },
                body: JSON.stringify({ url: repoUrl, question }),
            });
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || `HTTP error! Status: ${response.status}`);
            }
            const data = await response.json();
            setAnswer(data.answer);
        } catch (err) {
            console.error("Q&A failed:", err);
            setQnaError(err.message || 'Failed to get an answer.');
        } finally {
            setIsAsking(false);
        }
    };

    const getIndexingMessage = () => {
        switch (indexingStatus) {
            case 'indexing': return indexingDetail || 'Cloning and analyzing codebase...';
            case 'completed': return 'Codebase analysis complete. You can now ask questions.';
            case 'failed': return `Error: ${indexingDetail || 'Codebase analysis failed.'}`;
            case 'empty': return 'Analysis complete, but no indexable code files were found.';
            default: return 'Initializing code analysis...';
        }
    };

    return (
        <div className="bg-gray-900 text-white min-h-screen font-sans">
            <div className="container mx-auto p-4 md:p-8">
                <header className="flex justify-between items-center mb-8">
                    <h1 className="text-2xl md:text-3xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-purple-400 to-indigo-600">GitHub Intelligence Assistant</h1>
                    <div>
                        {user ? (
                            <div className="flex items-center space-x-4">
                                <img src={user.avatar_url} alt={user.name} className="w-10 h-10 rounded-full border-2 border-indigo-500"/>
                                <button onClick={handleLogout} className="flex items-center space-x-2 bg-gray-700 hover:bg-gray-600 text-white font-semibold py-2 px-4 rounded-lg transition-colors">
                                    <LogoutIcon /><span>Logout</span>
                                </button>
                            </div>
                        ) : (
                            <button onClick={handleLogin} className="bg-indigo-600 hover:bg-indigo-700 text-white font-bold py-2 px-4 rounded-lg flex items-center transition-colors">
                                <GithubIcon /><span className="ml-2">Login with GitHub</span>
                            </button>
                        )}
                    </div>
                </header>
                <div className="text-center mb-12">
                     <p className="text-gray-400 mt-4 max-w-2xl mx-auto">
                        {user ? `Welcome, ${user.name}! Analyze one of your private repositories or any public one.` : "Login to analyze private repositories, or paste a public URL to get started."}
                    </p>
                </div>
                <div className="max-w-2xl mx-auto">
                    <div className="flex items-center bg-gray-800 border border-gray-700 rounded-lg p-2 shadow-lg">
                        <span className="pl-2 text-gray-500"><GithubIcon /></span>
                        <input type="text" value={repoUrl} onChange={(e) => setRepoUrl(e.target.value)}
                            placeholder={user ? "e.g., https://github.com/your-name/your-private-repo" : "e.g., https://github.com/langchain-ai/langchain"}
                            className="w-full p-2 bg-transparent text-white focus:outline-none"
                            onKeyPress={(e) => e.key === 'Enter' && handleAnalyze()}
                        />
                        <button onClick={handleAnalyze} disabled={isLoading} className="bg-indigo-600 hover:bg-indigo-700 disabled:bg-indigo-900 disabled:cursor-not-allowed text-white font-bold py-2 px-4 rounded-md flex items-center transition-colors duration-300">
                            {isLoading ? <LoadingSpinner/> : <SearchIcon />}<span className="ml-2">{isLoading ? "Analyzing..." : "Analyze"}</span>
                        </button>
                    </div>
                </div>

                {error.type === 'connection' ? (
                    <ConnectionError apiUrl={API_BASE_URL} />
                ) : (
                    <>
                        {error.type === 'generic' && <p className="text-red-400 text-center mt-4">{error.message}</p>}
                        
                        {analysis && (
                            <div className="mt-12 space-y-8">
                                <div className="bg-gray-800/50 border border-gray-700 rounded-lg p-6">
                                    <h2 className="text-3xl font-bold text-white mb-2">{analysis.projectName}</h2>
                                    <p className="text-gray-400">{analysis.description}</p>
                                </div>
                                
                                <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
                                    <div className="lg:col-span-2 bg-gray-800/50 border border-gray-700 rounded-lg p-6">
                                        <h3 className="text-xl font-bold text-white mb-4">README.md</h3>
                                    <div className="prose prose-invert prose-sm md:prose-base max-w-none text-gray-300 h-96 overflow-auto" dangerouslySetInnerHTML={{ __html: marked(analysis.readmeContent) }} />
                                    </div>
                                    <div className="lg:col-span-1 space-y-8">
                                        <div className="bg-gray-800/50 border border-gray-700 rounded-lg p-6">
                                            <h3 className="text-xl font-bold text-white mb-4">Recent Commits</h3>
                                            <ul className="space-y-3">{analysis.recentCommits.map(c => <li key={c.sha} className="text-sm"><p className="font-mono text-indigo-400 truncate" title={c.message}>{c.message}</p><p className="text-gray-500">by {c.author}</p></li>)}</ul>
                                        </div>
                                        <div className="bg-gray-800/50 border border-gray-700 rounded-lg p-6">
                                            <h3 className="text-xl font-bold text-white mb-4">Open Issues</h3>
                                            <ul className="space-y-3">{analysis.openIssues.length > 0 ? analysis.openIssues.map(i => <li key={i.number}><a href={i.url} target="_blank" rel="noopener noreferrer" className="text-sm text-gray-300 hover:text-indigo-400 transition-colors">#{i.number}: {i.title}</a></li>) : <p className="text-sm text-gray-500">No open issues found.</p>}</ul>
                                        </div>
                                    </div>
                                </div>
                                
                                <div className="bg-gray-800/50 border border-gray-700 rounded-lg p-6">
                                    <h2 className="text-2xl font-bold text-white mb-4 flex items-center"><SparklesIcon /><span className="ml-2">Code-Aware Q&A</span></h2>
                                    <div className={`p-3 rounded-md mb-4 flex items-center text-sm ${indexingStatus === 'completed' ? 'bg-green-900/50' : indexingStatus === 'failed' ? 'bg-red-900/50' : 'bg-yellow-900/50'}`}>
                                        {indexingStatus === 'indexing' && <LoadingSpinner />}<p className="ml-3 text-gray-300">{getIndexingMessage()}</p>
                                    </div>
                                    <div className="flex items-center space-x-2">
                                        <input type="text" value={question} onChange={(e) => setQuestion(e.target.value)}
                                            placeholder={indexingStatus === 'completed' ? "Ask about the code..." : "Waiting for analysis to complete..."}
                                            className="w-full p-2 bg-gray-700 border border-gray-600 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-indigo-500 disabled:opacity-50"
                                            onKeyPress={(e) => e.key === 'Enter' && handleAskQuestion()}
                                            disabled={indexingStatus !== 'completed' || isAsking}
                                        />
                                        <button onClick={handleAskQuestion} disabled={indexingStatus !== 'completed' || isAsking} className="bg-purple-600 hover:bg-purple-700 disabled:bg-purple-900/50 disabled:cursor-not-allowed text-white font-bold py-2 px-4 rounded-md flex items-center transition-colors">
                                            {isAsking ? <LoadingSpinner/> : "Ask AI"}
                                        </button>
                                    </div>
                                    {qnaError && <p className="text-red-400 mt-2">{qnaError}</p>}
                                    {isAsking && <p className="text-gray-400 mt-4">Searching codebase and generating answer...</p>}
                                    {answer && <div className="mt-4 p-4 bg-gray-900 rounded-md border border-gray-700"><p className="prose prose-invert max-w-none text-gray-300 whitespace-pre-wrap">{answer}</p></div>}
                                </div>
                            </div>
                        )}
                    </>
                )}
            </div>
        </div>
    );
}
