<div align="center">

# 🤖 **GitChat AI**

### *Your GitHub Project Intelligence Assistant*

**A full-stack AI tool to analyze GitHub repositories, generate READMEs, and answer technical questions with Groq-powered insights.**

</div>

---

## 🎥 GitChat AI Demo

<a href="https://drive.google.com/file/d/1HlOD2-X7cDsjLv1VyEFyynIg58RZynNQ/view?usp=drive_link" target="_blank">
<img src="https://drive.google.com/uc?export=view&id=1HlOD2-X7cDsjLv1VyEFyynIg58RZynNQ" alt="GitChat AI Demo" width="920"/>
</a>

<p align="center">
  <a href="https://drive.google.com/file/d/1HlOD2-X7cDsjLv1VyEFyynIg58RZynNQ/view?usp=drive_link" target="_blank">
    ▶️ <strong>Watch the Full Demo</strong>
  </a>
</p>

---

<div align="center">

🛠️ Built With  
<p align="center">
<img alt="Python" src="https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white" />
<img alt="FastAPI" src="https://img.shields.io/badge/FastAPI-Backend-009688?logo=fastapi&logoColor=white" />
<img alt="React" src="https://img.shields.io/badge/React-Frontend-61DAFB?logo=react&logoColor=black" />
<img alt="Groq" src="https://img.shields.io/badge/Groq-LLM_API-E91E63?logo=openai&logoColor=white" />
<img alt="LangChain" src="https://img.shields.io/badge/LangChain-Framework-FFB300?logo=python&logoColor=black" />
<img alt="Redis" src="https://img.shields.io/badge/Redis-Rate_Limiting-DC382D?logo=redis&logoColor=white" />
<img alt="FAISS" src="https://img.shields.io/badge/FAISS-Vector_Search-2B2D42?logo=facebook&logoColor=white" />
<img alt="Sentence Transformers" src="https://img.shields.io/badge/SentenceTransformers-Embeddings-764ABC?logo=python&logoColor=white" />
</p>
</div>

---

## 🚀 Features

| 🧩 Feature                     | 📌 Description                                                                 |
|------------------------------|------------------------------------------------------------------------------|
| 🔍 **Repo Analysis**          | Clone & analyze any public/private GitHub repo with Groq + LangChain         |
| 📚 **README Generator**       | Autogenerates professional README.md using repo structure + AI               |
| 💬 **Code Q&A**               | Ask contextual questions and get answers using local vector search + Groq    |
| 🚦 **Rate Limiting**          | Robust Redis-based & in-memory fallback rate limiter                         |
| 🧠 **AI-Powered Summaries**   | Uses Llama-3 via Groq API for natural language insights                       |
| 🔐 **GitHub OAuth**           | Secure login and repo access for private repositories                        |
| 🌐 **Modern Frontend**        | React-based UI with Tailwind for sleek and responsive interactions           |

---

## 🛠️ Technology Stack

```text
Frontend         : React + TailwindCSS  
Backend          : FastAPI  
AI & Embeddings  : Groq API + SentenceTransformers  
LangChain        : LangChain, FAISS, HuggingFaceEmbeddings  
Rate Limiting    : Redis + In-memory fallback  
OAuth            : GitHub OAuth2  
```

---

## 📋 Prerequisites

* Python 3.10+
* Node.js 16+
* Redis (Optional but recommended)
* Groq API Key (Free from [console.groq.com](https://console.groq.com))
* GitHub OAuth App (Client ID + Secret)

---

## 🔧 Backend Installation

```bash
# 1. Clone this repo
git clone https://github.com/your-username/gitchat-ai.git
cd gitchat-ai

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install backend dependencies
pip install -r requirements.txt

# 4. Create .env file and add your keys
cp .env.example .env
# Fill in GROQ_API_KEY, GITHUB_CLIENT_ID, SECRET_KEY, etc.

# 5. Start FastAPI server
uvicorn main:app --reload
```

---

## 🌐 Frontend Installation

```bash
# In another terminal
cd frontend
npm install
npm run dev
```

---

## 🧠 API Key Setup

Edit your `.env` file in the root directory:

```bash
GROQ_API_KEY=your_groq_api_key
GITHUB_CLIENT_ID=your_github_client_id
GITHUB_CLIENT_SECRET=your_github_client_secret
FRONTEND_URL=http://localhost:3000
SECRET_KEY=very_long_secure_key_here
```

---

## ▶️ Run the App

1. Start FastAPI server:  
   ```bash
   uvicorn main:app --reload
   ```

2. Run React app:  
   ```bash
   cd frontend && npm run dev
   ```

3. Visit: [http://localhost:3000](http://localhost:3000)

---

## 🗂️ Project Structure

```bash
gitchat-ai/
├── main.py              # FastAPI backend with Groq + LangChain logic
├── .env                 # Environment variables
├── requirements.txt     # Backend dependencies
├── frontend/
│   ├── App.js           # React frontend (Login, Analyze, README, Q&A)
│   └── ...
```

---

## 🛡️ Security Highlights

✔️ OAuth2 GitHub Login for secure repo access  
✔️ Tokens securely encrypted via `jose` and `itsdangerous`  
✔️ Rate limiting via Redis fallback to in-memory store  
✔️ No external storage of user data or code  

---

## 🔮 Roadmap

* [ ] 🧪 Codebase Testing Coverage  
* [ ] 📄 Export README as PDF  
* [ ] 🧠 RAG with multiple repos  
* [ ] 📊 Dashboard for developer analytics  
* [ ] 📦 Dockerize full stack  

---

## 📄 License

Licensed under the [MIT License](LICENSE)

---

## 🙏 Acknowledgments

* [FastAPI](https://fastapi.tiangolo.com/)  
* [Groq API](https://console.groq.com/)  
* [LangChain](https://www.langchain.com/)  
* [React](https://react.dev)  
* [Tailwind CSS](https://tailwindcss.com)  
* [Redis](https://redis.io)

---

<div align="center">

⭐ Star this repo if you like it  
🐛 [Report a Bug](https://github.com/your-username/gitchat-ai/issues)  
📢 [Request a Feature](https://github.com/your-username/gitchat-ai/issues)

</div>
