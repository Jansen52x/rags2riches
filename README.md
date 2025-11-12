# Rags2Riches: AI-Powered Sales Presentation Assistant

A comprehensive AI system that helps salespeople fact-check claims, generate evidence-backed materials, and create professional presentation content for client meetings.

## 🎯 Project Overview

Rags2Riches is an intelligent sales support platform that combines:
- **Fact-checking** of sales claims using multiple web sources
- **Intelligent material recommendations** based on verified claims
- **Automated content generation** (charts, infographics, AI images, animated videos)
- **RAG (Retrieval-Augmented Generation)** for document-based knowledge retrieval

The system uses **LangGraph** to orchestrate multiple AI agents that work together to transform raw claims into professional, verified presentation materials.

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit Frontend                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │ Fact Checker │  │ RAG Query    │  │ Materials    │       │
│  │   Page       │  │   Page       │  │ Decision     │       │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘       │
└─────────┼─────────────────┼─────────────────┼───────────────┘
          │                 │                 │
          ▼                 ▼                 ▼
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI Backend                          │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  /check-claim → Fact Checker Agent                   │   │
│  │  /query_rag → RAG Service                            │   │
│  │  /generate-materials → Materials + Content Agents    │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────┬──────────────────┬──────────────────┬─────────────┘
          │                  │                  │
          ▼                  ▼                  ▼
┌─────────────────────────────────────────────────────────────┐
│                    AI Agents (LangGraph)                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │ Fact Checker │  │ Materials    │  │ Content      │       │
│  │   Agent      │  │ Decision     │  │ Generation   │       │
│  │              │  │   Agent      │  │   Agent      │       │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘       │
└─────────┼─────────────────┼─────────────────┼───────────────┘
          │                 │                 │
          ▼                 ▼                 ▼
┌─────────────────────────────────────────────────────────────┐
│              External Services & Databases                  │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐     │
│  │PostgreSQL│  │ ChromaDB │  │  Web     │  │  LLMs    │     │
│  │ (Claims) │  │ (Vectors)│  │  Search  │  │ (Claude, │     │
│  │          │  │          │  │  Tools   │  │ Gemini)  │     │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘     │
└─────────────────────────────────────────────────────────────┘
```

## 🔄 Workflow

### 1. Fact-Checking Flow

```
User Input (Claim) 
    ↓
Fact Checker Agent
    ├─→ Analyze claim & determine search strategy
    ├─→ Search web (DuckDuckGo, Tavily, Wikipedia, NewsAPI)
    ├─→ Process evidence & make verdict (TRUE/FALSE/CANNOT BE DETERMINED)
    └─→ Save to PostgreSQL
    ↓
Verified Claims (only TRUE claims pass through)
```

### 2. Materials Decision Flow

```
Verified Claims
    ↓
Materials Decision Agent
    ├─→ Analyze claims & recommend materials
    ├─→ Prioritize by impact & time constraints
    ├─→ Create generation queue
    └─→ Save decision to database
    ↓
Generation Queue (chart specs, video specs, AI image prompts)
```

### 3. Content Generation Flow

```
Generation Queue
    ↓
Content Generation Agent
    ├─→ Generate AI images (if prompts provided)
    ├─→ Plan content generation
    ├─→ Generate charts (market share, SWOT, growth trends, etc.)
    ├─→ Generate animated videos
    └─→ Collect all generated files
    ↓
Generated Content (PNG charts, MP4 videos, AI images)
```

## 📦 Core Components

### 1. Fact Checker Agent (`fast_api/agents/fact_checker.py`)

**Purpose**: Verifies sales claims using multiple web sources

**Workflow**:
1. **Analyze**: Understands the claim and determines search strategy
2. **Search**: Uses multiple tools (DuckDuckGo, Tavily, Wikipedia, NewsAPI, RAG)
3. **Process**: Evaluates evidence and makes verdict
4. **Save**: Stores result in PostgreSQL

**Tools Used**:
- `duckduckgo_search_text`: Web search via DuckDuckGo
- `tavily_search`: AI-powered web search
- `search_wikipedia`: Wikipedia article summaries
- `get_news_articles`: Recent news articles
- `query_rag_system`: Internal document retrieval

**Output**: Verdict (TRUE/FALSE/CANNOT BE DETERMINED) with evidence

### 2. Materials Decision Agent (`fast_api/agents/materials_decision_agent.py`)

**Purpose**: Recommends presentation materials based on verified claims

**Workflow**:
1. **Analyze Claims**: Determines best materials for each claim type
2. **Prioritize**: Ranks materials by impact and time constraints
3. **Create Queue**: Builds generation queue with specifications
4. **Trigger Generation**: Calls content generation agent
5. **Save Decision**: Stores recommendations in database

**Material Types**:
- Charts (market share, growth trends, competitive matrix, SWOT)
- Infographics
- Slides
- Animated videos
- Social media posts
- Presentation decks

**Output**: Generation queue with detailed specifications

### 3. Content Generation Agent (`fast_api/agents/content_generation/`)

**Purpose**: Generates visual content (charts, images, videos)

**Workflow**:
1. **Generate AI Images**: Creates images from prompts (if provided)
2. **Planning**: Analyzes requirements and plans content generation
3. **Tools**: Generates charts and videos using specialized tools
4. **Finalize**: Collects all generated file paths

**Tools Available**:
- `generate_market_share_chart`: Market share visualization
- `generate_growth_trend_chart`: Growth over time
- `generate_competitive_matrix`: 2x2 strategic positioning
- `generate_swot_analysis`: SWOT analysis visualization
- `generate_financial_comparison`: Financial metrics comparison
- `generate_animated_video`: Animated presentation videos
- `generate_ai_image`: AI-generated images (via Replicate)

**Output**: List of generated file paths (PNG, MP4)

### 4. RAG Service (`fast_api/rag_services/`)

**Purpose**: Retrieval-Augmented Generation for document-based queries

**Components**:
- **EmbeddingService**: Converts text to vectors (NVIDIA NIM)
- **LLMService**: Language model for answer generation
- **RAGService**: Orchestrates retrieval and generation
- **QueryBuilder**: Advanced query construction with filters

**Features**:
- Semantic search in ChromaDB
- Re-ranking with cross-encoders (optional)
- Metadata filtering
- Source citation

## 🚀 Getting Started

### Prerequisites

- Docker and Docker Compose
- Python 3.12+
- API Keys (see `secrets.env` below)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd rags2riches
   ```

2. **Install and set up Ollama (Local LLM)**
   
   The project uses Ollama for local LLM inference. Install Ollama and pull the required models:
   
   ```bash
   # Install Ollama (if not already installed)
   # macOS/Linux: curl https://ollama.ai/install.sh | sh
   # Or download from https://ollama.ai/download
   
   # Pull required models
   ollama pull llama3.2:3b      # Used by fact checker agent
   ollama pull qwen3:0.6b       # Used by materials decision agent
   
   # Start Ollama server (runs on port 11434 by default)
   ollama serve
   ```
   
   **Note**: Keep the `ollama serve` command running in a separate terminal. The Docker containers connect to Ollama via `host.docker.internal:11434`.

3. **Set up environment variables**
   
   Create `secrets.env` in the project root:
   ```env
   # AI API Keys
   ANTHROPIC_API_KEY=your-anthropic-key
   GOOGLE_API_KEY=your-google-key
   REPLICATE_API_TOKEN=your-replicate-token
   NVIDIA_API_KEY=your-nvidia-key
   
   # Search APIs
   NEWS_API_KEY=your-newsapi-key
   TAVILY_API_KEY=your-tavily-key
   ```

4. **Start services with Docker Compose**
   ```bash
   docker-compose up -d
   ```

   This starts:
   - PostgreSQL (port 5432)
   - ChromaDB (port 8000)
   - FastAPI service (port 8001)
   - Streamlit UI (port 8501)
   - pgAdmin (port 5050)

5. **Initialize the database**
   ```bash
   # The db_init service runs automatically, or manually:
   python init_db.py
   ```

6. **Ingest documents (optional)**
   ```bash
   # Ingest synthetic data for RAG
   cd data
   python ingest_synthetic_data.py
   ```

### Access the Application

- **Streamlit UI**: http://localhost:8501
- **FastAPI Docs**: http://localhost:8001/docs
- **pgAdmin**: http://localhost:5050 (admin@admin.com / admin)

## 📁 Project Structure

```
rags2riches/
├── fast_api/                    # FastAPI backend
│   ├── agents/                  # LangGraph agents
│   │   ├── fact_checker.py     # Fact-checking agent
│   │   ├── materials_decision_agent.py  # Materials recommendation
│   │   └── content_generation/ # Content generation agent
│   │       ├── content_generation_agent.py
│   │       ├── content_tools.py
│   │       ├── animated_video_generator.py
│   │       └── ai_image_tool.py
│   ├── rag_services/          # RAG implementation
│   │   ├── rag_service.py
│   │   ├── embedding_service.py
│   │   ├── llm_service.py
│   │   └── query_builder.py
│   ├── main.py                 # FastAPI app
│   └── generated_content/      # Generated files (charts, videos, images)
│
├── streamlit/                  # Streamlit frontend
│   ├── 1_RAG.py               # RAG query interface
│   └── pages/
│       ├── 2_Fact_Checker.py  # Fact-checking UI
│       └── 3_Marketing_Decision.py  # Materials decision UI
│
├── data/                       # Data ingestion
│   ├── ingest_synthetic_data.py
│   ├── embedding_service.py
│   └── document_service.py
│
├── synthetic-data/             # Synthetic data generation
│   ├── generate_data.py
│   └── output/                # Generated PDFs and images
│
├── docker-compose.yml          # Docker services configuration
├── secrets.env                 # API keys (not in git)
└── requirements.txt            # Python dependencies
```

## 🔌 API Endpoints

### Fact Checking

**POST** `/check-claim`
- Verifies a sales claim
- Returns streaming progress updates
- **Request**:
  ```json
  {
    "claim": "Singapore's e-commerce market reached SGD 9 billion in 2023",
    "salesperson_id": "SP12345",
    "client_context": "E-commerce startup looking to expand"
  }
  ```

### RAG Queries

**POST** `/query_rag`
- Simple RAG query
- **Request**:
  ```json
  {
    "query": "What is the market size of e-commerce in Singapore?",
    "k": 5,
    "include_sources": true
  }
  ```

**POST** `/query_rag/builder`
- Advanced RAG query with filters
- **Request**:
  ```json
  {
    "query": "Market trends",
    "filters": {"sector": "Technology"},
    "k": 10,
    "score_threshold": 0.7
  }
  ```

### Materials Generation

**POST** `/generate-materials`
- Generates presentation materials from verified claims
- **Request**:
  ```json
  {
    "verified_claims": [
      {
        "claim_id": "claim_001",
        "claim": "Singapore's e-commerce market reached SGD 9 billion",
        "verdict": "TRUE",
        "confidence": 0.9
      }
    ],
    "salesperson_id": "SP12345",
    "client_context": "E-commerce startup",
    "user_prompt": "1. Modern office collaboration\n2. Digital transformation"
  }
  ```

## 🛠️ Technologies Used

### AI & ML
- **LangGraph**: Agent orchestration and workflow management
- **LangChain**: LLM integration and tooling
- **Claude (Anthropic)**: Primary LLM for content generation
- **Gemini (Google)**: Fact-checking LLM
- **Ollama**: Local LLM for materials decision and other operations
- **NVIDIA NIM**: Embeddings and LLM inference

### Databases
- **PostgreSQL**: Fact-checking results and materials decisions
- **ChromaDB**: Vector database for RAG

### Web & APIs
- **FastAPI**: Backend API framework
- **Streamlit**: Frontend UI
- **DuckDuckGo**: Web search
- **Tavily**: AI-powered search
- **NewsAPI**: News articles
- **Wikipedia API**: Encyclopedia content

### Content Generation
- **Replicate**: AI image generation
- **Matplotlib/Seaborn**: Chart generation
- **MoviePy**: Video generation
- **Pillow**: Image processing

### Infrastructure
- **Docker & Docker Compose**: Containerization
- **psycopg**: PostgreSQL driver
- **python-dotenv**: Environment management

## 🔍 Usage Examples

### 1. Fact-Check a Claim

```python
import requests

response = requests.post(
    "http://localhost:8001/check-claim",
    json={
        "claim": "Shopee leads Singapore e-commerce with 35% market share",
        "salesperson_id": "SP12345",
        "client_context": "E-commerce startup"
    },
    stream=True
)

for line in response.iter_lines():
    if line:
        data = json.loads(line)
        if data["type"] == "progress":
            print(f"Progress: {data['text']}")
        elif data["type"] == "result":
            print(f"Verdict: {data['final_verdict']['overall_verdict']}")
```

### 2. Query RAG System

```python
response = requests.post(
    "http://localhost:8001/query_rag",
    json={
        "query": "What are the key trends in Singapore e-commerce?",
        "k": 5
    }
)

result = response.json()
print(result["answer"])
print(f"Sources: {result['sources']}")
```

### 3. Generate Materials

```python
response = requests.post(
    "http://localhost:8001/generate-materials",
    json={
        "verified_claims": [
            {
                "claim_id": "claim_001",
                "claim": "Market size is SGD 9 billion",
                "verdict": "TRUE",
                "confidence": 0.9
            }
        ],
        "salesperson_id": "SP12345",
        "client_context": "E-commerce startup",
        "user_prompt": "Modern, professional style"
    }
)

result = response.json()
print(f"Generated files: {result['generated_files']}")
```

## 🧪 Testing

### Test Fact Checker
```bash
cd fact-checker-agent
python test_agent.py
```

### Test Materials Agent
```bash
cd materials-agent/evaluation
python test_agent.py
```

### Generate Workflow Diagrams
```bash
cd fast_api/agents
python -m workflow_visualizer all
```

This generates Mermaid diagrams for all agent workflows.

## 🔧 Configuration

### Environment Variables

Key variables in `secrets.env`:
- `ANTHROPIC_API_KEY`: Claude API access
- `GOOGLE_API_KEY`: Gemini API access
- `REPLICATE_API_TOKEN`: AI image generation
- `NVIDIA_API_KEY`: Embeddings and LLM
- `NEWS_API_KEY`: News article access
- `TAVILY_API_KEY`: Web search

### Docker Services

Edit `docker-compose.yml` to:
- Change ports
- Add/remove services
- Configure volumes
- Set environment variables

## 📊 Data Flow

1. **User submits claim** → Streamlit UI
2. **Fact-checking** → Multiple web sources → PostgreSQL
3. **Materials recommendation** → LLM analysis → Generation queue
4. **Content generation** → Charts/videos/images → `generated_content/`
5. **Files served** → FastAPI static mount → Streamlit display

## 🐛 Troubleshooting

### Common Issues

1. **Ollama Connection Errors**
   - Ensure Ollama is running: `ollama serve` (must be running before starting Docker)
   - Verify models are installed: `ollama list` (should show `llama3.2:3b` and `qwen3:0.6b`)
   - Check Ollama is accessible: `curl http://localhost:11434/api/tags`
   - On Docker, ensure `OLLAMA_HOST=http://host.docker.internal:11434` is set correctly
   - If using Linux, you may need to use `host.docker.internal` or the host's IP address

2. **API Key Errors**
   - Ensure `secrets.env` exists in project root
   - Check all required keys are set
   - Restart Docker containers after updating keys

3. **Database Connection Errors**
   - Ensure PostgreSQL container is running: `docker-compose ps`
   - Check connection string in environment variables

4. **ChromaDB Connection Errors**
   - Verify ChromaDB is running: `curl http://localhost:8000/api/v1/heartbeat`
   - Check `CHROMADB_HOST` and `CHROMADB_PORT` settings

5. **Import Errors**
   - Ensure all dependencies are installed: `pip install -r requirements.txt`
   - Check Python path and virtual environment

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📝 License

[Add your license information here]

## 🙏 Acknowledgments

- LangChain/LangGraph for agent framework
- ChromaDB for vector database
- All open-source contributors

---

**Note**: This is an active development project. API endpoints and workflows may change. Check the FastAPI docs at `/docs` for the latest API specification.

