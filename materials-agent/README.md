# Meeting Materials Decision Making Agent

This agent helps salespeople decide what presentation materials to create based on fact-checked claims, then coordinates with image/video generation agents to produce the materials.

## 🎯 Overview

The Materials Decision Agent takes verified claims from the fact-checker and intelligently recommends the best presentation materials to create for a client meeting. It considers:

- **Claim types and evidence strength**
- **Client context and preferences**
- **Time constraints**
- **Material effectiveness for different data types**
- **Professional presentation standards**

## 🏗️ Architecture

```
Fact-Checker Output → Integration Bridge → Materials Decision Agent → Generation Queue
                                              ↓
                                         Streamlit UI
                                              ↓
                                    Image/Video Agents
```

## 📁 Files Structure

```
materials-agent/
├── materials_decision_agent.py  # Core LangGraph workflow
├── integration_bridge.py        # Connects with fact-checker
├── materials_ui.py              # Streamlit web interface
├── simple_demo.py               # No-AI fallback logic used by the UI when deps are missing
└── README.md                    # This file
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd /Users/[replace_with_user]/rags2riches
pip install -r requirements.txt
pip install streamlit  # For the UI
```

### 2. Launch the UI

```bash
streamlit run materials_ui.py
```

Then open <http://localhost:8501> in your browser.

## 🔧 Core Components

### Materials Decision Agent (`materials_decision_agent.py`)

A LangGraph workflow that:

1. **Analyzes Claims** - Determines what materials work best for each claim type
2. **Prioritizes Materials** - Ranks recommendations by impact and time constraints
3. **Creates Generation Queue** - Prepares detailed instructions for creation agents
4. **Saves Decisions** - Persists choices to database

**Supported Material Types:**

- 📊 **Slides** - Single impactful presentation slides
- 📈 **Charts** - Data visualizations and graphs
- 🎨 **Infographics** - Visual summaries with statistics
- 🎬 **Video Explainers** - Short animated explanations
- 📱 **Social Media Posts** - Shareable content
- 📋 **Presentation Decks** - Complete slide sets

### Integration Bridge (`integration_bridge.py`)

Handles the handoff between fact-checker and materials agent:

- Extracts verified claims from fact-checker output
- Calculates confidence scores based on evidence quality
- Formats data for materials agent consumption
- Provides end-to-end pipeline orchestration

### Streamlit UI (`materials_ui.py`)

Professional web interface for salespeople:

- **Input Management** - Load verified claims from fact-checker
- **Visual Recommendations** - See material suggestions with details
- **Approval Workflow** - Approve, reject, or modify recommendations
- **Session Management** - Save and restore work sessions
- **Generation Tracking** - Monitor material creation progress

## 📊 Example Usage

### From Python Code

```python
from materials_decision_agent import MaterialsDecisionState, create_materials_decision_workflow

# Sample verified claims
verified_claims = [
    {
        "claim_id": "claim_001",
        "claim": "Singapore's e-commerce market reached SGD 9 billion in 2023",
        "verdict": "TRUE",
        "confidence": 0.92,
        "evidence": [
            {
                "source": "Singapore Department of Statistics",
                "summary": "Official data confirms SGD 9.1B e-commerce sales"
            }
        ]
    }
]

# Create workflow state
initial_state = MaterialsDecisionState(
    session_id="session_001",
    salesperson_id="SP12345",
    client_context="Small e-commerce startup in Singapore",
    verified_claims=verified_claims,
    material_recommendations=[],
    selected_materials=[],
    generation_queue=[],
  generated_files=[],
  generation_status=None,
  decision_complete=False,
  user_feedback=None
)

# Run workflow
workflow = create_materials_decision_workflow()
final_state = workflow.invoke(initial_state)

# Access results
recommendations = final_state['material_recommendations']
selected_materials = final_state['selected_materials']
```

### Integration with Fact-Checker

```python
from integration_bridge import run_complete_pipeline

# Fact-checker output
fact_check_state = {
    "salesperson_id": "SP12345",
    "client_context": "E-commerce startup",
    "claim_verdicts": [
        {
            "overall_verdict": "TRUE",
            "explanation": "Verified by official sources",
            "main_evidence": [...],
            "pass_to_materials_agent": True
        }
    ]
}

# Run complete pipeline
result = run_complete_pipeline(fact_check_state)
print(f"Generated {result['recommendations_count']} material recommendations")
```

## 🎨 Material Recommendation Logic

The agent uses AI reasoning to recommend materials based on:

### Claim Analysis

- **Statistical claims** → Charts and infographics
- **Market trends** → Presentation decks with trend analysis
- **Product comparisons** → Side-by-side infographics
- **Case studies** → Video explainers or detailed slides

### Client Context

- **Startup clients** → Modern, data-heavy materials
- **Enterprise clients** → Professional, conservative designs
- **Technical audience** → Detailed charts and specifications
- **Executive audience** → High-level summaries and key metrics

### Time Constraints

- **High priority** materials get created first
- **Total time estimation** to fit available preparation time
- **Quick wins** (low time, high impact) prioritized

## 🔄 Integration Points

### Input: From Fact-Checker Agent

```json
{
  "claim_verdicts": [
    {
      "overall_verdict": "TRUE|FALSE|CANNOT BE DETERMINED",
      "explanation": "Reasoning for verdict",
      "main_evidence": [{ "source": "...", "summary": "..." }],
      "pass_to_materials_agent": true
    }
  ]
}
```

### Output: To Image/Video Generation Agents

```json
{
  "generation_queue": [
    {
      "task_id": "uuid",
      "type": "slide|infographic|chart|video_explainer",
      "title": "Material title",
      "content_requirements": {
        "style": "professional",
        "data_visualization": "required",
        "color_scheme": "corporate"
      },
      "instructions": "Detailed creation instructions..."
    }
  ]
}
```

## 🗄️ Database Schema

The agent creates and uses this database table:

```sql
CREATE TABLE materials_decisions (
    decision_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id TEXT NOT NULL,
    salesperson_id TEXT NOT NULL,
    recommendations JSONB NOT NULL,
    selected_materials JSONB NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
```

## 🧪 Testing

- Use the Streamlit UI to load sample claims and generate materials.
- Or use the “From Python Code” example above for a programmatic run.
