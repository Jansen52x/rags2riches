from langchain.tools import tool
from typing import Dict, Any
import json
import replicate
import requests
from pathlib import Path

# Use relative import within package
from content_generator import ContentGenerator


STATIC_ROOT = Path(__file__).resolve().parents[2] / "generated_content"
STATIC_ROOT.mkdir(parents=True, exist_ok=True)


def _to_public_path(path: Path) -> str:
    """Convert absolute asset path to FastAPI public URL path."""
    try:
        relative = path.resolve().relative_to(STATIC_ROOT)
        return f"/generated_content/{relative.as_posix()}"
    except Exception:
        return str(path)


# Initialize generator rooted at static directory
generator = ContentGenerator(output_dir=STATIC_ROOT)

# Directory for AI generated imagery served via FastAPI static mount
AI_IMAGE_DIR = STATIC_ROOT / "ai_images"
AI_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
print(f"🔧 AI_IMAGE_DIR set to: {AI_IMAGE_DIR.absolute()}")

@tool
def generate_market_share_chart(data: str) -> str:
    """
    Generate a market share bar chart.
    
    Args:
        data: JSON string with structure:
        {
            "companies": ["Company A", "Company B", ...],
            "market_share": [30, 25, 20, ...],
            "title": "Market Share Analysis",
            "client_name": "Client Co"
        }
    
    Returns:
        Path to generated image file
    """
    try:
        data_dict = json.loads(data)
        
        spec = {
            "type": "market_share",
            "title": data_dict.get("title", "Market Share Analysis"),
            "client_name": data_dict.get("client_name", "client"),
            "data": {
                "companies": data_dict["companies"],
                "market_share": data_dict["market_share"]
            }
        }

        file_path = Path(generator.generate(spec))
        return f"✅ Generated market share chart: {_to_public_path(file_path)}"
        
    except Exception as e:
        return f"❌ Error generating market share chart: {str(e)}"


@tool
def generate_growth_trend_chart(data: str) -> str:
    """
    Generate a line chart showing growth trends over time.
    
    Args:
        data: JSON string with structure:
        {
            "years": [2021, 2022, 2023, ...],
            "entities": [
                {"name": "Client", "values": [100, 120, 145, ...]},
                {"name": "Industry Avg", "values": [100, 115, 130, ...]}
            ],
            "title": "Growth Trends",
            "client_name": "Client Co"
        }
    
    Returns:
        Path to generated image file
    """
    try:
        data_dict = json.loads(data)
        
        spec = {
            "type": "growth_trend",
            "title": data_dict.get("title", "Growth Trend Analysis"),
            "client_name": data_dict.get("client_name", "client"),
            "data": {
                "years": data_dict["years"],
                "entities": data_dict["entities"],
                "y_axis_label": data_dict.get("y_axis_label", "Revenue ($M)")
            }
        }

        file_path = Path(generator.generate(spec))
        return f"✅ Generated growth trend chart: {_to_public_path(file_path)}"
        
    except Exception as e:
        return f"❌ Error generating growth trend chart: {str(e)}"


@tool
def generate_competitive_matrix(data: str) -> str:
    """
    Generate a 2x2 competitive positioning matrix.
    
    Args:
        data: JSON string with structure:
        {
            "competitors": [
                {"name": "Company A", "x": 7, "y": 8, "color": "#2E86AB"},
                {"name": "Company B", "x": 5, "y": 6, "color": "#F18F01"}
            ],
            "x_axis_label": "Capability",
            "y_axis_label": "Strategic Value",
            "title": "Competitive Positioning",
            "client_name": "Client Co"
        }
    
    Returns:
        Path to generated image file
    """
    try:
        data_dict = json.loads(data)
        
        spec = {
            "type": "competitive_matrix",
            "title": data_dict.get("title", "Competitive Positioning Matrix"),
            "client_name": data_dict.get("client_name", "client"),
            "data": {
                "competitors": data_dict["competitors"],
                "x_axis_label": data_dict.get("x_axis_label", "Market Capability"),
                "y_axis_label": data_dict.get("y_axis_label", "Strategic Value")
            }
        }

        file_path = Path(generator.generate(spec))
        return f"✅ Generated competitive matrix: {_to_public_path(file_path)}"
        
    except Exception as e:
        return f"❌ Error generating competitive matrix: {str(e)}"


@tool
def generate_swot_analysis(data: str) -> str:
    """
    Generate a SWOT analysis visualization.
    
    Args:
        data: JSON string with structure:
        {
            "strengths": ["strength1", "strength2", ...],
            "weaknesses": ["weakness1", "weakness2", ...],
            "opportunities": ["opportunity1", "opportunity2", ...],
            "threats": ["threat1", "threat2", ...],
            "company_name": "Company Name",
            "client_name": "Client Co"
        }
    
    Returns:
        Path to generated image file
    """
    try:
        data_dict = json.loads(data)
        
        spec = {
            "type": "swot_analysis",
            "title": f"SWOT Analysis: {data_dict.get('company_name', 'Client')}",
            "client_name": data_dict.get("client_name", "client"),
            "company_name": data_dict.get("company_name", "Client"),
            "data": {
                "strengths": data_dict.get("strengths", []),
                "weaknesses": data_dict.get("weaknesses", []),
                "opportunities": data_dict.get("opportunities", []),
                "threats": data_dict.get("threats", [])
            }
        }

        file_path = Path(generator.generate(spec))
        return f"✅ Generated SWOT analysis: {_to_public_path(file_path)}"
        
    except Exception as e:
        return f"❌ Error generating SWOT analysis: {str(e)}"


@tool
def generate_financial_comparison(data: str) -> str:
    """
    Generate a grouped bar chart comparing financial metrics.
    
    Args:
        data: JSON string with structure:
        {
            "metrics": ["Revenue", "Profit", "Growth"],
            "entities": [
                {"name": "Client", "values": [100, 15, 20]},
                {"name": "Competitor", "values": [120, 18, 15]}
            ],
            "title": "Financial Comparison",
            "client_name": "Client Co"
        }
    
    Returns:
        Path to generated image file
    """
    try:
        data_dict = json.loads(data)
        
        spec = {
            "type": "financial_comparison",
            "title": data_dict.get("title", "Financial Metrics Comparison"),
            "client_name": data_dict.get("client_name", "client"),
            "data": {
                "metrics": data_dict["metrics"],
                "entities": data_dict["entities"]
            }
        }

        file_path = Path(generator.generate(spec))
        return f"✅ Generated financial comparison: {_to_public_path(file_path)}"
        
    except Exception as e:
        return f"❌ Error generating financial comparison: {str(e)}"


@tool
def generate_animated_video(data: str) -> str:
    """
    Generate an ANIMATED video presentation with dynamic chart animations.
    This creates a video with animated charts (bars growing, lines drawing, etc.)
    rather than just static images in a slideshow.
    
    Args:
        data: JSON string with structure:
        {
            "title": "Presentation Title",
            "client_name": "Client Co",
            "sections": [
                {
                    "type": "market_share",
                    "title": "Market Share Analysis",
                    "data": {"companies": [...], "market_share": [...]}
                },
                {
                    "type": "swot_analysis",
                    "title": "SWOT Analysis",
                    "data": {"strengths": [...], "weaknesses": [...], ...}
                },
                ...
            ],
            "duration_per_section": 5
        }
    
    Returns:
        Path to generated animated MP4 video file
    """
    try:
        data_dict = json.loads(data)
        
        spec = {
            "type": "animated_video",
            "title": data_dict.get("title", "Sales Presentation"),
            "client_name": data_dict.get("client_name", "client"),
            "sections": data_dict.get("sections", []),
            "duration_per_section": data_dict.get("duration_per_section", 5)
        }

        file_path = Path(generator.generate(spec))
        return f"✅ Generated animated video presentation: {_to_public_path(file_path)}"
        
    except Exception as e:
        return f"❌ Error generating animated video: {str(e)}"


# List of all available tools
content_generation_tools = [
    generate_market_share_chart,
    generate_growth_trend_chart,
    generate_competitive_matrix,
    generate_swot_analysis,
    generate_financial_comparison,
    generate_animated_video,
]