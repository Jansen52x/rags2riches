from langchain.tools import tool
from typing import Dict, Any
import json
import replicate
import requests
from pathlib import Path
# Use relative import within package
from .content_generator import ContentGenerator

# Initialize generator
generator = ContentGenerator()

# Create directory for AI-generated images using absolute path
# Get the directory where this file (content_tools.py) is located
TOOLS_DIR = Path(__file__).parent
AI_IMAGE_DIR = TOOLS_DIR / "generated_content" / "ai_images"
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
        
        file_path = generator.generate(spec)
        return f"✅ Generated market share chart: {file_path}"
        
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
        
        file_path = generator.generate(spec)
        return f"✅ Generated growth trend chart: {file_path}"
        
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
        
        file_path = generator.generate(spec)
        return f"✅ Generated competitive matrix: {file_path}"
        
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
        
        file_path = generator.generate(spec)
        return f"✅ Generated SWOT analysis: {file_path}"
        
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
        
        file_path = generator.generate(spec)
        return f"✅ Generated financial comparison: {file_path}"
        
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
        
        file_path = generator.generate(spec)
        return f"✅ Generated animated video presentation: {file_path}"
        
    except Exception as e:
        return f"❌ Error generating animated video: {str(e)}"


@tool
def generate_ai_image(data: str) -> str:
    """
    Generate a custom AI image using Google's Imagen-4 via Replicate API.
    This tool creates photorealistic or artistic images from text descriptions.
    Use this for custom illustrations, marketing visuals, concept art, or any creative imagery.
    
    Args:
        data: JSON string with structure:
        {
            "prompt": "Detailed description of the image to generate",
            "aspect_ratio": "16:9" (optional, options: "1:1", "16:9", "9:16", "4:3", "3:4", default: "16:9"),
            "safety_filter_level": "block_medium_and_above" (optional, default: "block_medium_and_above"),
            "filename": "custom_name" (optional, will auto-generate if not provided)
        }
    
    Returns:
        Path to generated image file or error message
        
    Example prompts:
    - "modern office building with glass facade, architectural visualization, high quality"
    - "professional business person in a suit, corporate photography style, confident"
    - "abstract data visualization with blue and purple colors, modern design, geometric"
    - "team of diverse professionals collaborating around a table, natural lighting"
    """
    try:
        data_dict = json.loads(data)
        
        # Validate required field
        if "prompt" not in data_dict:
            error_msg = "❌ Error: 'prompt' field is required"
            print(error_msg)
            return error_msg
        
        # Prepare input for Replicate
        replicate_input = {
            "prompt": data_dict["prompt"],
            "aspect_ratio": data_dict.get("aspect_ratio", "16:9"),
            "safety_filter_level": data_dict.get("safety_filter_level", "block_medium_and_above")
        }
        
        print(f"🎨 Generating AI image with prompt: '{data_dict['prompt'][:60]}...'")
        print(f"   Input: {replicate_input}")
        
        # Run Replicate model
        try:
            output = replicate.run(
                "google/imagen-4",
                input=replicate_input
            )
            print(f"   ✓ Replicate API call successful")
        except Exception as replicate_error:
            error_msg = f"❌ Replicate API error: {str(replicate_error)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            return error_msg
        
        # Handle both single URL and list of URLs
        # Replicate returns a FileOutput object with a .url attribute
        if isinstance(output, list):
            image_url = output[0]
        else:
            image_url = output
        
        # Extract URL from FileOutput object if needed
        if hasattr(image_url, 'url'):
            print(f"   ✓ Output is FileOutput object, extracting URL...")
            image_url = image_url.url
        
        print(f"   ✓ Image URL received: {str(image_url)[:80]}...")
        
        # Generate filename
        if "filename" in data_dict:
            filename = f"{data_dict['filename']}.png"
        else:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"ai_image_{timestamp}.png"
        
        print(f"   → Filename: {filename}")
        
        # Full path
        file_path = AI_IMAGE_DIR / filename
        print(f"   → Full path: {file_path}")
        
        # Download the image
        try:
            print(f"   → Downloading image...")
            response = requests.get(image_url, timeout=30)
            response.raise_for_status()
            print(f"   ✓ Download successful ({len(response.content)} bytes)")
        except Exception as download_error:
            error_msg = f"❌ Error downloading image: {str(download_error)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            return error_msg
        
        # Save the image
        try:
            print(f"   → Saving to disk...")
            with open(file_path, "wb") as f:
                f.write(response.content)
            print(f"   ✓ Saved to: {file_path}")
        except Exception as save_error:
            error_msg = f"❌ Error saving image: {str(save_error)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            return error_msg
        
        success_msg = f"✅ Generated AI image: {file_path}\nPrompt: {data_dict['prompt']}"
        print(success_msg)
        return success_msg
        
    except json.JSONDecodeError as e:
        error_msg = f"❌ Invalid JSON format: {str(e)}"
        print(error_msg)
        return error_msg
    except Exception as e:
        error_msg = f"❌ Unexpected error in generate_ai_image: {str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        return error_msg


# List of all available tools
content_generation_tools = [
    generate_market_share_chart,
    generate_growth_trend_chart,
    generate_competitive_matrix,
    generate_swot_analysis,
    generate_financial_comparison,
    generate_animated_video,
    generate_ai_image
]