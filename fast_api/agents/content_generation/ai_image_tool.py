from langchain.tools import tool
from typing import Dict, Any
import json
import replicate
import requests
from pathlib import Path


STATIC_ROOT = Path(__file__).resolve().parents[2] / "generated_content"
STATIC_ROOT.mkdir(parents=True, exist_ok=True)
# Directory for AI generated imagery served via FastAPI static mount
AI_IMAGE_DIR = STATIC_ROOT / "ai_images"
AI_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
print(f"🔧 AI_IMAGE_DIR set to: {AI_IMAGE_DIR.absolute()}")

def _to_public_path(path: Path) -> str:
    """Convert absolute asset path to FastAPI public URL path."""
    try:
        relative = path.resolve().relative_to(STATIC_ROOT)
        return f"/generated_content/{relative.as_posix()}"
    except Exception:
        return str(path)

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
        
        success_msg = f"✅ Generated AI image: {_to_public_path(file_path)}\nPrompt: {data_dict['prompt']}"
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