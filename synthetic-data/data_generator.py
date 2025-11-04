"""
Data Generator Module
Handles synthetic data generation using Gemini AI or Faker library as fallback.
"""
from faker import Faker
import pandas as pd
import os
import time
import random

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    # Load from parent directory (rags2riches/.env)
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
    load_dotenv(env_path)
    print(f"✓ Loaded environment variables from: {env_path}")
except ImportError:
    print("⚠️  python-dotenv not installed. Using system environment variables.")
    print("   Install with: pip install python-dotenv")

# Try to import Gemini AI
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("⚠️  google-generativeai not installed. Using Faker fallback.")
    print("   Install with: pip install google-generativeai")

fake = Faker()

# Configuration - Load from environment variables
USE_GEMINI = os.environ.get('USE_GEMINI_AI', 'true').lower() == 'true'
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', None)

# Initialize Gemini if available and configured
gemini_model = None
if GEMINI_AVAILABLE and USE_GEMINI and GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        # Use the latest fast model (gemini-2.0-flash is fast and free)
        gemini_model = genai.GenerativeModel('gemini-2.0-flash')
        print("✓ Gemini AI initialized successfully (using gemini-2.0-flash)")
    except Exception as e:
        print(f"⚠️  Failed to initialize Gemini AI: {e}")
        print("   Falling back to Faker")
        gemini_model = None
elif USE_GEMINI and not GEMINI_API_KEY:
    print("⚠️  GEMINI_API_KEY not found in environment variables")
    print("   Set it with: $env:GEMINI_API_KEY='your-api-key-here' (PowerShell)")
    print("   Or: export GEMINI_API_KEY='your-api-key-here' (Linux/Mac)")
    print("   Falling back to Faker")


def generate_client_record_with_gemini():
    """
    Generates a synthetic client record using Gemini AI.
    Returns more realistic and coherent company data.
    """
    try:
        prompt = """Generate realistic company information in JSON format with these fields:
- company_name: A creative, professional company name
- industry: A specific industry/business sector (e.g., "cloud computing solutions", "sustainable energy")
- contact_person: A full name
- contact_email: A professional email address
- company_description: A 2-3 sentence description of what the company does

Make it realistic and varied. Return ONLY valid JSON as a single object, no markdown formatting."""
        
        response = gemini_model.generate_content(prompt)
        
        # Parse the JSON response
        import json
        # Remove markdown code blocks if present
        text = response.text.strip()
        if text.startswith('```'):
            text = text.split('```')[1]
            if text.startswith('json'):
                text = text[4:]
            text = text.strip()
        
        data = json.loads(text)
        
        # Ensure we have a dictionary, not a list
        if isinstance(data, list):
            if len(data) > 0:
                data = data[0]  # Take the first element if it's a list
            else:
                raise ValueError("Gemini returned an empty list")
        
        # Validate required fields
        required_fields = ['company_name', 'industry', 'contact_person', 'contact_email', 'company_description']
        if not all(field in data for field in required_fields):
            raise ValueError(f"Missing required fields. Got: {list(data.keys())}")
        
        # Add small delay to avoid rate limiting
        time.sleep(0.5)
        
        return data
        
    except Exception as e:
        print(f"⚠️  Gemini API error: {e}. Using Faker fallback for this record.")
        return generate_client_record_with_faker()


def generate_client_record_with_faker():
    """Generates a synthetic client record using Faker (fallback method)."""
    return {
        'company_name': fake.company(),
        'industry': fake.bs(),  # Using bs() for a business-sounding industry description
        'contact_person': fake.name(),
        'contact_email': fake.email(),
        'company_description': fake.catch_phrase() + ". " + fake.text(max_nb_chars=100)
    }


def generate_client_record():
    """
    Generates a synthetic client record.
    Uses Gemini AI if available, otherwise falls back to Faker.
    """
    if gemini_model is not None:
        return generate_client_record_with_gemini()
    else:
        return generate_client_record_with_faker()


def generate_industry_overview_with_gemini(industry):
    """
    Generates a realistic industry overview using Gemini AI.
    
    Args:
        industry (str): The industry name to generate overview for
        
    Returns:
        str: A synthetic industry overview text
    """
    try:
        prompt = f"""Write a brief 3-sentence industry overview for the {industry} sector.
Include current trends, key challenges, and future outlook.
Make it sound professional and realistic. Return only the text, no formatting."""
        
        response = gemini_model.generate_content(prompt)
        time.sleep(0.5)  # Rate limiting
        return response.text.strip()
        
    except Exception as e:
        print(f"⚠️  Gemini API error: {e}. Using Faker fallback for industry overview.")
        return generate_industry_overview_with_faker(industry)


def generate_industry_overview_with_faker(industry):
    """
    Generates a synthetic industry overview using Faker (fallback method).
    
    Args:
        industry (str): The industry name to generate overview for
        
    Returns:
        str: A synthetic industry overview text
    """
    overview = (
        f"The {industry} industry is currently experiencing {fake.word()} trends. "
        f"Key players are focusing on {fake.catch_phrase().lower()}. "
        f"Recent developments indicate a shift towards {fake.bs().lower()} solutions."
    )
    return overview


def generate_industry_overview(industry):
    """
    Generates a synthetic industry overview.
    Uses Gemini AI if available, otherwise falls back to Faker.
    
    Args:
        industry (str): The industry name to generate overview for
        
    Returns:
        str: A synthetic industry overview text
    """
    if gemini_model is not None:
        return generate_industry_overview_with_gemini(industry)
    else:
        return generate_industry_overview_with_faker(industry)


def generate_synthetic_dataset(num_records=75):
    """
    Generates a complete synthetic dataset with client data and industry overviews.
    
    Args:
        num_records (int): Number of records to generate (default: 75)
        
    Returns:
        pd.DataFrame: DataFrame containing synthetic data
    """
    synthetic_data = []
    
    for _ in range(num_records):
        client_data = generate_client_record()
        industry_overview = generate_industry_overview(client_data['industry'])
        synthetic_data.append({
            'client_data': client_data,
            'industry_overview': industry_overview
        })
    
    df_synthetic = pd.DataFrame(synthetic_data)
    return df_synthetic
