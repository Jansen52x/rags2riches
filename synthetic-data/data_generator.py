"""
Data Generator Module
Handles synthetic data generation using Gemini AI or Faker library as fallback.
"""
from faker import Faker
import pandas as pd
import os
import time
import random
import csv

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

# Always load seed companies from CSV (real companies like Microsoft, Shopee, etc.)
SEED_FILE = os.path.join(os.path.dirname(__file__), 'seed_companies.csv')
_seed_companies = []
if os.path.exists(SEED_FILE):
    try:
        with open(SEED_FILE, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            _seed_companies = [row for row in reader if row.get('company_name')]
        if _seed_companies:
            print(f"✓ Loaded {len(_seed_companies)} seed companies from CSV (will use Gemini for these)")
        else:
            print(f"⚠️ No seed companies found in {SEED_FILE}")
    except Exception as e:
        print(f"⚠️ Failed to load seed companies: {e}")
else:
    print(f"⚠️ Seed file not found: {SEED_FILE}")

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


def _normalize_domain(company_name: str) -> str:
    """Infer a simple domain from a company name (fallback if domain not provided)."""
    base = ''.join(ch.lower() for ch in company_name if ch.isalnum() or ch.isspace()).replace(' ', '')
    if not base:
        base = 'example'
    return f"{base}.com"


def generate_client_from_seed(index):
    """Generate a client record using a seeded (real) company entry from `seed_companies.csv`.

    Args:
        index (int): The index of the seed company to use
    
    All data comes from the CSV - NO FAKE DATA for real companies.
    CSV columns: company_name, industry, domain, description, contact_email, phone
    """
    if not _seed_companies or index >= len(_seed_companies):
        return None  # Return None if no seed company available

    seed = _seed_companies[index]
    company_name = seed.get('company_name')
    industry = seed.get('industry')
    domain = seed.get('domain')
    contact_email = seed.get('contact_email', f"contact@{domain}")
    phone = seed.get('phone', 'N/A')
    company_description = seed.get('description', '')

    # Validate we have required fields
    if not company_name or not industry:
        print(f"⚠️ Incomplete seed company data at index {index}, skipping")
        return None

    return {
        'company_name': company_name,
        'industry': industry,
        'contact_email': contact_email,
        'phone': phone,
        'domain': domain,
        'company_description': company_description,
        'is_seed': True  # Flag to indicate this is from seed CSV (real company)
    }


# Track which company we're generating (for seed vs synthetic)
_company_counter = 0
_gemini_failed_for_seed = False  # Track if Gemini failed for any seed company

def generate_client_record():
    """
    Generates a client record using seed companies first, then synthetic ones.
    - Seed companies (from CSV): 100% real data, use Gemini for documents
    - Synthetic companies (Faker): Fake data, use Faker for documents
    
    If Gemini fails for a seed company's documents, we skip remaining seed companies
    and move directly to synthetic companies.
    """
    global _company_counter, _gemini_failed_for_seed
    
    # If Gemini already failed for a seed company, skip to synthetic
    if _gemini_failed_for_seed:
        _company_counter += 1
        return generate_client_record_with_faker()
    
    # If we still have seed companies to use, return one
    if _company_counter < len(_seed_companies):
        seed_company = generate_client_from_seed(_company_counter)
        _company_counter += 1
        if seed_company is None:
            # Skip this seed company, try next
            return generate_client_record()
        return seed_company
    
    # Otherwise, generate synthetic company with Faker
    _company_counter += 1
    return generate_client_record_with_faker()


def mark_gemini_failed():
    """Mark that Gemini has failed for a seed company - skip remaining seed companies."""
    global _gemini_failed_for_seed
    _gemini_failed_for_seed = True
    print("\n" + "="*80)
    print("⚠️ GEMINI RATE LIMIT REACHED")
    print("   Skipping remaining real companies and switching to synthetic data generation")
    print("="*80 + "\n")


def reset_company_counter():
    """Reset the company counter (useful for testing or regeneration)."""
    global _company_counter, _gemini_failed_for_seed
    _company_counter = 0
    _gemini_failed_for_seed = False


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
    First uses all seed companies from CSV (with is_seed=True), then generates synthetic ones.
    
    Args:
        num_records (int): Total number of records to generate (default: 75)
        
    Returns:
        pd.DataFrame: DataFrame containing synthetic data
    """
    # Reset counter to ensure we start from the first seed company
    reset_company_counter()
    
    synthetic_data = []
    num_seed = len(_seed_companies)
    
    print(f"\n📊 Generation Plan:")
    print(f"   - First {num_seed} companies: Real companies from CSV (using Gemini for documents)")
    print(f"   - Remaining {num_records - num_seed} companies: Synthetic (using Faker for documents)")
    print()
    
    for i in range(num_records):
        client_data = generate_client_record()
        industry_overview = generate_industry_overview(client_data['industry'])
        synthetic_data.append({
            'client_data': client_data,
            'industry_overview': industry_overview
        })
    
    df_synthetic = pd.DataFrame(synthetic_data)
    return df_synthetic
