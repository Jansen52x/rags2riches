"""
Data Generator Module
Handles synthetic data generation using Faker library.
"""
from faker import Faker
import pandas as pd

fake = Faker()


def generate_client_record():
    """Generates a synthetic client record using Faker."""
    return {
        'company_name': fake.company(),
        'industry': fake.bs(),  # Using bs() for a business-sounding industry description
        'contact_person': fake.name(),
        'contact_email': fake.email(),
        'company_description': fake.catch_phrase() + ". " + fake.text(max_nb_chars=100)
    }


def generate_industry_overview(industry):
    """
    Generates a synthetic industry overview based on a given industry name.
    
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
