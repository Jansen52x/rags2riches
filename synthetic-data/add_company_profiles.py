import os
import pandas as pd
import json
import time

from advanced_document_generator import generate_company_profile
from multi_document_generator import generate_document_image, generate_document_pdf

# Get the directory where this script is located (synthetic-data folder)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def main():
  # Paths
  output_dir = os.path.join(SCRIPT_DIR, 'output')
  images_dir = os.path.join(output_dir, 'document_images')
  pdfs_dir = os.path.join(output_dir, 'document_pdfs')
  metadata_file = os.path.join(output_dir, 'all_documents_metadata.json')
  raw_csv = os.path.join(output_dir, 'synthetic_data_raw.csv')
  
  # Check if output exists
  if not os.path.exists(output_dir):
    print(" Output directory not found. Please run generate_data.py first.")
    return
  
  print("=" * 80)
  print("ADDING COMPANY PROFILE DOCUMENTS (Non-Destructive)")
  print("=" * 80)
  print(f"\n Found existing output directory: {output_dir}")
  
  # Load existing metadata
  existing_documents = []
  if os.path.exists(metadata_file):
    with open(metadata_file, 'r', encoding='utf-8') as f:
      existing_documents = json.load(f)
    print(f" Loaded {len(existing_documents)} existing documents")
  
  # Load company data
  if not os.path.exists(raw_csv):
    print(f" Raw data file not found: {raw_csv}")
    return
  
  df = pd.read_csv(raw_csv)
  print(f" Loaded {len(df)} companies from CSV")
  
  # Check if profiles already exist
  existing_profiles = [doc for doc in existing_documents if doc.get('document_type') == 'company_profile']
  if existing_profiles:
    print(f"\n Found {len(existing_profiles)} existing company profiles")
    response = input("Do you want to regenerate them? (yes/no): ").strip().lower()
    if response != 'yes':
      print(" Skipping profile generation. Exiting.")
      return
    # Remove old profiles from list
    existing_documents = [doc for doc in existing_documents if doc.get('document_type') != 'company_profile']
    print(f" Removed old profiles. Will regenerate.")
  
  # Get next document ID
  next_doc_id = max([doc.get('document_id', 0) for doc in existing_documents], default=-1) + 1
  print(f"\n Starting new documents at ID: {next_doc_id}")
  
  # Generate company profiles
  print("\n" + "=" * 80)
  print("Generating Company Profile Documents...")
  print("=" * 80)
  
  new_documents = []
  start_time = time.time()
  
  for idx, row in df.iterrows():
    # Parse client_data (it's stored as string in CSV)
    client_data = eval(row['client_data']) if isinstance(row['client_data'], str) else row['client_data']
    company_name = client_data.get('company_name', 'Unknown')
    is_seed_company = client_data.get('is_seed', False)
    
    print(f"\n[{idx+1}/{len(df)}] Generating profile for: {company_name} ({' Real' if is_seed_company else ' Synthetic'})")
    
    # Generate company profile
    # CRITICAL: Always use Gemini (use_gemini=True) - NO FAKER for company profiles
    try:
      profile = generate_company_profile(client_data, use_gemini=True)
      profile['company_id'] = idx
      profile['document_id'] = next_doc_id
      
      # Generate image
      img_filename = f"company_profile_{next_doc_id:04d}.png"
      img_path = os.path.join(images_dir, img_filename)
      generate_document_image(profile, img_path)
      profile['image_path'] = img_path
      
      # Generate PDF
      pdf_filename = f"company_profile_{next_doc_id:04d}.pdf"
      pdf_path = os.path.join(pdfs_dir, pdf_filename)
      generate_document_pdf(profile, pdf_path)
      profile['pdf_path'] = pdf_path
      
      new_documents.append(profile)
      next_doc_id += 1
      
      print(f"  Generated profile (ID: {profile['document_id']})")
      
    except Exception as e:
      print(f"  Failed: {e}")
      continue
  
  total_time = time.time() - start_time
  
  # Combine with existing documents
  all_documents = existing_documents + new_documents
  
  # Save updated metadata
  print("\n" + "=" * 80)
  print("Saving Updated Metadata...")
  print("=" * 80)
  
  with open(metadata_file, 'w', encoding='utf-8') as f:
    json.dump(all_documents, f, indent=2)
  
  print(f" Saved {len(all_documents)} total documents to: {metadata_file}")
  print(f"  • Previous documents: {len(existing_documents)}")
  print(f"  • New profiles added: {len(new_documents)}")
  print(f"  • Total documents now: {len(all_documents)}")
  
  print("\n" + "=" * 80)
  print("COMPANY PROFILES ADDED SUCCESSFULLY!")
  print("=" * 80)
  print(f" Total time: {total_time/60:.1f} minutes")
  print(f" Images: {images_dir}")
  print(f" PDFs: {pdfs_dir}")
  print(f" Metadata: {metadata_file}")
  print("\n All existing documents preserved!")
  print(f" Added {len(new_documents)} new company profile documents")


if __name__ == "__main__":
  main()
