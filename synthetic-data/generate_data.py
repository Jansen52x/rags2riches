"""
Main Script for Generating Synthetic Data
Orchestrates data generation, processing, image generation, PDF generation, and saving to files.
Now supports multiple documents per company and shared documents.
"""
import os
import shutil
import pandas as pd

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
    load_dotenv(env_path)
except ImportError:
    pass  # Will use default values if dotenv not available

from data_generator import generate_synthetic_dataset
from text_processor import process_dataframe
from image_generator import generate_all_materials
from pdf_generator import generate_all_pdf_brochures
from advanced_document_generator import generate_all_documents_for_company, generate_shared_documents
from multi_document_generator import generate_document_image, generate_document_pdf

# Get the directory where this script is located (synthetic-data folder)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def main(num_records=75, output_dir='output', generate_images=True, generate_pdfs=True, 
         clean_output=True, multi_docs_per_company=True, docs_per_company=(5, 10), 
         generate_partnerships=True, num_partnerships=10):
    """
    Main function to generate, process, and save synthetic data.
    
    Args:
        num_records (int): Number of synthetic records to generate
        output_dir (str): Directory to save output files (relative to synthetic-data folder)
        generate_images (bool): Whether to generate marketing images (brochures/flyers)
        generate_pdfs (bool): Whether to generate PDF brochures
        clean_output (bool): Whether to clean/overwrite existing output directory
        multi_docs_per_company (bool): Generate multiple diverse documents per company
        docs_per_company (tuple): Min and max number of documents per company
        generate_partnerships (bool): Generate shared partnership documents
        num_partnerships (int): Number of partnership documents to create
    """
    # Ensure output_dir is relative to the synthetic-data folder
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(SCRIPT_DIR, output_dir)
    
    # Clean output directory if it exists and clean_output is True
    if clean_output and os.path.exists(output_dir):
        print(f"Cleaning existing output directory: {output_dir}")
        shutil.rmtree(output_dir)
        print("✓ Cleaned previous output")
    
    print(f"\nGenerating {num_records} synthetic records...")
    print(f"Output will be saved to: {output_dir}")
    
    # Step 1: Generate synthetic data
    df_synthetic = generate_synthetic_dataset(num_records)
    print(f"✓ Generated {len(df_synthetic)} records")
    
    # Step 2: Display preview of raw data
    print("\n" + "="*80)
    print("PREVIEW: Raw Generated Data (First 5 records)")
    print("="*80)
    print(df_synthetic.head().to_string())
    
    # Step 3: Process the data (clean text and tokenize)
    print("\n" + "="*80)
    print("Processing text data...")
    print("="*80)
    df_processed = process_dataframe(df_synthetic.copy())
    print("✓ Text processing complete")
    
    # Step 4: Display preview of processed data
    print("\n" + "="*80)
    print("PREVIEW: Processed Data (First 5 records)")
    print("="*80)
    preview_df = df_processed[['client_data', 'industry_overview', 'document_text']].head()
    print(preview_df.to_string())
    
    # Step 5: Generate multiple diverse documents per company
    all_documents = []
    document_metadata = []
    
    if multi_docs_per_company:
        print("\n" + "="*80)
        print(f"Generating {docs_per_company[0]}-{docs_per_company[1]} diverse documents per company...")
        print("="*80)
        
        doc_count = 0
        for idx, row in df_synthetic.iterrows():
            client_data = row['client_data']
            
            # Use Gemini for seed companies (real companies), Faker for synthetic ones
            is_seed_company = client_data.get('is_seed', False)
            documents = generate_all_documents_for_company(
                client_data, 
                docs_per_company, 
                use_gemini=is_seed_company
            )
            
            # If documents is None, Gemini failed for a seed company - skip it
            if documents is None:
                company_name = client_data.get('company_name', 'Unknown')
                print(f"  ⏭️  Skipped real company '{company_name}' (Gemini limit reached)")
                continue
            
            for doc_idx, doc in enumerate(documents):
                doc['company_id'] = idx
                doc['document_id'] = doc_count
                all_documents.append(doc)
                doc_count += 1
            
            if (idx + 1) % 10 == 0:
                print(f"  ✓ Generated documents for {idx + 1}/{len(df_synthetic)} companies ({doc_count} total docs)")
        
        print(f"✓ Generated {doc_count} documents across {len(df_synthetic)} companies")
        
        # Generate partnership documents (only between synthetic companies, not real ones)
        if generate_partnerships:
            print(f"\nGenerating {num_partnerships} partnership documents...")
            # Filter to only synthetic companies (not from seed CSV)
            synthetic_companies_only = [
                company for company in df_synthetic['client_data'].tolist() 
                if not company.get('is_seed', False)
            ]
            
            if len(synthetic_companies_only) >= 2:
                partnership_docs = generate_shared_documents(synthetic_companies_only, num_partnerships)
                
                for doc in partnership_docs:
                    doc['company_id'] = -1  # Special ID for shared documents
                    doc['document_id'] = doc_count
                    all_documents.append(doc)
                    doc_count += 1
                
                print(f"✓ Generated {len(partnership_docs)} partnership documents (synthetic companies only)")
            else:
                print(f"⏭️  Skipped partnership documents (need at least 2 synthetic companies, have {len(synthetic_companies_only)})")
    
    # Step 6: Generate images for all documents
    if generate_images and multi_docs_per_company:
        print("\n" + "="*80)
        print(f"Generating images for {len(all_documents)} documents...")
        print("="*80)
        images_dir = os.path.join(output_dir, 'document_images')
        os.makedirs(images_dir, exist_ok=True)
        
        for idx, doc in enumerate(all_documents):
            doc_type = doc.get('document_type', 'unknown')
            doc_id = doc.get('document_id', idx)
            img_path = os.path.join(images_dir, f'{doc_type}_{doc_id:04d}.png')
            
            try:
                generate_document_image(doc, img_path)
                doc['image_path'] = img_path
            except Exception as e:
                print(f"  Warning: Failed to generate image for document {doc_id}: {e}")
            
            if (idx + 1) % 50 == 0:
                print(f"  ✓ Generated {idx + 1}/{len(all_documents)} images")
        
        print(f"✓ All document images saved to: {os.path.abspath(images_dir)}")
    elif generate_images and not multi_docs_per_company:
        # Original single brochure/flyer generation
        print("\n" + "="*80)
        print("Generating Marketing Materials (Brochures & Flyers)...")
        print("="*80)
        marketing_dir = os.path.join(output_dir, 'marketing_materials')
        df_synthetic = generate_all_materials(df_synthetic, marketing_dir)
    
    # Step 7: Generate PDFs for all documents
    if generate_pdfs and multi_docs_per_company:
        print("\n" + "="*80)
        print(f"Generating PDFs for {len(all_documents)} documents...")
        print("="*80)
        pdfs_dir = os.path.join(output_dir, 'document_pdfs')
        os.makedirs(pdfs_dir, exist_ok=True)
        
        for idx, doc in enumerate(all_documents):
            doc_type = doc.get('document_type', 'unknown')
            doc_id = doc.get('document_id', idx)
            pdf_path = os.path.join(pdfs_dir, f'{doc_type}_{doc_id:04d}.pdf')
            
            try:
                generate_document_pdf(doc, pdf_path)
                doc['pdf_path'] = pdf_path
            except Exception as e:
                print(f"  Warning: Failed to generate PDF for document {doc_id}: {e}")
            
            if (idx + 1) % 50 == 0:
                print(f"  ✓ Generated {idx + 1}/{len(all_documents)} PDFs")
        
        print(f"✓ All document PDFs saved to: {os.path.abspath(pdfs_dir)}")
    elif generate_pdfs and not multi_docs_per_company:
        # Original single PDF generation
        print("\n" + "="*80)
        print("Generating PDF Brochures...")
        print("="*80)
        pdf_dir = os.path.join(output_dir, 'pdf_brochures')
        df_synthetic = generate_all_pdf_brochures(df_synthetic, pdf_dir)
    
    # Step 8: Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"\n✓ Created output directory: {output_dir}")
    
    # Step 9: Save the data
    raw_csv_path = os.path.join(output_dir, 'synthetic_data_raw.csv')
    processed_csv_path = os.path.join(output_dir, 'synthetic_data_processed.csv')
    processed_json_path = os.path.join(output_dir, 'synthetic_data_processed.json')
    
    # Save raw data
    df_synthetic.to_csv(raw_csv_path, index=False)
    print(f"✓ Saved raw data to: {raw_csv_path}")
    
    # Save processed data (convert tokens list to string for CSV)
    df_export = df_processed.copy()
    df_export['document_text'] = df_export['document_text'].apply(lambda x: ' '.join(x) if isinstance(x, list) else x)
    df_export.to_csv(processed_csv_path, index=False)
    print(f"✓ Saved processed data to: {processed_csv_path}")
    
    # Save as JSON for better structure preservation
    df_processed.to_json(processed_json_path, orient='records', indent=2)
    print(f"✓ Saved processed data to: {processed_json_path}")
    
    # Save all documents metadata
    if multi_docs_per_company and all_documents:
        documents_json_path = os.path.join(output_dir, 'all_documents_metadata.json')
        df_documents = pd.DataFrame(all_documents)
        df_documents.to_json(documents_json_path, orient='records', indent=2)
        print(f"✓ Saved documents metadata to: {documents_json_path}")
    
    print("\n" + "="*80)
    print("DATA GENERATION COMPLETE!")
    print("="*80)
    print(f"Total companies: {len(df_synthetic)}")
    if multi_docs_per_company:
        print(f"Total documents generated: {len(all_documents)}")
        print(f"Average documents per company: {len(all_documents) / len(df_synthetic):.1f}")
    print(f"Output location: {os.path.abspath(output_dir)}")
    print(f"\nAll files are saved under the synthetic-data folder:")
    print(f"  - Data files: {os.path.relpath(output_dir, SCRIPT_DIR)}/")
    if multi_docs_per_company:
        if generate_images:
            print(f"  - Document images: {os.path.relpath(images_dir, SCRIPT_DIR)}/")
        if generate_pdfs:
            print(f"  - Document PDFs: {os.path.relpath(pdfs_dir, SCRIPT_DIR)}/")
    else:
        if generate_images:
            print(f"  - PNG materials: {os.path.relpath(marketing_dir, SCRIPT_DIR)}/")
        if generate_pdfs:
            print(f"  - PDF brochures: {os.path.relpath(pdf_dir, SCRIPT_DIR)}/")
    
    return df_synthetic, df_processed, all_documents if multi_docs_per_company else None


if __name__ == "__main__":
    # Load configuration from .env file (with fallback defaults)
    NUM_RECORDS = int(os.environ.get('NUM_COMPANIES', 75))
    OUTPUT_DIR = 'output'  # Output directory relative to synthetic-data folder
    GENERATE_IMAGES = os.environ.get('GENERATE_IMAGES', 'true').lower() == 'true'
    GENERATE_PDFS = os.environ.get('GENERATE_PDFS', 'true').lower() == 'true'
    CLEAN_OUTPUT = os.environ.get('CLEAN_OUTPUT_ON_START', 'true').lower() == 'true'
    
    # Multi-document generation settings
    MULTI_DOCS_PER_COMPANY = True  # Generate multiple diverse documents per company
    MIN_DOCS = int(os.environ.get('MIN_DOCS_PER_COMPANY', 5))
    MAX_DOCS = int(os.environ.get('MAX_DOCS_PER_COMPANY', 10))
    DOCS_PER_COMPANY = (MIN_DOCS, MAX_DOCS)
    GENERATE_PARTNERSHIPS = True   # Generate shared partnership documents
    NUM_PARTNERSHIPS = int(os.environ.get('NUM_PARTNERSHIPS', 10))
    
    # Run the data generation pipeline
    raw_data, processed_data, documents = main(
        num_records=NUM_RECORDS, 
        output_dir=OUTPUT_DIR, 
        generate_images=GENERATE_IMAGES, 
        generate_pdfs=GENERATE_PDFS,
        clean_output=CLEAN_OUTPUT,
        multi_docs_per_company=MULTI_DOCS_PER_COMPANY,
        docs_per_company=DOCS_PER_COMPANY,
        generate_partnerships=GENERATE_PARTNERSHIPS,
        num_partnerships=NUM_PARTNERSHIPS
    )
