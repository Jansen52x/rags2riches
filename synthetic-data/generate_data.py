"""
Main Script for Generating Synthetic Data
Orchestrates data generation, processing, image generation, PDF generation, and saving to files.
"""
import os
import shutil
import pandas as pd
from data_generator import generate_synthetic_dataset
from text_processor import process_dataframe
from image_generator import generate_all_materials
from pdf_generator import generate_all_pdf_brochures

# Get the directory where this script is located (synthetic-data folder)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def main(num_records=75, output_dir='output', generate_images=True, generate_pdfs=True, clean_output=True):
    """
    Main function to generate, process, and save synthetic data.
    
    Args:
        num_records (int): Number of synthetic records to generate
        output_dir (str): Directory to save output files (relative to synthetic-data folder)
        generate_images (bool): Whether to generate marketing images (brochures/flyers)
        generate_pdfs (bool): Whether to generate PDF brochures
        clean_output (bool): Whether to clean/overwrite existing output directory
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
    
    # Step 5: Generate marketing materials (brochures and flyers)
    if generate_images:
        print("\n" + "="*80)
        print("Generating Marketing Materials (Brochures & Flyers)...")
        print("="*80)
        marketing_dir = os.path.join(output_dir, 'marketing_materials')
        df_synthetic = generate_all_materials(df_synthetic, marketing_dir)
    
    # Step 6: Generate PDF brochures
    if generate_pdfs:
        print("\n" + "="*80)
        print("Generating PDF Brochures...")
        print("="*80)
        pdf_dir = os.path.join(output_dir, 'pdf_brochures')
        df_synthetic = generate_all_pdf_brochures(df_synthetic, pdf_dir)
    
    # Step 7: Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"\n✓ Created output directory: {output_dir}")
    
    # Step 8: Save the data
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
    
    print("\n" + "="*80)
    print("DATA GENERATION COMPLETE!")
    print("="*80)
    print(f"Total records: {len(df_synthetic)}")
    print(f"Output location: {os.path.abspath(output_dir)}")
    print(f"\nAll files are saved under the synthetic-data folder:")
    print(f"  - Data files: {os.path.relpath(output_dir, SCRIPT_DIR)}/")
    if generate_images:
        print(f"  - PNG materials: {os.path.relpath(marketing_dir, SCRIPT_DIR)}/")
    if generate_pdfs:
        print(f"  - PDF brochures: {os.path.relpath(pdf_dir, SCRIPT_DIR)}/")
    
    return df_synthetic, df_processed


if __name__ == "__main__":
    # Configure parameters here
    NUM_RECORDS = 75  # Choose a number between 50 and 100
    OUTPUT_DIR = 'output'  # Output directory relative to synthetic-data folder
    GENERATE_IMAGES = True  # Set to False to skip image generation
    GENERATE_PDFS = True    # Set to False to skip PDF generation
    CLEAN_OUTPUT = True     # Set to False to keep existing files and add new ones
    
    # Run the data generation pipeline
    raw_data, processed_data = main(
        num_records=NUM_RECORDS, 
        output_dir=OUTPUT_DIR, 
        generate_images=GENERATE_IMAGES, 
        generate_pdfs=GENERATE_PDFS,
        clean_output=CLEAN_OUTPUT
    )
