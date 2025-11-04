"""
Integration Script: Ingest Synthetic Data into RAG System
Loads data from synthetic-data/output/ and ingests it into ChromaDB via the data service
"""
import os
import sys
import json
import argparse
import logging
from pathlib import Path

# Add parent directory to path to import from data/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'data'))

from document_service import DocumentService
from embedding_service import EmbeddingService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_project_root():
    """Get the project root directory"""
    return Path(__file__).parent.parent


def ingest_json_file(document_service: DocumentService, json_path: str):
    """
    Ingest a JSON file containing synthetic data

    Args:
        document_service: DocumentService instance
        json_path: Path to JSON file
    """
    logger.info(f"Ingesting JSON file: {json_path}")

    if not os.path.exists(json_path):
        logger.error(f"File not found: {json_path}")
        return None

    try:
        result = document_service.process_json(json_path)
        logger.info(f"✓ Ingested {result['documents_processed']} documents with {result['total_chunks']} chunks")

        if result.get('errors'):
            logger.warning(f"Encountered {len(result['errors'])} errors during ingestion")
            for error in result['errors'][:5]:  # Show first 5 errors
                logger.warning(f"  - {error}")

        return result

    except Exception as e:
        logger.error(f"Error ingesting JSON file: {e}")
        return None


def ingest_csv_file(document_service: DocumentService, csv_path: str):
    """
    Ingest a CSV file containing synthetic data

    Args:
        document_service: DocumentService instance
        csv_path: Path to CSV file
    """
    logger.info(f"Ingesting CSV file: {csv_path}")

    if not os.path.exists(csv_path):
        logger.error(f"File not found: {csv_path}")
        return None

    try:
        result = document_service.process_csv(csv_path)
        logger.info(f"✓ Ingested {result['documents_processed']} documents with {result['total_chunks']} chunks")

        if result.get('errors'):
            logger.warning(f"Encountered {len(result['errors'])} errors during ingestion")
            for error in result['errors'][:5]:
                logger.warning(f"  - {error}")

        return result

    except Exception as e:
        logger.error(f"Error ingesting CSV file: {e}")
        return None


def ingest_pdf_directory(document_service: DocumentService, pdf_dir: str):
    """
    Ingest all PDF files from a directory

    Args:
        document_service: DocumentService instance
        pdf_dir: Path to directory containing PDFs
    """
    logger.info(f"Ingesting PDFs from directory: {pdf_dir}")

    if not os.path.exists(pdf_dir):
        logger.error(f"Directory not found: {pdf_dir}")
        return None

    try:
        result = document_service.process_batch_pdfs(pdf_dir)
        logger.info(f"✓ Ingested {result['documents_processed']} PDFs with {result['total_chunks']} chunks")

        if result.get('errors'):
            logger.warning(f"Encountered {len(result['errors'])} errors during ingestion")
            for error in result['errors'][:5]:
                logger.warning(f"  - {error}")

        return result

    except Exception as e:
        logger.error(f"Error ingesting PDFs: {e}")
        return None


def ingest_image_directory(document_service: DocumentService, image_dir: str):
    """
    Ingest all image files from a directory using OCR

    Args:
        document_service: DocumentService instance
        image_dir: Path to directory containing images
    """
    logger.info(f"Ingesting images from directory: {image_dir}")

    if not os.path.exists(image_dir):
        logger.error(f"Directory not found: {image_dir}")
        return None

    try:
        result = document_service.process_batch_images(image_dir)
        logger.info(f"✓ Ingested {result['documents_processed']} images with {result['total_chunks']} chunks")

        if result.get('errors'):
            logger.warning(f"Encountered {len(result['errors'])} errors during ingestion")
            for error in result['errors'][:5]:
                logger.warning(f"  - {error}")

        return result

    except Exception as e:
        logger.error(f"Error ingesting images: {e}")
        return None


def main():
    """Main function to orchestrate synthetic data ingestion"""
    parser = argparse.ArgumentParser(
        description='Ingest synthetic data into RAG system'
    )
    parser.add_argument(
        '--json',
        type=str,
        default='synthetic-data/output/synthetic_data_processed.json',
        help='Path to JSON file (relative to project root)'
    )
    parser.add_argument(
        '--csv',
        type=str,
        default='synthetic-data/output/synthetic_data_processed.csv',
        help='Path to CSV file (relative to project root)'
    )
    parser.add_argument(
        '--pdfs',
        type=str,
        default='synthetic-data/output/pdf_brochures',
        help='Path to PDF directory (relative to project root)'
    )
    parser.add_argument(
        '--images',
        type=str,
        default='synthetic-data/output/marketing_materials',
        help='Path to images directory (relative to project root)'
    )
    parser.add_argument(
        '--format',
        type=str,
        choices=['json', 'csv', 'pdfs', 'images', 'all'],
        default='json',
        help='Which format to ingest (default: json)'
    )
    parser.add_argument(
        '--clear',
        action='store_true',
        help='Clear existing data before ingestion'
    )

    args = parser.parse_args()

    # Get project root
    project_root = get_project_root()

    # Resolve paths relative to project root
    json_path = project_root / args.json
    csv_path = project_root / args.csv
    pdf_dir = project_root / args.pdfs
    image_dir = project_root / args.images

    logger.info("="*70)
    logger.info("Synthetic Data Ingestion Script")
    logger.info("="*70)
    logger.info(f"Project root: {project_root}")
    logger.info(f"Format: {args.format}")

    try:
        # Initialize services
        logger.info("Initializing services...")
        embedding_service = EmbeddingService()
        document_service = DocumentService(embedding_service)

        # Health check
        if not document_service.health_check():
            logger.error("ChromaDB health check failed. Is ChromaDB running?")
            logger.error("Start ChromaDB with: docker-compose up chromadb")
            return

        logger.info("✓ Services initialized successfully")

        # Clear existing data if requested
        if args.clear:
            logger.warning("Clearing existing data...")
            document_service.rebuild_index()
            logger.info("✓ Cleared existing data")

        # Get stats before ingestion
        stats_before = document_service.get_stats()
        logger.info(f"Current state: {stats_before['total_documents']} documents, {stats_before['total_chunks']} chunks")

        # Ingest based on format
        results = []

        if args.format == 'json' or args.format == 'all':
            logger.info("\n" + "="*70)
            logger.info("Ingesting JSON data...")
            logger.info("="*70)
            result = ingest_json_file(document_service, str(json_path))
            if result:
                results.append(result)

        if args.format == 'csv' or args.format == 'all':
            logger.info("\n" + "="*70)
            logger.info("Ingesting CSV data...")
            logger.info("="*70)
            result = ingest_csv_file(document_service, str(csv_path))
            if result:
                results.append(result)

        if args.format == 'pdfs' or args.format == 'all':
            logger.info("\n" + "="*70)
            logger.info("Ingesting PDF brochures...")
            logger.info("="*70)
            result = ingest_pdf_directory(document_service, str(pdf_dir))
            if result:
                results.append(result)

        if args.format == 'images' or args.format == 'all':
            logger.info("\n" + "="*70)
            logger.info("Ingesting marketing images (PNG) with OCR...")
            logger.info("="*70)
            result = ingest_image_directory(document_service, str(image_dir))
            if result:
                results.append(result)

        # Get stats after ingestion
        stats_after = document_service.get_stats()

        # Summary
        logger.info("\n" + "="*70)
        logger.info("INGESTION COMPLETE")
        logger.info("="*70)

        total_docs = sum(r.get('documents_processed', 0) for r in results)
        total_chunks = sum(r.get('total_chunks', 0) for r in results)

        logger.info(f"Documents ingested: {total_docs}")
        logger.info(f"Chunks created: {total_chunks}")
        logger.info(f"Total documents in DB: {stats_after['total_documents']}")
        logger.info(f"Total chunks in DB: {stats_after['total_chunks']}")
        logger.info(f"Collection: {stats_after['collection_name']}")

        logger.info("\n✓ Synthetic data successfully ingested into RAG system!")
        logger.info("\nYou can now query the data using the RAG query service:")
        logger.info("  curl -X POST http://localhost:8001/query \\")
        logger.info("    -H 'Content-Type: application/json' \\")
        logger.info("    -d '{\"query\": \"Tell me about companies in the robotics industry\"}'")

    except KeyboardInterrupt:
        logger.info("\nIngestion interrupted by user")
    except Exception as e:
        logger.error(f"Error during ingestion: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
