"""
Multimodal Processor for RAG System

Extracts images from PDFs and converts them to text using OCR and image captioning.
This allows images to be indexed as regular text chunks in the existing pipeline.
"""

from typing import List, Dict, Any, Optional
import logging
from PIL import Image
import io
import os
from pypdf import PdfReader
import base64

logger = logging.getLogger(__name__)


class MultimodalProcessor:
    """Process images from PDFs and convert to text"""

    def __init__(self, config):
        self.config = config
        self.enable_ocr = getattr(config, 'ENABLE_OCR', True)
        self.enable_captioning = getattr(config, 'ENABLE_IMAGE_CAPTIONING', False)
        self.image_storage_path = getattr(config, 'IMAGE_STORAGE_PATH', 'image_store')

        # Create image storage directory
        os.makedirs(self.image_storage_path, exist_ok=True)

        # Initialize OCR if enabled
        if self.enable_ocr:
            try:
                import pytesseract
                # Set tesseract path if configured
                if hasattr(config, 'TESSERACT_CMD') and config.TESSERACT_CMD:
                    pytesseract.pytesseract.tesseract_cmd = config.TESSERACT_CMD
                    logger.info(f"OCR enabled with pytesseract at: {config.TESSERACT_CMD}")
                else:
                    logger.info("OCR enabled with pytesseract (using PATH)")
                self.pytesseract = pytesseract
            except ImportError:
                logger.warning("pytesseract not available, OCR disabled")
                self.enable_ocr = False
                self.pytesseract = None

        # Initialize captioning service if enabled
        if self.enable_captioning:
            try:
                from openai import OpenAI
                nvidia_api_key = getattr(config, 'NVIDIA_API_KEY', '')
                nvidia_base_url = getattr(config, 'NVIDIA_BASE_URL', '')
                if nvidia_api_key:
                    self.vision_client = OpenAI(
                        base_url=nvidia_base_url,
                        api_key=nvidia_api_key
                    )
                    self.vision_model = getattr(config, 'VISION_MODEL', 'microsoft/phi-3-vision-128k-instruct')
                    logger.info(f"Image captioning enabled with model: {self.vision_model}")
                else:
                    logger.warning("NVIDIA API key not configured, image captioning disabled")
                    self.enable_captioning = False
            except ImportError:
                logger.warning("OpenAI client not available, image captioning disabled")
                self.enable_captioning = False

    def extract_images_from_pdf(self, pdf_path: str, document_id: str) -> List[Dict[str, Any]]:
        """
        Extract images from PDF file

        Args:
            pdf_path: Path to PDF file
            document_id: Unique document identifier

        Returns:
            List of image metadata dictionaries
        """
        images = []

        try:
            reader = PdfReader(pdf_path)

            for page_num, page in enumerate(reader.pages):
                # Extract images from page
                if '/XObject' in page['/Resources']:
                    xobjects = page['/Resources']['/XObject'].get_object()

                    for obj_name in xobjects:
                        obj = xobjects[obj_name]

                        if obj['/Subtype'] == '/Image':
                            try:
                                # Extract image data
                                image_data = self._extract_image_data(obj)

                                if image_data:
                                    # Save image
                                    image_filename = f"{document_id}_page{page_num + 1}_img{len(images)}.png"
                                    image_path = os.path.join(self.image_storage_path, image_filename)
                                    image_data.save(image_path, 'PNG')

                                    images.append({
                                        'image_path': image_path,
                                        'page_number': page_num + 1,
                                        'image_index': len(images),
                                        'document_id': document_id
                                    })

                                    logger.debug(f"Extracted image from page {page_num + 1}")

                            except Exception as e:
                                logger.warning(f"Could not extract image from page {page_num + 1}: {e}")
                                continue

            logger.info(f"Extracted {len(images)} images from PDF")

        except Exception as e:
            logger.error(f"Error extracting images from PDF: {e}")

        return images

    def _extract_image_data(self, image_obj) -> Optional[Image.Image]:
        """
        Extract PIL Image from PDF image object

        Args:
            image_obj: PDF image object

        Returns:
            PIL Image or None
        """
        try:
            size = (image_obj['/Width'], image_obj['/Height'])
            data = image_obj.get_data()

            # Handle different color spaces
            if image_obj['/ColorSpace'] == '/DeviceRGB':
                mode = 'RGB'
            elif image_obj['/ColorSpace'] == '/DeviceGray':
                mode = 'L'
            else:
                mode = 'RGB'

            return Image.frombytes(mode, size, data)

        except Exception as e:
            logger.debug(f"Could not convert image data: {e}")
            return None

    def process_image_to_text(self, image_path: str, page_number: int = None) -> str:
        """
        Convert image to text using OCR and/or captioning

        Args:
            image_path: Path to image file
            page_number: Page number where image was found

        Returns:
            Text description of the image
        """
        text_parts = []

        # Add context
        if page_number:
            text_parts.append(f"[Image from Page {page_number}]")

        # Extract text from image using OCR
        if self.enable_ocr:
            ocr_text = self._extract_text_with_ocr(image_path)
            if ocr_text and ocr_text.strip():
                text_parts.append(f"Text in image: {ocr_text.strip()}")

        # Generate image caption/description
        if self.enable_captioning:
            caption = self._generate_image_caption(image_path)
            if caption:
                text_parts.append(f"Image description: {caption}")

        # If no text was extracted, add a placeholder
        if len(text_parts) <= 1:  # Only has page number
            text_parts.append("Image content (no text or caption available)")

        return "\n".join(text_parts)

    def _extract_text_with_ocr(self, image_path: str) -> str:
        """
        Extract text from image using OCR

        Args:
            image_path: Path to image file

        Returns:
            Extracted text
        """
        if not self.enable_ocr or not self.pytesseract:
            return ""

        try:
            image = Image.open(image_path)
            text = self.pytesseract.image_to_string(image)
            logger.debug(f"OCR extracted {len(text)} characters from {image_path}")
            return text

        except Exception as e:
            logger.warning(f"OCR failed for {image_path}: {e}")
            return ""

    def _generate_image_caption(self, image_path: str) -> str:
        """
        Generate caption for image using vision model

        Args:
            image_path: Path to image file

        Returns:
            Generated caption
        """
        if not self.enable_captioning:
            return ""

        try:
            # Read and encode image
            with open(image_path, 'rb') as f:
                image_data = base64.b64encode(f.read()).decode('utf-8')

            # Call vision model
            response = self.vision_client.chat.completions.create(
                model=self.vision_model,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Describe this image in detail. Include any text, charts, diagrams, or important visual elements."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_data}"
                            }
                        }
                    ]
                }],
                max_tokens=300,
                temperature=0.2
            )

            caption = response.choices[0].message.content
            logger.debug(f"Generated caption for {image_path}")
            return caption

        except Exception as e:
            logger.warning(f"Image captioning failed for {image_path}: {e}")
            return ""

    def process_pdf_with_images(
        self,
        pdf_path: str,
        document_id: str,
        text_content: str
    ) -> Dict[str, Any]:
        """
        Process PDF to extract both text and images, converting images to text

        Args:
            pdf_path: Path to PDF file
            document_id: Document identifier
            text_content: Already extracted text content

        Returns:
            Dictionary with combined content and metadata
        """
        # Extract images
        images = self.extract_images_from_pdf(pdf_path, document_id)

        if not images:
            logger.info(f"No images found in PDF {document_id}")
            return {
                'content': text_content,
                'has_images': False,
                'image_count': 0
            }

        # Process each image to text
        image_texts = []
        for img_info in images:
            img_text = self.process_image_to_text(
                img_info['image_path'],
                img_info['page_number']
            )
            image_texts.append(img_text)

            # Update image info with text
            img_info['extracted_text'] = img_text

        # Combine text and image descriptions
        combined_content = text_content
        if image_texts:
            combined_content += "\n\n--- Images Found in Document ---\n\n"
            combined_content += "\n\n".join(image_texts)

        logger.info(f"Processed {len(images)} images from PDF {document_id}")

        return {
            'content': combined_content,
            'has_images': True,
            'image_count': len(images),
            'images': images
        }

    def health_check(self) -> Dict[str, bool]:
        """
        Check health of multimodal components

        Returns:
            Dictionary with component health status
        """
        return {
            'ocr_available': self.enable_ocr,
            'captioning_available': self.enable_captioning,
            'storage_accessible': os.path.exists(self.image_storage_path)
        }
