import base64
import time
import pdf2image
import io
from typing import List, Optional

from google.cloud import documentai_v1
from google.cloud.documentai_v1 import DocumentProcessorServiceClient
from google.oauth2 import service_account

from log_utils import logger

class DocumentAiService:
    """
    A service class for processing documents using Google Cloud Document AI
    with additional PDF to image conversion capabilities.
    """

    def __init__(self, config: dict):
        """
        Initialize the DocumentAiService with configuration.

        :param config: Configuration dictionary containing GCP and Document AI settings
        """
        # Configure logging
        self.logger = setup_logging(self.__class__.__name__)

        # Extract configuration
        self.project_name = config.get('gcp', {}).get('project_name')
        self.project_number = config.get('gcp', {}).get('project_number')
        self.location = config.get('document_ai', {}).get('location')
        self.processor_id = config.get('document_ai', {}).get('processor_id')
        self.service_account_path = config.get('document_ai', {}).get('path_to_service_account')

        # Validate configuration
        if not all([self.project_name, self.project_number, 
                    self.location, self.processor_id, self.service_account_path]):
            raise ValueError("Incomplete configuration for DocumentAiService")

        # Set up credentials
        try:
            credentials = service_account.Credentials.from_service_account_file(
                self.service_account_path
            )
        except FileNotFoundError:
            self.logger.error(f"Service account file not found: {self.service_account_path}")
            raise

        # Initialize client
        self.client_options = {"api_endpoint": f"{self.location}-documentai.googleapis.com"}
        self.documentai_client = documentai_v1.DocumentProcessorServiceClient(
            client_options=self.client_options,
            credentials=credentials
        )

        # Construct processor name
        self.processor_name = (
            f"projects/{self.project_number}/locations/{self.location}/"
            f"processors/{self.processor_id}"
        )
        self.logger.info(f"Processor name: {self.processor_name}")

    def init(self):
        """
        Initialize and verify connection by listing processors.

        :raises Exception: If connection cannot be verified
        """
        try:
            self.logger.info('Verifying Document AI connection')
            request = documentai_v1.ListProcessorsRequest(
                parent=f"projects/{self.project_number}/locations/{self.location}"
            )
            processors = list(self.documentai_client.list_processors(request))
            
            if not processors:
                raise ValueError("No processors found")
            
            for processor in processors:
                self.logger.debug(f"Connected to processor: {processor.display_name}")
            
            self.logger.info('Document AI connection verified')
        except Exception as e:
            self.logger.error(f"Connection verification failed: {e}")
            raise

    def recognize(self, image_buffer: bytes) -> str:
        """
        Recognize text from an image buffer.

        :param image_buffer: Image buffer to process
        :return: Extracted text
        :raises Exception: If processing fails
        """
        try:
            self.logger.info('Processing image with Document AI')
            
            # Prepare request
            raw_document = documentai_v1.RawDocument(
                content=base64.b64encode(image_buffer).decode(),
                mime_type='image/jpeg'  # Adjust as needed
            )
            
            request = documentai_v1.ProcessRequest(
                name=self.processor_name,
                raw_document=raw_document
            )
            
            # Process document
            result = self.documentai_client.process_document(request)
            document = result.document

            if not document or not document.text:
                self.logger.error('No text extracted from image')
                return ''

            # Extract text from paragraphs
            extracted_text = ''
            for page in document.pages:
                for paragraph in page.paragraphs:
                    # Extract text using text anchors
                    start_index = paragraph.layout.text_anchor.text_segments[0].start_index
                    end_index = paragraph.layout.text_anchor.text_segments[0].end_index
                    extracted_text += document.text[start_index:end_index] + '\n'

            self.logger.info('Document AI processing completed')
            return extracted_text.strip()

        except Exception as e:
            self.logger.error(f"Document AI processing error: {e}")
            raise

    def recognize_multiple(self, image_buffers: List[bytes]) -> List[str]:
        """
        Recognize text from multiple images.

        :param image_buffers: List of image buffers
        :return: List of extracted texts
        """
        try:
            self.logger.info(f'Processing {len(image_buffers)} images')
            results = []
            
            for buffer in image_buffers:
                result = self.recognize(buffer)
                results.append(result)
                
                # Wait 1 minute between recognitions
                time.sleep(60)
            
            return results
        
        except Exception as e:
            self.logger.error(f"Multiple image processing error: {e}")
            raise

    def pdf_to_images(self, pdf_path: str, dpi: int = 300) -> List[bytes]:
        """
        Convert PDF pages to images.

        :param pdf_path: Path to the PDF file
        :param dpi: Resolution of the images (default 300)
        :return: List of image buffers for each PDF page
        """
        try:
            self.logger.info(f'Converting PDF to images: {pdf_path}')
            
            # Convert PDF to images
            images = pdf2image.convert_from_path(pdf_path, dpi=dpi)
            
            # Convert images to byte buffers
            image_buffers = []
            for image in images:
                # Save image to bytes buffer
                buffer = io.BytesIO()
                image.save(buffer, format='JPEG')
                image_buffers.append(buffer.getvalue())
            
            self.logger.info(f'Converted {len(image_buffers)} pages')
            return image_buffers
        
        except Exception as e:
            self.logger.error(f"PDF to image conversion error: {e}")
            raise

    def process_pdf(self, pdf_path: str) -> List[str]:
        """
        Process a PDF by converting to images and recognizing text with delays.

        :param pdf_path: Path to the PDF file
        :return: List of extracted texts for each page
        """
        try:
            # Convert PDF to images
            image_buffers = self.pdf_to_images(pdf_path)
            
            # Recognize text from images with delays
            return self.recognize_multiple(image_buffers)
        
        except Exception as e:
            self.logger.error(f"PDF processing error: {e}")
            raise

# Optional: If you want to create a singleton instance
document_ai_service = None

def get_document_ai_service(config: dict) -> DocumentAiService:
    """
    Get or create a singleton instance of DocumentAiService
    
    :param config: Configuration dictionary
    :return: DocumentAiService instance
    """
    global document_ai_service
    if document_ai_service is None:
        document_ai_service = DocumentAiService(config)
    return document_ai_service# services/document_ai_service.py
