import os
import torch
import easyocr
import traceback
import fitz
from pathlib import Path
from datetime import datetime
from typing import List, Any, Optional, Tuple
from PIL import Image

from log_utils import get_logger

logger = get_logger()

class OCRHandler:
    """Handles OCR processing of PDF pages"""
    
    def __init__(self, ocr_engine):
        """Initialize OCR processor with appropriate OCR engine"""
        self.ocr_engine = ocr_engine
    
    def process_page(self, page: fitz.Page) -> Tuple[str, List]:
        """
        Process a single page with OCR
        
        Args:
            page: PDF page object
            
        Returns:
            Tuple of extracted text and raw OCR results
        """
        # Convert page to image
        logger.info(f"Converting PDF page {page.number+1} to image")
        pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        img_array = np.array(img)
        
        # Run OCR
        logger.debug(f"===>>>>Running OCR on page {page.number+1}")
        ocr_result = self.ocr_engine.process_image(img_array)
        
        # Log sample result for debugging
        if ocr_result:
            sample = ocr_result[:min(5, len(ocr_result))]
            
        # Extract text from OCR results
        page_text = ' '.join([text[1] for text in ocr_result])
        
        # Log text sample for debugging
        if page_text:
            logger.debug(f"Text sample for page {page.number+1}: {page_text[:100]}...")
            
        return page_text, ocr_result

class OCRProcessor:
    """
    A robust OCR processor with GPU and multi-language support
    Implemented as a Singleton to ensure only one instance exists
    """
    
    # Class variable to hold the singleton instance
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        """
        Override __new__ to implement the singleton pattern
        """
        if cls._instance is None:
            logger.info("Creating new OCRProcessor instance")
            cls._instance = super(OCRProcessor, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(
        self, 
        languages: List[str] = None, 
        use_gpu: Optional[bool] = None,
    ):
        """
        Initialize OCR processor with specified settings.
        Will only initialize once, subsequent calls will be ignored.
        
        :param languages: List of languages to support (default: ['th', 'en'])
        :param use_gpu: Force GPU usage (None = auto-detect)
        """
        # Only initialize once
        if self._initialized:
            logger.debug("OCRProcessor already initialized, skipping initialization")
            return
            
        # Set languages and GPU settings
        self.languages = languages or ['th', 'en']
        self.process_id = os.getpid()
        
        # Determine GPU usage
        self.use_gpu = self._setup_gpu() if use_gpu is None else use_gpu
        
        # Ensure models directory exists
        self.model_dir = Path('models').resolve()
        self.model_dir.mkdir(exist_ok=True)
        # Initialize OCR reader
        self._init_reader()
        
        # Mark as initialized
        self._initialized = True
        logger.info(f"OCRProcessor initialized with languages: {self.languages}")
    
    def _setup_gpu(self) -> bool:
        """
        Configure GPU settings for optimal performance.
        
        :return: Boolean indicating GPU availability
        """
        try:
            
            if torch.cuda.is_available():
                logger.info(f"CUDA {torch.version.cuda} is available")
                # Get GPU count
                gpu_count = torch.cuda.device_count()
                logger.debug(f"CUDA available with {gpu_count} GPUs")
                if gpu_count == 0:
                    logger.warning("CUDA reports available but no GPUs found")
                    return False
                
                # Select GPU with most free memory
                gpu_id = 0
                if gpu_count > 1:
                    max_free_mem = 0
                    for i in range(gpu_count):
                        torch.cuda.set_device(i)
                        torch.cuda.empty_cache()
                        free_mem = torch.cuda.get_device_properties(i).total_memory - torch.cuda.memory_allocated(i)
                        if free_mem > max_free_mem:
                            max_free_mem = free_mem
                            gpu_id = i
                
                # Configure selected GPU
                torch.cuda.set_device(gpu_id)
                os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
                torch.backends.cudnn.benchmark = True
                
                # Log GPU details
                gpu_name = torch.cuda.get_device_name(gpu_id)
                memory_allocated = torch.cuda.memory_allocated(gpu_id) / (1024 ** 2)
                memory_total = torch.cuda.get_device_properties(gpu_id).total_memory / (1024 ** 2)
                
                logger.info(f"Using GPU {gpu_id}: {gpu_name}")
                logger.info(
                    f"GPU Memory: {memory_allocated:.2f}MB allocated / "
                    f"{memory_total:.2f}MB total"
                )
                
                return True
            else:
                logger.warning("No GPU available - using CPU")
                return False
        
        except Exception as e:
            logger.error(f"Error setting up GPU: {str(e)}")
            return False
    
    def _init_reader(self):
        """
        Initialize EasyOCR reader with proper configuration.
        """
        try:
            logger.info(
                f"Process {self.process_id}: Initializing OCR "
                f"(GPU={self.use_gpu}, languages={self.languages})"
            )
            
            self.reader = easyocr.Reader(
                self.languages,
                gpu=self.use_gpu,
                model_storage_directory=str(self.model_dir),
                download_enabled=True,
                detector=True,
                recognizer=True,
                verbose=False
            )
            
            # Log available characters for debugging
            logger.debug(f"Available characters: {self.reader.lang_char}")
            logger.info(f"Process {self.process_id}: OCR reader initialized successfully")
        
        except Exception as e:
            logger.error(
                f"Process {self.process_id}: Failed to initialize OCR reader: {str(e)}"
            )
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
    
    def process_image(self, img_array, detail: int = 1) -> List[Any]:
        """
        Process an image array with OCR.
        
        :param img_array: NumPy array of the image
        :param detail: Level of detail in OCR result (0=basic, 1=with confidence)
        :return: OCR processing results
        """
        try:
            start_time = datetime.now()
            
            # Perform OCR
            result = self.reader.readtext(img_array, detail=detail, batch_size = 4)
            
            # Calculate processing duration
            duration = (datetime.now() - start_time).total_seconds()
            
            logger.debug(
                f"Process {self.process_id}: OCR processed image in {duration:.2f} seconds"
            )
            
            return result
        
        except Exception as e:
            logger.error(
                f"Process {self.process_id}: OCR processing failed: {str(e)}"
            )
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
    
    @classmethod
    def get_instance(cls, languages: List[str] = None, use_gpu: Optional[bool] = None):
        """
        Get or create the singleton instance of OCRProcessor.
        
        :param languages: List of languages to support (default: ['th', 'en'])
        :param use_gpu: Force GPU usage (None = auto-detect)
        :return: Singleton OCRProcessor instance
        """
        # This will either create a new instance or return the existing one
        return cls(languages=languages, use_gpu=use_gpu)