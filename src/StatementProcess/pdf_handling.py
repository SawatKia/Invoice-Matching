import fitz  # from PyNuPDF
from pathlib import Path
from log_utils import get_logger

logger = get_logger()

class PDFValidator:
    """Handles PDF file validation and document management"""

    def __init__(self, pdf_path: str, password: str = None):
        """
        Initialize PDFValidator with PDF path and optional password
        
        Args:
            pdf_path: Path to PDF file
            password: Password for encrypted PDF
        """
        self.pdf_path = self.validate_path(pdf_path)
        self.password = password
        self._document = None  # Use private attribute for document
        self.page_count = self.get_page_count()
    
    @property
    def document(self):
        """
        Property to ensure document is always open when accessed
        
        Returns:
            Open PDF document
        """
        if self._document is None or self._document.is_closed:
            self._open_document()
        return self._document
    
    def validate_path(self, pdf_path: str) -> Path:
        """
        Validate PDF file exists and has .pdf extension
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Resolved Path object
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file doesn't have .pdf extension
        """
        try:
            # Convert to Path and resolve
            path = Path(pdf_path).resolve()
            
            # Check file exists
            if not path.exists():
                logger.error(f"PDF file not found: {path}")
                raise FileNotFoundError(f"PDF file not found: {path}")
                
            # Check file extension
            if path.suffix.lower() != '.pdf':
                logger.error(f"File is not a PDF: {path}")
                raise ValueError(f"File must have .pdf extension: {path}")
                
            logger.debug(f"Validated PDF path: {path}")
            return path
            
        except Exception as e:
            logger.error(f"Error validating PDF path: {e}")
            raise

    def _open_document(self) -> None:
        """
        Open PDF document with proper authentication
        """
        try:
            doc = fitz.open(str(self.pdf_path))
            
            if doc.needs_pass:
                logger.info("PDF requires password")
                if not self.password:
                    logger.error("Password required to open PDF but not provided")
                    raise ValueError("Password required to open PDF")
                    
                if not doc.authenticate(self.password):
                    logger.error("Invalid password provided for PDF")
                    raise ValueError("Invalid password provided")
                    
                logger.info("PDF successfully authenticated with password")
                
            self._document = doc
            logger.debug(f"PDF metadata: {self._document.metadata}")
            
        except Exception as e:
            logger.error(f"Error opening PDF: {e}")
            raise
    
    def get_page_count(self) -> int:
        """
        Get total number of pages in PDF
        
        Returns:
            Number of pages in PDF
        """
        logger.info(
            f"Getting page count for PDF: {self.pdf_path}, "
            f"password: {'provided' if self.password else 'not provided'}"
        )
        
        # Ensure document is open
        doc = self.document
        page_count = doc.page_count
        
        logger.info(f"Total pages in PDF: {page_count}")
        return page_count
    
    def is_extractable(self) -> bool:
        """
        Check if text can be directly extracted from PDF
        
        Returns:
            Boolean indicating if text is directly extractable
        """
        logger.info(f"Checking if text is directly extractable from PDF: {self.pdf_path}")
        
        # Ensure document is open
        doc = self.document
        sample_pages = min(3, doc.page_count)
        extractable = False
        
        for page_num in range(sample_pages):
            page = doc[page_num]
            text = page.get_text()
            # If we get substantial text, consider it extractable
            if len(text.strip()) > 50:  # Arbitrary threshold
                extractable = True
                logger.debug(f"Extractable text found on page {page_num + 1}: {text}...")
                break
        
        logger.info(f"PDF text extractability check: {'extractable' if extractable else 'not extractable'}")
        return extractable
    
    def __del__(self):
        """Cleanup method to ensure document is closed"""
        if hasattr(self, '_document') and self._document is not None:
            self._document.close()
            self._document = None