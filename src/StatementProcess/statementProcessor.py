import os
import psutil
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple, Union
from alive_progress import alive_bar
from pathlib import Path

from Ocr import OCRProcessor, OCRHandler
from .dataCleaner_Extractor import DataCleaner, TransactionExtractor, TransactionVerifier
from .pdf_handling import PDFValidator
from .file_operations import FileManager
from log_utils import get_logger, get_bangkok_time

logger = get_logger()

class StatementProcessor:
    """Processes financial statements with OCR and transaction extraction"""
    
    def __init__(self, file_path: str, password: str = None, languages: List[str] = None):
        """
        Initialize processor with file path and languages
        
        Args:
            file_path: Path to PDF or Excel file
            password: Password for encrypted PDF
            languages: List of languages for OCR
        """
        self.file_path = file_path
        self.file_extension = os.path.splitext(file_path)[1].lower()
        
        if self.file_extension in ['.pdf']:
            self.pdf_validator = PDFValidator(file_path, password)
            self.pdf_path = self.pdf_validator.pdf_path
            self.languages = languages or ['th', 'en']
            self.ocr_engine = OCRProcessor.get_instance(languages=self.languages, use_gpu=True)
            self.ocr_handler = OCRHandler(self.ocr_engine)
            self.total_pages = self.pdf_validator.page_count
            self.document = self.pdf_validator.document
            logger.info(f"Initialized for PDF: {self.pdf_path} with {self.total_pages} pages")
        elif self.file_extension in ['.xlsx', '.xls']:
            self.excel_path = file_path
            self.excel_df = self._initialize_excel()
            logger.info(f"Initialized for Excel: {self.excel_path}")
        else:
            logger.error(f"Unsupported file type: {self.file_extension}. Supported types: .pdf, .xlsx, .xls")
            raise ValueError(f"Unsupported file type: {self.file_extension}. Supported types: .pdf, .xlsx, .xls")
           

    def _initialize_excel(self) -> pd.DataFrame:
        try:
            if not os.path.exists(self.excel_path):
                logger.error(f"Excel file not found: {self.excel_path}")
                raise FileNotFoundError(f"Excel file not found: {self.excel_path}")
            df = pd.read_excel(self.excel_path, engine='openpyxl')
            if df.empty:
                logger.error(f"Excel file is empty: {self.excel_path}")
                raise ValueError(f"Excel file is empty: {self.excel_path}")
            logger.info(f"Excel file loaded successfully: {self.excel_path}")
            logger.debug(f"Excel file info: \n")
            df.info()
            logger.debug(f"Excel file head: \n{df.head(5)}")
            df = self.process_excel_statement(df)
            return df
        except Exception as e:
            logger.error(f"Error loading Excel file: {e}")
            raise
    
    def process_excel_statement(self, df: pd.DataFrame) -> pd.DataFrame:
         """
         Process statement from Excel file
        
         Returns:
             DataFrame with standardized transaction data
         """
         logger.info(f"Processing Excel statement with {len(df)} rows")
         logger.debug(f"Excel file info: \n")
         df.info()

         try:
             logger.debug(f"Excel file columns: {df.columns.tolist()}")

             required_columns = ['วันที่','เวลา','ถอน','ฝาก','คงเหลือ','หน้าที่']
             missing_columns = [col for col in required_columns if col not in df.columns]

             if missing_columns:
                    logger.error(f"Missing required columns in Excel file: {missing_columns}")
                    raise ValueError(f"Missing required columns in Excel file: {missing_columns}")
             logger.info(f"All required columns(TH) are present in the Excel file")
             # Clean and standardize data
             df = self._standardize_excel_data(df)
             
             logger.info(f"Successfully processed Excel statement with {len(df)} transactions")
             return df
            
         except Exception as e:
            logger.error(f"Error processing Excel statement: {e}")
            raise
    
    def _standardize_excel_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize Excel data to match transaction format
        
        Args:
            df: Raw Excel DataFrame
            
        Returns:
            Standardized DataFrame
        """
        # Create a copy to avoid modifying the original
        standardized_df = df.copy()
        
        # Define column name mappings (Thai to English)
        column_mappings = {
            'วันที่': 'date',
            'เวลา': 'time', 
            'ถอน': 'withdrawal',
            'ฝาก': 'deposit',
            'คงเหลือ': 'balance',
            'หน้าที่': 'page'
        }
        
        # Rename columns to English
        standardized_df = standardized_df.rename(columns=column_mappings)
        standardized_df = standardized_df[['date', 'time', 'withdrawal', 'deposit', 'balance', 'page']]
        logger.debug(f"Columns after renamed to English: {standardized_df.columns.tolist()}")
        
        # Process date column - ensure dd-mm-yy format
        logger.info("Standardizing date format to dd-mm-yy")
        if pd.api.types.is_datetime64_any_dtype(standardized_df['date']):
            standardized_df['date'] = standardized_df['date'].dt.strftime('%d-%m-%y')
        else:
            try:
                standardized_df['date'] = pd.to_datetime(standardized_df['date']).dt.strftime('%d-%m-%y')
            except Exception as e:
                logger.warning(f"Could not standardize date format: {e}")
        
        # Process time column - ensure HH:MM format, default to 0:00 if missing
        logger.info("Standardizing time format to HH:MM")
        if 'time' not in standardized_df.columns or standardized_df['time'].isnull().all():
            logger.debug("Time column is missing or all values are null, setting default time to 0:00")
            standardized_df['time'] = '0:00'
        else:
            unique_times = standardized_df['time'].unique()
            logger.debug(f"Original time values: {', '.join(map(str, unique_times[:10]))}... (and {len(unique_times) - 10} more from {len(unique_times)} total)")
            
            try:
                # First try to parse as HH:MM:SS
                standardized_df['time'] = pd.to_datetime(
                    standardized_df['time'], 
                    format='%H:%M:%S', 
                    errors='coerce'
                )
                
                # If any values failed, try HH:MM format
                mask = standardized_df['time'].isna()
                if mask.any():
                    logger.info("Some times not in HH:MM:SS format, trying HH:MM format")
                    logger.debug(f"Failed time values: {', '.join(map(str, standardized_df.loc[mask, 'time'].unique()))}")
                    # Attempt to parse HH:MM format
                    standardized_df.loc[mask, 'time'] = pd.to_datetime(
                        standardized_df.loc[mask, 'time'],
                        format='%H:%M',
                        errors='coerce'
                    )
                
                # Convert to HH:MM string format
                standardized_df['time'] = standardized_df['time'].dt.strftime('%H:%M')
                standardized_df['time'].fillna('0:00', inplace=True)
                
                unique_times = standardized_df['time'].unique()
                logger.debug(f"Standardized time values: {', '.join(map(str, unique_times[:10]))}... (and {len(unique_times) - 10} more from {len(unique_times)} total)")
                
            except Exception as e:
                logger.warning(f"Could not standardize time format: {e}")
                standardized_df['time'].fillna('0:00', inplace=True)
        
        # Ensure monetary columns are numeric
        for col in ['withdrawal', 'deposit', 'balance']:
            logger.info(f"Converting {col} to numeric format")
            standardized_df[col] = pd.to_numeric(standardized_df[col], errors='coerce').fillna(0)
        
        # Ensure page column is integer
        logger.info("Standardizing page column")
        standardized_df['page'] = pd.to_numeric(standardized_df['page'], errors='coerce').fillna(1).astype(int)
        
        pd.set_option('display.max_rows', None)  # Show all rows
        logger.debug(f"Standardized dataframe sample:\n{standardized_df.head(10)}")

        logger.info("Checking for missing values in date and time columns")
        # Log rows with NaN before filling
        nan_rows = standardized_df[standardized_df['date'].isna() | standardized_df['time'].isna()]
        if not nan_rows.empty:
            logger.warning(f"Found {len(nan_rows)} rows with missing date/time:")
            logger.warning(f"Sample of problematic rows:\n{nan_rows.head(50)}")

        # Create datetime column by combining date and time
        standardized_df['datetime'] = pd.to_datetime(
            standardized_df['date'].astype(str) + ' ' + standardized_df['time'],
            format='%d-%m-%y %H:%M',
            errors='raise'
        )
        logger.debug(f"Datetime column created with values (date and time): \n{standardized_df['datetime'].dt.strftime('%d-%m-%y %H:%M').head(10)}")
        
        # Reorder columns to match expected format
        column_order = [
            'datetime', 'withdrawal', 
            'deposit', 'balance', 'page'
        ]
        standardized_df = standardized_df[column_order]
        
        logger.debug(f"Standardized Excel data example:\n{standardized_df.head(10)}")
        return standardized_df
        
    def _get_memory_usage(self) -> str:
        """
        Get current memory usage in MB
        
        Returns:
            String with memory usage
        """
        process = psutil.Process()
        return f"{process.memory_info().rss / (1024 * 1024):.1f} MB"

    def _process_page(self, page_num: int) -> Dict[str, Any]:
        """
        Process a single PDF page
        
        Args:
            page_num: Page number to process
            
        Returns:
            Dictionary with page processing info
        """
        page_info = {
            'page_num': page_num,
            'status': 'started',
            'start_time': get_bangkok_time(),
            'text': None,
            'error': None
        }
        
        try:
            # Use stored document instead of reopening the file
            page_data = self.document[page_num]
                
            # Process page with OCR
            page_text, ocr_result = self.ocr_handler.process_page(page_data)
                
            # Update page info
            page_info.update({
                'status': 'completed',
                'end_time': get_bangkok_time(),
                'text': page_text,
                'text_length': len(page_text),
                'duration': get_bangkok_time() - page_info['start_time']
            })
        except Exception as e:
            page_info = self._handle_page_error(page_info, e)
            
        return page_info
    
    def _handle_page_error(self, page_info: Dict[str, Any], error: Exception) -> Dict[str, Any]:
        """
        Handle errors during page processing
        
        Args:
            page_info: Current page info dictionary
            error: Exception that occurred
            
        Returns:
            Updated page info with error details
        """
        page_info.update({
            'status': 'failed',
            'end_time': get_bangkok_time(),
            'error': str(error),
            'error_type': type(error).__name__,
            'duration': get_bangkok_time() - page_info['start_time']
        })
        logger.error(f"Error processing page {page_info['page_num']+1}: {error}")
        return page_info

    def _calculate_eta(self, completed_pages: int, avg_duration: float) -> str:
        """
        Calculate estimated time of completion
        
        Args:
            completed_pages: Number of completed pages
            avg_duration: Average duration per page
            
        Returns:
            ETA string
        """
        remaining_pages = self.total_pages - completed_pages
        remaining_seconds = remaining_pages * avg_duration
        eta = datetime.now() + timedelta(seconds=remaining_seconds)
        return eta.strftime('%H:%M:%S')

    def process_statement(self, force_ocr: bool = False) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        """
        Process statement with direct text extraction, OCR, or Excel import
        
        Args:
            force_ocr: Force OCR even if text can be extracted directly
            
        Returns:
            Tuple of DataFrame with extracted texts and processing results
        """
        # Handle Excel files differently
        if hasattr(self, 'excel_path'):
            logger.info("no need to process Excel file with OCR")
            # Return in same format as PDF processing but with empty results list
            return pd.DataFrame({'text': ['Excel file processed']}), []
        
        # Check for cached OCR results
        cache_path = f"{os.path.splitext(str(self.pdf_path))[0]}_ocr.csv"
        
        if not force_ocr and (df := FileManager.load_dataframe(cache_path)) is not None:
            logger.info("Using cached OCR results")
            # Log truncated text results for debugging, but keep the original intact
            truncated_df = df.copy()
            truncated_df['text'] = truncated_df['text'].apply(lambda x: x[:50] + '...' if len(x) > 50 else x)
            logger.debug(f"Sample cached OCR results: {truncated_df.head(5).to_dict()}...(more {len(df)-5} rows)")
            # Return cached results
            return df, []
        
        # Check if text is directly extractable
        if not force_ocr and self.pdf_validator.is_extractable():
            logger.info("Text can be extracted directly from PDF, skipping OCR")
            return self._extract_text_directly()
        
        # Fallback to OCR processing
        logger.info("Text cannot be extracted directly, using OCR")
        return self._process_pdf_with_ocr()

    def _extract_text_directly(self) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        """
        Extract text directly from PDF without OCR
        
        Returns:
            Tuple of DataFrame with extracted texts and processing results
        """
        start_time = get_bangkok_time()
        logger.info(f"Starting direct text extraction at {start_time} for {self.total_pages} pages")
        
        results = []
        extracted_texts = []
        
        with alive_bar(
            total=self.total_pages, 
            title=f"Extracting text from {self.total_pages} pages",
            bar='smooth',
            enrich_print=True,
            receipt=True,
            ) as bar:
            
            for page_num in range(self.total_pages):
                page_info = {
                    'page_num': page_num,
                    'status': 'started',
                    'start_time': get_bangkok_time(),
                    'text': None,
                    'error': None
                }
                
                try:
                    page = self.document[page_num]
                    text = page.get_text()
                    
                    page_info.update({
                        'status': 'completed',
                        'end_time': get_bangkok_time(),
                        'text': text,
                        'text_length': len(text),
                        'duration': get_bangkok_time() - page_info['start_time']
                    })
                    
                    extracted_texts.append({'page_number': page_num + 1, 'text': text})
                    
                except Exception as e:
                    page_info = self._handle_page_error(page_info, e)
                
                results.append(page_info)
                bar()
        
        # Log summary
        self._log_processing_summary(results, start_time)
        
        # Cache results for future use
        extracted_texts_df = pd.DataFrame(extracted_texts)
        self._cache_ocr_results(extracted_texts_df['text'].tolist())
        
        return extracted_texts_df, results
    
    def _process_pdf_with_ocr(self) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        """
        Process PDF with OCR and detailed tracking
        
        Returns:
            Tuple of DataFrame with extracted texts and processing results
        """
        start_time = get_bangkok_time()
        logger.info(f"Starting OCR processing at {start_time} for {self.total_pages} pages")
        
        results = []
        extracted_texts = []
        durations = []
        
        with alive_bar(
            total=self.total_pages, 
            title=f"Processing {self.total_pages} pages",
            bar='smooth',
            enrich_print=True,
            receipt=True,
            monitor=True,
            elapsed=True,
            stats=True,
            ) as bar:
            
            for page_num in range(self.total_pages):
                # Process current page
                result = self._process_page(page_num)
                results.append(result)
                
                if result['text']:
                    extracted_texts.append({'page_number': page_num + 1, 'text': result['text']})
                
                # Update statistics for progress tracking
                if result['status'] == 'completed':
                    durations.append(result['duration'].total_seconds())
                    avg_duration = sum(durations) / len(durations)
                    eta = self._calculate_eta(page_num + 1, avg_duration)
                    
                    # Log detailed progress
                    logger.info(
                        f"Progress: {page_num+1}/{self.total_pages} pages | "
                        f"Avg: {avg_duration:.2f}s/page | "
                        f"ETA: {eta} | "
                        f"Memory: {self._get_memory_usage()}"
                    )
                
                bar()
                
        # Log summary
        self._log_processing_summary(results, start_time)
        
        # Cache results for future use
        extracted_texts_df = pd.DataFrame(extracted_texts)
        self._cache_ocr_results(extracted_texts_df['text'].tolist())

        return extracted_texts_df, results
    
    def _log_processing_summary(self, results: List[Dict[str, Any]], start_time: datetime) -> None:
        """
        Log summary of processing results
        
        Args:
            results: List of page processing results
            start_time: Processing start time
        """
        success_count = sum(1 for r in results if r['status'] == 'completed')
        success_rate = success_count / len(results) * 100
        total_duration = get_bangkok_time() - start_time
        
        logger.info(
            f"Processing completed - Success: {success_count}/{len(results)} "
            f"({success_rate:.1f}%), Duration: {total_duration}, "
            f"Final memory: {self._get_memory_usage()}"
        )
    
    def _cache_ocr_results(self, extracted_texts: List[str]) -> None:
        """
        Cache OCR results for future use
        
        Args:
            extracted_texts: List of extracted texts
        """
        cache_path = f"{os.path.splitext(str(self.pdf_path))[0]}_ocr.csv"
        df = pd.DataFrame({'text': extracted_texts})
        FileManager.save_dataframe(df, cache_path)

    def extract_transactions(self, extracted_texts: List[str]) -> pd.DataFrame:
        """
        Coordinate transaction extraction and cleaning from OCR text or return Excel data
        
        Args:
            extracted_texts: List of text from each page (for PDF)
                    
        Returns:
            DataFrame of cleaned transactions
        """
        # If this is an Excel file, we've already processed the transactions
        if hasattr(self, 'excel_path'):
            logger.info("Returning pre-processed Excel transactions")
            return self.excel_df
        
        logger.info("Starting transaction extraction process")
        logger.debug(f"Processing {len(extracted_texts)} pages of OCR text")
        
        # Initialize Gemini client and process pages
        transactions = TransactionExtractor.extract_with_gemini(extracted_texts, Path(self.pdf_path).stem)
        
        # Clean and format transactions
        cleaned_df = DataCleaner.clean_transactions(transactions)

        # Add page number to transactions if not already present
        if 'page' not in cleaned_df.columns:
            logger.warning("Page number not present in transactions, adding placeholder")
            cleaned_df['page'] = 1  # Default to page 1
            logger.debug(f'df.info():\n')
            cleaned_df.info()

        # Verify transaction balances page by page
        logger.info("Verifying transaction balances page by page")
        cleaned_df = TransactionVerifier.verify_transactions_by_page(cleaned_df)
        
        logger.info(f"Extracted {len(cleaned_df)} transactions")
        return cleaned_df

    def save_data(self, data: Union[List[str], str, pd.DataFrame], 
                  file_path: str, data_type: str = 'text') -> None:
        """
        Save data with appropriate format
        
        Args:
            data: Data to save (text, list of texts, or DataFrame)
            file_path: Path to save file
            data_type: Type of data ('text', 'ocr', or 'dataframe')
        """
        try:
            if data_type == 'text':
                if isinstance(data, list):
                    text_data = '\n\n'.join(data)
                else:
                    text_data = data
                FileManager.save_text(text_data, file_path)
                
            elif data_type == 'ocr':
                FileManager.save_ocr_results(data, file_path)
                
            elif data_type == 'dataframe':
                FileManager.save_dataframe(data, file_path)
                
            else:
                raise ValueError("Invalid data_type. Must be 'text', 'ocr', or 'dataframe'.")
                
        except Exception as e:
            logger.error(f"Error saving data: {e}")
            raise

