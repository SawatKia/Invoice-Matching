import os
from pathlib import Path
from log_utils import get_logger
import pandas as pd
from typing import List, Optional

logger = get_logger()
class FileManager:
    """Handles file operations for saving/loading data"""
    
    @staticmethod
    def ensure_directory_exists(file_path: str) -> None:
        """
        Ensure directory exists for file path
        
        Args:
            file_path: Path to file
        """
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    
    @staticmethod
    def save_text(data: str, file_path: str, encoding: str = 'utf-8') -> None:
        """
        Save text data to file
        
        Args:
            data: Text data to save
            file_path: Path to save file
            encoding: File encoding
        """
        FileManager.ensure_directory_exists(file_path)
        with open(file_path, 'w', encoding=encoding) as f:
            f.write(data)
        logger.info(f"Saved text data to {file_path}")
    
    @staticmethod
    def save_dataframe(df: pd.DataFrame, file_path: str) -> None:
        """
        Save DataFrame to CSV, formatting datetime columns to %d-%m-%y
        
        Args:
            df: DataFrame to save
            file_path: Path to save CSV
        """
        FileManager.ensure_directory_exists(file_path)
        
        # Format datetime columns
        for column in df.select_dtypes(include=['datetime']):
            logger.debug(f"Formatting datetime column: {column} to this strftime format: '%d-%m-%y %H:%M'")
            df[column] = df[column].dt.strftime('%d-%m-%y %H:%M')
        
        df.to_csv(file_path, index=False)
        logger.info(f"Saved DataFrame to {file_path}")
    
    @staticmethod
    def load_dataframe(file_path: str) -> Optional[pd.DataFrame]:
        """
        Load DataFrame from CSV if exists
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            DataFrame or None if file doesn't exist
        """
        if Path(file_path).exists():
            logger.info(f"Loading DataFrame from {file_path}")
            return pd.read_csv(file_path)
        return None
    
    @staticmethod
    def save_ocr_results(pages_text: List[str], file_path: str, encoding: str = 'utf-8') -> None:
        """
        Save OCR results to text file with page markers
        
        Args:
            pages_text: List of text for each page
            file_path: Path to save file
            encoding: File encoding
        """
        FileManager.ensure_directory_exists(file_path)
        with open(file_path, 'w', encoding=encoding) as f:
            for page_num, text in enumerate(pages_text, 1):
                f.write(f"\n{'='*50}\nPage {page_num}\n{'='*50}\n\n")
                f.write(f"{text[:500]}...")
                f.write(f"\n...(remaining {len(text[500:])} characters)...\n")
        logger.info(f"Saved OCR results to {file_path}")