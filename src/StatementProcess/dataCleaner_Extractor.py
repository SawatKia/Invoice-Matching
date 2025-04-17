import os
import json
from typing import List, Dict, Any
from alive_progress import alive_bar

import pandas as pd

from log_utils import get_logger
from .GeminiService import GeminiClient

logger = get_logger()
class TransactionExtractor:
    """Extracts transaction data from OCR text"""
    
    # Regular expression pattern for transaction data in OCR later use if need
    TRANSACTION_PATTERN = r'(\d{2}-\d{2}-\d{2})(?:\s+(\d{2}:\d{2}))?\s+([^0-9\n]+)?(?:\s+((?:\d[\d,.;]*|\.\d{2}|\d{1,3}(?:[,.;]\d{3})*(?:\.\d{2})?)))?(?:\s+((?:\d[\d,.;]*|\.\d{2}|\d{1,3}(?:[,.;]\d{3})*(?:\.\d{2})?)))?\s+(.*?)(?=\d{2}-\d{2}-\d{2}|$)'
    
    @staticmethod
    def extract_with_gemini(extracted_texts: List[str], filename: str) -> List[Dict[str, Any]]:
        """
        Extract transactions from OCR text using Gemini AI
        
        Args:
            extracted_texts: List of text from each page
            filename: Name of the file being processed
                
        Returns:
            List of transaction dictionaries
        """
        logger.info("Extracting transactions using Gemini AI")
        
        # Initialize Gemini client and variables
        gemini_client = GeminiClient()
        all_transactions = []
        checkpoint_file = f"data/files/output/{filename}_transactions_checkpoint.json"
        checkpoint_interval = 5  # Save every 5 pages
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(checkpoint_file), exist_ok=True)
        
        # Load existing transactions if any
        if os.path.exists(checkpoint_file):
            try:
                with open(checkpoint_file, 'r', encoding='utf-8') as f:
                    all_transactions = json.load(f)
                logger.info(f"Loaded {len(all_transactions)} transactions from checkpoint")
                # Get the last processed page
                last_page = max(t.get('page', 0) for t in all_transactions)
                logger.info(f"Resuming from page {last_page + 1}")
            except Exception as e:
                logger.error(f"Error loading checkpoint file: {e}")
                all_transactions = []
        
        # Process pages with progress bar
        with alive_bar(
            total=len(extracted_texts), 
            title=f"detect transactions from {len(extracted_texts)} pages",
            bar='smooth',
            enrich_print=True,
            receipt=True,
            monitor=True,
            elapsed=True,
            stats=True
            ) as bar:
            
            for page_num, page_text in enumerate(extracted_texts, 1):
                # Skip already processed pages
                if any(t.get('page') == page_num for t in all_transactions):
                    logger.debug(f"Skipping already processed page {page_num}")
                    bar()
                    continue
                    
                try:
                    # Generate prompt for this page
                    prompt = f"""
                    Extract financial transactions from this Thai bank statement page.
                    Convert all dates to YYYY-MM-DD format and ensure proper formatting of numbers.
                    Return the result as a JSON array of transactions.
    
                    Text to process:
                    {page_text}
                    """
                    
                    # Get response from Gemini
                    logger.debug(f"Sending request to Gemini for page \x1b[1;38;5;48m{page_num}\x1b[0m")
                    response = gemini_client.generate_content(prompt)
                    
                    # Parse JSON response and add page number
                    try:
                        response = json.loads(response)
                        logger.debug(f"page {page_num} metadata: {response['metadata']}\nnumber of transactions in response: {len(response['transactions'])}")
                        transactions = response['transactions']
                        if not transactions:
                            logger.warning(f"No transactions found in response for page {page_num}")
                            continue
                            
                        # Add page number to each transaction
                        for transaction in transactions:
                            transaction['page'] = page_num
                        all_transactions.extend(transactions)
                        
                        # Save checkpoint after interval
                        if page_num % checkpoint_interval == 0:
                            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                                json.dump(all_transactions, f, ensure_ascii=False, indent=2)
                            logger.info(f"Saved checkpoint after page {page_num}")
                            
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse Gemini response for page {page_num}: {e}")
                        continue
                        
                except Exception as e:
                    logger.error(f"Error processing page {page_num}: {e}")
                    continue
                
                bar()
        
        # Save final results
        try:
            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(all_transactions, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved {len(all_transactions)} transactions to {checkpoint_file}")
        except Exception as e:
            logger.error(f"Error saving final results: {e}")
        
        return all_transactions


class DataCleaner:
    """Cleans and formats transaction data"""
    
    @staticmethod
    def clean_transactions(transactions: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Clean and format transaction data
        
        Args:
            transactions: List of transaction dictionaries
            
        Returns:
            Cleaned DataFrame
        """
        try:
            logger.info("Starting transaction cleaning process")
            df = pd.DataFrame(transactions)
            logger.debug(f"Initial DataFrame shape: {df.shape}")
            logger.debug(f"Initial columns: {df.columns.tolist()}")
            
            # Process datetime first
            df = DataCleaner._process_datetime(df)
            logger.debug(f"Columns after datetime processing: {df.columns.tolist()}")
            
            # Process numeric columns
            for col in ['amount', 'balance']:
                if col in df.columns:
                    if not pd.api.types.is_numeric_dtype(df[col]):
                        logger.info(f"Converting {col} to numeric")
                        df = DataCleaner._clean_numeric_columns(df)
                        
            # Convert deposit boolean to transaction type
            if 'deposit' in df.columns:
                logger.info("Converting deposit column to transaction type")
                df['type'] = df['deposit'].map({True: 'Deposit', False: 'Withdrawal'})
                df = df.drop('deposit', axis=1)
                
            # Verify final columns
            expected_columns = [
                'datetime', 'description', 'type', 'amount', 
                'balance', 'additional_info', 'page'
            ]
            DataCleaner._verify_columns(df, expected_columns)
            
            # Return columns in expected order
            logger.info("Transaction cleaning completed successfully")
            return df[expected_columns]
            
        except Exception as e:
            logger.error(f"Error cleaning transactions: {e}")
            raise

    @staticmethod
    def _verify_columns(df: pd.DataFrame, expected_columns: List[str]) -> None:
        """
        Verify that DataFrame has all expected columns
        
        Args:
            df: DataFrame to verify
            expected_columns: List of expected column names
            
        Raises:
            ValueError: If any expected columns are missing
        """
        logger.debug(f"expected columns: {expected_columns}")
        logger.debug(f"df columns: {df.columns.tolist()}")
        missing_columns = [col for col in expected_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"Missing columns in DataFrame: {missing_columns}")
            raise ValueError(f"DataFrame is missing expected columns: {missing_columns}")
            
    @staticmethod
    def _process_datetime(df: pd.DataFrame) -> pd.DataFrame:
        """
        Process date and time into datetime column
        
        Args:
            df: DataFrame with date and time columns
            
        Returns:
            DataFrame with datetime column
        """
        try:
            # Create a copy to preserve original columns
            result_df = df.copy()
            
            # Find columns with 'date' or 'time' in their names
            date_columns = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
            
            if len(date_columns) >= 2:
                logger.debug(f"Found date/time columns: {date_columns}. Combining into new datetime column.")
                result_df['datetime_processed'] = pd.to_datetime(
                    df[date_columns[0]] + ' ' + df[date_columns[1]], 
                    format='%d-%m-%y %H:%M', 
                    errors='coerce'
                )
                # Only drop original columns if conversion was successful
                if not result_df['datetime_processed'].isna().all():
                    result_df = result_df.drop(date_columns, axis=1)
                    result_df = result_df.rename(columns={'datetime_processed': 'datetime'})
                else:
                    logger.warning("DateTime conversion failed, keeping original columns")
                    result_df = result_df.drop('datetime_processed', axis=1)
                    
            elif len(date_columns) == 1:
                logger.debug(f"Found single datetime column: {date_columns[0]}. Converting to datetime.")
                result_df['datetime_processed'] = pd.to_datetime(df[date_columns[0]], errors='coerce')
                # Only drop original column if conversion was successful
                if not result_df['datetime_processed'].isna().all():
                    result_df = result_df.drop(date_columns, axis=1)
                    result_df = result_df.rename(columns={'datetime_processed': 'datetime'})
                else:
                    logger.warning("DateTime conversion failed, keeping original column")
                    result_df = result_df.drop('datetime_processed', axis=1)
                    
            else:
                # Fallback: Look for a column with 'datetime' in its name
                datetime_column = next((col for col in df.columns if 'datetime' in col.lower()), None)
                if datetime_column:
                    logger.debug(f"Found datetime column: {datetime_column}. Converting in place.")
                    result_df[datetime_column] = pd.to_datetime(df[datetime_column], errors='coerce')
                else:
                    logger.error("No suitable date/time or datetime columns found.")
                    raise ValueError("No suitable date/time or datetime columns found for processing.")
                    
            # Verify datetime column exists and is valid
            if 'datetime' in result_df.columns:
                if result_df['datetime'].isna().all():
                    logger.error("DateTime conversion failed completely")
                    raise ValueError("DateTime conversion failed completely")
                    
            logger.info("Datetime processing completed successfully")
            return result_df
            
        except Exception as e:
            logger.error(f"Error processing datetime: {e}")
            raise

class TransactionVerifier:
    """Handles verification of transaction calculations"""
    
    @staticmethod
    def verify_transactions_by_page(df: pd.DataFrame) -> pd.DataFrame:
        """
        Verify transaction calculations page by page using Balance carry-forward rows
        
        Args:
            df: DataFrame with transactions
            
        Returns:
            DataFrame with verification results
        """
        df = df.copy()
        # Sort by page number and then by datetime
        df = df.sort_values(['page', 'datetime'])
        
        # Add columns for verification
        df['calculated_balance'] = None
        df['needs_review'] = False
        
        # Process each page separately
        pages = df['page'].unique()
        logger.info(f"Verifying transactions across {len(pages)}[{', '.join(map(str, pages))}] pages")
        
        for page in pages:
            page_df = df[df['page'] == page]
            page_indices = page_df.index
            
            # Find carry-forward balance row
            carry_forward_rows = page_df[page_df['additional_info'] == 'Balance carry-forward']
            
            if len(carry_forward_rows) == 0:
                logger.warning(f"No Balance carry-forward row found on page {page}, skipping verification")
                df.loc[page_indices, 'needs_review'] = True
                continue
                
            # Use the first carry-forward row as starting point
            start_idx = carry_forward_rows.index[0]
            start_balance = carry_forward_rows.iloc[0]['balance']
            
            if pd.isna(start_balance):
                logger.warning(f"Invalid starting balance on page {page}, skipping verification")
                df.loc[page_indices, 'needs_review'] = True
                continue
                
            # Set calculated balance for carry-forward row
            df.loc[start_idx, 'calculated_balance'] = start_balance
            
            # Verify subsequent transactions on this page
            current_balance = start_balance
            for idx in page_indices:
                if idx <= start_idx:
                    continue  # Skip rows before or equal to carry-forward
                    
                row = df.loc[idx]
                amount = row['amount']
                
                # Skip rows with missing amount
                if pd.isna(amount):
                    df.loc[idx, 'needs_review'] = True
                    logger.warning(f"Missing amount at row {idx}, marking for review")
                    continue
                    
                # Calculate expected balance
                if row['type'] == 'Deposit':
                    current_balance += amount
                elif row['type'] == 'Withdrawal':  
                    current_balance -= amount
                else:
                    # for balance carry-forward rows, we don't change the balance
                    pass

                # Update calculated balance
                df.loc[idx, 'calculated_balance'] = current_balance
                
                # Check if calculated balance matches recorded balance
                recorded_balance = row['balance']
                if not pd.isna(recorded_balance) and abs(current_balance - recorded_balance) > 0.01:
                    df.loc[idx, 'needs_review'] = True
                    logger.warning(f"Balance discrepancy on page {page}, row {idx}: \n{'        '*11}calculated={current_balance:.2f}, recorded={recorded_balance:.2f}")
        
        # Report summary
        review_count = df['needs_review'].sum()
        if review_count > 0:
            logger.warning(f"Found {review_count} transactions that need human review")
        else:
            logger.info("All transactions verified successfully")
        logger.debug(f"transaction verification results:\n{df[['page', 'datetime', 'amount', 'balance', 'calculated_balance', 'needs_review']].head(50)}")
                
        return df