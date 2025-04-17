import os
import time
from typing import Tuple, Dict
from pathlib import Path
import pandas as pd

from BillingProcess import BillingProcessor
from StatementProcess import StatementProcessor, DataReviewer, GeminiClient
from Ocr import OCRProcessor
from PaymentMatching import PaymentValidator, PaymentMatcher, PaymentAnalyzer

from log_utils import get_logger

logger = get_logger()

CONFIG = {
    'billing_file': 'data/files/billings/ขาย1-9-ทรัพย์.xlsx',
    # PDF statement examples
    # 'statement_file': 'data/files/statements/20250319195146_merged.pdf',
    # 'statement_password': None, 
    # PDF statement examples with password
    # 'statement_file': 'data/files/statements/STM_SA6066_01JAN25_13MAR25.pdf',
    # 'statement_password': '14032000',  # Set to actual password when needed
    # Excel statement examples
    'statement_file': 'data/files/statements/20250319195146_merged_manual.xlsx',
    'output_dir': 'data/files/output',
    'credit_days': 30,  # Maximum days between invoice and payment
    'tolerance_percent': 0.1,  # 10% tolerance for payment amount differences
}

def initialize_services() -> None:
    """Initialize services"""
    # Initialize OCR processor
    logger.info("Initializing OCR processor")
    ocr_processor = OCRProcessor()
    logger.info("OCR processor initialized")


    logger.info("Initializing Gemini client")
    # Initialize Gemini client
    gemini_client = GeminiClient()
    logger.info("Gemini client initialized")
    
    # Log initialization
    logger.info("All Services initialized successfully")

def process_billing_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Process billing data"""
    processor = BillingProcessor(CONFIG['billing_file'])
    sell_df = processor.process_sell_billing()
    paid_df = processor.process_paid_billing()
    
    return sell_df, paid_df

def process_statement_data() -> pd.DataFrame:
    """Process statement data with cached OCR if available"""
    # Output paths
    output_dir = Path(CONFIG['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    statement_file = Path(CONFIG['statement_file'])
    file_extension = statement_file.suffix.lower()
    
    statement_df_path = output_dir / f"{statement_file.stem}_statement_df.csv"
    
    # Create processor
    processor = StatementProcessor(
        CONFIG['statement_file'],
        CONFIG['statement_password'] if 'statement_password' in CONFIG else None,
        )
    
    # Check if cached data exists
    if statement_df_path.exists():
        logger.info(f"Using cached statement data from {statement_df_path}")
        statement_df = pd.read_csv(statement_df_path)
        reviewer = DataReviewer(statement_df)
        statement_df = reviewer.review_data()
        return statement_df
    
    # Process based on file type
    if file_extension in ['.xlsx', '.xls']:
        logger.info(f"Processing Excel statement file: {statement_file}")
        statement_df = processor.extract_transactions("no needs to process")
    else:
        # Process from PDF
        df_text, _ = processor.process_statement()
        raw_text = df_text['text'].to_list()
        statement_df = processor.extract_transactions(raw_text)
    
    # Review extracted data
    reviewer = DataReviewer(statement_df)
    statement_df = reviewer.review_data()
    
    # Save processed data
    processor.save_data(statement_df, str(statement_df_path), data_type='dataframe')
    
    return statement_df

def match_payments(sale_df: pd.DataFrame, paid_df: pd.DataFrame, statement_df: pd.DataFrame) -> None:
    """
    Match deposits in the statement with invoices and payments from sale and paid dataframes.
    
    Args:
        sale_df: DataFrame with sale/invoice information
        paid_df: DataFrame with payment information
        statement_df: DataFrame with bank statement information
    """
    output_dir = Path(CONFIG['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Starting payment matching process")
    
    # Initialize the payment matcher
    matcher = PaymentMatcher(
        sale_df, 
        paid_df, 
        statement_df, 
        credit_days=CONFIG['credit_days'],
        tolerance_percent=CONFIG['tolerance_percent']
    )
    
    # Match payments and generate report
    start_time = time.time()
    matches = matcher.match_payments()
    end_time = time.time()
    
    logger.info(f"Found {len(matches)} potential matches in {end_time - start_time:.2f} seconds")
    
    # Ensure "payment_matching_report" subfolder exists
    payment_matching_report_dir = output_dir / "payment_matching_report"
    payment_matching_report_dir.mkdir(parents=True, exist_ok=True)

    # Generate and save matching report
    report_path = payment_matching_report_dir / "payment_matches.csv"
    matcher.save_report(str(report_path))
    

    # Validate matches
    validator = PaymentValidator(matcher)
    validation_path = payment_matching_report_dir / "payment_validation.csv"
    validator.save_validation_report(str(validation_path))
    
    # Analyze payment patterns
    analyzer = PaymentAnalyzer(matcher)
    company_analysis = analyzer.analyze_company_payments()
    analysis_path = payment_matching_report_dir / "company_payment_analysis.csv"
    
    if not company_analysis.empty:
        company_analysis.to_csv(str(analysis_path), index=False)
        logger.info(f"Company payment analysis saved to {analysis_path}")
    
    # Find unmatched invoices
    unmatched_invoices = analyzer.find_unmatched_invoices()
    unmatched_path = payment_matching_report_dir / "unmatched_invoices.csv"
    
    if not unmatched_invoices.empty:
        unmatched_invoices.to_csv(str(unmatched_path), index=False)
        logger.info(f"Found {len(unmatched_invoices)} unmatched invoices, saved to {unmatched_path}")
    
    logger.info("Payment matching process completed")
    
    # Print summary statistics
    report_df = matcher.generate_report()
    status_counts = report_df['status'].value_counts()
    
    logger.info("Payment Matching Summary:")
    for status, count in status_counts.items():
        logger.info(f"  {status}: {count}")
    
    total_deposits = statement_df[statement_df['deposit'] > 0]['deposit'].sum()
    matched_deposits = report_df[report_df['status'] == 'Matched']['deposit_amount'].sum()
    
    logger.info(f"Total deposit amount: {total_deposits:.2f}")
    logger.info(f"Matched deposit amount: {matched_deposits:.2f}")
    logger.info(f"Match rate: {(matched_deposits / total_deposits * 100):.2f}%")

def main():
    """Main processing function"""
    try:
        start_time = time.time()
        logger.info("Starting main processing")
        # Initialize services
        initialize_services()

        # Process billing data
        sale_df, paid_df = process_billing_data()
        
        # Process statement data
        statement_df = process_statement_data()

        logger.info("Preparing input data for processing:")
        logger.debug(f"sale DataFrame: \n{sale_df.head(10)}")
        sale_df.info()
        logger.debug(f"Paid DataFrame: \n{paid_df.head(10)}")
        paid_df.info()
        logger.debug(f"Statement DataFrame: \n{statement_df.head(10)}\n{'='*50}")
        statement_df.info()

        # Match payments with invoices
        match_payments(sale_df, paid_df, statement_df)



        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.info(f"Processing completed successfully in \x1b[1;38;5;226m%.2f\x1b[0m seconds", elapsed_time)
        
    except Exception as e:
        logger.error(f"Error in main processing: {e}")
        raise

if __name__ == "__main__":
    main()