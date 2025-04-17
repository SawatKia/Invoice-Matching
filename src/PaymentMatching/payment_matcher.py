"""
Payment Matcher Module

This module contains classes and methods to match deposits in bank statements 
with invoices and payments in sale and paid dataframes.
"""

from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Set, Optional

from log_utils import get_logger

logger = get_logger()

class PaymentMatcher:
    """
    Class for matching deposits with invoices and companies.
    - Preprocesses the sale, paid, and statement dataframes for easier matching
    - Implements algorithms to match deposits with invoices and companies
    - Handles multiple invoices paid in a single transaction
    - Accounts for payment timing (using a configurable credit window)
    - Handles slight differences in payment amounts due to fees
    """
    
    def __init__(self, sale_df: pd.DataFrame, paid_df: pd.DataFrame, statement_df: pd.DataFrame,
                 credit_days: int = 30, tolerance_percent: float = 0.02):
        """
        Initialize the PaymentMatcher.
        
        Args:
            sale_df: DataFrame with sale/invoice information
            paid_df: DataFrame with payment information
            statement_df: DataFrame with bank statement information
            credit_days: Maximum number of days between invoice and payment
            tolerance_percent: Tolerance for payment amount differences as a percentage
        """
        self.sale_df = self._preprocess_sale_df(sale_df)
        self.paid_df = self._preprocess_paid_df(paid_df)
        self.statement_df = self._preprocess_statement_df(statement_df)
        self.credit_days = credit_days
        self.tolerance_percent = tolerance_percent
        self.results = []
        logger.info("PaymentMatcher initialized with credit_days: %d, tolerance_percent: %.2f",
                    credit_days, tolerance_percent)
        
    def _preprocess_sale_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the sale dataframe.
        
        Args:
            df: Raw sale dataframe
            
        Returns:
            Preprocessed sale dataframe
        """
        # Create a copy to avoid modifying the original
        sale_df = df.copy()
        
        # Convert date to datetime
        sale_df['invoice_date'] = pd.to_datetime(sale_df['วัน/เดือน/ปี'], format='%d/%m/%y', errors='coerce')
        
        # Clean up the dataframe - drop rows with invalid or zero values
        sale_df = sale_df[sale_df['มูลค่าสินค้า'] > 0]
        
        # Rename all Thai columns to English
        sale_df = sale_df.rename(columns={
            'ลำดับ': 'order_number',
            'วัน/เดือน/ปี': 'invoice_date_str',
            'เลขที่ใบกำกับภาษี': 'invoice_number',
            'ชื่อผู้ซื้อสินค้า/ผู้รับบริการ': 'company_name',
            'มูลค่าสินค้า': 'product_value',
            'ภาษีมูลค่าเพิ่ม': 'vat',
            'จำนวนเงิน': 'total_amount',
            'หัก 3%': 'withholding_tax',
            'คงเหลือ': 'net_amount'
        })
        
        # Mark all invoices as unmatched initially
        sale_df['matched'] = False
        
        return sale_df
    
    def _preprocess_paid_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the paid dataframe.
        
        Args:
            df: Raw paid dataframe
            
        Returns:
            Preprocessed paid dataframe
        """
        # Create a copy to avoid modifying the original
        paid_df = df.copy()
        
        # Convert date to datetime
        paid_df['paid_date'] = pd.to_datetime(paid_df['ว/ด/ป'], format='%d/%m/%y', errors='coerce')
        
        # Rename all Thai columns to English
        paid_df = paid_df.rename(columns={
            'ว/ด/ป': 'paid_date_str',
            'ชื่อและที่อยู่ผู้หักภาษี': 'company_name',
            'เลขประจำตัวผู้เสียภาษี': 'tax_id',
            'จำนวนเงิน': 'amount',
            'หัก ณ ที่จ่าย': 'withholding_tax',
            'ยอดโอน': 'transfer_amount'
        })
        
        # Mark all paid records as unmatched initially
        paid_df['matched'] = False
        
        return paid_df
    
    def _preprocess_statement_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the statement dataframe.
        
        Args:
            df: Raw statement dataframe
            
        Returns:
            Preprocessed statement dataframe
        """
        # Create a copy to avoid modifying the original
        statement_df = df.copy()
        
        # Convert date to datetime
        statement_df['deposit_date'] = pd.to_datetime(statement_df['datetime'], format='%d-%m-%y %H:%M', errors='coerce')
        
        # Keep only deposits (positive amounts)
        statement_df = statement_df[statement_df['deposit'] > 0]
        
        return statement_df
    
    def _is_amount_match(self, amount1: float, amount2: float) -> bool:
        """
        Check if two amounts match within the tolerance.
        
        Args:
            amount1: First amount
            amount2: Second amount
            
        Returns:
            True if amounts match within tolerance, False otherwise
        """
        if amount1 == 0 or amount2 == 0:
            logger.debug(f"Amount match failed: one of the amounts is zero (amount1: {amount1}, amount2: {amount2})")
            return False
            
        diff_percent = abs(amount1 - amount2) / max(amount1, amount2)
        is_match = diff_percent <= self.tolerance_percent
        logger.debug(f"Amount match check: amount1={amount1}, amount2={amount2}, "
                 f"diff_percent={diff_percent:.4f}, tolerance={self.tolerance_percent}, result={'' if is_match else 'not '}matched")
        return is_match
    
    def _find_invoice_combinations_with_backtracking(self, invoices_df: pd.DataFrame, 
                                               target_amount: float, 
                                               tolerance_percent: float) -> List[List[dict]]:
        """
        Find combinations of invoices that sum to target_amount within tolerance using backtracking.
        This is more comprehensive than the simple approach in _find_invoice_combinations.
        
        Args:
            invoices_df: DataFrame of available invoices
            target_amount: Target amount to match
            tolerance_percent: Tolerance percentage for matching
            
        Returns:
            List of invoice combinations that match the target amount
        """
        invoices = invoices_df.to_dict('records')
        
        # Define a recursive helper function for backtracking
        def backtrack(start_idx, current_sum, current_combination):
            # Check if we've found a valid combination
            if self._is_amount_match(current_sum, target_amount):
                return [current_combination[:]]  # Return a copy of the current combination
            
            # If we've gone too far over the target, stop this branch
            if current_sum > target_amount * (1 + tolerance_percent):
                return []
            
            # If we've reached the end of the invoices, stop this branch
            if start_idx >= len(invoices):
                return []
            
            results = []
            
            # Try adding each remaining invoice
            for i in range(start_idx, len(invoices)):
                # Add this invoice to our combination
                current_combination.append(invoices[i])
                current_sum += invoices[i]['net_amount']
                
                # Recursively find combinations with this invoice included
                new_results = backtrack(i + 1, current_sum, current_combination)
                if new_results:
                    results.extend(new_results)
                    # If we found a match, no need to try more combinations
                    break
                
                # Backtrack: remove this invoice and try the next one
                current_sum -= invoices[i]['net_amount']
                current_combination.pop()
                
                # For efficiency, limit combinations to max 5 invoices
                if len(current_combination) >= 5:
                    break
            
            return results
        
        # Try each invoice as a starting point
        all_combinations = []
        for i in range(len(invoices)):
            # Simple case: single invoice match
            if self._is_amount_match(invoices[i]['net_amount'], target_amount):
                return [[invoices[i]]]
            
            # Try combinations starting with this invoice
            combos = backtrack(i, 0, [])
            all_combinations.extend(combos)
            
            # If we found some combinations, no need to try more starting points
            if all_combinations:
                break
        
        return all_combinations
    
    def match_payments(self) -> List[Dict]:
        """
        Match deposits in the statement with invoices and payments.
        
        Returns:
            List of matching results
        """
        results = []
        # Track matched invoices to prevent reuse
        matched_invoice_ids = set()
        
        # Sort deposits by date to ensure chronological processing
        sorted_deposits = self.statement_df.sort_values('deposit_date')
        
        for _, deposit in sorted_deposits.iterrows():
            deposit_amount = deposit['deposit']
            deposit_date = deposit['deposit_date']
            
            logger.info(f"Processing deposit: {deposit_date.strftime('%d-%m-%y')} amount: {deposit_amount}")
            
            # Try to find matching paid records within a 3-day window (can be adjusted)
            possible_paids = self.paid_df[
                (self.paid_df['paid_date'] <= deposit_date + timedelta(days=3)) &
                (self.paid_df['paid_date'] >= deposit_date - timedelta(days=3)) &
                (~self.paid_df['matched'])  # Only consider unmatched payments
            ]
            
            matches = []
            
            # If we have possible paid records, use them to narrow down the search
            if not possible_paids.empty:
                logger.debug(f"Found {len(possible_paids)} possible paid records within date window")
                
                for _, paid in possible_paids.iterrows():
                    # Check if the paid amount matches the deposit
                    if self._is_amount_match(paid['transfer_amount'], deposit_amount):
                        company_name = paid['company_name']
                        logger.debug(f"Found potential match with company: {company_name}")
                        
                        # Find invoices for this company that haven't been matched yet
                        company_invoices = self.sale_df[
                            (self.sale_df['company_name'].str.contains(company_name, case=False, na=False)) &
                            (~self.sale_df['invoice_number'].isin(matched_invoice_ids)) &
                            (self.sale_df['invoice_date'] <= deposit_date) &
                            (self.sale_df['invoice_date'] >= deposit_date - timedelta(days=self.credit_days))
                        ]
                        
                        if not company_invoices.empty:
                            # Try to find combinations of these invoices that match the deposit amount
                            invoice_combinations = self._find_invoice_combinations_with_backtracking(
                                company_invoices, deposit_amount, self.tolerance_percent
                            )
                            
                            if invoice_combinations:
                                for invoices in invoice_combinations:
                                    invoice_ids = [inv['invoice_number'] for inv in invoices]
                                    invoice_total = sum(inv['net_amount'] for inv in invoices)
                                    
                                    # Add this match to our results
                                    matches.append({
                                        'deposit_date': deposit_date,
                                        'deposit_amount': deposit_amount,
                                        'company_name': company_name,
                                        'invoice_numbers': invoice_ids,
                                        'invoice_total': invoice_total,
                                        'paid_amount': paid['transfer_amount'],
                                        'difference': deposit_amount - invoice_total,
                                        'tax_id': paid['tax_id'],
                                        'paid_date': paid['paid_date'],
                                        'status': 'Matched'
                                    })
                                    
                                    # Mark these invoices as matched to prevent reuse
                                    matched_invoice_ids.update(invoice_ids)
                                    
                                    # Mark the paid record as matched
                                    self.paid_df.loc[self.paid_df['tax_id'] == paid['tax_id'], 'matched'] = True
                                    
                                    # Since we found a match, we can break the loop
                                    break
                                
                                # If we found matches for this paid record, move to the next deposit
                                if matches:
                                    break
            
            # If no matches found through paid records, try direct matching with sale records
            if not matches:
                logger.debug(f"No matches found via paid records, trying direct invoice matching")
                
                # Try to match with each company in the sale dataframe
                for company_name in self.sale_df['company_name'].unique():
                    # Find unmatched invoices for this company within the credit period
                    company_invoices = self.sale_df[
                        (self.sale_df['company_name'] == company_name) &
                        (~self.sale_df['invoice_number'].isin(matched_invoice_ids)) &
                        (self.sale_df['invoice_date'] <= deposit_date) &
                        (self.sale_df['invoice_date'] >= deposit_date - timedelta(days=self.credit_days))
                    ]
                    
                    if not company_invoices.empty:
                        invoice_combinations = self._find_invoice_combinations_with_backtracking(
                            company_invoices, deposit_amount, self.tolerance_percent
                        )
                        
                        if invoice_combinations:
                            for invoices in invoice_combinations:
                                invoice_ids = [inv['invoice_number'] for inv in invoices]
                                invoice_total = sum(inv['net_amount'] for inv in invoices)
                                
                                # Add this match to our results
                                matches.append({
                                    'deposit_date': deposit_date,
                                    'deposit_amount': deposit_amount,
                                    'company_name': company_name,
                                    'invoice_numbers': invoice_ids,
                                    'invoice_total': invoice_total,
                                    'paid_amount': None,  # No matching paid record
                                    'difference': deposit_amount - invoice_total,
                                    'tax_id': None,
                                    'paid_date': None,
                                    'status': 'Matched (No Payment Record)'
                                })
                                
                                # Mark these invoices as matched to prevent reuse
                                matched_invoice_ids.update(invoice_ids)
                                
                                # Since we found a match, we can break the loop
                                break
                            
                            # If we found matches for this company, move to the next deposit
                            if matches:
                                break
            
            # If we found matches, add them to the results
            if matches:
                logger.info(f"Found {len(matches)} matches for deposit on {deposit_date.strftime('%d-%m-%y')}")
                results.extend(matches)
            else:
                # No match found, add a placeholder
                logger.info(f"No matches found for deposit on {deposit_date.strftime('%d-%m-%y')}")
                results.append({
                    'deposit_date': deposit_date,
                    'deposit_amount': deposit_amount,
                    'company_name': None,
                    'invoice_numbers': [],
                    'invoice_total': 0,
                    'paid_amount': None,
                    'difference': deposit_amount,
                    'tax_id': None,
                    'paid_date': None,
                    'status': 'Unmatched'
                })
        
        self.results = results
        return results
    
    def generate_report(self) -> pd.DataFrame:
        """
        Generate a report of payment matches.
        
        Returns:
            DataFrame with matching results
        """
        logger.info("generate report")
        if not self.results:
            self.match_payments()
        
        # Convert results to DataFrame
        report_df = pd.DataFrame(self.results)
        
        # Add status column if not already present
        if 'status' not in report_df.columns:
            conditions = [
                (report_df['company_name'].isna()),
                (report_df['difference'].abs() > (report_df['deposit_amount'] * self.tolerance_percent)),
                (report_df['paid_amount'].isna())
            ]
            
            choices = [
                'Unmatched',
                'Partial Match',
                'Missing Payment Record'
            ]
            
            default = 'Matched'
            report_df['status'] = np.select(conditions, choices, default=default)
        
        # Format invoice numbers as comma-separated strings
        if 'invoice_numbers' in report_df.columns:
            report_df['invoice_numbers'] = report_df['invoice_numbers'].apply(
                lambda x: ', '.join(x) if isinstance(x, list) else ''
            )
        logger.debug(f"Payment matching report: \n{report_df.head(10)}\n ...(showing 10 out of {len(report_df)})")
        
        return report_df
    
    def save_report(self, filepath: str) -> None:
        """
        Save the comprehensive matching report to files.
        
        Args:
            filepath: Base path to save the reports
        """
        logger.info(f"Saving payment matching reports to {filepath}")
        
        # Generate the main report
        report_df = self.generate_report()
        report_df.to_csv(filepath, index=False)
        logger.info(f"Main payment matching report saved to {filepath}")
        
        # Initialize the analyzer to generate additional reports
        analyzer = PaymentAnalyzer(self)
        
        # Get unmatched invoices
        unmatched_invoices = analyzer.find_unmatched_invoices()
        invoice_filepath = filepath.replace('.csv', '_unmatched_invoices.csv')
        if not unmatched_invoices.empty:
            unmatched_invoices.to_csv(invoice_filepath, index=False)
            logger.info(f"Unmatched invoices report saved to {invoice_filepath}")
        
        # Get unmatched payments
        unmatched_payments = analyzer.find_unmatched_payments()
        payment_filepath = filepath.replace('.csv', '_unmatched_payments.csv')
        if not unmatched_payments.empty:
            unmatched_payments.to_csv(payment_filepath, index=False)
            logger.info(f"Unmatched payments report saved to {payment_filepath}")
        
        # Get unmatched deposits
        unmatched_deposits = analyzer.find_unmatched_deposits()
        deposit_filepath = filepath.replace('.csv', '_unmatched_deposits.csv')
        if not unmatched_deposits.empty:
            unmatched_deposits.to_csv(deposit_filepath, index=False)
            logger.info(f"Unmatched deposits report saved to {deposit_filepath}")


class PaymentValidator:
    """
    Class for validating payment matches and identifying discrepancies.
    - Validates payment matches against expected criteria
    - Identifies issues like unmatched deposits, partial matches, or missing payment records
    - Generates detailed validation reports
    """
    
    def __init__(self, matcher: PaymentMatcher):
        """
        Initialize the PaymentValidator.
        
        Args:
            matcher: PaymentMatcher instance with matching results
        """
        self.matcher = matcher
        self.validation_results = []
    
    def validate_matches(self) -> pd.DataFrame:
        """
        Validate payment matches and identify discrepancies.
        
        Returns:
            DataFrame with validation results
        """
        if not self.matcher.results:
            self.matcher.match_payments()
        
        report_df = self.matcher.generate_report()
        
        # Perform additional validation
        validations = []
        
        for _, row in report_df.iterrows():
            validation = {
                'deposit_date': row['deposit_date'],
                'deposit_amount': row['deposit_amount'],
                'company_name': row['company_name'],
                'invoice_numbers': row['invoice_numbers'],
                'status': row['status'],
                'issues': []
            }
            
            # Check for issues based on status
            if row['status'] == 'Unmatched':
                validation['issues'].append('No matching invoice or company found')
            
            elif row['status'] == 'Partial Match':
                validation['issues'].append(
                    f"Amount mismatch: Deposit {row['deposit_amount']:.2f}, "
                    f"Invoice total {row['invoice_total']:.2f}, "
                    f"Difference {row['difference']:.2f}"
                )
            
            elif row['status'] == 'Missing Payment Record':
                validation['issues'].append('Found matching invoice but no payment record')
            
            # Check for date discrepancies in matched records
            if row['status'] == 'Matched' and pd.notna(row['paid_date']):
                date_diff = abs((row['deposit_date'] - row['paid_date']).days)
                if date_diff > 3:
                    validation['issues'].append(
                        f"Date discrepancy: Deposit date {row['deposit_date'].date()}, "
                        f"Paid date {row['paid_date'].date()}, "
                        f"Difference {date_diff} days"
                    )
            
            # Add validation to results
            validation['issues'] = ', '.join(validation['issues']) if validation['issues'] else 'No issues'
            validations.append(validation)
        
        self.validation_results = validations
        return pd.DataFrame(validations)
    
    def save_validation_report(self, filepath: str) -> None:
        """
        Save the validation report to a file.
        
        Args:
            filepath: Path to save the report
        """
        logger.info("Saving payment validation report")
        if not self.validation_results:
            validation_df = self.validate_matches()
        else:
            validation_df = pd.DataFrame(self.validation_results)
            logger.debug(f"Payment validation report: \n{validation_df.head(10)}\n ...(showing 10 out of {len(validation_df)})")
            
        validation_df.to_csv(filepath, index=False)
        logger.info(f"Payment validation report saved to {filepath}")


class PaymentAnalyzer:
    """
    Class for analyzing payment patterns and generating insights.
    - Analyzes payment patterns by company
    - Identifies unmatched invoices that haven't been paid yet
    - Generates company-specific payment analytics
    """
    
    def __init__(self, matcher: PaymentMatcher):
        """
        Initialize the PaymentAnalyzer.
        
        Args:
            matcher: PaymentMatcher instance with matching results
        """
        self.matcher = matcher
    
    def analyze_company_payments(self) -> pd.DataFrame:
        """
        Analyze payment patterns by company.
        
        Returns:
            DataFrame with company payment analysis
        """
        logger.info("Analyzing company payment patterns")
        if not self.matcher.results:
            self.matcher.match_payments()
            
        report_df = self.matcher.generate_report()
        
        # Filter out unmatched records
        matched_df = report_df[report_df['company_name'].notna()]
        
        if matched_df.empty:
            return pd.DataFrame()
        
        # Group by company and analyze
        company_stats = []
        
        for company_name, group in matched_df.groupby('company_name'):
            stats = {
                'company_name': company_name,
                'total_payments': len(group),
                'total_amount': group['deposit_amount'].sum(),
                'avg_payment': group['deposit_amount'].mean(),
                'avg_difference': group['difference'].mean(),
                'matched_count': (group['status'] == 'Matched').sum(),
                'partial_matches': (group['status'] == 'Partial Match').sum(),
                'missing_records': (group['status'] == 'Missing Payment Record').sum()
            }
            
            # Calculate payment timing if we have paid dates
            if 'paid_date' in group.columns and group['paid_date'].notna().any():
                paid_dates = group[group['paid_date'].notna()]
                if not paid_dates.empty:
                    # Calculate average days between invoice and payment
                    # This assumes we have access to invoice dates, which we might need to get from sale_df
                    stats['avg_payment_days'] = None  # Placeholder
                    
            company_stats.append(stats)
            logger.info(f"Payment analysis for {company_name}: {stats}")
        
        return pd.DataFrame(company_stats) if company_stats else pd.DataFrame()
    
    def find_unmatched_invoices(self) -> pd.DataFrame:
        """
        Find invoices that haven't been matched to any deposit.
        
        Returns:
            DataFrame with unmatched invoices
        """
        logger.info("Finding unmatched invoices")
        if not self.matcher.results:
            self.matcher.match_payments()
            
        report_df = self.matcher.generate_report()
        
        # Get all matched invoice numbers
        matched_invoices = set()
        for inv_list in report_df['invoice_numbers'].str.split(', '):
            if isinstance(inv_list, list) and inv_list[0]:  # Check if it's a list and not empty
                matched_invoices.update(inv_list)
        
        # Filter sale_df to find unmatched invoices
        sale_df = self.matcher.sale_df
        unmatched = sale_df[~sale_df['invoice_number'].isin(matched_invoices)]
        
        # Calculate days since invoice date
        today = pd.Timestamp.now().normalize()
        unmatched['days_outstanding'] = (today - unmatched['invoice_date']).dt.days
        
        # Sort by company and then by days outstanding
        unmatched = unmatched.sort_values(['company_name', 'days_outstanding'], ascending=[True, False])
        
        logger.debug(f"Found {len(unmatched)} unmatched invoices")
        
        # Add category based on days outstanding
        def categorize_age(days):
            if days <= 30:
                return 'Current'
            elif days <= 60:
                return '31-60 days'
            elif days <= 90:
                return '61-90 days'
            else:
                return 'Over 90 days'
        
        unmatched['age_category'] = unmatched['days_outstanding'].apply(categorize_age)
        
        # Format dates for output
        unmatched['invoice_date_formatted'] = unmatched['invoice_date'].dt.strftime('%d-%m-%y')
        
        return unmatched[['invoice_number', 'company_name', 'invoice_date_formatted', 
                        'net_amount', 'days_outstanding', 'age_category']]

    def find_unmatched_payments(self) -> pd.DataFrame:
        """
        Find paid records that haven't been matched to any deposit.
        
        Returns:
            DataFrame with unmatched payments
        """
        logger.info("Finding unmatched payments")
        
        # Get paid records that aren't marked as matched
        unmatched_paid = self.matcher.paid_df[~self.matcher.paid_df['matched']]
        
        # Calculate days since payment
        today = pd.Timestamp.now().normalize()
        unmatched_paid['days_since_payment'] = (today - unmatched_paid['paid_date']).dt.days
        
        # Sort by company and then by days since payment
        unmatched_paid = unmatched_paid.sort_values(['company_name', 'days_since_payment'], ascending=[True, False])
        
        logger.debug(f"Found {len(unmatched_paid)} unmatched payments")
        
        # Format dates for output
        unmatched_paid['paid_date_formatted'] = unmatched_paid['paid_date'].dt.strftime('%d-%m-%y')
        
        return unmatched_paid[['company_name', 'paid_date_formatted', 'transfer_amount', 
                            'tax_id', 'days_since_payment']]

    def find_unmatched_deposits(self) -> pd.DataFrame:
        """
        Identify deposits that couldn't be matched to any invoice or payment.
        
        Returns:
            DataFrame with unmatched deposits
        """
        logger.info("Finding unmatched deposits")
        if not self.matcher.results:
            self.matcher.match_payments()
            
        report_df = self.matcher.generate_report()
        
        # Filter for unmatched deposits
        unmatched_deposits = report_df[report_df['status'] == 'Unmatched']
        
        # Format dates for output
        if not unmatched_deposits.empty:
            unmatched_deposits['deposit_date_formatted'] = unmatched_deposits['deposit_date'].dt.strftime('%d-%m-%y')
        
        logger.debug(f"Found {len(unmatched_deposits)} unmatched deposits")
        
        return unmatched_deposits[['deposit_date_formatted', 'deposit_amount']] if not unmatched_deposits.empty else pd.DataFrame()