import pandas as pd
from typing import List, Dict, Any, Set
from log_utils import get_logger

logger = get_logger()

class DataReviewer:
    """Interactive data reviewer for validating and correcting DataFrame content"""
    
    def __init__(self, df: pd.DataFrame, rows_per_page: int = 10):
        """
        Initialize data reviewer
        
        Args:
            df: DataFrame to review
            rows_per_page: Number of rows to display at once
        """
        self.df = df.copy()
        self.rows_per_page = rows_per_page
        self.total_rows = len(df)
        self.modified_rows = set()
    
    def review_data(self) -> pd.DataFrame:
        """
        Start interactive review process
        
        Returns:
            DataFrame with user corrections
        """
        logger.info(f"Starting data review for {self.total_rows} rows")
        
        # Check if user wants to review
        response = input("Do you want to review and correct the data? (y/n): ").strip().lower()
        if response != 'y':
            logger.info("Data review skipped by user")
            return self.df
            
        # Review in paginated manner
        current_row = 0
        while current_row < self.total_rows:
            # Display current page of rows
            end_row = min(current_row + self.rows_per_page, self.total_rows)
            self._display_rows(current_row, end_row)
            
            # Get rows to edit
            rows_to_edit = self._get_rows_to_edit(current_row, end_row)
            
            # Edit selected rows
            if rows_to_edit:
                min_row_edited = min([r-1 for r in rows_to_edit]) if rows_to_edit else 0
                self._edit_rows(rows_to_edit)
                # Update balances for rows after the modified ones
                self._update_subsequent_balances(min_row_edited)
                # Review modified rows
                self._review_modifications(rows_to_edit)
            
            # Ask if user wants to continue with next page
            if end_row < self.total_rows:
                response = input(f"\nPress Enter to continue to the next {self.rows_per_page} rows, or type 'n' to stop: ").strip().lower()
                if response == 'n':
                    break
                    
            current_row = end_row
        
        logger.info(f"Data review completed. Modified {len(self.modified_rows)} rows")
        return self.df
    
    def _display_rows(self, start_row: int, end_row: int) -> None:
        """
        Display rows in the current page
        
        Args:
            start_row: First row index to display (0-based)
            end_row: Last row index to display (exclusive)
        """
        print(f"\n{'='*80}")
        print(f"Displaying rows {start_row+1} to {end_row} of {self.total_rows}")
        print(f"{'='*80}")
        
        # Reset display options for better viewing
        with pd.option_context('display.max_columns', None, 
                               'display.width', None,
                               'display.max_colwidth', 30):
            
            # Add row numbers (1-based) for user reference
            display_df = self.df.iloc[start_row:end_row].copy()
            display_df.index = range(start_row+1, end_row+1)  # 1-based row numbers
            # Format datetime columns if they exist
            for col in display_df.select_dtypes(include=['datetime']).columns:
                display_df[col] = display_df[col].dt.strftime('%d-%m-%y %H:%M')
            
            print(display_df)
    
    def _get_rows_to_edit(self, start_row: int, end_row: int) -> List[int]:
        """
        Get list of rows user wants to edit
        
        Args:
            start_row: First row index in current page (0-based)
            end_row: Last row index in current page (exclusive)
            
        Returns:
            List of row numbers to edit (1-based)
        """
        print("\nEnter row numbers to edit (comma-separated, e.g. '1,3,5'), or press Enter to skip:")
        row_input = input("> ").strip()
        
        if not row_input:
            return []
            
        try:
            # Parse and validate row numbers
            row_numbers = [int(x.strip()) for x in row_input.split(',') if x.strip()]
            valid_rows = [r for r in row_numbers if start_row+1 <= r <= end_row]
            
            if len(valid_rows) != len(row_numbers):
                invalid = set(row_numbers) - set(valid_rows)
                print(f"WARNING: Skipping invalid row numbers: {invalid}")
                
            return valid_rows
            
        except ValueError:
            print("ERROR: Invalid input. Please enter comma-separated row numbers.")
            return self._get_rows_to_edit(start_row, end_row)
    
    def _edit_rows(self, row_numbers: List[int]) -> None:
        """
        Edit selected rows column by column
        
        Args:
            row_numbers: List of row numbers to edit (1-based)
        """
        for row_num in row_numbers:
            # Convert to 0-based index for DataFrame
            idx = row_num - 1
            
            print(f"\n{'='*40}")
            print(f"Editing row {row_num}:")
            print(f"{'='*40}")
            
            # Display the current row
            with pd.option_context('display.max_columns', None, 'display.width', None):
                row_df = self.df.iloc[[idx]].copy()
                row_df.index = [row_num]  # Display with 1-based row number
                print(row_df)
            
            # Track original amount for balance recalculation
            original_amount = self.df.at[idx, 'amount'] if not pd.isna(self.df.at[idx, 'amount']) else 0
            original_type = self.df.at[idx, 'type']
            amount_changed = False
            type_changed = False
            
            # Edit each column
            modified = False
            for col in self.df.columns:
                current_value = self.df.at[idx, col]
                print(f"\nEditing '{col}' (current: {current_value})")
                print("Press Enter to keep current value, or enter new value:")
                new_value = input("> ").strip()
                
                if new_value and new_value != str(current_value):
                    try:
                        # For numeric columns
                        if pd.api.types.is_numeric_dtype(self.df[col]):
                            self.df.at[idx, col] = pd.to_numeric(new_value, errors='coerce')
                        # For datetime columns
                        elif pd.api.types.is_datetime64_dtype(self.df[col]):
                            self.df.at[idx, col] = pd.to_datetime(new_value, errors='coerce')
                        # For all other columns
                        else:
                            self.df.at[idx, col] = new_value
                            
                        modified = True
                        
                        # Track if amount or type changed
                        if col == 'amount':
                            amount_changed = True
                        elif col == 'type':
                            type_changed = True
                            
                        logger.debug(f"Modified row {row_num}, column '{col}': '{current_value}' → '{new_value}'")
                    except Exception as e:
                        logger.error(f"Error updating row {row_num}, column '{col}': {str(e)}")
                        print(f"ERROR: Could not set value. {str(e)}")
            
            # If amount or type changed, update the balance for this row only
            if modified and (amount_changed or type_changed):
                self._update_single_row_balance(idx)
                
            if modified:
                self.modified_rows.add(idx)
                logger.info(f"Row {row_num} modified")
    
    def _update_single_row_balance(self, idx: int) -> None:
        """
        Update the balance for a single row based on the previous row's balance
        and the current row's amount and transaction type
        
        Args:
            idx: 0-based row index to update the balance for
        """
        # Cannot update balance if it's the first row or has no amount
        if idx == 0 or pd.isna(self.df.at[idx, 'amount']):
            logger.debug(f"Skipping balance update for row {idx+1}: first row or no amount")
            return
            
        # Get the previous row's balance
        prev_balance = None
        for i in range(idx-1, -1, -1):
            if not pd.isna(self.df.at[i, 'balance']):
                prev_balance = self.df.at[i, 'balance']
                break
                
        if prev_balance is None:
            logger.warning(f"Cannot update balance for row {idx+1}: no previous balance found")
            return
            
        # Calculate new balance based on transaction type
        amount = self.df.at[idx, 'amount']
        trans_type = self.df.at[idx, 'type']
        new_balance = prev_balance
        if trans_type == 'Deposit':
            logger.debug(f"Row {idx+1}: Deposit of {amount} → Previous balance: {prev_balance} → New balance: {prev_balance + amount}")
            new_balance = prev_balance + amount
        elif trans_type == 'Withdrawal':
            logger.debug(f"Row {idx+1}: Withdrawal of {amount} → Previous balance: {prev_balance} → New balance: {prev_balance - amount}")
            new_balance = prev_balance - amount
            
        # Update the balance
        old_balance = self.df.at[idx, 'balance']
        self.df.at[idx, 'balance'] = new_balance
        
        # Update calculated_balance if it exists
        if 'calculated_balance' in self.df.columns:
            self.df.at[idx, 'calculated_balance'] = new_balance
            
        # Update needs_review flag if it exists
        if 'needs_review' in self.df.columns:
            self.df.at[idx, 'needs_review'] = False  # Reset since we just updated it
            
        logger.info(f"Updated balance for row {idx+1}: {old_balance} → {new_balance}")
        
    def _update_subsequent_balances(self, start_idx: int) -> None:
        """
        Update balances for all rows after the specified index
        
        Args:
            start_idx: 0-based index of the first row to be updated
        """
        logger.info(f"Updating balances for rows after row {start_idx+1}")
        
        # Find the first row with a balance before start_idx
        prev_balance = None
        prev_idx = None
        
        for i in range(start_idx-1, -1, -1):
            if not pd.isna(self.df.at[i, 'balance']):
                prev_balance = self.df.at[i, 'balance']
                prev_idx = i
                break
                
        if prev_balance is None or prev_idx is None:
            # If we can't find a previous balance, just start from the current row
            if not pd.isna(self.df.at[start_idx, 'balance']):
                prev_balance = self.df.at[start_idx, 'balance']
                prev_idx = start_idx
            else:
                logger.warning(f"Cannot update subsequent balances: no valid starting balance found")
                return
                
        # Update balances for all rows after prev_idx
        current_balance = prev_balance
        update_count = 0
        
        for idx in range(prev_idx + 1, len(self.df)):
            # Skip rows with no amount
            if pd.isna(self.df.at[idx, 'amount']):
                continue
                
            amount = self.df.at[idx, 'amount']
            trans_type = self.df.at[idx, 'type']
            
            # Calculate new balance based on transaction type
            if trans_type == 'Deposit':
                current_balance += amount
            elif trans_type == 'Withdrawal':
                current_balance -= amount
            
            # Update the balance
            old_balance = self.df.at[idx, 'balance']
            if abs(old_balance - current_balance) > 0.01:  # Only log if different
                logger.debug(f"Row {idx+1}: Updated balance {old_balance} → {current_balance}")
                
            self.df.at[idx, 'balance'] = current_balance
            
            # Update calculated_balance if it exists
            if 'calculated_balance' in self.df.columns:
                self.df.at[idx, 'calculated_balance'] = current_balance
                
            # Update needs_review flag if it exists
            if 'needs_review' in self.df.columns:
                self.df.at[idx, 'needs_review'] = False
                
            update_count += 1
            
        logger.info(f"Updated balances for {update_count} rows after row {prev_idx+1}")
        
    def _recalculate_balances(self) -> None:
        """
        Recalculate all balances based on transaction types and amounts
        """
        logger.info("Recalculating all balances after edits")
        
        # Sort by datetime to ensure chronological order
        # if 'datetime' in self.df.columns:
        #     self.df = self.df.sort_values('datetime').reset_index(drop=True)
        
        # Find initial balance (first row that has a balance but no amount)
        initial_balance = 0
        first_row_idx = -1
        for idx in range(len(self.df)):
            if pd.isna(self.df.at[idx, 'amount']) and not pd.isna(self.df.at[idx, 'balance']):
                initial_balance = self.df.at[idx, 'balance']
                first_row_idx = idx
                break
        
        # If we didn't find an initial balance row, use the first row's balance
        if first_row_idx == -1 and len(self.df) > 0:
            initial_balance = self.df.at[0, 'balance'] if not pd.isna(self.df.at[0, 'balance']) else 0
            first_row_idx = 0
        
        # Recalculate all balances from the row after the first_row_idx
        current_balance = initial_balance
        for idx in range(first_row_idx + 1, len(self.df)):
            # Skip rows with no amount
            if pd.isna(self.df.at[idx, 'amount']):
                continue
                
            amount = self.df.at[idx, 'amount']
            trans_type = self.df.at[idx, 'type']
            
            # Update balance based on transaction type
            if trans_type == 'Deposit':
                current_balance += amount
            elif trans_type == 'Withdrawal':
                current_balance -= amount
            
            # Update the balance in the DataFrame
            old_balance = self.df.at[idx, 'balance']
            self.df.at[idx, 'balance'] = current_balance
            
            # Update the calculated balance column if it exists
            if 'calculated_balance' in self.df.columns:
                self.df.at[idx, 'calculated_balance'] = current_balance
                
            # Check if the calculated balance differs from the recorded balance
            if 'needs_review' in self.df.columns:
                if abs(old_balance - current_balance) > 0.01:  # Using a small threshold for floating point comparison
                    self.df.at[idx, 'needs_review'] = True
                else:
                    self.df.at[idx, 'needs_review'] = False
                    
            logger.debug(f"Row {idx+1}: {trans_type} of {amount} → Balance: {current_balance}")
        
        logger.info("Balance recalculation completed")
    
    def _review_modifications(self, row_numbers: List[int]) -> None:
        """
        Review modified rows and allow further corrections if needed
        
        Args:
            row_numbers: List of row numbers to review (1-based)
        """
        # Display modified rows
        print("\n" + "="*40)
        print("MODIFIED ROWS REVIEW")
        print("="*40)
        
        # Filter only rows that were actually modified
        modified_row_numbers = [r for r in row_numbers if r-1 in self.modified_rows]
        
        if not modified_row_numbers:
            logger.debug("No rows were modified during editing")
            return
            
        # Display all modified rows
        self._display_specific_rows(modified_row_numbers)
        
        # Check if any rows still need correction
        print("\nAre there any rows still incorrect? (y/n): ")
        response = input("> ").strip().lower()
        
        if response == 'y':
            # Get rows that need further correction
            print("\nEnter row numbers to edit again (comma-separated):")
            rows_to_fix = self._parse_row_numbers(input("> ").strip())
            
            if rows_to_fix:
                logger.debug(f"Re-editing rows: {rows_to_fix}")
                self._edit_rows(rows_to_fix)
                # Recalculate subsequent balances if needed
                self._update_subsequent_balances(min([r-1 for r in rows_to_fix]))
                self._review_modifications(rows_to_fix)  # Recursive call for further review
    
    def _display_specific_rows(self, row_numbers: List[int]) -> None:
        """
        Display specific rows by their 1-based row numbers
        
        Args:
            row_numbers: List of 1-based row numbers to display
        """
        if not row_numbers:
            print("No rows to display")
            return
            
        # Convert to 0-based indices for slicing
        indices = [r-1 for r in row_numbers]
        
        with pd.option_context('display.max_columns', None, 
                              'display.width', None,
                              'display.max_colwidth', 30):
            
            display_df = self.df.iloc[indices].copy()
            # Set index to show 1-based row numbers
            display_df.index = row_numbers
            print(display_df)
            
        logger.debug(f"Displayed rows: {row_numbers}")
    
    def _parse_row_numbers(self, row_input: str) -> List[int]:
        """
        Parse and validate comma-separated row numbers
        
        Args:
            row_input: String of comma-separated row numbers
            
        Returns:
            List of valid row numbers (1-based)
        """
        if not row_input:
            return []
            
        try:
            # Parse row numbers
            row_numbers = [int(x.strip()) for x in row_input.split(',') if x.strip()]
            
            # Validate row numbers are within range
            valid_rows = [r for r in row_numbers if 1 <= r <= self.total_rows]
            
            if len(valid_rows) != len(row_numbers):
                invalid = set(row_numbers) - set(valid_rows)
                print(f"WARNING: Skipping invalid row numbers: {invalid}")
                logger.warning(f"Skipped invalid row numbers: {invalid}")
                
            return valid_rows
            
        except ValueError:
            print("ERROR: Invalid input. Please enter comma-separated row numbers.")
            logger.error("Invalid row number input format")
            return []