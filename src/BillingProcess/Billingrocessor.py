import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

from log_utils import get_logger

logger = get_logger()

class BillingProcessor:
    """
    A class to process billing information from Excel files
    """
    def __init__(self, billing_file: str):
        """
        Initialize the BillingProcessor with a billing file path

        :param billing_file: Path to the Excel billing file
        """
        self.billing_file = billing_file
        
        # Define constants
        self.SELL_BILL_COLUMNS = [
            "ลำดับ", "วัน/เดือน/ปี", "เลขที่ใบกำกับภาษี", 
            "ชื่อผู้ซื้อสินค้า/ผู้รับบริการ", "มูลค่าสินค้า", 
            "ภาษีมูลค่าเพิ่ม", "จำนวนเงิน", "หัก 3%", "คงเหลือ"
        ]
        
        self.PAID_BILL_COLUMNS = None  # Will be set during processing
        logger.info(f"BillingProcessor initialized with file: {self.billing_file}")

    @staticmethod
    def convert_thai_date(date_str: str) -> str:
        """
        Convert Thai Buddhist calendar date to a standardized format

        :param date_str: Input date string
        :return: Converted date string
        """
        try:
            # Handle Excel datetime objects (date with "-")
            if isinstance(date_str, str) and "-" in date_str:
                try:
                    # Parse Excel datetime format
                    date_obj = pd.to_datetime(date_str)
                    dd = date_obj.day
                    mm = date_obj.month
                    # Convert year to Buddhist calendar (CE + 43)
                    yy = (date_obj.year - 1900) - 43  # Adjust for 2-digit year
                    return f"{dd:02d}/{mm:02d}/{yy:02d}"
                except:
                    return date_str
            
            # Handle string dates with "/" (General type)
            elif isinstance(date_str, str) and "/" in date_str:
                dd, mm, yy = date_str.split("/")
                # Ensure yy is 2 digits and convert from BE to CE
                yy = int(yy) - 43 if len(yy) == 2 else int(yy[-2:]) - 43
                return f"{int(dd):02d}/{int(mm):02d}/{yy:02d}"
                
            return date_str
        except Exception as e:
            logger.warning(f"Date conversion failed for {date_str}: {str(e)}")
            return date_str

    def process_sell_billing(self, sheet_name: Optional[str] = None) -> pd.DataFrame:
        """
        Process the sell billing sheet from the Excel file

        :param sheet_name: Optional specific sheet name to process
        :return: Processed sell billing DataFrame
        """
        logger.info(f"Processing sell billing from file: {self.billing_file}")
        # Use default sheet name if not provided
        if sheet_name is None:
            sheet_name = r"ขาย1-9-ทรัพย์ (2)"

        # Read Excel with all columns as string
        sellBil_df = pd.read_excel(
            self.billing_file, 
            sheet_name=sheet_name, 
            engine="openpyxl",
            dtype=str  # Convert all columns to string
        )
        logger.info(f"Read sell billing Sheet: ")
        sellBil_df.info()
        logger.debug(f"Billing DataFrame head:\n{sellBil_df.head(50)}")

        # Ensure DataFrame has exactly 9 columns before setting BILL_COLUMNS
        sellBil_df = sellBil_df.iloc[:, :9]  # Keep only first 9 columns
        sellBil_df.columns = self.SELL_BILL_COLUMNS
        sellBil_df.dropna(how='any', inplace=True)  # Drop rows that are all NaN
        
        logger.info(f"Header row set to: {sellBil_df.columns}")
        logger.info(f"Rows after dropping NaN rows: {len(sellBil_df)}")
        logger.info(f"Columns after dropping NaN rows: {len(sellBil_df.columns)}")
        sellBil_df.info()

        # Format columns according to requirements
        sellBil_df["ลำดับ"] = pd.to_numeric(sellBil_df["ลำดับ"], errors='coerce').fillna(0).astype(int)
        sellBil_df["วัน/เดือน/ปี"] = sellBil_df["วัน/เดือน/ปี"].apply(self.convert_thai_date)
        
        for col in ["เลขที่ใบกำกับภาษี", "ชื่อผู้ซื้อสินค้า/ผู้รับบริการ"]:
            sellBil_df[col] = sellBil_df[col].str.strip()
            sellBil_df[col] = sellBil_df[col].astype(str)

        # Convert numeric columns to float with 2 decimal places
        numeric_columns = ["มูลค่าสินค้า", "ภาษีมูลค่าเพิ่ม", "จำนวนเงิน", "หัก 3%", "คงเหลือ"]
        for col in numeric_columns:
            sellBil_df[col] = pd.to_numeric(sellBil_df[col], errors='coerce').fillna(0).round(2)

        logger.info("Column formatting completed")

        logger.debug(f"Billing DataFrame head:\n{sellBil_df.head(50)}\n{'='*100}")
        logger.debug(f"Billing DataFrame tail:\n{sellBil_df.tail(50)}")

        return sellBil_df

    def process_paid_billing(self, sheet_name: Optional[str] = None) -> pd.DataFrame:
        """
        Process the paid billing sheet from the Excel file

        :param sheet_name: Optional specific sheet name to process
        :return: Processed paid billing DataFrame
        """
        logger.info(f"Processing paid billing from file: {self.billing_file}")
        # Use default sheet name if not provided
        if sheet_name is None:
            sheet_name = r"หัก ณ ที่จ่าย"

        # Read paid billing sheet
        paidBil_df = pd.read_excel(
            self.billing_file, 
            sheet_name=sheet_name, 
            engine="openpyxl",
            dtype=str  # Convert all columns to string
        )
        logger.info(f"Read paid billing Sheet: ")
        paidBil_df.info()
        
        paidBil_df = paidBil_df.iloc[:, :5]  # Keep only first 5 columns
        logger.debug(f"Paid Billing DataFrame head:\n{paidBil_df.head(50)}")

        paidBil_df.dropna(how='any', inplace=True)
        logger.info("Paid Billing DataFrame after drop na:")
        logger.info(f"Header row set to: {paidBil_df.columns}")
        logger.info(f"Rows after dropping NaN rows: {len(paidBil_df)}")
        logger.info(f"Columns after dropping NaN rows: {len(paidBil_df.columns)}")
        paidBil_df.info()
        
        # Format columns according to requirements
        paidBil_df["ว/ด/ป"] = paidBil_df["ว/ด/ป"].apply(self.convert_thai_date)
        logger.info("Converting columns to appropriate types")
        
        for col in ["ชื่อและที่อยู่ผู้หักภาษี", "เลขประจำตัวผู้เสียภาษี"]:
            paidBil_df[col] = paidBil_df[col].str.strip()
            paidBil_df[col] = paidBil_df[col].astype(str)

        # Convert numeric columns to float with 2 decimal places
        logger.info("Converting numeric columns to float")
        for col in ["จำนวนเงิน", "หัก ณ ที่จ่าย"]:
            paidBil_df[col] = pd.to_numeric(paidBil_df[col], errors='coerce').fillna(0).round(2)

        # Define a constant for the column name "ยอดโอน"
        TRANSFER_AMOUNT_COLUMN = "ยอดโอน"

        # Calculate the transfer amount correctly
        logger.info("Calculating transfer amount")
        paidBil_df[TRANSFER_AMOUNT_COLUMN] = (
            (paidBil_df["จำนวนเงิน"].astype(float) * 1.07) - paidBil_df["หัก ณ ที่จ่าย"].astype(float)
        ).round(2)
        paidBil_df["ยอดโอน"] = paidBil_df["ยอดโอน"].astype(float).round(2)

        paidBil_df.info()
        logger.debug(f"Paid Billing DataFrame head:\n{paidBil_df.head(50)}\n{'='*100}")
        logger.debug(f"Paid Billing DataFrame tail:\n{paidBil_df.tail(50)}")

        return paidBil_df

# Example usage
def main():
    # Configuration 
    BILLING_FILE = r'data\files\billings\ขาย1-9-ทรัพย์.xlsx'
    
    # Create an instance of BillingProcessor
    processor = BillingProcessor(BILLING_FILE)
    
    # Process sell billing
    sell_billing_df = processor.process_sell_billing()
    
    # Process paid billing
    paid_billing_df = processor.process_paid_billing()

if __name__ == "__main__":
    main()