import yfinance as yf
import pandas as pd
from datetime import datetime

TICKER = "AAPL"
START_DATE = "2015-01-01"
END_DATE = datetime.now().strftime("%Y-%m-%d")
OUTPUT_FILE = f"{TICKER}_data.csv"

print(f"Fetching data for {TICKER} from {START_DATE} to {END_DATE}")

try: 
  stock_data = yf.download(TICKER, start=START_DATE, end=END_DATE)

  if stock_data.empty:
      print("No data found for the specified date range.")
  else:
      stock_data.to_csv(OUTPUT_FILE)
      print(f"Data saved to {OUTPUT_FILE}")
      print("Data preview:")
      print(stock_data.head())
      print("Data information:")
      print(stock_data.info())
except Exception as e:
    print(f"An error occurred while fetching data: {e}")


print("Script execution completed.")