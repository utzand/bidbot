"""
S&P 500 Symbols List
This file contains a comprehensive list of S&P 500 symbols for the stock screener.
"""

# Comprehensive S&P 500 symbols list (as of 2024)
SP500_SYMBOLS = [
    # Technology (Top 10 by market cap)
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'AVGO', 'ASML', 'CRM',
    
    # Financial Services
    'BRK.B', 'JPM', 'BAC', 'WFC', 'GS', 'MS', 'SCHW', 'BLK', 'C', 'USB',
    
    # Healthcare
    'UNH', 'JNJ', 'PFE', 'ABBV', 'TMO', 'DHR', 'ABT', 'LLY', 'AMGN', 'GILD',
    
    # Consumer Discretionary
    'HD', 'DIS', 'NKE', 'MCD', 'SBUX', 'LOW', 'TJX', 'BKNG', 'MAR', 'HLT',
    
    # Communication Services
    'GOOGL', 'META', 'NFLX', 'CMCSA', 'T', 'TMUS', 'VZ', 'CHTR', 'PARA', 'FOX',
    
    # Consumer Staples
    'PG', 'KO', 'PEP', 'WMT', 'COST', 'PM', 'MO', 'MDLZ', 'GIS', 'K',
    
    # Industrials
    'UPS', 'UNP', 'RTX', 'CAT', 'HON', 'LMT', 'BA', 'DE', 'GE', 'MMM',
    
    # Energy
    'XOM', 'CVX', 'COP', 'EOG', 'SLB', 'PSX', 'VLO', 'MPC', 'OXY', 'HAL',
    
    # Materials
    'LIN', 'APD', 'FCX', 'NEM', 'DOW', 'DD', 'NUE', 'BLL', 'ALB', 'ECL',
    
    # Real Estate
    'PLD', 'AMT', 'CCI', 'EQIX', 'DLR', 'PSA', 'O', 'SPG', 'WELL', 'EQR',
    
    # Utilities
    'NEE', 'DUK', 'SO', 'D', 'AEP', 'SRE', 'XEL', 'WEC', 'DTE', 'ED',
    
    # ETFs and Index Funds
    'SPY', 'QQQ', 'IWM', 'DIA', 'VTI', 'VOO', 'VEA', 'VWO', 'BND', 'AGG',
    
    # Additional Major Companies
    'ADBE', 'ISRG', 'VRTX', 'REGN', 'ADI', 'ZTS', 'CI', 'BKNG', 'MDLZ', 'SCHW',
    'TMUS', 'SO', 'DUK', 'D', 'NEE', 'AEP', 'SRE', 'XEL', 'WEC', 'DTE',
    'ED', 'PEG', 'AEE', 'EIX', 'PCG', 'CNP', 'NI', 'CMS', 'ATO', 'LNT',
    'IBM', 'RTX', 'CAT', 'SPGI', 'GS', 'AMGN', 'PLD', 'T', 'DE', 'GILD',
    'ADI', 'ISRG', 'VRTX', 'REGN', 'LMT', 'SCHW', 'BKNG', 'MDLZ', 'CI', 'ZTS',
    'TMUS', 'SO', 'DUK', 'D', 'NEE', 'AEP', 'SRE', 'XEL', 'WEC', 'DTE',
    'ED', 'PEG', 'AEE', 'EIX', 'PCG', 'CNP', 'NI', 'CMS', 'ATO', 'LNT'
]

# Remove duplicates while preserving order
SP500_SYMBOLS = list(dict.fromkeys(SP500_SYMBOLS))

def get_sp500_symbols():
    """Get the list of S&P 500 symbols"""
    return SP500_SYMBOLS.copy()

def get_sector_symbols(sector):
    """Get symbols for a specific sector (simplified)"""
    sector_map = {
        'technology': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'AVGO', 'ASML', 'CRM', 'ADBE', 'ISRG', 'VRTX', 'REGN', 'ADI'],
        'financial': ['BRK.B', 'JPM', 'BAC', 'WFC', 'GS', 'MS', 'SCHW', 'BLK', 'C', 'USB'],
        'healthcare': ['UNH', 'JNJ', 'PFE', 'ABBV', 'TMO', 'DHR', 'ABT', 'LLY', 'AMGN', 'GILD', 'ZTS', 'CI'],
        'consumer': ['HD', 'DIS', 'NKE', 'MCD', 'SBUX', 'LOW', 'TJX', 'BKNG', 'MAR', 'HLT', 'PG', 'KO', 'PEP', 'WMT', 'COST'],
        'etfs': ['SPY', 'QQQ', 'IWM', 'DIA', 'VTI', 'VOO', 'VEA', 'VWO', 'BND', 'AGG']
    }
    return sector_map.get(sector.lower(), [])

if __name__ == "__main__":
    print(f"S&P 500 Symbols: {len(SP500_SYMBOLS)} symbols")
    print("First 20 symbols:", SP500_SYMBOLS[:20])
    print("Last 20 symbols:", SP500_SYMBOLS[-20:])
