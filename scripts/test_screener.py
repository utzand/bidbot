#!/usr/bin/env python3
"""
Test script for the stock screener
"""

import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from options_trader.stock_screener import StockScreener

def test_screener():
    """Test the stock screener"""
    print("Testing Stock Screener with S&P 500 Universe...")
    print("=" * 60)
    
    # Create screener
    screener = StockScreener()
    
    # Get S&P 500 symbols
    symbols = screener.get_sp500_symbols()
    print(f"Loaded {len(symbols)} S&P 500 symbols")
    
    # Test screening a few individual stocks
    print("\nTesting individual stock screening...")
    test_symbols = ['AAPL', 'TSLA', 'SPY', 'MSFT', 'NVDA']
    
    for symbol in test_symbols:
        print(f"\nScreening {symbol}...")
        result = screener.screen_stock(symbol)
        if result:
            print(f"  Price: ${result['price']:.2f}")
            print(f"  RSI: {result['rsi']:.1f}")
            print(f"  Signal: {result['signal']}")
            print(f"  Score: {result['total_score']:.3f}")
            if result['signal'] != 'hold':
                print(f"  Action: {result['signal'].upper()} {result['option_type']} @ ${result['strike']:.2f}")
        else:
            print(f"  No data available")
    
    # Test full universe screening
    print(f"\n{'='*60}")
    print("Screening entire S&P 500 universe...")
    print("This may take a few minutes...")
    
    signals = screener.get_top_signals(max_results=10)
    
    # Print results
    screener.print_screening_results(signals)
    
    return signals

if __name__ == "__main__":
    test_screener()
