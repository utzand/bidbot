#!/usr/bin/env python3
"""
Test script to run the trading strategy for a few cycles
"""

import sys
import os
import time

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from options_trader.trading_strategy import TradingStrategy

def test_strategy():
    """Test the trading strategy for a few cycles"""
    print("Testing Options Trading Strategy...")
    print("=" * 50)
    
    # Load config from config directory
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'strategy_config.json')
    
    # Create strategy with shorter interval for testing
    strategy = TradingStrategy(config_file=config_path)
    strategy.execution_interval = 60  # 1 minute for testing
    strategy.symbols = ['SPY', 'AAPL']  # Fewer symbols for testing
    
    print(f"Testing with symbols: {strategy.symbols}")
    print(f"Test interval: {strategy.execution_interval} seconds")
    print(f"Log file: {strategy.log_file}")
    
    try:
        # Run a few cycles manually
        for i in range(3):
            print(f"\n--- Test Cycle {i+1}/3 ---")
            strategy.run_strategy_cycle()
            
            if i < 2:  # Don't sleep after the last cycle
                print(f"Waiting {strategy.execution_interval} seconds...")
                time.sleep(strategy.execution_interval)
        
        # Show final status
        status = strategy.get_status()
        print(f"\n--- Final Status ---")
        print(f"Positions: {len(status['positions'])}")
        print(f"Account equity: ${status['account_info'].get('equity', 0):,.2f}")
        print(f"Log file: {status['log_file']}")
        
    except Exception as e:
        print(f"Error during test: {e}")

if __name__ == "__main__":
    test_strategy()
