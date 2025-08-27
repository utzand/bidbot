#!/usr/bin/env python3
"""
Simple script to run the options trading strategy
"""

import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from options_trader.trading_strategy import TradingStrategy

def main():
    """Run the trading strategy"""
    print("Starting Options Trading Strategy...")
    print("=" * 50)
    
    # Load config from config directory
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'strategy_config.json')
    
    # Create and start the strategy
    strategy = TradingStrategy(config_file=config_path)
    
    try:
        strategy.start()
    except KeyboardInterrupt:
        print("\nStrategy stopped by user")
        strategy.stop()
    except Exception as e:
        print(f"Error running strategy: {e}")
        strategy.stop()

if __name__ == "__main__":
    main()
