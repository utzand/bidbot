import os
import json
import datetime as dt
import pandas as pd
import numpy as np
import time
import threading
import logging
from typing import Dict, List, Optional, Tuple
import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import APIError

# Import our option trader
from .option_trader import OptionTrader
from .stock_screener import StockScreener

class TradingStrategy:
    """
    Aggressive RSI + Breakout Options Trading Strategy
    
    Strategy Logic:
    - RSI oversold (< 30) + price breakout above resistance = BUY CALL
    - RSI overbought (> 70) + price breakdown below support = BUY PUT
    - Take profit at 50% gain or stop loss at 25% loss
    - Position sizing: 5% of portfolio per trade
    """
    
    def __init__(self, config_file: str = None):
        """Initialize the trading strategy"""
        self.config = self._load_config(config_file)
        self.trader = OptionTrader()
        self.running = False
        self.last_execution = None
        
        # Strategy parameters
        self.rsi_period = self.config.get('rsi_period', 14)
        self.rsi_oversold = self.config.get('rsi_oversold', 30)
        self.rsi_overbought = self.config.get('rsi_overbought', 70)
        self.breakout_threshold = self.config.get('breakout_threshold', 0.02)  # 2%
        self.position_size_pct = self.config.get('position_size_pct', 0.05)  # 5%
        self.take_profit_pct = self.config.get('take_profit_pct', 0.50)  # 50%
        self.stop_loss_pct = self.config.get('stop_loss_pct', 0.25)  # 25%
        self.execution_interval = self.config.get('execution_interval', 300)  # 5 minutes
        
        # Trading symbols (will be populated by screener)
        self.symbols = self.config.get('symbols', [])  # Empty by default, will be filled by screener
        self.option_expiry_days = self.config.get('option_expiry_days', 30)  # 30 days out
        
        # Initialize stock screener
        self.screener = StockScreener()
        self.max_signals_per_cycle = self.config.get('max_signals_per_cycle', 5)
        
        # Data storage
        self.price_history = {}
        self.indicators = {}
        self.signals = {}
        self.positions = {}
        
        # Setup logging
        self._setup_logging()
        
        # Initialize Alpaca API for data
        self._init_alpaca_api()
        
        print(f"Trading Strategy initialized with {len(self.symbols)} symbols")
        print(f"RSI: {self.rsi_period} period, oversold: {self.rsi_oversold}, overbought: {self.rsi_overbought}")
        print(f"Breakout threshold: {self.breakout_threshold*100}%, Position size: {self.position_size_pct*100}%")
        print(f"Take profit: {self.take_profit_pct*100}%, Stop loss: {self.stop_loss_pct*100}%")
    
    def _load_config(self, config_file: str = None) -> Dict:
        """Load configuration from file or use defaults"""
        if config_file and os.path.exists(config_file):
            with open(config_file, 'r') as f:
                return json.load(f)
        
        # Default configuration
        return {
            'rsi_period': 14,
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'breakout_threshold': 0.02,
            'position_size_pct': 0.05,
            'take_profit_pct': 0.50,
            'stop_loss_pct': 0.25,
            'execution_interval': 300,
            'symbols': ['SPY', 'AAPL', 'TSLA', 'MSFT', 'AMZN'],
            'option_expiry_days': 30,
            'max_positions': 5,
            'enable_trading': True,
            'paper_trading': True
        }
    
    def _setup_logging(self):
        """Setup CSV logging for strategy execution"""
        # Look for logs directory in project root
        log_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'logs')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # Create CSV log file
        timestamp = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file = os.path.join(log_dir, f'trading_strategy_{timestamp}.csv')
        
        # Create CSV header
        header = [
            'timestamp', 'symbol', 'action', 'signal_type', 'rsi', 'price', 
            'breakout_level', 'position_size', 'option_symbol', 'strike', 
            'option_type', 'quantity', 'price_per_contract', 'total_cost',
            'account_cash', 'account_equity', 'success', 'error_message'
        ]
        
        with open(self.log_file, 'w') as f:
            f.write(','.join(header) + '\n')
        
        print(f"Strategy logging to: {self.log_file}")
    
    def _init_alpaca_api(self):
        """Initialize Alpaca API for market data"""
        try:
            api_key = os.environ.get('ALPACA_API_KEY')
            api_secret = os.environ.get('ALPACA_API_SECRET')
            base_url = os.environ.get('ALPACA_API_BASE_URL', 'https://paper-api.alpaca.markets')
            
            if api_key and api_secret:
                self.api = tradeapi.REST(api_key, api_secret, base_url, api_version='v2')
                print("Alpaca API initialized for market data")
            else:
                print("No Alpaca API credentials found - using simulated data")
                self.api = None
        except Exception as e:
            print(f"Error initializing Alpaca API: {e}")
            self.api = None
    
    def _log_execution(self, data: Dict):
        """Log strategy execution to CSV"""
        try:
            with open(self.log_file, 'a') as f:
                row = [
                    data.get('timestamp', dt.datetime.now().isoformat()),
                    data.get('symbol', ''),
                    data.get('action', ''),
                    data.get('signal_type', ''),
                    data.get('rsi', ''),
                    data.get('price', ''),
                    data.get('breakout_level', ''),
                    data.get('position_size', ''),
                    data.get('option_symbol', ''),
                    data.get('strike', ''),
                    data.get('option_type', ''),
                    data.get('quantity', ''),
                    data.get('price_per_contract', ''),
                    data.get('total_cost', ''),
                    data.get('account_cash', ''),
                    data.get('account_equity', ''),
                    data.get('success', ''),
                    data.get('error_message', '')
                ]
                f.write(','.join([str(x) for x in row]) + '\n')
        except Exception as e:
            print(f"Error logging execution: {e}")
    
    def get_market_data(self, symbol: str, lookback_days: int = 30) -> pd.DataFrame:
        """Get historical market data for a symbol"""
        if self.api:
            try:
                end_date = dt.datetime.now()
                start_date = end_date - dt.timedelta(days=lookback_days)
                
                bars = self.api.get_bars(
                    symbol, 
                    '1Day', 
                    start=start_date.strftime('%Y-%m-%d'),
                    end=end_date.strftime('%Y-%m-%d')
                ).df
                
                if not bars.empty:
                    return bars
            except Exception as e:
                print(f"Error getting market data for {symbol}: {e}")
        
        # Fallback to simulated data
        return self._generate_simulated_data(symbol, lookback_days)
    
    def _generate_simulated_data(self, symbol: str, lookback_days: int) -> pd.DataFrame:
        """Generate simulated market data for testing"""
        end_date = dt.datetime.now()
        dates = pd.date_range(end=end_date, periods=lookback_days, freq='D')
        
        # Generate realistic price movements
        base_price = 150.0  # Base price
        prices = [base_price]
        
        for i in range(1, len(dates)):
            # Random walk with some trend
            change = np.random.normal(0, 0.02)  # 2% daily volatility
            new_price = prices[-1] * (1 + change)
            prices.append(max(new_price, 10))  # Minimum price of $10
        
        # Generate OHLC data
        data = []
        for i, date in enumerate(dates):
            price = prices[i]
            high = price * (1 + abs(np.random.normal(0, 0.01)))
            low = price * (1 - abs(np.random.normal(0, 0.01)))
            open_price = price * (1 + np.random.normal(0, 0.005))
            volume = int(np.random.uniform(1000000, 10000000))
            
            data.append({
                'open': open_price,
                'high': high,
                'low': low,
                'close': price,
                'volume': volume
            })
        
        df = pd.DataFrame(data, index=dates)
        return df
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI indicator"""
        if len(prices) < period + 1:
            return 50.0  # Neutral RSI if not enough data
        
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.iloc[-1]
    
    def detect_breakout(self, prices: pd.Series, threshold: float = 0.02) -> Tuple[bool, str, float]:
        """Detect price breakouts above resistance or breakdowns below support"""
        if len(prices) < 20:
            return False, 'insufficient_data', 0.0
        
        # Calculate recent high/low levels
        recent_high = prices.tail(10).max()
        recent_low = prices.tail(10).min()
        current_price = prices.iloc[-1]
        
        # Check for breakout above resistance
        if current_price > recent_high * (1 + threshold):
            return True, 'breakout_up', recent_high
        
        # Check for breakdown below support
        if current_price < recent_low * (1 - threshold):
            return True, 'breakdown_down', recent_low
        
        return False, 'no_breakout', 0.0
    
    def generate_signals(self) -> Dict[str, Dict]:
        """Generate trading signals using the stock screener"""
        print("Screening S&P 500 universe for trading opportunities...")
        
        # Get top signals from screener
        top_signals = self.screener.get_top_signals(max_results=self.max_signals_per_cycle)
        
        # Convert to our signal format
        signals = {}
        for signal in top_signals:
            symbol = signal['symbol']
            signals[symbol] = {
                'symbol': symbol,
                'price': signal['price'],
                'rsi': signal['rsi'],
                'breakout': signal['breakout'],
                'breakout_type': signal['breakout_type'],
                'breakout_level': signal['breakout_level'],
                'action': signal['signal'],
                'option_type': signal['option_type'],
                'strike': signal['strike'],
                'reason': signal['reason'],
                'total_score': signal['total_score']
            }
        
        print(f"Found {len(signals)} high-quality trading signals")
        return signals
    
    def calculate_position_size(self, symbol: str, signal: Dict) -> int:
        """Calculate position size based on account equity and risk management"""
        try:
            account_info = self.trader.get_account_info()
            equity = account_info.get('equity', 100000)  # Default if not available
            
            # Calculate position size in dollars
            position_dollars = equity * self.position_size_pct
            
            # Estimate option price (simplified)
            current_price = signal['price']
            strike = signal['strike']
            
            if signal['option_type'] == 'call':
                # Rough estimate: intrinsic value + time value
                intrinsic = max(0, current_price - strike)
                time_value = current_price * 0.1  # 10% of underlying price
                option_price = intrinsic + time_value
            else:  # put
                intrinsic = max(0, strike - current_price)
                time_value = current_price * 0.1
                option_price = intrinsic + time_value
            
            # Calculate number of contracts (each contract is 100 shares)
            contracts = int(position_dollars / (option_price * 100))
            
            return max(1, contracts)  # Minimum 1 contract
            
        except Exception as e:
            print(f"Error calculating position size: {e}")
            return 1  # Default to 1 contract
    
    def get_option_expiry(self) -> str:
        """Get option expiry date (30 days from now)"""
        expiry_date = dt.datetime.now() + dt.timedelta(days=self.option_expiry_days)
        return expiry_date.strftime('%Y-%m-%d')
    
    def execute_signals(self, signals: Dict) -> List[Dict]:
        """Execute trading signals"""
        executions = []
        
        for symbol, signal in signals.items():
            try:
                if signal['action'] == 'hold':
                    continue
                
                # Calculate position size
                quantity = self.calculate_position_size(symbol, signal)
                if quantity == 0:
                    continue
                
                # Execute the trade
                result = self.trader.buy_option(
                    ticker=symbol,
                    expiration_date='2025-09-26',  # Fixed expiration
                    strike_price=signal['strike'],
                    option_type=signal['option_type'],
                    quantity=quantity
                )
                
                # ONLY LOG SUCCESSFUL ALPACA TRADES
                if result.get('success', False) and result.get('status') == 'filled':
                    # Get account info for logging
                    account_info = self.trader.get_account_info()
                    
                    # Log the successful execution
                    log_data = {
                        'timestamp': dt.datetime.now().isoformat(),
                        'symbol': symbol,
                        'action': 'buy',
                        'signal_type': signal['reason'],
                        'rsi': signal['rsi'],
                        'price': signal['price'],
                        'breakout_level': signal['breakout_level'],
                        'position_size': quantity,
                        'option_symbol': result.get('symbol', ''),
                        'strike': signal['strike'],
                        'option_type': signal['option_type'],
                        'quantity': quantity,
                        'price_per_contract': result.get('filled_avg_price', 0),
                        'total_cost': result.get('filled_avg_price', 0) * quantity * 100 if result.get('filled_avg_price') else 0,
                        'account_cash': account_info.get('cash', 0),
                        'account_equity': account_info.get('equity', 0),
                        'success': True,
                        'error_message': ''
                    }
                    
                    self._log_execution(log_data)
                    executions.append(result)
                    
                    print(f"EXECUTED: {signal['action'].upper()} {quantity} {symbol} {signal['option_type']} @ ${signal['strike']}")
                    print(f"Reason: {signal['reason']}")
                    print(f"Result: {result.get('status', 'unknown')}")
                else:
                    # Log failed trades for debugging (but don't count as executions)
                    print(f"TRADE FAILED: {symbol} - {result.get('error_message', 'Unknown error')}")
                    print(f"Status: {result.get('status', 'unknown')}")
                    print(f"Success: {result.get('success', False)}")
                
            except Exception as e:
                print(f"Error executing signal for {symbol}: {e}")
                # Don't log failed executions
        
        return executions
    
    def check_exit_signals(self) -> List[Dict]:
        """Check for exit signals (take profit/stop loss) and execute them"""
        exits = []
        positions = self.trader.get_positions()
        
        for position in positions:
            try:
                # Parse option symbol to get strike and type
                symbol = position['symbol']
                current_price = position['current_price']
                entry_price = position['avg_entry_price']
                quantity = int(position['qty'])
                
                # Calculate P&L percentage
                if entry_price > 0:
                    pnl_pct = (current_price - entry_price) / entry_price
                else:
                    continue
                
                # Check take profit
                if pnl_pct >= self.take_profit_pct:
                    print(f"TAKE PROFIT SIGNAL: {symbol} at {pnl_pct*100:.1f}% gain")
                    
                    # Parse option symbol to get details for selling
                    # Format: SYMBOL + YYMMDD + C/P + STRIKE
                    # Example: AAPL250926C00200000
                    if len(symbol) >= 15:
                        ticker = symbol[:symbol.find('250926')]  # Extract ticker
                        expiration_date = '2025-09-26'  # Fixed expiration
                        option_type = 'call' if 'C' in symbol else 'put'
                        
                        # Extract strike price (last 8 digits)
                        strike_str = symbol[-8:]
                        strike_price = float(strike_str) / 1000.0
                        
                        # Execute sell order
                        result = self.trader.sell_option(
                            ticker=ticker,
                            expiration_date=expiration_date,
                            strike_price=strike_price,
                            option_type=option_type,
                            quantity=quantity
                        )
                        
                        exits.append({
                            'symbol': symbol,
                            'action': 'sell',
                            'reason': 'take_profit',
                            'pnl_pct': pnl_pct,
                            'result': result
                        })
                        
                        # Log the exit
                        account_info = self.trader.get_account_info()
                        log_data = {
                            'timestamp': dt.datetime.now().isoformat(),
                            'symbol': ticker,
                            'action': 'sell',
                            'signal_type': f'Take profit ({pnl_pct*100:.1f}%)',
                            'rsi': 0,  # Not relevant for exits
                            'price': current_price,
                            'breakout_level': 0,
                            'position_size': quantity,
                            'option_symbol': symbol,
                            'strike': strike_price,
                            'option_type': option_type,
                            'quantity': quantity,
                            'price_per_contract': result.get('filled_avg_price', 0),
                            'total_cost': result.get('filled_avg_price', 0) * quantity * 100 if result.get('filled_avg_price') else 0,
                            'account_cash': account_info.get('cash', 0),
                            'account_equity': account_info.get('equity', 0),
                            'success': result.get('success', False),
                            'error_message': result.get('error_message', '')
                        }
                        self._log_execution(log_data)
                
                # Check stop loss
                elif pnl_pct <= -self.stop_loss_pct:
                    print(f"STOP LOSS SIGNAL: {symbol} at {pnl_pct*100:.1f}% loss")
                    
                    # Parse option symbol to get details for selling
                    if len(symbol) >= 15:
                        ticker = symbol[:symbol.find('250926')]  # Extract ticker
                        expiration_date = '2025-09-26'  # Fixed expiration
                        option_type = 'call' if 'C' in symbol else 'put'
                        
                        # Extract strike price (last 8 digits)
                        strike_str = symbol[-8:]
                        strike_price = float(strike_str) / 1000.0
                        
                        # Execute sell order
                        result = self.trader.sell_option(
                            ticker=ticker,
                            expiration_date=expiration_date,
                            strike_price=strike_price,
                            option_type=option_type,
                            quantity=quantity
                        )
                        
                        exits.append({
                            'symbol': symbol,
                            'action': 'sell',
                            'reason': 'stop_loss',
                            'pnl_pct': pnl_pct,
                            'result': result
                        })
                        
                        # Log the exit
                        account_info = self.trader.get_account_info()
                        log_data = {
                            'timestamp': dt.datetime.now().isoformat(),
                            'symbol': ticker,
                            'action': 'sell',
                            'signal_type': f'Stop loss ({pnl_pct*100:.1f}%)',
                            'rsi': 0,  # Not relevant for exits
                            'price': current_price,
                            'breakout_level': 0,
                            'position_size': quantity,
                            'option_symbol': symbol,
                            'strike': strike_price,
                            'option_type': option_type,
                            'quantity': quantity,
                            'price_per_contract': result.get('filled_avg_price', 0),
                            'total_cost': result.get('filled_avg_price', 0) * quantity * 100 if result.get('filled_avg_price') else 0,
                            'account_cash': account_info.get('cash', 0),
                            'account_equity': account_info.get('equity', 0),
                            'success': result.get('success', False),
                            'error_message': result.get('error_message', '')
                        }
                        self._log_execution(log_data)
                
            except Exception as e:
                print(f"Error checking exit signals for {position.get('symbol', 'unknown')}: {e}")
        
        return exits
    
    def run_strategy_cycle(self):
        """Run one complete strategy cycle"""
        try:
            print(f"\n--- Strategy Cycle: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")
            
            # Generate signals
            signals = self.generate_signals()
            
            # Log signals
            for symbol, signal in signals.items():
                if signal['action'] != 'hold':
                    print(f"SIGNAL: {symbol} - {signal['action'].upper()} {signal['option_type']} @ ${signal['strike']}")
                    print(f"  RSI: {signal['rsi']:.1f}, Price: ${signal['price']:.2f}")
                    print(f"  Reason: {signal['reason']}")
            
            # Execute signals
            executions = self.execute_signals(signals)
            
            # Check exit signals
            exits = self.check_exit_signals()
            
            # Log summary
            print(f"Cycle complete: {len([s for s in signals.values() if s['action'] != 'hold'])} signals, {len(executions)} executions, {len(exits)} exits")
            
            self.last_execution = dt.datetime.now()
            
        except Exception as e:
            print(f"Error in strategy cycle: {e}")
    
    def start(self):
        """Start the trading strategy"""
        if self.running:
            print("Strategy is already running")
            return
        
        self.running = True
        print(f"Starting trading strategy - executing every {self.execution_interval} seconds")
        print(f"Trading symbols: {', '.join(self.symbols)}")
        print(f"Log file: {self.log_file}")
        
        # Run initial cycle
        self.run_strategy_cycle()
        
        # Start continuous execution
        while self.running:
            try:
                time.sleep(self.execution_interval)
                if self.running:
                    self.run_strategy_cycle()
            except KeyboardInterrupt:
                print("\nStopping trading strategy...")
                self.stop()
                break
            except Exception as e:
                print(f"Error in strategy loop: {e}")
                time.sleep(60)  # Wait a minute before retrying
    
    def stop(self):
        """Stop the trading strategy"""
        self.running = False
        print("Trading strategy stopped")
    
    def get_status(self) -> Dict:
        """Get current strategy status"""
        return {
            'running': self.running,
            'last_execution': self.last_execution.isoformat() if self.last_execution else None,
            'symbols': self.symbols,
            'execution_interval': self.execution_interval,
            'log_file': self.log_file,
            'positions': self.trader.get_positions(),
            'account_info': self.trader.get_account_info()
        }


def main():
    """Main entry point for the trading strategy"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Options Trading Strategy')
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--symbols', nargs='+', help='Trading symbols')
    parser.add_argument('--interval', type=int, default=300, help='Execution interval in seconds')
    parser.add_argument('--test', action='store_true', help='Run in test mode')
    
    args = parser.parse_args()
    
    # Create strategy
    strategy = TradingStrategy(config_file=args.config)
    
    # Override symbols if provided
    if args.symbols:
        strategy.symbols = args.symbols
    
    # Override interval if provided
    if args.interval:
        strategy.execution_interval = args.interval
    
    # Start strategy
    try:
        strategy.start()
    except KeyboardInterrupt:
        print("\nStrategy interrupted by user")
        strategy.stop()


if __name__ == "__main__":
    main()
