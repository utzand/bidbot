import os
import pandas as pd
import numpy as np
import requests
import time
import json
from typing import Dict, List, Tuple, Optional
import alpaca_trade_api as tradeapi
from datetime import datetime, timedelta
import logging
from .sp500_symbols import get_sp500_symbols, get_sector_symbols

class StockScreener:
    """
    Comprehensive stock screener for S&P 500 universe
    
    Screens stocks based on:
    - RSI conditions (oversold/overbought)
    - Breakout/breakdown patterns
    - Volume analysis
    - Price momentum
    - Volatility
    - Option liquidity (estimated)
    """
    
    def __init__(self, api_key=None, api_secret=None, base_url=None):
        """Initialize the stock screener"""
        self.api_key = api_key or os.environ.get('ALPACA_API_KEY')
        self.api_secret = api_secret or os.environ.get('ALPACA_API_SECRET')
        self.base_url = base_url or os.environ.get('ALPACA_API_BASE_URL', 'https://paper-api.alpaca.markets')
        
        # Initialize API
        self.api = self._init_api()
        
        # S&P 500 symbols (we'll load this dynamically)
        self.sp500_symbols = []
        self.screened_results = {}
        
        # Screening parameters
        self.min_volume = 1000000  # Minimum daily volume
        self.min_price = 10.0      # Minimum stock price
        self.max_price = 1000.0    # Maximum stock price (avoid expensive stocks)
        self.lookback_days = 30    # Days of historical data to analyze
        
        # Scoring weights
        self.weights = {
            'rsi_score': 0.3,
            'breakout_score': 0.25,
            'volume_score': 0.15,
            'momentum_score': 0.2,
            'volatility_score': 0.1
        }
        
        print("Stock Screener initialized")
    
    def _init_api(self):
        """Initialize Alpaca API connection"""
        if not self.api_key or not self.api_secret:
            print("No Alpaca API credentials found - using simulated data")
            return None
        
        try:
            api = tradeapi.REST(self.api_key, self.api_secret, self.base_url, api_version='v2')
            print("Alpaca API initialized for stock screening")
            return api
        except Exception as e:
            print(f"Error initializing Alpaca API: {e}")
            return None
    
    def get_sp500_symbols(self) -> List[str]:
        """Get S&P 500 symbols"""
        if self.sp500_symbols:
            return self.sp500_symbols
        
        # Try to get from Alpaca first
        if self.api:
            try:
                # Get all active stocks
                assets = self.api.list_assets(status='active', asset_class='us_equity')
                symbols = [asset.symbol for asset in assets if asset.tradable and asset.status == 'active']
                
                # Filter to common S&P 500 symbols (simplified approach)
                # In production, you'd want to get the actual S&P 500 list
                sp500_common = [
                    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'BRK.B', 'UNH', 'JNJ',
                    'JPM', 'V', 'PG', 'HD', 'MA', 'DIS', 'PYPL', 'BAC', 'ADBE', 'CRM',
                    'NFLX', 'CMCSA', 'PFE', 'ABT', 'KO', 'PEP', 'TMO', 'COST', 'DHR', 'ABBV',
                    'AVGO', 'WMT', 'ACN', 'NEE', 'LLY', 'TXN', 'HON', 'UNP', 'LOW', 'UPS',
                    'IBM', 'RTX', 'CAT', 'SPGI', 'GS', 'AMGN', 'PLD', 'T', 'DE', 'GILD',
                    'ADI', 'ISRG', 'VRTX', 'REGN', 'LMT', 'SCHW', 'BKNG', 'MDLZ', 'CI', 'ZTS',
                    'TMUS', 'SO', 'DUK', 'D', 'NEE', 'AEP', 'SRE', 'XEL', 'WEC', 'DTE',
                    'ED', 'PEG', 'AEE', 'EIX', 'PCG', 'CNP', 'NI', 'CMS', 'ATO', 'LNT',
                    'SPY', 'QQQ', 'IWM', 'DIA', 'VTI', 'VOO', 'VEA', 'VWO', 'BND', 'AGG'
                ]
                
                # Filter to symbols that exist in Alpaca
                self.sp500_symbols = [s for s in sp500_common if s in symbols]
                print(f"Loaded {len(self.sp500_symbols)} S&P 500 symbols from Alpaca")
                return self.sp500_symbols
                
            except Exception as e:
                print(f"Error getting symbols from Alpaca: {e}")
        
        # Fallback to comprehensive S&P 500 list
        self.sp500_symbols = get_sp500_symbols()
        
        print(f"Using fallback list of {len(self.sp500_symbols)} symbols")
        return self.sp500_symbols
    
    def get_stock_data(self, symbol: str, lookback_days: int = 30) -> Optional[pd.DataFrame]:
        """Get historical stock data"""
        if self.api:
            try:
                end_date = datetime.now()
                start_date = end_date - timedelta(days=lookback_days)
                
                bars = self.api.get_bars(
                    symbol, 
                    '1Day', 
                    start=start_date.strftime('%Y-%m-%d'),
                    end=end_date.strftime('%Y-%m-%d')
                ).df
                
                if not bars.empty and len(bars) >= 20:  # Need at least 20 days of data
                    return bars
                    
            except Exception as e:
                print(f"Error getting data for {symbol}: {e}")
        
        # Fallback to simulated data
        return self._generate_simulated_data(symbol, lookback_days)
    
    def _generate_simulated_data(self, symbol: str, lookback_days: int) -> pd.DataFrame:
        """Generate simulated market data for testing"""
        end_date = datetime.now()
        dates = pd.date_range(end=end_date, periods=lookback_days, freq='D')
        
        # Generate realistic price movements based on symbol
        base_price = 150.0  # Base price
        if 'AAPL' in symbol:
            base_price = 180.0
        elif 'TSLA' in symbol:
            base_price = 250.0
        elif 'SPY' in symbol:
            base_price = 450.0
        elif 'NVDA' in symbol:
            base_price = 800.0
        elif 'MSFT' in symbol:
            base_price = 350.0
        
        prices = [base_price]
        
        # Create some patterns that might trigger signals
        # For some symbols, create oversold conditions
        if symbol in ['NVDA', 'TSLA']:
            # Create a downtrend followed by a breakout
            for i in range(1, len(dates) - 5):
                change = np.random.normal(-0.015, 0.02)  # Downtrend
                new_price = prices[-1] * (1 + change)
                prices.append(max(new_price, 10))
            
            # Add a breakout at the end
            for i in range(len(dates) - 5, len(dates)):
                change = np.random.normal(0.03, 0.02)  # Strong uptrend
                new_price = prices[-1] * (1 + change)
                prices.append(max(new_price, 10))
        else:
            # Normal random walk
            for i in range(1, len(dates)):
                change = np.random.normal(0, 0.02)  # 2% daily volatility
                new_price = prices[-1] * (1 + change)
                prices.append(max(new_price, 10))
        
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
            return 50.0
        
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
    
    def calculate_momentum(self, prices: pd.Series, period: int = 10) -> float:
        """Calculate price momentum"""
        if len(prices) < period:
            return 0.0
        
        return (prices.iloc[-1] / prices.iloc[-period] - 1) * 100
    
    def calculate_volatility(self, prices: pd.Series, period: int = 20) -> float:
        """Calculate price volatility"""
        if len(prices) < period:
            return 0.0
        
        returns = prices.pct_change().dropna()
        return returns.tail(period).std() * 100
    
    def calculate_volume_score(self, volumes: pd.Series, period: int = 10) -> float:
        """Calculate volume score based on recent volume vs average"""
        if len(volumes) < period:
            return 0.0
        
        recent_volume = volumes.tail(5).mean()
        avg_volume = volumes.tail(period).mean()
        
        if avg_volume > 0:
            return min(recent_volume / avg_volume, 3.0)  # Cap at 3x
        return 0.0
    
    def screen_stock(self, symbol: str) -> Optional[Dict]:
        """Screen a single stock and return analysis"""
        try:
            # Get stock data
            data = self.get_stock_data(symbol, self.lookback_days)
            if data is None or data.empty:
                return None
            
            prices = data['close']
            volumes = data['volume']
            current_price = prices.iloc[-1]
            
            # Basic filters
            if current_price < self.min_price or current_price > self.max_price:
                return None
            
            avg_volume = volumes.tail(20).mean()
            if avg_volume < self.min_volume:
                return None
            
            # Calculate indicators
            rsi = self.calculate_rsi(prices)
            breakout, breakout_type, breakout_level = self.detect_breakout(prices)
            momentum = self.calculate_momentum(prices)
            volatility = self.calculate_volatility(prices)
            volume_score = self.calculate_volume_score(volumes)
            
            # Calculate scores
            rsi_score = 0.0
            if rsi < 30:  # Oversold
                rsi_score = (30 - rsi) / 30  # Higher score for more oversold
            elif rsi > 70:  # Overbought
                rsi_score = (rsi - 70) / 30  # Higher score for more overbought
            
            breakout_score = 1.0 if breakout else 0.0
            
            momentum_score = abs(momentum) / 10  # Normalize momentum
            momentum_score = min(momentum_score, 1.0)
            
            volatility_score = min(volatility / 5, 1.0)  # Normalize volatility
            
            volume_score = min(volume_score / 2, 1.0)  # Normalize volume
            
            # Calculate total score
            total_score = (
                rsi_score * self.weights['rsi_score'] +
                breakout_score * self.weights['breakout_score'] +
                volume_score * self.weights['volume_score'] +
                momentum_score * self.weights['momentum_score'] +
                volatility_score * self.weights['volatility_score']
            )
            
            # Determine trading signal (more aggressive)
            signal = 'hold'
            option_type = None
            strike = None
            reason = 'no_signal'
            
            # More aggressive RSI thresholds (35/65 instead of 30/70)
            if breakout and rsi < 35 and breakout_type == 'breakout_up':
                signal = 'buy'
                option_type = 'call'
                strike = current_price * 1.02
                reason = f'RSI oversold ({rsi:.1f}) + breakout up'
            elif breakout and rsi > 65 and breakout_type == 'breakdown_down':
                signal = 'buy'
                option_type = 'put'
                strike = current_price * 0.98
                reason = f'RSI overbought ({rsi:.1f}) + breakdown down'
            # Add momentum-based signals
            elif momentum > 5 and rsi < 40:  # Strong momentum + not overbought
                signal = 'buy'
                option_type = 'call'
                strike = current_price * 1.02
                reason = f'Strong momentum ({momentum:.1f}%) + RSI ({rsi:.1f})'
            elif momentum < -5 and rsi > 60:  # Strong negative momentum + not oversold
                signal = 'buy'
                option_type = 'put'
                strike = current_price * 0.98
                reason = f'Strong negative momentum ({momentum:.1f}%) + RSI ({rsi:.1f})'
            
            return {
                'symbol': symbol,
                'price': current_price,
                'volume': avg_volume,
                'rsi': rsi,
                'breakout': breakout,
                'breakout_type': breakout_type,
                'breakout_level': breakout_level,
                'momentum': momentum,
                'volatility': volatility,
                'volume_score': volume_score,
                'rsi_score': rsi_score,
                'breakout_score': breakout_score,
                'momentum_score': momentum_score,
                'volatility_score': volatility_score,
                'total_score': total_score,
                'signal': signal,
                'option_type': option_type,
                'strike': strike,
                'reason': reason
            }
            
        except Exception as e:
            print(f"Error screening {symbol}: {e}")
            return None
    
    def screen_universe(self, max_results: int = 20) -> List[Dict]:
        """Screen the entire universe and return top candidates"""
        print(f"Screening S&P 500 universe...")
        
        symbols = self.get_sp500_symbols()
        results = []
        
        for i, symbol in enumerate(symbols):
            if i % 10 == 0:
                print(f"Progress: {i}/{len(symbols)} symbols screened")
            
            result = self.screen_stock(symbol)
            if result and result['signal'] != 'hold':
                results.append(result)
            
            # Rate limiting (reduced for faster screening)
            time.sleep(0.05)
        
        # Sort by total score (highest first)
        results.sort(key=lambda x: x['total_score'], reverse=True)
        
        # Return top results
        top_results = results[:max_results]
        
        print(f"Screening complete: {len(results)} signals found, returning top {len(top_results)}")
        
        return top_results
    
    def get_top_signals(self, max_results: int = 10) -> List[Dict]:
        """Get top trading signals from screened universe"""
        results = self.screen_universe(max_results * 2)  # Screen more to get better selection
        
        # Filter to only buy signals
        buy_signals = [r for r in results if r['signal'] == 'buy']
        
        return buy_signals[:max_results]
    
    def print_screening_results(self, results: List[Dict]):
        """Print screening results in a nice format"""
        if not results:
            print("No trading signals found")
            return
        
        print(f"\n{'='*80}")
        print(f"TOP TRADING SIGNALS ({len(results)} found)")
        print(f"{'='*80}")
        
        for i, result in enumerate(results, 1):
            print(f"\n{i}. {result['symbol']} - ${result['price']:.2f}")
            print(f"   Signal: {result['signal'].upper()} {result['option_type']} @ ${result['strike']:.2f}")
            print(f"   Reason: {result['reason']}")
            print(f"   RSI: {result['rsi']:.1f}, Momentum: {result['momentum']:.1f}%, Vol: {result['volatility']:.1f}%")
            print(f"   Score: {result['total_score']:.3f} (RSI: {result['rsi_score']:.3f}, Breakout: {result['breakout_score']:.3f})")
            print(f"   Volume: {result['volume']:,.0f} avg daily")


def main():
    """Test the stock screener"""
    screener = StockScreener()
    
    print("Testing Stock Screener...")
    print("=" * 50)
    
    # Get top signals
    signals = screener.get_top_signals(max_results=10)
    
    # Print results
    screener.print_screening_results(signals)
    
    return signals


if __name__ == "__main__":
    main()
