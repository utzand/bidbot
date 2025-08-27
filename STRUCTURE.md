# BidBot Repository Structure

## Overview
This repository contains an options trading bot with RSI + breakout strategy, built with Python 3.11 and Alpaca API.

## Directory Structure

```
bidbot/
├── config/                 # Configuration files
│   └── strategy_config.json
├── data/                   # Portfolio and data files
│   ├── portfolio.json
│   └── simulated_portfolio.json
├── deploy/                 # Deployment scripts
│   ├── options-bidbot.service
│   ├── setup_droplet.sh
│   └── update_and_restart.sh
├── logs/                   # Strategy execution logs (auto-created)
├── scripts/                # Executable scripts
│   ├── run_strategy.py     # Main strategy runner
│   ├── test_strategy.py    # Strategy tester
│   ├── run_dash_local.sh   # Start dashboard
│   └── run_cli_local.sh    # Start CLI monitor
├── src/                    # Source code
│   └── options_trader/
│       ├── option_trader.py      # Core trading functionality
│       ├── trading_strategy.py   # RSI + breakout strategy
│       ├── options_monitor.py    # Dashboard (port 8050)
│       ├── cli_monitor.py        # Command-line monitor
│       └── requirements.txt      # Python dependencies
├── tests/                  # Test files
│   ├── test_integration.py
│   ├── test_options_trading.py
│   └── test_runner.py
├── README.md               # Main documentation
├── requirements.txt        # Root dependencies
└── .gitignore             # Git ignore rules
```

## Quick Start

### 1. Setup Environment
```bash
# Create virtual environment
python3.11 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r src/options_trader/requirements.txt
```

### 2. Configure API Keys
Create a `.env` file in the root directory:
```bash
ALPACA_API_KEY=your_api_key_here
ALPACA_API_SECRET=your_secret_here
ALPACA_API_BASE_URL=https://paper-api.alpaca.markets
```

### 3. Test the Strategy
```bash
python scripts/test_strategy.py
```

### 4. Run the Strategy
```bash
python scripts/run_strategy.py
```

### 5. Monitor (Optional)
```bash
# Start dashboard on port 8050
bash scripts/run_dash_local.sh

# Or use CLI monitor
bash scripts/run_cli_local.sh
```

## Key Files

### Strategy Configuration (`config/strategy_config.json`)
- RSI thresholds (oversold/overbought)
- Breakout detection parameters
- Position sizing and risk management
- Trading symbols and execution interval

### Main Strategy (`src/options_trader/trading_strategy.py`)
- Implements RSI + breakout logic
- Runs every 5 minutes by default
- Logs all trades to CSV
- Handles take profit/stop loss

### Trading Engine (`src/options_trader/option_trader.py`)
- Core buy/sell functionality
- Portfolio management
- Alpaca API integration
- Simulation mode fallback

## Logging

Strategy execution logs are saved to `logs/trading_strategy_YYYYMMDD_HHMMSS.csv` with:
- Timestamp and symbol
- RSI values and breakout levels
- Trade details (strike, quantity, price)
- Account information
- Success/error status

## Deployment

Use the scripts in `deploy/` to set up the bot on a DigitalOcean droplet:
```bash
bash deploy_to_droplet.sh
```

## Development

Run tests:
```bash
python -m pytest tests/
```

## File Organization Benefits

1. **Clear Separation**: Scripts, config, data, and source code are separated
2. **Easy Navigation**: Related files are grouped logically
3. **Scalable**: Easy to add new strategies, configs, or scripts
4. **Deployment Ready**: Clean structure for production deployment
5. **Maintainable**: Clear where to find and modify different components
