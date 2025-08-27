# BidBot - Options Trading Bot

A Python-based options trading bot using the Alpaca API for paper trading.

## Local Development

### Setup
1. Clone the repository: `git clone https://github.com/utzand/bidbot.git`
2. Create virtual environment: `python3 -m venv .venv`
3. Activate virtual environment: `source .venv/bin/activate`
4. Install dependencies: `pip install -r src/requirements.txt && pip install -e ./src`
5. Copy `.env.example` to `.env` and add your Alpaca API credentials

### Running Locally
- CLI Monitor: `./run_cli_local.sh`
- Web Dashboard: `./run_dash_local.sh`

## Deployment

### DigitalOcean Droplet Setup
1. SSH into your droplet: `ssh root@144.126.214.90`
2. Run setup script: `bash <(curl -s https://raw.githubusercontent.com/utzand/bidbot/main/deploy/setup_droplet.sh)`
3. Edit `/opt/bidbot/.env` with your Alpaca credentials
4. Enable and start service: `systemctl enable options-bidbot && systemctl start options-bidbot`

### Updates
- Local changes: `git add . && git commit -m "message" && git push origin main`
- Deploy to droplet: SSH into droplet and run `/opt/bidbot/deploy/update_and_restart.sh`

## Configuration

Edit `.env` file with your Alpaca API credentials:
```
ALPACA_API_KEY=your_key_here
ALPACA_API_SECRET=your_secret_here
ALPACA_API_BASE_URL=https://paper-api.alpaca.markets/v2
TICKERS=AAPL,MSFT,GOOGL,TSLA
REFRESH_SECONDS=60
DASH_PORT=8050
TRADING_MODE=paper
```

## Service Management

- Check status: `systemctl status options-bidbot`
- View logs: `journalctl -u options-bidbot -f`
- Restart: `systemctl restart options-bidbot`
- Stop: `systemctl stop options-bidbot`
