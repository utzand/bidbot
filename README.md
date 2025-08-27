# BidBot - Options Trading Bot

A Python-based options trading bot using the Alpaca API for paper trading.

## Quick Start

### Local Development
```bash
# Run CLI Monitor
./run_cli_local.sh

# Run Web Dashboard
./run_dash_local.sh
```

### Automated Workflow
```bash
# Run tests
./workflow.sh test

# Commit and push changes
./workflow.sh push "your commit message"

# Deploy to droplet
./workflow.sh deploy

# Full workflow (test + push + deploy)
./workflow.sh full "your commit message"
```

## Local Development

### Setup
1. Clone the repository: `git clone git@github.com:utzand/bidbot.git`
2. Create virtual environment: `python3 -m venv .venv`
3. Activate virtual environment: `source .venv/bin/activate`
4. Install dependencies: `pip install -r src/requirements.txt && pip install -e ./src`
5. Copy `.env.example` to `.env` and add your Alpaca API credentials

### Configuration
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

## Deployment

### Initial Droplet Setup
```bash
# Deploy to DigitalOcean droplet
./deploy_to_droplet.sh
```

This script will:
1. SSH into the droplet (144.126.214.90)
2. Clone the repository to `/opt/bidbot`
3. Set up Python virtual environment
4. Install dependencies
5. Configure systemd service
6. Enable the service

### Manual Droplet Setup (if needed)
```bash
# SSH into droplet
ssh -i ~/.ssh/digitalocean_bidbot root@144.126.214.90

# Run setup script
bash <(curl -s https://raw.githubusercontent.com/utzand/bidbot/main/deploy/setup_droplet.sh)

# Edit environment file
nano /opt/bidbot/.env

# Start service
systemctl start options-bidbot
```

## Service Management

### Check Status
```bash
ssh -i ~/.ssh/digitalocean_bidbot root@144.126.214.90 "systemctl status options-bidbot"
```

### View Logs
```bash
ssh -i ~/.ssh/digitalocean_bidbot root@144.126.214.90 "journalctl -u options-bidbot -f"
```

### Restart Service
```bash
ssh -i ~/.ssh/digitalocean_bidbot root@144.126.214.90 "systemctl restart options-bidbot"
```

### Stop Service
```bash
ssh -i ~/.ssh/digitalocean_bidbot root@144.126.214.90 "systemctl stop options-bidbot"
```

## Development Workflow

1. **Make Changes Locally**
   ```bash
   # Edit files in /Users/andrewutz/bidbot
   ```

2. **Test Changes**
   ```bash
   ./workflow.sh test
   ```

3. **Commit and Push**
   ```bash
   ./workflow.sh push "your commit message"
   ```

4. **Deploy to Production**
   ```bash
   ./workflow.sh deploy
   ```

5. **Or Do Everything at Once**
   ```bash
   ./workflow.sh full "your commit message"
   ```

## File Structure
```
/Users/andrewutz/bidbot/
├── .env                    # Alpaca credentials (not in git)
├── .env.example           # Template for credentials
├── run_cli_local.sh       # Run CLI locally
├── run_dash_local.sh      # Run dashboard locally
├── workflow.sh            # Automated workflow script
├── deploy_to_droplet.sh   # Deploy to droplet
├── deploy/
│   ├── setup_droplet.sh   # Initial droplet setup
│   ├── update_and_restart.sh # Update and restart
│   └── options-bidbot.service # systemd service
└── src/                   # options_alg_trader source
```

## Troubleshooting

### SSH Issues
- Ensure SSH keys are properly configured
- Test GitHub SSH: `ssh -T git@github.com`
- Test Droplet SSH: `ssh -i ~/.ssh/digitalocean_bidbot root@144.126.214.90`

### Service Issues
- Check service status: `systemctl status options-bidbot`
- View logs: `journalctl -u options-bidbot -f`
- Restart service: `systemctl restart options-bidbot`

### Alpaca API Issues
- Verify credentials in `.env` file
- Check API endpoint: `https://paper-api.alpaca.markets/v2`
- Ensure paper trading mode is enabled
