# 🚀 DigitalOcean Deployment Guide

## Prerequisites

1. **DigitalOcean Account** with a droplet created
2. **SSH Access** to your droplet
3. **Alpaca API Keys** configured in `.env` file

## Step 1: Create a DigitalOcean Droplet

### Recommended Specifications:
- **OS**: Ubuntu 22.04 LTS
- **Size**: Basic Plan (1GB RAM, 1 vCPU, 25GB SSD)
- **Region**: Choose closest to you
- **Authentication**: SSH Key (recommended) or Password

### Create Droplet:
1. Go to [DigitalOcean Console](https://cloud.digitalocean.com/)
2. Click "Create" → "Droplets"
3. Choose Ubuntu 22.04 LTS
4. Select Basic Plan ($6/month)
5. Choose your preferred region
6. Add your SSH key or set password
7. Click "Create Droplet"

## Step 2: Get Your Droplet IP

After creation, note your droplet's IP address from the DigitalOcean dashboard.

## Step 3: Deploy Your Bot

### Option A: Automated Deployment (Recommended)

```bash
# Make sure you're in the bidbot directory
cd /Users/andrewutz/bidbot

# Run the deployment script
./deploy_to_droplet.sh
```

The script will:
- ✅ Upload setup files to your droplet
- ✅ Install Python and dependencies
- ✅ Configure the trading bot service
- ✅ Start the bot automatically
- ✅ Show you monitoring commands

### Option B: Manual Deployment

If you prefer manual deployment:

```bash
# SSH into your droplet
ssh root@YOUR_DROPLET_IP

# Clone the repository
git clone https://github.com/utzand/bidbot.git /opt/bidbot
cd /opt/bidbot

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r src/requirements.txt
pip install -e ./src

# Copy your .env file (from your local machine)
# scp .env root@YOUR_DROPLET_IP:/opt/bidbot/

# Set up systemd service
cp deploy/options-bidbot.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable options-bidbot
systemctl start options-bidbot
```

## Step 4: Verify Deployment

### Check Bot Status:
```bash
ssh root@YOUR_DROPLET_IP
systemctl status options-bidbot
```

### View Live Logs:
```bash
journalctl -u options-bidbot -f
```

### Access Dashboard:
Open your browser and go to: `http://YOUR_DROPLET_IP:8050`

## Step 5: Monitoring & Management

### Useful Commands:

```bash
# SSH into droplet
ssh root@YOUR_DROPLET_IP

# Check bot status
systemctl status options-bidbot

# View real-time logs
journalctl -u options-bidbot -f

# Restart bot
systemctl restart options-bidbot

# Stop bot
systemctl stop options-bidbot

# View recent logs
journalctl -u options-bidbot --since "1 hour ago"

# Check disk usage
df -h

# Check memory usage
free -h

# Check running processes
ps aux | grep python
```

### Dashboard Access:
- **URL**: `http://YOUR_DROPLET_IP:8050`
- **Features**: Real-time monitoring, positions, trading history
- **Auto-refresh**: Every 30 seconds

## Step 6: Updates & Maintenance

### Update Bot Code:
```bash
# SSH into droplet
ssh root@YOUR_DROPLET_IP

# Navigate to bot directory
cd /opt/bidbot

# Pull latest changes
git pull origin main

# Update dependencies
source .venv/bin/activate
pip install -r src/requirements.txt
pip install -e ./src

# Restart service
systemctl restart options-bidbot
```

### Update Environment Variables:
```bash
# Edit .env file
nano /opt/bidbot/.env

# Restart service after changes
systemctl restart options-bidbot
```

## Troubleshooting

### Bot Won't Start:
```bash
# Check service status
systemctl status options-bidbot

# View error logs
journalctl -u options-bidbot -n 50

# Check if .env exists
ls -la /opt/bidbot/.env

# Verify Python environment
/opt/bidbot/.venv/bin/python --version
```

### Dashboard Not Accessible:
```bash
# Check if port 8050 is open
netstat -tlnp | grep 8050

# Check firewall
ufw status

# If needed, open port 8050
ufw allow 8050
```

### API Connection Issues:
```bash
# Test Alpaca API connection
cd /opt/bidbot
source .venv/bin/activate
python -c "from options_trader.option_trader import OptionTrader; ot = OptionTrader(); print('API Status:', ot.api is not None)"
```

## Security Considerations

1. **Firewall**: Configure UFW to only allow necessary ports
2. **SSH**: Use SSH keys instead of passwords
3. **Environment**: Keep `.env` file secure (chmod 600)
4. **Updates**: Regularly update your droplet's system packages

## Cost Optimization

- **Basic Droplet**: $6/month (sufficient for testing)
- **Standard Droplet**: $12/month (recommended for production)
- **Monitoring**: Use DigitalOcean's built-in monitoring

## Support

If you encounter issues:
1. Check the logs: `journalctl -u options-bidbot -f`
2. Verify your `.env` configuration
3. Ensure your Alpaca API keys are valid
4. Check DigitalOcean's status page for any outages
