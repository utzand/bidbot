#!/bin/bash
set -e

echo "Setting up bidbot on DigitalOcean droplet..."

# Update system
apt-get update
apt-get install -y python3 python3-pip python3-venv git curl

# Create bidbot directory
mkdir -p /opt/bidbot
cd /opt/bidbot

# Clone repository
if [ -d ".git" ]; then
    echo "Repository exists, pulling latest changes..."
    git pull origin main
else
    echo "Cloning repository..."
    git clone https://github.com/utzand/bidbot.git .
fi

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r src/requirements.txt
pip install -e ./src

# Copy environment file if it doesn't exist
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "Created .env from template. Please edit with your Alpaca credentials."
fi

# Copy systemd service
cp deploy/options-bidbot.service /etc/systemd/system/

# Reload systemd and enable service
systemctl daemon-reload
systemctl enable options-bidbot

echo "Setup complete!"
echo "Next steps:"
echo "1. Edit /opt/bidbot/.env with your Alpaca credentials"
echo "2. Start the service: systemctl start options-bidbot"
echo "3. Check status: systemctl status options-bidbot"
