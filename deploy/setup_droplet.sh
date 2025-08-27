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

# Copy environment file
cp .env.example .env

echo "Setup complete! Please edit /opt/bidbot/.env with your Alpaca credentials."
echo "Then run: systemctl enable options-bidbot && systemctl start options-bidbot"
