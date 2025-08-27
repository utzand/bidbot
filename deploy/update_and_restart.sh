#!/bin/bash
set -e

echo "Updating bidbot on DigitalOcean droplet..."

cd /opt/bidbot

# Pull latest changes
git pull origin main

# Activate virtual environment and update dependencies
source .venv/bin/activate
pip install -r src/requirements.txt
pip install -e ./src

# Restart the service
systemctl restart options-bidbot

echo "Update complete! Service restarted."
