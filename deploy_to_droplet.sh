#!/bin/bash
set -e

echo "Deploying bidbot to DigitalOcean droplet..."

# SSH into droplet and run setup
ssh -i ~/.ssh/digitalocean_bidbot root@144.126.214.90 << 'REMOTE_SCRIPT'
    cd /opt/bidbot
    
    if [ ! -d ".git" ]; then
        echo "Initial setup - cloning repository..."
        git clone https://github.com/utzand/bidbot.git .
        
        # Create virtual environment
        python3 -m venv .venv
        source .venv/bin/activate
        
        # Install dependencies
        pip install --upgrade pip
        pip install -r src/requirements.txt
        pip install -e ./src
        
        # Copy environment file
        cp .env.example .env
        
        # Copy systemd service
        cp deploy/options-bidbot.service /etc/systemd/system/
        
        # Reload systemd and enable service
        systemctl daemon-reload
        systemctl enable options-bidbot
        
        echo "Initial setup complete!"
        echo "Please edit /opt/bidbot/.env with your Alpaca credentials"
        echo "Then run: systemctl start options-bidbot"
    else
        echo "Updating existing installation..."
        git pull origin main
        
        # Update dependencies
        source .venv/bin/activate
        pip install -r src/requirements.txt
        pip install -e ./src
        
        # Restart service
        systemctl restart options-bidbot
        
        echo "Update complete! Service restarted."
    fi
    
    echo "Service status:"
    systemctl status options-bidbot --no-pager
REMOTE_SCRIPT

echo "Deployment completed!"
echo "To view logs: ssh -i ~/.ssh/digitalocean_bidbot root@144.126.214.90 'journalctl -u options-bidbot -f'"
