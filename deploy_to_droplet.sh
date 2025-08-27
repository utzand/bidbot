#!/bin/bash
set -e

echo "🚀 Deploying BidBot to DigitalOcean..."

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "❌ Error: .env file not found!"
    echo "Please create .env file with your Alpaca credentials first."
    exit 1
fi

# Get droplet IP from user
echo "📝 Enter your DigitalOcean droplet IP address:"
read -p "Droplet IP: " DROPLET_IP

if [ -z "$DROPLET_IP" ]; then
    echo "❌ Error: No IP address provided"
    exit 1
fi

echo "🔧 Setting up BidBot on droplet $DROPLET_IP..."

# Copy setup script to droplet
echo "📤 Uploading setup script..."
scp deploy/setup_droplet.sh root@$DROPLET_IP:/tmp/

# Copy environment file to droplet
echo "📤 Uploading environment file..."
scp .env root@$DROPLET_IP:/tmp/

# Execute setup on droplet
echo "🔧 Running setup on droplet..."
ssh root@$DROPLET_IP << 'EOF'
    # Make setup script executable
    chmod +x /tmp/setup_droplet.sh
    
    # Run setup
    /tmp/setup_droplet.sh
    
    # Copy environment file
    cp /tmp/.env /opt/bidbot/.env
    
    # Set proper permissions
    chown -R root:root /opt/bidbot
    chmod 600 /opt/bidbot/.env
    
    # Start the service
    systemctl start options-bidbot
    
    # Check status
    systemctl status options-bidbot --no-pager
EOF

echo "✅ Deployment complete!"
echo ""
echo "📊 To monitor your bot:"
echo "   SSH: ssh root@$DROPLET_IP"
echo "   Check logs: journalctl -u options-bidbot -f"
echo "   Check status: systemctl status options-bidbot"
echo ""
echo "🌐 Dashboard will be available at: http://$DROPLET_IP:8050"
echo ""
echo "📝 Next steps:"
echo "   1. SSH into your droplet: ssh root@$DROPLET_IP"
echo "   2. Check bot status: systemctl status options-bidbot"
echo "   3. View logs: journalctl -u options-bidbot -f"
echo "   4. Access dashboard: http://$DROPLET_IP:8050"
