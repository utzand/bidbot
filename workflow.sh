#!/bin/bash
set -e

echo "=== BidBot Development Workflow ==="
echo "1. Local Development"
echo "2. Test Changes"
echo "3. Commit and Push to GitHub"
echo "4. Deploy to DigitalOcean Droplet"
echo ""

# Check if we're in the right directory
if [ ! -f "run_cli_local.sh" ]; then
    echo "Error: Please run this script from /Users/andrewutz/bidbot"
    exit 1
fi

# Function to run tests
run_tests() {
    echo "Running tests..."
    cd src
    source ../.venv/bin/activate
    python tests/test_runner.py
    cd ..
}

# Function to push to GitHub
push_to_github() {
    echo "Pushing to GitHub..."
    git add .
    git commit -m "$1"
    git push origin main
}

# Function to deploy to droplet
deploy_to_droplet() {
    echo "Deploying to DigitalOcean droplet..."
    ./deploy_to_droplet.sh
}

# Main workflow
case "${1:-help}" in
    "test")
        run_tests
        ;;
    "push")
        if [ -z "$2" ]; then
            echo "Usage: $0 push 'commit message'"
            exit 1
        fi
        push_to_github "$2"
        ;;
    "deploy")
        deploy_to_droplet
        ;;
    "full")
        if [ -z "$2" ]; then
            echo "Usage: $0 full 'commit message'"
            exit 1
        fi
        echo "Running full workflow..."
        run_tests
        push_to_github "$2"
        deploy_to_droplet
        ;;
    "help"|*)
        echo "Usage:"
        echo "  $0 test                    - Run tests"
        echo "  $0 push 'message'          - Commit and push to GitHub"
        echo "  $0 deploy                  - Deploy to droplet"
        echo "  $0 full 'message'          - Run full workflow (test + push + deploy)"
        echo ""
        echo "Examples:"
        echo "  $0 test"
        echo "  $0 push 'Fix trading algorithm bug'"
        echo "  $0 deploy"
        echo "  $0 full 'Add new trading strategy'"
        ;;
esac
