#!/bin/bash
cd /Users/andrewutz/bidbot
source .venv/bin/activate
cd src && python -m options_trader.options_monitor
