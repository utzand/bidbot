#!/bin/bash
cd "$(dirname "$0")/.."
source .venv/bin/activate
cd src && python -m options_trader.options_monitor
