#!/bin/bash
# Build libroutecrack.so
set -e
cd "$(dirname "$0")"
g++ -std=c++17 -O2 -shared -fPIC -o libroutecrack.so rc_predictor_lib.cpp -lpthread
echo "Built: $(pwd)/libroutecrack.so"
