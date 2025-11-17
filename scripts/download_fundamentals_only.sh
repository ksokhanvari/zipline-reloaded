#!/bin/bash
# Download Sharadar fundamentals for an existing bundle (Docker wrapper)
# Usage: ./scripts/download_fundamentals_only.sh [bundle_name]

set -e

BUNDLE_NAME="${1:-sharadar}"

echo "=================================================="
echo "Sharadar Fundamentals Download (Standalone)"
echo "=================================================="
echo "Bundle: $BUNDLE_NAME"
echo ""

# Run Python script inside Docker container
docker exec zipline-reloaded-jupyter python /app/scripts/download_fundamentals_only.py \
    --bundle "$BUNDLE_NAME"

echo ""
echo "✓ Done! Fundamentals added to bundle: $BUNDLE_NAME"
