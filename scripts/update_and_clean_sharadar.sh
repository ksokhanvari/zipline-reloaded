#!/bin/bash
# Update Sharadar bundle and clean old ingestions
# Usage: ./scripts/update_and_clean_sharadar.sh

set -e  # Exit on error

BUNDLE_NAME="${1:-sharadar}"
KEEP_LAST="${2:-2}"

echo "=================================================="
echo "Sharadar Update & Cleanup"
echo "=================================================="
echo "Bundle: $BUNDLE_NAME"
echo "Keep last: $KEEP_LAST ingestions"
echo ""

# Step 1: Ingest new data
echo "Step 1/2: Ingesting new data..."
docker exec zipline-reloaded-jupyter zipline ingest -b "$BUNDLE_NAME"

echo ""
echo "Step 2/2: Cleaning old ingestions..."
docker exec zipline-reloaded-jupyter zipline clean -b "$BUNDLE_NAME" --keep-last "$KEEP_LAST"

echo ""
echo "=================================================="
echo "✓ Update complete!"
echo "=================================================="

# Show current disk usage
echo ""
echo "Current bundle size:"
docker exec zipline-reloaded-jupyter du -sh /root/.zipline/data/"$BUNDLE_NAME"

echo ""
echo "Remaining ingestions:"
docker exec zipline-reloaded-jupyter bash -c "
for dir in /root/.zipline/data/$BUNDLE_NAME/*/; do
    size=\$(du -sh \"\$dir\" | cut -f1)
    dirname=\$(basename \"\$dir\")
    echo \"  \$size - \$dirname\"
done
"
