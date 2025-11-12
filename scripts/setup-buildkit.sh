#!/bin/bash
#
# Setup Docker BuildKit environment variables
# This script auto-detects your shell and adds the necessary exports
#

set -e

echo "🔧 Docker BuildKit Setup"
echo "======================="
echo ""

# Detect the shell
SHELL_NAME=$(basename "$SHELL")
SHELL_RC=""

case "$SHELL_NAME" in
    zsh)
        SHELL_RC="$HOME/.zshrc"
        echo "✓ Detected shell: zsh"
        ;;
    bash)
        SHELL_RC="$HOME/.bashrc"
        echo "✓ Detected shell: bash"
        ;;
    *)
        echo "⚠️  Unknown shell: $SHELL_NAME"
        echo "Please manually add these lines to your shell config:"
        echo ""
        echo "  export DOCKER_BUILDKIT=1"
        echo "  export COMPOSE_DOCKER_CLI_BUILD=1"
        echo ""
        exit 1
        ;;
esac

# Check if already configured
if grep -q "DOCKER_BUILDKIT" "$SHELL_RC" 2>/dev/null; then
    echo "✓ BuildKit already configured in $SHELL_RC"
    echo ""

    # Show current values
    if grep -q "export DOCKER_BUILDKIT=1" "$SHELL_RC" 2>/dev/null; then
        echo "Current configuration:"
        grep "DOCKER_BUILDKIT\|COMPOSE_DOCKER_CLI_BUILD" "$SHELL_RC" | sed 's/^/  /'
        echo ""
        echo "✅ Configuration looks good!"
    else
        echo "⚠️  Warning: DOCKER_BUILDKIT found but may not be set to 1"
        echo "Please check your $SHELL_RC file"
    fi
else
    echo "📝 Adding BuildKit configuration to $SHELL_RC"
    echo ""

    # Add the configuration
    cat >> "$SHELL_RC" << 'EOF'

# Docker BuildKit - enables faster builds with caching
export DOCKER_BUILDKIT=1
export COMPOSE_DOCKER_CLI_BUILD=1
EOF

    echo "✅ Added to $SHELL_RC:"
    echo "  export DOCKER_BUILDKIT=1"
    echo "  export COMPOSE_DOCKER_CLI_BUILD=1"
    echo ""
fi

# Check if currently set in environment
if [ "$DOCKER_BUILDKIT" = "1" ]; then
    echo "✅ DOCKER_BUILDKIT is currently enabled in your session"
else
    echo "⚠️  DOCKER_BUILDKIT is not set in current session"
    echo ""
    echo "To enable for this session, run:"
    echo "  source $SHELL_RC"
    echo ""
    echo "Or start a new terminal window"
fi

echo ""
echo "🚀 Next steps:"
echo "  1. Reload your shell configuration:"
echo "     source $SHELL_RC"
echo ""
echo "  2. Build Docker image with caching:"
echo "     docker-compose build"
echo ""
echo "  3. First build takes ~15-20 min (creates cache)"
echo "     Subsequent builds take ~2-3 min (5-7x faster!)"
echo ""
echo "For more info, see: DOCKER-QUICK-START.md"
