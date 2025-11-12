# Docker Quick Start - Build Optimization

## TL;DR - Fast Builds

Enable BuildKit for **5-7x faster** rebuilds:

### Option 1: Automated Setup (Easiest)

```bash
# Run the setup script (auto-detects zsh/bash)
./scripts/setup-buildkit.sh

# Reload your shell
source ~/.zshrc  # Mac (zsh)
# OR
source ~/.bashrc  # Linux (bash)

# Build with caching
docker-compose build
```

### Option 2: Manual Setup

```bash
# For zsh (Mac default):
echo 'export DOCKER_BUILDKIT=1' >> ~/.zshrc
echo 'export COMPOSE_DOCKER_CLI_BUILD=1' >> ~/.zshrc
source ~/.zshrc

# For bash (Linux):
echo 'export DOCKER_BUILDKIT=1' >> ~/.bashrc
echo 'export COMPOSE_DOCKER_CLI_BUILD=1' >> ~/.bashrc
source ~/.bashrc

# Build with caching
docker-compose build
```

**Performance:**
- First build: ~15-20 min (downloads everything)
- Subsequent builds: ~2-3 min (uses cache) **← 5-7x faster!**

## What Changed?

✅ **Dockerfile**: Removed `--no-cache-dir`, added BuildKit cache mounts
✅ **docker-compose.yml**: Added persistent pip cache volume
✅ **Build speed**: 5-7x faster on rebuilds

## How It Works

1. **BuildKit cache mounts** - Caches pip downloads during build
2. **Persistent volume** - Saves pip cache across builds
3. **Layer optimization** - Smart Dockerfile ordering

## Common Commands

```bash
# Build (with cache)
DOCKER_BUILDKIT=1 docker-compose build

# Clean build (no cache)
DOCKER_BUILDKIT=1 docker-compose build --no-cache

# Start containers
docker-compose up -d

# View build cache size
docker buildx du

# Clear build cache (if needed)
docker builder prune
```

## Verify It's Working

During build, you should see:
```
=> CACHED [stage-0  5/12] RUN --mount=type=cache...
```

The `CACHED` keyword means it's working!

## More Details

See `docs/docker-build-optimization.md` for comprehensive documentation.

## Troubleshooting

**Not seeing speedup?**
- Ensure `DOCKER_BUILDKIT=1` is set
- Check Docker version: `docker version` (needs 18.09+)
- First build always takes full time (creates cache)

**Want to start fresh?**
```bash
docker-compose down
docker volume rm zipline-reloaded_pip-cache
docker builder prune
docker-compose build --no-cache
```
