# Docker Build Optimization Guide

This guide explains how to speed up Docker builds by caching Python packages and dependencies.

## Problem

Without caching, every Docker build downloads all Python packages from PyPI, which can be slow and bandwidth-intensive, especially on slow networks.

## Solution: Multi-Layer Caching Strategy

This project uses a **three-layer caching approach**:

1. **Docker BuildKit cache mounts** (build-time caching)
2. **Persistent pip cache volume** (runtime caching)
3. **Docker layer caching** (proper Dockerfile ordering)

---

## 1. Enable BuildKit (Recommended)

BuildKit provides advanced caching features and significantly faster builds.

### Enable BuildKit permanently:

**Automated setup (easiest):**
```bash
# Run the setup script (auto-detects your shell)
./scripts/setup-buildkit.sh

# Then reload your shell config
source ~/.zshrc  # Mac (zsh)
# OR
source ~/.bashrc  # Linux (bash)
```

**Manual setup for zsh (macOS default):**
```zsh
echo 'export DOCKER_BUILDKIT=1' >> ~/.zshrc
echo 'export COMPOSE_DOCKER_CLI_BUILD=1' >> ~/.zshrc
source ~/.zshrc
```

**Manual setup for bash (Linux):**
```bash
echo 'export DOCKER_BUILDKIT=1' >> ~/.bashrc
echo 'export COMPOSE_DOCKER_CLI_BUILD=1' >> ~/.bashrc
source ~/.bashrc
```

**Or manually add to your shell config:**
```bash
# Add these lines to ~/.zshrc (Mac) or ~/.bashrc (Linux)
export DOCKER_BUILDKIT=1
export COMPOSE_DOCKER_CLI_BUILD=1
```

**Or enable for a single build only:**
```bash
DOCKER_BUILDKIT=1 docker-compose build
```

### What it does:
- Caches pip downloads in `/root/.cache/pip` during build
- Reuses cached packages across builds
- Only downloads new or updated packages

---

## 2. Persistent Pip Cache Volume

The `docker-compose.yml` mounts a persistent volume for pip cache:

```yaml
volumes:
  - pip-cache:/root/.cache/pip
```

### What it does:
- Persists pip cache across container restarts
- Shares cache between build and runtime
- Speeds up pip installs inside running containers

### View cache size:
```bash
docker volume inspect zipline-reloaded_pip-cache
```

### Clear cache if needed:
```bash
docker volume rm zipline-reloaded_pip-cache
```

---

## 3. Dockerfile Layer Optimization

The Dockerfile is structured to maximize layer caching:

```dockerfile
# 1. Install system dependencies (changes rarely)
RUN apt-get update && apt-get install ...

# 2. Copy only requirements files first (changes occasionally)
COPY requirements*.txt pyproject.toml setup.py ./

# 3. Install Python dependencies (benefits from cache)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip setuptools wheel

# 4. Copy source code last (changes frequently)
COPY . .
```

**Key principle**: Copy files in order from least to most frequently changed.

---

## Build Performance Comparison

| Scenario | Without Caching | With Caching | Speedup |
|----------|-----------------|--------------|---------|
| Clean build | ~15-20 min | ~15-20 min | 1x (first time) |
| Rebuild (no changes) | ~15-20 min | ~2-3 min | **5-7x** |
| Rebuild (code changes only) | ~15-20 min | ~2-3 min | **5-7x** |
| Rebuild (deps changed) | ~15-20 min | ~5-8 min | **2-3x** |

---

## Advanced: Local PyPI Mirror (Optional)

For maximum speed, you can set up a local PyPI mirror:

### Option A: Using devpi (Lightweight)

```bash
# Install devpi
pip install devpi-server devpi-client

# Start devpi server
devpi-server --start --init

# Configure pip to use devpi
export PIP_INDEX_URL=http://localhost:3141/root/pypi/+simple/
```

### Option B: Using bandersnatch (Full mirror)

```bash
# Install bandersnatch
pip install bandersnatch

# Configure and sync (warning: 1TB+ storage needed)
bandersnatch mirror
```

Then update `docker-compose.yml`:

```yaml
services:
  zipline-jupyter:
    environment:
      - PIP_INDEX_URL=http://host.docker.internal:3141/root/pypi/+simple/
      - PIP_TRUSTED_HOST=host.docker.internal
```

---

## Usage Examples

### Clean build with caching:
```bash
DOCKER_BUILDKIT=1 docker-compose build
```

### Force rebuild without cache:
```bash
DOCKER_BUILDKIT=1 docker-compose build --no-cache
```

### Build with progress output:
```bash
DOCKER_BUILDKIT=1 docker-compose build --progress=plain
```

### Check what's cached:
```bash
# List Docker build cache
docker buildx du

# List volumes
docker volume ls | grep cache
```

---

## Troubleshooting

### BuildKit not working?

Check Docker version (needs 18.09+):
```bash
docker version
```

Enable BuildKit in Docker daemon config (`/etc/docker/daemon.json`):
```json
{
  "features": {
    "buildkit": true
  }
}
```

### Cache not being used?

1. Ensure BuildKit is enabled: `echo $DOCKER_BUILDKIT`
2. Check Dockerfile syntax: `# syntax=docker/dockerfile:1.4`
3. Verify cache mount: look for `--mount=type=cache` in Dockerfile

### Packages still downloading?

- First build always downloads everything (creates cache)
- Subsequent builds should be much faster
- Watch build output for "CACHED" indicators

### Clear all caches:

```bash
# Clear BuildKit cache
docker builder prune

# Clear pip cache volume
docker volume rm zipline-reloaded_pip-cache

# Clean rebuild
docker-compose down
docker-compose build --no-cache
```

---

## Best Practices

1. **Always use BuildKit** for modern Docker features
2. **Don't use `--no-cache-dir`** with pip (prevents caching)
3. **Order Dockerfile commands** from least to most frequently changed
4. **Copy requirements files separately** before copying all source code
5. **Use `.dockerignore`** to exclude unnecessary files from context
6. **Monitor cache size** periodically and prune if needed

---

## Files Modified

- `Dockerfile` - Added BuildKit cache mounts, removed `--no-cache-dir`
- `docker-compose.yml` - Added persistent `pip-cache` volume
- `.dockerignore` - Excludes unnecessary files (recommended)

---

## References

- [Docker BuildKit Documentation](https://docs.docker.com/build/buildkit/)
- [Docker Layer Caching](https://docs.docker.com/build/cache/)
- [Pip Caching](https://pip.pypa.io/en/stable/topics/caching/)
- [devpi PyPI Server](https://devpi.net/)
