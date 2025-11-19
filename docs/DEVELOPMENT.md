# Development Guide

Guide for developing with zipline-reloaded, including Docker build optimization and development workflow.

---

## Docker Build Optimization

### Enable BuildKit (Required)

BuildKit provides advanced caching and faster builds.

```bash
# Run setup script (auto-detects shell)
./scripts/setup-buildkit.sh

# Reload shell config
source ~/.zshrc  # Mac
# or
source ~/.bashrc  # Linux
```

**Or manually add to your shell config:**
```bash
export DOCKER_BUILDKIT=1
export COMPOSE_DOCKER_CLI_BUILD=1
```

### Multi-Layer Caching Strategy

This project uses a three-layer caching approach:

1. **Docker BuildKit cache mounts** (build-time)
2. **Persistent pip cache volume** (runtime)
3. **Docker layer caching** (proper Dockerfile ordering)

### Build Performance

| Scenario | Without Caching | With Caching | Speedup |
|----------|-----------------|--------------|---------|
| Clean build | ~15-20 min | ~15-20 min | 1x |
| Rebuild (no changes) | ~15-20 min | ~2-3 min | **5-7x** |
| Rebuild (code only) | ~15-20 min | ~2-3 min | **5-7x** |
| Rebuild (deps changed) | ~15-20 min | ~5-8 min | **2-3x** |

### Build Commands

```bash
# Standard build with caching
docker compose build

# Force rebuild without cache
docker compose build --no-cache

# Build with progress output
docker compose build --progress=plain

# Check build cache
docker buildx du
```

### Cache Management

```bash
# View pip cache volume
docker volume inspect zipline-reloaded_pip-cache

# Clear pip cache
docker volume rm zipline-reloaded_pip-cache

# Clear BuildKit cache
docker builder prune
```

---

## Development Workflow

### When to Rebuild vs Restart

| Change Type | Action |
|-------------|--------|
| Python source code | Rebuild container |
| Requirements/dependencies | Rebuild container |
| Docker config | Rebuild container |
| Strategy code (mounted) | Restart container |
| Data files | Restart container |
| Environment variables | Restart container |

### Quick Commands

```bash
# Restart container (keeps image)
docker compose restart zipline-jupyter

# Rebuild container
docker compose build zipline-jupyter
docker compose up -d

# View logs
docker compose logs -f zipline-jupyter

# Enter container shell
docker exec -it zipline-reloaded-jupyter bash
```

---

## Project Structure

```
zipline-reloaded/
├── src/zipline/          # Main source code
├── examples/             # Example notebooks and strategies
│   ├── getting_started/  # Quickstart tutorials
│   ├── strategies/       # Production strategies
│   ├── lseg_fundamentals/# LSEG data loading
│   ├── shared_modules/   # Reusable code
│   └── utils/            # Utility scripts
├── docs/                 # Documentation
├── scripts/              # Shell scripts
├── tests/                # Test suite
└── data/                 # Output directory (persists)
```

### Volume Mounts

Only data is persisted via Docker volumes:

```yaml
volumes:
  - ./data:/data                    # Strategy output
  - zipline-data:/root/.zipline     # Bundles
  - pip-cache:/root/.cache/pip      # Package cache
```

---

## Testing

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_algorithm.py

# Run with coverage
pytest --cov=zipline tests/
```

---

## Common Tasks

### Adding New Dependencies

1. Add to `requirements.txt` or `pyproject.toml`
2. Rebuild container: `docker compose build`

### Creating New Strategies

1. Create file in `examples/strategies/`
2. Use shared modules from `examples/shared_modules/`
3. Output to `/data/backtest_results/`

### Modifying Core Zipline

1. Edit source in `src/zipline/`
2. Rebuild container to recompile Cython extensions
3. Run relevant tests

---

## Troubleshooting

### "Module not found" after rebuild

Rebuild failed to compile Cython extensions:
```bash
docker compose build --no-cache zipline-jupyter
```

### Slow builds

Ensure BuildKit is enabled:
```bash
echo $DOCKER_BUILDKIT  # Should show "1"
```

### Cache not working

Check Dockerfile syntax and cache mount:
```bash
docker compose build --progress=plain
# Look for "CACHED" indicators
```

---

## Best Practices

1. **Always use BuildKit** for modern Docker features
2. **Order Dockerfile commands** from least to most frequently changed
3. **Copy requirements separately** before copying source code
4. **Use `.dockerignore`** to exclude unnecessary files
5. **Monitor cache size** and prune periodically

---

## Git Workflow

### Commit Message Format

```
type: Short description

Longer description if needed.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>
```

Types: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`

### Branch Strategy

- `main`: Stable release
- `claude/*`: Claude Code development branches
- Feature branches from `main`
