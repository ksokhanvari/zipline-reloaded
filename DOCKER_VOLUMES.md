# Docker Volume Mounts - Important Information

## Volume Mount Behavior

The Docker containers use **bind mounts** that map your local directories directly into the container:

```yaml
volumes:
  - ./notebooks:/notebooks          # Your local notebooks → container
  - ./examples:/app/examples        # Your local examples → container
  - ./scripts:/scripts              # Your local scripts → container
  - ./data:/data                    # Your local data → container
  - zipline-data:/root/.zipline     # Named volume (persists bundle data)
```

### What's Mounted vs What's Not

**✅ Mounted from your local repo** (changes appear immediately):
- `./notebooks/` → `/notebooks` - Jupyter notebooks
- `./examples/` → `/app/examples` - Example notebooks and scripts
- `./scripts/` → `/scripts` - Management scripts
- `./data/` → `/data` - Data files and custom databases

**❌ NOT mounted** (copied during build, requires restart to see changes):
- `./src/` → `/app/src` - Python source code
- `./tests/` → `/app/tests` - Test files
- `./docs/` → `/app/docs` - Documentation

**🔒 Named volumes** (persist data, not from your repo):
- `zipline-data` → `/root/.zipline` - Bundle data, survives container deletion
- `pip-cache` → `/root/.cache/pip` - Python package cache

## What This Means

### ✅ **Advantages:**
- Changes to files on your Mac **immediately appear** in the container
- No need to rebuild Docker when you edit code
- Easy to edit files with your favorite editor on Mac

### ⚠️ **Important Implications:**

1. **Rebuilding Docker does NOT update these files**
   - `docker-compose build` copies files during build
   - But volume mounts **override** those files at runtime
   - The container always uses what's in your local directories

2. **Git changes require container restart**
   - After `git pull`, files on your Mac are updated
   - Running containers see the new files immediately
   - BUT Jupyter may have old notebooks cached in memory

3. **Always restart Jupyter kernel after git pull**
   ```bash
   # In Jupyter: Kernel → Shutdown
   # Then: Close and reopen the notebook
   ```

## Workflow Best Practices

### When You Git Pull Changes:

```bash
# 1. Pull latest code
git pull origin main

# 2. Restart containers (if needed)
docker-compose restart

# 3. In Jupyter:
#    - Kernel → Shutdown
#    - Close notebook tab
#    - Reopen notebook from file browser
#    - Run cells fresh
```

### When Working on Code:

```bash
# Edit files directly on your Mac
code examples/custom_data/my_file.py

# Changes appear immediately in container
# No rebuild needed!

# Just restart Python kernel if needed
```

## Understanding the File Flow

```
┌─────────────────────────────────────────────────────────────┐
│ YOUR MAC (Host)                                             │
│                                                              │
│  /Users/kamran/.../zipline-reloaded/                        │
│  ├── examples/                                              │
│  │   └── 6_custom_data/                                     │
│  │       └── load_csv_fundamentals.ipynb  ← YOU EDIT HERE  │
│  ├── notebooks/                                             │
│  └── scripts/                                               │
│                                                              │
│         │                                                    │
│         │ Bind Mount (Live Link)                            │
│         ▼                                                    │
│                                                              │
│  ┌────────────────────────────────────────────────────┐    │
│  │ DOCKER CONTAINER                                    │    │
│  │                                                      │    │
│  │  /app/examples/6_custom_data/                       │    │
│  │  └── load_csv_fundamentals.ipynb ← SEES YOUR FILE  │    │
│  │                                                      │    │
│  │  Changes on Mac = Changes in Container              │    │
│  │  (No rebuild needed!)                               │    │
│  └────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

## Troubleshooting

### Issue: "I pulled latest code but Jupyter shows old version"

**Cause:** Jupyter has the notebook cached in memory

**Solution:**
1. Kernel → Shutdown
2. Close notebook tab
3. Refresh Jupyter file browser (F5)
4. Reopen notebook

### Issue: "I rebuilt Docker but files didn't update"

**Cause:** Volume mounts override the built-in files

**Solution:**
1. Don't rebuild - just edit files on your Mac
2. Files update immediately via volume mount
3. Restart kernel if needed

### Issue: "Docker shows old code after git pull"

**Cause:** Need to restart to clear caches

**Solution:**
```bash
docker-compose restart
```

## Quick Reference

| Action | Do You Need to... |
|--------|------------------|
| Edit files in mounted dirs (`./examples/`, `./notebooks/`, `./scripts/`) | ✗ No restart, just save file and restart Jupyter kernel |
| Edit Python source (`./src/`) | ✗ No rebuild, just `docker-compose restart` |
| Git pull new changes | ✗ No rebuild, restart containers/kernel |
| Change dependencies (`requirements.txt`) | ✓ Yes, rebuild (`docker-compose build && docker-compose up -d`) |
| Change Dockerfile | ✓ Yes, rebuild (`docker-compose build && docker-compose up -d`) |

## Restart vs Rebuild: When to Use Each

### Restart Containers (Fast - ~5 seconds)

```bash
docker-compose restart
```

**Use when you changed:**
- ✅ Python source code in `./src/zipline/`
- ✅ Configuration files
- ✅ Any code that needs Python to reload modules

**What it does:**
- Stops running containers
- Starts them again using the **existing image**
- Reloads Python modules from your local files

**Why it works:**
- The Dockerfile copies `./src/` during build
- But at runtime, Python imports from those copied files
- Restarting clears Python's module cache and reloads everything

---

### Rebuild Containers (Slow - ~5 minutes)

```bash
docker-compose build
docker-compose up -d
```

**Use when you changed:**
- ✅ `Dockerfile` (changed base image, added OS packages, etc.)
- ✅ `requirements.txt` (added or removed Python packages)
- ✅ System dependencies or OS-level configuration
- ✅ Docker build steps or layers

**What it does:**
- Rebuilds the Docker image from scratch
- Reinstalls all dependencies
- Creates new containers from the new image

**When you DON'T need it:**
- ❌ Python code changes in `./src/` → Just restart
- ❌ Mounted directory changes (`./examples/`, `./notebooks/`) → Just restart kernel
- ❌ Most code changes → Just restart

---

### Summary: Restart vs Rebuild Decision Tree

```
Did you change a file?
│
├─ Is it in a mounted directory? (./examples/, ./notebooks/, ./scripts/)
│  └─ YES → Just restart Jupyter kernel (instant)
│
├─ Is it Python source code? (./src/)
│  └─ YES → docker-compose restart (~5 sec)
│
├─ Is it requirements.txt or Dockerfile?
│  └─ YES → docker-compose build && docker-compose up -d (~5 min)
│
└─ Not sure?
   └─ Try restart first, rebuild only if that doesn't work
```

---

## Git Hook for Automatic Reminders

Install a post-merge hook to automatically remind you to restart containers after `git pull`:

```bash
# Create the hook
cat > .git/hooks/post-merge << 'EOF'
#!/bin/bash
# Auto-restart containers after git pull to ensure latest code is used

echo "Git pull completed. Checking if Docker containers need restart..."

# Check if any mounted directories changed
if git diff --name-only HEAD@{1} HEAD | grep -E '^(examples|notebooks|scripts)/' > /dev/null; then
    echo "⚠️  Mounted directories changed. Consider restarting Docker containers:"
    echo "   docker-compose restart"
fi
EOF

# Make it executable
chmod +x .git/hooks/post-merge
```

## Remember

**The container ALWAYS uses files from your Mac's directories, not from the Docker image!**

This is by design - it makes development easier. But it means:
- Git pull updates your Mac files → Container sees them immediately
- No need to rebuild Docker for code changes
- Just restart Jupyter kernel to clear caches
