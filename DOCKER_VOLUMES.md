# Docker Volume Mounts - Important Information

## Volume Mount Behavior

The Docker containers use **bind mounts** that map your local directories directly into the container:

```yaml
volumes:
  - ./notebooks:/notebooks          # Your local notebooks → container
  - ./examples:/app/examples        # Your local examples → container
  - ./scripts:/scripts              # Your local scripts → container
  - ./data:/data                    # Your local data → container
```

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
| Edit Python file | ✗ No rebuild, just save file |
| Edit Jupyter notebook | ✗ No rebuild, restart kernel |
| Git pull new changes | ✗ No rebuild, restart containers/kernel |
| Change dependencies | ✓ Yes, rebuild (`docker-compose build`) |
| Change Dockerfile | ✓ Yes, rebuild |

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
