# Docker Disk Space Fix

## The Problem

Your Docker build failed with:
```
E: You don't have enough free space in /var/cache/apt/archives/.
```

This means Docker Desktop on your Mac has run out of allocated disk space.

## Quick Fix (Do this first)

### 1. Clean Up Docker Cache

Run these commands on your Mac:

```bash
# Remove all stopped containers
docker container prune -f

# Remove all unused images
docker image prune -a -f

# Remove all unused volumes (BE CAREFUL - this deletes data!)
docker volume prune -f

# Remove all build cache
docker builder prune -a -f

# Or clean everything at once (NUCLEAR OPTION - deletes everything!)
docker system prune -a --volumes -f
```

**⚠️ WARNING:** `docker volume prune -f` will DELETE your ingested Sharadar data if it's not currently in use by a running container. If you've already ingested data, skip this command or stop containers first.

**Safer option:**
```bash
# Clean everything except volumes
docker system prune -a -f
```

### 2. Check Disk Usage

```bash
docker system df
```

You should see something like:
```
TYPE            TOTAL     ACTIVE    SIZE      RECLAIMABLE
Images          10        2         5GB       3GB (60%)
Containers      5         1         100MB     80MB (80%)
Local Volumes   3         1         2GB       1GB (50%)
Build Cache     50        0         10GB      10GB (100%)
```

### 3. Increase Docker Desktop Disk Size

If cleaning doesn't help, increase Docker's disk allocation:

1. **Open Docker Desktop**
2. Click **Settings** (gear icon)
3. Go to **Resources** → **Disk image size**
4. Increase from default (usually 64GB) to **100GB+**
5. Click **Apply & Restart**

**Note:** This creates a new disk image, so you'll need to re-build and re-ingest data.

---

## Alternative: Build Without Cache (Smaller)

If you're still running into space issues, try a slimmer build:

### Option 1: Use Pre-built Image (if available)

If there's a pre-built image, pull it instead:
```bash
docker pull ghcr.io/your-repo/zipline-reloaded:latest
```

### Option 2: Build in Stages

Build only what you need right now:

```bash
# Build just the base without installing everything
docker compose build --no-cache zipline-jupyter
```

---

## What to Do Now

### Step 1: Clean Docker
```bash
# Safe clean (keeps volumes)
docker system prune -a -f

# Check space
docker system df
```

### Step 2: Try Building Again
```bash
docker compose build --no-cache
```

### Step 3: If Still Fails

**Increase Docker Desktop disk size:**
1. Docker Desktop → Settings → Resources
2. Disk image size: **100 GB** (or more)
3. Apply & Restart

Then retry:
```bash
docker compose build --no-cache
```

---

## Why This Happened

Docker Desktop on Mac uses a VM with a virtual disk. The default size (64GB) can fill up quickly when:
- Building large images (Zipline has many dependencies)
- Multiple builds without cleaning cache
- Storing data in volumes

**The build needs:**
- ~2 GB for base Python image
- ~600 MB for system packages (gcc, gfortran, etc.)
- ~1 GB for Python packages (numpy, pandas, etc.)
- Plus build cache

Total: ~5-10 GB for a clean build.

---

## After Fixing

Once you have space and the build succeeds, your Sharadar data (if you had any) might be gone if you used `docker volume prune`. You'll need to re-ingest:

```bash
docker compose exec zipline-jupyter zipline ingest -b sharadar
```

Takes 60-90 minutes first time.

---

## Quick Commands Summary

```bash
# 1. Clean up Docker (safe - keeps volumes)
docker system prune -a -f

# 2. Check space
docker system df

# 3. Build
docker compose build --no-cache

# 4. If fails, increase disk in Docker Desktop Settings
# Then retry build
```
