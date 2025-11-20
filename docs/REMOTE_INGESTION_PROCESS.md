# Remote Data Ingestion Process

Guide for offloading Zipline data ingestion to remote cloud infrastructure with S3 storage.

---

## Overview

This document outlines approaches for running data ingestion on remote machines (EC2, Lambda) and downloading the processed data to local Docker containers for backtesting.

---

## Option 1: Pre-built Data Bundle on S3

**How it works:**
1. Build the Zipline data bundle on a remote machine (EC2, Lambda, etc.)
2. Upload the entire `~/.zipline` directory to S3
3. Container downloads and extracts on startup

```python
# In container startup script
import boto3
import os

s3 = boto3.client('s3')
s3.download_file('your-bucket', 'zipline-data/sharadar-bundle.tar.gz', '/tmp/bundle.tar.gz')
os.system('tar -xzf /tmp/bundle.tar.gz -C /root/.zipline')
```

**Pros:** Fast startup after download, no ingestion needed locally
**Cons:** Large download (~10GB), need to manage bundle versions

---

## Option 2: S3-Mounted Volume (s3fs or goofys)

**How it works:**
Mount S3 bucket as a filesystem using FUSE

```dockerfile
# In Dockerfile
RUN apt-get install -y s3fs

# In docker-compose.yml
volumes:
  - type: bind
    source: /mnt/s3-zipline
    target: /root/.zipline
```

```bash
# Mount S3 on host
s3fs your-bucket /mnt/s3-zipline -o iam_role=auto
```

**Pros:** Transparent access, always up-to-date
**Cons:** Slow random access (S3 latency), not ideal for SQLite files

---

## Option 3: Hybrid - S3 Sync on Startup

**How it works:**
Use AWS CLI to sync data from S3 to local volume on container startup

```bash
# entrypoint.sh
#!/bin/bash
aws s3 sync s3://your-bucket/zipline-data /root/.zipline --size-only
exec "$@"
```

```dockerfile
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["jupyter", "lab", ...]
```

**Pros:** Local performance after sync, incremental updates
**Cons:** Startup delay for sync

---

## Option 4: Remote Ingestion Service (Recommended)

**Architecture:**
```
┌─────────────────┐     ┌──────────────┐     ┌─────────────────┐
│  EC2/Lambda     │     │     S3       │     │  Local Docker   │
│  Ingestion      │ ──► │   Bucket     │ ──► │   Container     │
│  Worker         │     │              │     │                 │
└─────────────────┘     └──────────────┘     └─────────────────┘
      │                                              │
      │  1. Ingest data                             │  3. Download
      │  2. Upload to S3                            │     on startup
      ▼                                              ▼
   Scheduled                                    Fast local
   (cron/EventBridge)                          backtesting
```

### Implementation

#### 1. Remote Worker (EC2 or Lambda)

```python
# ingest_worker.py
import subprocess
import boto3
import tarfile
from datetime import datetime

# Ingest data
subprocess.run(['zipline', 'ingest', '-b', 'sharadar'])

# Package and upload
with tarfile.open('/tmp/sharadar-bundle.tar.gz', 'w:gz') as tar:
    tar.add('/root/.zipline', arcname='.')

s3 = boto3.client('s3')
s3.upload_file('/tmp/sharadar-bundle.tar.gz', 'your-bucket',
               f'zipline-data/sharadar-{datetime.now():%Y%m%d}.tar.gz')
```

#### 2. Local Container Startup

```python
# download_bundle.py
import boto3
import os

def download_latest_bundle():
    s3 = boto3.client('s3')

    # Get latest bundle
    response = s3.list_objects_v2(
        Bucket='your-bucket',
        Prefix='zipline-data/sharadar-'
    )
    latest = sorted(response['Contents'], key=lambda x: x['LastModified'])[-1]

    # Check if we need to update
    local_marker = '/root/.zipline/.bundle_version'
    if os.path.exists(local_marker):
        with open(local_marker) as f:
            if f.read() == latest['Key']:
                print("Bundle is up to date")
                return

    # Download and extract
    print(f"Downloading {latest['Key']}...")
    s3.download_file('your-bucket', latest['Key'], '/tmp/bundle.tar.gz')
    os.system('tar -xzf /tmp/bundle.tar.gz -C /root/.zipline')

    # Mark version
    with open(local_marker, 'w') as f:
        f.write(latest['Key'])

if __name__ == '__main__':
    download_latest_bundle()
```

---

## Recommendation

For production use, **Option 4 (Remote Ingestion Service)** is recommended because:

1. **Offloads ingestion** - Heavy data processing happens on cloud
2. **Fast local backtesting** - Data is local after initial download
3. **Incremental updates** - Only download when new data available
4. **Cost effective** - Use spot instances or Lambda for ingestion
5. **Version control** - Keep multiple bundle versions in S3

---

## Cost Estimates

| Resource | Usage | Monthly Cost |
|----------|-------|--------------|
| S3 storage | 10 GB | ~$2.30 |
| EC2 t3.medium | 30 min/day ingestion | ~$0.60 |
| Data transfer | Initial download | ~$0.90 (one-time) |
| Data transfer | Daily updates | Negligible |

**Total estimated monthly cost: ~$3-5/month**

---

## AWS Services Required

- **S3**: Data storage
- **EC2 or Lambda**: Ingestion worker
- **EventBridge**: Scheduling (optional)
- **IAM**: Access control

---

## Security Considerations

1. Use IAM roles instead of access keys
2. Enable S3 bucket encryption
3. Restrict bucket access to specific IPs/VPCs
4. Enable S3 versioning for data recovery
5. Use VPC endpoints for private S3 access

---

## Next Steps

1. Create S3 bucket with appropriate permissions
2. Set up EC2 instance or Lambda function for ingestion
3. Configure scheduling (EventBridge or cron)
4. Modify Docker container to download on startup
5. Test end-to-end workflow

---

**Last Updated:** 2025-11-20

**Status:** Planning - Implementation pending
