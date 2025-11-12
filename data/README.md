# Data Directory

This directory is mounted as a persistent volume in the Docker container and is used for storing data files that need to persist across container restarts.

## Directory Structure

```
data/
├── csv/                    # CSV files for custom fundamentals
│   └── (your CSV files here)
├── custom_databases/       # Custom SQLite databases (optional)
└── README.md              # This file
```

## Docker Volume Mapping

- **Container path**: `/data/`
- **Host path**: `./data/` (relative to docker-compose.yml)
- **Absolute host path**: `/home/user/zipline-reloaded/data/`

## Environment Variable

The Docker container sets:
```bash
ZIPLINE_CUSTOM_DATA_DIR=/data/custom_databases
```

## Usage

### Storing CSV Files for Fundamentals

1. **On your host machine**, place CSV files in:
   ```
   ./data/csv/
   ```

2. **Inside the Docker container**, they will be accessible at:
   ```
   /data/csv/
   ```

3. Use the `notebooks/load_csv_fundamentals.ipynb` notebook to load CSV files into a custom SQLite database.

### Custom Databases

Custom SQLite databases are automatically created in:
- **Container**: `/root/.zipline/data/custom/`
- **Persisted via**: Docker named volume `zipline-data`

## Example Workflow

1. Copy your CSV fundamentals files to `./data/csv/` on your host
2. Open JupyterLab (http://localhost:9000)
3. Run `notebooks/load_csv_fundamentals.ipynb`
4. The notebook will:
   - Read CSV files from `/data/csv/`
   - Map symbols to Zipline SIDs
   - Create database in `~/.zipline/data/custom/`
5. Use the custom database in your backtests

## Persistent Storage

- ✅ **CSV files in `/data/`** persist across container restarts (bind mount)
- ✅ **SQLite databases in `~/.zipline/data/`** persist across container restarts (named volume)
- ✅ **Notebooks in `/notebooks/`** persist across container restarts (bind mount)
- ✅ **Bundle data** persists in `~/.zipline/` (named volume)

## Notes

- The `csv/` subdirectory is for organization; you can create other subdirectories as needed
- The Docker bind mount ensures all files in this directory survive container restarts
- The Zipline bundle data and custom databases are stored separately in the `zipline-data` named volume
