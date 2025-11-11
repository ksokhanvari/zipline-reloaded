# Bundle Registration Guide

## Problem

The `extension.py` file (which registers bundles like 'sharadar') is only loaded by Zipline CLI commands (like `zipline bundles` or `zipline ingest`), but NOT when you import Zipline in Python scripts or notebooks.

This causes `UnknownBundle` errors when running standalone scripts.

## Solution

Use the centralized `register_bundles.py` utility in all your scripts.

### Option 1: Auto-registration (Recommended)

Simply import the module - bundles are registered automatically:

```python
from register_bundles import ensure_bundles_registered
from zipline.data.bundles import load as load_bundle

# Bundles are already registered from the import above!
bundle_data = load_bundle('sharadar')
```

### Option 2: Explicit registration

For more control or to see registration status:

```python
from register_bundles import ensure_bundles_registered

# Explicitly register with verbose output
ensure_bundles_registered(verbose=True)

# Now load bundles
from zipline.data.bundles import load as load_bundle
bundle_data = load_bundle('sharadar')
```

## Files Updated

All example scripts have been updated to use this pattern:

1. **create_fundamentals_db.py** - Creates fundamentals database
2. **check_bundle_data.py** - Diagnostics for bundle data
3. **test_fundamentals_only.py** - Tests fundamentals loader
4. **backtest_with_fundamentals.py** - Main strategy file
5. **backtest_helpers.py** - Backtest utility functions

## When to Use

Import `register_bundles` at the top of any Python file that:

- Uses `load_bundle()` to load a bundle
- Runs `run_algorithm()` with a bundle parameter
- Imports modules that use bundles

## Why This Works

The utility:
- Registers the base 'sharadar' bundle (all tickers)
- Registers variant bundles (sharadar-tech, sharadar-sp500, etc.)
- Is idempotent (safe to call multiple times)
- Auto-registers on import for convenience

## Example Pattern

```python
#!/usr/bin/env python
"""
My Zipline script that uses bundles.
"""

import pandas as pd
import numpy as np

# ALWAYS import this first, before other zipline imports
from register_bundles import ensure_bundles_registered

# Now import zipline modules
from zipline import run_algorithm
from zipline.data.bundles import load as load_bundle

# Rest of your script...
```

## Notebooks

For Jupyter notebooks, add this to your first cell:

```python
# Cell 1: Setup
from register_bundles import ensure_bundles_registered
ensure_bundles_registered(verbose=True)
```

Then all subsequent cells can use bundles without issues.

## Technical Details

**Why doesn't extension.py work automatically?**

- `extension.py` is loaded by `zipline.utils.extensions.load_user_extension()`
- This is called by CLI entry points (defined in pyproject.toml)
- Python imports don't trigger this loading automatically
- Scripts need to explicitly register bundles or import a module that does

**How does register_bundles.py work?**

```python
# On import, this runs automatically:
if __name__ != '__main__':
    ensure_bundles_registered(verbose=False)
```

This means any import of the module registers bundles, making it convenient for most use cases.
