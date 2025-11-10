"""
Zipline Extensions - Custom bundle registration.

INSTALLATION:
Copy this file to ~/.zipline/extension.py

    cp extension.py ~/.zipline/extension.py

This file is automatically loaded by Zipline on startup to register
custom data bundles and other extensions.
"""

from zipline.data.bundles import register
from zipline.data.bundles.sharadar_bundle import sharadar_bundle


# Register the main Sharadar bundle (all tickers, incremental updates)
register(
    'sharadar',
    sharadar_bundle(
        tickers=None,  # None = all tickers
        incremental=True,  # Enable incremental updates
        include_funds=True,  # Include ETFs and funds
    ),
)

# Register Sharadar equities only (no funds)
register(
    'sharadar-equities',
    sharadar_bundle(
        tickers=None,
        incremental=True,
        include_funds=False,
    ),
)

# Register tech stocks sample (faster for testing)
register(
    'sharadar-tech',
    sharadar_bundle(
        tickers=[
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META',
            'NVDA', 'TSLA', 'NFLX', 'ADBE', 'CRM',
            'INTC', 'AMD', 'ORCL', 'CSCO', 'AVGO',
        ],
        incremental=True,
        include_funds=False,
    ),
)

print("✓ Sharadar bundles registered: sharadar, sharadar-equities, sharadar-tech")
