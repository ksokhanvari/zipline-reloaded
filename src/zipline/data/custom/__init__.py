"""
Custom Data Module for Zipline

This module provides functionality to load and manage custom tabular data
(e.g., fundamentals, alternative data) in local SQLite databases and expose
them as Zipline Pipeline data sources.

Main components:
- db_manager: Create and manage custom data databases
- loader: Load CSV data into custom databases
- query: Query custom data
- pipeline_integration: Integrate custom data with Zipline Pipeline

Quick Start (Recommended):
    For easy multi-source integration with Sharadar and custom data, use the
    high-level multi_source module:

    >>> from zipline.pipeline import multi_source as ms
    >>>
    >>> class CustomFundamentals(ms.Database):
    ...     CODE = "fundamentals"
    ...     LOOKBACK_WINDOW = 252
    ...     ROE = ms.Column(float)
    >>>
    >>> # Run backtest - auto-detects all sources!
    >>> results = run_algorithm(
    ...     ...,
    ...     custom_loader=ms.setup_auto_loader(),
    ... )

    See docs/MULTI_SOURCE_DATA.md for complete documentation.

Advanced Usage (Low-level API):
    >>> from zipline.data.custom import create_custom_db, load_csv_to_db
    >>> from zipline.data.custom import make_custom_dataset_class, CustomSQLiteLoader
    >>>
    >>> # Create database
    >>> create_custom_db('fundamentals', '1 day', {
    ...     'Revenue': 'int',
    ...     'EPS': 'float',
    ...     'Currency': 'text'
    ... })
    >>>
    >>> # Load CSV data
    >>> load_csv_to_db('data.csv', 'fundamentals', sid_map=securities_map)
    >>>
    >>> # Use in Pipeline
    >>> FundamentalsDataSet = make_custom_dataset_class('fundamentals', {...})
    >>> loader = CustomSQLiteLoader('fundamentals')
"""

from .config import DEFAULT_DB_DIR, get_custom_data_dir
from .db_manager import (
    create_custom_db,
    get_db_path,
    connect_db,
    list_custom_dbs,
    describe_custom_db,
)
from .loader import load_csv_to_db
from .query import get_prices, get_prices_reindexed_like, get_latest_values
from .pipeline_integration import (
    make_custom_dataset_class,
    CustomSQLiteLoader,
    register_custom_loader,
)

# Import auto_loader if available (may not be in older installations)
try:
    from zipline.pipeline.loaders.auto_loader import setup_auto_loader, AutoLoader
    _HAS_AUTO_LOADER = True
except ImportError:
    _HAS_AUTO_LOADER = False
    setup_auto_loader = None
    AutoLoader = None

__all__ = [
    # Config
    'DEFAULT_DB_DIR',
    'get_custom_data_dir',
    # Database management
    'create_custom_db',
    'get_db_path',
    'connect_db',
    'list_custom_dbs',
    'describe_custom_db',
    # Data loading
    'load_csv_to_db',
    # Querying
    'get_prices',
    'get_prices_reindexed_like',
    'get_latest_values',
    # Pipeline integration
    'make_custom_dataset_class',
    'CustomSQLiteLoader',
    'register_custom_loader',
]

# Add auto_loader to exports if available
if _HAS_AUTO_LOADER:
    __all__.extend(['setup_auto_loader', 'AutoLoader'])
