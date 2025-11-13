"""
Configuration for custom data module.

Provides default paths and settings for custom data storage.
"""

import os
from pathlib import Path
from typing import Union

# Default directory for custom history databases
# Can be overridden via ZIPLINE_CUSTOM_DATA_DIR environment variable
DEFAULT_DB_DIR = Path(
    os.environ.get(
        'ZIPLINE_CUSTOM_DATA_DIR',
        os.path.join(os.path.expanduser('~'), '.zipline', 'custom_data')
    )
)

# Default chunk size for CSV processing
DEFAULT_CHUNK_SIZE = 100000

# Supported column types for custom data
SUPPORTED_TYPES = {
    'int': 'INTEGER',
    'float': 'REAL',
    'text': 'TEXT',
    'date': 'TEXT',  # Stored as ISO 8601 text
    'datetime': 'TEXT',  # Stored as ISO 8601 text
}

# Column name validation pattern
VALID_COLUMN_PATTERN = r'^[A-Za-z_][A-Za-z0-9_]*$'


def get_custom_data_dir() -> Path:
    """
    Get the custom data directory path.

    Creates the directory if it doesn't exist.

    Returns
    -------
    Path
        Path to custom data directory
    """
    DEFAULT_DB_DIR.mkdir(parents=True, exist_ok=True)
    return DEFAULT_DB_DIR


def get_db_filename(db_code: str) -> str:
    """
    Get the database filename for a given db_code.

    Parameters
    ----------
    db_code : str
        Database code/identifier

    Returns
    -------
    str
        Database filename (e.g., 'fundamentals.sqlite')
    """
    return f"{db_code}.sqlite"


def validate_column_name(name: str) -> bool:
    """
    Validate that a column name follows Python identifier rules.

    Parameters
    ----------
    name : str
        Column name to validate

    Returns
    -------
    bool
        True if valid, False otherwise
    """
    import re
    return bool(re.match(VALID_COLUMN_PATTERN, name))


def validate_column_type(type_str: str) -> bool:
    """
    Validate that a column type is supported.

    Parameters
    ----------
    type_str : str
        Type string (e.g., 'int', 'float', 'text', 'date', 'datetime')

    Returns
    -------
    bool
        True if supported, False otherwise
    """
    return type_str in SUPPORTED_TYPES


def get_sql_type(type_str: str) -> str:
    """
    Get SQL type string for a given type.

    Parameters
    ----------
    type_str : str
        Type string (e.g., 'int', 'float')

    Returns
    -------
    str
        SQL type string (e.g., 'INTEGER', 'REAL')

    Raises
    ------
    ValueError
        If type is not supported
    """
    if not validate_column_type(type_str):
        raise ValueError(
            f"Unsupported column type: {type_str}. "
            f"Supported types: {', '.join(SUPPORTED_TYPES.keys())}"
        )
    return SUPPORTED_TYPES[type_str]
