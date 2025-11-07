"""
Database management for custom data.

Handles creation, connection, and introspection of custom data SQLite databases.
"""

import json
import logging
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Union

from .config import (
    DEFAULT_DB_DIR,
    get_custom_data_dir,
    get_db_filename,
    get_sql_type,
    validate_column_name,
    validate_column_type,
)

log = logging.getLogger(__name__)


def create_custom_db(
    db_code: str,
    bar_size: str,
    columns: Dict[str, str],
    db_dir: Union[str, Path] = None,
) -> Path:
    """
    Create a new custom data SQLite database.

    Parameters
    ----------
    db_code : str
        Database code/identifier (e.g., 'fundamentals')
    bar_size : str
        Bar size/frequency (e.g., '1 day', '1 week')
    columns : dict
        Dictionary mapping column names to types.
        Types must be one of: 'int', 'float', 'text', 'date', 'datetime'
    db_dir : str or Path, optional
        Directory to create database in. Defaults to DEFAULT_DB_DIR.

    Returns
    -------
    Path
        Path to created database file

    Raises
    ------
    ValueError
        If column names or types are invalid
    FileExistsError
        If database already exists

    Examples
    --------
    >>> create_custom_db('fundamentals', '1 day', {
    ...     'Revenue': 'int',
    ...     'EPS': 'float',
    ...     'Currency': 'text'
    ... })
    PosixPath('.../.zipline/custom_data/quant_fundamentals.sqlite')
    """
    if db_dir is None:
        db_dir = get_custom_data_dir()
    else:
        db_dir = Path(db_dir)
        db_dir.mkdir(parents=True, exist_ok=True)

    # Validate inputs
    if not db_code or not isinstance(db_code, str):
        raise ValueError("db_code must be a non-empty string")

    # Validate column names and types
    for col_name, col_type in columns.items():
        if not validate_column_name(col_name):
            raise ValueError(
                f"Invalid column name '{col_name}'. "
                "Column names must start with letter or underscore "
                "and contain only letters, numbers, and underscores."
            )
        if not validate_column_type(col_type):
            raise ValueError(
                f"Invalid column type '{col_type}' for column '{col_name}'. "
                f"Supported types: int, float, text, date, datetime"
            )

    # Build database path
    db_path = db_dir / get_db_filename(db_code)

    if db_path.exists():
        raise FileExistsError(f"Database already exists: {db_path}")

    # Create database
    conn = sqlite3.connect(str(db_path))
    try:
        cursor = conn.cursor()

        # Create ConfigBlob table
        cursor.execute("""
            CREATE TABLE ConfigBlob (
                id INTEGER PRIMARY KEY,
                config_json TEXT NOT NULL
            )
        """)

        # Store configuration
        config = {
            'db_code': db_code,
            'bar_size': bar_size,
            'columns': columns,
            'version': '1.0',
        }
        cursor.execute(
            "INSERT INTO ConfigBlob (id, config_json) VALUES (?, ?)",
            (1, json.dumps(config))
        )

        # Build Price table schema
        column_defs = ["Sid VARCHAR(20) NOT NULL", "Date DATETIME NOT NULL"]
        for col_name, col_type in columns.items():
            sql_type = get_sql_type(col_type)
            column_defs.append(f"{col_name} {sql_type}")

        # Create Price table with primary key
        create_table_sql = f"""
            CREATE TABLE Price (
                {', '.join(column_defs)},
                PRIMARY KEY (Sid, Date)
            )
        """
        cursor.execute(create_table_sql)

        # Create indices for better query performance
        cursor.execute("CREATE INDEX idx_price_date ON Price(Date)")
        cursor.execute("CREATE INDEX idx_price_sid ON Price(Sid)")

        conn.commit()
        log.info(f"Created custom database: {db_path}")
        log.info(f"  Columns: {', '.join(columns.keys())}")

    except Exception as e:
        conn.close()
        # Clean up failed database creation
        if db_path.exists():
            db_path.unlink()
        raise

    finally:
        conn.close()

    return db_path


def get_db_path(db_code: str, db_dir: Union[str, Path] = None) -> Path:
    """
    Get path to a custom data database.

    Parameters
    ----------
    db_code : str
        Database code/identifier
    db_dir : str or Path, optional
        Directory containing database. Defaults to DEFAULT_DB_DIR.

    Returns
    -------
    Path
        Path to database file

    Raises
    ------
    FileNotFoundError
        If database doesn't exist
    """
    if db_dir is None:
        db_dir = get_custom_data_dir()
    else:
        db_dir = Path(db_dir)

    db_path = db_dir / get_db_filename(db_code)

    if not db_path.exists():
        raise FileNotFoundError(
            f"Database not found: {db_path}\n"
            f"Available databases: {', '.join(list_custom_dbs(db_dir)) or 'none'}"
        )

    return db_path


def connect_db(db_code: str, db_dir: Union[str, Path] = None) -> sqlite3.Connection:
    """
    Connect to a custom data database.

    Parameters
    ----------
    db_code : str
        Database code/identifier
    db_dir : str or Path, optional
        Directory containing database. Defaults to DEFAULT_DB_DIR.

    Returns
    -------
    sqlite3.Connection
        Connection to database

    Raises
    ------
    FileNotFoundError
        If database doesn't exist
    """
    db_path = get_db_path(db_code, db_dir)
    return sqlite3.connect(str(db_path))


def list_custom_dbs(db_dir: Union[str, Path] = None) -> List[str]:
    """
    List all custom data databases.

    Parameters
    ----------
    db_dir : str or Path, optional
        Directory to search. Defaults to DEFAULT_DB_DIR.

    Returns
    -------
    list of str
        List of database codes
    """
    if db_dir is None:
        db_dir = get_custom_data_dir()
    else:
        db_dir = Path(db_dir)

    if not db_dir.exists():
        return []

    dbs = []
    for db_file in db_dir.glob("quant_*.sqlite"):
        # Extract db_code from filename
        db_code = db_file.stem.replace("quant_", "")
        dbs.append(db_code)

    return sorted(dbs)


def describe_custom_db(db_code: str, db_dir: Union[str, Path] = None) -> Dict:
    """
    Get metadata and schema information for a custom database.

    Parameters
    ----------
    db_code : str
        Database code/identifier
    db_dir : str or Path, optional
        Directory containing database. Defaults to DEFAULT_DB_DIR.

    Returns
    -------
    dict
        Dictionary containing database metadata:
        - db_code: str
        - db_path: str
        - bar_size: str
        - columns: dict mapping column names to types
        - row_count: int
        - date_range: tuple of (min_date, max_date) or None
        - sids: list of unique Sids

    Raises
    ------
    FileNotFoundError
        If database doesn't exist
    """
    db_path = get_db_path(db_code, db_dir)
    conn = sqlite3.connect(str(db_path))

    try:
        cursor = conn.cursor()

        # Get configuration
        cursor.execute("SELECT config_json FROM ConfigBlob WHERE id = 1")
        config_row = cursor.fetchone()

        if not config_row:
            raise ValueError(f"No configuration found in database: {db_code}")

        config = json.loads(config_row[0])

        # Get table info
        cursor.execute("PRAGMA table_info(Price)")
        table_info = cursor.fetchall()

        # Extract column information (skip Sid and Date)
        schema_columns = {}
        for col in table_info:
            col_name = col[1]
            col_type = col[2]

            if col_name not in ('Sid', 'Date'):
                # Map SQL type back to our type system
                type_map = {
                    'INTEGER': 'int',
                    'REAL': 'float',
                    'TEXT': 'text',  # Ambiguous, but we store date/datetime as text too
                }
                schema_columns[col_name] = type_map.get(col_type, col_type.lower())

        # Get statistics
        cursor.execute("SELECT COUNT(*) FROM Price")
        row_count = cursor.fetchone()[0]

        # Get date range
        cursor.execute("SELECT MIN(Date), MAX(Date) FROM Price")
        date_range = cursor.fetchone()

        # Get unique Sids
        cursor.execute("SELECT DISTINCT Sid FROM Price ORDER BY Sid")
        sids = [row[0] for row in cursor.fetchall()]

        return {
            'db_code': db_code,
            'db_path': str(db_path),
            'bar_size': config.get('bar_size', 'unknown'),
            'columns': config.get('columns', schema_columns),
            'row_count': row_count,
            'date_range': date_range if date_range[0] else None,
            'sids': sids,
            'num_sids': len(sids),
        }

    finally:
        conn.close()
