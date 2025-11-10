"""
Tests for zipline.data.custom.db_manager module.
"""

import pytest
import sqlite3
from pathlib import Path

from zipline.data.custom.db_manager import (
    create_custom_db,
    get_db_path,
    connect_db,
    list_custom_dbs,
    describe_custom_db,
)


@pytest.fixture
def temp_db_dir(tmp_path):
    """Temporary directory for test databases."""
    db_dir = tmp_path / 'custom_data'
    db_dir.mkdir()
    return db_dir


class TestCreateCustomDB:
    """Test database creation."""

    def test_create_basic_db(self, temp_db_dir):
        """Test creating a basic database."""
        columns = {
            'Revenue': 'int',
            'EPS': 'float',
            'Currency': 'text',
        }

        db_path = create_custom_db(
            db_code='test_fundamentals',
            bar_size='1 day',
            columns=columns,
            db_dir=temp_db_dir,
        )

        assert db_path.exists()
        assert db_path.name == 'quant_test_fundamentals.sqlite'

        # Verify database structure
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        # Check ConfigBlob table
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        assert 'ConfigBlob' in tables
        assert 'Price' in tables

        # Check Price table columns
        cursor.execute("PRAGMA table_info(Price)")
        price_columns = {row[1]: row[2] for row in cursor.fetchall()}
        assert 'Sid' in price_columns
        assert 'Date' in price_columns
        assert 'Revenue' in price_columns
        assert 'EPS' in price_columns
        assert 'Currency' in price_columns

        # Check column types
        assert price_columns['Revenue'] == 'INTEGER'
        assert price_columns['EPS'] == 'REAL'
        assert price_columns['Currency'] == 'TEXT'

        conn.close()

    def test_create_db_with_all_types(self, temp_db_dir):
        """Test creating database with all supported column types."""
        columns = {
            'IntCol': 'int',
            'FloatCol': 'float',
            'TextCol': 'text',
            'DateCol': 'date',
            'DateTimeCol': 'datetime',
        }

        db_path = create_custom_db(
            db_code='test_all_types',
            bar_size='1 hour',
            columns=columns,
            db_dir=temp_db_dir,
        )

        assert db_path.exists()

        # Verify column types
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(Price)")
        price_columns = {row[1]: row[2] for row in cursor.fetchall()}

        assert price_columns['IntCol'] == 'INTEGER'
        assert price_columns['FloatCol'] == 'REAL'
        assert price_columns['TextCol'] == 'TEXT'
        assert price_columns['DateCol'] == 'TEXT'
        assert price_columns['DateTimeCol'] == 'TEXT'

        conn.close()

    def test_create_db_already_exists(self, temp_db_dir):
        """Test that creating existing database raises error."""
        columns = {'Value': 'float'}

        # Create first time
        create_custom_db(
            db_code='test_duplicate',
            bar_size='1 day',
            columns=columns,
            db_dir=temp_db_dir,
        )

        # Try to create again
        with pytest.raises(FileExistsError, match="already exists"):
            create_custom_db(
                db_code='test_duplicate',
                bar_size='1 day',
                columns=columns,
                db_dir=temp_db_dir,
            )

    def test_create_db_invalid_column_name(self, temp_db_dir):
        """Test that invalid column names raise error."""
        columns = {'123Invalid': 'int'}  # Starts with number

        with pytest.raises(ValueError, match="Invalid column name"):
            create_custom_db(
                db_code='test_invalid',
                bar_size='1 day',
                columns=columns,
                db_dir=temp_db_dir,
            )

    def test_create_db_invalid_column_type(self, temp_db_dir):
        """Test that invalid column types raise error."""
        columns = {'Value': 'invalid_type'}

        with pytest.raises(ValueError, match="Invalid column type"):
            create_custom_db(
                db_code='test_invalid_type',
                bar_size='1 day',
                columns=columns,
                db_dir=temp_db_dir,
            )

    def test_create_db_empty_code(self, temp_db_dir):
        """Test that empty db_code raises error."""
        with pytest.raises(ValueError, match="db_code must be a non-empty string"):
            create_custom_db(
                db_code='',
                bar_size='1 day',
                columns={'Value': 'float'},
                db_dir=temp_db_dir,
            )

    def test_create_db_indices(self, temp_db_dir):
        """Test that indices are created."""
        columns = {'Value': 'float'}

        db_path = create_custom_db(
            db_code='test_indices',
            bar_size='1 day',
            columns=columns,
            db_dir=temp_db_dir,
        )

        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        # Check indices exist
        cursor.execute("SELECT name FROM sqlite_master WHERE type='index'")
        indices = [row[0] for row in cursor.fetchall()]

        # Should have indices on Date and Sid
        assert any('date' in idx.lower() for idx in indices)
        assert any('sid' in idx.lower() for idx in indices)

        conn.close()


class TestGetDBPath:
    """Test database path retrieval."""

    def test_get_db_path_exists(self, temp_db_dir):
        """Test getting path to existing database."""
        columns = {'Value': 'float'}
        create_custom_db('test_path', '1 day', columns, temp_db_dir)

        db_path = get_db_path('test_path', temp_db_dir)
        assert db_path.exists()
        assert db_path.name == 'quant_test_path.sqlite'

    def test_get_db_path_not_exists(self, temp_db_dir):
        """Test getting path to non-existent database raises error."""
        with pytest.raises(FileNotFoundError, match="Database not found"):
            get_db_path('nonexistent', temp_db_dir)


class TestConnectDB:
    """Test database connection."""

    def test_connect_db(self, temp_db_dir):
        """Test connecting to database."""
        columns = {'Value': 'float'}
        create_custom_db('test_connect', '1 day', columns, temp_db_dir)

        conn = connect_db('test_connect', temp_db_dir)
        assert conn is not None
        assert isinstance(conn, sqlite3.Connection)

        # Verify we can query
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        assert 'Price' in tables

        conn.close()

    def test_connect_db_not_exists(self, temp_db_dir):
        """Test connecting to non-existent database raises error."""
        with pytest.raises(FileNotFoundError):
            connect_db('nonexistent', temp_db_dir)


class TestListCustomDBs:
    """Test listing databases."""

    def test_list_empty_dir(self, temp_db_dir):
        """Test listing empty directory."""
        dbs = list_custom_dbs(temp_db_dir)
        assert dbs == []

    def test_list_multiple_dbs(self, temp_db_dir):
        """Test listing multiple databases."""
        columns = {'Value': 'float'}

        # Create multiple databases
        create_custom_db('test_a', '1 day', columns, temp_db_dir)
        create_custom_db('test_b', '1 day', columns, temp_db_dir)
        create_custom_db('test_c', '1 day', columns, temp_db_dir)

        dbs = list_custom_dbs(temp_db_dir)
        assert len(dbs) == 3
        assert 'test_a' in dbs
        assert 'test_b' in dbs
        assert 'test_c' in dbs

        # Should be sorted
        assert dbs == sorted(dbs)

    def test_list_nonexistent_dir(self, tmp_path):
        """Test listing non-existent directory."""
        nonexistent = tmp_path / 'nonexistent'
        dbs = list_custom_dbs(nonexistent)
        assert dbs == []


class TestDescribeCustomDB:
    """Test database description."""

    def test_describe_db(self, temp_db_dir):
        """Test describing a database."""
        columns = {
            'Revenue': 'int',
            'EPS': 'float',
            'Currency': 'text',
        }

        create_custom_db(
            db_code='test_describe',
            bar_size='1 quarter',
            columns=columns,
            db_dir=temp_db_dir,
        )

        info = describe_custom_db('test_describe', temp_db_dir)

        assert info['db_code'] == 'test_describe'
        assert info['bar_size'] == '1 quarter'
        assert info['columns'] == columns
        assert info['row_count'] == 0  # Empty database
        assert info['date_range'] is None  # No data
        assert info['sids'] == []
        assert info['num_sids'] == 0

    def test_describe_db_with_data(self, temp_db_dir):
        """Test describing database with data."""
        columns = {'Value': 'float'}

        db_path = create_custom_db(
            db_code='test_with_data',
            bar_size='1 day',
            columns=columns,
            db_dir=temp_db_dir,
        )

        # Insert test data
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO Price (Sid, Date, Value) VALUES (?, ?, ?)",
            ('101', '2020-01-01', 100.0)
        )
        cursor.execute(
            "INSERT INTO Price (Sid, Date, Value) VALUES (?, ?, ?)",
            ('102', '2020-01-02', 200.0)
        )
        conn.commit()
        conn.close()

        info = describe_custom_db('test_with_data', temp_db_dir)

        assert info['row_count'] == 2
        assert info['date_range'] == ('2020-01-01', '2020-01-02')
        assert set(info['sids']) == {'101', '102'}
        assert info['num_sids'] == 2

    def test_describe_db_not_exists(self, temp_db_dir):
        """Test describing non-existent database raises error."""
        with pytest.raises(FileNotFoundError):
            describe_custom_db('nonexistent', temp_db_dir)
