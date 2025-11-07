"""
Tests for zipline.data.custom.insert_utils module.
"""

import pytest
import sqlite3
import pandas as pd

from zipline.data.custom.insert_utils import (
    insert_or_replace,
    insert_or_ignore,
    insert_or_fail,
    get_insert_function,
    batch_insert,
)
from zipline.data.custom.db_manager import create_custom_db


@pytest.fixture
def test_db(tmp_path):
    """Create a test database."""
    db_dir = tmp_path / 'custom_data'
    db_dir.mkdir()

    columns = {'Value': 'float', 'Volume': 'int'}
    db_path = create_custom_db(
        db_code='test_insert',
        bar_size='1 day',
        columns=columns,
        db_dir=db_dir,
    )

    conn = sqlite3.connect(str(db_path))
    yield conn
    conn.close()


class TestInsertOrReplace:
    """Test INSERT OR REPLACE strategy."""

    def test_insert_new_rows(self, test_db):
        """Test inserting new rows."""
        df = pd.DataFrame({
            'Sid': ['101', '102'],
            'Date': ['2020-01-01', '2020-01-01'],
            'Value': [100.5, 200.5],
            'Volume': [1000, 2000],
        })

        rows_affected = insert_or_replace(df, 'Price', test_db)
        assert rows_affected == 2

        # Verify data
        cursor = test_db.cursor()
        cursor.execute("SELECT COUNT(*) FROM Price")
        assert cursor.fetchone()[0] == 2

    def test_replace_existing_rows(self, test_db):
        """Test replacing existing rows."""
        # Insert initial data
        df1 = pd.DataFrame({
            'Sid': ['101'],
            'Date': ['2020-01-01'],
            'Value': [100.0],
            'Volume': [1000],
        })
        insert_or_replace(df1, 'Price', test_db)

        # Replace with new data
        df2 = pd.DataFrame({
            'Sid': ['101'],
            'Date': ['2020-01-01'],
            'Value': [150.0],  # Changed
            'Volume': [1500],  # Changed
        })
        rows_affected = insert_or_replace(df2, 'Price', test_db)
        assert rows_affected == 1

        # Verify updated values
        cursor = test_db.cursor()
        cursor.execute("SELECT Value, Volume FROM Price WHERE Sid='101'")
        value, volume = cursor.fetchone()
        assert value == 150.0
        assert volume == 1500

    def test_empty_dataframe(self, test_db):
        """Test inserting empty DataFrame."""
        df = pd.DataFrame(columns=['Sid', 'Date', 'Value', 'Volume'])
        rows_affected = insert_or_replace(df, 'Price', test_db)
        assert rows_affected == 0


class TestInsertOrIgnore:
    """Test INSERT OR IGNORE strategy."""

    def test_insert_new_rows(self, test_db):
        """Test inserting new rows."""
        df = pd.DataFrame({
            'Sid': ['101', '102'],
            'Date': ['2020-01-01', '2020-01-01'],
            'Value': [100.5, 200.5],
            'Volume': [1000, 2000],
        })

        rows_affected = insert_or_ignore(df, 'Price', test_db)
        assert rows_affected == 2

    def test_ignore_existing_rows(self, test_db):
        """Test ignoring existing rows."""
        # Insert initial data
        df1 = pd.DataFrame({
            'Sid': ['101'],
            'Date': ['2020-01-01'],
            'Value': [100.0],
            'Volume': [1000],
        })
        insert_or_ignore(df1, 'Price', test_db)

        # Try to insert duplicate (should be ignored)
        df2 = pd.DataFrame({
            'Sid': ['101'],
            'Date': ['2020-01-01'],
            'Value': [150.0],  # Different value
            'Volume': [1500],
        })
        rows_affected = insert_or_ignore(df2, 'Price', test_db)
        assert rows_affected == 0  # Nothing inserted

        # Verify original values unchanged
        cursor = test_db.cursor()
        cursor.execute("SELECT Value, Volume FROM Price WHERE Sid='101'")
        value, volume = cursor.fetchone()
        assert value == 100.0
        assert volume == 1000


class TestInsertOrFail:
    """Test INSERT OR FAIL strategy."""

    def test_insert_new_rows(self, test_db):
        """Test inserting new rows."""
        df = pd.DataFrame({
            'Sid': ['101', '102'],
            'Date': ['2020-01-01', '2020-01-01'],
            'Value': [100.5, 200.5],
            'Volume': [1000, 2000],
        })

        rows_affected = insert_or_fail(df, 'Price', test_db)
        assert rows_affected == 2

    def test_fail_on_duplicate(self, test_db):
        """Test failing on duplicate rows."""
        # Insert initial data
        df1 = pd.DataFrame({
            'Sid': ['101'],
            'Date': ['2020-01-01'],
            'Value': [100.0],
            'Volume': [1000],
        })
        insert_or_fail(df1, 'Price', test_db)

        # Try to insert duplicate (should raise error)
        df2 = pd.DataFrame({
            'Sid': ['101'],
            'Date': ['2020-01-01'],
            'Value': [150.0],
            'Volume': [1500],
        })

        with pytest.raises(ValueError, match="Duplicate key violation"):
            insert_or_fail(df2, 'Price', test_db)


class TestGetInsertFunction:
    """Test insert function factory."""

    def test_get_replace_function(self):
        """Test getting replace function."""
        func = get_insert_function('replace')
        assert func == insert_or_replace

    def test_get_ignore_function(self):
        """Test getting ignore function."""
        func = get_insert_function('ignore')
        assert func == insert_or_ignore

    def test_get_fail_function(self):
        """Test getting fail function."""
        func = get_insert_function('fail')
        assert func == insert_or_fail

    def test_invalid_strategy(self):
        """Test invalid strategy raises error."""
        with pytest.raises(ValueError, match="Invalid on_duplicate strategy"):
            get_insert_function('invalid')


class TestBatchInsert:
    """Test batch insert functionality."""

    def test_batch_insert_small_data(self, test_db):
        """Test batch insert with small dataset."""
        df = pd.DataFrame({
            'Sid': ['101', '102', '103'],
            'Date': ['2020-01-01', '2020-01-01', '2020-01-01'],
            'Value': [100.0, 200.0, 300.0],
            'Volume': [1000, 2000, 3000],
        })

        total_rows = batch_insert(
            df, 'Price', test_db,
            on_duplicate='replace',
            batch_size=2,
        )

        assert total_rows == 3

        # Verify all data inserted
        cursor = test_db.cursor()
        cursor.execute("SELECT COUNT(*) FROM Price")
        assert cursor.fetchone()[0] == 3

    def test_batch_insert_large_data(self, test_db):
        """Test batch insert with large dataset."""
        # Create large dataset
        n_rows = 1000
        df = pd.DataFrame({
            'Sid': [f'{i}' for i in range(n_rows)],
            'Date': ['2020-01-01'] * n_rows,
            'Value': [float(i) for i in range(n_rows)],
            'Volume': [i * 100 for i in range(n_rows)],
        })

        total_rows = batch_insert(
            df, 'Price', test_db,
            on_duplicate='replace',
            batch_size=100,
        )

        assert total_rows == n_rows

        # Verify count
        cursor = test_db.cursor()
        cursor.execute("SELECT COUNT(*) FROM Price")
        assert cursor.fetchone()[0] == n_rows

    def test_batch_insert_with_ignore(self, test_db):
        """Test batch insert with ignore strategy."""
        # Insert initial data
        df1 = pd.DataFrame({
            'Sid': ['101', '102'],
            'Date': ['2020-01-01', '2020-01-01'],
            'Value': [100.0, 200.0],
            'Volume': [1000, 2000],
        })
        batch_insert(df1, 'Price', test_db, on_duplicate='replace')

        # Insert with duplicates (should ignore)
        df2 = pd.DataFrame({
            'Sid': ['101', '102', '103'],  # First two are duplicates
            'Date': ['2020-01-01', '2020-01-01', '2020-01-01'],
            'Value': [150.0, 250.0, 300.0],
            'Volume': [1500, 2500, 3000],
        })
        total_rows = batch_insert(df2, 'Price', test_db, on_duplicate='ignore')

        # Only one new row inserted
        assert total_rows == 1

        # Verify total count
        cursor = test_db.cursor()
        cursor.execute("SELECT COUNT(*) FROM Price")
        assert cursor.fetchone()[0] == 3
