"""
Tests for zipline.data.custom.loader module.
"""

import pytest
import pandas as pd
from pathlib import Path

from zipline.data.custom.db_manager import create_custom_db, connect_db
from zipline.data.custom.loader import load_csv_to_db


@pytest.fixture
def temp_db_dir(tmp_path):
    """Temporary directory for test databases."""
    db_dir = tmp_path / 'custom_data'
    db_dir.mkdir()
    return db_dir


@pytest.fixture
def test_csv(tmp_path):
    """Create test CSV file."""
    csv_path = tmp_path / 'test_data.csv'
    df = pd.DataFrame({
        'Symbol': ['AAPL', 'MSFT', 'GOOGL', 'AAPL', 'MSFT'],
        'Date': ['2020-01-01', '2020-01-01', '2020-01-01', '2020-01-02', '2020-01-02'],
        'Revenue': [1000000, 2000000, 3000000, 1100000, 2100000],
        'EPS': [1.5, 2.0, 3.0, 1.6, 2.1],
    })
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def securities_csv(tmp_path):
    """Create securities mapping CSV."""
    csv_path = tmp_path / 'securities.csv'
    df = pd.DataFrame({
        'Symbol': ['AAPL', 'MSFT', 'GOOGL'],
        'Sid': [24, 5061, 26890],
    })
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def test_db(temp_db_dir):
    """Create test database."""
    columns = {'Revenue': 'int', 'EPS': 'float'}
    create_custom_db(
        db_code='test_loader',
        bar_size='1 day',
        columns=columns,
        db_dir=temp_db_dir,
    )
    return temp_db_dir


class TestLoadCSVToDB:
    """Test CSV loading functionality."""

    def test_load_with_csv_mapping(self, test_csv, securities_csv, test_db):
        """Test loading CSV with securities CSV mapping."""
        result = load_csv_to_db(
            csv_path=test_csv,
            db_code='test_loader',
            sid_map=pd.read_csv(securities_csv),
            id_col='Symbol',
            date_col='Date',
            db_dir=test_db,
        )

        assert result['rows_inserted'] == 5
        assert result['rows_skipped'] == 0
        assert len(result['unmapped_ids']) == 0
        assert len(result['errors']) == 0

        # Verify data in database
        conn = connect_db('test_loader', test_db)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM Price")
        assert cursor.fetchone()[0] == 5
        conn.close()

    def test_load_with_dict_mapping(self, test_csv, test_db):
        """Test loading CSV with dictionary mapping."""
        sid_map = {'AAPL': 24, 'MSFT': 5061, 'GOOGL': 26890}

        result = load_csv_to_db(
            csv_path=test_csv,
            db_code='test_loader',
            sid_map=sid_map,
            id_col='Symbol',
            date_col='Date',
            db_dir=test_db,
        )

        assert result['rows_inserted'] == 5
        assert result['rows_skipped'] == 0

    def test_load_without_mapping(self, tmp_path, test_db):
        """Test loading CSV without mapping (using identifiers as Sids)."""
        # Create CSV with integer identifiers
        csv_path = tmp_path / 'no_mapping.csv'
        df = pd.DataFrame({
            'Sid': [100, 101, 102],
            'Date': ['2020-01-01', '2020-01-01', '2020-01-01'],
            'Revenue': [1000, 2000, 3000],
            'EPS': [1.0, 2.0, 3.0],
        })
        df.to_csv(csv_path, index=False)

        result = load_csv_to_db(
            csv_path=csv_path,
            db_code='test_loader',
            sid_map=None,
            id_col='Sid',
            date_col='Date',
            db_dir=test_db,
        )

        assert result['rows_inserted'] == 3

    def test_load_with_unmapped_ids(self, tmp_path, test_db):
        """Test loading CSV with unmapped identifiers."""
        # Create CSV with symbol not in mapping
        csv_path = tmp_path / 'unmapped.csv'
        df = pd.DataFrame({
            'Symbol': ['AAPL', 'UNKNOWN'],
            'Date': ['2020-01-01', '2020-01-01'],
            'Revenue': [1000, 2000],
            'EPS': [1.0, 2.0],
        })
        df.to_csv(csv_path, index=False)

        sid_map = {'AAPL': 24}

        # With fail_on_unmapped=False
        result = load_csv_to_db(
            csv_path=csv_path,
            db_code='test_loader',
            sid_map=sid_map,
            id_col='Symbol',
            date_col='Date',
            fail_on_unmapped=False,
            db_dir=test_db,
        )

        assert result['rows_inserted'] == 1  # Only AAPL
        assert 'UNKNOWN' in result['unmapped_ids']

        # With fail_on_unmapped=True
        with pytest.raises(ValueError, match="unmapped identifiers"):
            load_csv_to_db(
                csv_path=csv_path,
                db_code='test_loader',
                sid_map=sid_map,
                id_col='Symbol',
                date_col='Date',
                fail_on_unmapped=True,
                db_dir=test_db,
            )

    def test_load_with_duplicates_replace(self, test_csv, test_db):
        """Test loading with duplicates using replace strategy."""
        sid_map = {'AAPL': 24, 'MSFT': 5061, 'GOOGL': 26890}

        # Load first time
        result1 = load_csv_to_db(
            csv_path=test_csv,
            db_code='test_loader',
            sid_map=sid_map,
            on_duplicate='replace',
            db_dir=test_db,
        )
        assert result1['rows_inserted'] == 5

        # Load again (should replace)
        result2 = load_csv_to_db(
            csv_path=test_csv,
            db_code='test_loader',
            sid_map=sid_map,
            on_duplicate='replace',
            db_dir=test_db,
        )
        assert result2['rows_inserted'] == 5

        # Verify still only 5 rows
        conn = connect_db('test_loader', test_db)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM Price")
        assert cursor.fetchone()[0] == 5
        conn.close()

    def test_load_with_duplicates_ignore(self, test_csv, test_db):
        """Test loading with duplicates using ignore strategy."""
        sid_map = {'AAPL': 24, 'MSFT': 5061, 'GOOGL': 26890}

        # Load first time
        load_csv_to_db(
            csv_path=test_csv,
            db_code='test_loader',
            sid_map=sid_map,
            on_duplicate='replace',
            db_dir=test_db,
        )

        # Load again with ignore (should skip all)
        result2 = load_csv_to_db(
            csv_path=test_csv,
            db_code='test_loader',
            sid_map=sid_map,
            on_duplicate='ignore',
            db_dir=test_db,
        )
        assert result2['rows_inserted'] == 0  # All ignored

    def test_load_with_duplicates_fail(self, test_csv, test_db):
        """Test loading with duplicates using fail strategy."""
        sid_map = {'AAPL': 24, 'MSFT': 5061, 'GOOGL': 26890}

        # Load first time
        load_csv_to_db(
            csv_path=test_csv,
            db_code='test_loader',
            sid_map=sid_map,
            on_duplicate='replace',
            db_dir=test_db,
        )

        # Load again with fail (should raise error)
        with pytest.raises(ValueError, match="Duplicate key violation"):
            load_csv_to_db(
                csv_path=test_csv,
                db_code='test_loader',
                sid_map=sid_map,
                on_duplicate='fail',
                db_dir=test_db,
            )

    def test_load_with_custom_date_format(self, tmp_path, test_db):
        """Test loading with custom date format."""
        csv_path = tmp_path / 'custom_date.csv'
        df = pd.DataFrame({
            'Symbol': ['AAPL'],
            'Date': ['01/15/2020'],  # MM/DD/YYYY format
            'Revenue': [1000],
            'EPS': [1.0],
        })
        df.to_csv(csv_path, index=False)

        sid_map = {'AAPL': 24}

        result = load_csv_to_db(
            csv_path=csv_path,
            db_code='test_loader',
            sid_map=sid_map,
            date_format='%m/%d/%Y',
            db_dir=test_db,
        )

        assert result['rows_inserted'] == 1

        # Verify date stored correctly
        conn = connect_db('test_loader', test_db)
        cursor = conn.cursor()
        cursor.execute("SELECT Date FROM Price")
        date_str = cursor.fetchone()[0]
        assert '2020-01-15' in date_str
        conn.close()

    def test_load_with_chunking(self, tmp_path, test_db):
        """Test loading large CSV with chunking."""
        # Create large CSV
        csv_path = tmp_path / 'large.csv'
        n_rows = 250
        df = pd.DataFrame({
            'Symbol': ['AAPL'] * n_rows,
            'Date': [f'2020-01-{i % 28 + 1:02d}' for i in range(n_rows)],
            'Revenue': [1000 + i for i in range(n_rows)],
            'EPS': [1.0 + i * 0.1 for i in range(n_rows)],
        })
        df.to_csv(csv_path, index=False)

        sid_map = {'AAPL': 24}

        result = load_csv_to_db(
            csv_path=csv_path,
            db_code='test_loader',
            sid_map=sid_map,
            chunk_size=100,  # Process in chunks of 100
            db_dir=test_db,
        )

        assert result['rows_inserted'] == n_rows

    def test_load_missing_columns(self, tmp_path, test_db):
        """Test loading CSV with missing required columns."""
        csv_path = tmp_path / 'missing_cols.csv'
        df = pd.DataFrame({
            'Symbol': ['AAPL'],
            'Date': ['2020-01-01'],
            # Missing Revenue and EPS
        })
        df.to_csv(csv_path, index=False)

        sid_map = {'AAPL': 24}

        with pytest.raises(Exception):  # Should raise some error
            load_csv_to_db(
                csv_path=csv_path,
                db_code='test_loader',
                sid_map=sid_map,
                db_dir=test_db,
            )

    def test_load_nonexistent_csv(self, test_db):
        """Test loading non-existent CSV raises error."""
        with pytest.raises(FileNotFoundError):
            load_csv_to_db(
                csv_path='nonexistent.csv',
                db_code='test_loader',
                db_dir=test_db,
            )

    def test_load_nonexistent_db(self, test_csv, temp_db_dir):
        """Test loading to non-existent database raises error."""
        with pytest.raises(FileNotFoundError):
            load_csv_to_db(
                csv_path=test_csv,
                db_code='nonexistent',
                db_dir=temp_db_dir,
            )
