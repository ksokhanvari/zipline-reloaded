"""
Tests for zipline.data.custom.query module.
"""

import pytest
import pandas as pd
import sqlite3

from zipline.data.custom.db_manager import create_custom_db, connect_db
from zipline.data.custom.query import (
    get_prices,
    get_prices_reindexed_like,
    get_latest_values,
)


@pytest.fixture
def test_db_with_data(tmp_path):
    """Create test database with sample data."""
    db_dir = tmp_path / 'custom_data'
    db_dir.mkdir()

    # Create database
    columns = {'Revenue': 'int', 'EPS': 'float', 'Currency': 'text'}
    db_path = create_custom_db(
        db_code='test_query',
        bar_size='1 quarter',
        columns=columns,
        db_dir=db_dir,
    )

    # Insert test data
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    test_data = [
        ('24', '2020-01-01', 1000000, 1.5, 'USD'),
        ('24', '2020-04-01', 1100000, 1.6, 'USD'),
        ('24', '2020-07-01', 1200000, 1.7, 'USD'),
        ('5061', '2020-01-01', 2000000, 2.0, 'USD'),
        ('5061', '2020-04-01', 2100000, 2.1, 'USD'),
        ('26890', '2020-01-01', 3000000, 3.0, 'USD'),
    ]

    for row in test_data:
        cursor.execute(
            "INSERT INTO Price (Sid, Date, Revenue, EPS, Currency) VALUES (?, ?, ?, ?, ?)",
            row
        )

    conn.commit()
    conn.close()

    return db_dir


class TestGetPrices:
    """Test get_prices function."""

    def test_get_all_prices(self, test_db_with_data):
        """Test getting all prices without filters."""
        df = get_prices('test_query', db_dir=test_db_with_data)

        assert len(df) == 6
        assert list(df.columns) == ['Sid', 'Date', 'Revenue', 'EPS', 'Currency']
        assert df['Sid'].nunique() == 3

    def test_get_prices_date_range(self, test_db_with_data):
        """Test filtering by date range."""
        df = get_prices(
            'test_query',
            start_date='2020-04-01',
            end_date='2020-07-01',
            db_dir=test_db_with_data,
        )

        assert len(df) == 3  # 2 for AAPL, 1 for MSFT
        dates = pd.to_datetime(df['Date'])
        assert dates.min() >= pd.Timestamp('2020-04-01')
        assert dates.max() <= pd.Timestamp('2020-07-01')

    def test_get_prices_specific_sids(self, test_db_with_data):
        """Test filtering by specific Sids."""
        df = get_prices(
            'test_query',
            sids=[24, 5061],
            db_dir=test_db_with_data,
        )

        assert len(df) == 5  # 3 for AAPL, 2 for MSFT
        assert set(df['Sid'].astype(int)) == {24, 5061}

    def test_get_prices_specific_fields(self, test_db_with_data):
        """Test retrieving specific fields."""
        df = get_prices(
            'test_query',
            fields=['Revenue', 'EPS'],
            db_dir=test_db_with_data,
        )

        # Should have Sid, Date, and requested fields
        assert set(df.columns) == {'Sid', 'Date', 'Revenue', 'EPS'}
        assert 'Currency' not in df.columns

    def test_get_prices_combined_filters(self, test_db_with_data):
        """Test combining multiple filters."""
        df = get_prices(
            'test_query',
            start_date='2020-01-01',
            end_date='2020-04-01',
            sids=[24],
            fields=['Revenue'],
            db_dir=test_db_with_data,
        )

        assert len(df) == 2  # AAPL in Q1 and Q2
        assert set(df.columns) == {'Sid', 'Date', 'Revenue'}
        assert (df['Sid'].astype(int) == 24).all()

    def test_get_prices_empty_result(self, test_db_with_data):
        """Test query with no matching data."""
        df = get_prices(
            'test_query',
            start_date='2025-01-01',
            end_date='2025-12-31',
            db_dir=test_db_with_data,
        )

        assert df.empty

    def test_get_prices_nonexistent_db(self, tmp_path):
        """Test querying non-existent database raises error."""
        db_dir = tmp_path / 'custom_data'
        db_dir.mkdir()

        with pytest.raises(FileNotFoundError):
            get_prices('nonexistent', db_dir=db_dir)


class TestGetPricesReindexedLike:
    """Test get_prices_reindexed_like function."""

    def test_reindex_basic(self, test_db_with_data):
        """Test reindexing to match template."""
        # Create template with specific dates and Sids
        dates = pd.to_datetime(['2020-01-01', '2020-04-01', '2020-07-01'])
        sids = [24, 5061]
        index = pd.MultiIndex.from_product([dates, sids], names=['Date', 'Sid'])
        template = pd.DataFrame(index=index)

        df = get_prices_reindexed_like(
            'test_query',
            template,
            db_dir=test_db_with_data,
        )

        # Should match template shape
        assert len(df) == 6  # 3 dates * 2 sids
        assert isinstance(df.index, pd.MultiIndex)

    def test_reindex_with_fields(self, test_db_with_data):
        """Test reindexing with specific fields."""
        dates = pd.to_datetime(['2020-01-01', '2020-04-01'])
        sids = [24]
        index = pd.MultiIndex.from_product([dates, sids], names=['Date', 'Sid'])
        template = pd.DataFrame(index=index)

        df = get_prices_reindexed_like(
            'test_query',
            template,
            fields=['Revenue'],
            db_dir=test_db_with_data,
        )

        assert 'Revenue' in df.columns
        assert 'EPS' not in df.columns

    def test_reindex_with_missing_data(self, test_db_with_data):
        """Test reindexing when some data is missing."""
        # Include date/sid combinations that don't exist
        dates = pd.to_datetime(['2020-01-01', '2020-10-01'])  # Oct doesn't exist
        sids = [24, 99999]  # 99999 doesn't exist
        index = pd.MultiIndex.from_product([dates, sids], names=['Date', 'Sid'])
        template = pd.DataFrame(index=index)

        df = get_prices_reindexed_like(
            'test_query',
            template,
            db_dir=test_db_with_data,
        )

        # Should still return data, with NaN for missing
        assert len(df) == 4  # 2 dates * 2 sids
        assert df.isnull().any().any()  # Some values should be NaN


class TestGetLatestValues:
    """Test get_latest_values function."""

    def test_get_latest_all_sids(self, test_db_with_data):
        """Test getting latest values for all Sids."""
        df = get_latest_values(
            'test_query',
            as_of_date='2020-07-01',
            db_dir=test_db_with_data,
        )

        # Should have one row per Sid with latest available data
        assert len(df) == 3
        assert set(df['Sid'].astype(int)) == {24, 5061, 26890}

        # Check AAPL has July data
        aapl_row = df[df['Sid'].astype(int) == 24].iloc[0]
        assert aapl_row['Revenue'] == 1200000

    def test_get_latest_specific_sids(self, test_db_with_data):
        """Test getting latest values for specific Sids."""
        df = get_latest_values(
            'test_query',
            as_of_date='2020-04-01',
            sids=[24, 5061],
            db_dir=test_db_with_data,
        )

        assert len(df) == 2
        assert set(df['Sid'].astype(int)) == {24, 5061}

    def test_get_latest_specific_fields(self, test_db_with_data):
        """Test getting latest values for specific fields."""
        df = get_latest_values(
            'test_query',
            as_of_date='2020-07-01',
            fields=['Revenue'],
            db_dir=test_db_with_data,
        )

        assert 'Revenue' in df.columns
        assert 'EPS' not in df.columns

    def test_get_latest_early_date(self, test_db_with_data):
        """Test getting latest values for date before any data."""
        df = get_latest_values(
            'test_query',
            as_of_date='2019-01-01',
            db_dir=test_db_with_data,
        )

        # Should be empty since no data before this date
        assert df.empty

    def test_get_latest_point_in_time(self, test_db_with_data):
        """Test point-in-time correctness."""
        # As of April 1, GOOGL should only have Jan data
        df = get_latest_values(
            'test_query',
            as_of_date='2020-04-01',
            sids=[26890],
            db_dir=test_db_with_data,
        )

        assert len(df) == 1
        googl_row = df.iloc[0]
        # Should have Jan data (3000000), not later data
        assert googl_row['Revenue'] == 3000000
        # Date should be Jan, not April
        assert '2020-01-01' in str(googl_row['Date'])
