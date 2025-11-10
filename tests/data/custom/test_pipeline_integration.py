"""
Tests for zipline.data.custom.pipeline_integration module.
"""

import pytest
import numpy as np
import pandas as pd
import sqlite3

from zipline.data.custom.db_manager import create_custom_db
from zipline.data.custom.pipeline_integration import (
    make_custom_dataset_class,
    CustomSQLiteLoader,
    TYPE_TO_DTYPE,
)
from zipline.pipeline.data import DataSet, Column
from zipline.lib.adjusted_array import AdjustedArray


@pytest.fixture
def test_db_with_data(tmp_path):
    """Create test database with sample data."""
    db_dir = tmp_path / 'custom_data'
    db_dir.mkdir()

    # Create database
    columns = {'Revenue': 'int', 'EPS': 'float', 'Rating': 'text'}
    db_path = create_custom_db(
        db_code='test_pipeline',
        bar_size='1 day',
        columns=columns,
        db_dir=db_dir,
    )

    # Insert test data (3 assets, 3 days)
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    test_data = [
        # Asset 24 (AAPL)
        ('24', '2020-01-01', 100, 1.0, 'A'),
        ('24', '2020-01-02', 110, 1.1, 'A'),
        ('24', '2020-01-03', 120, 1.2, 'A'),
        # Asset 5061 (MSFT)
        ('5061', '2020-01-01', 200, 2.0, 'B'),
        ('5061', '2020-01-02', 210, 2.1, 'B'),
        ('5061', '2020-01-03', 220, 2.2, 'B'),
        # Asset 26890 (GOOGL) - missing day 2
        ('26890', '2020-01-01', 300, 3.0, 'C'),
        ('26890', '2020-01-03', 320, 3.2, 'C'),
    ]

    for row in test_data:
        cursor.execute(
            "INSERT INTO Price (Sid, Date, Revenue, EPS, Rating) VALUES (?, ?, ?, ?, ?)",
            row
        )

    conn.commit()
    conn.close()

    return db_dir


class TestMakeCustomDatasetClass:
    """Test DataSet class creation."""

    def test_create_basic_dataset(self):
        """Test creating basic DataSet class."""
        columns = {'Revenue': 'int', 'EPS': 'float'}

        DataSetClass = make_custom_dataset_class(
            db_code='test',
            columns=columns,
        )

        # Verify it's a DataSet subclass
        assert issubclass(DataSetClass, DataSet)

        # Verify columns exist
        assert hasattr(DataSetClass, 'Revenue')
        assert hasattr(DataSetClass, 'EPS')

        # Verify column types
        assert hasattr(DataSetClass.Revenue, 'dtype')
        assert hasattr(DataSetClass.EPS, 'dtype')

    def test_dataset_column_dtypes(self):
        """Test that columns have correct dtypes."""
        columns = {
            'IntCol': 'int',
            'FloatCol': 'float',
            'TextCol': 'text',
        }

        DataSetClass = make_custom_dataset_class(
            db_code='test',
            columns=columns,
        )

        # Check dtype mappings
        assert DataSetClass.IntCol.dtype == TYPE_TO_DTYPE['int']
        assert DataSetClass.FloatCol.dtype == TYPE_TO_DTYPE['float']
        assert DataSetClass.TextCol.dtype == TYPE_TO_DTYPE['text']

    def test_dataset_column_missing_values(self):
        """Test that columns have appropriate missing values."""
        columns = {
            'IntCol': 'int',
            'FloatCol': 'float',
            'TextCol': 'text',
        }

        DataSetClass = make_custom_dataset_class(
            db_code='test',
            columns=columns,
        )

        # Int columns should use -1 as missing value
        assert DataSetClass.IntCol.missing_value == -1

        # Float columns should use NaN as missing value
        assert np.isnan(DataSetClass.FloatCol.missing_value)

        # Text columns should use None as missing value
        assert DataSetClass.TextCol.missing_value is None

    def test_dataset_custom_name(self):
        """Test custom DataSet class name."""
        columns = {'Value': 'float'}

        DataSetClass = make_custom_dataset_class(
            db_code='my_data',
            columns=columns,
            base_name='CustomData',
        )

        assert DataSetClass.__name__ == 'CustomDataDataSet'

    def test_dataset_auto_name(self):
        """Test automatic DataSet class naming."""
        columns = {'Value': 'float'}

        # Test snake_case to PascalCase conversion
        DataSetClass = make_custom_dataset_class(
            db_code='my_custom_data',
            columns=columns,
        )

        assert DataSetClass.__name__ == 'MyCustomDataDataSet'


class TestCustomSQLiteLoader:
    """Test CustomSQLiteLoader class."""

    def test_loader_initialization(self, test_db_with_data):
        """Test loader initialization."""
        loader = CustomSQLiteLoader('test_pipeline', test_db_with_data)

        assert loader.db_code == 'test_pipeline'
        assert loader.db_dir == test_db_with_data
        assert isinstance(loader._cache, dict)

    def test_load_adjusted_array_basic(self, test_db_with_data):
        """Test loading data as AdjustedArrays."""
        # Create DataSet and loader
        columns = {'Revenue': 'int', 'EPS': 'float', 'Rating': 'text'}
        DataSetClass = make_custom_dataset_class(
            db_code='test_pipeline',
            columns=columns,
        )
        loader = CustomSQLiteLoader('test_pipeline', test_db_with_data)

        # Prepare parameters for load_adjusted_array
        dates = pd.DatetimeIndex([
            pd.Timestamp('2020-01-01'),
            pd.Timestamp('2020-01-02'),
            pd.Timestamp('2020-01-03'),
        ])
        sids = pd.Int64Index([24, 5061, 26890])
        mask = np.ones((len(dates), len(sids)), dtype=bool)
        columns_to_load = [DataSetClass.Revenue]

        # Load data
        result = loader.load_adjusted_array(
            domain=None,  # Not used in our implementation
            columns=columns_to_load,
            dates=dates,
            sids=sids,
            mask=mask,
        )

        # Verify result structure
        assert isinstance(result, dict)
        assert DataSetClass.Revenue in result

        # Verify AdjustedArray
        adj_array = result[DataSetClass.Revenue]
        assert isinstance(adj_array, AdjustedArray)

        # Verify shape: (dates, sids)
        assert adj_array.data.shape == (3, 3)

        # Verify data values (AAPL Revenue on day 1)
        assert adj_array.data[0, 0] == 100  # AAPL, 2020-01-01

    def test_load_multiple_columns(self, test_db_with_data):
        """Test loading multiple columns."""
        columns = {'Revenue': 'int', 'EPS': 'float'}
        DataSetClass = make_custom_dataset_class(
            db_code='test_pipeline',
            columns=columns,
        )
        loader = CustomSQLiteLoader('test_pipeline', test_db_with_data)

        dates = pd.DatetimeIndex([pd.Timestamp('2020-01-01')])
        sids = pd.Int64Index([24, 5061])
        mask = np.ones((1, 2), dtype=bool)
        columns_to_load = [DataSetClass.Revenue, DataSetClass.EPS]

        result = loader.load_adjusted_array(
            domain=None,
            columns=columns_to_load,
            dates=dates,
            sids=sids,
            mask=mask,
        )

        # Should have both columns
        assert DataSetClass.Revenue in result
        assert DataSetClass.EPS in result

        # Verify Revenue values
        assert result[DataSetClass.Revenue].data[0, 0] == 100  # AAPL
        assert result[DataSetClass.Revenue].data[0, 1] == 200  # MSFT

        # Verify EPS values
        assert result[DataSetClass.EPS].data[0, 0] == 1.0  # AAPL
        assert result[DataSetClass.EPS].data[0, 1] == 2.0  # MSFT

    def test_load_with_missing_data(self, test_db_with_data):
        """Test loading when some data is missing."""
        columns = {'Revenue': 'int'}
        DataSetClass = make_custom_dataset_class(
            db_code='test_pipeline',
            columns=columns,
        )
        loader = CustomSQLiteLoader('test_pipeline', test_db_with_data)

        # Include day 2, where GOOGL is missing
        dates = pd.DatetimeIndex([
            pd.Timestamp('2020-01-01'),
            pd.Timestamp('2020-01-02'),
            pd.Timestamp('2020-01-03'),
        ])
        sids = pd.Int64Index([26890])  # GOOGL
        mask = np.ones((3, 1), dtype=bool)

        result = loader.load_adjusted_array(
            domain=None,
            columns=[DataSetClass.Revenue],
            dates=dates,
            sids=sids,
            mask=mask,
        )

        data = result[DataSetClass.Revenue].data

        # Day 1 should have value
        assert data[0, 0] == 300

        # Day 2 should be NaN (missing)
        assert np.isnan(data[1, 0])

        # Day 3 should have value
        assert data[2, 0] == 320

    def test_load_with_date_range(self, test_db_with_data):
        """Test loading data for specific date range."""
        columns = {'Revenue': 'int'}
        DataSetClass = make_custom_dataset_class(
            db_code='test_pipeline',
            columns=columns,
        )
        loader = CustomSQLiteLoader('test_pipeline', test_db_with_data)

        # Query only days 2 and 3
        dates = pd.DatetimeIndex([
            pd.Timestamp('2020-01-02'),
            pd.Timestamp('2020-01-03'),
        ])
        sids = pd.Int64Index([24])  # AAPL
        mask = np.ones((2, 1), dtype=bool)

        result = loader.load_adjusted_array(
            domain=None,
            columns=[DataSetClass.Revenue],
            dates=dates,
            sids=sids,
            mask=mask,
        )

        data = result[DataSetClass.Revenue].data

        # Should have correct values for days 2 and 3
        assert data[0, 0] == 110  # Day 2
        assert data[1, 0] == 120  # Day 3

    def test_load_nonexistent_column(self, test_db_with_data):
        """Test loading non-existent column."""
        columns = {'Revenue': 'int', 'NonExistent': 'float'}
        DataSetClass = make_custom_dataset_class(
            db_code='test_pipeline',
            columns=columns,
        )
        loader = CustomSQLiteLoader('test_pipeline', test_db_with_data)

        dates = pd.DatetimeIndex([pd.Timestamp('2020-01-01')])
        sids = pd.Int64Index([24])
        mask = np.ones((1, 1), dtype=bool)

        # This should not raise, but return missing values for NonExistent
        result = loader.load_adjusted_array(
            domain=None,
            columns=[DataSetClass.NonExistent],
            dates=dates,
            sids=sids,
            mask=mask,
        )

        # Should have the column with all missing values
        assert DataSetClass.NonExistent in result
        data = result[DataSetClass.NonExistent].data
        # All values should be NaN (missing)
        assert np.isnan(data[0, 0])

    def test_adjusted_array_properties(self, test_db_with_data):
        """Test AdjustedArray properties."""
        columns = {'Revenue': 'int'}
        DataSetClass = make_custom_dataset_class(
            db_code='test_pipeline',
            columns=columns,
        )
        loader = CustomSQLiteLoader('test_pipeline', test_db_with_data)

        dates = pd.DatetimeIndex([pd.Timestamp('2020-01-01')])
        sids = pd.Int64Index([24])
        mask = np.ones((1, 1), dtype=bool)

        result = loader.load_adjusted_array(
            domain=None,
            columns=[DataSetClass.Revenue],
            dates=dates,
            sids=sids,
            mask=mask,
        )

        adj_array = result[DataSetClass.Revenue]

        # Check that there are no adjustments (custom data doesn't have splits, etc.)
        assert adj_array.adjustments == {}

        # Check missing value is set correctly
        assert adj_array.missing_value == -1  # For int columns
