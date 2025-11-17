"""
Automatic loader setup for multi-source Pipeline backtests.

This module provides automatic detection and configuration of data loaders,
making it easy to mix Sharadar fundamentals with custom data sources.

Usage
-----
In your algorithm, instead of manually setting up loaders, just import and use:

    from zipline.pipeline.loaders.auto_loader import setup_auto_loader

    # In run_algorithm() call
    results = run_algorithm(
        ...,
        custom_loader=setup_auto_loader(),  # That's it!
    )

This automatically:
- Loads Sharadar fundamentals
- Detects and loads custom databases
- Handles SID translation for run_algorithm()
- Routes columns to the appropriate loaders

Example
-------
>>> from zipline import run_algorithm
>>> from zipline.pipeline.loaders.auto_loader import setup_auto_loader
>>>
>>> results = run_algorithm(
...     start=pd.Timestamp('2023-01-01'),
...     end=pd.Timestamp('2024-01-01'),
...     initialize=initialize,
...     bundle='sharadar',
...     custom_loader=setup_auto_loader(),
... )
"""

from pathlib import Path
import logging

from zipline.data.bundles import load as load_bundle
from zipline.data.custom import CustomSQLiteLoader
from zipline.pipeline.loaders.sharadar_fundamentals import make_sharadar_fundamentals_loader

log = logging.getLogger(__name__)


class AutoLoader(dict):
    """
    Automatic multi-source loader that routes Pipeline columns to appropriate loaders.

    This loader automatically:
    - Detects Sharadar fundamentals columns
    - Detects custom database columns
    - Handles SID translation for custom databases
    - Caches loader instances for performance

    Parameters
    ----------
    bundle_name : str, optional
        Bundle name for Sharadar data. Default: 'sharadar'
    custom_db_dir : str or Path, optional
        Directory containing custom databases. Default: ~/.zipline/data/custom
    enable_sid_translation : bool, optional
        Enable automatic SID translation for custom loaders. Default: True
    """

    def __init__(
        self,
        bundle_name='sharadar',
        custom_db_dir=None,
        enable_sid_translation=True,
    ):
        super().__init__()

        self.bundle_name = bundle_name
        self.enable_sid_translation = enable_sid_translation

        # Set custom database directory
        if custom_db_dir is None:
            custom_db_dir = Path.home() / '.zipline' / 'data' / 'custom'
        self.custom_db_dir = Path(custom_db_dir)

        # Loader cache
        self._sharadar_loader = None
        self._custom_loaders = {}  # db_code -> loader
        self._asset_finder = None

        log.info("AutoLoader initialized")
        log.info(f"  Bundle: {bundle_name}")
        log.info(f"  Custom DB directory: {self.custom_db_dir}")
        log.info(f"  SID translation: {'enabled' if enable_sid_translation else 'disabled'}")

    def _get_asset_finder(self):
        """Lazy load asset_finder for SID translation."""
        if self._asset_finder is None and self.enable_sid_translation:
            try:
                bundle_data = load_bundle(self.bundle_name)
                self._asset_finder = bundle_data.asset_finder
                log.debug("Asset finder loaded for SID translation")
            except Exception as e:
                log.warning(f"Could not load asset finder for SID translation: {e}")
        return self._asset_finder

    def _get_sharadar_loader(self):
        """Lazy load Sharadar fundamentals loader."""
        if self._sharadar_loader is None:
            self._sharadar_loader = make_sharadar_fundamentals_loader(self.bundle_name)
            log.debug("Sharadar fundamentals loader created")
        return self._sharadar_loader

    def _get_custom_loader(self, db_code):
        """
        Lazy load custom database loader.

        Parameters
        ----------
        db_code : str
            Database code/identifier

        Returns
        -------
        CustomSQLiteLoader
            Loader for the specified database
        """
        if db_code not in self._custom_loaders:
            # Create loader with optional SID translation
            asset_finder = self._get_asset_finder() if self.enable_sid_translation else None

            loader = CustomSQLiteLoader(
                db_code=db_code,
                db_dir=self.custom_db_dir,
                asset_finder=asset_finder,
            )

            self._custom_loaders[db_code] = loader
            log.debug(f"Custom loader created for '{db_code}' database")

        return self._custom_loaders[db_code]

    def get(self, key, default=None):
        """
        Get the appropriate loader for a Pipeline column.

        This method routes columns to the correct loader based on their dataset.

        Parameters
        ----------
        key : BoundColumn
            Pipeline column
        default : optional
            Default value if no loader found

        Returns
        -------
        PipelineLoader
            Appropriate loader for the column
        """
        # Check if it's a registered column first
        if key in self:
            return self[key]

        # Check if it's a BoundColumn with a dataset
        if not hasattr(key, 'dataset'):
            raise KeyError(f"No loader for {key}")

        # Extract dataset name
        dataset_name = self._get_dataset_name(key.dataset)

        # Route to Sharadar loader
        if 'Sharadar' in dataset_name:
            return self._get_sharadar_loader()

        # Route to custom loader
        # Check if dataset has a CODE attribute (custom Database classes)
        if hasattr(key.dataset, 'CODE'):
            db_code = key.dataset.CODE
            return self._get_custom_loader(db_code)

        # Check if dataset name contains "DataSet" suffix from Database classes
        # Database classes create datasets with names like "LSEGFundamentalsDataSet"
        # We need to extract the CODE from the parent Database class
        if 'DataSet' in dataset_name:
            # Try to find the CODE by checking the dataset's attributes
            for attr_name in dir(key.dataset):
                if attr_name == 'CODE':
                    db_code = getattr(key.dataset, attr_name)
                    if db_code:
                        return self._get_custom_loader(db_code)

        # Try to match by column name in registered loaders
        if hasattr(key, 'name'):
            for registered_col, loader in self.items():
                if hasattr(registered_col, 'name') and registered_col.name == key.name:
                    return loader

        # No loader found
        raise KeyError(f"No loader for {key} (dataset: {dataset_name})")

    def _get_dataset_name(self, dataset):
        """
        Extract dataset name from dataset object.

        Parameters
        ----------
        dataset : DataSet
            Pipeline dataset

        Returns
        -------
        str
            Dataset name
        """
        # Try __name__ attribute first
        if hasattr(dataset, '__name__'):
            return dataset.__name__

        # Parse from string representation
        dataset_str = str(dataset)
        if "'" in dataset_str:
            return dataset_str.split("'")[1]

        return dataset_str


def setup_auto_loader(
    bundle_name='sharadar',
    custom_db_dir=None,
    enable_sid_translation=True,
):
    """
    Create an AutoLoader for automatic multi-source Pipeline backtests.

    This is the main entry point for easy multi-source data integration.

    Parameters
    ----------
    bundle_name : str, optional
        Bundle name for Sharadar data. Default: 'sharadar'
    custom_db_dir : str or Path, optional
        Directory containing custom databases. Default: ~/.zipline/data/custom
    enable_sid_translation : bool, optional
        Enable automatic SID translation for custom loaders. Default: True

    Returns
    -------
    AutoLoader
        Configured loader ready for use with run_algorithm()

    Examples
    --------
    Basic usage:

    >>> from zipline import run_algorithm
    >>> from zipline.pipeline.loaders.auto_loader import setup_auto_loader
    >>>
    >>> results = run_algorithm(
    ...     start=pd.Timestamp('2023-01-01'),
    ...     end=pd.Timestamp('2024-01-01'),
    ...     initialize=initialize,
    ...     bundle='sharadar',
    ...     custom_loader=setup_auto_loader(),
    ... )

    With custom settings:

    >>> loader = setup_auto_loader(
    ...     bundle_name='sharadar',
    ...     custom_db_dir='/path/to/custom/dbs',
    ...     enable_sid_translation=True,
    ... )
    """
    return AutoLoader(
        bundle_name=bundle_name,
        custom_db_dir=custom_db_dir,
        enable_sid_translation=enable_sid_translation,
    )
