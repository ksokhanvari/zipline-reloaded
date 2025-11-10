"""
Database class pattern for custom data integration.

This module provides a class-based approach to defining custom datasets
that is more Pythonic and easier to work with than the functional approach.

Example:
    >>> from zipline.pipeline.data.db import Database, Column
    >>>
    >>> class Fundamentals(Database):
    ...     CODE = "fundamentals"
    ...     LOOKBACK_WINDOW = 240
    ...
    ...     ROE = Column(float)
    ...     PERatio = Column(float)
    ...     Sector = Column(str)
    >>>
    >>> # Use in pipeline
    >>> roe = Fundamentals.ROE.latest
"""

from zipline.data.custom import make_custom_dataset_class


class Column:
    """
    Column descriptor for Database class.

    Parameters
    ----------
    dtype : type
        Python type for the column (float, int, str, etc.)
    """
    def __init__(self, dtype):
        self.dtype = dtype
        self.name = None

    def __set_name__(self, owner, name):
        self.name = name


class DatabaseMeta(type):
    """Metaclass for Database that converts Column descriptors to dataset columns."""

    def __new__(mcs, name, bases, namespace):
        # Skip for base Database class
        if name == 'Database':
            return super().__new__(mcs, name, bases, namespace)

        # Extract CODE and columns
        code = namespace.get('CODE')
        if not code:
            raise ValueError(f"Database class {name} must define CODE attribute")

        # Build columns dict from Column descriptors
        columns = {}
        for attr_name, attr_value in list(namespace.items()):
            if isinstance(attr_value, Column):
                # Map Python types to Zipline types
                dtype = attr_value.dtype
                if dtype == float:
                    zipline_type = 'float'
                elif dtype == int:
                    zipline_type = 'int'
                elif dtype == str:
                    zipline_type = 'text'
                elif dtype == bool:
                    zipline_type = 'bool'
                else:
                    # Default to float for unknown types
                    zipline_type = 'float'

                columns[attr_name] = zipline_type

        if not columns:
            raise ValueError(f"Database class {name} must define at least one Column")

        # Create the actual dataset class using make_custom_dataset_class
        dataset_class = make_custom_dataset_class(
            db_code=code,
            columns=columns,
            base_name=name
        )

        # Copy over the dataset's column attributes to our class namespace
        for col_name in columns:
            namespace[col_name] = getattr(dataset_class, col_name)

        # Store the generated dataset class
        namespace['_dataset_class'] = dataset_class

        return super().__new__(mcs, name, bases, namespace)


class Database(metaclass=DatabaseMeta):
    """
    Base class for defining custom data sources.

    Subclass this and define columns using the Column descriptor:

    Example:
        >>> class Fundamentals(Database):
        ...     CODE = "fundamentals"  # Database code (required)
        ...     LOOKBACK_WINDOW = 240  # Optional: days to look back
        ...
        ...     # Define columns with types
        ...     ROE = Column(float)
        ...     PERatio = Column(float)
        ...     Sector = Column(str)
        ...
        >>> # Use in pipeline
        >>> roe = Fundamentals.ROE.latest
        >>> high_roe = (roe > 15.0)

    Attributes
    ----------
    CODE : str
        Database code that matches the custom database created with
        create_custom_db(). This is required.

    LOOKBACK_WINDOW : int, optional
        Number of days to look back for data. Default is 1.
    """
    CODE = None
    LOOKBACK_WINDOW = 1


__all__ = ['Database', 'Column']
