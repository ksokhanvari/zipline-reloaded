"""
Custom Pipeline Factors for Sharadar Ticker Metadata Filtering

Provides custom factors to filter universe by exchange, category, and ADR status.
These factors work around the limitation that string columns don't support .in_() method.

Usage:
    from sharadar_filters import ExchangeFilter, CategoryFilter, ADRFilter

    # In your strategy file, pass the metadata columns from your ms.Database:
    exchange_filter = ExchangeFilter(CustomFundamentals.sharadar_exchange)
    category_filter = CategoryFilter(CustomFundamentals.sharadar_category)
    adr_filter = ADRFilter(CustomFundamentals.sharadar_is_adr)

    base_universe = exchange_filter & category_filter & adr_filter
"""

from zipline.pipeline.filters import CustomFilter
import numpy as np


def ExchangeFilter(exchange_column):
    """
    Filter stocks by exchange (NYSE, NASDAQ, NYSEMKT).

    Parameters
    ----------
    exchange_column : BoundColumn
        The exchange column from your fundamentals database (e.g., CustomFundamentals.sharadar_exchange)

    Returns
    -------
    CustomFilter
        Filter that returns True if stock is listed on one of the major US exchanges.
    """
    class _ExchangeFilter(CustomFilter):
        window_length = 1

        def compute(self, today, assets, out, exchange):
            # Check if exchange is in our target list
            # exchange is a 2D array (1 day x N assets)
            valid_exchanges = {'NYSE', 'NASDAQ', 'NYSEMKT'}

            # Handle object dtype from SQLite
            exchange_vals = exchange[-1]

            # Vectorized string comparison
            out[:] = np.array([str(val) in valid_exchanges if val is not None else False
                               for val in exchange_vals], dtype=bool)

    # Set inputs as class attribute after class definition
    _ExchangeFilter.inputs = [exchange_column]
    return _ExchangeFilter()


def CategoryFilter(category_column):
    """
    Filter to domestic common stocks only.

    Parameters
    ----------
    category_column : BoundColumn
        The category column from your fundamentals database (e.g., CustomFundamentals.sharadar_category)

    Returns
    -------
    CustomFilter
        Filter that returns True if category is 'Domestic Common Stock'.
    """
    class _CategoryFilter(CustomFilter):
        window_length = 1

        def compute(self, today, assets, out, category):
            # Check if category is Domestic Common Stock
            # Handle object dtype from SQLite
            category_vals = category[-1]

            # Vectorized string comparison
            out[:] = np.array([str(val) == 'Domestic Common Stock' if val is not None else False
                               for val in category_vals], dtype=bool)

    _CategoryFilter.inputs = [category_column]
    return _CategoryFilter()


def ADRFilter(is_adr_column):
    """
    Filter to identify NON-ADRs (domestic stocks).

    Parameters
    ----------
    is_adr_column : BoundColumn
        The ADR flag column from your fundamentals database (e.g., CustomFundamentals.sharadar_is_adr)

    Returns
    -------
    CustomFilter
        Filter that returns True if stock is NOT an ADR (i.e., domestic stock).
        Use this filter directly without the ~ operator.
    """
    class _ADRFilter(CustomFilter):
        window_length = 1

        def compute(self, today, assets, out, is_adr):
            # Return True for NON-ADRs (inverted logic)
            # Handle various input types (bool, int, object)
            is_adr_vals = is_adr[-1]

            def is_not_adr(val):
                if isinstance(val, (bool, np.bool_)):
                    return not val
                elif isinstance(val, (int, np.integer, float, np.floating)):
                    return val == 0  # 0 = not ADR, 1 = ADR
                else:
                    return True  # Default to non-ADR if unclear

            out[:] = np.array([is_not_adr(val) for val in is_adr_vals], dtype=bool)

    _ADRFilter.inputs = [is_adr_column]
    return _ADRFilter()


def SectorFilter(sector_column, sectors=None):
    """
    Filter stocks by sector.

    Parameters
    ----------
    sector_column : BoundColumn
        The sector column from your fundamentals database (e.g., CustomFundamentals.sharadar_sector)
    sectors : list of str
        List of sectors to include (e.g., ['Technology', 'Healthcare'])

    Returns
    -------
    CustomFilter
        Filter that returns True if stock's sector is in the specified list.
    """
    class _SectorFilter(CustomFilter):
        window_length = 1
        params = {'sectors': sectors or []}

        def compute(self, today, assets, out, sector, sectors):
            # Check if sector is in target list
            # Handle object dtype from SQLite
            sector_vals = sector[-1]

            if sectors:
                out[:] = np.array([str(val) in sectors if val is not None else False
                                   for val in sector_vals], dtype=bool)
            else:
                # No sectors specified - accept all
                out[:] = True

    _SectorFilter.inputs = [sector_column]
    return _SectorFilter()


def ScaleMarketCapFilter(scalemarketcap_column, min_scale=1, max_scale=6):
    """
    Filter stocks by market cap scale.

    Sharadar market cap scales:
    - 1 - Nano (< $50M)
    - 2 - Micro ($50M - $300M)
    - 3 - Small ($300M - $2B)
    - 4 - Mid ($2B - $10B)
    - 5 - Large ($10B - $200B)
    - 6 - Mega (> $200B)

    Parameters
    ----------
    scalemarketcap_column : BoundColumn
        The market cap scale column from your fundamentals database (e.g., CustomFundamentals.sharadar_scalemarketcap)
    min_scale : int
        Minimum market cap scale (1-6)
    max_scale : int
        Maximum market cap scale (1-6)

    Returns
    -------
    CustomFilter
        Filter that returns True if stock's market cap scale is in the specified range.
    """
    class _ScaleMarketCapFilter(CustomFilter):
        window_length = 1
        params = {'min_scale': min_scale, 'max_scale': max_scale}

        def compute(self, today, assets, out, scalemarketcap, min_scale, max_scale):
            # Parse scale from string like "6 - Mega"
            # Extract first number
            def get_scale(s):
                if isinstance(s, str) and s:
                    try:
                        return int(s[0])
                    except:
                        return 0
                else:
                    return 0

            scales = np.array([get_scale(s) for s in scalemarketcap[-1]], dtype=int)

            # Filter by scale range
            out[:] = (scales >= min_scale) & (scales <= max_scale)

    _ScaleMarketCapFilter.inputs = [scalemarketcap_column]
    return _ScaleMarketCapFilter()


# Convenience function to create standard Sharadar universe
def create_sharadar_universe(
    fundamentals_dataset,
    exchanges=None,
    include_adrs=False,
    sectors=None,
    min_market_cap_scale=None,
    max_market_cap_scale=None,
):
    """
    Create a standard Sharadar universe filter.

    Parameters
    ----------
    fundamentals_dataset : ms.Database
        The fundamentals dataset (e.g., CustomFundamentals) that contains the metadata columns
    exchanges : list of str, optional
        Exchanges to include (default: ['NYSE', 'NASDAQ', 'NYSEMKT'])
    include_adrs : bool, default False
        Include American Depositary Receipts
    sectors : list of str, optional
        Sectors to include (default: all sectors)
    min_market_cap_scale : int, optional
        Minimum market cap scale 1-6 (default: no minimum)
    max_market_cap_scale : int, optional
        Maximum market cap scale 1-6 (default: no maximum)

    Returns
    -------
    Filter
        Combined filter for universe selection

    Examples
    --------
    >>> # Standard US equity universe (no ADRs, major exchanges)
    >>> universe = create_sharadar_universe(CustomFundamentals)

    >>> # Large-cap tech stocks only
    >>> universe = create_sharadar_universe(
    ...     CustomFundamentals,
    ...     sectors=['Technology'],
    ...     min_market_cap_scale=5,  # Large + Mega cap
    ... )

    >>> # Include all stocks including ADRs
    >>> universe = create_sharadar_universe(CustomFundamentals, include_adrs=True)
    """
    filters = []

    # Exchange filter (default to major US exchanges)
    if exchanges is None:
        exchanges = ['NYSE', 'NASDAQ', 'NYSEMKT']

    if exchanges:
        filters.append(ExchangeFilter(fundamentals_dataset.sharadar_exchange))

    # Category filter - always domestic common stock
    filters.append(CategoryFilter(fundamentals_dataset.sharadar_category))

    # ADR filter (ADRFilter already returns True for non-ADRs)
    if not include_adrs:
        filters.append(ADRFilter(fundamentals_dataset.sharadar_is_adr))

    # Sector filter
    if sectors:
        filters.append(SectorFilter(fundamentals_dataset.sharadar_sector, sectors=sectors))

    # Market cap scale filter
    if min_market_cap_scale is not None or max_market_cap_scale is not None:
        min_s = min_market_cap_scale if min_market_cap_scale is not None else 1
        max_s = max_market_cap_scale if max_market_cap_scale is not None else 6
        filters.append(ScaleMarketCapFilter(
            fundamentals_dataset.sharadar_scalemarketcap,
            min_scale=min_s,
            max_scale=max_s
        ))

    # Combine all filters with AND logic
    if len(filters) == 0:
        return None
    elif len(filters) == 1:
        return filters[0]
    else:
        # Combine with & operator
        result = filters[0]
        for f in filters[1:]:
            result = result & f
        return result
