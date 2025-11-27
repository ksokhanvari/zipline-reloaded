"""
Custom Pipeline Factors for Sharadar Ticker Metadata Filtering

Provides custom factors to filter universe by exchange, category, and ADR status.
These factors work around the limitation that string columns don't support .in_() method.

Usage:
    from sharadar_filters import ExchangeFilter, CategoryFilter, ADRFilter

    # Filter to NYSE/NASDAQ domestic common stocks, no ADRs
    exchange_filter = ExchangeFilter()
    category_filter = CategoryFilter()
    adr_filter = ADRFilter()

    base_universe = exchange_filter & category_filter & ~adr_filter
"""

from zipline.pipeline.filters import CustomFilter
from zipline.pipeline.data import DataSet, Column
import numpy as np


class SharadarTickers(DataSet):
    """Sharadar ticker metadata for universe filtering."""

    CODE = "fundamentals"
    LOOKBACK_WINDOW = 1  # Metadata is static, doesn't change daily

    exchange = Column(object, missing_value='')
    category = Column(object, missing_value='')
    is_adr = Column(bool)
    location = Column(object, missing_value='')
    sector = Column(object, missing_value='')
    industry = Column(object, missing_value='')
    sicsector = Column(object, missing_value='')
    sicindustry = Column(object, missing_value='')
    scalemarketcap = Column(object, missing_value='')


class ExchangeFilter(CustomFilter):
    """
    Filter stocks by exchange (NYSE, NASDAQ, NYSEMKT).

    Returns True if stock is listed on one of the major US exchanges.
    """
    inputs = [SharadarTickers.exchange]
    window_length = 1

    def compute(self, today, assets, out, exchange):
        # Check if exchange is in our target list
        # exchange is a 2D array (1 day x N assets)
        valid_exchanges = {'NYSE', 'NASDAQ', 'NYSEMKT'}
        out[:] = np.isin(exchange[-1], list(valid_exchanges))


class CategoryFilter(CustomFilter):
    """
    Filter to domestic common stocks only.

    Returns True if category is 'Domestic Common Stock'.
    """
    inputs = [SharadarTickers.category]
    window_length = 1

    def compute(self, today, assets, out, category):
        # Check if category is Domestic Common Stock
        out[:] = (category[-1] == 'Domestic Common Stock')


class ADRFilter(CustomFilter):
    """
    Filter to identify NON-ADRs (domestic stocks).

    Returns True if stock is NOT an ADR (i.e., domestic stock).
    Use this filter directly without the ~ operator.
    """
    inputs = [SharadarTickers.is_adr]
    window_length = 1

    def compute(self, today, assets, out, is_adr):
        # Return True for NON-ADRs (inverted logic)
        out[:] = ~is_adr[-1].astype(bool)


class SectorFilter(CustomFilter):
    """
    Filter stocks by sector.

    Parameters
    ----------
    sectors : list of str
        List of sectors to include (e.g., ['Technology', 'Healthcare'])
    """
    inputs = [SharadarTickers.sector]
    window_length = 1
    params = {'sectors': []}

    def compute(self, today, assets, out, sector, sectors):
        # Check if sector is in target list
        if sectors:
            out[:] = np.isin(sector[-1], sectors)
        else:
            # No sectors specified - accept all
            out[:] = True


class ScaleMarketCapFilter(CustomFilter):
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
    min_scale : int
        Minimum market cap scale (1-6)
    max_scale : int
        Maximum market cap scale (1-6)
    """
    inputs = [SharadarTickers.scalemarketcap]
    window_length = 1
    params = {'min_scale': 1, 'max_scale': 6}

    def compute(self, today, assets, out, scalemarketcap, min_scale, max_scale):
        # Parse scale from string like "6 - Mega"
        # Extract first number
        scales = np.zeros(len(scalemarketcap[-1]), dtype=int)

        for i, s in enumerate(scalemarketcap[-1]):
            if isinstance(s, str) and s:
                try:
                    # Extract first digit
                    scales[i] = int(s[0])
                except:
                    scales[i] = 0
            else:
                scales[i] = 0

        # Filter by scale range
        out[:] = (scales >= min_scale) & (scales <= max_scale)


# Convenience function to create standard Sharadar universe
def create_sharadar_universe(
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
    >>> universe = create_sharadar_universe()

    >>> # Large-cap tech stocks only
    >>> universe = create_sharadar_universe(
    ...     sectors=['Technology'],
    ...     min_market_cap_scale=5,  # Large + Mega cap
    ... )

    >>> # Include all stocks including ADRs
    >>> universe = create_sharadar_universe(include_adrs=True)
    """
    filters = []

    # Exchange filter (default to major US exchanges)
    if exchanges is None:
        exchanges = ['NYSE', 'NASDAQ', 'NYSEMKT']

    if exchanges:
        filters.append(ExchangeFilter())

    # Category filter - always domestic common stock
    filters.append(CategoryFilter())

    # ADR filter (ADRFilter already returns True for non-ADRs)
    if not include_adrs:
        filters.append(ADRFilter())

    # Sector filter
    if sectors:
        filters.append(SectorFilter(sectors=sectors))

    # Market cap scale filter
    if min_market_cap_scale is not None or max_market_cap_scale is not None:
        min_s = min_market_cap_scale if min_market_cap_scale is not None else 1
        max_s = max_market_cap_scale if max_market_cap_scale is not None else 6
        filters.append(ScaleMarketCapFilter(min_scale=min_s, max_scale=max_s))

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
