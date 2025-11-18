"""
Bundle Registration Utility

This module provides a centralized way to register Zipline bundles for use in
standalone scripts and notebooks.

The extension.py file is only loaded by Zipline CLI commands (like 'zipline bundles'),
but NOT when importing zipline in Python scripts. This utility ensures bundles are
registered consistently across all example code.

Usage
-----
Import this at the top of any script that uses bundles:

    from register_bundles import ensure_bundles_registered
    ensure_bundles_registered()

Or use it as a context manager:

    with ensure_bundles_registered():
        bundle_data = load_bundle('sharadar')
"""

from zipline.data.bundles import register
from zipline.data.bundles.sharadar_bundle import sharadar_bundle, register_sharadar_bundles


_BUNDLES_REGISTERED = False


def ensure_bundles_registered(verbose=False):
    """
    Ensure Sharadar bundles are registered.

    This function is idempotent - calling it multiple times is safe.

    Parameters
    ----------
    verbose : bool, optional
        If True, print registration status. Default is False.

    Examples
    --------
    >>> from register_bundles import ensure_bundles_registered
    >>> ensure_bundles_registered()
    >>> from zipline.data.bundles import load as load_bundle
    >>> bundle_data = load_bundle('sharadar')
    """
    global _BUNDLES_REGISTERED

    if _BUNDLES_REGISTERED:
        if verbose:
            print("✓ Bundles already registered")
        return

    try:
        # Register the base sharadar bundle (normally done in extension.py)
        register('sharadar', sharadar_bundle(
            tickers=None,  # All tickers
            incremental=True,
            include_funds=True,
        ))

        # Register the bundle variants (tech, sp500, all)
        register_sharadar_bundles()

        _BUNDLES_REGISTERED = True

        if verbose:
            print("✓ Sharadar bundles registered")

    except Exception as e:
        if verbose:
            print(f"⚠ Warning: Could not register Sharadar bundles: {e}")
        raise


def __enter__():
    """Context manager entry - register bundles."""
    ensure_bundles_registered()
    return None


def __exit__(exc_type, exc_val, exc_tb):
    """Context manager exit - nothing to clean up."""
    return False


# Auto-register when module is imported (convenience)
# Scripts that want explicit control can import without triggering this
if __name__ != '__main__':
    ensure_bundles_registered(verbose=False)
