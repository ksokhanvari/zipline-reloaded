"""
Automatic Symbol Mapping for LSEG Data Loads

This module provides intelligent, automated symbol mapping that:
1. Detects unmapped symbols in your CSV
2. Attempts fuzzy matching based on company names
3. Generates/updates a persistent mapping file
4. Applies mappings automatically on future loads

NO MORE manual point fixes for individual symbols!
"""

import pandas as pd
import json
from pathlib import Path
from difflib import SequenceMatcher


class AutoSymbolMapper:
    """
    Automatically maps CSV symbols to bundle symbols using company names.

    Learns from your data and builds a persistent mapping file that works
    for all future loads.
    """

    def __init__(self, asset_finder, mapping_file='/data/custom_databases/symbol_mappings.json'):
        """
        Initialize auto mapper.

        Args:
            asset_finder: Zipline AssetFinder instance
            mapping_file: Path to persistent JSON mapping file
        """
        self.asset_finder = asset_finder
        self.mapping_file = Path(mapping_file)
        self.mappings = self._load_mappings()
        self._bundle_symbols_cache = None

    def _load_mappings(self):
        """Load existing mappings from file."""
        if self.mapping_file.exists():
            with open(self.mapping_file, 'r') as f:
                return json.load(f)
        return {}

    def _save_mappings(self):
        """Save mappings to file."""
        self.mapping_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.mapping_file, 'w') as f:
            json.dump(self.mappings, f, indent=2)

    def _get_bundle_symbols(self):
        """Get all symbols and names from bundle (cached)."""
        if self._bundle_symbols_cache is not None:
            return self._bundle_symbols_cache

        print("  Building bundle symbol cache...")
        bundle_symbols = {}

        for sid in self.asset_finder.sids:
            try:
                asset = self.asset_finder.retrieve_asset(sid)
                # Try to get symbol from asset
                if hasattr(asset, 'symbol') and asset.symbol:
                    symbol = asset.symbol
                    name = getattr(asset, 'asset_name', '').upper()
                    bundle_symbols[symbol] = {
                        'sid': sid,
                        'name': name,
                        'symbol': symbol
                    }
            except:
                continue

        self._bundle_symbols_cache = bundle_symbols
        print(f"  Cached {len(bundle_symbols):,} bundle symbols")
        return bundle_symbols

    def _fuzzy_match_by_name(self, csv_name, csv_symbol, threshold=0.8):
        """
        Find best bundle symbol match based on company name similarity.

        Args:
            csv_name: Company name from CSV
            csv_symbol: Symbol from CSV (for context)
            threshold: Minimum similarity score (0-1)

        Returns:
            Tuple of (bundle_symbol, similarity_score) or (None, 0)
        """
        if not csv_name or csv_name == '':
            return None, 0.0

        csv_name_upper = csv_name.upper()
        bundle_symbols = self._get_bundle_symbols()

        best_match = None
        best_score = 0.0

        for bundle_symbol, info in bundle_symbols.items():
            bundle_name = info['name']
            if not bundle_name:
                continue

            # Calculate similarity
            score = SequenceMatcher(None, csv_name_upper, bundle_name).ratio()

            # Bonus if symbols are similar
            symbol_similarity = SequenceMatcher(None, csv_symbol.upper(), bundle_symbol.upper()).ratio()
            if symbol_similarity > 0.5:
                score += 0.1  # Small bonus

            if score > best_score:
                best_score = score
                best_match = bundle_symbol

        if best_score >= threshold:
            return best_match, best_score
        return None, best_score

    def detect_unmapped_symbols(self, df, symbol_col='Symbol', name_col='CompanyCommonName',
                                verbose=True):
        """
        Detect symbols in CSV that aren't directly in the bundle.

        Args:
            df: DataFrame with symbol and company name columns
            symbol_col: Name of symbol column
            name_col: Name of company name column
            verbose: Print progress

        Returns:
            DataFrame with unmapped symbols and suggested mappings
        """
        if verbose:
            print("\n" + "="*80)
            print("DETECTING UNMAPPED SYMBOLS")
            print("="*80)

        bundle_symbols = self._get_bundle_symbols()

        # Get unique symbols from CSV
        unique_symbols = df[[symbol_col, name_col]].drop_duplicates(subset=[symbol_col])

        unmapped = []

        for _, row in unique_symbols.iterrows():
            csv_symbol = row[symbol_col]
            csv_name = row[name_col]

            # Check if already in our mappings
            if csv_symbol in self.mappings:
                continue

            # Check if directly in bundle
            if csv_symbol in bundle_symbols:
                continue

            # Not found - try fuzzy match
            best_match, score = self._fuzzy_match_by_name(csv_name, csv_symbol, threshold=0.7)

            unmapped.append({
                'csv_symbol': csv_symbol,
                'csv_name': csv_name,
                'suggested_symbol': best_match,
                'confidence': score,
                'action': 'auto' if score >= 0.85 else 'review'
            })

        unmapped_df = pd.DataFrame(unmapped)

        if verbose and len(unmapped_df) > 0:
            print(f"\nFound {len(unmapped_df)} unmapped symbols")
            print(f"  High confidence (>=0.85): {(unmapped_df['confidence'] >= 0.85).sum()}")
            print(f"  Needs review (0.7-0.85): {((unmapped_df['confidence'] >= 0.7) & (unmapped_df['confidence'] < 0.85)).sum()}")
            print(f"  No match (<0.7): {(unmapped_df['confidence'] < 0.7).sum()}")

        return unmapped_df

    def apply_suggestions(self, unmapped_df, auto_threshold=0.85, verbose=True):
        """
        Apply high-confidence suggestions to mappings.

        Args:
            unmapped_df: DataFrame from detect_unmapped_symbols()
            auto_threshold: Confidence threshold for auto-acceptance
            verbose: Print actions

        Returns:
            Tuple of (auto_applied, needs_review)
        """
        if verbose:
            print("\n" + "="*80)
            print("APPLYING SYMBOL MAPPINGS")
            print("="*80)

        auto_applied = []
        needs_review = []

        for _, row in unmapped_df.iterrows():
            csv_symbol = row['csv_symbol']
            suggested = row['suggested_symbol']
            confidence = row['confidence']

            if confidence >= auto_threshold and suggested:
                # Auto-apply high confidence matches
                self.mappings[csv_symbol] = suggested
                auto_applied.append(row)

                if verbose:
                    print(f"  ✓ {csv_symbol} → {suggested} (confidence: {confidence:.2f})")

            elif confidence >= 0.7 and suggested:
                # Flag for review
                needs_review.append(row)

            else:
                # No good match
                if verbose:
                    print(f"  ✗ {csv_symbol}: No match found (best: {confidence:.2f})")

        # Save updated mappings
        if auto_applied:
            self._save_mappings()
            if verbose:
                print(f"\n✓ Auto-applied {len(auto_applied)} mappings")
                print(f"  Saved to: {self.mapping_file}")

        if needs_review and verbose:
            print(f"\n⚠ {len(needs_review)} symbols need manual review:")
            review_df = pd.DataFrame(needs_review)
            print(review_df[['csv_symbol', 'csv_name', 'suggested_symbol', 'confidence']].to_string(index=False))
            print(f"\nTo manually add mappings, edit: {self.mapping_file}")
            print("Format: {\"CSV_SYMBOL\": \"BUNDLE_SYMBOL\"}")

        return auto_applied, needs_review

    def map_symbols(self, df, symbol_col='Symbol', verbose=True):
        """
        Apply all known mappings to dataframe symbols.

        Args:
            df: DataFrame to map
            symbol_col: Name of symbol column
            verbose: Print stats

        Returns:
            DataFrame with mapped symbols
        """
        if not self.mappings:
            if verbose:
                print("  No mappings to apply")
            return df

        df = df.copy()

        # Apply mappings
        mapped_count = 0
        for csv_symbol, bundle_symbol in self.mappings.items():
            mask = df[symbol_col] == csv_symbol
            if mask.any():
                df.loc[mask, symbol_col] = bundle_symbol
                mapped_count += mask.sum()

        if verbose and mapped_count > 0:
            print(f"  ✓ Applied {len(self.mappings)} symbol mappings to {mapped_count:,} rows")

        return df

    def auto_learn_and_map(self, df, symbol_col='Symbol', name_col='CompanyCommonName',
                          auto_threshold=0.85, verbose=True):
        """
        Complete workflow: detect, learn, and apply mappings.

        This is the main method to use. It:
        1. Detects unmapped symbols
        2. Finds best matches using company names
        3. Auto-applies high-confidence matches
        4. Saves mappings for future use
        5. Returns mapped dataframe

        Args:
            df: DataFrame with symbols and company names
            symbol_col: Symbol column name
            name_col: Company name column name
            auto_threshold: Confidence for auto-apply (0.85 = 85% match)
            verbose: Print detailed progress

        Returns:
            DataFrame with symbols mapped to bundle symbols
        """
        if verbose:
            print("="*80)
            print("AUTO SYMBOL MAPPER")
            print("="*80)
            print(f"Mapping file: {self.mapping_file}")
            print(f"Current mappings: {len(self.mappings)}")

        # Detect unmapped symbols
        unmapped = self.detect_unmapped_symbols(df, symbol_col, name_col, verbose)

        # Apply suggestions
        if len(unmapped) > 0:
            auto_applied, needs_review = self.apply_suggestions(
                unmapped,
                auto_threshold,
                verbose
            )

        # Apply all mappings to dataframe
        if verbose:
            print("\n" + "="*80)
            print("MAPPING DATAFRAME")
            print("="*80)

        df_mapped = self.map_symbols(df, symbol_col, verbose)

        if verbose:
            print("="*80)

        return df_mapped


def generate_mapping_report(mapping_file='/root/.zipline/data/custom/symbol_mappings.json'):
    """
    Generate a readable report of current symbol mappings.

    Args:
        mapping_file: Path to mapping JSON file

    Returns:
        DataFrame with mapping details
    """
    mapping_path = Path(mapping_file)

    if not mapping_path.exists():
        print(f"No mapping file found at: {mapping_file}")
        return pd.DataFrame()

    with open(mapping_path, 'r') as f:
        mappings = json.load(f)

    if not mappings:
        print("Mapping file is empty")
        return pd.DataFrame()

    report = pd.DataFrame([
        {'CSV_Symbol': k, 'Bundle_Symbol': v}
        for k, v in sorted(mappings.items())
    ])

    print("="*80)
    print("CURRENT SYMBOL MAPPINGS")
    print("="*80)
    print(f"Source: {mapping_file}")
    print(f"Total mappings: {len(mappings)}\n")
    print(report.to_string(index=False))
    print("="*80)

    return report


# Example usage
if __name__ == '__main__':
    print(__doc__)
    print("\nUsage in load_csv_fundamentals.ipynb:")
    print("""
# After loading CSV, before SID mapping:

from symbol_mapper_auto import AutoSymbolMapper

# Create auto mapper
auto_mapper = AutoSymbolMapper(asset_finder)

# Auto-detect and apply mappings (one line!)
custom_data = auto_mapper.auto_learn_and_map(
    custom_data,
    symbol_col='Symbol',
    name_col='CompanyCommonName',
    auto_threshold=0.85,  # 85% confidence for auto-apply
    verbose=True
)

# Now proceed with normal SID mapping - all symbols are normalized!
""")
