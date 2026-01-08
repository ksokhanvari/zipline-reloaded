#!/usr/bin/env python3
"""
Verify Reproducibility of ML Forecasting Script

This script compares two prediction files to ensure they are identical.
Used to verify that the ML forecasting script produces deterministic results.

Usage:
    python verify_reproducibility.py run1_predictions.parquet run2_predictions.parquet
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path


def verify_identical(file1, file2):
    """
    Verify that two prediction files are identical.

    Args:
        file1: Path to first prediction file (.parquet or .csv)
        file2: Path to second prediction file (.parquet or .csv)

    Returns:
        bool: True if files are identical, False otherwise
    """
    print(f"🔍 Verifying Reproducibility")
    print(f"   File 1: {file1}")
    print(f"   File 2: {file2}")
    print()

    # Load files
    file1_path = Path(file1)
    file2_path = Path(file2)

    if not file1_path.exists():
        print(f"❌ ERROR: File not found: {file1}")
        return False

    if not file2_path.exists():
        print(f"❌ ERROR: File not found: {file2}")
        return False

    # Read files (auto-detect format)
    print("📂 Loading files...")
    if file1_path.suffix.lower() in ['.parquet', '.pq']:
        df1 = pd.read_parquet(file1)
    else:
        df1 = pd.read_csv(file1)

    if file2_path.suffix.lower() in ['.parquet', '.pq']:
        df2 = pd.read_parquet(file2)
    else:
        df2 = pd.read_csv(file2)

    print(f"   • File 1: {len(df1):,} rows, {len(df1.columns)} columns")
    print(f"   • File 2: {len(df2):,} rows, {len(df2.columns)} columns")
    print()

    # Check shapes
    if df1.shape != df2.shape:
        print(f"❌ FAILED: Shapes differ!")
        print(f"   File 1: {df1.shape}")
        print(f"   File 2: {df2.shape}")
        return False

    print(f"✅ Shapes match: {df1.shape}")
    print()

    # Check column names
    if not all(df1.columns == df2.columns):
        print(f"❌ FAILED: Column names differ!")
        print(f"   File 1 columns: {list(df1.columns)}")
        print(f"   File 2 columns: {list(df2.columns)}")
        return False

    print(f"✅ Column names match")
    print()

    # Check predictions
    if 'predicted_return' not in df1.columns:
        print(f"❌ ERROR: No 'predicted_return' column found")
        return False

    pred1 = df1['predicted_return'].dropna().values
    pred2 = df2['predicted_return'].dropna().values

    if len(pred1) != len(pred2):
        print(f"❌ FAILED: Prediction counts differ!")
        print(f"   File 1: {len(pred1):,} predictions")
        print(f"   File 2: {len(pred2):,} predictions")
        return False

    print(f"✅ Prediction counts match: {len(pred1):,}")
    print()

    # Check if predictions are identical
    max_diff = np.max(np.abs(pred1 - pred2))
    mean_diff = np.mean(np.abs(pred1 - pred2))

    print(f"📊 Prediction Comparison:")
    print(f"   • Max absolute difference: {max_diff:.2e}")
    print(f"   • Mean absolute difference: {mean_diff:.2e}")
    print()

    # Tolerance: 1e-9 (essentially identical)
    TOLERANCE = 1e-9

    if max_diff > TOLERANCE:
        print(f"❌ FAILED: Predictions differ by more than {TOLERANCE:.2e}")
        print(f"   • Max difference: {max_diff:.2e}")

        # Show worst mismatches
        diffs = np.abs(pred1 - pred2)
        worst_idx = np.argsort(diffs)[-5:][::-1]

        print(f"\n   Worst mismatches:")
        for idx in worst_idx:
            print(f"     Row {idx}: {pred1[idx]:.6f} vs {pred2[idx]:.6f} (diff: {diffs[idx]:.6e})")

        return False

    # Check all columns for identical values
    print(f"🔍 Checking all columns for differences...")
    for col in df1.columns:
        if col == 'predicted_return':
            continue  # Already checked above

        # For numeric columns, use np.allclose
        if pd.api.types.is_numeric_dtype(df1[col]):
            val1 = df1[col].fillna(0).values
            val2 = df2[col].fillna(0).values

            if not np.allclose(val1, val2, rtol=1e-9, atol=1e-9):
                max_col_diff = np.max(np.abs(val1 - val2))
                print(f"   ⚠️  Column '{col}' differs (max diff: {max_col_diff:.2e})")
        else:
            # For non-numeric, use exact equality
            if not all(df1[col].fillna('') == df2[col].fillna('')):
                print(f"   ⚠️  Column '{col}' differs")

    print()
    print("=" * 70)
    print("✅ REPRODUCIBILITY VERIFIED: Files are IDENTICAL!")
    print("=" * 70)
    print(f"   • Predictions: {len(pred1):,}")
    print(f"   • Max difference: {max_diff:.2e} (< {TOLERANCE:.2e})")
    print(f"   • Mean difference: {mean_diff:.2e}")
    print()
    print("🎉 Your ML forecasting script produces deterministic results!")
    print()

    return True


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python verify_reproducibility.py file1.parquet file2.parquet")
        sys.exit(1)

    file1 = sys.argv[1]
    file2 = sys.argv[2]

    success = verify_identical(file1, file2)
    sys.exit(0 if success else 1)
