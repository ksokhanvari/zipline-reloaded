"""
ALTERNATIVE OPTIMIZATION: MultiIndex Mapping
============================================

This approach avoids merge() entirely by using pandas MultiIndex for O(1) lookup.
Memory usage: Minimal (no temporary dataframe creation).

Replace the merge section (lines 1155-1189) with this code:
"""

# ============================================================
# ALTERNATIVE: MultiIndex mapping (avoids merge entirely)
# ============================================================
# Filter previous predictions to only rows with non-NaN predicted_return
prev_with_preds = previous_predictions[
    previous_predictions['predicted_return'].notna()
].copy()
print(f"  • Previous predictions with values: {len(prev_with_preds):,}")

# Ensure types match for indexing
prev_with_preds['Date'] = pd.to_datetime(prev_with_preds['Date'])
prev_with_preds['Symbol'] = prev_with_preds['Symbol'].astype(str)
df['Date'] = pd.to_datetime(df['Date'])
df['Symbol'] = df['Symbol'].astype(str)

# Create MultiIndex Series for fast O(1) lookup
prev_with_preds = prev_with_preds.set_index(['Date', 'Symbol'])
prediction_series = prev_with_preds['predicted_return']

# Create MultiIndex for df
df_index = pd.MultiIndex.from_arrays([df['Date'], df['Symbol']])

# Map predictions using MultiIndex (fast vectorized operation)
# This is memory-efficient: no temporary dataframe created
previous_predictions_array = df_index.map(
    lambda idx: prediction_series.get(idx, np.nan)
).to_numpy()

# Clean up
del prev_with_preds, prediction_series, df_index
import gc
gc.collect()

matches = np.sum(~np.isnan(previous_predictions_array))
print(f"  • Matched {matches:,} / {len(df):,} rows ({matches/len(df)*100:.1f}%)")
