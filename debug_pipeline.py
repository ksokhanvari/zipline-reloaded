"""
Debug script to check pipeline output before running backtest.
"""
import pandas as pd
from zipline.pipeline import Pipeline
from zipline.pipeline.engine import SimplePipelineEngine
from zipline.data.bundles import load as load_bundle
from zipline.pipeline import multi_source as ms

# Register bundle
from zipline.data.bundles import register
from zipline.data.bundles.sharadar_bundle import sharadar_bundle

register('sharadar', sharadar_bundle(tickers=None, incremental=True, include_funds=True))

# Define LSEG database
class LSEGFundamentals(ms.Database):
    CODE = "fundamentals"
    LOOKBACK_WINDOW = 252
    ReturnOnEquity_SmartEstimat = ms.Column(float)
    ForwardPEG_DailyTimeSeriesRatio_ = ms.Column(float)
    CompanyMarketCap = ms.Column(float)
    Debt_Total = ms.Column(float)

# Load bundle
bundle_data = load_bundle('sharadar')

# Create a relaxed pipeline for debugging
def make_debug_pipeline():
    """Pipeline with relaxed filters to see what data is available."""
    # Sharadar
    s_roe = ms.SharadarFundamentals.roe.latest
    s_fcf = ms.SharadarFundamentals.fcf.latest
    s_marketcap = ms.SharadarFundamentals.marketcap.latest
    s_pe = ms.SharadarFundamentals.pe.latest

    # LSEG
    l_roe = LSEGFundamentals.ReturnOnEquity_SmartEstimat.latest
    l_peg = LSEGFundamentals.ForwardPEG_DailyTimeSeriesRatio_.latest

    # Very relaxed universe - top 500
    universe = s_marketcap.top(500)

    # Just Sharadar filter (no LSEG requirement yet)
    sharadar_quality = (s_roe > 15.0) & (s_fcf > 0) & (s_pe > 0) & (s_pe < 30)

    # LSEG filter (separate)
    lseg_quality = (l_roe > 15.0) & (l_peg > 0) & (l_peg < 2.5)

    # Consensus
    both_confirm = sharadar_quality & lseg_quality

    return ms.Pipeline(
        columns={
            's_roe': s_roe,
            's_fcf': s_fcf,
            's_marketcap': s_marketcap,
            's_pe': s_pe,
            'l_roe': l_roe,
            'l_peg': l_peg,
            'sharadar_quality': sharadar_quality,
            'lseg_quality': lseg_quality,
            'both_confirm': both_confirm,
        },
        screen=universe,  # Just universe, no quality filter yet
    )

# Create engine
engine = SimplePipelineEngine(
    get_loader=ms.setup_auto_loader().get,
    asset_finder=bundle_data.asset_finder,
)

# Run pipeline for a single day
test_date = pd.Timestamp('2023-01-05')
print(f"\nRunning pipeline for {test_date.date()}...")

try:
    result = engine.run_pipeline(
        make_debug_pipeline(),
        test_date,
        test_date
    )

    print(f"\n✓ Pipeline returned {len(result)} stocks\n")

    if len(result) > 0:
        print("="*80)
        print("PIPELINE OUTPUT SAMPLE (first 10 stocks)")
        print("="*80)
        print(result.head(10).to_string())

        print("\n" + "="*80)
        print("DATA AVAILABILITY SUMMARY")
        print("="*80)

        # Check for NaN values
        print(f"\nSharadar data availability:")
        print(f"  ROE: {result['s_roe'].notna().sum()} / {len(result)} ({result['s_roe'].notna().sum()/len(result)*100:.1f}%)")
        print(f"  FCF: {result['s_fcf'].notna().sum()} / {len(result)} ({result['s_fcf'].notna().sum()/len(result)*100:.1f}%)")
        print(f"  MarketCap: {result['s_marketcap'].notna().sum()} / {len(result)} ({result['s_marketcap'].notna().sum()/len(result)*100:.1f}%)")
        print(f"  PE: {result['s_pe'].notna().sum()} / {len(result)} ({result['s_pe'].notna().sum()/len(result)*100:.1f}%)")

        print(f"\nLSEG data availability:")
        print(f"  ROE: {result['l_roe'].notna().sum()} / {len(result)} ({result['l_roe'].notna().sum()/len(result)*100:.1f}%)")
        print(f"  PEG: {result['l_peg'].notna().sum()} / {len(result)} ({result['l_peg'].notna().sum()/len(result)*100:.1f}%)")

        print(f"\nFilter results:")
        print(f"  Sharadar quality: {result['sharadar_quality'].sum()} stocks")
        print(f"  LSEG quality: {result['lseg_quality'].sum()} stocks")
        print(f"  Both confirm: {result['both_confirm'].sum()} stocks")

        if result['both_confirm'].sum() > 0:
            print(f"\n✓ Found {result['both_confirm'].sum()} stocks with dual confirmation!")
            print("\nTop 5 by Sharadar ROE with dual confirmation:")
            confirmed = result[result['both_confirm']]
            top5 = confirmed.nlargest(5, 's_roe')
            print(top5[['s_roe', 'l_roe', 's_fcf', 's_pe', 'l_peg']].to_string())
        else:
            print("\n⚠️  No stocks found with dual confirmation!")
            print("\nTrying to find the issue...")

            # Show stocks with Sharadar quality
            if result['sharadar_quality'].sum() > 0:
                print(f"\nStocks passing Sharadar filter: {result['sharadar_quality'].sum()}")
                sharadar_pass = result[result['sharadar_quality']].head(5)
                print(sharadar_pass[['s_roe', 'l_roe', 's_fcf', 's_pe', 'l_peg']].to_string())

            # Show stocks with LSEG quality
            if result['lseg_quality'].sum() > 0:
                print(f"\nStocks passing LSEG filter: {result['lseg_quality'].sum()}")
                lseg_pass = result[result['lseg_quality']].head(5)
                print(lseg_pass[['s_roe', 'l_roe', 's_fcf', 's_pe', 'l_peg']].to_string())
    else:
        print("⚠️  Pipeline returned no stocks!")
        print("\nThis could mean:")
        print("1. No Sharadar data available for this date")
        print("2. Universe filter is too restrictive")
        print("3. Data loader issue")

except Exception as e:
    print(f"\n❌ Pipeline failed: {e}")
    import traceback
    traceback.print_exc()
