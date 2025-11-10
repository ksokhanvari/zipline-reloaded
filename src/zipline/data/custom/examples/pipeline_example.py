"""
Example: Using Custom Data in Zipline Pipeline

This example demonstrates how to:
1. Create a custom data database
2. Load CSV data with symbol-to-sid mapping
3. Create a Pipeline DataSet from the custom data
4. Use custom data in a Pipeline
5. Integrate with a Zipline algorithm

Prerequisites:
- Zipline installed with custom data module
- A data bundle ingested (e.g., 'quandl')
"""

import pandas as pd
from zipline import run_algorithm
from zipline.api import (
    order_target_percent,
    record,
    symbol,
    get_datetime,
)
from zipline.pipeline import Pipeline
from zipline.pipeline.data import EquityPricing
from zipline.pipeline.factors import SimpleMovingAverage
from zipline.data.custom import (
    create_custom_db,
    load_csv_to_db,
    make_custom_dataset_class,
    CustomSQLiteLoader,
    describe_custom_db,
)


# ============================================================================
# Step 1: Create Database and Load Data (Run once)
# ============================================================================

def setup_custom_data():
    """
    Set up the custom fundamentals database.

    This only needs to be run once to create the database and load data.
    After that, you can query the data or add more data as needed.
    """
    print("Setting up custom fundamentals database...")

    # Define the schema
    columns = {
        'Revenue': 'int',
        'EPS': 'float',
        'MarketCap': 'int',
        'Currency': 'text',
    }

    # Create database (skip if already exists)
    try:
        create_custom_db(
            db_code='fundamentals',
            bar_size='1 quarter',  # Quarterly fundamental data
            columns=columns,
        )
        print("✓ Database created")
    except FileExistsError:
        print("✓ Database already exists")

    # Load data from CSV
    # Note: You'll need to provide a securities.csv with Symbol-to-Sid mapping
    # This can be generated from your bundle's asset database

    result = load_csv_to_db(
        csv_path='example_fundamentals.csv',
        db_code='fundamentals',
        sid_map=pd.read_csv('securities.csv'),  # Symbol -> Sid mapping
        id_col='Symbol',
        date_col='Date',
        on_duplicate='replace',  # Update existing records
    )

    print(f"✓ Loaded {result['rows_inserted']} rows")

    # Show database info
    info = describe_custom_db('fundamentals')
    print(f"\nDatabase: {info['db_code']}")
    print(f"Columns: {', '.join(info['columns'].keys())}")
    print(f"Date range: {info['date_range']}")
    print(f"Sids: {info['num_sids']}")


# ============================================================================
# Step 2: Create Pipeline with Custom Data
# ============================================================================

def make_pipeline():
    """
    Create a Pipeline that uses custom fundamental data.

    This pipeline combines:
    - Price data from the bundle (EquityPricing)
    - Fundamental data from custom database
    """
    # Get database info to know available columns
    info = describe_custom_db('fundamentals')

    # Create DataSet class dynamically
    FundamentalsDataSet = make_custom_dataset_class(
        db_code='fundamentals',
        columns=info['columns'],
    )

    # Define factors

    # Price factors (from bundle)
    close_price = EquityPricing.close.latest
    sma_20 = SimpleMovingAverage(
        inputs=[EquityPricing.close],
        window_length=20,
    )

    # Fundamental factors (from custom data)
    eps = FundamentalsDataSet.EPS.latest
    revenue = FundamentalsDataSet.Revenue.latest
    market_cap = FundamentalsDataSet.MarketCap.latest

    # Combined factor: Price-to-Sales ratio
    # (Market Cap / Revenue, where revenue is in millions)
    price_to_sales = market_cap / revenue

    # Return pipeline
    return Pipeline(
        columns={
            'close': close_price,
            'sma_20': sma_20,
            'eps': eps,
            'revenue': revenue,
            'market_cap': market_cap,
            'price_to_sales': price_to_sales,
        },
    )


# ============================================================================
# Step 3: Use in Zipline Algorithm
# ============================================================================

def initialize(context):
    """
    Initialize algorithm with custom data pipeline.
    """
    # Create custom data loader
    loader = CustomSQLiteLoader('fundamentals')

    # Attach pipeline (Note: This requires proper engine setup)
    # In practice, you'll need to register the loader with the engine
    context.pipeline = make_pipeline()

    # Define universe
    context.universe = [
        symbol('AAPL'),
        symbol('MSFT'),
        symbol('GOOGL'),
        symbol('AMZN'),
    ]

    print("Algorithm initialized with custom fundamentals pipeline")


def before_trading_start(context, data):
    """
    Run pipeline before market opens.

    Note: Full pipeline execution requires proper SimplePipelineEngine
    configuration with the custom loader registered.
    """
    # This is a simplified example
    # In production, you would:
    # 1. Configure SimplePipelineEngine with custom loader
    # 2. Run pipeline: context.output = pipeline_output('my_pipeline')
    # 3. Use the output for trading decisions

    pass


def handle_data(context, data):
    """
    Trading logic using pipeline output.
    """
    # Example: Equal-weight portfolio of stocks with P/S ratio < 10
    # (This is simplified - in practice you'd use pipeline output)

    for stock in context.universe:
        if data.can_trade(stock):
            # Placeholder logic
            order_target_percent(stock, 0.25)

    # Record metrics
    record(
        portfolio_value=context.portfolio.portfolio_value,
        cash=context.portfolio.cash,
    )


# ============================================================================
# Step 4: Run the Algorithm
# ============================================================================

def run_backtest():
    """
    Run the backtest with custom data.

    Note: This is a simplified example. Full integration requires:
    1. Custom loader registered with SimplePipelineEngine
    2. Pipeline execution in before_trading_start
    3. Proper bundle and calendar configuration
    """
    results = run_algorithm(
        start=pd.Timestamp('2020-01-01', tz='UTC'),
        end=pd.Timestamp('2020-12-31', tz='UTC'),
        initialize=initialize,
        handle_data=handle_data,
        before_trading_start=before_trading_start,
        capital_base=100000,
        bundle='quandl',
        data_frequency='daily',
    )

    return results


# ============================================================================
# Alternative: Query Custom Data Directly
# ============================================================================

def query_custom_data_example():
    """
    Example of querying custom data directly without Pipeline.

    This is useful for analysis or when you don't need full Pipeline features.
    """
    from zipline.data.custom import get_prices

    # Get all fundamentals for specific stocks
    df = get_prices(
        db_code='fundamentals',
        start_date='2020-01-01',
        end_date='2020-12-31',
        sids=[24, 5061, 26890, 16841],  # AAPL, MSFT, GOOGL, AMZN
    )

    print("\nFundamentals Data:")
    print(df.head())

    # Calculate metrics
    df['PE_Ratio'] = df['MarketCap'] / (df['EPS'] * 4)  # Annualized
    df['Price_to_Sales'] = df['MarketCap'] / df['Revenue']

    print("\nCalculated Metrics:")
    print(df[['Sid', 'Date', 'PE_Ratio', 'Price_to_Sales']].head())


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == 'setup':
        # Run setup to create database and load data
        setup_custom_data()

    elif len(sys.argv) > 1 and sys.argv[1] == 'query':
        # Run query example
        query_custom_data_example()

    elif len(sys.argv) > 1 and sys.argv[1] == 'backtest':
        # Run full backtest
        print("Running backtest...")
        results = run_backtest()
        print(f"\nBacktest complete!")
        print(f"Final portfolio value: ${results.portfolio_value.iloc[-1]:,.2f}")

    else:
        print("Usage:")
        print("  python pipeline_example.py setup     # Create DB and load data")
        print("  python pipeline_example.py query     # Query data directly")
        print("  python pipeline_example.py backtest  # Run full backtest")
