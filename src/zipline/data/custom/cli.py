"""
CLI commands for custom data management.

Provides command-line interface for creating databases, loading data,
and querying custom data sources.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

import click
import pandas as pd

from .config import get_custom_data_dir
from .db_manager import (
    create_custom_db,
    describe_custom_db,
    list_custom_dbs,
    connect_db,
)
from .loader import load_csv_to_db
from .pipeline_integration import (
    make_custom_dataset_class,
    CustomSQLiteLoader,
)

log = logging.getLogger(__name__)


@click.group(name='custom-data')
def custom_data_group():
    """Manage custom data sources for Zipline."""
    pass


@custom_data_group.command(name='create-db')
@click.argument('db-code')
@click.option(
    '--bar-size',
    default='1 day',
    show_default=True,
    help='Bar size/frequency (e.g., "1 day", "1 week")',
)
@click.option(
    '--columns',
    required=True,
    help='Column definitions as "name:type" pairs, comma-separated. '
         'Example: Revenue:int,EPS:float,Currency:text',
)
@click.option(
    '--db-dir',
    type=click.Path(file_okay=False, dir_okay=True, path_type=str),
    help='Directory for database. Defaults to ~/.zipline/custom_data/',
)
def create_db_command(db_code, bar_size, columns, db_dir):
    """
    Create a new custom data database.

    Example:

        zipline custom-data create-db fundamentals \\
            --columns "Revenue:int,EPS:float,Currency:text"
    """
    try:
        # Parse columns string
        column_dict = {}
        for col_spec in columns.split(','):
            col_spec = col_spec.strip()
            if ':' not in col_spec:
                click.echo(
                    f"Error: Invalid column specification '{col_spec}'. "
                    "Format should be 'name:type'",
                    err=True
                )
                sys.exit(1)

            name, type_str = col_spec.split(':', 1)
            column_dict[name.strip()] = type_str.strip()

        # Create database
        db_path = create_custom_db(
            db_code=db_code,
            bar_size=bar_size,
            columns=column_dict,
            db_dir=db_dir,
        )

        click.echo(f"✓ Created custom database: {db_path}")
        click.echo(f"  DB Code: {db_code}")
        click.echo(f"  Bar Size: {bar_size}")
        click.echo(f"  Columns: {', '.join(column_dict.keys())}")

    except FileExistsError as e:
        click.echo(f"Error: {e}", err=True)
        click.echo("Use 'zipline custom-data list-dbs' to see existing databases.", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error creating database: {e}", err=True)
        sys.exit(1)


@custom_data_group.command(name='load-csv')
@click.argument('csv-path', type=click.Path(exists=True, dir_okay=False, path_type=str))
@click.argument('db-code')
@click.option(
    '--securities-csv',
    type=click.Path(exists=True, dir_okay=False, path_type=str),
    help='CSV file with symbol-to-sid mapping (columns: Symbol, Sid)',
)
@click.option(
    '--id-col',
    default='Symbol',
    show_default=True,
    help='Column name for identifier (e.g., Symbol, Ticker)',
)
@click.option(
    '--date-col',
    default='Date',
    show_default=True,
    help='Column name for dates',
)
@click.option(
    '--date-format',
    help='Date format string (e.g., "%%Y-%%m-%%d"). If not specified, pandas will infer.',
)
@click.option(
    '--tz',
    help='Timezone for dates (e.g., "America/New_York", "UTC")',
)
@click.option(
    '--chunk-size',
    type=int,
    default=100000,
    show_default=True,
    help='Number of rows to process at a time',
)
@click.option(
    '--on-duplicate',
    type=click.Choice(['replace', 'ignore', 'fail']),
    default='replace',
    show_default=True,
    help='Strategy for handling duplicate (Sid, Date) pairs',
)
@click.option(
    '--fail-on-unmapped/--skip-unmapped',
    default=False,
    show_default=True,
    help='Fail if identifiers cannot be mapped to Sids',
)
@click.option(
    '--db-dir',
    type=click.Path(file_okay=False, dir_okay=True, path_type=str),
    help='Directory containing database',
)
def load_csv_command(
    csv_path,
    db_code,
    securities_csv,
    id_col,
    date_col,
    date_format,
    tz,
    chunk_size,
    on_duplicate,
    fail_on_unmapped,
    db_dir,
):
    """
    Load CSV data into a custom database.

    Example:

        zipline custom-data load-csv data.csv fundamentals \\
            --securities-csv securities.csv \\
            --id-col Symbol \\
            --date-col Date
    """
    try:
        # Load securities mapping if provided
        sid_map = None
        if securities_csv:
            try:
                securities_df = pd.read_csv(securities_csv)
                if 'Symbol' not in securities_df.columns or 'Sid' not in securities_df.columns:
                    click.echo(
                        "Error: Securities CSV must have 'Symbol' and 'Sid' columns",
                        err=True
                    )
                    sys.exit(1)
                sid_map = securities_df
                click.echo(f"Loaded {len(sid_map)} securities from {securities_csv}")
            except Exception as e:
                click.echo(f"Error loading securities CSV: {e}", err=True)
                sys.exit(1)

        # Load CSV data
        click.echo(f"Loading data from {csv_path} into {db_code}...")

        result = load_csv_to_db(
            csv_path=csv_path,
            db_code=db_code,
            sid_map=sid_map,
            id_col=id_col,
            date_col=date_col,
            date_format=date_format,
            tz=tz,
            chunk_size=chunk_size,
            on_duplicate=on_duplicate,
            fail_on_unmapped=fail_on_unmapped,
            db_dir=db_dir,
        )

        # Report results
        click.echo("\n✓ Data loading complete")
        click.echo(f"  Rows inserted: {result['rows_inserted']:,}")
        click.echo(f"  Rows skipped: {result['rows_skipped']:,}")

        if result['unmapped_ids']:
            click.echo(f"\n⚠  Warning: {len(result['unmapped_ids'])} unmapped identifiers:")
            for uid in result['unmapped_ids'][:10]:
                click.echo(f"    - {uid}")
            if len(result['unmapped_ids']) > 10:
                click.echo(f"    ... and {len(result['unmapped_ids']) - 10} more")

        if result['errors']:
            click.echo(f"\n⚠  Encountered {len(result['errors'])} errors:")
            for err in result['errors'][:5]:
                click.echo(f"    - {err}")
            if len(result['errors']) > 5:
                click.echo(f"    ... and {len(result['errors']) - 5} more")

    except FileNotFoundError as e:
        click.echo(f"Error: {e}", err=True)
        click.echo("Use 'zipline custom-data list-dbs' to see available databases.", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error loading CSV: {e}", err=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)


@custom_data_group.command(name='describe-db')
@click.argument('db-code')
@click.option(
    '--db-dir',
    type=click.Path(file_okay=False, dir_okay=True, path_type=str),
    help='Directory containing database',
)
def describe_db_command(db_code, db_dir):
    """
    Show metadata and statistics for a custom database.

    Example:

        zipline custom-data describe-db fundamentals
    """
    try:
        info = describe_custom_db(db_code, db_dir)

        click.echo(f"\nDatabase: {info['db_code']}")
        click.echo(f"Path: {info['db_path']}")
        click.echo(f"Bar Size: {info['bar_size']}")
        click.echo(f"\nColumns:")
        for col_name, col_type in info['columns'].items():
            click.echo(f"  - {col_name}: {col_type}")

        click.echo(f"\nStatistics:")
        click.echo(f"  Total rows: {info['row_count']:,}")
        click.echo(f"  Unique Sids: {info['num_sids']}")

        if info['date_range']:
            click.echo(f"  Date range: {info['date_range'][0]} to {info['date_range'][1]}")
        else:
            click.echo(f"  Date range: (empty)")

        if info['sids']:
            click.echo(f"\nSample Sids:")
            for sid in info['sids'][:10]:
                click.echo(f"  - {sid}")
            if info['num_sids'] > 10:
                click.echo(f"  ... and {info['num_sids'] - 10} more")

    except FileNotFoundError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error describing database: {e}", err=True)
        sys.exit(1)


@custom_data_group.command(name='list-dbs')
@click.option(
    '--db-dir',
    type=click.Path(file_okay=False, dir_okay=True, path_type=str),
    help='Directory to search for databases',
)
def list_dbs_command(db_dir):
    """
    List all available custom databases.

    Example:

        zipline custom-data list-dbs
    """
    try:
        dbs = list_custom_dbs(db_dir)

        if not dbs:
            click.echo("No custom databases found.")
            db_path = db_dir if db_dir else get_custom_data_dir()
            click.echo(f"Searched in: {db_path}")
            click.echo("\nCreate a database with:")
            click.echo("  zipline custom-data create-db <db-code> --columns <cols>")
            return

        click.echo(f"Found {len(dbs)} custom database(s):\n")

        for db_code in dbs:
            try:
                info = describe_custom_db(db_code, db_dir)
                click.echo(f"{db_code}")
                click.echo(f"  Rows: {info['row_count']:,}")
                click.echo(f"  Sids: {info['num_sids']}")
                click.echo(f"  Columns: {', '.join(info['columns'].keys())}")
                if info['date_range']:
                    click.echo(f"  Dates: {info['date_range'][0]} to {info['date_range'][1]}")
                click.echo()
            except Exception as e:
                click.echo(f"{db_code}")
                click.echo(f"  Error: {e}")
                click.echo()

    except Exception as e:
        click.echo(f"Error listing databases: {e}", err=True)
        sys.exit(1)


@custom_data_group.command(name='dump-db')
@click.argument('db-code')
@click.argument('output-path', type=click.Path(dir_okay=False, path_type=str))
@click.option(
    '--start-date',
    help='Start date (YYYY-MM-DD)',
)
@click.option(
    '--end-date',
    help='End date (YYYY-MM-DD)',
)
@click.option(
    '--sids',
    help='Comma-separated list of Sids to export',
)
@click.option(
    '--db-dir',
    type=click.Path(file_okay=False, dir_okay=True, path_type=str),
    help='Directory containing database',
)
def dump_db_command(db_code, output_path, start_date, end_date, sids, db_dir):
    """
    Export database data to CSV.

    Example:

        zipline custom-data dump-db fundamentals output.csv \\
            --start-date 2020-01-01 \\
            --end-date 2020-12-31
    """
    try:
        from .query import get_prices

        # Parse sids if provided
        sid_list = None
        if sids:
            try:
                sid_list = [int(s.strip()) for s in sids.split(',')]
            except ValueError:
                click.echo("Error: Sids must be comma-separated integers", err=True)
                sys.exit(1)

        # Query data
        click.echo(f"Exporting data from {db_code}...")
        df = get_prices(
            db_code=db_code,
            start_date=start_date,
            end_date=end_date,
            sids=sid_list,
            db_dir=db_dir,
        )

        if df.empty:
            click.echo("No data found matching criteria.", err=True)
            sys.exit(1)

        # Export to CSV
        df.to_csv(output_path, index=True)

        click.echo(f"\n✓ Exported {len(df):,} rows to {output_path}")

    except FileNotFoundError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error exporting data: {e}", err=True)
        sys.exit(1)


@custom_data_group.command(name='run-example')
@click.argument('db-code')
@click.option(
    '--start-date',
    required=True,
    help='Start date (YYYY-MM-DD)',
)
@click.option(
    '--end-date',
    required=True,
    help='End date (YYYY-MM-DD)',
)
@click.option(
    '--db-dir',
    type=click.Path(file_okay=False, dir_okay=True, path_type=str),
    help='Directory containing database',
)
def run_example_command(db_code, start_date, end_date, db_dir):
    """
    Run a simple Pipeline example using custom data.

    Example:

        zipline custom-data run-example fundamentals \\
            --start-date 2020-01-01 \\
            --end-date 2020-12-31
    """
    try:
        from zipline.pipeline import Pipeline
        from zipline.pipeline.engine import SimplePipelineEngine
        from zipline.utils.calendar_utils import get_calendar

        # Get database info
        info = describe_custom_db(db_code, db_dir)

        click.echo(f"\nRunning Pipeline example for {db_code}")
        click.echo(f"Columns: {', '.join(info['columns'].keys())}")

        # Create DataSet class
        dataset_class = make_custom_dataset_class(
            db_code=db_code,
            columns=info['columns'],
        )

        # Create loader
        loader = CustomSQLiteLoader(db_code, db_dir)

        # Create a simple pipeline using first column
        first_col = list(info['columns'].keys())[0]
        column_obj = getattr(dataset_class, first_col)

        pipeline = Pipeline(
            columns={
                f'{first_col}_latest': column_obj.latest,
            }
        )

        click.echo(f"\nPipeline definition:")
        click.echo(f"  - {first_col}_latest")

        # For this example, we would need a properly configured
        # SimplePipelineEngine with asset finder and domain
        click.echo("\nNote: Full Pipeline execution requires:")
        click.echo("  - Asset finder (bundle)")
        click.echo("  - Trading calendar")
        click.echo("  - Pipeline domain")
        click.echo("\nSee documentation for complete integration example.")

        click.echo("\nDataSet class created successfully!")
        click.echo("You can use it in your algorithms like:")
        click.echo(f"\n  from zipline.data.custom import make_custom_dataset_class")
        click.echo(f"  {dataset_class.__name__} = make_custom_dataset_class('{db_code}', ...)")
        click.echo(f"  pipeline = Pipeline(columns={{")
        click.echo(f"      '{first_col}': {dataset_class.__name__}.{first_col}.latest,")
        click.echo(f"  }})")

    except FileNotFoundError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error running example: {e}", err=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)


# For backwards compatibility, also register individual commands
# This allows both "zipline custom-data create-db" and direct import
create_db = create_db_command
load_csv = load_csv_command
describe_db = describe_db_command
list_dbs = list_dbs_command
dump_db = dump_db_command
run_example = run_example_command
