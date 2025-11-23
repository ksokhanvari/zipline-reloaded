#!/usr/bin/env python
"""
FlightLog - Real-time log viewer for Zipline backtests.

A standalone log server that displays backtest logs in a separate terminal,
similar to QuantRocket's flightlog.

Single Channel Usage:
    # Start the log server
    python scripts/flightlog.py

    # Or with Docker
    docker compose run --rm flightlog

    # Custom options
    python scripts/flightlog.py --port 9021 --file logs/backtest.log

Multi-Channel Usage:
    # Terminal 1: Algorithm logs only
    python scripts/flightlog.py --port 9020 --channel algorithm --logger-filter algorithm

    # Terminal 2: Progress logs only
    python scripts/flightlog.py --port 9021 --channel progress --logger-filter zipline.progress

    # Terminal 3: Errors only
    python scripts/flightlog.py --port 9022 --channel errors --level ERROR

For complete multi-channel guide, see: scripts/FLIGHTLOG_MULTI_CHANNEL_GUIDE.md
"""

import argparse
import logging
import logging.handlers
import pickle
import socketserver
import struct
import sys
from datetime import datetime


class ColoredFormatter(logging.Formatter):
    """Formatter that adds color codes for different log levels."""

    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
    }
    RESET = '\033[0m'
    BOLD = '\033[1m'

    def __init__(self, use_colors=True):
        super().__init__(
            fmt='%(asctime)s %(levelname)-8s %(name)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        self.use_colors = use_colors

    def format(self, record):
        if self.use_colors:
            levelname = record.levelname
            if levelname in self.COLORS:
                # Color the level name
                record.levelname = f"{self.COLORS[levelname]}{levelname}{self.RESET}"

        return super().format(record)


class ProgressFilter(logging.Filter):
    """Filter to exclude progress logs."""

    def filter(self, record):
        # Exclude zipline.progress logs
        return record.name != 'zipline.progress'


class LoggerFilter(logging.Filter):
    """Filter to only show logs from specific logger."""

    def __init__(self, logger_name):
        super().__init__()
        self.logger_name = logger_name

    def filter(self, record):
        # Only show logs from the specified logger (or its children)
        return record.name == self.logger_name or record.name.startswith(self.logger_name + '.')


class ExcludeLoggerFilter(logging.Filter):
    """Filter to exclude logs from specific logger."""

    def __init__(self, logger_name):
        super().__init__()
        self.logger_name = logger_name

    def filter(self, record):
        # Exclude logs from the specified logger (and its children)
        return not (record.name == self.logger_name or record.name.startswith(self.logger_name + '.'))


class LogRecordStreamHandler(socketserver.StreamRequestHandler):
    """Handler for streaming log records."""

    def handle(self):
        """
        Handle multiple log records from a single connection.
        Each record is pickled and prefixed with a 4-byte length.
        """
        while True:
            try:
                # Read the 4-byte length prefix
                chunk = self.connection.recv(4)
                if len(chunk) < 4:
                    break

                # Unpack the length
                slen = struct.unpack('>L', chunk)[0]

                # Read the log record data
                chunk = self.connection.recv(slen)
                while len(chunk) < slen:
                    chunk += self.connection.recv(slen - len(chunk))

                # Unpickle and log the record
                record = logging.makeLogRecord(pickle.loads(chunk))
                self.server.logger.handle(record)

            except Exception as e:
                # Connection closed or error
                break


class LogRecordSocketReceiver(socketserver.ThreadingTCPServer):
    """
    Simple TCP socket-based logging receiver.
    """

    allow_reuse_address = True

    def __init__(self, host='0.0.0.0', port=9020, handler=LogRecordStreamHandler):
        socketserver.ThreadingTCPServer.__init__(self, (host, port), handler)
        self.timeout = 1
        self.logger = logging.getLogger()


def main():
    """Main entry point for FlightLog server."""
    parser = argparse.ArgumentParser(
        description='FlightLog - Real-time log viewer for Zipline backtests'
    )
    parser.add_argument(
        '--host',
        default='0.0.0.0',
        help='Host to bind to (default: 0.0.0.0)'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=9020,
        help='Port to listen on (default: 9020)'
    )
    parser.add_argument(
        '--level',
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help='Minimum log level to display (default: INFO)'
    )
    parser.add_argument(
        '--file',
        help='Save logs to file (optional)'
    )
    parser.add_argument(
        '--no-color',
        action='store_true',
        help='Disable color output'
    )
    parser.add_argument(
        '--filter-progress',
        action='store_true',
        help='Filter out progress logs (only show algorithm logs)'
    )
    parser.add_argument(
        '--channel',
        default='default',
        help='Channel name for this listener (default: default)'
    )
    parser.add_argument(
        '--logger-filter',
        help='Only show logs from specific logger (e.g., "algorithm", "zipline.progress")'
    )
    parser.add_argument(
        '--exclude-logger',
        help='Exclude logs from specific logger (e.g., "zipline.finance")'
    )

    args = parser.parse_args()

    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, args.level))

    # Console handler with colors
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(ColoredFormatter(use_colors=not args.no_color))

    # Add filters as requested
    if args.filter_progress:
        console_handler.addFilter(ProgressFilter())

    if args.logger_filter:
        console_handler.addFilter(LoggerFilter(args.logger_filter))

    if args.exclude_logger:
        console_handler.addFilter(ExcludeLoggerFilter(args.exclude_logger))

    root_logger.addHandler(console_handler)

    # Optional file handler
    if args.file:
        file_handler = logging.FileHandler(args.file)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s %(levelname)-8s %(name)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        root_logger.addHandler(file_handler)

    # Print banner
    print("=" * 70)
    print(f"FlightLog Server - Channel: {args.channel.upper()}")
    print("=" * 70)
    print(f"Listening on: {args.host}:{args.port}")
    print(f"Log level: {args.level}")
    if args.file:
        print(f"Saving to: {args.file}")

    # Show active filters
    filters_active = []
    if args.filter_progress:
        filters_active.append("Hiding progress logs")
    if args.logger_filter:
        filters_active.append(f"Only showing '{args.logger_filter}' logger")
    if args.exclude_logger:
        filters_active.append(f"Excluding '{args.exclude_logger}' logger")

    if filters_active:
        print()
        print("Active Filters:")
        for f in filters_active:
            print(f"  • {f}")

    print()
    print("Waiting for backtest connections...")
    print("Press Ctrl+C to stop")
    print("=" * 70)
    print()

    # Start server
    server = LogRecordSocketReceiver(args.host, args.port)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n")
        print("=" * 70)
        print("FlightLog Server Stopped")
        print("=" * 70)
        server.shutdown()
        sys.exit(0)


if __name__ == '__main__':
    main()
